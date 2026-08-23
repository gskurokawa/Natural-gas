#%%#%%
"""
tiered_rag_langchain.py -- two-tier RAG over a fixed corpus of research papers.

Primary papers (papers1) are authoritative; background literature (papers2)
gives context. When both layers return something, one cheap LLM call asks
whether they disagree; if they do, an investigator agent runs its own
follow-up searches before the answer is written.

What the system does, and how the defaults below were measured:
see tiered_rag_langchain_doc.md.

Install:
    python -m pip install langchain langchain-classic langchain-community langchain-text-splitters
    python -m pip install langchain-ollama langchain-openai langchain-anthropic langchain-google-genai
    python -m pip install faiss-cpu rank_bm25 pypdf pymupdf

Targets LangChain 1.0+: EnsembleRetriever moved to langchain-classic, and
agents are built with langchain.agents.create_agent.

Usage:
    python tiered_rag_langchain.py serve                  # browser front-end
    python tiered_rag_langchain.py ingest
    python tiered_rag_langchain.py ask "your question"
    python tiered_rag_langchain.py ask -kp 4 -kb 2 --alpha 0.5 --tp 0.35 --ts 0.25
    python tiered_rag_langchain.py ask --no-investigate   # skip the agent step
    python tiered_rag_langchain.py ask --show             # print retrieved chunks

Testing API -- the CLI is a thin wrapper over these, which return data instead
of printing, so a sweep can call them directly:

    from tiered_rag_langchain import load_corpus, retrieve, answer, RunConfig

    corpus = load_corpus()                       # or pass index dirs explicitly
    cfg = RunConfig(kp=5, kb=4, alpha=0.5)       # tuned defaults if omitted
    r = retrieve("your question", corpus, cfg)   # retrieval only, no LLM
    a = answer("your question", corpus, cfg)     # full pipeline

RunConfig holds the five retrieval-time settings and can change per call.
Chunk size and overlap are baked into the index at ingest time, so sweeping
those means re-running ingest with CHUNK_WORDS / CHUNK_OVERLAP set and loading
that configuration's index:

    corpus = load_corpus(Path("index_p_400_50"), Path("index_s_400_50"))
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.agents import create_agent


# ============================================================================
# CONFIGURATION -- same hyperparameters as rag_tiered.py, same env-var pattern
# ============================================================================

def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/0, true/false, yes/no, on/off."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text expects "search_document: " on stored passages and
# "search_query: " on questions; omitting them degrades retrieval. LangChain's
# OllamaEmbeddings does not add them, so get_embeddings() does. Auto-on for any
# model named *nomic*; force with EMBED_PREFIXES=1/0. Most other embedding
# models do NOT want these. Changing this requires a re-ingest.
_PREFIX_ENV = os.environ.get("EMBED_PREFIXES")
USE_EMBED_PREFIXES = (
    "nomic" in EMBED_MODEL.lower()
    if _PREFIX_ENV is None
    else _PREFIX_ENV.strip().lower() in ("1", "true", "yes", "on")
)
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# --- TUNED VALUES ------------------------------------------------------------
# Measured, not guessed. Evidence in tiered_rag_langchain_doc.md.
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", 300))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
# LangChain's splitter counts CHARACTERS, not words -- convert at ~5.5 chars/word.
CHUNK_SIZE_CHARS = int(CHUNK_WORDS * 5.5)
CHUNK_OVERLAP_CHARS = int(CHUNK_OVERLAP * 5.5)

PAPERS_PRIMARY_DIR = Path(os.environ.get("PAPERS_PRIMARY_DIR", "papers1"))
PAPERS_SECONDARY_DIR = Path(os.environ.get("PAPERS_SECONDARY_DIR", "papers2"))
# Named for the chunk settings that built them, matching sweep_retrieval.py, so
# an index cannot be silently queried with the wrong chunk size.
INDEX_PRIMARY_DIR = Path(os.environ.get("INDEX_PRIMARY_DIR", "index_p_300_50"))
INDEX_SECONDARY_DIR = Path(os.environ.get("INDEX_SECONDARY_DIR", "index_s_300_50"))

# Config D. Set K_PRIMARY_DEFAULT=3 in .env for the leaner config E.
K_PRIMARY_DEFAULT = int(os.environ.get("K_PRIMARY_DEFAULT", 8))
K_SECONDARY_DEFAULT = int(os.environ.get("K_SECONDARY_DEFAULT", 10))

# No measured effect at any value; 0.5 is a midpoint of a flat line.
ALPHA_DEFAULT = float(os.environ.get("ALPHA_DEFAULT", 0.5))

# Inert: no cutoff separates the fake questions from the genuine ones. The
# system prompt is what actually refuses. Left as a cheap floor.
PRIMARY_SCORE_THRESHOLD = float(os.environ.get("PRIMARY_SCORE_THRESHOLD", 0.3))
SECONDARY_SCORE_THRESHOLD = float(os.environ.get("SECONDARY_SCORE_THRESHOLD", 0.2))

MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS_DEFAULT", 900))
TEMPERATURE_DEFAULT = float(os.environ.get("TEMPERATURE_DEFAULT", 0.0))

LLM_DEFAULTS = {
    "ollama": "llama3.1:8b",
    "google": "gemini-2.5-flash-lite",
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-haiku-4-5",
}

SYSTEM_PROMPT = """You answer questions using only the excerpts from academic papers provided.

Rules:
- Ground every claim in the excerpts. Cite them inline as [1], [2], etc.
- If the excerpts do not contain the answer, say so plainly. Do not fill the gap from memory.
- Distinguish what a paper claims from what it demonstrates.
- Be concise. No preamble, no restating the question."""


# ============================================================================
# PDF EXTRACTION AND CLEANING -- unchanged from rag_tiered.py.
# No generic LangChain loader replicates this, so it stays hand-written.
# ============================================================================

_COMMON_WORDS = set("""
the be to of and a in that have it for not on with he as you do at this but his
by from they we say her she or an will my one all would there their what so up
out if about who get which go me when make can like time no just him know take
people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even
new want because any these give day most us is are was were been has had did
does said can may might must shall should will would am being where why while
each such between both under above again very more many much less least own
same too here both during before against among within without across per
model models data set sets test tests result results paper study
""".split())


def fix_split_caps(text: str) -> str:
    def join_lower(m):
        letter, rest = m.group(1), m.group(2)
        if letter in ("A", "I") or rest.lower() in _COMMON_WORDS:
            return m.group(0)
        return letter + rest

    def join_upper(m):
        letter, rest = m.group(1), m.group(2)
        if letter in ("A", "I"):
            return m.group(0)
        return letter + rest

    text = re.sub(r"\b([A-Z]) ([a-z]{2,})\b", join_lower, text)
    text = re.sub(r"\b([A-Z]) ([A-Z]{2,3})\b", join_upper, text)
    return text


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = fix_split_caps(text)
    return text.strip()


def extract_pdf(path: Path) -> tuple[list[tuple[int, str]], str]:
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        pages = []
        with pymupdf.open(path) as doc:
            meta = (doc.metadata or {}).get("title") or ""
            for i, page in enumerate(doc, start=1):
                pages.append((i, page.get_text("text")))
        return pages, meta.strip()
    except ImportError:
        pass

    from pypdf import PdfReader
    reader = PdfReader(str(path))
    meta = ""
    try:
        meta = (reader.metadata.title or "").strip()
    except Exception:
        pass
    pages = [(i, (p.extract_text() or "")) for i, p in enumerate(reader.pages, 1)]
    return pages, meta


def guess_title(first_page: str, meta_title: str, fallback: str) -> str:
    if 10 < len(meta_title) < 250 and not meta_title.lower().endswith(".pdf"):
        return meta_title
    lines = [fix_split_caps(ln.strip())
             for ln in first_page.splitlines()[:12] if ln.strip()]
    for i, line in enumerate(lines):
        if not (15 < len(line) < 200):
            continue
        if line.lower().startswith(("arxiv", "http", "doi", "preprint")):
            continue
        ends_with_marker = line[-1] in "⋆*†‡§"
        clean_line = line.rstrip("⋆*†‡§ ")
        if i + 1 < len(lines) and not ends_with_marker:
            nxt = lines[i + 1]
            looks_like_authors = bool(re.search(r"\b[A-Z]\.\s*[A-Za-z]", nxt))
            if (clean_line[-1] not in ".?!"
                    and len(nxt) < 90
                    and "@" not in nxt
                    and not looks_like_authors
                    and not re.match(r"^\s*(abstract|\d)", nxt, re.I)):
                return f"{clean_line} {nxt.rstrip('⋆*†‡§ ')}"
        return clean_line
    return fallback


_REF_HEADING = re.compile(
    r"(?mi)^[ \t]*(?:\d+[.)]?[ \t]*)?(references|bibliography)[ \t]*:?[ \t]*$"
)
_POST_REF_HEADING = re.compile(
    r"(?mi)^[ \t]*(?:\d+[.)]?[ \t]*)?"
    r"(appendix|data availability|supplementary|acknowledge?ments?)\b.*$"
)


def cut_references(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    total = sum(len(t) for _, t in pages) or 1
    seen = 0
    cut_at = None
    for idx, (_, text) in enumerate(pages):
        for m in _REF_HEADING.finditer(text):
            if seen + m.start() > total * 0.25:
                cut_at = (idx, m.start())
        seen += len(text)
    if cut_at is None:
        return pages
    idx, offset = cut_at
    kept = list(pages[:idx])
    head = pages[idx][1][:offset]
    if head.strip():
        kept.append((pages[idx][0], head))
    tail_pages = [(pages[idx][0], pages[idx][1][offset:])] + list(pages[idx + 1:])
    for page_no, text in tail_pages:
        m = _POST_REF_HEADING.search(text)
        if m:
            kept.append((page_no, text[m.start():]))
    return kept


def load_layer_documents(papers_dir: Path, layer: str, keep_refs: bool = False) -> list[Document]:
    """Read all PDFs in a directory, clean them, and return LangChain Documents
    (one per page) -- these get split into chunks later by the text splitter."""
    pdfs = sorted(p for p in papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {papers_dir.resolve()}")
        return []

    docs: list[Document] = []
    for pdf in pdfs:
        pages, meta_title = extract_pdf(pdf)
        if not any(t.strip() for _, t in pages):
            print(f"  {pdf.name}: no extractable text (scanned image? needs OCR)")
            continue
        title = guess_title(pages[0][1], meta_title, pdf.stem)
        body_pages = pages if keep_refs else cut_references(pages)

        before = len(docs)
        for page_no, raw in body_pages:
            body = clean(raw)
            if not body:
                continue
            docs.append(Document(
                page_content=body,
                metadata={"source": pdf.name, "title": title, "page": page_no, "layer": layer},
            ))
        print(f"  {pdf.name}: {len(pages)} pages -> {len(docs) - before} page-documents  [{title[:60]}]")
    return docs


# ============================================================================
# LLM DISPATCH -- mirrors your PROVIDER/MODEL env-var pattern, using
# LangChain's Chat* wrapper classes instead of raw HTTP/SDK calls.
# ============================================================================

def get_llm(for_tools: bool = False):
    """Build the chat model.

    for_tools=True routes OpenAI through the Responses API. Newer reasoning
    models reject function tools on /v1/chat/completions, which the
    investigator agent needs, and langchain-openai does not yet auto-detect
    this (langchain issue #35584). Plain chains stay on the default endpoint.
    """
    provider = os.environ.get("PROVIDER", "ollama").lower()
    model = os.environ.get("MODEL") or LLM_DEFAULTS.get(provider, "")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=TEMPERATURE_DEFAULT)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": model}
        # OPENAI_RESPONSES_API=0 forces Chat Completions, =1 forces Responses.
        if env_flag("OPENAI_RESPONSES_API", for_tools):
            kwargs["use_responses_api"] = True
        # OPENAI_SEND_TEMPERATURE=0 if the model rejects a custom temperature.
        if env_flag("OPENAI_SEND_TEMPERATURE", True):
            kwargs["temperature"] = TEMPERATURE_DEFAULT
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=TEMPERATURE_DEFAULT)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=TEMPERATURE_DEFAULT)

    raise SystemExit(f"Unknown PROVIDER={provider!r}. Choose one of {list(LLM_DEFAULTS)}.")


_TOOL_LLM = None


def get_tool_llm():
    """The model used by the investigator agent. Built once, then reused."""
    global _TOOL_LLM
    if _TOOL_LLM is None:
        _TOOL_LLM = get_llm(for_tools=True)
    return _TOOL_LLM


# ============================================================================
# EMBEDDINGS -- one factory for BOTH ingest and ask. A prefixing mismatch
# between the two silently degrades every search.
# ============================================================================

def get_embeddings():
    """Embedding object, with nomic task prefixes where appropriate.

    Prefixes are applied at embedding time only; stored text is untouched.
    """
    from langchain_ollama import OllamaEmbeddings

    if not USE_EMBED_PREFIXES:
        return OllamaEmbeddings(model=EMBED_MODEL)

    class PrefixedOllamaEmbeddings(OllamaEmbeddings):
        """Prepends nomic's task prefixes before calling Ollama."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents([DOC_PREFIX + t for t in texts])

        def embed_query(self, text: str) -> list[float]:
            return super().embed_query(QUERY_PREFIX + text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await super().aembed_documents([DOC_PREFIX + t for t in texts])

        async def aembed_query(self, text: str) -> list[float]:
            return await super().aembed_query(QUERY_PREFIX + text)

    return PrefixedOllamaEmbeddings(model=EMBED_MODEL)


# --- index/embedding compatibility guard -----------------------------------
# An index built with one embedding setup and queried with another returns
# plausible-looking but degraded results, with no error. These two functions
# record the settings at ingest time and warn at ask time if they've changed.

INDEX_CONFIG_NAME = "embedding_config.json"


def write_index_config(index_dir: Path) -> None:
    (index_dir / INDEX_CONFIG_NAME).write_text(
        json.dumps({
            "embed_model": EMBED_MODEL,
            "prefixes": USE_EMBED_PREFIXES,
            "chunk_words": CHUNK_WORDS,
            "chunk_overlap": CHUNK_OVERLAP,
        }, indent=2),
        encoding="utf-8",
    )


def check_index_config(index_dir: Path) -> None:
    path = index_dir / INDEX_CONFIG_NAME
    if not path.exists():
        print(f"  ! {index_dir.name} was built before prefix support was added.")
        print("    Re-run ingest, or retrieval quality will be worse than it should be.")
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if cfg.get("embed_model") != EMBED_MODEL or cfg.get("prefixes") != USE_EMBED_PREFIXES:
        print(f"  ! {index_dir.name} was built with "
              f"model={cfg.get('embed_model')!r}, prefixes={cfg.get('prefixes')}; "
              f"you are now querying with "
              f"model={EMBED_MODEL!r}, prefixes={USE_EMBED_PREFIXES}.")
        print("    Re-run ingest to rebuild the index with the current settings.")


# ============================================================================
# INGEST
# ============================================================================

EMBED_BATCH = int(os.environ.get("EMBED_BATCH", 16))


def build_faiss_batched(chunks: list[Document], embeddings, batch: int = EMBED_BATCH) -> FAISS:
    """Embed in small batches.

    OllamaEmbeddings.embed_documents() sends every text in one request, which
    at ~1300 chunks kills Ollama's model-runner subprocess.
    """
    vectorstore = FAISS.from_documents(chunks[:batch], embeddings)
    for i in range(batch, len(chunks), batch):
        vectorstore.add_documents(chunks[i:i + batch])
        print(f"    embedded {min(i + batch, len(chunks))}/{len(chunks)}",
              end="\r", flush=True)
    print(f"    embedded {len(chunks)}/{len(chunks)}")
    return vectorstore


def cmd_ingest(args) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    t0 = time.time()
    embeddings = get_embeddings()
    print(f"Embedding model: {EMBED_MODEL}  "
          f"(nomic task prefixes: {'on' if USE_EMBED_PREFIXES else 'off'})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS
    )

    print("=== PRIMARY LAYER ===")
    print(f"Reading PDFs from {PAPERS_PRIMARY_DIR.resolve()}")
    primary_docs = load_layer_documents(PAPERS_PRIMARY_DIR, "primary", keep_refs=args.keep_refs)
    primary_chunks = splitter.split_documents(primary_docs)

    print("\n=== SECONDARY LAYER ===")
    print(f"Reading PDFs from {PAPERS_SECONDARY_DIR.resolve()}")
    secondary_docs = load_layer_documents(PAPERS_SECONDARY_DIR, "secondary", keep_refs=args.keep_refs)
    secondary_chunks = splitter.split_documents(secondary_docs)

    print(f"\n{len(primary_chunks)} primary chunks, {len(secondary_chunks)} secondary chunks")

    if primary_chunks:
        print("\nEmbedding primary layer...")
        build_faiss_batched(primary_chunks, embeddings).save_local(str(INDEX_PRIMARY_DIR))
        write_index_config(INDEX_PRIMARY_DIR)
        print(f"Primary index saved to {INDEX_PRIMARY_DIR.resolve()}")

    if secondary_chunks:
        print("\nEmbedding secondary layer...")
        build_faiss_batched(secondary_chunks, embeddings).save_local(str(INDEX_SECONDARY_DIR))
        write_index_config(INDEX_SECONDARY_DIR)
        print(f"Secondary index saved to {INDEX_SECONDARY_DIR.resolve()}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print('Now run:  python rag_tiered_langchain.py ask "your question"')


# ============================================================================
# RETRIEVAL HELPERS
# ============================================================================

def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{i+1}] {d.metadata.get('title', '')} — {d.metadata.get('source', '')}, "
        f"p.{d.metadata.get('page', '?')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def build_ensemble_retriever(vectorstore: FAISS, bm25: BM25Retriever,
                             k: int, alpha: float) -> EnsembleRetriever:
    """Vector search + BM25 keyword search blended by alpha.

    BM25 is passed in already built -- tokenising 1,300 chunks per question
    would dominate a sweep. Its k is mutated here, safe only while calls are
    sequential (see _ASK_LOCK).
    """
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[alpha, 1 - alpha],
    )


def load_layer(index_dir: Path, embeddings, quiet: bool = False
               ) -> tuple[FAISS | None, list[Document]]:
    if not index_dir.exists():
        return None, []
    if not quiet:
        check_index_config(index_dir)
    vectorstore = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    chunks = list(vectorstore.docstore._dict.values())
    return vectorstore, chunks


# ============================================================================
# TESTING API -- retrieval and answering as functions that return data rather
# than printing. The CLI below is a thin wrapper over these.
# ============================================================================

@dataclass
class RunConfig:
    """The five retrieval-time settings, changeable per call without
    re-indexing. Chunk size and overlap are fixed at ingest, so not here."""
    kp: int = K_PRIMARY_DEFAULT
    kb: int = K_SECONDARY_DEFAULT
    alpha: float = ALPHA_DEFAULT
    primary_threshold: float = PRIMARY_SCORE_THRESHOLD
    secondary_threshold: float = SECONDARY_SCORE_THRESHOLD
    investigate: bool = True

    @classmethod
    def from_args(cls, args) -> "RunConfig":
        return cls(
            kp=args.kp,
            kb=args.kb,
            alpha=args.alpha,
            primary_threshold=args.tp,
            secondary_threshold=args.ts,
            investigate=not args.no_investigate,
        )

    def label(self) -> str:
        """Compact identifier for logging a sweep row."""
        return (f"kp{self.kp}_kb{self.kb}_a{self.alpha:g}"
                f"_tp{self.primary_threshold:g}_ts{self.secondary_threshold:g}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Corpus:
    """Both indexes, loaded once and reused across every question and config."""
    primary_vs: FAISS | None = None
    primary_chunks: list[Document] = field(default_factory=list)
    primary_bm25: BM25Retriever | None = None
    secondary_vs: FAISS | None = None
    secondary_chunks: list[Document] = field(default_factory=list)
    secondary_bm25: BM25Retriever | None = None
    primary_index_dir: Path | None = None
    secondary_index_dir: Path | None = None

    def index_settings(self) -> dict:
        """Chunk settings recorded when these indexes were built. Useful for
        labelling sweep rows, since chunk size is not in RunConfig."""
        out = {}
        for name, d in (("primary", self.primary_index_dir),
                        ("secondary", self.secondary_index_dir)):
            if d and (d / INDEX_CONFIG_NAME).exists():
                try:
                    out[name] = json.loads((d / INDEX_CONFIG_NAME).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        return out


def load_corpus(primary_index_dir: Path | None = None,
                secondary_index_dir: Path | None = None,
                embeddings=None,
                quiet: bool = False) -> Corpus:
    """Load both indexes and build the BM25 retrievers once.

    Pass explicit directories when sweeping chunk size.
    """
    primary_index_dir = primary_index_dir or INDEX_PRIMARY_DIR
    secondary_index_dir = secondary_index_dir or INDEX_SECONDARY_DIR
    embeddings = embeddings or get_embeddings()

    corpus = Corpus(primary_index_dir=primary_index_dir,
                    secondary_index_dir=secondary_index_dir)
    corpus.primary_vs, corpus.primary_chunks = load_layer(
        primary_index_dir, embeddings, quiet=quiet)
    corpus.secondary_vs, corpus.secondary_chunks = load_layer(
        secondary_index_dir, embeddings, quiet=quiet)

    if corpus.primary_chunks:
        corpus.primary_bm25 = BM25Retriever.from_documents(corpus.primary_chunks)
    if corpus.secondary_chunks:
        corpus.secondary_bm25 = BM25Retriever.from_documents(corpus.secondary_chunks)
    return corpus


def doc_pages(docs: list[Document]) -> list[tuple[str, int | None]]:
    """(source_pdf, page) for each retrieved chunk -- what Hit Rate and MRR
    are scored against. Page-level labels survive re-chunking; chunk ids do not."""
    return [(d.metadata.get("source", ""), d.metadata.get("page")) for d in docs]


@dataclass
class Retrieval:
    """Everything retrieval produced, with no LLM involved. Free to compute,
    which is what makes the k / alpha / threshold sweeps cheap."""
    question: str
    primary_docs: list[Document] = field(default_factory=list)
    secondary_docs: list[Document] = field(default_factory=list)
    primary_top_score: float = 0.0
    secondary_top_score: float = 0.0
    has_primary: bool = False
    has_secondary: bool = False
    seconds: float = 0.0
    # retrievers are kept so the investigator agent can reuse them
    primary_retriever: EnsembleRetriever | None = None
    secondary_retriever: EnsembleRetriever | None = None

    @property
    def primary_pages(self): return doc_pages(self.primary_docs)

    @property
    def secondary_pages(self): return doc_pages(self.secondary_docs)

    @property
    def refused(self) -> bool:
        """True when both layers fell below threshold -- the 'not in your
        papers' path, and the pass condition for the unanswerable questions."""
        return not self.has_primary and not self.has_secondary


def retrieve(question: str, corpus: Corpus, cfg: RunConfig | None = None) -> Retrieval:
    """Run both layers and apply the thresholds. No LLM call, no printing."""
    cfg = cfg or RunConfig()
    t0 = time.time()
    r = Retrieval(question=question)

    if corpus.primary_vs is not None and corpus.primary_bm25 is not None:
        r.primary_retriever = build_ensemble_retriever(
            corpus.primary_vs, corpus.primary_bm25, cfg.kp, cfg.alpha)
        r.primary_docs = r.primary_retriever.invoke(question)
        scored = corpus.primary_vs.similarity_search_with_relevance_scores(question, k=1)
        if scored:
            r.primary_top_score = scored[0][1]

    if corpus.secondary_vs is not None and corpus.secondary_bm25 is not None:
        r.secondary_retriever = build_ensemble_retriever(
            corpus.secondary_vs, corpus.secondary_bm25, cfg.kb, cfg.alpha)
        r.secondary_docs = r.secondary_retriever.invoke(question)
        scored = corpus.secondary_vs.similarity_search_with_relevance_scores(question, k=1)
        if scored:
            r.secondary_top_score = scored[0][1]

    r.has_primary = bool(r.primary_docs) and r.primary_top_score > cfg.primary_threshold
    r.has_secondary = bool(r.secondary_docs) and r.secondary_top_score > cfg.secondary_threshold
    r.seconds = time.time() - t0
    return r


@dataclass
class AnswerResult:
    """The full pipeline's output, including the intermediate steps that the
    agent test in step 8 needs to score."""
    question: str
    text: str = ""
    refused: bool = False
    retrieval: Retrieval | None = None
    disagreement: str | None = None
    investigation: str | None = None
    investigation_failed: str | None = None
    prompt_chars: int = 0
    generation_seconds: float = 0.0


# ============================================================================
# INVESTIGATOR AGENT -- invoked only when a disagreement is detected. It
# chooses its own follow-up searches.
# ============================================================================

def detect_disagreement(llm, primary_text: str, secondary_text: str, question: str) -> str | None:
    """One LLM call: do these excerpts disagree? Description, or None."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the two sets of excerpts below. If they disagree "
                   "on something relevant to the question, describe the "
                   "disagreement in one sentence. If they do not disagree, "
                   "respond with exactly: NONE"),
        ("human", "PRIMARY:\n{primary}\n\nBACKGROUND:\n{secondary}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {"primary": primary_text, "secondary": secondary_text, "question": question}
    ).strip()
    return None if result.upper().startswith("NONE") else result


def build_investigator(llm, primary_retriever, secondary_retriever):
    """An agent with two search tools, which it queries as it chooses.

    The tool docstrings below are read by the model to pick a tool -- they are
    part of the prompt, not documentation. Edit them with that in mind.
    """

    @tool
    def search_primary_papers(query: str) -> str:
        """Search Kurokawa's primary papers for additional context -- e.g.
        methodology, time period, sample data, or evaluation metric."""
        docs = primary_retriever.invoke(query)
        return format_docs(docs) if docs else "No results."

    @tool
    def search_background_literature(query: str) -> str:
        """Search the background literature for additional context -- e.g.
        methodology, time period, sample data, or evaluation metric."""
        docs = secondary_retriever.invoke(query)
        return format_docs(docs) if docs else "No results."

    tools = [search_primary_papers, search_background_literature]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You investigate a disagreement between Kurokawa's primary papers "
            "and background literature. Use the search tools -- with your own "
            "chosen queries -- to look for details that might explain the "
            "disagreement (different time period, different methodology, "
            "different loss metric, different data). Search at least once "
            "before concluding. Finish with a short paragraph giving the most "
            "likely reason for the disagreement."
        ),
    )


# ============================================================================
# ASK
# ============================================================================

def build_prompt(question: str, r: Retrieval, investigation: str | None) -> tuple[str, str]:
    """Assemble the tiered prompt. Returns (system, user)."""
    primary_text = format_docs(r.primary_docs) if r.has_primary else ""
    secondary_text = format_docs(r.secondary_docs) if r.has_secondary else ""

    if r.has_primary:
        prompt_text = f"PRIMARY (Kurokawa 2026):\n\n{primary_text}"
        if r.has_secondary:
            prompt_text += f"\n\n---\n\nBACKGROUND LITERATURE:\n\n{secondary_text}"
        if investigation:
            prompt_text += f"\n\n---\n\nINVESTIGATION OF DISAGREEMENT:\n\n{investigation}"
        prompt_text += f"\n\n---\n\nQuestion: {question}"
        system = (SYSTEM_PROMPT +
                  "\n\nYour answer should be grounded in Kurokawa's PRIMARY work. "
                  "You MUST MUST MUST reference BACKGROUND LITERATURE if it exists to support or contextualize, "
                  "but Kurokawa's findings are authoritative.")
    else:
        prompt_text = f"BACKGROUND LITERATURE:\n\n{secondary_text}\n\n---\n\nQuestion: {question}"
        system = (SYSTEM_PROMPT +
                  "\n\nYour answer is based on BACKGROUND LITERATURE only, "
                  "not Kurokawa's primary papers.")
    return system, prompt_text


REFUSAL_TEXT = "Information is not available in your papers."


def answer(question: str, corpus: Corpus, cfg: RunConfig | None = None,
           llm=None, verbose: bool = False, show: bool = False) -> AnswerResult:
    """Full pipeline: retrieve, optionally investigate, generate.

    Returns an AnswerResult. verbose=True adds the CLI's running commentary.
    """
    cfg = cfg or RunConfig()
    res = AnswerResult(question=question)

    r = retrieve(question, corpus, cfg)
    res.retrieval = r

    if show:
        print("\n--- PRIMARY LAYER ---")
        for i, d in enumerate(r.primary_docs, 1):
            print(f"[{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}\n    {d.page_content[:200]}...\n")
        print("--- SECONDARY LAYER ---")
        for i, d in enumerate(r.secondary_docs, 1):
            print(f"[{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}\n    {d.page_content[:200]}...\n")

    if r.refused:
        res.refused = True
        res.text = REFUSAL_TEXT
        if verbose:
            print("\n--- No relevant passages found in either layer. ---")
            print(REFUSAL_TEXT)
        return res

    llm = llm or get_llm()
    primary_text = format_docs(r.primary_docs) if r.has_primary else ""
    secondary_text = format_docs(r.secondary_docs) if r.has_secondary else ""

    # --- investigate disagreements, only when both layers are actually in play ---
    if cfg.investigate and r.has_primary and r.has_secondary:
        res.disagreement = detect_disagreement(llm, primary_text, secondary_text, question)
        if res.disagreement:
            if verbose:
                print(f"\n--- possible disagreement detected: {res.disagreement} ---")
                print("--- investigating (agent may run its own follow-up searches)... ---")
            # get_tool_llm(), not llm -- the agent binds function tools, which
            # some OpenAI models only accept via the Responses API.
            investigator = build_investigator(
                get_tool_llm(), r.primary_retriever, r.secondary_retriever
            )
            try:
                agent_out = investigator.invoke({
                    "messages": [{
                        "role": "user",
                        "content": f"Disagreement to investigate: {res.disagreement}\nOriginal question: {question}",
                    }]
                })
                # create_agent returns the full conversation state; the agent's
                # final reply is the last message.
                res.investigation = agent_out["messages"][-1].content
            except Exception as exc:
                # A failed investigation should degrade to "no investigation",
                # not lose the answer the user actually asked for. Recorded on
                # the result so a sweep can count how often this happens.
                res.investigation_failed = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"--- investigation failed ({res.investigation_failed}) ---")
                    print("--- continuing without it; use --no-investigate to skip this step ---")

    system, prompt_text = build_prompt(question, r, res.investigation)
    res.prompt_chars = len(prompt_text)

    if verbose:
        provider = os.environ.get("PROVIDER", "ollama")
        print(f"\nAsking {provider} with ~{res.prompt_chars // 4} tokens...")

    final_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])
    chain = final_prompt | llm | StrOutputParser()

    t0 = time.time()
    res.text = chain.invoke({"input": prompt_text})
    res.generation_seconds = time.time() - t0

    if verbose:
        print(f"\n--- answer ({res.generation_seconds:.0f}s) ---")
        print(res.text)
    return res


def cmd_ask(args) -> None:
    cfg = RunConfig.from_args(args)
    corpus = load_corpus(
        Path(args.index_primary) if args.index_primary else None,
        Path(args.index_secondary) if args.index_secondary else None,
    )
    llm = get_llm()

    if corpus.primary_vs:
        print(f"{len(corpus.primary_chunks)} primary chunks loaded.")
    else:
        print("Warning: No primary index found. Only searching secondary layer.")
    if corpus.secondary_vs:
        print(f"{len(corpus.secondary_chunks)} secondary chunks loaded.")
    else:
        print("Warning: No secondary index found. Only searching primary layer.")

    question = args.q or " ".join(args.question)
    if question:
        answer(question, corpus, cfg, llm=llm, verbose=True, show=args.show)
        return

    print("Ask away (blank line or Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        answer(q, corpus, cfg, llm=llm, verbose=True, show=args.show)


# ============================================================================
# WEB FRONT-END -- one page from the standard library. No Flask, no build
# step. Calls the same cmd_ingest() and answer() the CLI does.
#
#     python tiered_rag_langchain.py serve
# ============================================================================

# Printed on startup and at the foot of the page, to confirm which file is
# actually running.
BUILD = "v5 — config D (kp 8 / kb 10)"

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiered RAG</title>
<style>
  :root {
    --bg:#f6f8fa; --card:#fff; --ink:#16202b; --soft:#5a6673; --line:#d3dce6;
    --accent:#1f55c7; --accent-soft:#e5edfc; --warn:#a94a22; --ok:#226b52;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#0f151c; --card:#161f29; --ink:#e3e9f0; --soft:#a9b7c6; --line:#2c3945;
      --accent:#85aaf5; --accent-soft:#1b2a44; --warn:#e09468; --ok:#6ec49f;
    }
  }
  * { box-sizing:border-box; }
  body {
    margin:0; padding:1.5rem 1rem 3rem; background:var(--bg); color:var(--ink);
    font:14px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif;
  }
  .wrap { max-width:64rem; margin:0 auto; display:flex; flex-direction:column; gap:1.15rem; }
  /* One column on a narrow window; main + sidebar side by side on a wide one. */
  .cols { display:grid; grid-template-columns:1fr; gap:1.15rem; align-items:start; }
  /* 3fr / 2fr = a 60 / 40 split. minmax(0,...) stops a long unbroken
     filename from forcing its column wider than its share. */
  @media (min-width:60rem) { .cols { grid-template-columns:minmax(0,3fr) minmax(0,2fr); } }
  .main { display:flex; flex-direction:column; gap:1.15rem; min-width:0; }
  .side { display:flex; flex-direction:column; gap:1.15rem; }
  @media (min-width:60rem) { .side { position:sticky; top:1.5rem; } }
  h1 { font-size:1.15rem; margin:0; letter-spacing:-.02em; }
  .sub { color:var(--soft); font-size:.82rem; margin:.25rem 0 0; }
  .card {
    background:var(--card); border:1px solid var(--line); border-radius:8px;
    padding:1rem 1.1rem; display:flex; flex-direction:column; gap:.7rem;
  }
  .row { display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; }
  h2 { font-size:.66rem; text-transform:uppercase; letter-spacing:.1em;
       color:var(--soft); margin:0; font-weight:600; }
  button {
    font:inherit; font-weight:600; padding:.45rem .95rem; border-radius:6px;
    border:1px solid var(--accent); background:var(--accent); color:#fff;
    cursor:pointer;
  }
  button.ghost { background:transparent; color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  textarea {
    font:inherit; width:100%; min-height:4.5rem; resize:vertical; padding:.6rem;
    border:1px solid var(--line); border-radius:6px;
    background:var(--bg); color:var(--ink);
  }
  textarea:focus, button:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
  label.check { display:flex; gap:.4rem; align-items:center; font-size:.82rem; color:var(--soft); }
  pre.log {
    margin:0; max-height:14rem; overflow:auto; font-size:.72rem; line-height:1.45;
    font-family:ui-monospace, Consolas, monospace; white-space:pre-wrap;
    background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:.7rem;
  }
  .answer { font-size:.92rem; }
  .answer .cite { color:var(--accent); font-weight:600; }
  .meta { font-size:.75rem; color:var(--soft); display:flex; gap:1rem; flex-wrap:wrap; }
  .note { border-left:2px solid var(--warn); background:var(--card);
          padding:.55rem .75rem; font-size:.82rem; border-radius:4px; }
  .note b { color:var(--warn); font-size:.65rem; text-transform:uppercase;
            letter-spacing:.08em; display:block; }
  .hide { display:none; }
  /* --- papers sidebar --- */
  .pgroup + .pgroup { margin-top:.9rem; }
  .plabel { font-size:.66rem; text-transform:uppercase; letter-spacing:.09em;
            color:var(--soft); font-weight:600; display:flex;
            justify-content:space-between; gap:.5rem;
            border-bottom:1px solid var(--line); padding-bottom:.3rem;
            margin-bottom:.45rem; }
  .plabel .count { font-variant-numeric:tabular-nums; }
  .paper { padding:.3rem 0; border-bottom:1px solid var(--line); }
  .paper:last-child { border-bottom:none; }
  .ptitle { display:block; font-size:.78rem; line-height:1.35; }
  .pfile { display:block; font-size:.68rem; color:var(--soft); word-break:break-all;
           font-family:ui-monospace, Consolas, monospace; margin-top:.1rem; }
  .dot { width:.5rem; height:.5rem; border-radius:50%; background:var(--soft); }
  .dot.busy { background:var(--accent); animation:p 1s ease-in-out infinite; }
  .dot.ok { background:var(--ok); } .dot.bad { background:var(--warn); }
  @keyframes p { 50% { opacity:.25; } }
  @media (prefers-reduced-motion: reduce) { .dot.busy { animation:none; } }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Multi-tiered RAG</h1>
    <p class="sub">A multi-tiered Retrieval-Augmented Generation (RAG) script which answers questions about the primary papers and uses background literature as secondary material. Hyperparameters have already been tuned to the specific \papers1 and \papers2.</p>
    <p class="sub">Ensure the proper PDFs are in <code>\papers1</code> (primary papers) and <code>\papers2 </code> (secondary papers).</p>
  </header>

  <div class="cols">
  <div class="main">

  <!-- 1. INGEST -->
  <section class="card">
    <h2>Ingest the primary and secondary papers and turn them into a vector database</h2>
    <div class="row">
      <button id="ingestBtn">Ingest</button>
      <span class="sub" id="ingestMsg">Ingestion is needed only once.</span>
    </div>
    <pre class="log hide" id="ingestLog"></pre>
  </section>

  <!-- 2. ASK -->
  <section class="card">
    <h2>Ask any question relating to the primary papers, background literature, or anything else.</h2>
    <textarea id="q" placeholder="e.g. Is week-ahead TTF price volatility forecastable?"></textarea>
    <div class="row">
      <button id="askBtn">Ask</button>
      <label class="check"><input type="checkbox" id="inv" checked> Investigate disagreements with an AI agent</label>
      <span class="sub" id="askMsg"></span>
    </div>
  </section>

  <!-- 3. ANSWER -->
  <section class="card hide" id="answerCard">
    <h2>Answer</h2>
    <div class="answer" id="answer"></div>
    <div class="note hide" id="disagreement"><b>Disagreement detected</b><span></span></div>
    <div class="note hide" id="investigation"><b>Investigation</b><span></span></div>
    <div class="meta" id="meta"></div>
  </section>

  </div><!-- /main -->

  <!-- 4. THE PAPERS -->
  <aside class="side">
    <section class="card">
      <h2>Papers in the corpus</h2>
      <div id="papers"><p class="sub">Loading…</p></div>
    </section>
  </aside>

  </div><!-- /cols -->

  <p class="sub" id="build"></p>

</div>
<script>
const $ = id => document.getElementById(id);

// Answers arrive as plain text with **bold** and [1] citations. Escape first,
// then apply those two, so nothing in a PDF can inject markup.
function render(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>')
          .replace(/\n/g, "<br>");
}

// Tolerates a missing element, so the status dots can be removed from the
// HTML without every call site having to be edited too.
function setDot(el, state) { if (el) el.className = "dot" + (state ? " " + state : ""); }

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

// --- ingest, with polling because it takes minutes -------------------------
let pollTimer = null;

async function pollIngest() {
  const s = await (await fetch("/api/status")).json();
  $("ingestLog").textContent = s.log.join("\n");
  $("ingestLog").scrollTop = $("ingestLog").scrollHeight;
  if (s.running) return;
  clearInterval(pollTimer); pollTimer = null;
  $("ingestBtn").disabled = false;
  setDot($("ingestDot"), s.error ? "bad" : "ok");
  $("ingestMsg").textContent = s.error ? ("Failed: " + s.error) : "Done.";
  loadPapers();          // chunk counts have changed
}

$("ingestBtn").onclick = async () => {
  $("ingestBtn").disabled = true;
  $("ingestLog").classList.remove("hide");
  $("ingestLog").textContent = "";
  setDot($("ingestDot"), "busy");
  $("ingestMsg").textContent = "Reading PDFs and embedding — this takes a couple of minutes.";
  try {
    await post("/api/ingest");
    pollTimer = setInterval(pollIngest, 1000);
  } catch (e) {
    $("ingestBtn").disabled = false;
    setDot($("ingestDot"), "bad");
    $("ingestMsg").textContent = e.message;
  }
};

// --- ask -------------------------------------------------------------------
async function ask() {
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  $("askBtn").disabled = true;
  setDot($("askDot"), "busy");
  $("askMsg").textContent = "Searching and writing…";
  try {
    const d = await post("/api/ask", {question: q, investigate: $("inv").checked});
    $("answerCard").classList.remove("hide");
    $("answer").innerHTML = render(d.answer);
    for (const [id, val] of [["disagreement", d.disagreement], ["investigation", d.investigation]]) {
      const box = $(id);
      box.classList.toggle("hide", !val);
      if (val) box.querySelector("span").innerHTML = render(val);
    }
    $("meta").textContent =
      `${d.n_primary} primary + ${d.n_secondary} background chunks · ${d.seconds}s`
      + (d.refused ? " · refused (nothing relevant found)" : "");
    setDot($("askDot"), "ok");
    $("askMsg").textContent = "";
  } catch (e) {
    setDot($("askDot"), "bad");
    $("askMsg").textContent = e.message;
  } finally {
    $("askBtn").disabled = false;
  }
}

$("askBtn").onclick = ask;
// Ctrl/Cmd+Enter submits, plain Enter keeps making new lines.
$("q").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask(); }
});

// --- the papers sidebar ----------------------------------------------------
// Titles are lifted out of the PDFs themselves, so they are untrusted text.
// Everything here is built with createElement/textContent -- never innerHTML --
// so a title containing "<script>" ends up as visible characters, not markup.
function paperGroup(label, info, indexed) {
  const g = document.createElement("div");
  g.className = "pgroup";

  const head = document.createElement("div");
  head.className = "plabel";
  const name = document.createElement("span");
  name.textContent = label;
  const count = document.createElement("span");
  count.className = "count";
  const n = info.papers.length;
  count.textContent = n + (n === 1 ? " PDF" : " PDFs")
                      + (indexed != null ? " · " + indexed + " chunks" : "");
  head.append(name, count);
  g.append(head);

  if (!info.exists) {
    const p = document.createElement("p");
    p.className = "sub";
    p.textContent = "Folder " + info.dir + " not found.";
    g.append(p);
    return g;
  }
  if (!n) {
    const p = document.createElement("p");
    p.className = "sub";
    p.textContent = "No PDFs in " + info.dir + ".";
    g.append(p);
    return g;
  }

  for (const paper of info.papers) {
    const row = document.createElement("div");
    row.className = "paper";
    const t = document.createElement("span");
    t.className = "ptitle";
    t.textContent = paper.title;
    const f = document.createElement("span");
    f.className = "pfile";
    f.textContent = paper.file;
    row.append(t, f);
    g.append(row);
  }
  return g;
}

async function loadPapers() {
  const box = $("papers");
  try {
    const r = await fetch("/api/papers");
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
    box.textContent = "";
    box.append(paperGroup("Primary", d.primary, d.indexed.primary));
    box.append(paperGroup("Secondary", d.secondary, d.indexed.secondary));
    $("build").textContent = d.build + " · " + d.script;
  } catch (e) {
    box.textContent = "";
    const p = document.createElement("p");
    p.className = "sub";
    p.textContent = "Could not list the papers: " + e.message;
    box.append(p);
  }
}

loadPapers();
</script>
</body>
</html>
"""


# The corpus takes seconds to load and the BM25 index seconds more, so both are
# built once on the first question and reused. Cleared after an ingest, since
# the files on disk have changed underneath them.
_WEB = {"corpus": None, "llm": None}

# build_ensemble_retriever mutates bm25.k, so two questions answered at the
# same time could read each other's k. ThreadingHTTPServer makes that possible,
# hence the lock.
_ASK_LOCK = threading.Lock()

_INGEST = {"running": False, "log": [], "error": None}


class _LogTee:
    """Collects cmd_ingest's print output for the browser.

    A tee, not a redirect: redirect_stdout swaps sys.stdout process-wide, so a
    plain redirect would swallow everything else printed during an ingest.
    """

    def __init__(self, sink: list[str], limit: int = 400, echo=None):
        self.sink, self.limit, self.buf = sink, limit, ""
        self.echo = echo

    def write(self, s: str) -> int:
        if self.echo is not None:
            try:
                self.echo.write(s)
                self.echo.flush()
            except Exception:
                pass
        self.buf += s
        # ingest uses \r for its embedding counter, so split on both
        while True:
            i = min((x for x in (self.buf.find("\n"), self.buf.find("\r")) if x >= 0),
                    default=-1)
            if i < 0:
                break
            line, self.buf = self.buf[:i].rstrip(), self.buf[i + 1:]
            if line:
                self.sink.append(line)
                del self.sink[:-self.limit]
        return len(s)

    def flush(self) -> None:
        pass


def _run_ingest() -> None:
    import contextlib
    from types import SimpleNamespace

    _INGEST.update(running=True, log=[], error=None)
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(_LogTee(_INGEST["log"], echo=real_stdout)):
            cmd_ingest(SimpleNamespace(keep_refs=False))
        _WEB["corpus"] = None          # force a reload with the new index
    except Exception as exc:
        _INGEST["error"] = f"{type(exc).__name__}: {exc}"
        _INGEST["log"].append(f"ERROR: {_INGEST['error']}")
    finally:
        _INGEST["running"] = False


# --- the papers sidebar -----------------------------------------------------
# Display only; nothing here affects retrieval. guess_title() takes the first
# plausible line of page 1, which fails on cover pages, mastheads and junk
# metadata -- those are overridden here. Anything unlisted falls back to
# guess_title(). Keys are filenames without ".pdf".
_TITLE_OVERRIDES: dict[str, str] = {
    "baumeister":
        "Forecasting Natural Gas Prices in Real Time",
    "nick_thoenes":
        "What Drives Natural Gas Prices? A Structural VAR Approach",
    "rubaszek_2024":
        "A Note on Natural Gas Price Transmission from TTF to Other European Hubs",
    "rootjournalmanager42_ijeep_10192_obadi_okey":
        "Driving Fundamentals of Natural Gas Price in Europe",
    "giignl_2026_annual_report":
        "GIIGNL Annual Report 2026 — The LNG Industry",
    "igu_world_lng_report_2026":
        "IGU World LNG Report 2026",
}


def _norm_key(stem: str) -> str:
    """Letters and digits only, so renames like "Nick Thoenes (1).pdf" match."""
    return re.sub(r"[^a-z0-9]", "", stem.lower())


_TITLE_OVERRIDES_NORM = {_norm_key(k): v for k, v in _TITLE_OVERRIDES.items()}


def _lookup_override(stem: str) -> str | None:
    key = _norm_key(stem)
    if key in _TITLE_OVERRIDES_NORM:
        return _TITLE_OVERRIDES_NORM[key]
    # Longest key first, so a short key cannot shadow a longer one.
    for k in sorted(_TITLE_OVERRIDES_NORM, key=len, reverse=True):
        if key.startswith(k) or k in key:
            return _TITLE_OVERRIDES_NORM[k]
    return None

# Cached on (path, mtime), so replacing a PDF invalidates its entry.
_TITLE_CACHE: dict[tuple[str, float], str] = {}


def _pdf_title(path: Path) -> str:
    """Best guess at a paper's title. Page 1 only, for speed."""
    override = _lookup_override(path.stem.strip())
    if override:
        return override

    key = (str(path), path.stat().st_mtime)
    if key in _TITLE_CACHE:
        return _TITLE_CACHE[key]

    meta_title, first_page = "", ""
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        with pymupdf.open(path) as doc:
            meta_title = (doc.metadata or {}).get("title") or ""
            if doc.page_count:
                first_page = doc[0].get_text("text")
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            meta_title = getattr(reader.metadata, "title", "") or ""
            if reader.pages:
                first_page = reader.pages[0].extract_text() or ""
        except Exception:
            pass

    try:
        title = guess_title(first_page, meta_title.strip(), path.stem)
    except Exception:
        title = path.stem
    _TITLE_CACHE[key] = title
    return title


def _papers_payload() -> dict:
    """What is on disk, plus how much is indexed. Never loads the corpus."""
    out: dict = {}
    for key, folder in (("primary", PAPERS_PRIMARY_DIR),
                        ("secondary", PAPERS_SECONDARY_DIR)):
        entry = {"dir": str(folder), "exists": folder.is_dir(), "papers": []}
        if entry["exists"]:
            for pdf in sorted(folder.glob("*.pdf")):
                entry["papers"].append({"file": pdf.name, "title": _pdf_title(pdf)})
        out[key] = entry

    corpus = _WEB["corpus"]
    out["indexed"] = {
        "primary": len(corpus.primary_chunks) if corpus else None,
        "secondary": len(corpus.secondary_chunks) if corpus else None,
    }
    out["build"] = BUILD
    out["script"] = Path(__file__).name
    return out


def _web_answer(question: str, investigate: bool) -> dict:
    if _WEB["corpus"] is None:
        _WEB["corpus"] = load_corpus()
    if _WEB["llm"] is None:
        _WEB["llm"] = get_llm()
    corpus = _WEB["corpus"]
    if corpus.primary_vs is None and corpus.secondary_vs is None:
        raise RuntimeError("No index found. Press Ingest first.")

    cfg = RunConfig(investigate=investigate)
    with _ASK_LOCK:
        res = answer(question, corpus, cfg, llm=_WEB["llm"])

    r = res.retrieval
    return {
        "answer": res.text,
        "refused": res.refused,
        "disagreement": res.disagreement,
        "investigation": res.investigation or res.investigation_failed,
        "n_primary": len(r.primary_docs) if r else 0,
        "n_secondary": len(r.secondary_docs) if r else 0,
        "seconds": round((r.seconds if r else 0) + res.generation_seconds, 1),
    }


def cmd_serve(args) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import webbrowser

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Without this the browser will happily serve you the page it
            # cached from an earlier version of this script, and edits appear
            # to have no effect.
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._json(200, dict(_INGEST))
            elif self.path == "/api/papers":
                try:
                    self._json(200, _papers_payload())
                except Exception as exc:
                    self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})

            if self.path == "/api/ingest":
                if _INGEST["running"]:
                    return self._json(409, {"error": "an ingest is already running"})
                threading.Thread(target=_run_ingest, daemon=True).start()
                return self._json(200, {"started": True})

            if self.path == "/api/ask":
                question = (payload.get("question") or "").strip()
                if not question:
                    return self._json(400, {"error": "no question"})
                try:
                    return self._json(200, _web_answer(
                        question, bool(payload.get("investigate", True))))
                except Exception as exc:
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

            self._json(404, {"error": "unknown endpoint"})

        def log_message(self, fmt, *a):
            pass       # the browser's own console is enough

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving at {url}   (Ctrl-C to stop)")
    print(f"  build:           {BUILD}")
    print(f"  script:          {Path(__file__).resolve()}")
    print(f"  primary papers:  {PAPERS_PRIMARY_DIR.resolve()}")
    print(f"  secondary papers:{PAPERS_SECONDARY_DIR.resolve()}")
    print(f"  primary index:   {INDEX_PRIMARY_DIR}")
    print(f"  secondary index: {INDEX_SECONDARY_DIR}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="LangChain multi-layer RAG with disagreement-investigating agent")
    subparsers = ap.add_subparsers(dest="command")

    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("--keep-refs", action="store_true")

    ask_p = subparsers.add_parser("ask")
    ask_p.add_argument("question", nargs="*")
    ask_p.add_argument("-q", dest="q")
    ask_p.add_argument("-kp", type=int, default=K_PRIMARY_DEFAULT,
                       help="chunks retrieved from the primary layer")
    ask_p.add_argument("-kb", type=int, default=K_SECONDARY_DEFAULT,
                       help="chunks retrieved from the secondary layer")
    ask_p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                       help="vector-vs-BM25 blend; 1.0 = vector only, 0.0 = keyword only")
    ask_p.add_argument("--tp", type=float, default=PRIMARY_SCORE_THRESHOLD,
                       help="primary relevance threshold")
    ask_p.add_argument("--ts", type=float, default=SECONDARY_SCORE_THRESHOLD,
                       help="secondary relevance threshold")
    ask_p.add_argument("--index-primary", default=None,
                       help="primary index folder (default: INDEX_PRIMARY_DIR). "
                            "The sweep writes index_p_<chunk>_<overlap>.")
    ask_p.add_argument("--index-secondary", default=None,
                       help="secondary index folder (default: INDEX_SECONDARY_DIR)")
    ask_p.add_argument("--show", action="store_true")
    ask_p.add_argument("--no-investigate", action="store_true",
                        help="skip the disagreement-investigating agent")

    serve_p = subparsers.add_parser("serve", help="browser front-end (ingest + ask)")
    serve_p.add_argument("--host", default="127.0.0.1",
                         help="127.0.0.1 keeps it on this machine only")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--no-browser", action="store_true",
                         help="don't open a browser window automatically")

    args = ap.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())