# A multi-tier RAG system for research papers — what it is, and how it was tuned

## The multi-tier RAG system

This RAG — retrieval-augmented generation — model not only extracts any relevant passages from the econometric natural gas papers, but also any relevant extracts from background literature papers to respond to a user's question in the prompt. The primary econometric gas papers and the secondary background literature papers must be loaded in \papers1 and \papers2 for the RAG system to function. The primary and secondary papers are the primary and secondary 'tiers'.

If there is no relevant passage in response to the user's question, the model will return a response to that effect. If there is a discrepany between the primary and secondary papers, an AI agent will called to address the discrepancy, should this option be used (it will appear as a box to tick in the user interface).

- **`papers1`** — two Kurokawa PDF papers on forecasting Dutch TTF gas prices. The *primary* tier. The model is told these are authoritative.
- **`papers2`** — eight background PDF papers from the wider literature. The *secondary* tier. Context and comparison.
  
For this particular multi-tiered RAG to work as intended, the exact 2 + 8 = 10 papers must be used. The RAG is tuned to exactly those 10 papers.

## How the script works

The whole system is one file, `tiered_rag_langchain.py`, and it runs two ways: from the command line (`ingest`, then `ask "…"`), or as a simple local web page with an Ingest button, a question box, an answer pane, and the corpus listed down the side. There is also an option to call an AI agent if there is a discrepancy between the primary and secondary papers. 

`ingest`

Each PDF is read and cleaned: hyphens rejoined, broken line breaks repaired, reference lists cut off. The text is then chunked — cut into 300-word pieces, because whole papers are far too long to paste into a prompt. The 300-word size comes from tuning the multi-tiered RAG model.

Every chunk is then converted into an embedding: a list of numbers representing its meaning, arranged so that passages about similar things end up numerically close together. The embeddings are vectors. Searching is then a matter of finding the nearest vectors using the dot product. 

`ask`

1. Search both tiers. Each question triggers a search of the primary and secondary tiers every time simultaneously. The primary tier returns its best 3 chunks, the secondary its best 10. The 3 and 10 are tuned hyperparameters.
2. Check relevance. If neither tier matches well enough, the system says the answer isn't in the papers and stops.
3. Check for conflict. If both tiers have something, the model asks whether they disagree. If they do, an AI agent given specified search tools in the code runs follow-up searches to work out why: different market, different period, different method, etc.
4. Write the answer. Everything retrieved goes into one prompt, primary marked authoritative, and the model answers with citations `[1]`, `[2]`.

---

## How the multi-tiered RAG model was evaluated

For the evaluation, 50 test questions with known answers, each labelled with the paper and page, were generated. There were 20 questions relating to the primary tier (with 5 that overlap with the secondary tier), 20 to the secondary tier (with 5 that overlap with the primary tier), 5 where there is conflict between the tiers, and 5 where there is no answer in either tier.

These questions were then used to test and tune the multi-tiered RAG model.

## Stage one: non-LLM screen for best hyperparameter tuning candidates

The k's for each of the primary and secondary tiers, chunk size, overlap size, alpha (the balance between BM25 and exact-word searches) were tested. About 700 configurations were tested and hit rate and MRR (mean reciprocal rank) were calculated. These tests were used as they are (1) fast and (2) don't require LLMs - which usually require paid tokens.

- Hit Rate — did a chunk from the correct page come back at all?
- MRR — how near the top of the results was it?

**How many chunks you take is the only setting that matters.** Going from 2-and-1 chunks to 8-and-10 moves the score from 0.467 to 0.911. So, increasing chunk number improves model performance.

**Three settings did nothing measurable.** Chunk size anywhere from 150 to 400 words. Overlap between neighbouring chunks from 25 to 100 words. And alpha, the blend between meaning-search and keyword-search: *identical* results at every value across 640 configurations. So, these three hyperparameters did not measurably change model performance.

So, the chunk number is the hyperparamter to tune in the next step.

## Stage two: Hyperparameter tuning with LLM judging

Five configurations of varying primary and secondary chunk numbers went to RAGAS — a library that uses an LLM as a judge — for two measures:

- Precision — of the chunks retrieved, how many were actually useful?
- Recall — did they contain everything the answer needed?

About 2,300 judge calls were used.

| config | kp | kb | P prec | P rec | S prec | S rec | average |
|---|---|---|---|---|---|---|---|
| A | 3 | 3 | 0.960 | 0.873 | 0.958 | 0.755 | **0.8866** |
| B | 8 | 5 | 0.952 | 0.882 | 0.957 | 0.767 | **0.8899** |
| C | 8 | 5 | 0.927 | 0.924 | 0.921 | 0.696 | **0.8671** |
| D | 8 | 10 | 0.954 | 0.910 | 0.948 | 0.889 | **0.9253** |
| E | 3 | 10 | 0.962 | 0.795 | 0.948 | 0.853 | **0.8893** |

Every evaluation gain came from changing hyperparameters in the secondary tier. Taking more background chunks lifts its recall from 0.755 to 0.889. Taking more primary chunks does nothing because the primary tier is so small; 3 already returns a sixth of it and more just repeats.

Precision is a weak testing instrument. Its entire range across five configurations is 0.909 to 0.960.

When the primary and secondary precision and recall scores are averaged for each of A to E, D is the highest. They are all however close. D may win partly on having more chunks, and while we could test higher chunk numbers than in D, it would become more costly. So, D is chosen as the hyperparameter set for implementation.

## Stage three: Generation evaluation

Stages one and two only measured retrieval, i.e. whether the right passages were found. This stage measures what the model does with them in the answer generation.

Faithfulness asks whether each claim in the answer is actually supported by the chunks. It is the direct test for invented content.

D scored 0.935, evenly across both tiers: 0.927 on the primary questions, 0.945 on the secondary, and 0.921 on the conflict questions. One of the 45 judge calls failed, so the average is over 44.

All five deliberately unanswerable questions were correctly refused, each explaining what the papers cover instead. 

A second measure, response relevancy, was discarded. It runs a yes/no test for whether the answer sounds noncommittal, and any answer failing that test is forced to zero. The test cannot distinguish a negative finding from a non-answer, as in the present case where the main result of both primary papers is negative (i.e. TTF prices are not forecastable). "No, the price is not forecastable, and here is the evidence" has the same test results as "I don't know." Indeed, the measure zeroed the answers that state the main conclusion, with 14 of 45 scored zero.

The instruction in the prompt telling the model to decline when the excerpts do not cover something works reliably: all five unanswerable questions were correctly refused, each explaining what the papers do cover instead. 

## Finalised hyperparameters

**300-word chunks · 50-word overlap · 8 primary chunks · 10 secondary · 0.5 blend**
