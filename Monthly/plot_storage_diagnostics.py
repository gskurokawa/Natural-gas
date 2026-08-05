"""
plot_storage_diagnostics.py

Loads NG_m_final.csv and reproduces the storage-balance diagnostic charts
from the end of the original Natgas_monthly_data.py: these compare the
CALCULATED storage change (implied from the supply/demand balance --
production + LNG + pipeline flows - exports - distribution losses -
consumption) against the ACTUAL reported storage change (from AGSI), as a
sanity check on how well the constructed supply/demand balance lines up
with what storage data actually shows happened.

Usage:
    python plot_storage_diagnostics.py
"""

import pandas as pd
import plotly.express as px

DATA_PATH = "NG_m_final.csv"
POST_FILTER_START = "2022-01-01"  # the original script's code filters from
                                   # this date despite its chart title saying
                                   # "Post-2018" -- kept as-is here, see below


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_storage_change_comparison(df: pd.DataFrame):
    """Calculated vs. actual storage change over time (line plot)."""
    melted = df.melt(
        id_vars=["Date"],
        value_vars=["Calc_storage_change", "Act_storage_change"],
        var_name="Metric",
        value_name="Value",
    )
    fig = px.line(
        melted, x="Date", y="Value", color="Metric",
        title="Storage Change vs Average Storage",
        labels={"Value": "bcm", "Date": "Date"},
    )
    return fig


def plot_storage_change_scatter(df: pd.DataFrame, post_filter: bool = False):
    """
    Actual vs. calculated storage change (scatter, with an OLS trendline).

    Set post_filter=True to restrict to POST_FILTER_START onward -- useful
    for checking whether the calculated/actual relationship held up
    differently after the 2021-2023 European energy crisis.
    Requires the `statsmodels` package (used internally by plotly's
    trendline='ols').
    """
    data = df[df["Date"] >= POST_FILTER_START] if post_filter else df
    title = (
        f"Storage Change vs Average Storage (from {POST_FILTER_START})"
        if post_filter else "Storage Change vs Average Storage"
    )
    fig = px.scatter(
        data, x="Act_storage_change", y="Calc_storage_change",
        title=title,
        labels={
            "Act_storage_change": "Actual Storage Change (bcm)",
            "Calc_storage_change": "Calculated Storage Change (bcm)",
        },
        trendline="ols",
    )
    return fig


def plot_storage_discrepancy(df: pd.DataFrame):
    """Gap between actual and calculated storage change over time (line plot)."""
    fig = px.line(
        df, x="Date", y="Diff_storage_changes",
        title="Difference in Storage Changes over Time",
        labels={"Date": "Date", "Diff_storage_changes": "Difference in Storage Changes (bcm)"},
    )
    return fig


def main():
    df = load_data()

    figures = [
        plot_storage_change_comparison(df),
        plot_storage_discrepancy(df),
        plot_storage_change_scatter(df, post_filter=False),
        plot_storage_change_scatter(df, post_filter=True),
    ]
    for fig in figures:
        fig.show()


if __name__ == "__main__":
    main()
