#!/usr/bin/env python3
"""Reproducible A/B test analysis and PDF report for ab_data.csv.

The source data record a landing-page experiment (not a price or revenue test).
This script performs data-quality checks, cleans assignment mismatches and repeated
users, estimates the conversion effect, runs a two-sided two-proportion z-test,
computes confidence intervals, runs a binary-metric bootstrap, checks sample-ratio
balance, assesses detectable effect size, and writes the final report and figures.

Run from the repository root:
    python AB_Test_Analysis.py

Required packages are listed in requirements.txt.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from statistics import NormalDist
from textwrap import fill

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ab_data.csv"
REPORT_PATH = ROOT / "AB_Test_Pricing_Report_CORRECTED.pdf"
FIGURE_DIR = ROOT / "ab_test_figures"
SOURCE_URL = "https://github.com/aamir-gk2539-byte/DSML-Training/blob/main/ab_data.csv"
SOURCE_COMMIT = "94acc3eb1ec8bcba7e323018f20f63fa6e22be31"
REPORT_DATE = "28 September 2026"
AUTHOR = "Advik Singh"
REGISTRATION_NUMBER = "RA2411056030023"
BRANCH = "DS-A"
ALPHA = 0.05
BOOTSTRAP_REPS = 100_000
RANDOM_SEED = 42

# Visual system used both in the figures and in the PDF.
NAVY = "#12355B"
BLUE = "#2F80ED"
TEAL = "#13A8A8"
GOLD = "#F2B134"
RED = "#D9534F"
INK = "#1E293B"
SLATE = "#475569"
MIST = "#EFF6FF"
PALE_TEAL = "#EAF8F7"
GRID = "#D8E1EA"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Data preparation and inference
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    """Return a SHA-256 digest without loading the full file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function using only stdlib math."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_sided_normal_pvalue(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate the experiment data and return one valid, aligned record per user.

    A valid treatment exposure is treatment/new_page; a valid control exposure is
    control/old_page.  Assignment/page mismatches are excluded because they do not
    represent the intended experimental condition.  If a user remains more than
    once after that rule, the earliest timestamp is retained to keep one independent
    observation per experimental unit.
    """
    required_columns = {"user_id", "timestamp", "group", "landing_page", "converted"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"ab_data.csv is missing required columns: {sorted(missing_columns)}")

    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["converted"] = pd.to_numeric(data["converted"], errors="coerce")

    expected_groups = {"control", "treatment"}
    expected_pages = {"old_page", "new_page"}
    valid_structure = (
        data["user_id"].notna()
        & data["timestamp"].notna()
        & data["group"].isin(expected_groups)
        & data["landing_page"].isin(expected_pages)
        & data["converted"].isin([0, 1])
    )
    aligned_assignment = (
        ((data["group"] == "control") & (data["landing_page"] == "old_page"))
        | ((data["group"] == "treatment") & (data["landing_page"] == "new_page"))
    )

    structurally_valid = data.loc[valid_structure].copy()
    aligned = structurally_valid.loc[aligned_assignment.loc[structurally_valid.index]].copy()
    duplicate_rows = aligned.duplicated(subset="user_id", keep=False)
    duplicate_users = int(aligned.loc[duplicate_rows, "user_id"].nunique())
    cleaned = (
        aligned.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="user_id", keep="first")
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    cleaned["date"] = cleaned["timestamp"].dt.normalize()

    audit = {
        "raw_rows": int(len(data)),
        "missing_values": int(data[list(required_columns)].isna().sum().sum()),
        "invalid_structure_rows": int((~valid_structure).sum()),
        "assignment_mismatches": int((valid_structure & ~aligned_assignment).sum()),
        "rows_after_alignment": int(len(aligned)),
        "duplicate_users": duplicate_users,
        "duplicate_records": int(duplicate_rows.sum()),
        "duplicates_removed": int(len(aligned) - len(cleaned)),
        "analysis_rows": int(len(cleaned)),
        "analysis_users": int(cleaned["user_id"].nunique()),
    }
    return cleaned, audit


def group_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("group", observed=True)
        .agg(users=("user_id", "nunique"), conversions=("converted", "sum"))
        .reindex(["control", "treatment"])
    )
    summary["conversion_rate"] = summary["conversions"] / summary["users"]
    return summary


def sample_size_per_group_for_effect(baseline: float, absolute_effect: float) -> float:
    """Approximate sample size per equal-sized arm for 80% power, two-sided alpha.

    Uses the standard normal approximation for two independent proportions.
    """
    if absolute_effect <= 0:
        return float("inf")
    comparison = min(max(baseline + absolute_effect, 1e-9), 1 - 1e-9)
    pooled = (baseline + comparison) / 2
    z_alpha = NormalDist().inv_cdf(1 - ALPHA / 2)
    z_power = NormalDist().inv_cdf(0.80)
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(baseline * (1 - baseline) + comparison * (1 - comparison))
    ) ** 2
    return numerator / (absolute_effect**2)


def minimum_detectable_effect(baseline: float, users_per_group: int) -> float:
    """Solve for the absolute effect with an 80% powered equal-arm design."""
    lower, upper = 1e-7, min(0.25, 1 - baseline - 1e-7)
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if sample_size_per_group_for_effect(baseline, midpoint) > users_per_group:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def calculate_inference(summary: pd.DataFrame) -> dict[str, float | np.ndarray]:
    control_n = int(summary.loc["control", "users"])
    treatment_n = int(summary.loc["treatment", "users"])
    control_x = int(summary.loc["control", "conversions"])
    treatment_x = int(summary.loc["treatment", "conversions"])
    control_rate = control_x / control_n
    treatment_rate = treatment_x / treatment_n
    effect = treatment_rate - control_rate

    pooled_rate = (control_x + treatment_x) / (control_n + treatment_n)
    pooled_se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / control_n + 1 / treatment_n))
    z_statistic = effect / pooled_se
    p_value = two_sided_normal_pvalue(z_statistic)

    unpooled_se = math.sqrt(
        control_rate * (1 - control_rate) / control_n
        + treatment_rate * (1 - treatment_rate) / treatment_n
    )
    critical_value = NormalDist().inv_cdf(1 - ALPHA / 2)
    ci_low = effect - critical_value * unpooled_se
    ci_high = effect + critical_value * unpooled_se

    # A binary non-parametric bootstrap can be sampled from Binomial(n, p-hat).
    # This is equivalent to resampling the observed zero/one outcomes with replacement.
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_effects = (
        rng.binomial(treatment_n, treatment_rate, size=BOOTSTRAP_REPS) / treatment_n
        - rng.binomial(control_n, control_rate, size=BOOTSTRAP_REPS) / control_n
    )
    bootstrap_low, bootstrap_high = np.quantile(bootstrap_effects, [ALPHA / 2, 1 - ALPHA / 2])

    total = control_n + treatment_n
    expected = total / 2
    srm_chi_square = ((control_n - expected) ** 2 + (treatment_n - expected) ** 2) / expected
    srm_p_value = math.erfc(math.sqrt(srm_chi_square / 2))

    cohen_h = 2 * math.asin(math.sqrt(treatment_rate)) - 2 * math.asin(math.sqrt(control_rate))
    conservative_n = min(control_n, treatment_n)
    mde = minimum_detectable_effect(control_rate, conservative_n)
    needed_n = sample_size_per_group_for_effect(control_rate, abs(effect))

    return {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_x": control_x,
        "treatment_x": treatment_x,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "effect": effect,
        "relative_effect": effect / control_rate,
        "pooled_rate": pooled_rate,
        "pooled_se": pooled_se,
        "unpooled_se": unpooled_se,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_effects": bootstrap_effects,
        "bootstrap_low": float(bootstrap_low),
        "bootstrap_high": float(bootstrap_high),
        "srm_chi_square": srm_chi_square,
        "srm_p_value": srm_p_value,
        "cohen_h": cohen_h,
        "mde": mde,
        "needed_n": needed_n,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", length=0)


def draw_conversion_chart(axis: plt.Axes, summary: pd.DataFrame, stats: dict[str, float | np.ndarray]) -> None:
    rates = [float(summary.loc[group, "conversion_rate"]) for group in ["control", "treatment"]]
    ns = [int(summary.loc[group, "users"]) for group in ["control", "treatment"]]
    z_critical = NormalDist().inv_cdf(1 - ALPHA / 2)
    errors = [z_critical * math.sqrt(rate * (1 - rate) / n) for rate, n in zip(rates, ns)]
    positions = np.arange(2)
    bars = axis.bar(positions, np.array(rates) * 100, yerr=np.array(errors) * 100, capsize=6,
                    color=[NAVY, TEAL], width=0.56, edgecolor="none", error_kw={"ecolor": INK, "lw": 1.4})
    axis.set_xticks(positions, ["Control\n(old page)", "Treatment\n(new page)"])
    axis.set_ylabel("Conversion rate (%)")
    axis.set_title("Conversion rate with 95% confidence intervals", loc="left", fontsize=12, pad=12)
    upper = max(np.array(rates) * 100 + np.array(errors) * 100) + 0.8
    lower = max(0, min(np.array(rates) * 100 - np.array(errors) * 100) - 0.8)
    axis.set_ylim(lower, upper)
    for bar, rate in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.17, f"{rate * 100:.3f}%",
                  ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    style_axis(axis)


def draw_cumulative_chart(axis: plt.Axes, data: pd.DataFrame) -> None:
    daily = (
        data.groupby(["date", "group"], observed=True)["converted"]
        .agg([("conversions", "sum"), ("users", "size")])
        .reset_index()
        .sort_values("date")
    )
    for group, color, label in [
        ("control", NAVY, "Control (old page)"),
        ("treatment", TEAL, "Treatment (new page)"),
    ]:
        group_daily = daily.loc[daily["group"] == group].copy()
        group_daily["cumulative_rate"] = group_daily["conversions"].cumsum() / group_daily["users"].cumsum()
        axis.plot(group_daily["date"], group_daily["cumulative_rate"] * 100, color=color, marker="o",
                  markersize=3.2, linewidth=2, label=label)
    axis.set_ylabel("Cumulative conversion (%)")
    axis.set_title("Cumulative conversion rate through the test window", loc="left", fontsize=12, pad=12)
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.legend(frameon=False, loc="best", fontsize=8.5)
    style_axis(axis)


def draw_bootstrap_chart(axis: plt.Axes, stats: dict[str, float | np.ndarray]) -> None:
    effects = np.asarray(stats["bootstrap_effects"]) * 100
    axis.hist(effects, bins=60, density=True, color=BLUE, alpha=0.83, edgecolor="white", linewidth=0.35)
    axis.axvline(0, color=INK, linewidth=1.35, linestyle="--", label="No difference")
    axis.axvline(float(stats["effect"]) * 100, color=RED, linewidth=2, label="Observed difference")
    axis.axvspan(float(stats["bootstrap_low"]) * 100, float(stats["bootstrap_high"]) * 100,
                 color=GOLD, alpha=0.20, label="Bootstrap 95% interval")
    axis.set_xlabel("Treatment minus control (percentage points)")
    axis.set_ylabel("Density")
    axis.set_title("Bootstrap sampling distribution of the conversion difference", loc="left", fontsize=12, pad=12)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(axis)


def save_figures(data: pd.DataFrame, summary: pd.DataFrame, stats: dict[str, float | np.ndarray]) -> None:
    """Write standalone chart files used alongside the report."""
    FIGURE_DIR.mkdir(exist_ok=True)
    charts: list[tuple[str, callable]] = [
        ("01_conversion_rate_95ci.png", lambda ax: draw_conversion_chart(ax, summary, stats)),
        ("02_cumulative_conversion_trend.png", lambda ax: draw_cumulative_chart(ax, data)),
        ("03_bootstrap_effect_distribution.png", lambda ax: draw_bootstrap_chart(ax, stats)),
    ]
    for filename, drawer in charts:
        fig, axis = plt.subplots(figsize=(9, 5.3))
        drawer(axis)
        fig.tight_layout(pad=1.2)
        fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# PDF layout helpers
# ---------------------------------------------------------------------------
def new_page(page_title: str, page_number: int, subtitle: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.add_artist(Rectangle((0, 0.958), 1, 0.042, transform=fig.transFigure, color=NAVY, zorder=0))
    fig.text(0.07, 0.975, "A/B TEST REPORT", color="white", fontsize=8.5, fontweight="bold", va="center")
    fig.text(0.93, 0.975, f"ADVIK SINGH  |  {page_number:02d}", color="white", fontsize=7.4,
             ha="right", va="center")
    fig.text(0.07, 0.918, page_title, fontsize=19, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.07, 0.891, subtitle, fontsize=9.5, color=SLATE)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.074, 0.074], transform=fig.transFigure, color=GRID, linewidth=0.8))
    fig.text(0.07, 0.047, f"Prepared by {AUTHOR}  |  Reg. No. {REGISTRATION_NUMBER}  |  Branch: {BRANCH}",
             fontsize=7.3, color=SLATE)
    fig.text(0.93, 0.047, "Source: ab_data.csv", fontsize=7.3, color=SLATE, ha="right")
    canvas = fig.add_axes([0, 0, 1, 1], frameon=False)
    canvas.set_axis_off()
    return fig, canvas


def rounded_box(fig: plt.Figure, x: float, y: float, width: float, height: float,
                title: str, value: str, note: str = "", fill_color: str = MIST,
                value_color: str = NAVY) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=fill_color,
        edgecolor="none",
        zorder=1,
    )
    fig.add_artist(box)
    fig.text(x + 0.018, y + height - 0.036, title.upper(), fontsize=7.5, color=SLATE, fontweight="bold")
    fig.text(x + 0.018, y + height * 0.45, value, fontsize=17, color=value_color, fontweight="bold", va="center")
    if note:
        fig.text(x + 0.018, y + 0.022, note, fontsize=7.6, color=SLATE)


def add_paragraph(fig: plt.Figure, text: str, x: float, y: float, width: int = 90,
                  fontsize: float = 9.5, color: str = INK, line_spacing: float = 0.024,
                  weight: str = "normal") -> float:
    """Add wrapped text and return the y-coordinate underneath it."""
    lines = fill(text, width=width).splitlines()
    for index, line in enumerate(lines):
        fig.text(x, y - index * line_spacing, line, fontsize=fontsize, color=color, fontweight=weight, va="top")
    return y - len(lines) * line_spacing


def add_bullets(fig: plt.Figure, items: list[str], x: float, y: float, width: int = 83,
                fontsize: float = 9, item_gap: float = 0.012) -> float:
    current_y = y
    for item in items:
        wrapped = fill(item, width=width).splitlines()
        fig.text(x, current_y, "•", fontsize=fontsize + 1, color=TEAL, va="top", fontweight="bold")
        for line_number, line in enumerate(wrapped):
            fig.text(x + 0.019, current_y - line_number * 0.022, line, fontsize=fontsize, color=INK, va="top")
        current_y -= len(wrapped) * 0.022 + item_gap
    return current_y


def add_table(axis: plt.Axes, cell_text: list[list[str]], col_labels: list[str], col_widths: list[float] | None = None,
              font_size: float = 8.5, row_scale: float = 1.48) -> None:
    axis.axis("off")
    table = axis.table(cellText=cell_text, colLabels=col_labels, cellLoc="left", colLoc="left",
                       colWidths=col_widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_scale)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.PAD = 0.06
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "#EDF4FB")
            cell.get_text().set_color(INK)


def percent(value: float, decimals: int = 3) -> str:
    return f"{value * 100:.{decimals}f}%"


def pp(value: float, decimals: int = 3, signed: bool = False) -> str:
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f} pp"


# ---------------------------------------------------------------------------
# Report pages
# ---------------------------------------------------------------------------
def add_cover_page(pdf: PdfPages, data: pd.DataFrame, stats: dict[str, float | np.ndarray]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="#F8FBFF"))
    fig.add_artist(Rectangle((0, 0.79), 1, 0.21, transform=fig.transFigure, color=NAVY))
    fig.add_artist(Rectangle((0.07, 0.735), 0.16, 0.008, transform=fig.transFigure, color=GOLD))
    fig.text(0.07, 0.93, "DATA SCIENCE & MACHINE LEARNING", fontsize=9, color="#D9E9FF", fontweight="bold")
    fig.text(0.07, 0.866, "A/B Test Report", fontsize=29, color="white", fontweight="bold")
    fig.text(0.07, 0.823, "Landing-page conversion experiment", fontsize=14, color="#D9E9FF")
    fig.text(0.07, 0.758, "Requested pricing-study deliverable — analysed as conversion data", fontsize=9, color=SLATE)

    rounded_box(fig, 0.07, 0.603, 0.26, 0.105, "Control conversion", percent(float(stats["control_rate"])),
                "Old landing page", "#E7F0FA")
    rounded_box(fig, 0.37, 0.603, 0.26, 0.105, "Treatment conversion", percent(float(stats["treatment_rate"])),
                "New landing page", PALE_TEAL, TEAL)
    rounded_box(fig, 0.67, 0.603, 0.26, 0.105, "Estimated change", pp(float(stats["effect"]), signed=True),
                "Treatment minus control", "#FEF2F2", RED)

    fig.text(0.07, 0.538, "Executive decision", fontsize=15, color=NAVY, fontweight="bold")
    decision = (
        "Do not roll out the new landing page on the basis of conversion. The treatment estimate is "
        f"{pp(float(stats['effect']))} relative to control and the two-sided p-value is {float(stats['p_value']):.3f}. "
        "The result does not provide statistically significant evidence of a conversion improvement at alpha = 0.05."
    )
    add_paragraph(fig, decision, 0.07, 0.505, width=94, fontsize=10.5, line_spacing=0.028)

    fig.add_artist(FancyBboxPatch((0.07, 0.310), 0.86, 0.104, boxstyle="round,pad=0.012,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor="#FFF7E6", edgecolor="#F6D38C", linewidth=0.8))
    fig.text(0.09, 0.399, "Scope note", fontsize=9.4, color="#8A5A00", fontweight="bold")
    scope = (
        "The CSV contains user assignment, landing page, timestamp and conversion only. It has no price, revenue, "
        "margin or customer-segment fields. Accordingly, this report tests landing-page conversion only and makes no "
        "claim about pricing impact or profitability."
    )
    add_paragraph(fig, scope, 0.09, 0.375, width=103, fontsize=8.8, color=INK, line_spacing=0.020)

    fig.text(0.07, 0.282, "Prepared for", fontsize=8.5, color=SLATE, fontweight="bold")
    fig.text(0.07, 0.247, AUTHOR, fontsize=17, color=NAVY, fontweight="bold")
    fig.text(0.07, 0.215, f"Reg. No. {REGISTRATION_NUMBER}   |   Branch: {BRANCH}", fontsize=10.2, color=SLATE)
    fig.text(0.07, 0.151, f"Report date: {REPORT_DATE}", fontsize=8.5, color=SLATE)
    fig.text(0.07, 0.127, f"Observation window: {data['timestamp'].min():%d %b %Y} – {data['timestamp'].max():%d %b %Y}",
             fontsize=8.5, color=SLATE)
    fig.text(0.07, 0.103, f"Analysis population: {len(data):,} unique users after quality controls", fontsize=8.5, color=SLATE)
    fig.text(0.07, 0.043, "AB_Test_Pricing_Report_CORRECTED.pdf", fontsize=7.3, color=SLATE)
    fig.text(0.93, 0.043, "01", fontsize=7.3, color=SLATE, ha="right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_data_quality_page(pdf: PdfPages, data: pd.DataFrame, audit: dict[str, int],
                          summary: pd.DataFrame, stats: dict[str, float | np.ndarray]) -> None:
    fig, _ = new_page(
        "Experiment design & data quality",
        2,
        "Pre-analysis checks ensure that each retained user belongs to one intended experimental condition.",
    )
    fig.text(0.07, 0.842, "Design", fontsize=11.5, color=NAVY, fontweight="bold")
    design = (
        "Unit of randomisation: user ID. Control users were assigned the old page; treatment users were assigned "
        "the new page. The primary metric is the binary conversion indicator. The primary comparison is treatment "
        "conversion minus control conversion using a two-sided test at alpha = 0.05."
    )
    add_paragraph(fig, design, 0.07, 0.814, width=101, fontsize=9.2, line_spacing=0.023)

    fig.text(0.07, 0.711, "Cleaning audit", fontsize=11.5, color=NAVY, fontweight="bold")
    cleaning_rows = [
        ["Raw records read", f"{audit['raw_rows']:,}", "Rows in the supplied CSV"],
        ["Missing values", f"{audit['missing_values']:,}", "Across required fields"],
        ["Malformed records", f"{audit['invalid_structure_rows']:,}", "Excluded before assignment checks"],
        ["Assignment/page mismatches", f"{audit['assignment_mismatches']:,}", "Control/new or treatment/old"],
        ["Rows after alignment", f"{audit['rows_after_alignment']:,}", "Valid intended exposures"],
        ["Repeated-user records removed", f"{audit['duplicates_removed']:,}",
         f"{audit['duplicate_records']:,} rows across {audit['duplicate_users']:,} user(s); earliest retained"],
        ["Final analysis population", f"{audit['analysis_rows']:,}", "One aligned record per user"],
    ]
    table_axis = fig.add_axes([0.07, 0.47, 0.86, 0.215])
    add_table(table_axis, cleaning_rows, ["Check", "Count", "Treatment"], [0.36, 0.17, 0.47], font_size=7.7, row_scale=1.20)

    fig.text(0.07, 0.421, "Allocation check", fontsize=11.5, color=NAVY, fontweight="bold")
    allocation_rows = [
        ["Control (old page)", f"{int(summary.loc['control', 'users']):,}", percent(float(summary.loc['control', 'users']) / len(data), 2)],
        ["Treatment (new page)", f"{int(summary.loc['treatment', 'users']):,}", percent(float(summary.loc['treatment', 'users']) / len(data), 2)],
    ]
    allocation_axis = fig.add_axes([0.07, 0.278, 0.48, 0.105])
    add_table(allocation_axis, allocation_rows, ["Arm", "Users", "Share"], [0.55, 0.24, 0.21], font_size=8, row_scale=1.25)
    rounded_box(fig, 0.61, 0.278, 0.32, 0.105, "Sample-ratio check", f"p = {float(stats['srm_p_value']):.3f}",
                f"Chi-square = {float(stats['srm_chi_square']):.3f}; expected 50/50", "#EAF8F7", TEAL)

    fig.text(0.07, 0.222, "Quality conclusion", fontsize=11.5, color=NAVY, fontweight="bold")
    quality_conclusion = (
        "The retained sample is essentially evenly allocated, with no evidence of a sample-ratio mismatch. "
        "Removing assignment/page contradictions and repeated users is necessary to compare the intended new-page and "
        "old-page experiences. No customer covariates are available, so baseline balance beyond arm counts cannot be assessed."
    )
    add_paragraph(fig, quality_conclusion, 0.07, 0.194, width=103, fontsize=9.0, line_spacing=0.023)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_visual_evidence_page(pdf: PdfPages, data: pd.DataFrame, summary: pd.DataFrame,
                             stats: dict[str, float | np.ndarray]) -> None:
    fig, _ = new_page(
        "Visual evidence",
        3,
        "The treatment line remains below the control line over the observed period; visual separation is small.",
    )
    first_axis = fig.add_axes([0.10, 0.535, 0.80, 0.275])
    draw_conversion_chart(first_axis, summary, stats)
    fig.text(0.10, 0.496,
             "Error bars show 95% Wald confidence intervals for each arm's conversion rate. Overlapping arm intervals "
             "are descriptive; the formal difference test appears on the next page.",
             fontsize=8.2, color=SLATE)

    second_axis = fig.add_axes([0.10, 0.165, 0.80, 0.255])
    draw_cumulative_chart(second_axis, data)
    fig.text(0.10, 0.125,
             "Cumulative rates are descriptive monitoring only. The experiment's primary decision uses the full-window "
             "comparison, not repeated daily significance testing.",
             fontsize=8.2, color=SLATE)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_inference_page(pdf: PdfPages, summary: pd.DataFrame, stats: dict[str, float | np.ndarray]) -> None:
    fig, _ = new_page(
        "Primary inference: new page vs. old page",
        4,
        "Two independent proportions; treatment effect is always defined as new-page minus old-page conversion.",
    )
    fig.text(0.07, 0.843, "Observed outcomes", fontsize=11.5, color=NAVY, fontweight="bold")
    outcome_rows = [
        ["Control — old page", f"{int(stats['control_n']):,}", f"{int(stats['control_x']):,}", percent(float(stats["control_rate"]))],
        ["Treatment — new page", f"{int(stats['treatment_n']):,}", f"{int(stats['treatment_x']):,}", percent(float(stats["treatment_rate"]))],
    ]
    outcomes_axis = fig.add_axes([0.07, 0.673, 0.86, 0.135])
    add_table(outcomes_axis, outcome_rows, ["Arm", "Users", "Conversions", "Conversion rate"],
              [0.39, 0.18, 0.20, 0.23], font_size=8.5, row_scale=1.25)

    fig.text(0.07, 0.626, "Hypotheses and test", fontsize=11.5, color=NAVY, fontweight="bold")
    test_rows = [
        ["Null hypothesis", "H0: p(new page) − p(old page) = 0"],
        ["Alternative hypothesis", "H1: p(new page) − p(old page) ≠ 0"],
        ["Test and threshold", "Two-proportion z-test, two-sided alpha = 0.05"],
        ["Test statistic", f"z = {float(stats['z_statistic']):.3f}"],
        ["Two-sided p-value", f"{float(stats['p_value']):.4f}"],
        ["95% CI for difference", f"[{pp(float(stats['ci_low']))}, {pp(float(stats['ci_high']), signed=True)}]"],
        ["Bootstrap 95% interval", f"[{pp(float(stats['bootstrap_low']))}, {pp(float(stats['bootstrap_high']), signed=True)}]"],
        ["Relative conversion change", f"{float(stats['relative_effect']) * 100:+.2f}%"],
        ["Standardised effect (Cohen's h)", f"{float(stats['cohen_h']):.4f} (negligible practical magnitude)"],
    ]
    test_axis = fig.add_axes([0.07, 0.335, 0.49, 0.265])
    add_table(test_axis, test_rows, ["Measure", "Result"], [0.44, 0.56], font_size=7.6, row_scale=1.14)

    chart_axis = fig.add_axes([0.62, 0.348, 0.31, 0.235])
    draw_bootstrap_chart(chart_axis, stats)
    chart_axis.set_title("Bootstrap effect", loc="left", fontsize=9.5, pad=8)
    chart_axis.legend(fontsize=5.5, loc="upper left", frameon=False)
    chart_axis.set_ylabel("")
    chart_axis.tick_params(labelsize=6.5)

    fig.add_artist(FancyBboxPatch((0.07, 0.159), 0.86, 0.118, boxstyle="round,pad=0.012,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor="#FFF7E6", edgecolor="#F6D38C", linewidth=0.8))
    fig.text(0.09, 0.246, "Interpretation", fontsize=9.6, color="#8A5A00", fontweight="bold")
    interpretation = (
        f"Because p = {float(stats['p_value']):.4f} is greater than 0.05, the analysis does not reject the null hypothesis. "
        f"The 95% interval ranges from a {abs(float(stats['ci_low']) * 100):.3f} percentage-point treatment loss to a "
        f"{float(stats['ci_high']) * 100:.3f} percentage-point gain. It is therefore not evidence that the new page improves conversion; "
        "it is also not proof that the two pages are exactly equivalent."
    )
    add_paragraph(fig, interpretation, 0.09, 0.221, width=103, fontsize=8.8, line_spacing=0.020)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_decision_page(pdf: PdfPages, stats: dict[str, float | np.ndarray]) -> None:
    fig, _ = new_page(
        "Power, practical significance & recommendation",
        5,
        "Statistical non-significance should be interpreted alongside the size of the effect the experiment could reliably detect.",
    )
    observed_abs = abs(float(stats["effect"]))
    mde = float(stats["mde"])
    fig.text(0.07, 0.842, "Sensitivity of the experiment", fontsize=11.5, color=NAVY, fontweight="bold")
    rounded_box(fig, 0.07, 0.687, 0.26, 0.111, "Observed absolute change", pp(observed_abs),
                "Magnitude of the estimated difference", "#FEF2F2", RED)
    rounded_box(fig, 0.37, 0.687, 0.26, 0.111, "80% power MDE", pp(mde),
                "Two-sided alpha = 0.05", "#E7F0FA", NAVY)
    rounded_box(fig, 0.67, 0.687, 0.26, 0.111, "Needed per arm", f"{math.ceil(float(stats['needed_n'])):,}",
                "To detect observed magnitude at 80% power", "#EAF8F7", TEAL)

    sensitivity_text = (
        f"With about {min(int(stats['control_n']), int(stats['treatment_n'])):,} users per arm, this study has 80% power "
        f"to detect an absolute conversion change of roughly {pp(mde)} or larger. The observed absolute difference is "
        f"{pp(observed_abs)}, about {observed_abs / mde * 100:.0f}% of that threshold. A future experiment targeting an "
        f"effect this small would need approximately {math.ceil(float(stats['needed_n'])):,} users in each arm under the "
        "same baseline-rate assumptions."
    )
    add_paragraph(fig, sensitivity_text, 0.07, 0.645, width=103, fontsize=9.0, line_spacing=0.023)

    fig.text(0.07, 0.506, "Recommendation", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.add_artist(FancyBboxPatch((0.07, 0.342), 0.86, 0.132, boxstyle="round,pad=0.014,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor="#EFF6FF", edgecolor="#BBD8F7", linewidth=0.9))
    fig.text(0.09, 0.439, "Do not adopt the new page as a conversion improvement.", fontsize=11.3, color=NAVY, fontweight="bold")
    recommendation_text = (
        "The treatment point estimate is lower and the confidence interval includes zero. Retain the current page for the "
        "conversion objective, or continue a pre-registered experiment only if the business considers a small conversion "
        "difference decision-relevant. A conversion-neutral rollout cannot be established from a non-significant result alone."
    )
    add_paragraph(fig, recommendation_text, 0.09, 0.410, width=102, fontsize=8.7, line_spacing=0.020)

    fig.text(0.07, 0.285, "Actions before a follow-up test", fontsize=11.5, color=NAVY, fontweight="bold")
    add_bullets(fig, [
        "Pre-specify a minimum practical effect (MDE) jointly with product stakeholders, then size the experiment for that target rather than for the observed result.",
        "Capture revenue, price, margin, refunds and guardrail metrics (for example, latency or support contacts) if the decision is genuinely about pricing or profitability.",
        "Keep one primary conversion metric, a fixed stopping rule and a documented duration; treat any subgroup or daily views as exploratory unless they are pre-registered and multiplicity-controlled.",
    ], 0.07, 0.256, width=98, fontsize=8.8, item_gap=0.016)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_methodology_page(pdf: PdfPages, data_hash: str) -> None:
    fig, _ = new_page(
        "Methodology, provenance & reproducibility",
        6,
        "All calculations are reproducible from the source CSV and the included Python script.",
    )
    fig.text(0.07, 0.842, "Methods", fontsize=11.5, color=NAVY, fontweight="bold")
    methods = [
        "Cleaning: require the five expected fields; retain only control/old-page and treatment/new-page assignments; retain the earliest valid row for any repeated user ID.",
        "Primary estimate: treatment conversion rate minus control conversion rate. Significance test: pooled two-proportion z-test, two-sided alpha = 0.05.",
        "Uncertainty: 95% unpooled Wald confidence interval plus a 100,000-replicate binary non-parametric bootstrap (seed 42).",
        "Power: normal-approximation, equal-sized arms, 80% power and two-sided alpha = 0.05. It is planning context, not a post-hoc claim of achieved power.",
    ]
    add_bullets(fig, methods, 0.07, 0.812, width=99, fontsize=8.7, item_gap=0.013)

    fig.text(0.07, 0.548, "Data provenance", fontsize=11.5, color=NAVY, fontweight="bold")
    provenance_rows = [
        ["Dataset", "ab_data.csv"],
        ["Original repository", "aamir-gk2539-byte/DSML-Training"],
        ["Source file", "github.com/aamir-gk2539-byte/DSML-Training/blob/main/ab_data.csv"],
        ["Source commit", SOURCE_COMMIT[:12]],
        ["SHA-256", data_hash],
        ["Retrieved / report date", REPORT_DATE],
    ]
    provenance_axis = fig.add_axes([0.07, 0.347, 0.86, 0.17])
    add_table(provenance_axis, provenance_rows, ["Item", "Value"], [0.29, 0.71], font_size=7.7, row_scale=1.10)

    fig.text(0.07, 0.302, "Reproduce", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.add_artist(FancyBboxPatch((0.07, 0.213), 0.86, 0.06, boxstyle="round,pad=0.012,rounding_size=0.008",
                                  transform=fig.transFigure, facecolor="#0F172A", edgecolor="none"))
    fig.text(0.09, 0.243, "pip install -r requirements.txt     # then:     python AB_Test_Analysis.py",
             fontfamily="DejaVu Sans Mono", fontsize=8.2, color="white", va="center")

    fig.text(0.07, 0.162, "Limitations", fontsize=11.5, color=NAVY, fontweight="bold")
    limitation_text = (
        "This observational extract identifies assignment and conversion but does not document randomisation implementation, "
        "exposure quality, business costs, price, revenue, user characteristics or longer-term outcomes. Results generalise "
        "only to the observed test window and the recorded conversion event."
    )
    add_paragraph(fig, limitation_text, 0.07, 0.136, width=103, fontsize=8.7, line_spacing=0.021)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_report(data: pd.DataFrame, audit: dict[str, int], summary: pd.DataFrame,
                 stats: dict[str, float | np.ndarray], data_hash: str) -> None:
    with PdfPages(REPORT_PATH) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "A/B Test Report: Landing-page Conversion"
        metadata["Author"] = AUTHOR
        metadata["Subject"] = f"A/B conversion test | Reg. No. {REGISTRATION_NUMBER} | {BRANCH}"
        metadata["Keywords"] = "A/B test, conversion, landing page, experiment, DS-A"
        metadata["CreationDate"] = pd.Timestamp("2026-09-28").to_pydatetime()
        add_cover_page(pdf, data, stats)
        add_data_quality_page(pdf, data, audit, summary, stats)
        add_visual_evidence_page(pdf, data, summary, stats)
        add_inference_page(pdf, summary, stats)
        add_decision_page(pdf, stats)
        add_methodology_page(pdf, data_hash)


def write_summary_markdown(audit: dict[str, int], stats: dict[str, float | np.ndarray], data_hash: str) -> None:
    """Write a concise, reviewable text companion to the PDF."""
    summary_path = ROOT / "AB_Test_Analysis_Summary.md"
    summary_path.write_text(
        f"""# A/B Test Analysis Summary

**Student:** {AUTHOR}<br>
**Registration No.:** {REGISTRATION_NUMBER}<br>
**Branch:** {BRANCH}

## Decision

Do **not** roll out the new landing page as a conversion improvement. Its estimated
conversion effect is **{pp(float(stats['effect']))}** (treatment minus control), with
a two-sided two-proportion z-test p-value of **{float(stats['p_value']):.4f}**. At
alpha = 0.05, this is not statistically significant.

## Primary result

| Arm | Users | Conversions | Conversion rate |
| --- | ---: | ---: | ---: |
| Control (old page) | {int(stats['control_n']):,} | {int(stats['control_x']):,} | {percent(float(stats['control_rate']))} |
| Treatment (new page) | {int(stats['treatment_n']):,} | {int(stats['treatment_x']):,} | {percent(float(stats['treatment_rate']))} |

- **Effect (new − old):** {pp(float(stats['effect']), signed=True)} ({float(stats['relative_effect']) * 100:+.2f}% relative)
- **95% CI:** [{pp(float(stats['ci_low']))}, {pp(float(stats['ci_high']), signed=True)}]
- **z statistic / p-value:** {float(stats['z_statistic']):.3f} / {float(stats['p_value']):.4f}
- **Bootstrap 95% interval:** [{pp(float(stats['bootstrap_low']))}, {pp(float(stats['bootstrap_high']), signed=True)}]
- **Sample-ratio-mismatch p-value:** {float(stats['srm_p_value']):.4f}

## Data quality

- Raw records: {audit['raw_rows']:,}
- Assignment/page mismatches excluded: {audit['assignment_mismatches']:,}
- Repeated-user records removed: {audit['duplicates_removed']:,}
- Final population: {audit['analysis_rows']:,} unique users
- CSV SHA-256: `{data_hash}`

## Important scope limitation

The supplied CSV has no price, revenue, margin or cost field. The report title follows
the requested filename, but the valid inference is about **landing-page conversion only**,
not pricing or profitability.

See `AB_Test_Pricing_Report_CORRECTED.pdf` for the complete six-page report and
`AB_Test_Analysis.py` to reproduce it.
""",
        encoding="utf-8",
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected data file not found: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH)
    data_hash = sha256(DATA_PATH)
    data, audit = clean_data(raw)
    summary = group_summary(data)
    stats = calculate_inference(summary)
    save_figures(data, summary, stats)
    write_report(data, audit, summary, stats, data_hash)
    write_summary_markdown(audit, stats, data_hash)

    print("A/B test analysis complete")
    print(f"  CSV: {DATA_PATH.name} ({audit['raw_rows']:,} raw records)")
    print(f"  Analysis population: {audit['analysis_rows']:,} unique users")
    print(f"  Control conversion:   {percent(float(stats['control_rate']))}")
    print(f"  Treatment conversion: {percent(float(stats['treatment_rate']))}")
    print(f"  Difference (new-old): {pp(float(stats['effect']), signed=True)}")
    print(f"  Two-sided p-value:    {float(stats['p_value']):.4f}")
    print(f"  95% CI: [{pp(float(stats['ci_low']))}, {pp(float(stats['ci_high']), signed=True)}]")
    print(f"  Wrote: {REPORT_PATH.name}, {FIGURE_DIR.name}/, AB_Test_Analysis_Summary.md")


if __name__ == "__main__":
    main()
