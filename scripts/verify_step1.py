from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.markouts import DEFAULT_HORIZONS_SECONDS, compute_markouts

EXPECTED_SIGNATURES: dict[str, str] = {
    "HFT_A": "Strong short-horizon alpha; LP markouts should be negative quickly and stay the worst of the HF tags.",
    "HFT_B": "Negative short-horizon markouts, but milder than HFT_A.",
    "PB_C": "Moderate toxicity with markouts drifting modestly negative as horizon extends.",
    "RetailAgg_D": "Near-flat markouts across horizons; flow is close to random.",
    "Corp_E": "One-way corporate bias should skew long-horizon markouts in one direction.",
    "Bank_F": "Information shows up at medium horizons; 30s to 60s markouts should be negative.",
}


def _horizon_label(horizon_seconds: float) -> str:
    if float(horizon_seconds).is_integer():
        return str(int(horizon_seconds))
    return str(horizon_seconds).replace(".", "p")


def build_tag_horizon_mean_table(
    markouts: pl.DataFrame,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for horizon_seconds in horizons_seconds:
        label = _horizon_label(horizon_seconds)
        rows.append(
            markouts.group_by("tag")
            .agg(pl.col(f"markout_bp_{label}s").mean().alias("mean_markout_bp"))
            .with_columns(pl.lit(f"{label}s").alias("horizon"))
        )
    long_table = pl.concat(rows, how="vertical")
    return (
        long_table.pivot(on="horizon", index="tag", values="mean_markout_bp")
        .sort("tag")
        .select(["tag"] + [f"{_horizon_label(h)}s" for h in horizons_seconds])
        .with_columns(pl.exclude("tag").round(4))
    )


def build_tag_horizon_stats(
    markouts: pl.DataFrame,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for horizon_seconds in horizons_seconds:
        label = _horizon_label(horizon_seconds)
        column = f"markout_bp_{label}s"
        stats = (
            markouts.group_by("tag")
            .agg(
                [
                    pl.col(column).mean().alias("mean_markout_bp"),
                    pl.col(column).std(ddof=1).alias("std_markout_bp"),
                    pl.col(column).count().alias("observations"),
                    pl.col(column).median().alias("median_markout_bp"),
                    (100.0 * pl.col(column).lt(0).mean()).alias("pct_negative"),
                ]
            )
            .with_columns(
                [
                    pl.lit(horizon_seconds).alias("horizon_seconds"),
                    pl.lit(f"{label}s").alias("horizon"),
                ]
            )
        )
        frames.append(stats)
    return pl.concat(frames, how="vertical").sort(["tag", "horizon_seconds"])


def save_per_tag_plots(stats: pl.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    for tag in stats["tag"].unique().sort().to_list():
        tag_stats = stats.filter(pl.col("tag") == tag).sort("horizon_seconds")
        horizons = tag_stats["horizon_seconds"].to_numpy()
        means = tag_stats["mean_markout_bp"].to_numpy()
        std = np.nan_to_num(tag_stats["std_markout_bp"].to_numpy(), nan=0.0)
        observations = np.maximum(tag_stats["observations"].to_numpy(), 1)
        ci_half_width = 1.96 * std / np.sqrt(observations)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(horizons, means, color="#0b5cab", linewidth=2.2, marker="o")
        ax.fill_between(
            horizons,
            means - ci_half_width,
            means + ci_half_width,
            color="#7fb3d5",
            alpha=0.30,
            label="95% CI",
        )
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title(f"{tag} mean LP markout vs horizon")
        ax.set_xlabel("Horizon (seconds)")
        ax.set_ylabel("Mean markout (bp)")
        ax.legend(loc="best")
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}s"))
        fig.tight_layout()

        plot_path = output_dir / f"{tag.lower()}_markout_curve.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def evaluate_assertions(markouts: pl.DataFrame, mean_table: pl.DataFrame) -> list[tuple[str, bool, str]]:
    def mean_value(tag: str, label: str) -> float:
        row = mean_table.filter(pl.col("tag") == tag)
        return float(row.item(0, label))

    corp_metrics = (
        markouts.filter(pl.col("tag") == "Corp_E")
        .select(
            [
                pl.col("markout_bp_300s").mean().alias("mean_300s"),
                pl.col("markout_bp_300s").median().alias("median_300s"),
                (100.0 * pl.col("markout_bp_300s").lt(0).mean()).alias("pct_negative_300s"),
                (100.0 * pl.col("side").eq("BUY").mean()).alias("buy_share_pct"),
            ]
        )
        .row(0, named=True)
    )

    results = [
        (
            "HFT_A mean markout is negative at 1s and 5s",
            mean_value("HFT_A", "1s") < 0.0 and mean_value("HFT_A", "5s") < 0.0,
            f"measured: 1s={mean_value('HFT_A', '1s'):.4f}bp, 5s={mean_value('HFT_A', '5s'):.4f}bp",
        ),
        (
            "HFT_A 30s markout is more negative than HFT_B 30s",
            mean_value("HFT_A", "30s") < mean_value("HFT_B", "30s"),
            f"measured: HFT_A={mean_value('HFT_A', '30s'):.4f}bp vs HFT_B={mean_value('HFT_B', '30s'):.4f}bp",
        ),
        (
            "Bank_F mean markout is negative at 30s and 60s",
            mean_value("Bank_F", "30s") < 0.0 and mean_value("Bank_F", "60s") < 0.0,
            f"measured: 30s={mean_value('Bank_F', '30s'):.4f}bp, 60s={mean_value('Bank_F', '60s'):.4f}bp",
        ),
        (
            "RetailAgg_D mean markout stays within +/-0.1bp across horizons",
            all(abs(mean_value("RetailAgg_D", f"{_horizon_label(h)}s")) <= 0.1 for h in DEFAULT_HORIZONS_SECONDS),
            "measured: "
            + ", ".join(
                f"{_horizon_label(h)}s={mean_value('RetailAgg_D', f'{_horizon_label(h)}s'):.4f}bp"
                for h in DEFAULT_HORIZONS_SECONDS
            ),
        ),
        (
            "Corp_E shows one-way bias at 300s",
            abs(float(corp_metrics["mean_300s"])) >= 0.1
            and abs(float(corp_metrics["median_300s"])) >= 0.1
            and (
                float(corp_metrics["pct_negative_300s"]) >= 60.0
                or float(corp_metrics["pct_negative_300s"]) <= 40.0
            ),
            "measured: "
            f"mean_300s={float(corp_metrics['mean_300s']):.4f}bp, "
            f"median_300s={float(corp_metrics['median_300s']):.4f}bp, "
            f"pct_negative_300s={float(corp_metrics['pct_negative_300s']):.2f}%, "
            f"buy_share={float(corp_metrics['buy_share_pct']):.2f}%",
        ),
    ]
    return results


def print_signature_summary(mean_table: pl.DataFrame) -> None:
    print("Expected qualitative signatures vs measured means:")
    for row in mean_table.iter_rows(named=True):
        tag = str(row["tag"])
        measured = ", ".join(
            f"{column}={float(row[column]):.4f}bp"
            for column in mean_table.columns
            if column != "tag"
        )
        print(f"- {tag}: {EXPECTED_SIGNATURES[tag]}")
        print(f"  measured: {measured}")
    print()


def main() -> None:
    project_root = PROJECT_ROOT
    verification_dir = project_root / "data" / "verification"

    trades = pl.read_parquet(project_root / "data" / "trades.parquet")
    ticks = pl.read_parquet(project_root / "data" / "ticks.parquet")
    markouts = compute_markouts(trades=trades, ticks=ticks, horizons_seconds=DEFAULT_HORIZONS_SECONDS)

    mean_table = build_tag_horizon_mean_table(markouts)
    print("Mean markout table (bp):")
    print(mean_table)
    print()

    print_signature_summary(mean_table)

    stats = build_tag_horizon_stats(markouts)
    plot_paths = save_per_tag_plots(stats=stats, output_dir=verification_dir)

    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path}")
    print()

    results = evaluate_assertions(markouts=markouts, mean_table=mean_table)
    print("Pass/fail summary:")
    failed = False
    for description, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"- {status}: {description}")
        print(f"  {detail}")
        failed = failed or not passed

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
