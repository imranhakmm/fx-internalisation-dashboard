"""Synthetic G3 FX ticks and client flow generation for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

FX_SECONDS_PER_YEAR = 252 * 24 * 60 * 60
SESSION_HOURS = 10
TICK_MS = 100
TICK_SECONDS = TICK_MS / 1000
TOXICITY_WINDOW_SECONDS = 300
TOTAL_TRADES = 50_000


@dataclass(frozen=True)
class PairSpec:
    pair: str
    start_price: float
    annualized_vol: float
    half_spread_bp: float
    trade_weight: float


@dataclass(frozen=True)
class TagProfile:
    tag: str
    frequency_weight: float
    notional_range_usd: Tuple[float, float]
    buy_probability: float
    widen_bp: float
    toxicity_anchor_bps: Tuple[float, float, float, float, float, float]
    notes: str


@dataclass(frozen=True)
class ToxicityKernel:
    cumulative_impact_bps: np.ndarray
    return_kernel_log: np.ndarray


PAIR_SPECS: Dict[str, PairSpec] = {
    "EURUSD": PairSpec("EURUSD", start_price=1.0850, annualized_vol=0.06, half_spread_bp=0.20, trade_weight=0.38),
    "USDJPY": PairSpec("USDJPY", start_price=152.50, annualized_vol=0.08, half_spread_bp=0.30, trade_weight=0.32),
    "GBPUSD": PairSpec("GBPUSD", start_price=1.2700, annualized_vol=0.09, half_spread_bp=0.40, trade_weight=0.30),
}

TAG_WEIGHTS: Dict[str, float] = {
    "HFT_A": 0.20,
    "HFT_B": 0.15,
    "PB_C": 0.20,
    "RetailAgg_D": 0.25,
    "Corp_E": 0.05,
    "Bank_F": 0.15,
}


def build_session_timestamps(
    session_start: datetime,
    session_hours: int = SESSION_HOURS,
    tick_ms: int = TICK_MS,
) -> np.ndarray:
    """Return evenly spaced timestamps for a 10-hour London session."""
    n_ticks = int(session_hours * 60 * 60 * 1000 / tick_ms)
    offsets = np.arange(n_ticks, dtype=np.int64) * np.timedelta64(tick_ms, "ms")
    return np.datetime64(session_start, "ms") + offsets


def intraday_volatility_multiplier(
    n_ticks: int,
    tick_seconds: float = TICK_SECONDS,
) -> np.ndarray:
    """Create a smooth intraday volatility smile around London and NY opens."""
    session_hours = np.arange(n_ticks, dtype=np.float64) * tick_seconds / 3600.0
    london_open = np.exp(-0.5 * ((session_hours - 1.0) / 0.45) ** 2)
    ny_open = np.exp(-0.5 * ((session_hours - 6.5) / 0.55) ** 2)
    lunch_lull = np.exp(-0.5 * ((session_hours - 4.5) / 1.10) ** 2)
    multiplier = 1.0 + 0.55 * london_open + 0.50 * ny_open - 0.30 * lunch_lull
    return np.clip(multiplier, 0.70, 1.60)


def build_tag_profiles(rng: np.random.Generator) -> Dict[str, TagProfile]:
    """Return the six synthetic client tags with size, flow, and toxicity traits."""
    corp_buy_probability = 0.90 if rng.random() < 0.5 else 0.10
    return {
        "HFT_A": TagProfile(
            tag="HFT_A",
            frequency_weight=TAG_WEIGHTS["HFT_A"],
            notional_range_usd=(50_000.0, 500_000.0),
            buy_probability=0.50,
            widen_bp=0.12,
            toxicity_anchor_bps=(0.0, 0.80, 1.05, 1.25, 1.35, 1.50),
            notes="High-frequency flow with strong short-horizon alpha.",
        ),
        "HFT_B": TagProfile(
            tag="HFT_B",
            frequency_weight=TAG_WEIGHTS["HFT_B"],
            notional_range_usd=(50_000.0, 500_000.0),
            buy_probability=0.50,
            widen_bp=0.08,
            toxicity_anchor_bps=(0.0, 0.55, 0.75, 0.90, 1.00, 1.15),
            notes="High-frequency flow, less toxic than HFT_A.",
        ),
        "PB_C": TagProfile(
            tag="PB_C",
            frequency_weight=TAG_WEIGHTS["PB_C"],
            notional_range_usd=(100_000.0, 2_000_000.0),
            buy_probability=0.52,
            widen_bp=0.04,
            toxicity_anchor_bps=(0.0, 0.18, 0.32, 0.46, 0.56, 0.70),
            notes="Prime broker mix with moderate toxicity.",
        ),
        "RetailAgg_D": TagProfile(
            tag="RetailAgg_D",
            frequency_weight=TAG_WEIGHTS["RetailAgg_D"],
            notional_range_usd=(10_000.0, 200_000.0),
            buy_probability=0.50,
            widen_bp=0.00,
            toxicity_anchor_bps=(0.0, 0.28, 0.32, 0.38, 0.40, 0.43),
            notes="Retail aggregator flow that is nearly random.",
        ),
        "Corp_E": TagProfile(
            tag="Corp_E",
            frequency_weight=TAG_WEIGHTS["Corp_E"],
            notional_range_usd=(1_000_000.0, 20_000_000.0),
            buy_probability=corp_buy_probability,
            widen_bp=0.03,
            toxicity_anchor_bps=(0.0, 0.04, 0.09, 0.18, 0.28, 0.42),
            notes="Corporate flow with a one-way side bias for the day.",
        ),
        "Bank_F": TagProfile(
            tag="Bank_F",
            frequency_weight=TAG_WEIGHTS["Bank_F"],
            notional_range_usd=(500_000.0, 5_000_000.0),
            buy_probability=0.50,
            widen_bp=0.06,
            toxicity_anchor_bps=(0.0, 0.05, 0.14, 0.42, 0.60, 1.10),
            notes="Bank counterparty with longer-horizon information.",
        ),
    }


def build_toxicity_kernels(
    tag_profiles: Mapping[str, TagProfile],
    tick_seconds: float = TICK_SECONDS,
    window_seconds: int = TOXICITY_WINDOW_SECONDS,
) -> Dict[str, ToxicityKernel]:
    """Convert anchor-point impact curves into per-tick return kernels."""
    anchor_seconds = np.array([0.0, 1.0, 5.0, 30.0, 60.0, 300.0], dtype=np.float64)
    grid_seconds = np.arange(int(window_seconds / tick_seconds) + 1, dtype=np.float64) * tick_seconds
    kernels: Dict[str, ToxicityKernel] = {}
    for tag, profile in tag_profiles.items():
        cumulative_impact_bps = np.interp(grid_seconds, anchor_seconds, np.array(profile.toxicity_anchor_bps))
        return_kernel_log = np.diff(cumulative_impact_bps) * 1e-4
        kernels[tag] = ToxicityKernel(
            cumulative_impact_bps=cumulative_impact_bps,
            return_kernel_log=return_kernel_log,
        )
    return kernels


def generate_pair_baseline_ticks(
    pair_spec: PairSpec,
    timestamps: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate baseline log returns and mid prices before toxicity is injected."""
    n_ticks = len(timestamps)
    vol_multiplier = intraday_volatility_multiplier(n_ticks)
    sigma_per_tick = pair_spec.annualized_vol * vol_multiplier * np.sqrt(TICK_SECONDS / FX_SECONDS_PER_YEAR)
    log_returns = rng.normal(loc=0.0, scale=sigma_per_tick[:-1], size=n_ticks - 1)
    log_mid = np.empty(n_ticks, dtype=np.float64)
    log_mid[0] = np.log(pair_spec.start_price)
    log_mid[1:] = log_mid[0] + np.cumsum(log_returns)
    return log_returns, np.exp(log_mid)


def sample_notional_usd(
    low: float,
    high: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample notionals from a log-uniform distribution."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def trade_intensity_profile(n_ticks: int) -> np.ndarray:
    """Higher trade frequency around liquid sessions, lower at lunch."""
    vol_profile = intraday_volatility_multiplier(n_ticks)
    intensity = 0.80 + 0.40 * vol_profile
    return intensity / intensity.sum()


def generate_trade_schedule(
    timestamps: np.ndarray,
    pair_specs: Mapping[str, PairSpec],
    tag_profiles: Mapping[str, TagProfile],
    rng: np.random.Generator,
    total_trades: int = TOTAL_TRADES,
) -> pl.DataFrame:
    """Create the synthetic client flow schedule before trade prices are attached."""
    n_ticks = len(timestamps)
    pair_names = list(pair_specs)
    pair_weights = np.array([pair_specs[pair].trade_weight for pair in pair_names], dtype=np.float64)
    pair_weights = pair_weights / pair_weights.sum()
    pair_counts = rng.multinomial(total_trades, pair_weights)
    tag_names = list(tag_profiles)
    tag_weights = np.array([tag_profiles[tag].frequency_weight for tag in tag_names], dtype=np.float64)
    tag_weights = tag_weights / tag_weights.sum()
    tick_probabilities = trade_intensity_profile(n_ticks)

    frames = []
    for pair_name, pair_count in zip(pair_names, pair_counts):
        tick_indices = rng.choice(n_ticks - 1, size=pair_count, replace=True, p=tick_probabilities[:-1] / tick_probabilities[:-1].sum())
        tags = rng.choice(tag_names, size=pair_count, replace=True, p=tag_weights)
        notionals = np.empty(pair_count, dtype=np.float64)
        widen_bps = np.empty(pair_count, dtype=np.float64)
        buy_probabilities = np.empty(pair_count, dtype=np.float64)

        for tag_name, profile in tag_profiles.items():
            mask = tags == tag_name
            tag_size = int(mask.sum())
            if tag_size == 0:
                continue
            notionals[mask] = sample_notional_usd(
                low=profile.notional_range_usd[0],
                high=profile.notional_range_usd[1],
                size=tag_size,
                rng=rng,
            )
            widen_bps[mask] = profile.widen_bp
            buy_probabilities[mask] = profile.buy_probability

        buy_mask = rng.random(pair_count) < buy_probabilities
        sides = np.where(buy_mask, "BUY", "SELL")
        side_sign = np.where(buy_mask, 1, -1)
        # Keep size as a secondary amplifier, but avoid collapsing small-ticket toxicity toward zero.
        size_multiplier = np.clip(np.sqrt(notionals / 1_000_000.0), 0.75, 3.00)

        order = np.argsort(tick_indices, kind="stable")
        frames.append(
            pl.DataFrame(
                {
                    "tick_index": tick_indices[order],
                    "ts": timestamps[tick_indices[order]],
                    "pair": np.full(pair_count, pair_name),
                    "tag": tags[order],
                    "side": sides[order],
                    "side_sign": side_sign[order],
                    "notional_usd": notionals[order],
                    "size_multiplier": size_multiplier[order],
                    "widen_bp": widen_bps[order],
                }
            )
        )

    return pl.concat(frames, how="vertical").sort(["ts", "pair"])


def _fft_convolve_trimmed(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve and keep only the portion aligned to the original signal length."""
    target_length = signal.size + kernel.size - 1
    fft_length = 1 << (target_length - 1).bit_length()
    signal_fft = np.fft.rfft(signal, fft_length)
    kernel_fft = np.fft.rfft(kernel, fft_length)
    convolved = np.fft.irfft(signal_fft * kernel_fft, fft_length)[:target_length]
    return convolved[: signal.size]


def apply_toxicity_impacts(
    baseline_returns_by_pair: Mapping[str, np.ndarray],
    trade_schedule: pl.DataFrame,
    kernels: Mapping[str, ToxicityKernel],
    pair_specs: Mapping[str, PairSpec],
) -> Dict[str, np.ndarray]:
    """Inject client-favourable post-trade drift into each pair mid path."""
    final_mid_by_pair: Dict[str, np.ndarray] = {}
    for pair_name, baseline_returns in baseline_returns_by_pair.items():
        pair_schedule = trade_schedule.filter(pl.col("pair") == pair_name)
        adjusted_returns = baseline_returns.copy()
        n_returns = adjusted_returns.size
        tick_indices = pair_schedule["tick_index"].to_numpy()
        side_sign = pair_schedule["side_sign"].to_numpy()
        size_multiplier = pair_schedule["size_multiplier"].to_numpy()
        tags = pair_schedule["tag"].to_numpy()

        for tag_name, kernel in kernels.items():
            tag_mask = tags == tag_name
            if not np.any(tag_mask):
                continue
            impulses = np.zeros(n_returns, dtype=np.float64)
            np.add.at(
                impulses,
                tick_indices[tag_mask],
                side_sign[tag_mask] * size_multiplier[tag_mask],
            )
            adjusted_returns += _fft_convolve_trimmed(impulses, kernel.return_kernel_log)

        log_mid = np.empty(n_returns + 1, dtype=np.float64)
        log_mid[0] = np.log(pair_specs[pair_name].start_price)
        log_mid[1:] = log_mid[0] + np.cumsum(adjusted_returns)
        final_mid_by_pair[pair_name] = np.exp(log_mid)

    return final_mid_by_pair


def build_ticks_frame(
    timestamps: np.ndarray,
    final_mid_by_pair: Mapping[str, np.ndarray],
) -> pl.DataFrame:
    """Assemble all pair mid paths into one Polars DataFrame."""
    frames = []
    tick_index = np.arange(len(timestamps), dtype=np.int32)
    for pair_name, mids in final_mid_by_pair.items():
        frames.append(
            pl.DataFrame(
                {
                    "ts": timestamps,
                    "pair": np.full(len(timestamps), pair_name),
                    "tick_index": tick_index,
                    "mid": mids,
                }
            )
        )
    return pl.concat(frames, how="vertical").sort(["pair", "ts"])


def attach_trade_prices(
    trade_schedule: pl.DataFrame,
    final_mid_by_pair: Mapping[str, np.ndarray],
    pair_specs: Mapping[str, PairSpec],
) -> pl.DataFrame:
    """Attach the prevailing mid and executed price for every synthetic trade."""
    frames = []
    for pair_name, pair_spec in pair_specs.items():
        pair_schedule = trade_schedule.filter(pl.col("pair") == pair_name)
        pair_indices = pair_schedule["tick_index"].to_numpy()
        side_sign = pair_schedule["side_sign"].to_numpy()
        widen_bp = pair_schedule["widen_bp"].to_numpy()
        mid_at_trade = final_mid_by_pair[pair_name][pair_indices]
        effective_half_spread_bp = pair_spec.half_spread_bp + widen_bp
        executed_price = mid_at_trade * (1.0 + side_sign * effective_half_spread_bp * 1e-4)
        frames.append(
            pair_schedule.with_columns(
                [
                    pl.Series("mid_at_trade", mid_at_trade),
                    pl.Series("executed_price", executed_price),
                ]
            )
        )

    return (
        pl.concat(frames, how="vertical")
        .sort(["ts", "pair"])
        .select(["ts", "pair", "tag", "side", "notional_usd", "mid_at_trade", "executed_price"])
    )


def generate_synthetic_fx_data(
    seed: int = 7,
    session_start: Optional[datetime] = None,
    total_trades: int = TOTAL_TRADES,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Generate the full synthetic tick and trade datasets."""
    rng = np.random.default_rng(seed)
    if session_start is None:
        session_start = datetime(2024, 1, 15, 7, 0, 0)

    timestamps = build_session_timestamps(session_start=session_start)
    tag_profiles = build_tag_profiles(rng)
    toxicity_kernels = build_toxicity_kernels(tag_profiles)
    baseline_returns_by_pair: Dict[str, np.ndarray] = {}
    for pair_name, pair_spec in PAIR_SPECS.items():
        baseline_returns, _ = generate_pair_baseline_ticks(pair_spec=pair_spec, timestamps=timestamps, rng=rng)
        baseline_returns_by_pair[pair_name] = baseline_returns

    trade_schedule = generate_trade_schedule(
        timestamps=timestamps,
        pair_specs=PAIR_SPECS,
        tag_profiles=tag_profiles,
        rng=rng,
        total_trades=total_trades,
    )
    final_mid_by_pair = apply_toxicity_impacts(
        baseline_returns_by_pair=baseline_returns_by_pair,
        trade_schedule=trade_schedule,
        kernels=toxicity_kernels,
        pair_specs=PAIR_SPECS,
    )
    ticks = build_ticks_frame(timestamps=timestamps, final_mid_by_pair=final_mid_by_pair)
    trades = attach_trade_prices(
        trade_schedule=trade_schedule,
        final_mid_by_pair=final_mid_by_pair,
        pair_specs=PAIR_SPECS,
    )
    return ticks, trades


def summarise_trades_by_tag(trades: pl.DataFrame) -> pl.DataFrame:
    """Aggregate trade counts and total USD notional per client tag."""
    return (
        trades.group_by("tag")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.sum("notional_usd").alias("notional_usd"),
                pl.mean("notional_usd").alias("avg_notional_usd"),
            ]
        )
        .sort("notional_usd", descending=True)
    )


def write_dataset(
    ticks: pl.DataFrame,
    trades: pl.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write the synthetic dataset to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ticks_path = output_dir / "ticks.parquet"
    trades_path = output_dir / "trades.parquet"
    ticks.write_parquet(ticks_path)
    trades.write_parquet(trades_path)
    return ticks_path, trades_path


def generate_and_write_dataset(
    output_dir: Path,
    seed: int = 7,
    session_start: Optional[datetime] = None,
    total_trades: int = TOTAL_TRADES,
) -> Tuple[pl.DataFrame, pl.DataFrame, Path, Path]:
    """Convenience wrapper for generating and persisting the synthetic book."""
    ticks, trades = generate_synthetic_fx_data(
        seed=seed,
        session_start=session_start,
        total_trades=total_trades,
    )
    ticks_path, trades_path = write_dataset(ticks=ticks, trades=trades, output_dir=output_dir)
    return ticks, trades, ticks_path, trades_path


def _format_summary_table(summary: pl.DataFrame) -> str:
    return summary.with_columns(pl.col("notional_usd").round(2), pl.col("avg_notional_usd").round(2)).__repr__()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"
    ticks, trades, ticks_path, trades_path = generate_and_write_dataset(output_dir=output_dir)
    print(f"Generated {ticks.height:,} ticks -> {ticks_path}")
    print(f"Generated {trades.height:,} trades -> {trades_path}")
    print()
    print("Trade sample:")
    print(trades.head(10))
    print()
    print("Summary by tag:")
    print(_format_summary_table(summarise_trades_by_tag(trades)))


if __name__ == "__main__":
    main()
