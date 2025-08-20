# src/config.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
from collections import Counter


# ---------- locations ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "portfolio.yaml"

# ---------- public constants (populated at import) ----------
TICKERS: List[str]
BENCHMARKS: Dict[str, str]
DEFAULT_BENCH: str
RF_ANN_PCT: float
METADATA: Dict[str, Dict[str, str]]
DEFAULT_WEIGHTS: Dict[str, float]  # optional but handy

# ---------- internal helpers ----------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_metadata(tickers: List[str], meta: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for t in tickers:
        row = (meta or {}).get(t, {}) or {}
        out[t] = {
            "sector": row.get("sector", "Unknown"),
            "region": row.get("region", "Unknown"),
        }
    return out

def _find_duplicates(items):
    counts = Counter(items)
    return [k for k, v in counts.items() if v > 1]

def _warn_missing(section_name: str, expected: set, provided: set):
    missing = expected - provided
    extra = provided - expected
    if missing:
        logging.warning(f"[config] Missing in {section_name}: {sorted(missing)}")
    if extra:
        logging.warning(f"[config] Unknown entries in {section_name}: {sorted(extra)}")


def _normalize_weights(tickers: List[str], w: Dict[str, float] | None) -> Dict[str, float]:
    if not w:
        # equal weights if none provided
        eq = 1.0 / len(tickers) if tickers else 0.0
        return {t: eq for t in tickers}
    # keep only known tickers, fill missing as 0, then renormalize to 1
    vals = {t: float(w.get(t, 0.0)) for t in tickers}
    s = sum(vals.values())
    if s <= 0:
        logging.warning("default_weights sum <= 0; falling back to equal weights.")
        eq = 1.0 / len(tickers) if tickers else 0.0
        return {t: eq for t in tickers}
    return {t: v / s for t, v in vals.items()}

def _load() -> None:
    """Populate module-level constants from YAML with light validation."""
    cfg = _read_yaml(CONFIG_PATH)

    # ---- tickers
    tickers = cfg.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("`tickers` must be a non-empty list in portfolio.yaml")
    tickers = [str(t).strip() for t in tickers]

    # duplicates check
    dups = _find_duplicates(tickers)
    if dups:
        raise ValueError(f"`tickers` contains duplicates: {dups}")

    tickers_set = set(tickers)

    # ---- benchmarks (support old/new schema)
    benchmarks = cfg.get("benchmarks")
    if benchmarks is None:
        # backward compatibility: { "^GSPC": "S&P 500 ..." }
        benchmarks = cfg.get("benchmark", {})
    if not isinstance(benchmarks, dict) or not benchmarks:
        logging.warning("No `benchmarks` provided; defaulting to {'^GSPC': 'S&P 500 Index (US)'}")
        benchmarks = {"^GSPC": "S&P 500 Index (US)"}

    # defensive: drop empty/None keys
    benchmarks = {str(k).strip(): str(v).strip() for k, v in benchmarks.items() if k}
    if not benchmarks:
        raise ValueError("`benchmarks` must contain at least one valid symbol → description mapping")


    default_bench = cfg.get("default_benchmark")
    if not default_bench:
        # choose the first key deterministically
        default_bench = sorted(benchmarks.keys())[0]
    if default_bench not in benchmarks:
        raise ValueError(f"`default_benchmark` '{default_bench}' not found in `benchmarks` keys")

    # ---- risk-free (annual, decimal). Accept either key; prefer *_annual
    rf = cfg.get("risk_free_rate_annual", cfg.get("risk_free_rate", 0.0))
    try:
        rf = float(rf)
    except Exception as e:
        raise ValueError("`risk_free_rate_annual` (or `risk_free_rate`) must be a float (e.g., 0.04)") from e
    if rf < 0:
        logging.warning("Risk-free rate is negative; continuing anyway.")
    
    # ---- refresh intervals
    refresh_cfg = cfg.get("refresh_interval", {}) or {}
    refresh_market = int(refresh_cfg.get("market_open", 60))
    refresh_closed = int(refresh_cfg.get("market_closed", 600))
    if refresh_market <= 0 or refresh_closed <= 0:
        raise ValueError("Refresh intervals must be positive integers (seconds)")

    # ---- metadata
    metadata_raw = cfg.get("metadata", {}) or {}
    if not isinstance(metadata_raw, dict):
        raise ValueError("`metadata` must be a mapping of ticker -> {sector, region}")

    # warn on metadata keys that aren't in tickers, and vice versa
    _warn_missing("metadata", expected=tickers_set, provided=set(metadata_raw.keys()))

    metadata = _ensure_metadata(tickers, metadata_raw)

    # ---- default weights (optional)
    # default_w = _normalize_weights(tickers, cfg.get("default_weights", {}))
    # if abs(sum(default_w.values()) - 1.0) > 1e-6:
    #     logging.warning("Normalized `default_weights` do not sum to 1.0 exactly; renormalized internally.")

    default_w_raw = cfg.get("default_weights", {}) or {}
    if not isinstance(default_w_raw, dict):
        raise ValueError("`default_weights` must be a mapping of ticker -> float weight")

    # warn about missing/extra weight entries
    _warn_missing("default_weights", expected=tickers_set, provided=set(default_w_raw.keys()))

    default_w = _normalize_weights(tickers, default_w_raw)
    s = sum(default_w.values())
    if abs(s - 1.0) > 1e-6:
        logging.warning(f"[config] Weights renormalized to 1.0 (sum was {s:.6f})")

    # ---- export to module globals
    global TICKERS, BENCHMARKS, DEFAULT_BENCH, RF_ANN_PCT, METADATA, DEFAULT_WEIGHTS, REFRESH_SEC_MARKET, REFRESH_SEC_CLOSED
    TICKERS = tickers
    BENCHMARKS = benchmarks
    DEFAULT_BENCH = default_bench
    RF_ANN_PCT = rf
    METADATA = metadata
    DEFAULT_WEIGHTS = default_w
    REFRESH_SEC_MARKET = refresh_market
    REFRESH_SEC_CLOSED = refresh_closed

# populate at import
_load()

def reload_config() -> None:
    """Call this if you edit portfolio.yaml at runtime and want to refresh constants."""
    _load()
    logging.info("Configuration reloaded from portfolio.yaml")


# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#     print("TICKERS:", TICKERS)
#     print("DEFAULT_BENCH:", DEFAULT_BENCH)
#     print("RF_ANN_PCT:", RF_ANN_PCT)
#     print("Refresh intervals for Market - Open:", REFRESH_SEC_MARKET, "seconds and", "Closed:", REFRESH_SEC_CLOSED, "Seconds")
#     print("DEFAULT_WEIGHTS sum:", sum(DEFAULT_WEIGHTS.values()))

#     # find tickers with missing metadata fields
#     missing_meta = [t for t, meta in METADATA.items() if "sector" not in meta or "region" not in meta]
#     print("Missing metadata for tickers:", missing_meta if missing_meta else "None")

#     # print first 3 metadata entries as a sample
#     print("Metadata sample:", {k: METADATA[k] for k in list(METADATA)[:3]})
