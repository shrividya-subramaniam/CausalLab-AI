"""Loading, sampling and the arm split.

Three rules from the spec live here:
  * Never load the 2 GB CSV into memory: DuckDB streams it to Parquet.
  * Sample by search (srch_id), never by row, because rows in a search are not independent.
  * The firewall: each arm has its own loader, and every load is logged, so a test can
    prove the Q1 estimation path never read a randomly ranked row.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from .config import load_config, processed_dir, repo_path

Arm = Literal["observational", "experimental"]
ARM_VALUE = {"observational": 0, "experimental": 1}

# (arm, calling module) for every arm load in this process. Read by the firewall test.
ACCESS_LOG: list[tuple[str, str]] = []


def convert_csv_to_parquet(force: bool = False, cfg: dict | None = None) -> Path:
    """Stream train.csv to Parquet. The file marks missing values with the text NULL."""
    cfg = cfg or load_config()
    out = processed_dir(cfg) / "train.parquet"
    if out.exists() and not force:
        print(f"{out} already exists; pass force=True to rebuild.")
        return out
    src = repo_path(cfg["paths"]["raw_csv"])
    if not src.exists():
        raise FileNotFoundError(f"train.csv not found at {src}. Set paths.raw_csv in config.yaml.")
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    try:  # converting 2 GB: progress is useful here (DuckDB refuses inside Jupyter without ipywidgets)
        con.execute("SET enable_progress_bar = true")
    except duckdb.Error:
        pass
    con.execute(
        f"COPY (SELECT * FROM read_csv('{src.as_posix()}', nullstr='NULL', header=true, sample_size=-1)) "
        f"TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
    )
    return out


def search_hash_keep(srch_id: str, fraction: float, seed: int) -> str:
    """SQL expression keeping a fixed share of searches.

    A multiplicative hash of srch_id, so the same seed always keeps the same searches,
    on any machine and any DuckDB version.
    """
    threshold = int(fraction * 4294967296)
    return f"((({srch_id}::BIGINT * 2654435761 + {seed % 100000} * 40503) % 4294967296) < {threshold})"


def sample_path(cfg: dict | None = None, synthetic: bool = False) -> Path:
    cfg = cfg or load_config()
    tag = "synthetic" if synthetic else f"sample_{int(round(cfg['data']['sample_fraction'] * 1000)):03d}"
    return processed_dir(cfg) / f"{tag}.parquet"


def build_sample(force: bool = False, cfg: dict | None = None) -> Path:
    """Keep a share of whole searches from the full Parquet file."""
    cfg = cfg or load_config()
    out = sample_path(cfg)
    if out.exists() and not force:
        return out
    src = processed_dir(cfg) / "train.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing. Run `python -m causallab convert` first.")
    keep = search_hash_keep("srch_id", cfg["data"]["sample_fraction"], cfg["seed"])
    duckdb.execute(
        f"COPY (SELECT * FROM '{src.as_posix()}' WHERE {keep} ORDER BY srch_id, position) "
        f"TO '{out.as_posix()}' (FORMAT parquet)"
    )
    return out


def features_path(cfg: dict | None = None, synthetic: bool = False) -> Path:
    return sample_path(cfg, synthetic).with_name(sample_path(cfg, synthetic).stem + "_features.parquet")


def _caller() -> str:
    frame = inspect.stack()[2]
    mod = inspect.getmodule(frame.frame)
    return mod.__name__ if mod else "unknown"


def _read_arm(arm: Arm, cfg: dict, synthetic: bool, caller: str) -> pd.DataFrame:
    ACCESS_LOG.append((arm, caller))
    path = features_path(cfg, synthetic)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python -m causallab prepare` first.")
    df = pd.read_parquet(path, filters=[("random_bool", "==", ARM_VALUE[arm])])
    df.attrs["arm"] = arm
    return df


def load_arm(arm: Arm, cfg: dict | None = None, synthetic: bool = False) -> pd.DataFrame:
    """Load one arm of the feature table. Every call is logged for the firewall test."""
    return _read_arm(arm, cfg or load_config(), synthetic, _caller())


def loader_for(target_arm: Arm):
    """Return a loader bound to one arm. The agent's estimation tools receive only this,
    so a run whose state says 'observational' has no way to read the other arm."""
    def _load(cfg: dict | None = None, synthetic: bool = False) -> pd.DataFrame:
        return _read_arm(target_arm, cfg or load_config(), synthetic, _caller())
    _load.arm = target_arm  # type: ignore[attr-defined]
    return _load
