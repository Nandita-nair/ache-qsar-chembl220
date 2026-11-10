"""Data retrieval from ChEMBL and compound curation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from . import config

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

RAW_FIELDS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
    "standard_value", "standard_units", "assay_chembl_id", "assay_type",
    "document_chembl_id",
]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def fetch_chembl_ic50(target_id: str) -> tuple[pd.DataFrame, str]:
    """Download all IC50 records for a ChEMBL target. Returns (dataframe, chembl_version)."""
    from chembl_webresource_client.new_client import new_client  # lazy: allows offline use

    log.info("Querying ChEMBL for %s IC50 records ...", target_id)
    records = new_client.activity.filter(
        target_chembl_id=target_id, standard_type="IC50"
    ).only(RAW_FIELDS)
    df = pd.DataFrame.from_records(list(records))
    try:
        version = new_client.status.status["chembl_db_version"]
    except Exception:  # pragma: no cover - network dependent
        version = "unknown"
    log.info("Retrieved %d raw records (%s)", len(df), version)
    return df, version


def load_or_fetch(target_id: str, raw_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Use the cached raw snapshot if present so the study is reproducible offline."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / f"{target_id}_ic50_raw.csv"
    meta = raw_dir / f"{target_id}_ic50_raw.json"
    if csv.exists() and not refresh:
        log.info("Using cached raw data: %s", csv)
        version = json.loads(meta.read_text())["chembl_version"] if meta.exists() else "unknown"
        return pd.read_csv(csv, low_memory=False), version
    df, version = fetch_chembl_ic50(target_id)
    df.to_csv(csv, index=False)
    meta.write_text(json.dumps({"target": target_id, "chembl_version": version, "n_records": len(df)}))
    return df, version


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #
_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize_smiles(smiles: str) -> str | None:
    """Parent-structure standardisation: cleanup, strip salts, neutralise, canonical SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        if not (config.MIN_HEAVY_ATOMS <= n_heavy <= config.MAX_HEAVY_ATOMS):
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def ic50_nm_to_pic50(ic50_nm):
    """pIC50 = -log10(IC50 [M]) = 9 - log10(IC50 [nM])."""
    return 9.0 - np.log10(np.asarray(ic50_nm, dtype=float))


def assign_activity_class(pic50: float) -> str:
    if pic50 >= config.ACTIVE_PIC50:
        return "active"
    if pic50 <= config.INACTIVE_PIC50:
        return "inactive"
    return "intermediate"


def curate(raw: pd.DataFrame, max_std: float = config.MAX_REPLICATE_STD) -> tuple[pd.DataFrame, dict]:
    """Turn raw ChEMBL activity records into one clean row per unique parent compound.

    Steps (each count is recorded in the returned report):
      1. keep exact ('=') IC50 values in nM from biochemical assays (assay_type 'B')
      2. drop missing / non-positive values
      3. standardise structures (salts removed, neutralised, size filter)
      4. aggregate replicate measurements per compound (median pIC50)
      5. drop compounds whose replicates disagree by more than `max_std` log units
    """
    rep = {"raw_records": int(len(raw))}
    df = raw.copy()

    if "standard_relation" in df:
        df = df[df["standard_relation"] == "="]
    if "standard_units" in df:
        df = df[df["standard_units"] == "nM"]
    if "assay_type" in df:
        df = df[df["assay_type"] == "B"]
    rep["after_relation_units_assay_filter"] = int(len(df))

    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0]
    rep["after_missing_value_filter"] = int(len(df))

    mapping = {s: standardize_smiles(s) for s in df["canonical_smiles"].unique()}
    df["smiles"] = df["canonical_smiles"].map(mapping)
    df = df.dropna(subset=["smiles"]).copy()
    rep["after_standardisation"] = int(len(df))

    df["pIC50"] = ic50_nm_to_pic50(df["standard_value"])
    grouped = df.groupby("smiles")
    agg = grouped["pIC50"].agg(pIC50="median", n_measurements="size", pIC50_std="std").reset_index()
    agg["pIC50_std"] = agg["pIC50_std"].fillna(0.0)
    agg["molecule_chembl_id"] = agg["smiles"].map(grouped["molecule_chembl_id"].first())
    rep["unique_compounds_before_replicate_filter"] = int(len(agg))
    rep["compounds_with_replicates"] = int((agg["n_measurements"] > 1).sum())

    agg = agg[agg["pIC50_std"] <= max_std].reset_index(drop=True)
    rep["unique_compounds_final"] = int(len(agg))

    agg["activity_class"] = agg["pIC50"].map(assign_activity_class)
    rep["class_counts"] = {k: int(v) for k, v in agg["activity_class"].value_counts().items()}
    cols = ["molecule_chembl_id", "smiles", "pIC50", "n_measurements", "pIC50_std", "activity_class"]
    return agg[cols], rep
