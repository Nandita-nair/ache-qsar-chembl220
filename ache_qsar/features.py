"""Molecular featurisation: RDKit 2D descriptors and Morgan (ECFP4-style) count fingerprints."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from . import config

FEATURE_SETS = ("rdkit", "ecfp4", "combined")
_DESC_NAMES = [name for name, _ in Descriptors._descList]


def to_mols(smiles) -> list[Chem.Mol]:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("Invalid SMILES found - run curate() first.")
    return mols


def rdkit_descriptors(mols) -> pd.DataFrame:
    """All RDKit 2D descriptors; inf / absurdly large values become NaN (imputed in the pipeline)."""
    rows = []
    for m in mols:
        row = []
        for _, fn in Descriptors._descList:
            try:
                row.append(fn(m))
            except Exception:
                row.append(np.nan)
        rows.append(row)
    X = pd.DataFrame(rows, columns=_DESC_NAMES, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > 1e10)
    return X


def morgan_counts(mols, radius: int = config.MORGAN_RADIUS, n_bits: int = config.MORGAN_BITS) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.vstack([gen.GetCountFingerprintAsNumPy(m) for m in mols]).astype(float)
    return pd.DataFrame(arr, columns=[f"ecfp4_{i}" for i in range(n_bits)])


def build_features(smiles, kind: str) -> pd.DataFrame:
    if kind not in FEATURE_SETS:
        raise ValueError(f"kind must be one of {FEATURE_SETS}")
    mols = to_mols(smiles)
    if kind == "rdkit":
        return rdkit_descriptors(mols)
    if kind == "ecfp4":
        return morgan_counts(mols)
    return pd.concat([rdkit_descriptors(mols), morgan_counts(mols)], axis=1)


def tanimoto_max_to_reference(query_smiles, reference_smiles) -> np.ndarray:
    """Max Tanimoto similarity (Morgan r=2, bit vectors) of each query to the reference set."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_BITS)
    ref = [gen.GetFingerprint(m) for m in to_mols(reference_smiles)]
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), ref))
        for m in to_mols(query_smiles)
    ])
