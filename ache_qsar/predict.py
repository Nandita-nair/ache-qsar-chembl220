"""Score new molecules with the trained models:  python -m ache_qsar.predict --smiles "CCO" ..."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import standardize_smiles
from .features import build_features, tanimoto_max_to_reference

IN_DOMAIN_SIMILARITY = 0.4   # below this max-Tanimoto the prediction is an extrapolation


def predict(smiles: list[str], models_dir: Path = Path("models")) -> pd.DataFrame:
    std = [standardize_smiles(s) for s in smiles]
    out = pd.DataFrame({"input_smiles": smiles, "standardized_smiles": std})
    valid = out["standardized_smiles"].notna().to_numpy()
    if not valid.any():
        raise ValueError("None of the input SMILES could be parsed.")
    vs = out.loc[valid, "standardized_smiles"].tolist()

    reg_path, clf_path = models_dir / "best_regression.joblib", models_dir / "best_classification.joblib"
    if not reg_path.exists() and not clf_path.exists():
        raise FileNotFoundError(f"No trained models in {models_dir}/ - run `python run_pipeline.py` first.")

    if reg_path.exists():
        art = joblib.load(reg_path)
        pred = art["pipeline"].predict(build_features(vs, art["feature_set"]))
        out.loc[valid, "pred_pIC50"] = pred
        out.loc[valid, "pred_IC50_nM"] = 10 ** (9 - pred)
        sim = tanimoto_max_to_reference(vs, art["train_smiles"])
        out.loc[valid, "max_tanimoto_to_train"] = sim
        out.loc[valid, "in_applicability_domain"] = sim >= IN_DOMAIN_SIMILARITY
    if clf_path.exists():
        art = joblib.load(clf_path)
        out.loc[valid, "prob_active"] = art["pipeline"].predict_proba(build_features(vs, art["feature_set"]))[:, 1]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict AChE inhibition (pIC50, P(active)) for new molecules.")
    p.add_argument("--smiles", action="append", default=[], help="SMILES string (repeatable)")
    p.add_argument("--input-csv", help="CSV with a 'smiles' column")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--output", help="Write predictions to this CSV")
    a = p.parse_args(argv)
    smiles = list(a.smiles)
    if a.input_csv:
        smiles += pd.read_csv(a.input_csv)["smiles"].tolist()
    if not smiles:
        p.error("Provide --smiles and/or --input-csv")
    res = predict(smiles, Path(a.models_dir))
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.round(3).to_string(index=False))
    if a.output:
        res.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
