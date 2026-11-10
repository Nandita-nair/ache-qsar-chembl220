import numpy as np
import pandas as pd
import pytest

from ache_qsar import config
from ache_qsar.data import assign_activity_class, curate, ic50_nm_to_pic50, standardize_smiles
from ache_qsar.evaluate import bootstrap_ci, clf_metrics, reg_metrics
from ache_qsar.features import build_features
from ache_qsar.split import murcko_scaffolds, scaffold_split


def test_pic50_conversion():
    assert ic50_nm_to_pic50(1000) == pytest.approx(6.0)      # 1 uM
    assert ic50_nm_to_pic50(10) == pytest.approx(8.0)
    assert ic50_nm_to_pic50(10000) == pytest.approx(5.0)


def test_activity_classes_follow_config_thresholds():
    assert assign_activity_class(config.ACTIVE_PIC50) == "active"
    assert assign_activity_class(config.INACTIVE_PIC50) == "inactive"
    assert assign_activity_class(5.5) == "intermediate"


def test_standardize_strips_salts_and_neutralises():
    assert standardize_smiles("CCN(CC)CC.Cl") == standardize_smiles("CCN(CC)CC")
    assert standardize_smiles("CC(=O)[O-].[Na+]") == standardize_smiles("CC(=O)O")
    assert standardize_smiles("not a smiles") is None
    assert standardize_smiles("C") is None                    # below the minimum heavy-atom count


def _raw():
    return pd.DataFrame({
        "molecule_chembl_id": ["A", "A2", "B", "C", "C2", "D"],
        "canonical_smiles": ["c1ccccc1CCN", "c1ccccc1CCN.Cl", "CCOC(=O)c1ccccc1", "c1ccncc1CC", "c1ccncc1CC", "CCCCCCO"],
        "standard_value": [100.0, 400.0, 50.0, 10.0, 100000.0, 20.0],
        "standard_relation": ["=", "=", "=", "=", "=", ">"],      # D is censored -> dropped
        "standard_units": ["nM"] * 6,
        "assay_type": ["B"] * 6,
    })


def test_curate_filters_and_aggregates_replicates():
    cur, rep = curate(_raw())
    assert rep["raw_records"] == 6
    assert cur["smiles"].is_unique                              # one row per parent compound
    assert len(cur) == 2                                        # A (2 replicates merged), B; C dropped (SD > 1); D censored
    a = cur[cur["smiles"] == standardize_smiles("c1ccccc1CCN")].iloc[0]
    assert a["n_measurements"] == 2
    assert a["pIC50"] == pytest.approx(np.median(ic50_nm_to_pic50([100.0, 400.0])))


def test_scaffold_split_has_no_scaffold_leakage():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccncc1C", "c1ccncc1CC", "C1CCNCC1C",
              "C1CCNCC1CC", "c1ccc2ccccc2c1C", "c1ccsc1C", "c1ccoc1C", "c1cc[nH]c1C", "C1CCOCC1C"] * 5
    smiles = [standardize_smiles(s) for s in smiles]
    tr, te, scaf = scaffold_split(smiles, test_size=0.2, seed=1)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == len(smiles)
    assert set(scaf[tr]).isdisjoint(set(scaf[te]))
    assert len(te) > 0


def test_features_have_expected_shape_and_no_infinities():
    smiles = ["c1ccccc1CCN", "CCOC(=O)c1ccccc1"]
    X = build_features(smiles, "combined")
    assert X.shape[0] == 2 and X.shape[1] > 2000
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.4, 0.7, 0.3])
    assert clf_metrics(y, p)["roc_auc"] == pytest.approx(1.0)
    r = reg_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert r["rmse"] == pytest.approx(0.0) and r["r2"] == pytest.approx(1.0)
    ci = bootstrap_ci(y, p, clf_metrics, n_boot=50, seed=0)
    assert ci["roc_auc"][0] <= 1.0 <= ci["roc_auc"][1] + 1e-9
