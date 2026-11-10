"""End-to-end study: data -> curation -> features -> scaffold split -> tuned models -> validation -> report."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold

from . import __version__, config, plots
from .data import curate, load_or_fetch
from .evaluate import bootstrap_ci, clf_metrics, reg_metrics
from .features import FEATURE_SETS, build_features, tanimoto_max_to_reference
from .models import baseline, classifier_specs, make_pipeline, regressor_specs, shrink_grid
from .split import murcko_scaffolds, random_split, scaffold_split

log = logging.getLogger("ache_qsar")
TARGET_LABELS = {"CHEMBL220": "human AChE", "CHEMBL4078": "electric eel AChE", "CHEMBL4768": "bovine AChE"}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _fmt_ci(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})"


def update_readme(readme: Path, summary_md: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(f"{head}{README_START}\n{summary_md}\n{README_END}{tail}", encoding="utf-8")
    return True


def _fit_score(pipe, Xtr, y_perm, Xte, yte, task: str) -> float:
    m = clone(pipe).fit(Xtr, y_perm)
    if task == "classification":
        return clf_metrics(yte, m.predict_proba(Xte)[:, 1])["roc_auc"]
    return reg_metrics(yte, m.predict(Xte))["r2"]


def y_randomization(pipe, Xtr, ytr, Xte, yte, task: str, n: int, seed: int, n_jobs: int = 1) -> list[float]:
    """Refit the selected model on shuffled labels (in parallel); a valid model must score far worse than the real one."""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(ytr) for _ in range(n)]
    return Parallel(n_jobs=n_jobs)(delayed(_fit_score)(pipe, Xtr, yp, Xte, yte, task) for yp in perms)


# --------------------------------------------------------------------------- #
# main study
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    t0 = time.time()
    root = Path(args.out).resolve()
    fig_dir, res_dir = root / "results" / "figures", root / "results"
    for d in (fig_dir, root / "data" / "processed", root / "models"):
        d.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    n_boot = 200 if args.fast else config.N_BOOTSTRAP
    n_yrand = 5 if args.fast else config.N_Y_RANDOMIZATIONS
    folds = 3 if args.fast else args.folds
    fsets = [f.strip() for f in args.features.split(",")]
    for f in fsets:
        if f not in FEATURE_SETS:
            raise SystemExit(f"Unknown feature set '{f}'. Choose from {FEATURE_SETS}.")

    # ---- 1. data ---------------------------------------------------------- #
    if args.input_csv:
        raw, chembl_version = pd.read_csv(args.input_csv, low_memory=False), "user-provided file"
    else:
        raw, chembl_version = load_or_fetch(args.target, root / "data" / "raw", args.refresh)
    cur, cur_report = curate(raw)
    cur.to_csv(root / "data" / "processed" / f"{args.target}_curated.csv", index=False)
    (res_dir / "curation_report.json").write_text(json.dumps(cur_report, indent=2))
    log.info("Curation: %s", cur_report)
    if len(cur) < 100:
        raise SystemExit(f"Only {len(cur)} compounds after curation - too few for a QSAR study.")
    label = TARGET_LABELS.get(args.target, args.target)
    plots.plot_eda(cur, fig_dir / "01_eda.png", label)

    # ---- 2. features + split --------------------------------------------- #
    smiles = cur["smiles"].tolist()
    X = {fs: build_features(smiles, fs) for fs in fsets}
    scaff = np.array(murcko_scaffolds(smiles))
    if args.split == "scaffold":
        tr, te, _ = scaffold_split(smiles, seed=seed)
    else:
        tr, te = random_split(len(cur), seed=seed)
    split_info = {
        "split": args.split, "n_train": int(len(tr)), "n_test": int(len(te)),
        "scaffolds_shared_between_train_and_test": int(len(set(scaff[tr]) & set(scaff[te]))),
    }
    log.info("Split: %s", split_info)

    y_reg = cur["pIC50"].to_numpy()
    is_bin = (cur["activity_class"] != "intermediate").to_numpy()
    y_clf = (cur["activity_class"] == "active").astype(int).to_numpy()
    tasks = {
        "classification": dict(tr=tr[is_bin[tr]], te=te[is_bin[te]], y=y_clf, specs=classifier_specs(seed),
                               scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"}, refit="roc_auc"),
        "regression": dict(tr=tr, te=te, y=y_reg, specs=regressor_specs(seed),
                           scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, refit="rmse"),
    }

    # ---- 3. tune + evaluate ---------------------------------------------- #
    # Every finished (task, features, model) fit is checkpointed, so an interrupted run resumes where it stopped.
    ckpt_dir = res_dir / ".checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_sig = hashlib.md5(f"{smiles}{cur['pIC50'].round(6).tolist()}{seed}{folds}{args.fast}{args.split}".encode()).hexdigest()
    tables, fitted, test_pred = {}, {}, {}
    for task, cfg in tasks.items():
        itr, ite, y = cfg["tr"], cfg["te"], cfg["y"]
        if task == "classification" and (len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2):
            log.warning("Skipping classification: a split has only one class.")
            continue
        rows = []
        for fs in fsets:
            Xtr, Xte = X[fs].iloc[itr], X[fs].iloc[ite]
            if fs == fsets[0]:   # trivial baseline (class prior / training mean) - independent of features
                base = make_pipeline(baseline(task)).fit(Xtr, y[itr])
                bp = base.predict_proba(Xte)[:, 1] if task == "classification" else base.predict(Xte)
                m = clf_metrics(y[ite], bp) if task == "classification" else reg_metrics(y[ite], bp)
                rows.append({"task": task, "feature_set": "-", "model": "Baseline (prior/mean)", "baseline": True,
                             **{f"test_{k}": v for k, v in m.items()}})
            for name, (est, grid) in cfg["specs"].items():
                t1 = time.time()
                ck = ckpt_dir / f"{task}__{fs}__{name}.joblib"
                sig = hashlib.md5(f"{data_sig}{task}{fs}{name}{grid}".encode()).hexdigest()
                if ck.exists() and not args.no_resume:
                    try:
                        saved = joblib.load(ck)
                    except Exception:
                        saved = None
                    if saved and saved["sig"] == sig:
                        rows.append(saved["row"])
                        fitted[(task, fs, name)], test_pred[(task, fs, name)] = saved["estimator"], saved["pred"]
                        log.info("%-14s %-9s %-20s resumed from checkpoint", task, fs, name)
                        continue
                gs = GridSearchCV(make_pipeline(est), shrink_grid(grid) if args.fast else grid,
                                  cv=GroupKFold(n_splits=folds), scoring=cfg["scoring"], refit=cfg["refit"],
                                  n_jobs=args.n_jobs)
                gs.fit(Xtr, y[itr], groups=scaff[itr])
                res, bi = gs.cv_results_, gs.best_index_
                pred = gs.predict_proba(Xte)[:, 1] if task == "classification" else gs.predict(Xte)
                m = clf_metrics(y[ite], pred) if task == "classification" else reg_metrics(y[ite], pred)
                row = {"task": task, "feature_set": fs, "model": name, "baseline": False,
                       "best_params": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()}, default=str)}
                if task == "classification":
                    row.update(cv_roc_auc=res["mean_test_roc_auc"][bi], cv_roc_auc_std=res["std_test_roc_auc"][bi],
                               cv_balanced_accuracy=res["mean_test_balanced_accuracy"][bi])
                else:
                    row.update(cv_rmse=-res["mean_test_rmse"][bi], cv_rmse_std=res["std_test_rmse"][bi],
                               cv_r2=res["mean_test_r2"][bi])
                row.update({f"test_{k}": v for k, v in m.items()})
                rows.append(row)
                fitted[(task, fs, name)] = gs.best_estimator_
                test_pred[(task, fs, name)] = pred
                joblib.dump({"sig": sig, "row": row, "estimator": gs.best_estimator_, "pred": pred}, ck, compress=3)
                log.info("%-14s %-9s %-20s CV=%.3f  test=%s  (%.0fs)", task, fs, name,
                         row.get("cv_roc_auc", row.get("cv_rmse")),
                         f"{m['roc_auc']:.3f}" if task == "classification" else f"R2 {m['r2']:.3f}", time.time() - t1)
        df = pd.DataFrame(rows)
        real = df[~df["baseline"]]
        # model selection uses ONLY cross-validation on the training set - never the test set
        best_i = real["cv_roc_auc"].idxmax() if task == "classification" else real["cv_rmse"].idxmin()
        df["selected"] = df.index == best_i
        tables[task] = df
        df.drop(columns="baseline").to_csv(res_dir / f"{task}_results.csv", index=False)

    # ---- 4. selected-model analysis -------------------------------------- #
    analysis: dict = {"selected": {}}
    for task, df in tables.items():
        cfg = tasks[task]
        itr, ite, y = cfg["tr"], cfg["te"], cfg["y"]
        sel = df[df["selected"]].iloc[0]
        key = (task, sel["feature_set"], sel["model"])
        pipe, pred = fitted[key], test_pred[key]
        fn = clf_metrics if task == "classification" else reg_metrics
        metrics = fn(y[ite], pred)
        log.info("Selected %s model: %s (%s). Running bootstrap CIs + %d y-randomisation fits ...",
                 task, sel["model"], sel["feature_set"], n_yrand)
        ci = bootstrap_ci(y[ite], pred, fn, n_boot, seed)
        scr = y_randomization(pipe, X[sel["feature_set"]].iloc[itr], y[itr], X[sel["feature_set"]].iloc[ite], y[ite],
                              task, n_yrand, seed, args.n_jobs)
        log.info("  y-randomisation done.")
        real_score = metrics["roc_auc"] if task == "classification" else metrics["r2"]
        analysis["selected"][task] = {
            "model": sel["model"], "feature_set": sel["feature_set"], "best_params": sel["best_params"],
            "metrics": metrics, "ci95": ci,
            "y_randomization": {"metric": "ROC-AUC" if task == "classification" else "R2", "real": real_score,
                                "scrambled_mean": float(np.mean(scr)), "scrambled_max": float(np.max(scr)), "n": n_yrand},
        }
        tag = "clf" if task == "classification" else "reg"
        plots.plot_model_comparison(df[~df["baseline"]], task, fig_dir / f"02_{tag}_model_comparison.png")
        plots.plot_y_randomization(real_score, scr, "ROC-AUC" if task == "classification" else "R²",
                                   fig_dir / f"06_{tag}_y_randomization.png")
        if task == "classification":
            best_per_fs = df[~df["baseline"]].sort_values("cv_roc_auc", ascending=False).groupby("feature_set").head(1)
            curves = {f"{r.model} ({r.feature_set})": test_pred[(task, r.feature_set, r.model)] for r in best_per_fs.itertuples()}
            plots.plot_roc(y[ite], curves, fig_dir / "03_roc_curves.png")
            plots.plot_confusion(y[ite], pred, fig_dir / "04_confusion_matrix.png",
                                 f"{sel['model']} ({sel['feature_set']}) - test set")
        else:
            plots.plot_pred_vs_actual(y[ite], pred, metrics, fig_dir / "05_pred_vs_actual.png",
                                      f"{sel['model']} ({sel['feature_set']}) - scaffold test set")
            sim = tanimoto_max_to_reference([smiles[i] for i in ite], [smiles[i] for i in itr])
            err = np.abs(pred - y[ite])
            plots.plot_applicability_domain(sim, err, fig_dir / "07_applicability_domain.png")
            bins = pd.cut(sim, [0, 0.4, 0.6, 0.8, 1.0001], labels=["<0.4", "0.4-0.6", "0.6-0.8", "≥0.8"], right=False)
            ad = pd.DataFrame({"bin": bins, "se": (pred - y[ite]) ** 2}).groupby("bin", observed=True)["se"].agg(["size", "mean"])
            analysis["applicability_domain"] = [
                {"similarity": str(i), "n": int(r["size"]), "rmse": float(np.sqrt(r["mean"]))} for i, r in ad.iterrows()]
        artifact = {"task": task, "feature_set": sel["feature_set"], "model_name": sel["model"], "pipeline": pipe,
                    "train_smiles": [smiles[i] for i in itr], "target": args.target, "test_metrics": metrics,
                    "package_version": __version__}
        joblib.dump(artifact, root / "models" / f"best_{task}.joblib", compress=3)

    # interpretability: permutation importance of the best RDKit-descriptor regressor
    if "regression" in tables and "rdkit" in fsets:
        d = tables["regression"]
        d = d[(~d["baseline"]) & (d["feature_set"] == "rdkit")]
        r = d.loc[d["cv_rmse"].idxmin()]
        itr, ite = tasks["regression"]["tr"], tasks["regression"]["te"]
        log.info("Computing permutation importance for the best RDKit-descriptor regressor (%s) ...", r["model"])
        pi = permutation_importance(fitted[("regression", "rdkit", r["model"])], X["rdkit"].iloc[ite], y_reg[ite],
                                    scoring="neg_root_mean_squared_error", n_repeats=5, random_state=seed, n_jobs=args.n_jobs)
        imp = pd.Series(pi.importances_mean, index=X["rdkit"].columns).sort_values(ascending=False)
        imp.to_csv(res_dir / "descriptor_importance.csv", header=["importance"])
        plots.plot_importance(imp, fig_dir / "08_descriptor_importance.png",
                              f"Top RDKit descriptors - {r['model']} (test set)")
        analysis["importance_model"] = r["model"]
        analysis["top_descriptors"] = imp.head(10).index.tolist()

    # ---- 5. report -------------------------------------------------------- #
    run_info = {
        "package_version": __version__, "target": args.target, "target_label": label,
        "chembl_version": chembl_version, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "rdkit": rdkit.__version__, "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "seed": seed, "cv_folds": folds, "fast_mode": bool(args.fast),
        "feature_sets": fsets, "curation": cur_report, "split": split_info,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (res_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    (res_dir / "metrics_summary.json").write_text(json.dumps(analysis, indent=2, default=float))
    summary = render_summary(run_info, tables, analysis)
    (res_dir / "RESULTS.md").write_text(summary.replace("{FIG}", "figures/"), encoding="utf-8")
    if not args.no_readme_update and update_readme(root / "README.md", summary.replace("{FIG}", "results/figures/")):
        log.info("README.md results section updated.")
    log.info("Done in %.1f min. Outputs in %s", (time.time() - t0) / 60, res_dir)
    return {"run_info": run_info, "analysis": analysis, "tables": tables}


def render_summary(info: dict, tables: dict, analysis: dict) -> str:
    c, sp = info["curation"], info["split"]
    out = [f"_Auto-generated by `run_pipeline.py` on {info['timestamp_utc'][:10]} · {info['chembl_version']} · "
           f"seed {info['seed']} · RDKit {info['rdkit']} · scikit-learn {info['scikit_learn']}"
           + (" · **FAST/SMOKE-TEST MODE - not final results**" if info["fast_mode"] else "") + "_", ""]
    cc = c["class_counts"]
    out += ["### Dataset", "", _md_table(pd.DataFrame([
        ["Raw IC50 records downloaded", c["raw_records"]],
        ["After exact-value / nM / biochemical-assay filter", c["after_relation_units_assay_filter"]],
        ["After structure standardisation", c["after_standardisation"]],
        ["Unique parent compounds", c["unique_compounds_before_replicate_filter"]],
        ["**Final curated compounds** (replicate SD ≤ 1 log unit)", f"**{c['unique_compounds_final']}**"],
        ["Active / intermediate / inactive", f"{cc.get('active', 0)} / {cc.get('intermediate', 0)} / {cc.get('inactive', 0)}"],
    ], columns=["Curation step", "Count"])), "", "![EDA]({FIG}01_eda.png)", ""]
    out += ["### Validation protocol", "",
            f"- **{sp['split'].capitalize()} split**: {sp['n_train']} train / {sp['n_test']} test compounds; "
            f"{sp['scaffolds_shared_between_train_and_test']} Bemis-Murcko scaffolds shared between the two sets.",
            f"- Hyper-parameters tuned by **{info['cv_folds']}-fold scaffold-grouped cross-validation on the training set only**; "
            "the best model is chosen by CV score, and the test set is used once for reporting.",
            "- Imputation, variance filtering and scaling are fitted inside each CV fold (sklearn `Pipeline`).", ""]
    if "classification" in tables:
        d = tables["classification"].copy()
        d = d.sort_values(["baseline", "cv_roc_auc"], ascending=[True, False])
        t = pd.DataFrame({
            "Model": [("★ " if s else "") + m for s, m in zip(d["selected"], d["model"])],
            "Features": d["feature_set"],
            "CV ROC-AUC": [("-" if b else f"{v:.3f} ± {s:.3f}") for b, v, s in zip(d["baseline"], d["cv_roc_auc"], d["cv_roc_auc_std"])],
            "Test ROC-AUC": d["test_roc_auc"].map("{:.3f}".format),
            "Balanced acc.": d["test_balanced_accuracy"].map("{:.3f}".format),
            "MCC": d["test_mcc"].map("{:.3f}".format),
        })
        out += ["### Classification: active (IC50 ≤ 1 µM) vs inactive (IC50 ≥ 10 µM)", "", _md_table(t), "",
                "★ = model selected by cross-validation.", "",
                "![ROC]({FIG}03_roc_curves.png)", "", "![Model comparison]({FIG}02_clf_model_comparison.png)", ""]
    if "regression" in tables:
        d = tables["regression"].copy()
        d = d.sort_values(["baseline", "cv_rmse"], ascending=[True, True])
        t = pd.DataFrame({
            "Model": [("★ " if s else "") + m for s, m in zip(d["selected"], d["model"])],
            "Features": d["feature_set"],
            "CV RMSE": [("-" if b else f"{v:.3f} ± {s:.3f}") for b, v, s in zip(d["baseline"], d["cv_rmse"], d["cv_rmse_std"])],
            "Test R²": d["test_r2"].map("{:.3f}".format),
            "Test RMSE": d["test_rmse"].map("{:.3f}".format),
            "Test MAE": d["test_mae"].map("{:.3f}".format),
            "Pearson r": d["test_pearson_r"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        })
        out += ["### Regression: pIC50 prediction (all compounds)", "", _md_table(t), "",
                "![Predicted vs actual]({FIG}05_pred_vs_actual.png)", ""]
    out += ["### Selected models (held-out test set, bootstrap 95% CI)", ""]
    for task, a in analysis["selected"].items():
        keys = ["roc_auc", "balanced_accuracy", "mcc"] if task == "classification" else ["r2", "rmse", "mae"]
        pretty = {"roc_auc": "ROC-AUC", "balanced_accuracy": "balanced accuracy", "mcc": "MCC", "r2": "R²", "rmse": "RMSE", "mae": "MAE"}
        txt = "; ".join(f"{pretty[k]} {_fmt_ci(a['metrics'][k], a['ci95'][k])}" for k in keys)
        out.append(f"- **{task.capitalize()}**: {a['model']} on `{a['feature_set']}` features: {txt}")
    out += ["", "### Robustness checks", ""]
    for task, a in analysis["selected"].items():
        y = a["y_randomization"]
        out.append(f"- **Y-randomisation ({task})**: real {y['metric']} = {y['real']:.3f} vs. "
                   f"{y['scrambled_mean']:.3f} mean / {y['scrambled_max']:.3f} max over {y['n']} label-shuffled models.")
    if "applicability_domain" in analysis:
        ad = pd.DataFrame(analysis["applicability_domain"])
        t = pd.DataFrame({"Max Tanimoto to training set": ad["similarity"], "Test compounds": ad["n"], "RMSE": ad["rmse"].map("{:.2f}".format)})
        out += ["", "**Applicability domain** - test-set RMSE binned by each compound's maximum Tanimoto similarity to the training set:", "",
                _md_table(t), "", "![AD]({FIG}07_applicability_domain.png)"]
    if "top_descriptors" in analysis:
        out += ["", f"**Most influential RDKit descriptors** ({analysis['importance_model']}, permutation importance): "
                + ", ".join(f"`{d}`" for d in analysis["top_descriptors"]) + ".", "", "![Importance]({FIG}08_descriptor_importance.png)"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the AChE QSAR study end-to-end.")
    p.add_argument("--target", default=config.DEFAULT_TARGET, help="ChEMBL target ID (default CHEMBL220 = human AChE)")
    p.add_argument("--input-csv", help="Use a local raw activity CSV instead of querying ChEMBL "
                                       "(needs canonical_smiles + standard_value [nM] columns)")
    p.add_argument("--refresh", action="store_true", help="Re-download from ChEMBL even if a cached snapshot exists")
    p.add_argument("--features", default="rdkit,ecfp4", help=f"Comma-separated subset of {FEATURE_SETS}")
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=".", help="Repository root for data/, results/, models/")
    p.add_argument("--fast", action="store_true", help="Tiny grids / few repeats: smoke test only")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and retrain every model")
    p.add_argument("--no-readme-update", action="store_true")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(build_parser().parse_args(argv))
