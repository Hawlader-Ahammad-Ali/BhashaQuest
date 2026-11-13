import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"


@dataclass
class ModelBundle:
    name: str
    oof: np.ndarray
    test: np.ndarray | None = None


def load_train_labels():
    df = pd.read_csv(TRAIN_PATH)
    labels = sorted(df["label"].unique().tolist())
    y = df["label"].map({v: i for i, v in enumerate(labels)}).to_numpy(dtype=int)
    return df, labels, y


def load_prob(path: Path):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype=np.float64)


def validate_shapes(name, arr, n_rows, n_classes):
    if arr.shape[0] != n_rows:
        raise ValueError(f"[{name}] row count mismatch: expected {n_rows}, got {arr.shape[0]}")
    if arr.shape[1] != n_classes:
        raise ValueError(f"[{name}] class count mismatch: expected {n_classes}, got {arr.shape[1]}")


def macro_f1_from_probs(y_true, probs, label_order):
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = np.array([label_order[i] for i in pred_idx])
    y_true_labels = np.array([label_order[i] for i in y_true])
    macro = f1_score(y_true_labels, pred_labels, average="macro")
    return macro, pred_labels


def per_class_metrics(y_true, pred_labels, label_order):
    out = []
    for label in label_order:
        idx = np.where(np.array(label_order) == label)[0][0]
        gt = np.array([x == label for x in y_true])
        pr = np.array([x == label for x in pred_labels])
        p = precision_score(gt, pr, zero_division=0)
        r = recall_score(gt, pr, zero_division=0)
        f1 = f1_score(gt, pr, zero_division=0)
        out.append((label, p, r, f1))
    return out


def normalize_probs(arr, temperature=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    if temperature != 1.0:
        arr = np.power(arr + 1e-12, 1.0 / temperature)
    arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)
    return arr


def search_ensemble_weights(models, y_true, label_order):
    best = {"score": -1.0, "weights": None}
    candidates = []
    # simple, interpretable search for 4 pillars
    if len(models) == 4:
        ranges = [np.linspace(0.10, 0.45, 8), np.linspace(0.05, 0.25, 6), np.linspace(0.20, 0.45, 7), np.linspace(0.15, 0.35, 7)]
        for w1, w2, w3, w4 in product(*ranges):
            ws = np.array([w1, w2, w3, w4], dtype=np.float64)
            ws = ws / ws.sum()
            blended = np.zeros_like(models[0].oof, dtype=np.float64)
            for model, w in zip(models, ws):
                blended += w * model.oof
            macro, _ = macro_f1_from_probs(y_true, blended, label_order)
            candidates.append((macro, ws.copy()))
            if macro > best["score"]:
                best = {"score": macro, "weights": ws.copy()}
        return best, sorted(candidates, reverse=True)[:10]
    return best, []


def main():
    df, label_order, y_true = load_train_labels()
    print(f"labels: {label_order}")
    print(f"n_rows: {len(df)}, n_classes: {len(label_order)}")

    bundle_names = [
        "oof_bert_master.npy",
        "oof_xlmr_master.npy",
        "oof_tfidf_master.npy",
        "oof_v12_none.npy",
        "oof_v12_v1.npy",
        "oof_v12_v2.npy",
        "oof_mdeberta_master.npy",
    ]

    bundles = []
    for name in bundle_names:
        path = ROOT / name
        if not path.exists():
            print(f"skipping missing file: {name}")
            continue
        arr = load_prob(path)
        validate_shapes(name, arr, len(df), len(label_order))
        bundles.append(ModelBundle(name=name.replace(".npy", ""), oof=arr))

    # CatBoost view ensemble, if available
    cat_views = [b for b in bundles if b.name.startswith("oof_v12_")]
    if cat_views:
        cat_ensemble = sum((b.oof for b in cat_views), 0.0)
        # simple normalized weighting based on file naming order
        weights = {"oof_v12_none": 0.35, "oof_v12_v1": 0.35, "oof_v12_v2": 0.30}
        cat_names = [b.name for b in cat_views]
        merged = np.zeros_like(cat_views[0].oof)
        for b in cat_views:
            merged += weights.get(b.name, 1.0 / len(cat_views)) * b.oof
        merged = merged / sum(weights.get(b.name, 1.0 / len(cat_views)) for b in cat_views)
        print("\nCatBoost blended OOF Macro F1:")
        macro, pred = macro_f1_from_probs(y_true, merged, label_order)
        print(f"  macro_f1 = {macro:.6f}")
        print("  class-wise: ")
        for item in per_class_metrics([label_order[i] for i in y_true], pred, label_order):
            label, p, r, f = item
            print(f"    {label}: P={p:.4f} R={r:.4f} F1={f:.4f}")

    print("\nIndividual OOF Macro F1:")
    for b in bundles:
        macro, pred = macro_f1_from_probs(y_true, b.oof, label_order)
        print(f"  {b.name}: {macro:.6f}")

    # 4-pillar baseline using known weights from the repo workflow
    if len([b for b in bundles if b.name in {"oof_bert_master", "oof_xlmr_master", "oof_tfidf_master"}]) >= 3 and any(b.name.startswith("oof_v12_") for b in bundles):
        model_map = {}
        for b in bundles:
            model_map[b.name] = b
        four = [
            model_map["oof_bert_master"],
            model_map["oof_xlmr_master"],
            model_map["oof_tfidf_master"],
            model_map["oof_v12_v1"],
        ]
        weights = np.array([0.24, 0.10, 0.333, 0.327], dtype=np.float64)
        blended = np.zeros_like(four[0].oof, dtype=np.float64)
        for model, w in zip(four, weights):
            blended += w * model.oof
        blended = normalize_probs(blended, temperature=4.69)
        macro, pred = macro_f1_from_probs(y_true, blended, label_order)
        print("\n4-pillar baseline (repo-like weights):")
        print(f"  macro_f1 = {macro:.6f}")
        for item in per_class_metrics([label_order[i] for i in y_true], pred, label_order):
            label, p, r, f = item
            print(f"    {label}: P={p:.4f} R={r:.4f} F1={f:.4f}")

    # Optional: search best 5-pillar blend if mDeBERTa available
    if any(b.name == "oof_mdeberta_master" for b in bundles):
        models = [
            model_map["oof_bert_master"],
            model_map["oof_xlmr_master"],
            model_map["oof_tfidf_master"],
            model_map["oof_v12_none"],
            model_map["oof_mdeberta_master"],
        ]
        best, top = search_ensemble_weights(models[:4], y_true, label_order)
        print("\nEnsemble weight search (first 4 models):")
        print(f"  best_score = {best['score']:.6f}")
        print(f"  best_weights = {best['weights']}")
        for score, w in top[:5]:
            print(f"    score={score:.6f} weights={w}")


if __name__ == "__main__":
    main()


# End of script
