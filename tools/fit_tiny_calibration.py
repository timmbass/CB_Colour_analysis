#!/usr/bin/env python3
"""Fit tiny calibration params (alpha, bias, gamma) for season logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
    p = _softmax(logits)
    idx = (np.arange(len(y_true)), y_true)
    return float(-np.mean(np.log(np.clip(p[idx], 1e-12, 1.0))))


def _fit_bias_gd(z_mix: np.ndarray, y_true: np.ndarray, steps: int = 200, lr: float = 0.05) -> np.ndarray:
    bias = np.zeros(4, dtype=np.float64)
    n = float(len(y_true))
    y_onehot = np.zeros((len(y_true), 4), dtype=np.float64)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0

    for _ in range(steps):
        logits = z_mix + bias[None, :]
        probs = _softmax(logits)
        grad = np.sum(probs - y_onehot, axis=0) / n
        bias -= lr * grad
        bias -= np.mean(bias)
    return bias


def _choose_gamma(margins: np.ndarray, correct: np.ndarray) -> float:
    best_gamma = 3.0
    best_nll = float("inf")
    for g in np.linspace(0.1, 12.0, 240):
        conf = 1.0 / (1.0 + np.exp(-g * margins))
        nll = -np.mean(correct * np.log(np.clip(conf, 1e-9, 1.0)) + (1.0 - correct) * np.log(np.clip(1.0 - conf, 1e-9, 1.0)))
        if nll < best_nll:
            best_nll = float(nll)
            best_gamma = float(g)
    return best_gamma


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("empty dataset")

    y = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
    z_base = np.array([[float(r[f"z_base_{i}"]) for i in range(4)] for r in rows], dtype=np.float64)
    z_drape = np.array([[float(r[f"z_drape_{i}"]) for i in range(4)] for r in rows], dtype=np.float64)
    return y, z_base, z_drape


def _metrics(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    top1 = np.argmax(logits, axis=1)
    top1_acc = float(np.mean(top1 == y_true))
    top2 = np.argsort(logits, axis=1)[:, -2:]
    top2_acc = float(np.mean([y_true[i] in top2[i] for i in range(len(y_true))]))
    margins = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    correct = (top1 == y_true).astype(np.float64)
    return top1_acc, top2_acc, margins, correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="CSV with y_true,z_base_*,z_drape_*")
    parser.add_argument("--output", default=Path("calibration_params.json"), type=Path)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument("--gd-steps", type=int, default=220)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    args = parser.parse_args()

    y_true, z_base, z_drape = _load_csv(args.input)

    alphas = np.arange(0.0, 1.0 + 1e-9, args.alpha_step, dtype=np.float64)
    best = {"loss": float("inf"), "alpha": 0.5, "bias": np.zeros(4, dtype=np.float64)}

    for alpha in alphas:
        z_mix = alpha * z_base + (1.0 - alpha) * z_drape
        bias = _fit_bias_gd(z_mix, y_true, steps=args.gd_steps, lr=args.gd_lr)
        loss = _cross_entropy(z_mix + bias[None, :], y_true)
        if loss < best["loss"]:
            best = {"loss": float(loss), "alpha": float(alpha), "bias": bias.copy()}

    z_cal = best["alpha"] * z_base + (1.0 - best["alpha"]) * z_drape + best["bias"][None, :]
    top1_acc, top2_acc, margins, correct = _metrics(z_cal, y_true)
    gamma = _choose_gamma(margins, correct)

    out = {
        "alpha": float(best["alpha"]),
        "bias": [float(x) for x in best["bias"].tolist()],
        "gamma": float(gamma),
    }
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("best alpha:", out["alpha"])
    print("learned bias:", out["bias"])
    print("chosen gamma:", out["gamma"])
    print("training top1 accuracy:", f"{top1_acc:.4f}")
    print("training top2 accuracy:", f"{top2_acc:.4f}")
    print(
        "margin summary min/median/max:",
        f"{float(np.min(margins)):.4f}",
        f"{float(np.median(margins)):.4f}",
        f"{float(np.max(margins)):.4f}",
    )


if __name__ == "__main__":
    main()
