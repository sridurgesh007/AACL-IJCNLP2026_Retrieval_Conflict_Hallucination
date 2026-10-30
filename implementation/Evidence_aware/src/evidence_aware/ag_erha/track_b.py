from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evidence_aware.ag_erha.gating import compute_margin_from_logits_vector
from evidence_aware.ag_erha.utils import read_jsonl, roc_auc_binary, write_json, write_jsonl
from evidence_aware.phase2.common import Example


def _cohens_d(f_vals: np.ndarray, h_vals: np.ndarray) -> float:
    mean_diff = float(np.mean(f_vals) - np.mean(h_vals))
    pooled_std = float(np.sqrt((np.std(f_vals) ** 2 + np.std(h_vals) ** 2) / 2.0))
    return float(mean_diff / (pooled_std + 1e-10))


def build_evidence_set(tokenizer, evidence_answer_str: str) -> set[int]:
    variants = [
        evidence_answer_str,
        " " + evidence_answer_str,
        evidence_answer_str.capitalize(),
        " " + evidence_answer_str.capitalize(),
        evidence_answer_str.lower(),
        " " + evidence_answer_str.lower(),
    ]
    token_ids: set[int] = set()
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(int(ids[0]))
    return token_ids


def build_parametric_set(baseline_top_tokens: list[int], evidence_set: set[int], k: int = 5) -> set[int]:
    picked: list[int] = []
    for token_id in baseline_top_tokens:
        tid = int(token_id)
        if tid in evidence_set:
            continue
        picked.append(tid)
        if len(picked) == k:
            break
    return set(picked)


def _margin_from_selected_log_probs(
    log_probs_selected: list[float],
    selected_token_ids: list[int],
    evidence_set: set[int],
    parametric_set: set[int],
) -> float:
    idx_map = {int(tid): i for i, tid in enumerate(selected_token_ids)}
    e_idx = [idx_map[t] for t in evidence_set if t in idx_map]
    p_idx = [idx_map[t] for t in parametric_set if t in idx_map]
    if not e_idx or not p_idx:
        raise RuntimeError("Selected-token log-prob row does not contain full evidence/parametric token sets.")
    lp = np.array(log_probs_selected, dtype=np.float64)
    p_e = float(np.exp(lp[e_idx]).sum())
    p_p = float(np.exp(lp[p_idx]).sum())
    p_e = max(p_e, 1e-10)
    p_p = max(p_p, 1e-10)
    return float(np.log(p_e) - np.log(p_p))


def compute_margin_trajectory(trace: dict[str, Any], evidence_set: set[int], parametric_set: set[int]) -> dict[str, Any]:
    # Full-logits path.
    required_full = ["logits_pre_attn", "logits_post_attn", "logits_post_mlp"]
    if all(k in trace for k in required_full):
        logits_pre = trace["logits_pre_attn"]
        logits_attn = trace["logits_post_attn"]
        logits_mlp = trace["logits_post_mlp"]
        n_layers = min(len(logits_pre), len(logits_attn), len(logits_mlp))
        if n_layers <= 0:
            raise RuntimeError(f"Expected at least 1 layer in logits traces, got {n_layers}")

        m_pre, m_attn, m_mlp = [], [], []
        for layer_idx in range(n_layers):
            m_pre.append(compute_margin_from_logits_vector(logits_pre[layer_idx], evidence_set, parametric_set))
            m_attn.append(compute_margin_from_logits_vector(logits_attn[layer_idx], evidence_set, parametric_set))
            m_mlp.append(compute_margin_from_logits_vector(logits_mlp[layer_idx], evidence_set, parametric_set))
    else:
        # Sparse-logprob path.
        required_sparse = [
            "margin_token_ids",
            "log_probs_pre_attn_selected",
            "log_probs_post_attn_selected",
            "log_probs_post_mlp_selected",
        ]
        if not all(k in trace for k in required_sparse):
            raise RuntimeError(
                "Track B requires either full logits fields "
                "(logits_pre_attn/logits_post_attn/logits_post_mlp) or sparse selected-logprob fields "
                "(margin_token_ids/log_probs_*_selected)."
            )
        token_ids = [int(x) for x in trace["margin_token_ids"]]
        lp_pre = trace["log_probs_pre_attn_selected"]
        lp_attn = trace["log_probs_post_attn_selected"]
        lp_mlp = trace["log_probs_post_mlp_selected"]
        n_layers = min(len(lp_pre), len(lp_attn), len(lp_mlp))
        if n_layers <= 0:
            raise RuntimeError(f"Expected at least 1 layer in sparse log-prob traces, got {n_layers}")

        m_pre, m_attn, m_mlp = [], [], []
        for layer_idx in range(n_layers):
            m_pre.append(_margin_from_selected_log_probs(lp_pre[layer_idx], token_ids, evidence_set, parametric_set))
            m_attn.append(_margin_from_selected_log_probs(lp_attn[layer_idx], token_ids, evidence_set, parametric_set))
            m_mlp.append(_margin_from_selected_log_probs(lp_mlp[layer_idx], token_ids, evidence_set, parametric_set))

    delta_attn = [m_attn[i] - m_pre[i] for i in range(n_layers)]
    delta_mlp = [m_mlp[i] - m_attn[i] for i in range(n_layers)]
    return {
        "n_layers": int(n_layers),
        "M_pre_attn": m_pre,
        "M_post_attn": m_attn,
        "M_post_mlp": m_mlp,
        "delta_attn": delta_attn,
        "delta_mlp": delta_mlp,
    }


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([r[key] for r in rows], dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def _plot_track_b(
    out_dir: Path,
    faithful: list[dict[str, Any]],
    hallucinated: list[dict[str, Any]],
    auc_scalar: float,
    auc_margin_l27_post_mlp: float,
    focus_layer: int = 27,
    n_layers: int = 32,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    layers = np.arange(int(n_layers))

    f_mean, f_std = _mean_std(faithful, "M_post_mlp")
    h_mean, h_std = _mean_std(hallucinated, "M_post_mlp")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(layers, f_mean, color="blue", linewidth=2.0, label="faithful")
    ax.fill_between(layers, f_mean - f_std, f_mean + f_std, color="blue", alpha=0.18)
    ax.plot(layers, h_mean, color="red", linewidth=2.0, label="hallucinated")
    ax.fill_between(layers, h_mean - h_std, h_mean + h_std, color="red", alpha=0.18)
    ax.axvline(int(focus_layer), linestyle="--", color="black", linewidth=1.2)
    ax.set_title("Mean evidence-set margin by layer (post-MLP)")
    ax.set_xlabel("layer")
    ax.set_ylabel("margin M")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "margin_trajectory_by_layer.png", dpi=220)
    plt.close(fig)

    f_mlp, f_mlp_std = _mean_std(faithful, "delta_mlp")
    h_mlp, h_mlp_std = _mean_std(hallucinated, "delta_mlp")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(layers, f_mlp, color="blue", linewidth=2.0, label="faithful")
    ax.fill_between(layers, f_mlp - f_mlp_std, f_mlp + f_mlp_std, color="blue", alpha=0.18)
    ax.plot(layers, h_mlp, color="red", linewidth=2.0, label="hallucinated")
    ax.fill_between(layers, h_mlp - h_mlp_std, h_mlp + h_mlp_std, color="red", alpha=0.18)
    ax.axvline(int(focus_layer), linestyle="--", color="black", linewidth=1.2)
    ax.set_title("MLP delta of evidence-set margin by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("delta_mlp")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_mlp_margin_by_layer.png", dpi=220)
    plt.close(fig)

    f_attn, f_attn_std = _mean_std(faithful, "delta_attn")
    h_attn, h_attn_std = _mean_std(hallucinated, "delta_attn")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(layers, f_attn, color="blue", linewidth=2.0, label="faithful")
    ax.fill_between(layers, f_attn - f_attn_std, f_attn + f_attn_std, color="blue", alpha=0.18)
    ax.plot(layers, h_attn, color="red", linewidth=2.0, label="hallucinated")
    ax.fill_between(layers, h_attn - h_attn_std, h_attn + h_attn_std, color="red", alpha=0.18)
    ax.axvline(int(focus_layer), linestyle="--", color="black", linewidth=1.2)
    ax.set_title("Attention delta of evidence-set margin by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("delta_attn")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_attn_margin_by_layer.png", dpi=220)
    plt.close(fig)

    f_pre = np.array([x["M_post_attn"][focus_layer] for x in faithful], dtype=np.float64)
    f_post = np.array([x["M_post_mlp"][focus_layer] for x in faithful], dtype=np.float64)
    h_pre = np.array([x["M_post_attn"][focus_layer] for x in hallucinated], dtype=np.float64)
    h_post = np.array([x["M_post_mlp"][focus_layer] for x in hallucinated], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.hist(f_pre, bins=35, alpha=0.35, label="faithful pre-MLP")
    ax.hist(f_post, bins=35, alpha=0.35, label="faithful post-MLP")
    ax.hist(h_pre, bins=35, alpha=0.35, label="hallucinated pre-MLP")
    ax.hist(h_post, bins=35, alpha=0.35, label="hallucinated post-MLP")
    ax.set_title(f"Evidence-set margin at layer {focus_layer} - pre vs post MLP")
    ax.set_xlabel("margin")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"margin_at_layer{focus_layer}_distribution.png", dpi=220)
    if int(focus_layer) != 27:
        fig.savefig(plots_dir / "margin_at_mistral_top_head_layer.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.bar([0, 1], [auc_scalar, auc_margin_l27_post_mlp], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"scalar p_evidence@L{focus_layer}", f"margin M_post_mlp@L{focus_layer}"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Single-token scalar vs candidate-set margin - separation at layer {focus_layer}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "separation_scalar_vs_margin.png", dpi=220)
    plt.close(fig)


def run_track_b(
    tokenizer,
    examples_by_id: dict[int, Example],
    traces_path: Path,
    out_dir: Path,
    focus_layer: int = 27,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = read_jsonl(traces_path)

    processed: list[dict[str, Any]] = []
    faithful, hallucinated = [], []
    skipped = {
        "missing_example_row": 0,
        "missing_label": 0,
        "no_single_token_evidence": 0,
        "no_parametric_set": 0,
        "missing_baseline_top_tokens": 0,
        "missing_margin_payload": 0,
    }

    for trace in traces:
        example_id = int(trace["example_id"])
        ex = examples_by_id.get(example_id)
        if ex is None:
            skipped["missing_example_row"] += 1
            continue

        if "is_hallucinated" in trace:
            is_hall = bool(trace["is_hallucinated"])
        elif "label" in trace:
            is_hall = str(trace["label"]).strip().lower() == "hallucinated"
        else:
            skipped["missing_label"] += 1
            continue

        evidence_set = set(int(x) for x in trace.get("evidence_set_token_ids", []))
        if not evidence_set:
            evidence_set = build_evidence_set(tokenizer, ex.answer_liquid)
        if len(evidence_set) == 0:
            skipped["no_single_token_evidence"] += 1
            continue

        parametric_set = set(int(x) for x in trace.get("parametric_set_token_ids", []))
        if not parametric_set:
            baseline_top_tokens = trace.get("baseline_top_tokens", trace.get("top_tokens_final_layer"))
            if baseline_top_tokens is None:
                skipped["missing_baseline_top_tokens"] += 1
                continue
            parametric_set = build_parametric_set([int(t) for t in baseline_top_tokens], evidence_set, k=5)

        parametric_set = {t for t in parametric_set if t not in evidence_set}
        if len(parametric_set) == 0:
            skipped["no_parametric_set"] += 1
            continue

        try:
            margins = compute_margin_trajectory(trace, evidence_set, parametric_set)
        except RuntimeError as err:
            if "requires either full logits fields" in str(err):
                skipped["missing_margin_payload"] += 1
                continue
            raise

        row = {
            "example_id": example_id,
            "is_hallucinated": is_hall,
            "label": ("hallucinated" if is_hall else "faithful"),
            "evidence_set_token_ids": sorted(int(x) for x in evidence_set),
            "parametric_set_token_ids": sorted(int(x) for x in parametric_set),
            "evidence_set_size": len(evidence_set),
            "parametric_set_size": len(parametric_set),
            **margins,
        }
        processed.append(row)
        if row["is_hallucinated"]:
            hallucinated.append(row)
        else:
            faithful.append(row)

    if not faithful or not hallucinated:
        raise RuntimeError(
            "Track B needs both faithful and hallucinated rows. "
            f"faithful={len(faithful)}, hallucinated={len(hallucinated)}, skipped={skipped}"
        )

    n_layers = min(
        min(len(r["M_pre_attn"]), len(r["M_post_attn"]), len(r["M_post_mlp"]))
        for r in processed
    )
    if n_layers <= 0:
        raise RuntimeError("Track B computed empty margin trajectories.")
    if int(focus_layer) < 0 or int(focus_layer) >= int(n_layers):
        raise RuntimeError(
            f"focus_layer={focus_layer} out of range for n_layers={n_layers}. "
            f"Use focus_layer in [0, {n_layers - 1}]."
        )

    write_jsonl(out_dir / "margin_trajectories.jsonl", processed)

    separation_rows = []
    late_start = max(0, int(n_layers) - 8)
    for layer in range(late_start, int(n_layers)):
        for point in ["M_pre_attn", "M_post_attn", "M_post_mlp", "delta_attn", "delta_mlp"]:
            f_vals = np.array([r[point][layer] for r in faithful], dtype=np.float64)
            h_vals = np.array([r[point][layer] for r in hallucinated], dtype=np.float64)
            labels = [1] * len(f_vals) + [0] * len(h_vals)
            scores = f_vals.tolist() + h_vals.tolist()
            auc = roc_auc_binary(labels, scores)
            cohens_d = _cohens_d(f_vals, h_vals)
            separation_rows.append(
                {
                    "layer": layer,
                    "point": point,
                    "cohens_d": cohens_d,
                    "auc": auc,
                    "mean_faithful": float(np.mean(f_vals)),
                    "mean_hallucinated": float(np.mean(h_vals)),
                }
            )
    write_json(out_dir / "separation_metrics.json", separation_rows)

    scalar_scores = []
    scalar_labels = []
    for t in traces:
        if "p_evidence_post_attn" not in t:
            raise RuntimeError("Track B needs p_evidence_post_attn in Phase1 traces for scalar AUC comparison.")
        if "is_hallucinated" in t:
            is_hall = bool(t["is_hallucinated"])
        elif "label" in t:
            is_hall = str(t["label"]).strip().lower() == "hallucinated"
        else:
            continue
        scores = t.get("p_evidence_post_attn", [])
        if len(scores) <= int(focus_layer):
            continue
        scalar_scores.append(float(scores[int(focus_layer)]))
        scalar_labels.append(0 if is_hall else 1)

    if not scalar_scores or not scalar_labels:
        raise RuntimeError(
            "Track B scalar AUC calculation has no valid rows. "
            f"focus_layer={focus_layer}, traces={len(traces)}"
        )

    auc_scalar = roc_auc_binary(scalar_labels, scalar_scores)
    auc_margin_focus_post_attn = roc_auc_binary(
        [1] * len(faithful) + [0] * len(hallucinated),
        [x["M_post_attn"][focus_layer] for x in faithful] + [x["M_post_attn"][focus_layer] for x in hallucinated],
    )
    auc_margin_focus_post_mlp = roc_auc_binary(
        [1] * len(faithful) + [0] * len(hallucinated),
        [x["M_post_mlp"][focus_layer] for x in faithful] + [x["M_post_mlp"][focus_layer] for x in hallucinated],
    )
    mlp_erodes = float(np.mean([x["delta_mlp"][focus_layer] for x in hallucinated])) < float(
        np.mean([x["delta_mlp"][focus_layer] for x in faithful])
    )

    if auc_margin_focus_post_attn > auc_scalar + 0.05:
        verdict = "MARGIN_BETTER"
    elif abs(auc_margin_focus_post_attn - auc_scalar) <= 0.05:
        verdict = "MARGIN_COMPARABLE"
    else:
        verdict = "SCALAR_BETTER"

    cohens_d_focus = _cohens_d(
        np.array([x["M_post_attn"][focus_layer] for x in faithful]),
        np.array([x["M_post_attn"][focus_layer] for x in hallucinated]),
    )
    verdict_lines = [
        f"AUC_scalar_L{focus_layer}_post_attn: {auc_scalar}",
        f"AUC_margin_L{focus_layer}_post_attn: {auc_margin_focus_post_attn}",
        f"AUC_margin_L{focus_layer}_post_mlp: {auc_margin_focus_post_mlp}",
        f"COHENS_D_margin_L{focus_layer}: {cohens_d_focus}",
        f"MLP_ERODES_MARGIN_IN_HALLUCINATED: {'yes' if mlp_erodes else 'no'}",
        "",
        f"VERDICT: {verdict}",
    ]
    (out_dir / "track_b_verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    _plot_track_b(
        out_dir=out_dir,
        faithful=faithful,
        hallucinated=hallucinated,
        auc_scalar=auc_scalar,
        auc_margin_l27_post_mlp=auc_margin_focus_post_mlp,
        focus_layer=int(focus_layer),
        n_layers=int(n_layers),
    )

    return {
        "num_faithful": len(faithful),
        "num_hallucinated": len(hallucinated),
        "num_processed": len(processed),
        "n_layers": int(n_layers),
        "skipped": skipped,
        "focus_layer": int(focus_layer),
        "auc_scalar_post_attn": auc_scalar,
        "auc_margin_post_attn": auc_margin_focus_post_attn,
        "auc_margin_post_mlp": auc_margin_focus_post_mlp,
        f"auc_scalar_L{focus_layer}_post_attn": auc_scalar,
        f"auc_margin_L{focus_layer}_post_attn": auc_margin_focus_post_attn,
        f"auc_margin_L{focus_layer}_post_mlp": auc_margin_focus_post_mlp,
        f"cohens_d_margin_L{focus_layer}": cohens_d_focus,
        "mlp_erodes_margin_in_hallucinated": bool(mlp_erodes),
        "verdict": verdict,
    }
