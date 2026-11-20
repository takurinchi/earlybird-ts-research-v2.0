"""MNIST Early-Bird: Dense control vs train-time compaction."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from mnist_experiment import (
    ROOT,
    _append_csv,
    _make_loaders,
    _prepare_output,
    _sync,
    run_one,
)
from models.mnist_lenet import (
    create_lenet_bn,
    structurally_prune_lenet,
    flatten_channel_indices,
)
from pruning.channel_manager import BNChannelMaskManager
from pruning.mask_generator import bn_channel_masks
from pruning.mask_utils import load_masks, save_masks
from training.builders import build_optimizer, build_scheduler
from training.optimizer_state import (
    migrate_optimizer_state,
    migrate_scheduler_state,
)
from training.trainer import train_epoch
from evaluation.classification import evaluate_classifier
from evaluation.flops import count_dense_flops
from evaluation.inference_time import measure_inference
from evaluation.model_size import model_size_bytes
from evaluation.parameter_count import parameter_count
from evaluation.structured_metrics import (
    estimate_lenet_compact_flops,
    estimate_lenet_compact_parameters,
)
from analysis.plot_compaction_comparison import (
    plot_compaction_comparison,
)
from utils.config import load_config
from utils.seed import set_seed


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, rows):
    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(model, path, device):
    _sync(device)
    start = time.perf_counter()
    torch.save(model.state_dict(), path)
    _sync(device)
    return time.perf_counter() - start


def mask_digest(masks):
    digest = hashlib.sha256()
    for name in sorted(masks):
        digest.update(name.encode("utf-8"))
        digest.update(
            masks[name].detach().cpu()
            .to(torch.uint8).contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def same_masks(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[k].cpu().bool(), right[k].cpu().bool())
        for k in left
    )


def select_device(config):
    name = config["project"].get("device", "auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if (
        device.type != "cuda"
        and config.get("runtime", {}).get("require_cuda", False)
    ):
        raise RuntimeError("This configuration requires CUDA")
    return device


def load_baseline(config, baseline_path, run_id, device):
    seed = int(config["project"]["seed"])
    output_root = ROOT / config["experiment"].get(
        "output_root", "outputs"
    )

    if baseline_path:
        directory = Path(baseline_path)
        if not directory.is_absolute():
            directory = ROOT / directory

        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        source_config = yaml.safe_load(
            (directory / "config.yaml").read_text(encoding="utf-8")
        )

        if (
            summary.get("status") != "completed"
            or not summary.get("tm_detected")
        ):
            raise ValueError("Baseline must contain a detected t_m")
        if summary.get("dataset") != "MNIST":
            raise ValueError("A MNIST baseline is required")
        if int(summary["seed"]) != seed:
            raise ValueError("Baseline seed does not match")
        if (
            source_config["pruning"]["selected_ratio"]
            != config["pruning"]["selected_ratio"]
        ):
            raise ValueError("Baseline pruning ratio does not match")

        mask_path = directory / "masks" / "stable_mask.pt"
        if not mask_path.exists():
            mask_path = directory / "stable_mask.pt"
        masks = load_masks(mask_path)

    else:
        experiment_id = f"{run_id}_baseline_seed{seed}"
        summary, masks = run_one(
            config,
            experiment_id,
            "dense_baseline_with_tm_detection",
            detect=True,
            device=device,
        )
        directory = output_root / experiment_id

        if not summary.get("tm_detected") or masks is None:
            raise RuntimeError("Early-Bird mask was not detected")

    return summary, masks, directory


def run_condition(
    config, experiment_id, strategy, mode, ts, tm,
    baseline_masks, baseline,
):
    """1条件の学習、移行、最終評価を行う。"""
    seed = int(config["project"]["seed"])
    deterministic = bool(
        config.get("reproducibility", {}).get("deterministic", True)
    )
    set_seed(seed, deterministic=deterministic)
    torch.backends.cudnn.benchmark = not deterministic

    device = select_device(config)
    epochs = int(config["experiment"]["epochs"])
    evaluation = config["measurement"]["evaluation"]
    checkpoint_policy = config["measurement"]["checkpoint_policy"]
    labels = list(evaluation["labels"])

    if not 1 <= ts < epochs:
        raise ValueError("t_s must satisfy 1 <= t_s < epochs")

    start = time.perf_counter()
    out = _prepare_output(experiment_id, config)
    log_path = out / "run.log"
    log_path.touch()

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    environment_path = out / "environment.json"
    environment = json.loads(
        environment_path.read_text(encoding="utf-8")
    )
    environment.update({
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "deterministic_requested": deterministic,
        "training_strategy": strategy,
        "information_mode": mode,
    })
    save_json(environment_path, environment)

    model_config = config["experiment"]["model"]
    if model_config.get(
        "factory", "models.mnist_lenet:create_lenet_bn"
    ) != "models.mnist_lenet:create_lenet_bn":
        raise ValueError("Only MnistLeNetBN is supported")

    loaders = _make_loaders(config)
    model = create_lenet_bn(
        **model_config.get("kwargs", {"num_classes": 10})
    ).to(device)

    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    criterion = torch.nn.CrossEntropyLoss()

    times = dict.fromkeys([
        "dense_train_seconds",
        "compact_train_seconds",
        "training_seconds",
        "validation_seconds",
        "mask_generation_seconds",
        "compaction_seconds",
        "structural_metric_seconds",
        "final_compaction_seconds",
        "checkpoint_seconds",
        "evaluation_seconds",
    ], 0.0)

    times["checkpoint_seconds"] += save_checkpoint(
        model, out / "checkpoints" / "initial.pt", device
    )

    masks = None
    gate = None
    transition = {}
    _sync(device)
    loop_start = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):
            if epoch - 1 == ts:
                if checkpoint_policy.get("transition", False):
                    times["checkpoint_seconds"] += save_checkpoint(
                        model,
                        out / "checkpoints" / "transition_before.pt",
                        device,
                    )

                if mode == "online":
                    _sync(device)
                    mask_start = time.perf_counter()
                    masks = bn_channel_masks(
                        model, config["pruning"]["selected_ratio"]
                    )
                    _sync(device)
                    times["mask_generation_seconds"] += (
                        time.perf_counter() - mask_start
                    )
                    source_id = experiment_id
                elif mode == "posthoc":
                    masks = {
                        k: v.detach().clone()
                        for k, v in baseline_masks.items()
                    }
                    source_id = baseline["experiment_id"]
                else:
                    raise ValueError("Unknown information mode")

                if any(not bool(m.any()) for m in masks.values()):
                    raise ValueError("A mask removes all channels in a layer")

                save_masks(
                    masks, out / "masks" / "transition_mask.pt"
                )

                _sync(device)
                metric_start = time.perf_counter()
                shape = tuple(evaluation["input_shape"])

                params_before = sum(
                    p.numel() for p in model.parameters()
                )
                params_after = estimate_lenet_compact_parameters(
                    model, masks
                )
                flops_before = (
                    count_dense_flops(model, shape, device) / shape[0]
                )
                flops_after = (
                    estimate_lenet_compact_flops(model, masks, shape)
                    / shape[0]
                )

                _sync(device)
                times["structural_metric_seconds"] += (
                    time.perf_counter() - metric_start
                )

                if strategy == "dense_late_compaction":
                    gate = BNChannelMaskManager(model, masks)
                    indices = {
                        k: torch.where(v.cpu().bool())[0]
                        for k, v in masks.items()
                    }
                    indices["classifier.input"] = flatten_channel_indices(
                        indices["features.5"],
                        model.feature_hw[0] * model.feature_hw[1],
                    )

                elif strategy == "train_time_compaction":
                    _sync(device)
                    compact_start = time.perf_counter()

                    # モデル生成による余分な乱数消費を復元する。
                    with torch.random.fork_rng(devices=[]):
                        compact, slices, indices = (
                            structurally_prune_lenet(
                                model, masks,
                                return_parameter_slices=True,
                            )
                        )
                    compact = compact.to(device)

                    new_optimizer = build_optimizer(
                        compact, config["training"]
                    )
                    migrate_optimizer_state(
                        optimizer, new_optimizer,
                        model, compact, slices,
                    )

                    new_scheduler = build_scheduler(
                        new_optimizer, config["training"]
                    )
                    migrate_scheduler_state(
                        scheduler, new_scheduler,
                        optimizer, new_optimizer,
                    )

                    model = compact
                    optimizer = new_optimizer
                    scheduler = new_scheduler

                    _sync(device)
                    times["compaction_seconds"] += (
                        time.perf_counter() - compact_start
                    )

                    if sum(
                        p.numel() for p in model.parameters()
                    ) != params_after:
                        raise RuntimeError(
                            "Estimated and actual parameter counts differ"
                        )
                else:
                    raise ValueError("Unknown training strategy")

                save_json(
                    out / "masks" / "channel_indices.json",
                    {k: v.cpu().tolist() for k, v in indices.items()},
                )

                transition = {
                    "mask_source_run_id": source_id,
                    "mask_sha256": mask_digest(masks),
                    "channel_sparsity": 1 - (
                        sum(int(m.sum()) for m in masks.values())
                        / sum(m.numel() for m in masks.values())
                    ),
                    "parameter_count_before_compaction": params_before,
                    "parameter_count_after_compaction": params_after,
                    "parameter_count_actual_after_transition": sum(
                        p.numel() for p in model.parameters()
                    ),
                    "flops_before_compaction": flops_before,
                    "flops_after_compaction": flops_after,
                }
                log(f"transition_after_epoch={ts} {transition}")

            _sync(device)
            train_start = time.perf_counter()
            train = train_epoch(
                model, loaders["train"], optimizer, criterion, device
            )
            _sync(device)
            train_seconds = time.perf_counter() - train_start

            times["training_seconds"] += train_seconds
            phase_time = (
                "dense_train_seconds"
                if epoch <= ts else "compact_train_seconds"
            )
            times[phase_time] += train_seconds

            _sync(device)
            validation_start = time.perf_counter()
            validation = evaluate_classifier(
                model, loaders["validation"], device, labels=labels
            )
            _sync(device)
            validation_seconds = (
                time.perf_counter() - validation_start
            )
            times["validation_seconds"] += validation_seconds

            phase = (
                "compact"
                if strategy == "train_time_compaction" and epoch > ts
                else "dense"
            )
            row = {
                "experiment_id": experiment_id,
                "training_strategy": strategy,
                "information_mode": mode,
                "seed": seed,
                "epoch": epoch,
                "tm": tm,
                "ts": ts,
                "model_phase": phase,
                "transition_after_epoch": ts,
                "first_compact_epoch": (
                    ts + 1 if strategy == "train_time_compaction"
                    else None
                ),
                "transition_applied": int(epoch == ts + 1),
                "train_loss": train["train_loss"],
                "train_accuracy": train["train_accuracy"],
                "validation_loss": validation["loss"],
                "accuracy": validation["accuracy"],
                "f1_macro": validation["f1_macro"],
                "training_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "parameter_count": sum(
                    p.numel() for p in model.parameters()
                ),
                "channel_sparsity": transition.get(
                    "channel_sparsity", 0.0
                ),
            }
            _append_csv(out / "metrics.csv", row)
            log(
                f"epoch={epoch}/{epochs} phase={phase} "
                f"train_loss={train['train_loss']:.6f} "
                f"val_accuracy={validation['accuracy']:.6f} "
                f"train_s={train_seconds:.4f}"
            )
            scheduler.step()

        _sync(device)
        wall_seconds = time.perf_counter() - loop_start

        if masks is None:
            raise RuntimeError("The transition was not reached")

        # 対照群も、最終評価前には同じマスクで縮小する。
        if strategy == "dense_late_compaction":
            _sync(device)
            compact_start = time.perf_counter()
            with torch.random.fork_rng(devices=[]):
                final_model = structurally_prune_lenet(model, masks)
            final_model = final_model.to(device).eval()
            _sync(device)
            times["final_compaction_seconds"] = (
                time.perf_counter() - compact_start
            )
            gate.close()
        else:
            final_model = model.eval()

        _sync(device)
        evaluation_start = time.perf_counter()

        test = evaluate_classifier(
            final_model, loaders["test"], device, labels=labels
        )
        batch_size = int(evaluation["inference_batch_size"])
        sample = next(iter(loaders["test"]))[0][:batch_size].to(device)

        if list(sample.shape) != list(evaluation["input_shape"]):
            raise ValueError("Inference input shape does not match")

        inference = measure_inference(
            final_model, sample, device,
            warmup=int(evaluation["warmup"]),
            repeats=int(evaluation["repeats"]),
        )
        flops = (
            count_dense_flops(final_model, tuple(sample.shape), device)
            / batch_size
        )
        counts = parameter_count(final_model)

        _sync(device)
        times["evaluation_seconds"] = (
            time.perf_counter() - evaluation_start
        )

        checkpoint_path = out / "checkpoints" / "final.pt"
        times["checkpoint_seconds"] += save_checkpoint(
            final_model, checkpoint_path, device
        )

        summary = {
            "experiment_id": experiment_id,
            "status": "completed",
            "stage": "ts_compaction_comparison",
            "dataset": "MNIST",
            "model": "MNIST LeNet-BN",
            "seed": seed,
            "device": str(device),
            "tm": int(tm),
            "ts": int(ts),
            "transition_after_epoch": ts,
            "first_compact_epoch": (
                ts + 1 if strategy == "train_time_compaction"
                else None
            ),
            "training_strategy": strategy,
            "information_mode": mode,
            **transition,
            **test,
            **times,
            "parameter_count": counts["parameter_count"],
            "zero_parameter_count": counts["zero_parameter_count"],
            "parameter_sparsity": counts["sparsity"],
            "flops_per_sample": flops,
            "model_size_bytes": model_size_bytes(checkpoint_path),
            "wall_seconds": wall_seconds,
            "inference_seconds_per_sample": (
                inference["mean_seconds"] / batch_size
            ),
            "inference_repeats": inference["repeats"],
            "inference_batch_size": batch_size,
            "class_labels": labels,
            "checkpoint_format": evaluation["checkpoint_format"],
            "checkpoint": "checkpoints/final.pt",
            "transition_checkpoint_saved": bool(
                checkpoint_policy.get("transition", False)
            ),
        }

        for metric, output_name in (
            ("parameter_count", "parameter_count_reduction_percent"),
            ("flops_per_sample", "flops_per_sample_reduction_percent"),
            ("model_size_bytes", "model_size_bytes_reduction_percent"),
        ):
            base = baseline.get(metric)
            summary[output_name] = (
                100 * (1 - summary[metric] / base)
                if base else None
            )

        summary["end_to_end_seconds"] = time.perf_counter() - start
        save_json(out / "summary.json", summary)
        return summary, out

    finally:
        if gate is not None:
            gate.close()


def run_condition_safe(
    config, experiment_id, strategy, mode, ts, tm,
    baseline_masks, baseline,
):
    """失敗内容を保存し、次の比較条件へ進める。"""
    try:
        return run_condition(
            config, experiment_id, strategy, mode, ts, tm,
            baseline_masks, baseline,
        )
    except Exception as exc:
        out = (
            ROOT
            / config["experiment"].get("output_root", "outputs")
            / experiment_id
        )
        if not out.exists():
            out = _prepare_output(experiment_id, config)

        metrics_path = out / "metrics.csv"
        completed_epochs = 0
        if metrics_path.exists() and metrics_path.stat().st_size:
            with metrics_path.open(newline="", encoding="utf-8") as f:
                completed_epochs = sum(1 for _ in csv.DictReader(f))

        summary = {
            "experiment_id": experiment_id,
            "status": "failed",
            "stage": "ts_compaction_comparison",
            "training_strategy": strategy,
            "information_mode": mode,
            "seed": int(config["project"]["seed"]),
            "tm": tm,
            "ts": ts,
            "mask_source_run_id": baseline["experiment_id"],
            "completed_epochs": completed_epochs,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        for key in (
            "accuracy", "f1_macro", "parameter_count",
            "flops_per_sample", "model_size_bytes",
            "dense_train_seconds", "compact_train_seconds",
            "training_seconds", "validation_seconds",
            "compaction_seconds", "checkpoint_seconds",
            "wall_seconds", "end_to_end_seconds",
        ):
            summary[key] = None

        save_json(out / "summary.json", summary)
        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(f"status=failed error={exc}\n")

        return summary, out


def write_bundle(
    config, run_id, baseline, baseline_dir,
    conditions, summaries, run_dirs, pairs,
):
    """設定・結果・マスク・集計表をまとめる。"""
    bundle = (
        ROOT
        / config["experiment"].get("output_root", "outputs")
        / "results" / "mnist_compaction" / run_id
    )
    bundle.mkdir(parents=True, exist_ok=False)

    def copy_run(source, destination, include_trace=False):
        destination.mkdir(parents=True)
        names = [
            "config.yaml", "environment.json",
            "run.log", "metrics.csv", "summary.json",
        ]
        if include_trace:
            names.append("mask_distances.csv")

        for name in names:
            if (source / name).is_file():
                shutil.copy2(source / name, destination / name)

        for folder in ("masks", "plots"):
            if (source / folder).is_dir():
                shutil.copytree(source / folder, destination / folder)

        if (source / "stable_mask.pt").is_file():
            (destination / "masks").mkdir(exist_ok=True)
            shutil.copy2(
                source / "stable_mask.pt",
                destination / "masks" / "stable_mask.pt",
            )

    epoch_rows = []
    for experiment_id, directory in run_dirs.items():
        copy_run(directory, bundle / "runs" / experiment_id)

        path = directory / "metrics.csv"
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as f:
                epoch_rows.extend(
                    dict(row, source_experiment=experiment_id)
                    for row in csv.DictReader(f)
                )

    copy_run(baseline_dir, bundle / "baseline", include_trace=True)
    write_csv(bundle / "summary.csv", summaries)
    write_csv(bundle / "compaction_comparison.csv", pairs)
    if epoch_rows:
        write_csv(bundle / "metrics_all_epochs.csv", epoch_rows)

    save_json(bundle / "manifest.json", {
        "study": "MNIST Early-Bird train-time compaction",
        "run_id": run_id,
        "baseline_experiment_id": baseline["experiment_id"],
        "tm": baseline["tm"],
        "strategies": config["comparison"]["strategies"],
        "information_modes": config["comparison"]["information_modes"],
        "conditions": conditions,
        "run_ids": [s["experiment_id"] for s in summaries],
        "failed_or_incomplete_runs": [
            s["experiment_id"] for s in summaries
            if s["status"] != "completed"
        ],
        "checkpoint_policy": config["measurement"]["checkpoint_policy"],
        "large_checkpoints_copied": False,
    })

    (bundle / "README.md").write_text(
        "# MNIST compaction comparison\n\n"
        f"Baseline: {baseline['experiment_id']}, t_m={baseline['tm']}.\n"
        "summary.csv: per-run results.\n"
        "compaction_comparison.csv: paired results.\n"
        "metrics_all_epochs.csv: epoch metrics.\n"
        "Large model checkpoints remain in original run directories.\n",
        encoding="utf-8",
    )

    plot_compaction_comparison(
        bundle / "summary.csv", bundle / "plots"
    )
    return bundle


def run_compaction_suite(config, resume_baseline=None):
    measurement = config["measurement"]
    evaluation = measurement["evaluation"]
    policy = measurement["checkpoint_policy"]
    strategies = list(config["comparison"]["strategies"])
    modes = list(config["comparison"]["information_modes"])

    if (
        int(measurement["validation_frequency_epochs"]) != 1
        or int(measurement["epoch_log_frequency_epochs"]) != 1
    ):
        raise ValueError("Validation and logging must run every epoch")
    if not policy.get("initial") or not policy.get("final"):
        raise ValueError("Initial and final checkpoints are required")
    if evaluation["checkpoint_format"] != "torch_state_dict":
        raise ValueError("Only torch_state_dict is supported")
    if (
        len(evaluation["input_shape"]) != 4
        or evaluation["input_shape"][0]
        != evaluation["inference_batch_size"]
    ):
        raise ValueError("Invalid inference input shape")
    if set(strategies) != {
        "dense_late_compaction", "train_time_compaction"
    }:
        raise ValueError("Both training strategies are required")
    if set(modes) != {"online", "posthoc"}:
        raise ValueError("Both information modes are required")

    device = select_device(config)
    seed = int(config["project"]["seed"])
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")

    baseline, stable_masks, baseline_dir = load_baseline(
        config, resume_baseline, run_id, device
    )
    tm = int(baseline["tm"])
    relative = config["ts"]["relative_to_tm"]

    conditions = {
        "C1_before": max(1, tm - int(relative["before_offset"])),
        "C2_tm": max(1, tm + int(relative["near_offset"])),
        "C3_after": max(1, tm + int(relative["after_offset"])),
        "C4_fixed_ts": int(config["ts"]["fixed_epoch"]),
    }

    epochs = int(config["experiment"]["epochs"])
    if any(not 1 <= ts < epochs for ts in conditions.values()):
        raise ValueError(
            f"All transition times must satisfy 1 <= ts < {epochs}: "
            f"{conditions}"
        )

    summaries, run_dirs, pairs = [], {}, []
    pair_number = 0

    for mode in modes:
        for condition, ts in conditions.items():
            pair_number += 1
            order = (
                strategies if pair_number % 2
                else list(reversed(strategies))
            )
            pair = {}

            for strategy in order:
                experiment_id = (
                    f"{run_id}_{condition}_{mode}_{strategy}_seed{seed}"
                )
                summary, directory = run_condition_safe(
                    config, experiment_id, strategy, mode, ts, tm,
                    stable_masks, baseline,
                )
                pair[strategy] = summary
                summaries.append(summary)
                run_dirs[experiment_id] = directory

            control = pair["dense_late_compaction"]
            treatment = pair["train_time_compaction"]
            completed = all(
                s["status"] == "completed"
                for s in (control, treatment)
            )

            match = None
            if completed:
                match = same_masks(
                    load_masks(
                        run_dirs[control["experiment_id"]]
                        / "masks" / "transition_mask.pt"
                    ),
                    load_masks(
                        run_dirs[treatment["experiment_id"]]
                        / "masks" / "transition_mask.pt"
                    ),
                )

            # 同じマスクのペアについて速度比を計算する。
            speedup = (
                control["compact_train_seconds"]
                / treatment["compact_train_seconds"]
                if completed and match
                and treatment["compact_train_seconds"] else None
            )
            total_speedup = (
                control["end_to_end_seconds"]
                / treatment["end_to_end_seconds"]
                if completed and match
                and treatment["end_to_end_seconds"] else None
            )

            for arm, other in (
                (control, treatment), (treatment, control)
            ):
                arm.update({
                    "paired_mask_match": match,
                    "paired_experiment_id": other["experiment_id"],
                    "speedup_after_ts": speedup,
                    "total_speedup": total_speedup,
                })
                directory = run_dirs[arm["experiment_id"]]
                save_json(directory / "summary.json", arm)
                with (directory / "run.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(f"paired_mask_match={match}\n")

            pairs.append({
                "condition": condition,
                "information_mode": mode,
                "ts": ts,
                "status": (
                    "completed" if completed and match
                    else "incomplete_or_mask_mismatch"
                ),
                "paired_mask_match": match,
                "dense_control_id": control["experiment_id"],
                "train_time_compaction_id": treatment["experiment_id"],
                "dense_post_ts_train_seconds": control.get(
                    "compact_train_seconds"
                ),
                "compact_post_ts_train_seconds": treatment.get(
                    "compact_train_seconds"
                ),
                "compaction_seconds": treatment.get("compaction_seconds"),
                "dense_end_to_end_seconds": control.get(
                    "end_to_end_seconds"
                ),
                "compact_end_to_end_seconds": treatment.get(
                    "end_to_end_seconds"
                ),
                "speedup_after_ts": speedup,
                "total_speedup": total_speedup,
                "dense_test_accuracy": control.get("accuracy"),
                "compact_test_accuracy": treatment.get("accuracy"),
                "dense_error": control.get("error"),
                "compact_error": treatment.get("error"),
            })

    output_root = ROOT / config["experiment"].get(
        "output_root", "outputs"
    )
    comparison_path = (
        output_root / f"ts_compaction_comparison_{run_id}.csv"
    )
    write_csv(comparison_path, summaries)

    result = {
        "run_id": run_id,
        "baseline": baseline,
        "mask_source_run_id": baseline["experiment_id"],
        "conditions": summaries,
        "pairs": pairs,
        "comparison_csv": str(comparison_path),
    }
    save_json(output_root / f"run_summary_{run_id}.json", result)

    result["bundle"] = str(write_bundle(
        config, run_id, baseline, baseline_dir,
        conditions, summaries, run_dirs, pairs,
    ))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/ts_compaction.yaml"
    )
    parser.add_argument("--resume-baseline")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    result = run_compaction_suite(
        load_config(config_path),
        resume_baseline=args.resume_baseline,
    )
    print("Result bundle:", result["bundle"])