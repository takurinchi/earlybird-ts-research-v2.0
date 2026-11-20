"""MNIST training with Early-Bird mask stability detection."""
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from data.mnist import create_loaders
from models.mnist_lenet import create_lenet_bn
from training.builders import build_optimizer, build_scheduler
from training.trainer import train_epoch
from evaluation.classification import evaluate_classifier
from pruning.mask_generator import bn_channel_masks
from pruning.mask_utils import save_masks
from earlybird.stability_detector import StabilityDetector
from utils.config import load_config
from utils.seed import set_seed


ROOT = Path(__file__).resolve().parent


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def append_csv(path, row):
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def run_baseline(config):
    seed = int(config["project"]["seed"])
    epochs = int(config["experiment"]["epochs"])

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    set_seed(seed)

    requested = config["project"].get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    out = ROOT / config["experiment"].get("output_root", "outputs")
    out = out / f"{run_id}_baseline_seed{seed}"
    out.mkdir(parents=True, exist_ok=False)

    (out / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    save_json(out / "environment.json", {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
    })

    data_root = Path(config["data"].get("root", "data/mnist"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    loaders = create_loaders(
        root=str(data_root),
        seed=seed,
        **config["data"].get("kwargs", {}),
    )

    model = create_lenet_bn(num_classes=10).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    criterion = torch.nn.CrossEntropyLoss()

    detector = StabilityDetector(
        threshold=float(config["earlybird"]["epsilon"]),
        mask_window=int(config["earlybird"]["mask_window"]),
    )

    # 枝刈り前のチャネル構造
    channel_counts = {
        name: module.num_features
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    }
    save_json(out / "channel_counts.json", channel_counts)

    training_seconds = 0.0
    detection_seconds = 0.0

    sync(device)
    loop_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # 元コードと同じく、学習開始前の状態を検査する。
        # epoch - 1 は完了済みepoch数。
        sync(device)
        detection_start = time.perf_counter()

        candidate = bn_channel_masks(
            model,
            float(config["pruning"]["selected_ratio"]),
        )
        became_stable = detector.update(epoch - 1, candidate)

        sync(device)
        detection_seconds += time.perf_counter() - detection_start

        # last_distancesは、古い比較先から新しい比較先の順。
        distances = detector.last_distances
        for index, distance in enumerate(distances):
            offset = len(distances) - index

            append_csv(out / "mask_distances.csv", {
                "epoch": epoch - 1,
                "reference_epoch": epoch - 1 - offset,
                "mask_distance": distance,
            })

        if became_stable:
            save_masks(candidate, out / "stable_mask.pt")

            save_json(out / "stable_channels.json", {
                name: {
                    "total": mask.numel(),
                    "kept": int(mask.sum().item()),
                    "indices": (
                        torch.where(mask)[0].cpu().tolist()
                    ),
                }
                for name, mask in candidate.items()
            })

        sync(device)
        train_start = time.perf_counter()

        train_result = train_epoch(
            model, loaders["train"], optimizer, criterion, device
        )

        sync(device)
        epoch_seconds = time.perf_counter() - train_start
        training_seconds += epoch_seconds

        validation = evaluate_classifier(
            model, loaders["validation"], device,
            labels=list(range(10)),
        )

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["train_loss"],
            "train_accuracy": train_result["train_accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_f1_macro": validation["f1_macro"],
            "training_seconds": epoch_seconds,
        }
        append_csv(out / "metrics.csv", row)

        progress = (
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_result['train_loss']:.4f} "
            f"val_acc={validation['accuracy']:.4f} "
            f"tm={detector.tm}"
        )
        print(progress, flush=True)

        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(progress + "\n")

        scheduler.step()

    sync(device)
    loop_seconds = time.perf_counter() - loop_start

    test_result = evaluate_classifier(
        model, loaders["test"], device,
        labels=list(range(10)),
    )

    torch.save(model.state_dict(), out / "final.pt")

    summary = {
        "status": "completed",
        "dataset": "MNIST",
        "model": "MnistLeNetBN",
        "seed": seed,
        "epochs": epochs,
        "tm": detector.tm,
        "tm_detected": detector.detected,
        "training_seconds": training_seconds,
        "mask_detection_seconds": detection_seconds,
        "loop_seconds": loop_seconds,
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "test": test_result,
    }
    save_json(out / "summary.json", summary)

    return summary, out


if __name__ == "__main__":
    config = load_config(ROOT / "configs" / "earlybird.yaml")
    summary, output_dir = run_baseline(config)
    print("保存先:", output_dir)