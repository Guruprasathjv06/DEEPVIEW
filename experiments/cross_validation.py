"""Stratified 5-Fold Cross-Validation runner for DEEPVISION."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import logging
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from configs.mnist_config import MNISTConfig
from datasets.mnist import MNISTManager, MNIST_CLASSES
from models import get_model
from training.device import get_device
from training.trainer import ModelTrainer
from training.reproducibility import set_seed
from evaluation.evaluator import evaluate_model
from evaluation.visualizer import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_sample_predictions,
)
from experiments.wandb_tracker import WandbTracker

logger = logging.getLogger(__name__)


def run_cross_validation(
    config: MNISTConfig,
    dataset_manager: MNISTManager,
) -> Dict[str, Any]:
    """Execute stratified 5-fold cross-validation on the training set.

    Trains a fresh model instance for each fold, tracks metrics,
    saves checkpoints, and computes mean & standard deviation.

    Args:
        config: Experiment configuration dataclass.
        dataset_manager: MNISTManager instance with precomputed stratified folds.

    Returns:
        Dict[str, Any]: Cross-validation results summary across all folds.
    """
    device = get_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)
    plots_dir = os.path.join(config.output_dir, "plots", config.model_name)
    os.makedirs(plots_dir, exist_ok=True)

    cv_group_name = f"{config.dataset_name}_{config.model_name}_5fold_cv"
    logger.info(f"=== Starting Stratified {config.n_splits}-Fold Cross-Validation: {cv_group_name} ===")

    fold_metrics_list: List[Dict[str, Any]] = []

    for fold_idx in range(config.n_splits):
        logger.info(f"\n--- FOLD {fold_idx + 1} / {config.n_splits} ({config.model_name.upper()}) ---")

        # Ensure seed consistency per fold
        set_seed(config.seed + fold_idx)

        # 1. Instantiate fresh model
        model = get_model(config.model_name, in_channels=1, num_classes=10)

        # 2. Instantiate optimizer
        if config.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

        # 3. Instantiate learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )

        # 4. Initialize W&B tracker for this fold
        wandb_tracker = None
        if config.use_wandb:
            run_name = f"{config.model_name}_fold{fold_idx + 1}"
            wandb_tracker = WandbTracker(
                project_name=config.project_name,
                experiment_name=run_name,
                config={**config.to_dict(), "fold_idx": fold_idx + 1},
                group=cv_group_name,
                job_type="cv-fold",
                tags=[*config.wandb_tags, config.model_name, f"fold-{fold_idx + 1}"],
                mode=config.wandb_mode,
            )

        # 5. Get fold DataLoaders (official test set is NOT touched)
        train_loader, val_loader = dataset_manager.get_fold_loaders(
            fold_idx=fold_idx,
            batch_size=config.batch_size,
        )

        # Callback function for trainer
        def epoch_logger(ep: int, step_data: Dict[str, float]):
            if wandb_tracker:
                wandb_tracker.log_epoch(ep, step_data)

        # 6. Train model
        trainer = ModelTrainer(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            scheduler=scheduler,
            checkpoint_dir=config.checkpoint_dir,
            model_name=config.model_name,
            fold_idx=fold_idx + 1,
            epoch_callback=epoch_logger,
        )

        train_results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
        )

        # 7. Load best fold checkpoint for rigorous final fold validation
        best_ckpt_path = os.path.join(
            config.checkpoint_dir, f"{config.model_name}_fold{fold_idx + 1}_best.pt"
        )
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best checkpoint from {best_ckpt_path}")

        # 8. Complete evaluation on validation fold
        val_eval = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=MNIST_CLASSES,
            collect_samples=True,
            max_samples=16,
        )

        # 9. Plot and save training curves and confusion matrix for this fold
        curve_path = os.path.join(plots_dir, f"learning_curves_fold{fold_idx + 1}.png")
        plot_learning_curves(
            trainer.history,
            title_suffix=f"({config.model_name.upper()} - Fold {fold_idx + 1})",
            save_path=curve_path,
        )

        cm_path = os.path.join(plots_dir, f"confusion_matrix_fold{fold_idx + 1}.png")
        plot_confusion_matrix(
            np.array(val_eval["confusion_matrix"]),
            class_names=MNIST_CLASSES,
            title=f"Confusion Matrix ({config.model_name.upper()} - Fold {fold_idx + 1})",
            save_path=cm_path,
        )

        pred_path = os.path.join(plots_dir, f"sample_preds_fold{fold_idx + 1}.png")
        if "samples" in val_eval:
            plot_sample_predictions(
                images=val_eval["samples"]["images"],
                true_labels=val_eval["samples"]["true_labels"],
                pred_labels=val_eval["samples"]["pred_labels"],
                confidences=val_eval["samples"]["confidences"],
                class_names=MNIST_CLASSES,
                n_samples=16,
                title=f"Sample Val Predictions ({config.model_name.upper()} - Fold {fold_idx + 1})",
                save_path=pred_path,
            )

        # 10. Log plots and artifacts to W&B
        if wandb_tracker:
            wandb_tracker.log_confusion_matrix(
                cm=np.array(val_eval["confusion_matrix"]),
                class_names=MNIST_CLASSES,
                title=f"Val Confusion Matrix Fold {fold_idx + 1}",
            )
            if "samples" in val_eval:
                wandb_tracker.log_sample_predictions(
                    samples=val_eval["samples"],
                    class_names=MNIST_CLASSES,
                    title=f"Val Sample Predictions Fold {fold_idx + 1}",
                )
            wandb_tracker.log_summary({
                "val/loss": val_eval["loss"],
                "val/accuracy": val_eval["accuracy"],
                "val/precision_macro": val_eval["precision_macro"],
                "val/recall_macro": val_eval["recall_macro"],
                "val/f1_macro": val_eval["f1_macro"],
                "best_epoch": train_results["best_epoch"],
                "checkpoint_path": best_ckpt_path,
            })
            if config.log_checkpoint_artifacts and os.path.exists(best_ckpt_path):
                wandb_tracker.log_artifact(
                    best_ckpt_path,
                    name=f"{config.model_name}_fold{fold_idx + 1}_best",
                    artifact_type="model",
                )
            wandb_tracker.finish()

        fold_record = {
            "fold": fold_idx + 1,
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_precision_macro": val_eval["precision_macro"],
            "val_recall_macro": val_eval["recall_macro"],
            "val_f1_macro": val_eval["f1_macro"],
            "best_epoch": train_results["best_epoch"],
            "total_time_seconds": train_results["total_time_seconds"],
            "checkpoint_path": best_ckpt_path,
        }
        fold_metrics_list.append(fold_record)
        logger.info(
            f"Fold {fold_idx + 1} Results -> Acc: {val_eval['accuracy']*100:.2f}% | "
            f"Loss: {val_eval['loss']:.4f} | F1: {val_eval['f1_macro']:.4f}"
        )

    # 11. Aggregate metrics across all 5 folds
    accs = [f["val_accuracy"] for f in fold_metrics_list]
    losses = [f["val_loss"] for f in fold_metrics_list]
    precs = [f["val_precision_macro"] for f in fold_metrics_list]
    recs = [f["val_recall_macro"] for f in fold_metrics_list]
    f1s = [f["val_f1_macro"] for f in fold_metrics_list]

    summary = {
        "model_name": config.model_name,
        "n_splits": config.n_splits,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer_name,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "precision_macro_mean": float(np.mean(precs)),
        "precision_macro_std": float(np.std(precs)),
        "recall_macro_mean": float(np.mean(recs)),
        "recall_macro_std": float(np.std(recs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "fold_details": fold_metrics_list,
    }

    # 12. Save results in JSON and CSV format
    json_path = os.path.join(config.output_dir, f"{config.model_name}_cv_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved CV results JSON to {json_path}")

    csv_path = os.path.join(config.output_dir, f"{config.model_name}_cv_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "val_loss",
                "val_accuracy",
                "val_precision_macro",
                "val_recall_macro",
                "val_f1_macro",
                "best_epoch",
                "total_time_seconds",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        writer.writerows(fold_metrics_list)
        # Write mean & std summary row
        writer.writerow({
            "fold": "MEAN",
            "val_loss": round(summary["loss_mean"], 4),
            "val_accuracy": round(summary["accuracy_mean"], 4),
            "val_precision_macro": round(summary["precision_macro_mean"], 4),
            "val_recall_macro": round(summary["recall_macro_mean"], 4),
            "val_f1_macro": round(summary["f1_macro_mean"], 4),
            "best_epoch": "N/A",
            "total_time_seconds": round(sum(f["total_time_seconds"] for f in fold_metrics_list), 2),
            "checkpoint_path": "N/A",
        })
        writer.writerow({
            "fold": "STD",
            "val_loss": round(summary["loss_std"], 4),
            "val_accuracy": round(summary["accuracy_std"], 4),
            "val_precision_macro": round(summary["precision_macro_std"], 4),
            "val_recall_macro": round(summary["recall_macro_std"], 4),
            "val_f1_macro": round(summary["f1_macro_std"], 4),
            "best_epoch": "N/A",
            "total_time_seconds": 0.0,
            "checkpoint_path": "N/A",
        })
    logger.info(f"Saved CV results CSV to {csv_path}")

    logger.info(
        f"\n=== {config.model_name.upper()} 5-FOLD CV SUMMARY ===\n"
        f"Validation Accuracy: {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%\n"
        f"Validation Loss:     {summary['loss_mean']:.4f} ± {summary['loss_std']:.4f}\n"
        f"Validation F1-Score: {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}\n"
        f"=========================================="
    )

    return summary
