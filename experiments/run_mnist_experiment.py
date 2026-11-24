"""Main experiment orchestrator for MNIST under DEEPVISION."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import argparse
import logging
from typing import Dict, Any
import numpy as np
import torch
from configs.mnist_config import MNISTConfig
from datasets.mnist import MNISTManager, MNIST_CLASSES
from models import get_model
from training.device import get_device, get_device_info
from training.reproducibility import set_seed
from evaluation.evaluator import evaluate_model
from evaluation.visualizer import plot_confusion_matrix, plot_sample_predictions
from experiments.cross_validation import run_cross_validation
from experiments.wandb_tracker import WandbTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DEEPVISION.MNIST")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment execution."""
    parser = argparse.ArgumentParser(description="DeepVision MNIST Experiment Pipeline")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per fold (use 1-2 for verification, 5+ for full training)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--models", nargs="+", default=["basic", "improved"], help="Models to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    return parser.parse_args()


def run_full_mnist_pipeline(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    device_pref: str = "auto",
    wandb_mode: str = "online",
    models_to_run: list = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete MNIST pipeline: 5-Fold CV on Basic CNN & Improved CNN + Test Set Evaluation.

    Returns:
        Dict[str, Any]: Full experiment results dictionary.
    """
    if models_to_run is None:
        models_to_run = ["basic", "improved"]

    # 1. Device detection
    device = get_device(device_pref)
    dev_info = get_device_info(device)
    logger.info(f"Hardware setup: {dev_info}")

    set_seed(seed)

    # 2. Dataset initialization (downloads if needed, creates stratified folds)
    dataset_manager = MNISTManager(data_dir="data", n_splits=5, seed=seed)
    logger.info("MNIST dataset ready with 5 stratified folds on training set.")

    cv_results = {}
    best_overall_checkpoint = None
    best_overall_val_acc = -1.0
    best_overall_model_name = ""

    # 3. Run Stratified 5-Fold Cross-Validation for each requested model
    for model_name in models_to_run:
        config = MNISTConfig(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            device=device_pref,
            wandb_mode=wandb_mode,
            seed=seed,
            n_splits=5,
        )

        res = run_cross_validation(config=config, dataset_manager=dataset_manager)
        cv_results[model_name] = res

        # Check if this model produced the highest validation fold performance
        for fold_info in res["fold_details"]:
            if fold_info["val_accuracy"] > best_overall_val_acc:
                best_overall_val_acc = fold_info["val_accuracy"]
                best_overall_checkpoint = fold_info["checkpoint_path"]
                best_overall_model_name = model_name

    # 4. Final Evaluation on Untouched Official Test Set (10,000 samples)
    logger.info(
        f"\n=======================================================\n"
        f"FINAL EVALUATION ON OFFICIAL TEST SET (10,000 samples)\n"
        f"Selected Best Model: {best_overall_model_name.upper()} (Val Acc: {best_overall_val_acc*100:.2f}%)\n"
        f"Checkpoint: {best_overall_checkpoint}\n"
        f"======================================================="
    )

    test_loader = dataset_manager.get_test_loader(batch_size=128)
    best_model = get_model(best_overall_model_name, in_channels=1, num_classes=10)
    ckpt = torch.load(best_overall_checkpoint, map_location=device, weights_only=False)
    best_model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
        class_names=MNIST_CLASSES,
        collect_samples=True,
        max_samples=24,
    )

    logger.info(
        f"Official Test Set Performance:\n"
        f"  - Test Accuracy:        {test_metrics['accuracy']*100:.2f}%\n"
        f"  - Test Loss:            {test_metrics['loss']:.4f}\n"
        f"  - Test Macro F1:        {test_metrics['f1_macro']:.4f}\n"
        f"  - Test Macro Precision: {test_metrics['precision_macro']:.4f}\n"
        f"  - Test Macro Recall:    {test_metrics['recall_macro']:.4f}"
    )

    # 5. Plot and save final Test Set Confusion Matrix & Predictions
    test_plots_dir = os.path.join("outputs", "plots", "test_evaluation")
    os.makedirs(test_plots_dir, exist_ok=True)

    cm_test_path = os.path.join(test_plots_dir, f"test_confusion_matrix_{best_overall_model_name}.png")
    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names=MNIST_CLASSES,
        title=f"Test Set Confusion Matrix ({best_overall_model_name.upper()})",
        save_path=cm_test_path,
    )

    pred_test_path = os.path.join(test_plots_dir, f"test_sample_predictions_{best_overall_model_name}.png")
    if "samples" in test_metrics:
        plot_sample_predictions(
            images=test_metrics["samples"]["images"],
            true_labels=test_metrics["samples"]["true_labels"],
            pred_labels=test_metrics["samples"]["pred_labels"],
            confidences=test_metrics["samples"]["confidences"],
            class_names=MNIST_CLASSES,
            n_samples=20,
            title=f"Test Sample Predictions ({best_overall_model_name.upper()})",
            save_path=pred_test_path,
        )

    # 6. Build Comparison Table across Basic and Improved CNNs
    comparison_data = []
    for model_name, res in cv_results.items():
        comparison_data.append({
            "Architecture": model_name.upper(),
            "CV Val Accuracy": f"{res['accuracy_mean']*100:.2f}% ± {res['accuracy_std']*100:.2f}%",
            "CV Val Loss": f"{res['loss_mean']:.4f} ± {res['loss_std']:.4f}",
            "CV Val Macro F1": f"{res['f1_macro_mean']:.4f} ± {res['f1_macro_std']:.4f}",
            "CV Val Precision": f"{res['precision_macro_mean']:.4f} ± {res['precision_macro_std']:.4f}",
            "CV Val Recall": f"{res['recall_macro_mean']:.4f} ± {res['recall_macro_std']:.4f}",
        })

    # Save comparison as Markdown
    md_table = "| Architecture | CV Val Accuracy | CV Val Loss | CV Val Macro F1 | CV Val Precision | CV Val Recall |\n"
    md_table += "|---|---|---|---|---|---|\n"
    for row in comparison_data:
        md_table += (
            f"| {row['Architecture']} | {row['CV Val Accuracy']} | {row['CV Val Loss']} | "
            f"{row['CV Val Macro F1']} | {row['CV Val Precision']} | {row['CV Val Recall']} |\n"
        )

    md_path = os.path.join("outputs", "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("# DEEPVISION: MNIST Model Comparison Summary\n\n")
        f.write(md_table)
        f.write(f"\n### Official Untouched Test Set Results ({best_overall_model_name.upper()}):\n")
        f.write(f"- **Test Accuracy**: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"- **Test Loss**: {test_metrics['loss']:.4f}\n")
        f.write(f"- **Test Macro F1**: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"- **Test Macro Precision**: {test_metrics['precision_macro']:.4f}\n")
        f.write(f"- **Test Macro Recall**: {test_metrics['recall_macro']:.4f}\n")
        f.write(f"\n```\n{test_metrics['classification_report']}\n```\n")

    # Save comparison as CSV
    csv_comp_path = os.path.join("outputs", "comparison_summary.csv")
    with open(csv_comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_data[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_data)

    # Save complete experiment output as JSON
    final_output = {
        "experiment_name": "DEEPVISION_MNIST",
        "device_info": dev_info,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "cross_validation": cv_results,
        "official_test_set_results": {
            "evaluated_model": best_overall_model_name,
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "per_class_f1": test_metrics["per_class_f1"],
        },
    }
    json_comp_path = os.path.join("outputs", "final_experiment_results.json")
    with open(json_comp_path, "w") as f:
        json.dump(final_output, f, indent=4)

    # 7. Log summary run to W&B
    if wandb_mode != "disabled":
        summary_tracker = WandbTracker(
            project_name="DEEPVISION",
            experiment_name="mnist_final_comparison",
            config=final_output,
            group="mnist_experiments",
            job_type="evaluation",
            tags=["mnist", "comparison", "final_test"],
            mode=wandb_mode,
        )
        summary_tracker.log_summary({
            "test/best_model": best_overall_model_name,
            "test/accuracy": test_metrics["accuracy"],
            "test/loss": test_metrics["loss"],
            "test/f1_macro": test_metrics["f1_macro"],
        })
        summary_tracker.log_confusion_matrix(
            cm=np.array(test_metrics["confusion_matrix"]),
            class_names=MNIST_CLASSES,
            title="Official Test Set Confusion Matrix",
        )
        if "samples" in test_metrics:
            summary_tracker.log_sample_predictions(
                samples=test_metrics["samples"],
                class_names=MNIST_CLASSES,
                title="Official Test Set Sample Predictions",
            )
        summary_tracker.finish()

    logger.info(f"\nExperiment execution complete! Outputs saved to {os.path.abspath('outputs')}")
    return final_output


if __name__ == "__main__":
    args = parse_args()
    run_full_mnist_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        device_pref=args.device,
        wandb_mode=args.wandb_mode,
        models_to_run=args.models,
        seed=args.seed,
    )
