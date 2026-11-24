"""Weights & Biases (W&B) experiment tracking integration for DEEPVISION."""

import os
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import wandb
from dotenv import load_dotenv
from evaluation.visualizer import plot_confusion_matrix, plot_sample_predictions

logger = logging.getLogger(__name__)


class WandbTracker:
    """Manages W&B experiment runs, metric logging, artifacts, and visualizations."""

    def __init__(
        self,
        project_name: str = "DEEPVISION",
        experiment_name: str = "run",
        config: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        tags: Optional[List[str]] = None,
        mode: str = "online",
    ) -> None:
        """Initialize W&B tracker with fallback to offline mode.

        Args:
            project_name: Name of the W&B project ('DEEPVISION').
            experiment_name: Display name for the run.
            config: Hyperparameters dictionary.
            group: Group identifier (e.g. cross-validation experiment group).
            job_type: Type of job ('train', 'eval', 'cv-fold').
            tags: List of descriptive tags.
            mode: 'online', 'offline', or 'disabled'.
        """
        # Load environment variables without ever logging the key
        load_dotenv(override=False)

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.group = group
        self.job_type = job_type
        self.tags = tags or []
        self.mode = mode.lower()
        self.run = None

        self._initialize_run()

    def _initialize_run(self) -> None:
        """Initialize the wandb run securely."""
        if self.mode == "disabled":
            logger.info("W&B tracking is disabled.")
            return

        api_key = os.environ.get("WANDB_API_KEY")

        # Attempt online login if key is available and online mode requested
        if self.mode == "online" and api_key:
            try:
                # Login silently without printing the key
                wandb.login(key=api_key, relogin=False)
                logger.info("W&B authenticated successfully.")
            except Exception as e:
                logger.warning(f"W&B login encountered an issue ({e}). Falling back to offline mode.")
                self.mode = "offline"
        elif self.mode == "online" and not api_key:
            logger.warning("No WANDB_API_KEY detected in environment. Running W&B in offline mode.")
            self.mode = "offline"

        try:
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                group=self.group,
                job_type=self.job_type,
                tags=self.tags,
                mode=self.mode,
                reinit=True,
            )
            logger.info(f"W&B run initialized in '{self.mode}' mode: {self.experiment_name} (ID: {self.run.id})")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B ({e}). Switching to offline mode.")
            self.mode = "offline"
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                group=self.group,
                job_type=self.job_type,
                tags=self.tags,
                mode="offline",
                reinit=True,
            )

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log per-epoch metrics (train/loss, val/loss, acc, lr).

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
        """
        if not self.run:
            return

        formatted_metrics = {"epoch": epoch}
        for k, v in metrics.items():
            if k != "epoch":
                formatted_metrics[k] = v

        self.run.log(formatted_metrics, step=epoch)

    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        step: Optional[int] = None,
    ) -> None:
        """Log a confusion matrix plot and raw heatmap to W&B.

        Args:
            cm: Confusion matrix array.
            class_names: List of class labels.
            title: Title for plot.
            step: Optional step/epoch index.
        """
        if not self.run:
            return

        fig = plot_confusion_matrix(cm, class_names=class_names, title=title)
        self.run.log({f"plots/{title}": wandb.Image(fig)}, step=step)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def log_sample_predictions(
        self,
        samples: Dict[str, np.ndarray],
        class_names: List[str],
        title: str = "Sample Predictions",
        step: Optional[int] = None,
    ) -> None:
        """Log a grid of prediction images with predictions vs ground truth.

        Args:
            samples: Dictionary with 'images', 'true_labels', 'pred_labels', 'confidences'.
            class_names: List of class labels.
            title: Title for visualization.
            step: Optional step/epoch index.
        """
        if not self.run or not samples:
            return

        fig = plot_sample_predictions(
            images=samples["images"],
            true_labels=samples["true_labels"],
            pred_labels=samples["pred_labels"],
            confidences=samples["confidences"],
            class_names=class_names,
            n_samples=16,
            title=title,
        )
        self.run.log({f"plots/{title}": wandb.Image(fig)}, step=step)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def log_summary(self, summary_metrics: Dict[str, Any]) -> None:
        """Log final summary metrics to W&B run summary.

        Args:
            summary_metrics: Dictionary of final aggregated metrics.
        """
        if not self.run:
            return

        for k, v in summary_metrics.items():
            if isinstance(v, (int, float, str, bool)):
                self.run.summary[k] = v

    def log_artifact(
        self,
        filepath: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a file (e.g. checkpoint) as a W&B Artifact.

        Args:
            filepath: Path to the local file.
            name: Artifact name.
            artifact_type: Artifact category ('model', 'dataset', 'results').
            metadata: Optional dictionary with artifact metadata.
        """
        if not self.run or self.mode == "offline":
            return

        try:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            artifact.add_file(filepath)
            self.run.log_artifact(artifact)
            logger.info(f"Logged artifact '{name}' to W&B.")
        except Exception as e:
            logger.warning(f"Could not log artifact '{name}' to W&B: {e}")

    def finish(self) -> None:
        """Finish and close the W&B run."""
        if self.run:
            self.run.finish()
            self.run = None
