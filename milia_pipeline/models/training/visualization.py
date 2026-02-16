"""
Training Visualization Module

Provides visualization utilities for training metrics and loss curves:
- Loss curve plotting (matplotlib)
- Metrics curve plotting (matplotlib)
- Interactive plots (plotly)
- Learning rate visualization
- Export functionality (PNG, HTML, PDF)

Author: milia Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Visualization imports with availability checks
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server usage
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import kaleido

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING VISUALIZER
# =============================================================================


class TrainingVisualizer:
    """
    Visualization utilities for training metrics and loss curves.

    DYNAMIC: Supports both matplotlib (static) and plotly (interactive)
    PRODUCTION-READY: Handles missing dependencies gracefully
    FUTURE-PROOF: Extensible plot types, configurable styling

    Usage:
        >>> from milia_pipeline.models.training import TrainingVisualizer
        >>> visualizer = TrainingVisualizer(metrics_history)
        >>> visualizer.plot_loss_curves(save_path="loss_curves.png")
        >>> visualizer.plot_all_metrics(save_path="metrics.html", interactive=True)
        >>> visualizer.plot_training_summary(output_dir="./plots")
    """

    # Default style configuration
    DEFAULT_STYLE = {
        "figure_size": (12, 6),
        "dpi": 150,
        "line_width": 2,
        "marker_size": 4,
        "font_size": 12,
        "title_size": 14,
        "legend_size": 10,
        "grid_alpha": 0.3,
        "colors": {
            "train": "#2196F3",  # Blue
            "val": "#FF9800",  # Orange
            "test": "#4CAF50",  # Green
            "lr": "#9C27B0",  # Purple
        },
    }

    def __init__(
        self, metrics_history: dict[str, list[float]], style: dict[str, Any] | None = None
    ):
        """
        Initialize visualizer with metrics history.

        Args:
            metrics_history: Dictionary mapping metric names to lists of values
                           (from Trainer.metrics_history)
            style: Optional style configuration overrides
        """
        self.metrics_history = metrics_history
        self.style = {**self.DEFAULT_STYLE, **(style or {})}

        # Validate dependencies
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not available - static plots disabled. "
                "Install with: pip install matplotlib"
            )
        if not PLOTLY_AVAILABLE:
            logger.warning(
                "plotly not available - interactive plots disabled. "
                "Install with: pip install plotly"
            )

    def plot_loss_curves(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        title: str = "Training Loss Curves",
    ) -> plt.Figure | None:
        """
        Plot training and validation loss curves.

        Args:
            save_path: Optional path to save figure
            show: Whether to display plot (default: False for server usage)
            title: Plot title

        Returns:
            matplotlib Figure object, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot loss curves - matplotlib not available")
            return None

        fig, ax = plt.subplots(figsize=self.style["figure_size"], dpi=self.style["dpi"])

        epochs = None

        # Plot training loss
        if "train_loss" in self.metrics_history:
            train_loss = self.metrics_history["train_loss"]
            epochs = range(1, len(train_loss) + 1)
            ax.plot(
                epochs,
                train_loss,
                label="Train Loss",
                color=self.style["colors"]["train"],
                linewidth=self.style["line_width"],
                marker="o",
                markersize=self.style["marker_size"],
            )

        # Plot validation loss
        if "val_loss" in self.metrics_history:
            val_loss = self.metrics_history["val_loss"]
            if epochs is None:
                epochs = range(1, len(val_loss) + 1)
            ax.plot(
                epochs[: len(val_loss)],
                val_loss,
                label="Validation Loss",
                color=self.style["colors"]["val"],
                linewidth=self.style["line_width"],
                marker="s",
                markersize=self.style["marker_size"],
            )

        ax.set_xlabel("Epoch", fontsize=self.style["font_size"])
        ax.set_ylabel("Loss", fontsize=self.style["font_size"])
        ax.set_title(title, fontsize=self.style["title_size"])
        ax.legend(fontsize=self.style["legend_size"])
        ax.grid(True, alpha=self.style["grid_alpha"])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.style["dpi"], bbox_inches="tight")
            logger.info(f"Loss curves saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_metrics(
        self,
        metric_names: list[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
        title: str = "Training Metrics",
    ) -> plt.Figure | None:
        """
        Plot specified metrics over epochs.

        Args:
            metric_names: List of metric names to plot (None = all except loss)
            save_path: Optional path to save figure
            show: Whether to display plot
            title: Plot title

        Returns:
            matplotlib Figure object, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot metrics - matplotlib not available")
            return None

        # Filter metrics to plot (exclude loss by default)
        if metric_names is None:
            metric_names = [k for k in self.metrics_history if "loss" not in k.lower()]

        if not metric_names:
            logger.warning("No metrics to plot")
            return None

        # Determine subplot layout
        n_metrics = len(metric_names)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.style["figure_size"][0], self.style["figure_size"][1] * n_rows / 2),
            dpi=self.style["dpi"],
        )

        axes = [axes] if n_metrics == 1 else axes.flatten()

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]

            if metric_name in self.metrics_history:
                values = self.metrics_history[metric_name]
                epochs = range(1, len(values) + 1)

                # Determine color based on metric type
                if "train" in metric_name.lower():
                    color = self.style["colors"]["train"]
                elif "val" in metric_name.lower():
                    color = self.style["colors"]["val"]
                elif "test" in metric_name.lower():
                    color = self.style["colors"]["test"]
                else:
                    color = self.style["colors"]["train"]

                ax.plot(
                    epochs,
                    values,
                    label=metric_name,
                    color=color,
                    linewidth=self.style["line_width"],
                    marker="o",
                    markersize=self.style["marker_size"],
                )

            ax.set_xlabel("Epoch", fontsize=self.style["font_size"])
            ax.set_ylabel(metric_name, fontsize=self.style["font_size"])
            ax.set_title(metric_name, fontsize=self.style["title_size"])
            ax.grid(True, alpha=self.style["grid_alpha"])

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=self.style["title_size"] + 2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.style["dpi"], bbox_inches="tight")
            logger.info(f"Metrics plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_learning_rate(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        title: str = "Learning Rate Schedule",
    ) -> plt.Figure | None:
        """
        Plot learning rate over epochs.

        Args:
            save_path: Optional path to save figure
            show: Whether to display plot
            title: Plot title

        Returns:
            matplotlib Figure object, or None if unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot plot learning rate - matplotlib not available")
            return None

        # Check for learning rate in history
        lr_key = None
        for key in ["lr", "learning_rate", "LR"]:
            if key in self.metrics_history:
                lr_key = key
                break

        if lr_key is None:
            logger.warning("No learning rate data in metrics history")
            return None

        fig, ax = plt.subplots(figsize=self.style["figure_size"], dpi=self.style["dpi"])

        lr_values = self.metrics_history[lr_key]
        epochs = range(1, len(lr_values) + 1)

        ax.plot(
            epochs,
            lr_values,
            label="Learning Rate",
            color=self.style["colors"]["lr"],
            linewidth=self.style["line_width"],
            marker="o",
            markersize=self.style["marker_size"],
        )

        ax.set_xlabel("Epoch", fontsize=self.style["font_size"])
        ax.set_ylabel("Learning Rate", fontsize=self.style["font_size"])
        ax.set_title(title, fontsize=self.style["title_size"])
        ax.legend(fontsize=self.style["legend_size"])
        ax.grid(True, alpha=self.style["grid_alpha"])
        ax.set_yscale("log")  # Log scale for learning rate

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.style["dpi"], bbox_inches="tight")
            logger.info(f"Learning rate plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_interactive(
        self, save_path: str | Path | None = None, title: str = "Training Progress"
    ) -> Optional["go.Figure"]:
        """
        Create interactive plotly visualization.

        Args:
            save_path: Optional path to save HTML file
            title: Plot title

        Returns:
            plotly Figure object, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Cannot create interactive plot - plotly not available")
            return None

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1, subplot_titles=("Loss Curves", "Metrics"), vertical_spacing=0.15
        )

        # Add loss curves
        if "train_loss" in self.metrics_history:
            train_loss = self.metrics_history["train_loss"]
            epochs = list(range(1, len(train_loss) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_loss,
                    mode="lines+markers",
                    name="Train Loss",
                    line=dict(color=self.style["colors"]["train"]),
                ),
                row=1,
                col=1,
            )

        if "val_loss" in self.metrics_history:
            val_loss = self.metrics_history["val_loss"]
            epochs = list(range(1, len(val_loss) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_loss,
                    mode="lines+markers",
                    name="Validation Loss",
                    line=dict(color=self.style["colors"]["val"]),
                ),
                row=1,
                col=1,
            )

        # Add other metrics
        metric_count = 0
        for key, values in self.metrics_history.items():
            if "loss" not in key.lower() and values:
                epochs = list(range(1, len(values) + 1))
                fig.add_trace(
                    go.Scatter(x=epochs, y=values, mode="lines+markers", name=key), row=2, col=1
                )
                metric_count += 1

        fig.update_layout(title=title, height=800, showlegend=True)

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=1)

        if save_path:
            save_path = Path(save_path)
            if save_path.suffix == ".html":
                fig.write_html(str(save_path))
            elif (
                save_path.suffix == ".png"
                and KALEIDO_AVAILABLE
                or save_path.suffix == ".pdf"
                and KALEIDO_AVAILABLE
            ):
                fig.write_image(str(save_path))
            else:
                fig.write_html(str(save_path.with_suffix(".html")))
            logger.info(f"Interactive plot saved to: {save_path}")

        return fig

    def plot_training_summary(
        self, output_dir: str | Path, prefix: str = "", formats: list[str] = None
    ) -> dict[str, Path]:
        """
        Generate comprehensive training visualization summary.

        DYNAMIC: Generates all available plot types
        PRODUCTION-READY: Handles missing data gracefully
        FUTURE-PROOF: Returns paths for integration with reporting systems

        Args:
            output_dir: Directory to save plots
            prefix: Optional prefix for filenames
            formats: List of formats to generate (default: ['png', 'html'])

        Returns:
            Dictionary mapping plot types to saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["png", "html"]
        saved_paths = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = f"{prefix}_{timestamp}" if prefix else timestamp

        # Generate static plots (matplotlib)
        if "png" in formats and MATPLOTLIB_AVAILABLE:
            # Loss curves
            loss_path = output_dir / f"{file_prefix}_loss_curves.png"
            if self.plot_loss_curves(save_path=loss_path):
                saved_paths["loss_curves_png"] = loss_path

            # All metrics
            metrics_path = output_dir / f"{file_prefix}_metrics.png"
            if self.plot_metrics(save_path=metrics_path):
                saved_paths["metrics_png"] = metrics_path

            # Learning rate
            lr_path = output_dir / f"{file_prefix}_learning_rate.png"
            if self.plot_learning_rate(save_path=lr_path):
                saved_paths["learning_rate_png"] = lr_path

        # Generate interactive plots (plotly)
        if "html" in formats and PLOTLY_AVAILABLE:
            interactive_path = output_dir / f"{file_prefix}_interactive.html"
            if self.plot_interactive(save_path=interactive_path):
                saved_paths["interactive_html"] = interactive_path

        # Generate PDF if requested and kaleido available
        if "pdf" in formats and PLOTLY_AVAILABLE and KALEIDO_AVAILABLE:
            pdf_path = output_dir / f"{file_prefix}_summary.pdf"
            if self.plot_interactive(save_path=pdf_path):
                saved_paths["summary_pdf"] = pdf_path

        logger.info(f"Training summary generated: {len(saved_paths)} files saved to {output_dir}")

        return saved_paths


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def plot_training_summary(
    metrics_history: dict[str, list[float]],
    output_dir: str | Path,
    prefix: str = "",
    formats: list[str] = None,
) -> dict[str, Path]:
    """
    Convenience function to generate training visualization summary.

    Example:
        >>> from milia_pipeline.models.training import plot_training_summary
        >>> saved_paths = plot_training_summary(
        ...     trainer.metrics_history,
        ...     output_dir="./training_plots"
        ... )
    """
    visualizer = TrainingVisualizer(metrics_history)
    return visualizer.plot_training_summary(output_dir, prefix, formats)


def create_visualizer(
    metrics_history: dict[str, list[float]], style: dict[str, Any] | None = None
) -> TrainingVisualizer:
    """
    Convenience function to create a TrainingVisualizer.

    Example:
        >>> from milia_pipeline.models.training import create_visualizer
        >>> visualizer = create_visualizer(trainer.metrics_history)
        >>> visualizer.plot_loss_curves(save_path="loss.png")
    """
    return TrainingVisualizer(metrics_history, style)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"visualization module loaded - "
    f"matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}, "
    f"plotly: {'✓' if PLOTLY_AVAILABLE else '✗'}, "
    f"kaleido: {'✓' if KALEIDO_AVAILABLE else '✗'}"
)
