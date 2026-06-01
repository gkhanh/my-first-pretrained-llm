"""CSV and optional W&B training logger."""

from __future__ import annotations

import csv
from pathlib import Path


class TrainingLogger:
    """Logs training metrics to CSV. W&B is opt-in via use_wandb flag.

    Args:
        run_dir: Root run directory.
        use_wandb: If True, also logs to Weights & Biases.
        wandb_project: W&B project name (only used if use_wandb=True).
        run_name: Run name for display.
    """

    CSV_FIELDS = ["step", "tokens", "loss", "lr", "tokens_per_sec", "grad_norm"]

    def __init__(
        self,
        run_dir: str | Path,
        use_wandb: bool = False,
        wandb_project: str = "khanh-llm",
        run_name: str = "run",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.run_name = run_name

        self._csv_path = self.run_dir / "training_log.csv"
        self._init_csv()

        if use_wandb:
            self._init_wandb(wandb_project, run_name)

    def _init_csv(self) -> None:
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_FIELDS)

    def _init_wandb(self, project: str, name: str) -> None:
        try:
            import wandb  # type: ignore
            wandb.init(project=project, name=name)
        except ImportError:
            print("W&B not installed. Falling back to CSV-only logging.")
            self.use_wandb = False

    def log(
        self,
        step: int,
        tokens: int,
        max_tokens: int,
        loss: float,
        lr: float,
        tokens_per_sec: float,
        grad_norm: float,
        eta_seconds: float = 0.0,
    ) -> None:
        """Log a single training step."""
        row = {
            "step":          step,
            "tokens":        tokens,
            "loss":          loss,
            "lr":            lr,
            "tokens_per_sec": tokens_per_sec,
            "grad_norm":     grad_norm,
        }

        # CSV
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)

        # Absolute completion time formatting
        import datetime
        if eta_seconds > 0:
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            time_str = f"Finishes: {finish_time.strftime('%b %d, %H:%M')}"
        else:
            time_str = ""

        # Console
        print(
            f"step={step:>8,} | tokens={tokens/1e9:.3f}B / {max_tokens/1e9:.3f}B | "
            f"loss={loss:.4f} | lr={lr:.2e} | "
            f"tok/s={tokens_per_sec:,.0f} | grad_norm={grad_norm:.3f}"
            + (f" | {time_str}" if time_str else "")
        )

        # W&B
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                wandb.log(row, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                wandb.finish()
            except Exception:
                pass
