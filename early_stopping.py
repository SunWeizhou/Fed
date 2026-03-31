"""Validation-accuracy early stopping for FedViM training."""

import os
from datetime import datetime


class EarlyStoppingMonitor:
    """Single-metric early stopping monitor based on validation accuracy."""

    def __init__(self, patience=10, min_delta=0.001, save_dir=None):
        """
        Args:
            patience: Rounds to wait before stopping (default: 10)
            min_delta: Minimum change to qualify as improvement (default: 0.001)
            save_dir: Directory to save best checkpoint info (default: None)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir

        self.best_val_acc = 0.0
        self.best_round = 0

        self.wait = 0
        self.stopped_epoch = 0
        self.best_checkpoint_path = None
        self.history = {
            'rounds': [],
            'val_acc': [],
            'wait_count': []
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.log_path = os.path.join(save_dir, "early_stopping_log.txt")
        else:
            self.log_path = None

    def check(self, round_num, val_acc, checkpoint_path=None):
        """Check whether validation accuracy has plateaued."""
        self.history['rounds'].append(round_num)
        self.history['val_acc'].append(val_acc)
        self.history['wait_count'].append(self.wait)

        val_improved = val_acc > self.best_val_acc + self.min_delta

        if val_improved:
            self.best_val_acc = val_acc
            self.best_round = round_num
            self.wait = 0
            self.best_checkpoint_path = checkpoint_path

            msg = f"Validation improved to {val_acc:.4f}"
            self._log(round_num, msg, "improvement")
            return False, msg

        self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = round_num
            reason = (f"Early stopping: No improvement in validation accuracy for {self.wait} rounds\n"
                     f"  Best val_acc: {self.best_val_acc:.4f} (Round {self.best_round})\n"
                     f"  Current val_acc: {val_acc:.4f} (Round {round_num})")

            self._log(round_num, reason, "stopped")
            return True, reason

        msg = f"Waiting: {self.wait}/{self.patience} rounds without improvement"
        self._log(round_num, msg, "waiting")
        return False, msg

    def get_best_checkpoint(self):
        """Return path to best checkpoint"""
        return self.best_checkpoint_path

    def get_summary(self):
        """Get summary of validation-based early stopping."""
        return {
            'best_round': self.best_round,
            'best_val_acc': self.best_val_acc,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else None,
            'total_rounds': len(self.history['rounds']),
            'reason': self._get_stop_reason()
        }

    def _get_stop_reason(self):
        """Determine why training stopped."""
        if self.stopped_epoch == 0:
            return "Training completed all rounds"
        return f"Patience exceeded ({self.wait} rounds without improvement)"

    def _log(self, round_num, message, level="info"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Round {round_num} - {message}"

        if level == "stopped":
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING: {message}")
            print(f"{'='*70}\n")
        elif level == "improvement":
            print(f"  [ES Improvement] {message}")
        elif level == "waiting":
            if round_num % 5 == 0:  # Only print waiting status periodically
                print(f"  [ES Waiting] {message}")

        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(log_entry + "\n")

    def save_history(self):
        """Save training history to file"""
        if self.save_dir:
            import json
            history_path = os.path.join(self.save_dir, "early_stopping_history.json")

            # Add summary to history
            history_to_save = self.history.copy()
            history_to_save['summary'] = self.get_summary()

            with open(history_path, 'w') as f:
                json.dump(history_to_save, f, indent=2)

            print(f"Early stopping history saved to {history_path}")


if __name__ == "__main__":
    monitor = EarlyStoppingMonitor(patience=5, min_delta=0.001)

    for round_num, val_acc in enumerate([0.82, 0.86, 0.89, 0.901, 0.902, 0.901, 0.900, 0.900], start=1):
        should_stop, reason = monitor.check(round_num, val_acc, f"ckpt_{round_num}.pth")
        print(f"Round {round_num}: should_stop={should_stop}, {reason}")
        if should_stop:
            break

    print(f"\nSummary: {monitor.get_summary()}")
