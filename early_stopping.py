"""
Early stopping monitor for federated learning with multi-metric support

This module provides early stopping functionality that monitors:
- Validation accuracy (primary metric)
- Near-OOD AUROC (secondary metric)
- Far-OOD AUROC (tertiary metric)

Stops training when:
1. No improvement in validation accuracy for `patience` rounds
2. OR significant degradation in OOD metrics (> tolerance drop from best)
"""

import os
import torch
from datetime import datetime


class EarlyStoppingMonitor:
    """
    Multi-metric early stopping monitor for federated learning

    Monitors:
    - Validation accuracy (primary)
    - Near-OOD AUROC (secondary)
    - Far-OOD AUROC (tertiary)

    Stops training when:
    1. No improvement in validation accuracy for `patience` rounds
    2. AND significant degradation in OOD metrics (> tolerance drop from best)
    """

    def __init__(self, patience=10, min_delta=0.001,
                 ood_tolerance=0.05, restore_best_weights=True,
                 save_dir=None):
        """
        Args:
            patience: Rounds to wait before stopping (default: 10)
            min_delta: Minimum change to qualify as improvement (default: 0.001)
            ood_tolerance: Max allowed AUROC degradation from best (default: 5%)
            restore_best_weights: Whether to restore best model on stop (default: True)
            save_dir: Directory to save best checkpoint info (default: None)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.ood_tolerance = ood_tolerance
        self.restore_best_weights = restore_best_weights
        self.save_dir = save_dir

        # Best metrics
        self.best_val_acc = 0.0
        self.best_near_auroc = None
        self.best_far_auroc = None
        self.best_round = 0

        # Tracking
        self.wait = 0
        self.stopped_epoch = 0
        self.best_checkpoint_path = None
        self.history = {
            'rounds': [],
            'val_acc': [],
            'near_auroc': [],
            'far_auroc': [],
            'wait_count': []
        }

        # Logging
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.log_path = os.path.join(save_dir, "early_stopping_log.txt")
        else:
            self.log_path = None

    def check(self, round_num, val_acc, near_auroc=None, far_auroc=None, checkpoint_path=None):
        """
        Check if training should stop

        Args:
            round_num: Current communication round number
            val_acc: Validation accuracy this round
            near_auroc: Near-OOD AUROC this round (optional)
            far_auroc: Far-OOD AUROC this round (optional)
            checkpoint_path: Path to checkpoint file for this round

        Returns:
            should_stop (bool): True if training should stop
            reason (str): Explanation for stopping
        """
        # Update history
        self.history['rounds'].append(round_num)
        self.history['val_acc'].append(val_acc)
        self.history['near_auroc'].append(near_auroc)
        self.history['far_auroc'].append(far_auroc)
        self.history['wait_count'].append(self.wait)

        # Check for validation accuracy improvement
        val_improved = val_acc > self.best_val_acc + self.min_delta

        if val_improved:
            # New best model
            self.best_val_acc = val_acc
            if near_auroc is not None:
                if self.best_near_auroc is None:
                    self.best_near_auroc = near_auroc
                else:
                    self.best_near_auroc = max(self.best_near_auroc, near_auroc)
            if far_auroc is not None:
                if self.best_far_auroc is None:
                    self.best_far_auroc = far_auroc
                else:
                    self.best_far_auroc = max(self.best_far_auroc, far_auroc)
            self.best_round = round_num
            self.wait = 0
            self.best_checkpoint_path = checkpoint_path

            msg = f"Validation improved to {val_acc:.4f}"
            self._log(round_num, msg, "improvement")
            return False, msg

        # Check for OOD degradation (even if val didn't improve much)
        near_degraded = (
            near_auroc is not None
            and self.best_near_auroc is not None
            and (self.best_near_auroc - near_auroc) > self.ood_tolerance
        )
        far_degraded = (
            far_auroc is not None
            and self.best_far_auroc is not None
            and (self.best_far_auroc - far_auroc) > self.ood_tolerance
        )

        if near_degraded or far_degraded:
            reason_lines = ["OOD degradation detected"]
            if near_degraded:
                reason_lines.append(
                    f"  Near-OOD: {near_auroc:.4f} (best: {self.best_near_auroc:.4f}, "
                    f"change: {(near_auroc - self.best_near_auroc):.4f})"
                )
            if far_degraded:
                reason_lines.append(
                    f"  Far-OOD: {far_auroc:.4f} (best: {self.best_far_auroc:.4f}, "
                    f"change: {(far_auroc - self.best_far_auroc):.4f})"
                )
            reason = "\n".join(reason_lines)

            self.stopped_epoch = round_num
            self._log(round_num, reason, "stopped")
            return True, reason

        # Increment wait counter
        self.wait += 1

        # Check patience
        if self.wait >= self.patience:
            self.stopped_epoch = round_num
            reason = (f"Early stopping: No improvement in validation accuracy for {self.wait} rounds\n"
                     f"  Best val_acc: {self.best_val_acc:.4f} (Round {self.best_round})\n"
                     f"  Current val_acc: {val_acc:.4f} (Round {round_num})")

            self._log(round_num, reason, "stopped")
            return True, reason

        # Still waiting
        msg = f"Waiting: {self.wait}/{self.patience} rounds without improvement"
        self._log(round_num, msg, "waiting")
        return False, msg

    def get_best_checkpoint(self):
        """Return path to best checkpoint"""
        return self.best_checkpoint_path

    def get_summary(self):
        """Get summary of early stopping metrics"""
        return {
            'best_round': self.best_round,
            'best_val_acc': self.best_val_acc,
            'best_near_auroc': self.best_near_auroc,
            'best_far_auroc': self.best_far_auroc,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else None,
            'total_rounds': len(self.history['rounds']),
            'reason': self._get_stop_reason()
        }

    def _get_stop_reason(self):
        """Determine why training stopped"""
        if self.stopped_epoch == 0:
            return "Training completed all rounds"

        # Get metrics at stop
        stop_idx = len(self.history['rounds']) - 1
        val_acc = self.history['val_acc'][stop_idx]
        near_auroc = self.history['near_auroc'][stop_idx]
        far_auroc = self.history['far_auroc'][stop_idx]

        # Check what caused stop
        near_degraded = (
            near_auroc is not None
            and self.best_near_auroc is not None
            and (self.best_near_auroc - near_auroc) > self.ood_tolerance
        )
        far_degraded = (
            far_auroc is not None
            and self.best_far_auroc is not None
            and (self.best_far_auroc - far_auroc) > self.ood_tolerance
        )
        patience_exceeded = self.wait >= self.patience

        if near_degraded or far_degraded:
            return "OOD degradation"
        elif patience_exceeded:
            return f"Patience exceeded ({self.wait} rounds without improvement)"
        else:
            return "Unknown"

    def _log(self, round_num, message, level="info"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Round {round_num} - {message}"

        # Print to console
        if level == "stopped":
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING: {message}")
            print(f"{'='*70}\n")
        elif level == "improvement":
            print(f"  [ES Improvement] {message}")
        elif level == "waiting":
            if round_num % 5 == 0:  # Only print waiting status periodically
                print(f"  [ES Waiting] {message}")

        # Write to log file
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
    # Test early stopping monitor
    monitor = EarlyStoppingMonitor(patience=5, min_delta=0.001, ood_tolerance=0.05)

    # Simulate training
    import numpy as np

    # Rounds 1-10: Improving
    for r in range(1, 11):
        val_acc = 0.85 + r * 0.01
        near_auroc = 0.90 + r * 0.005
        far_auroc = 0.85 + r * 0.003
        should_stop, reason = monitor.check(r, val_acc, near_auroc, far_auroc, f"ckpt_{r}.pth")
        print(f"Round {r}: should_stop={should_stop}, {reason}")

    # Rounds 11-20: No improvement (validation stable, OOD degrading)
    for r in range(11, 21):
        val_acc = 0.94 + np.random.randn() * 0.002  # Small fluctuations
        near_auroc = 0.93 - (r - 10) * 0.01  # Gradual degradation
        far_auroc = 0.87 - (r - 10) * 0.008  # Gradual degradation
        should_stop, reason = monitor.check(r, val_acc, near_auroc, far_auroc, f"ckpt_{r}.pth")
        print(f"Round {r}: should_stop={should_stop}, {reason}")
        if should_stop:
            break

    print(f"\nSummary: {monitor.get_summary()}")
