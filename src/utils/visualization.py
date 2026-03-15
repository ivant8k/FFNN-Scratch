from pathlib import Path
import sys
 
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

class Visualizer:
    @staticmethod
    def _auto_colors(n: int) -> list:
        """Utilitas untuk generate n warna berbeda"""
        return [plt.cm.tab10(i) for i in range(n)]

    @staticmethod
    def _resolve_layer_keys(data_dict: dict, layer_keys: list | None) -> list:
        """
        Menentukan key layer mana yang akan diplot.
        Jika layer_keys=None, pakai semua key yang ada.
        Jika layer_keys diberikan, filter hanya yang valid (ada di dict).
        Guard ini penting karena key bisa berbeda antar model jika arsitektur berbeda.
        """
        if layer_keys is None:
            return sorted(data_dict.keys())
        return [k for k in layer_keys if k in data_dict]

    @staticmethod
    def _annotate_stats(ax, values: np.ndarray) -> None:
        """Utility untuk menambah anotasi mean dan std di pojok kanan atas histogram"""
        ax.text(
            0.97, 0.95,
            f"std  = {values.std():.4f}\nmean = {values.mean():.4f}",
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#cccccc'),
        )

    @staticmethod
    def _savefig(fig, save_path: str | None) -> None:
        """Simpan figure"""
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] saved in {save_path}")
        plt.show()

    def plot_weight_distribution(
            self, 
            weight_dict : dict, 
            layer_keys: list[str] | None = None,
            title: str = "Weight distribution",
            color: str = "steelblue",
            save_path: str | None = None,
            ) -> None:
        """Plot histogram distribusi bobot untuk setiap linear layer."""
        keys = self._resolve_layer_keys(weight_dict, layer_keys)

        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
        fig.suptitle(title, fontsize=13, fontweight='bold')

        for ax, k in zip(axes[0], keys):
            w = weight_dict[k]
            ax.hist(w, bins=50, color=color, alpha=0.8)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
            self._annotate_stats(ax, w)
            ax.set_title(k, fontsize=10)
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.3)
 
        plt.tight_layout()
        self._savefig(fig, save_path)

    def plot_gradient_distribution(
        self,
        grad_dict  : dict,
        layer_keys : list[str] | None = None,
        title      : str = "Gradient distribution",
        color      : str = "darkorange",
        save_path  : str | None = None,
        ) -> None:
        keys = self._resolve_layer_keys(grad_dict, layer_keys)
 
        fig, axes = plt.subplots(1, len(keys),
                                  figsize=(5 * len(keys), 4),
                                  squeeze=False)
        fig.suptitle(title, fontsize=13, fontweight='bold')
 
        for ax, k in zip(axes[0], keys):
            g = grad_dict[k]
            ax.hist(g, bins=50, color=color, alpha=0.8, edgecolor='none')
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
            self._annotate_stats(ax, g)
            ax.set_title(f"{k}  —  dL/dW", fontsize=10)
            ax.set_xlabel("Gradient value")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.3)
 
        plt.tight_layout()
        self._savefig(fig, save_path)

    def plot_loss_curve(
        self,
        history   : dict,
        title     : str = "Training & Validation Loss",
        show_acc  : bool = False,
        save_path : str | None = None,
        ) -> None:
        train_loss = history.get('train_loss', [])
        val_loss   = history.get('val_loss',   [])
        train_acc  = history.get('train_acc',  [])
        val_acc    = history.get('val_acc',    [])
 
        epochs = range(1, len(train_loss) + 1)
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title, fontsize=13, fontweight='bold')
 
        # kiri: training
        axes[0].plot(epochs, train_loss, color='steelblue', label='Train loss')
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        if show_acc and train_acc:
            ax2 = axes[0].twinx()
            ax2.plot(epochs, train_acc, color='steelblue',
                     linestyle='--', alpha=0.5, label='Train acc')
            ax2.set_ylabel("Accuracy", color='steelblue')
        axes[0].legend(loc='upper right')
 
        # kanan: validation
        if val_loss:
            val_epochs = range(1, len(val_loss) + 1)
            axes[1].plot(val_epochs, val_loss, color='tomato', label='Val loss')
            axes[1].set_title("Validation Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].grid(alpha=0.3)
            if show_acc and val_acc:
                ax3 = axes[1].twinx()
                ax3.plot(val_epochs, val_acc, color='tomato',
                         linestyle='--', alpha=0.5, label='Val acc')
                ax3.set_ylabel("Accuracy", color='tomato')
            axes[1].legend(loc='upper right')
        else:
            axes[1].text(0.5, 0.5, "val_loss tidak tersedia",
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=11, color='gray')
            axes[1].set_title("Validation Loss")
 
        plt.tight_layout()
        self._savefig(fig, save_path)

    def plot_comparison(
        self,
        results     : dict,
        layer_keys  : list[str] | None = None,
        mode        : str = 'all',
        save_prefix : str = "comparison",
        x_ref       : np.ndarray | None = None,
        y_ref       : np.ndarray | None = None,
        ) -> None:
        
        if not results:
            print("[Visualizer] plot_comparison: results kosong.")
            return
 
        labels = list(results.keys())
        colors = [
            results[lb].get('color', c)
            for lb, c in zip(labels, self._auto_colors(len(labels)))
        ]
 
        # A: loss curve
        if mode in ('loss', 'all'):
            fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{save_prefix} — Loss Curve",
                         fontsize=13, fontweight='bold')
 
            for lb, c in zip(labels, colors):
                h = results[lb]['history']
                epochs_tr  = range(1, len(h.get('train_loss', [])) + 1)
                epochs_val = range(1, len(h.get('val_loss',   [])) + 1)
                ax_tr.plot(epochs_tr,  h.get('train_loss', []), label=lb, color=c)
                ax_val.plot(epochs_val, h.get('val_loss',  []), label=lb, color=c)
 
            for ax, t in zip([ax_tr, ax_val], ['Training Loss', 'Validation Loss']):
                ax.set_title(t)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
 
            plt.tight_layout()
            self._savefig(fig, f"{save_prefix}_loss.png")
 
        # B: weight distribution
        if mode in ('weights', 'all'):
            wdicts = {lb: results[lb]['model'].get_weight_distribution()
                      for lb in labels}
            ref_keys = self._resolve_layer_keys(
                next(iter(wdicts.values())), layer_keys
            )

            n_cond   = len(labels)
            n_layers = len(ref_keys)
            fig, axes = plt.subplots(n_cond, n_layers,
                                          figsize=(5 * n_layers, 3.5 * n_cond),
                                          squeeze=False)
            fig.suptitle(f"{save_prefix} — Weight Distribution",
                             fontsize=13, fontweight='bold')
 
            for row, (lb, c) in enumerate(zip(labels, colors)):
                for col, k in enumerate(ref_keys):
                    ax = axes[row][col]
                    if k not in wdicts[lb]:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes)
                        continue
                    w = wdicts[lb][k]
                    ax.hist(w, bins=40, color=c, alpha=0.8, edgecolor='none')
                    ax.axvline(0, color='black', linewidth=0.8,
                               linestyle='--', alpha=0.5)
                    self._annotate_stats(ax, w)
                    ax.set_title(f"{lb}\n{k}", fontsize=8)
                    ax.set_xlabel("Weight value", fontsize=8)
                    ax.set_ylabel("Count", fontsize=8)
                    ax.grid(axis='y', alpha=0.3)
 
            plt.tight_layout()
            self._savefig(fig, f"{save_prefix}_weights.png")
 
        #C: gradient distribution
        if mode in ('grads', 'all'):
            # trigger fresh backward pass jika x_ref tersedia
            if x_ref is not None and y_ref is not None:
                for lb in labels:
                    m = results[lb]['model']
                    y_pred = m.forward(np.asarray(x_ref, dtype=np.float64))
                    m.backward(y_pred,
                               np.asarray(y_ref, dtype=np.float64).reshape(-1, 1))
 
            gdicts = {lb: results[lb]['model'].get_gradient_distribution()
                      for lb in labels}
 
            gdicts = {lb: gd for lb, gd in gdicts.items() if gd}
            if not gdicts:
                print("[Visualizer] plot_comparison (grads): semua grad_dict kosong. "
                      "Berikan x_ref dan y_ref untuk trigger backward pass.")
            else:
                ref_keys_g = self._resolve_layer_keys(
                    next(iter(gdicts.values())), layer_keys
                )
                n_cond   = len(gdicts)
                n_layers = len(ref_keys_g)
                fig, axes = plt.subplots(n_cond, n_layers,
                                          figsize=(5 * n_layers, 3.5 * n_cond),
                                          squeeze=False)
                fig.suptitle(f"{save_prefix} — Gradient Distribution",
                             fontsize=13, fontweight='bold')
 
                color_map = dict(zip(labels, colors))
                for row, lb in enumerate(gdicts):
                    c = color_map.get(lb, 'gray')
                    for col, k in enumerate(ref_keys_g):
                        ax = axes[row][col]
                        if k not in gdicts[lb]:
                            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=ax.transAxes)
                            continue
                        g = gdicts[lb][k]
                        ax.hist(g, bins=40, color=c, alpha=0.8, edgecolor='none')
                        ax.axvline(0, color='black', linewidth=0.8,
                                   linestyle='--', alpha=0.5)
                        self._annotate_stats(ax, g)
                        ax.set_title(f"{lb}\n{k}  dL/dW", fontsize=8)
                        ax.set_xlabel("Gradient value", fontsize=8)
                        ax.set_ylabel("Count", fontsize=8)
                        ax.grid(axis='y', alpha=0.3)
 
                plt.tight_layout()
                self._savefig(fig, f"{save_prefix}_grads.png")
 

    def plot_sklearn_comparison(
        self,
        y_true         : np.ndarray,
        y_pred_custom  : np.ndarray,
        y_pred_sklearn : np.ndarray,
        label_custom   : str = "FFNN (custom)",
        label_sklearn  : str = "sklearn MLP",
        save_path      : str | None = None,
        ) -> None:

        
        y_true    = np.asarray(y_true).ravel()
        y_pred_c  = np.asarray(y_pred_custom).ravel()
        y_pred_sk = np.asarray(y_pred_sklearn).ravel()
 
        def _metrics(y_t, y_p):
            return {
                'Accuracy' : accuracy_score(y_t, y_p),
                'Precision': precision_score(y_t, y_p, zero_division=0),
                'Recall'   : recall_score(y_t, y_p, zero_division=0),
                'F1 Score' : f1_score(y_t, y_p, zero_division=0),
            }
 
        mc = _metrics(y_true, y_pred_c)
        ms = _metrics(y_true, y_pred_sk)
 
        metric_names = list(mc.keys())
        vals_custom  = [mc[m] for m in metric_names]
        vals_sklearn = [ms[m] for m in metric_names]
 
        # layout: 1 baris, 3 kolom (bar chart | cm custom | cm sklearn)
        fig = plt.figure(figsize=(16, 5))
        gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)
        ax_bar = fig.add_subplot(gs[0])
        ax_cm1 = fig.add_subplot(gs[1])
        ax_cm2 = fig.add_subplot(gs[2])
 
        fig.suptitle("Perbandingan FFNN Custom vs sklearn MLP",
                     fontsize=13, fontweight='bold')
 
        # --- bar chart ---
        y_pos  = np.arange(len(metric_names))
        height = 0.35
        bars_c  = ax_bar.barh(y_pos + height / 2, vals_custom,
                               height, label=label_custom,  color='steelblue', alpha=0.85)
        bars_sk = ax_bar.barh(y_pos - height / 2, vals_sklearn,
                               height, label=label_sklearn, color='seagreen',  alpha=0.85)
 
        # anotasi nilai di ujung bar
        for bar in bars_c:
            v = bar.get_width()
            ax_bar.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va='center', fontsize=8)
        for bar in bars_sk:
            v = bar.get_width()
            ax_bar.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va='center', fontsize=8)
 
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(metric_names)
        ax_bar.set_xlim(0, 1.12)
        ax_bar.set_xlabel("Score")
        ax_bar.set_title("Metrics comparison")
        ax_bar.legend(fontsize=9)
        ax_bar.grid(axis='x', alpha=0.3)
 
        # --- confusion matrix ---
        cm_labels = np.unique(y_true)
        for ax, y_p, lbl, clr in [
            (ax_cm1, y_pred_c,  label_custom,  'Blues'),
            (ax_cm2, y_pred_sk, label_sklearn, 'Greens'),
        ]:
            cm = confusion_matrix(y_true, y_p, labels=cm_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=cm_labels)
            disp.plot(ax=ax, colorbar=False, cmap=clr)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
 
        self._savefig(fig, save_path)
