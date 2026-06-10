"""
STEP 5: ATTENTION-BASED FUSION LAYER + FINAL EVALUATION
=========================================================
Combines TFT and TabNet predictions intelligently:
- Learns how much to trust each model (dynamic weights)
- Produces final sepsis risk score
- Generates comprehensive evaluation metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = "./processed"
MODEL_DIR     = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 1. ATTENTION FUSION LAYER
#    Learns to weigh TFT vs TabNet dynamically
#    Instead of simple averaging: 0.5*TFT + 0.5*TabNet
#    It learns: w_tft * TFT + w_tabnet * TabNet
#    where weights are learned from both predictions together
# ──────────────────────────────────────────────
class AttentionFusionLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()

        # Context-aware weight network
        # Input: [tft_prob, tabnet_prob] → 2 features
        self.attention_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)             # outputs [w_tft, w_tabnet], sum=1
        )

        # Refinement layer after fusion
        # No Sigmoid — BCEWithLogitsLoss handles it during training
        # We apply sigmoid manually at evaluation/inference time
        self.fusion_fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tft_prob, tabnet_prob):
        """
        tft_prob    : (batch,) TFT predictions
        tabnet_prob : (batch,) TabNet predictions
        """
        # Stack predictions into (batch, 2)
        combined = torch.stack([tft_prob, tabnet_prob], dim=1)

        # Compute attention weights
        weights = self.attention_net(combined)   # (batch, 2)
        w_tft, w_tabnet = weights[:, 0], weights[:, 1]

        # Weighted combination
        fused = w_tft * tft_prob + w_tabnet * tabnet_prob   # (batch,)

        # Refine through a small network
        final = self.fusion_fc(fused.unsqueeze(1)).squeeze(1)  # (batch,)

        return final, weights


# ──────────────────────────────────────────────
# 2. TRAIN FUSION LAYER
# ──────────────────────────────────────────────
def train_fusion(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # FIX: pos_weight penalises missed sepsis 8x — same as TFT/TabNet
    # Without this, fusion learns to be conservative and pulls High scores toward Moderate
    pos_weight = torch.tensor([8.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0
    history  = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for tft_b, tab_b, y_b in train_loader:
            tft_b, tab_b, y_b = tft_b.to(DEVICE), tab_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(tft_b, tab_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for tft_b, tab_b, y_b in val_loader:
                tft_b, tab_b = tft_b.to(DEVICE), tab_b.to(DEVICE)
                logits, _ = model(tft_b, tab_b)
                val_loss += criterion(logits, y_b.to(DEVICE)).item()
                preds = torch.sigmoid(logits)   # sigmoid only for AUC evaluation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_b.numpy())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        val_auc   = roc_auc_score(all_labels, all_preds)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_auc'].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{MODEL_DIR}/fusion_best.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | "
                  f"Val Loss: {avg_val:.4f} | Val AUC: {val_auc:.4f}")

    print(f"\n  Best Fusion Val AUC: {best_auc:.4f}")
    return history


# ──────────────────────────────────────────────
# 3. COMPREHENSIVE EVALUATION PLOTS
# ──────────────────────────────────────────────
def full_evaluation(y_true, tft_probs, tabnet_probs, fusion_probs):
    """
    Generate complete evaluation dashboard:
    - ROC curves (all 3 models)
    - Precision-Recall curves
    - Confusion matrix
    - Risk score distribution
    - Model weight analysis
    """
    threshold = 0.60  # FIX: aligned with dashboard High Risk boundary

    # Binary predictions
    tft_preds    = (tft_probs    > threshold).astype(int)
    tabnet_preds = (tabnet_probs > threshold).astype(int)
    fusion_preds = (fusion_probs > threshold).astype(int)

    # AUC scores
    auc_tft    = roc_auc_score(y_true, tft_probs)
    auc_tabnet = roc_auc_score(y_true, tabnet_probs)
    auc_fusion = roc_auc_score(y_true, fusion_probs)
    ap_fusion  = average_precision_score(y_true, fusion_probs)

    print("\n" + "=" * 55)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  TFT ROC-AUC    : {auc_tft:.4f}")
    print(f"  TabNet ROC-AUC : {auc_tabnet:.4f}")
    print(f"  Fusion ROC-AUC : {auc_fusion:.4f}  ← Best")
    print(f"  Avg Precision  : {ap_fusion:.4f}")
    print("\n  Fusion Model Classification Report:")
    print(classification_report(y_true, fusion_preds,
                                target_names=['No Sepsis', 'Sepsis']))

    # ── FIGURE 1: ROC + PR + Confusion + Distribution ──
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Sepsis Prediction System — Final Evaluation Dashboard',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for probs, label, color, auc in [
        (tft_probs,    f'TFT    (AUC={auc_tft:.3f})',    'steelblue', auc_tft),
        (tabnet_probs, f'TabNet (AUC={auc_tabnet:.3f})', 'tomato',    auc_tabnet),
        (fusion_probs, f'Fusion (AUC={auc_fusion:.3f})', 'green',     auc_fusion),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax1.plot(fpr, tpr, label=label, color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Precision-Recall Curves
    ax2 = fig.add_subplot(gs[0, 1])
    for probs, label, color in [
        (tft_probs,    'TFT',    'steelblue'),
        (tabnet_probs, 'TabNet', 'tomato'),
        (fusion_probs, 'Fusion', 'green'),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ax2.plot(rec, prec, label=label, color=color, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Fusion Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, fusion_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: No Sepsis', 'Pred: Sepsis'],
                yticklabels=['True: No Sepsis', 'True: Sepsis'],
                ax=ax3, cbar=False, annot_kws={"size": 12})
    ax3.set_title('Fusion Confusion Matrix', fontweight='bold')

    # Panel 4: Risk Score Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(fusion_probs[y_true == 0], bins=40, alpha=0.6,
             color='steelblue', label='No Sepsis', density=True)
    ax4.hist(fusion_probs[y_true == 1], bins=40, alpha=0.6,
             color='tomato', label='Sepsis', density=True)
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax4.set_xlabel('Predicted Sepsis Risk Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Risk Score Distribution', fontweight='bold')
    ax4.legend()

    # Panel 5: AUC Comparison Bar Chart
    ax5 = fig.add_subplot(gs[1, 1])
    models = ['TFT', 'TabNet', 'Fusion']
    aucs   = [auc_tft, auc_tabnet, auc_fusion]
    colors = ['steelblue', 'tomato', 'green']
    bars = ax5.bar(models, aucs, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, aucs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax5.set_ylim(0.5, 1.0)
    ax5.set_ylabel('ROC-AUC')
    ax5.set_title('Model AUC Comparison', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Metrics Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    from sklearn.metrics import recall_score, precision_score, f1_score

    metrics_data = []
    for name, preds, probs in [
        ('TFT',    tft_preds,    tft_probs),
        ('TabNet', tabnet_preds, tabnet_probs),
        ('Fusion', fusion_preds, fusion_probs),
    ]:
        recall    = recall_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        f1        = f1_score(y_true, preds, zero_division=0)
        auc       = roc_auc_score(y_true, probs)
        metrics_data.append([name, f'{auc:.3f}', f'{recall:.3f}',
                              f'{precision:.3f}', f'{f1:.3f}'])

    table = ax6.table(
        cellText=metrics_data,
        colLabels=['Model', 'AUC', 'Recall', 'Precision', 'F1'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    # Highlight fusion row
    for col in range(5):
        table[3, col].set_facecolor('#90EE90')
    ax6.set_title('Metrics Summary', fontweight='bold', pad=20)

    plt.savefig(f"{MODEL_DIR}/final_evaluation_dashboard.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Evaluation dashboard saved!")

    return {
        'auc_tft': auc_tft, 'auc_tabnet': auc_tabnet, 'auc_fusion': auc_fusion,
        'ap_fusion': ap_fusion
    }


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 5: ATTENTION FUSION + FINAL EVALUATION")
    print("=" * 55)

    # FIX: Load proper val predictions (for fusion training) and
    # test predictions (for final unbiased evaluation) — no data leakage
    tft_val_probs    = np.load(f"{PROCESSED_DIR}/tft_val_probs.npy")
    tabnet_val_probs = np.load(f"{PROCESSED_DIR}/tabnet_val_probs.npy")
    tft_test_probs   = np.load(f"{PROCESSED_DIR}/tft_test_probs.npy")
    tabnet_test_probs= np.load(f"{PROCESSED_DIR}/tabnet_test_probs.npy")
    y_val            = np.load(f"{PROCESSED_DIR}/y_val.npy")
    y_test           = np.load(f"{PROCESSED_DIR}/y_test.npy")

    print(f"  Fusion train (val set) : {tft_val_probs.shape[0]} samples")
    print(f"  Final test set         : {tft_test_probs.shape[0]} samples")

    # Create DataLoaders
    # Fusion trains on val predictions, evaluates on test predictions
    def make_loader(tft, tab, y, shuffle=False):
        ds = TensorDataset(
            torch.FloatTensor(tft),
            torch.FloatTensor(tab),
            torch.FloatTensor(y)
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle)

    # 80/20 split of val set for fusion train/val
    n_val    = len(y_val)
    n_ftrain = int(0.8 * n_val)
    indices  = np.random.permutation(n_val)
    tr_idx, vl_idx = indices[:n_ftrain], indices[n_ftrain:]

    ftrain_loader = make_loader(tft_val_probs[tr_idx],
                                tabnet_val_probs[tr_idx],
                                y_val[tr_idx], shuffle=True)
    fval_loader   = make_loader(tft_val_probs[vl_idx],
                                tabnet_val_probs[vl_idx],
                                y_val[vl_idx])
    test_loader   = make_loader(tft_test_probs, tabnet_test_probs, y_test)

    # Train fusion layer
    fusion_model = AttentionFusionLayer(hidden_dim=32).to(DEVICE)
    print(f"\n  Training Attention Fusion Layer...")
    history = train_fusion(fusion_model, ftrain_loader, fval_loader,
                           epochs=50, lr=1e-3)

    # Load best fusion model and run final test evaluation
    fusion_model.load_state_dict(
        torch.load(f"{MODEL_DIR}/fusion_best.pt", map_location=DEVICE))
    fusion_model.eval()

    # Get final test predictions
    all_fusion_probs = []
    all_weights      = []
    with torch.no_grad():
        for tft_b, tab_b, _ in test_loader:
            tft_b, tab_b = tft_b.to(DEVICE), tab_b.to(DEVICE)
            logits, weights = fusion_model(tft_b, tab_b)
            probs = torch.sigmoid(logits)
            all_fusion_probs.extend(probs.cpu().numpy())
            all_weights.extend(weights.cpu().numpy())

    fusion_probs = np.array(all_fusion_probs)
    all_weights  = np.array(all_weights)

    # Save final fusion predictions
    np.save(f"{PROCESSED_DIR}/fusion_probs.npy", fusion_probs)

    # Print average model weights
    avg_weights = all_weights.mean(axis=0)
    print(f"\n  Average Attention Weights:")
    print(f"    TFT    weight: {avg_weights[0]:.3f}")
    print(f"    TabNet weight: {avg_weights[1]:.3f}")

    # Full evaluation on clean test set
    metrics = full_evaluation(y_test, tft_test_probs, tabnet_test_probs, fusion_probs)

    print("\n" + "=" * 55)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"  TFT    AUC : {metrics['auc_tft']:.4f}")
    print(f"  TabNet AUC : {metrics['auc_tabnet']:.4f}")
    print(f"  FUSION AUC : {metrics['auc_fusion']:.4f} ← Final Model")
    print(f"  Avg Prec   : {metrics['ap_fusion']:.4f}")
    print("\n✅ Fusion model training and evaluation complete!")


if __name__ == "__main__":
    main()