"""
 TABNET MODEL
Processes structured tabular clinical data
(aggregated lab values, demographics, vital statistics)

TabNet uses sequential attention to:
- Automatically select most important features at each step
- Learn complex feature interactions
- Provide feature importance scores for interpretability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import pickle, os
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
PROCESSED_DIR = "./processed"
MODEL_DIR     = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-4      # conservative LR — same as TFT
N_STEPS    = 5         # number of sequential attention steps
N_D        = 32        # feature processing dimension
N_A        = 32        # attention embedding dimension
GAMMA      = 1.5       # relaxation factor (controls feature reuse between steps)
print(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────
# HELPER: sanitize — replace NaN/Inf with 0
# ──────────────────────────────────────────────
def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=3.0, neginf=-3.0)


# ──────────────────────────────────────────────
# TABNET MODEL
# How it works (one step at a time):
#   1. Attention layer selects which features to focus on
#   2. Feature transformer processes those selected features
#   3. Output accumulates across all N_STEPS
#   4. Mask history controls feature reuse via prior_scales
# ──────────────────────────────────────────────
class TabNet(nn.Module):
    def __init__(self, input_dim, n_d=32, n_a=32, n_steps=5,
                 gamma=1.5, dropout=0.1):
        super().__init__()
        self.n_steps   = n_steps
        self.n_a       = n_a
        self.n_d       = n_d
        self.input_dim = input_dim
        self.gamma     = gamma

        # ── Input batch normalisation ──
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.02)

        # ── Shared feature transformer (applied before each step) ──
        # Input → n_d*2, then GLU halves it → n_d
        self.shared_fc  = nn.Linear(input_dim, n_d * 2, bias=False)
        self.shared_bn  = nn.BatchNorm1d(n_d * 2, momentum=0.02)

        # ── Per-step layers ──
        # Attention: (n_a) → (input_dim)  — produces feature mask
        self.attention_fc = nn.ModuleList([
            nn.Linear(n_a, input_dim, bias=False) for _ in range(n_steps)
        ])
        self.attention_bn = nn.ModuleList([
            nn.BatchNorm1d(input_dim, momentum=0.02) for _ in range(n_steps)
        ])

        # Step transformer:
        #   step_fc1 : input_dim → n_d * 2  (then GLU → n_d)
        #   step_fc2 : n_d       → n_d + n_a (then split: n_d output + n_a for next attn)
        # FIX: step_fc2 input is n_d (post-GLU), NOT n_d*2
        self.step_fc1 = nn.ModuleList([
            nn.Linear(input_dim, n_d * 2, bias=False) for _ in range(n_steps)
        ])
        self.step_bn1 = nn.ModuleList([
            nn.BatchNorm1d(n_d * 2, momentum=0.02) for _ in range(n_steps)
        ])
        self.step_fc2 = nn.ModuleList([
            nn.Linear(n_d, n_d + n_a, bias=False) for _ in range(n_steps)
        ])                                           # ← n_d input (post-GLU), not n_d*2
        self.step_bn2 = nn.ModuleList([
            nn.BatchNorm1d(n_d + n_a, momentum=0.02) for _ in range(n_steps)
        ])

        self.dropout = nn.Dropout(dropout)

        # ── Classifier ──
        # Input = n_d (accumulated step outputs averaged)
        # No Sigmoid — BCEWithLogitsLoss handles it
        self.classifier = nn.Sequential(
            nn.Linear(n_d, n_d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_d // 2, 1)
        )

    def forward(self, x):
        # Sanitize input
        x = sanitize(x)                              # (batch, input_dim)

        # Normalize input
        x_norm = self.initial_bn(x)                  # (batch, input_dim)

        # Initial attention context — zeros at step 0
        h_attn = torch.zeros(x.size(0), self.n_a,
                             device=x.device)        # (batch, n_a)

        # Prior scales — starts at 1, penalizes reused features
        prior_scales = torch.ones(x.size(0), self.input_dim,
                                  device=x.device)   # (batch, input_dim)

        aggregated = torch.zeros(x.size(0), self.n_d,
                                 device=x.device)    # (batch, n_d)
        all_masks  = []

        for step in range(self.n_steps):

            # ── Attention: which features matter this step? ──
            a    = self.attention_bn[step](
                       self.attention_fc[step](h_attn))  # (batch, input_dim)
            a    = a * prior_scales
            mask = torch.softmax(a, dim=-1)              # sparse feature weights
            all_masks.append(mask)

            # ── Masked input ──
            masked_x = mask * x_norm                     # (batch, input_dim)

            # ── Feature transformer ──
            # step_fc1: input_dim → n_d*2
            feat = self.step_bn1[step](
                       self.step_fc1[step](masked_x))    # (batch, n_d*2)

            # GLU activation: splits in half, gates with sigmoid
            # output shape: (batch, n_d)
            feat = (feat[:, :self.n_d] *
                    torch.sigmoid(feat[:, self.n_d:]))   # (batch, n_d)

            # step_fc2: n_d → n_d + n_a   ← FIX: input is n_d (post-GLU)
            feat = self.step_bn2[step](
                       self.step_fc2[step](feat))        # (batch, n_d + n_a)

            # Split: step output (n_d) + next attention context (n_a)
            step_out = torch.relu(feat[:, :self.n_d])    # (batch, n_d)
            h_attn   = feat[:, self.n_d:]                # (batch, n_a)

            # Accumulate
            aggregated = aggregated + step_out

            # Update prior scales — reduce weight of already-used features
            prior_scales = prior_scales * (self.gamma - mask)

        # Average over steps
        final = self.dropout(aggregated / self.n_steps)  # (batch, n_d)

        # Classify
        logit = self.classifier(final).squeeze(-1)       # (batch,)

        # Feature importance = mean attention mask across all steps
        importance = torch.stack(all_masks, dim=0).mean(dim=0)  # (batch, input_dim)

        return logit, importance


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_tabnet(model, train_loader, val_loader, epochs, lr):
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=5, factor=0.5)
    pos_weight = torch.tensor([8.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history      = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_auc     = 0.0
    patience_cnt = 0
    EARLY_STOP   = 10

    for epoch in range(epochs):

        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for X_b, y_b in train_loader:
            X_b = sanitize(X_b).to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            if torch.isnan(logits).any():
                continue
            loss = criterion(logits, y_b)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        # ── Validate ──
        model.eval()
        val_loss   = 0.0
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = sanitize(X_b).to(DEVICE)
                logits, _ = model(X_b)
                if torch.isnan(logits).any():
                    continue
                val_loss += criterion(logits, y_b.to(DEVICE)).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_b.numpy())

        if len(all_probs) == 0 or len(set(all_labels)) < 2:
            print(f"  Epoch {epoch+1:03d} — skipped (no valid predictions)")
            continue

        avg_train = train_loss / max(n_batches, 1)
        avg_val   = val_loss   / len(val_loader)
        val_auc   = roc_auc_score(all_labels, all_probs)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_auc'].append(val_auc)
        scheduler.step(avg_val)

        if val_auc > best_auc:
            best_auc     = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/tabnet_best.pt")
        else:
            patience_cnt += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {avg_val:.4f} | "
                  f"Val AUC: {val_auc:.4f}")

        if patience_cnt >= EARLY_STOP:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n  Best Val AUC: {best_auc:.4f}")
    return history


# ──────────────────────────────────────────────
# EVALUATION + FEATURE IMPORTANCE PLOT
# ──────────────────────────────────────────────
def evaluate_tabnet(model, test_loader, tab_cols):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_importances = []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = sanitize(X_b).to(DEVICE)
            logits, importance = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
            all_importances.append(importance.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    print("\n" + "=" * 40)
    print("  TABNET TEST RESULTS")
    print("=" * 40)
    print(f"  ROC-AUC: {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['No Sepsis', 'Sepsis']))

    # ── Feature importance plot (top 15) ──
    avg_imp  = np.concatenate(all_importances, axis=0).mean(axis=0)
    top_k    = min(15, len(avg_imp))
    top_idx  = np.argsort(avg_imp)[::-1][:top_k]
    top_feats = [tab_cols[i] if i < len(tab_cols) else f"feat_{i}"
                 for i in top_idx]
    top_scores = avg_imp[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_k))[::-1]
    ax.barh(range(top_k), top_scores, color=colors, edgecolor='white')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Average Attention Score')
    ax.set_title('Top 15 Feature Importances (TabNet)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/tabnet_feature_importance.png", dpi=150)
    plt.show()
    print(f"  Plot saved → {MODEL_DIR}/tabnet_feature_importance.png")

    return np.array(all_probs), np.array(all_labels)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  STEP 4: TABNET MODEL TRAINING")
    print("=" * 50)

    # ── Load data ──
    X_tab_train = np.load(f"{PROCESSED_DIR}/X_tab_train.npy")
    X_tab_val   = np.load(f"{PROCESSED_DIR}/X_tab_val.npy")
    X_tab_test  = np.load(f"{PROCESSED_DIR}/X_tab_test.npy")
    y_train     = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_val       = np.load(f"{PROCESSED_DIR}/y_val.npy")
    y_test      = np.load(f"{PROCESSED_DIR}/y_test.npy")

    with open(f"{PROCESSED_DIR}/tab_cols.pkl", 'rb') as f:
        tab_cols = pickle.load(f)

    # ── Data sanity check & clean ──
    print("\n[Data check]")
    for name, arr in [("X_tab_train", X_tab_train),
                      ("X_tab_val",   X_tab_val),
                      ("X_tab_test",  X_tab_test)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"  {name}: shape={arr.shape}  NaN={n_nan}  Inf={n_inf}")

    def clean(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
        arr = np.clip(arr, -5.0, 5.0)
        return arr.astype(np.float32)

    X_tab_train = clean(X_tab_train)
    X_tab_val   = clean(X_tab_val)
    X_tab_test  = clean(X_tab_test)

    input_dim = X_tab_train.shape[1]
    print(f"\n  Input dim   : {input_dim}")
    print(f"  Sepsis rate : {y_train.mean()*100:.1f}% (train)")

    # ── Convert to tensors ──
    X_tab_train = torch.FloatTensor(X_tab_train)
    X_tab_val   = torch.FloatTensor(X_tab_val)
    X_tab_test  = torch.FloatTensor(X_tab_test)
    y_train_t   = torch.FloatTensor(y_train)
    y_val_t     = torch.FloatTensor(y_val)
    y_test_t    = torch.FloatTensor(y_test)

    # ── DataLoaders ──
    # drop_last=True prevents BatchNorm1d crash on single-sample last batch
    train_loader = DataLoader(TensorDataset(X_tab_train, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_tab_val, y_val_t),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_tab_test, y_test_t),
                              batch_size=BATCH_SIZE)

    # ── Build model ──
    model = TabNet(
        input_dim=input_dim,
        n_d=N_D, n_a=N_A,
        n_steps=N_STEPS,
        gamma=GAMMA
    ).to(DEVICE)
    print(f"  TabNet Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ──
    print(f"\nTraining for up to {EPOCHS} epochs...")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('TabNet Training History', fontweight='bold')

    history = train_tabnet(model, train_loader, val_loader, EPOCHS, LR)

    if history['train_loss']:
        er = range(1, len(history['train_loss']) + 1)
        axes[0].plot(er, history['train_loss'], label='Train Loss', color='steelblue')
        axes[0].plot(er, history['val_loss'],   label='Val Loss',   color='tomato')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(er, history['val_auc'],    label='Val AUC',    color='green')
        axes[1].set_title('AUC'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f"{MODEL_DIR}/tabnet_training_curves.png", dpi=150)
        plt.show()
        print(f"  Plot saved → {MODEL_DIR}/tabnet_training_curves.png")

    # ── Evaluate ──
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/tabnet_best.pt", map_location=DEVICE))
    tabnet_test_probs, _ = evaluate_tabnet(model, test_loader, tab_cols)

    # ── Save val predictions for fusion layer ──
    model.eval()
    val_probs = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            logits, _ = model(sanitize(X_b).to(DEVICE))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy())

    np.save(f"{PROCESSED_DIR}/tabnet_val_probs.npy",  np.array(val_probs))
    np.save(f"{PROCESSED_DIR}/tabnet_test_probs.npy", tabnet_test_probs)
    print("\n✅ TabNet complete! Predictions saved to ./processed/")


if __name__ == "__main__":
    main()
