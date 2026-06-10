"""
TEMPORAL FUSION TRANSFORMER (TFT)
Processes time-series vital signs.
Architecture:
  Input (batch, 12, num_vitals)
    → Variable Selection Network (GRN)
    → LSTM encoder
    → Multi-head Self-Attention (pre-norm)
    → GRN post-attention
    → Output: sepsis risk logit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os
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
LR         = 1e-4      # conservative LR for stability
HIDDEN_DIM = 32        # smaller hidden dim — more stable on CPU
NUM_HEADS  = 4
DROPOUT    = 0.1
print(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────
# HELPER: sanitize tensor — replace NaN/Inf with 0
# Root cause fix: bad values in preprocessed .npy
# files cause all-NaN forward passes
# ──────────────────────────────────────────────
def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=3.0, neginf=-3.0)


# ──────────────────────────────────────────────
# 1. GATED RESIDUAL NETWORK (GRN)
#    Core building block of TFT
# ──────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate    = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim, eps=1e-6)
        self.skip    = nn.Linear(input_dim, output_dim) \
                       if input_dim != output_dim else nn.Identity()

        # Small weight init — prevents exploding activations at start
        for layer in [self.fc1, self.fc2, self.gate]:
            nn.init.xavier_uniform_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        residual = self.skip(x)
        h   = torch.relu(self.fc1(x))
        h   = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.norm(out + residual)


# ──────────────────────────────────────────────
# 2. VARIABLE SELECTION NETWORK (VSN)
#    Learns which vital signs matter most
# ──────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.grn     = GatedResidualNetwork(num_features, hidden_dim,
                                            num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, time, features)
        # Flatten to 2D so GRN processes each timestep independently
        B, T, F  = x.shape
        x_flat   = x.reshape(B * T, F)
        w_flat   = self.softmax(self.grn(x_flat))
        weights  = w_flat.reshape(B, T, F)
        return x * weights, weights


# ──────────────────────────────────────────────
# 3. TEMPORAL FUSION TRANSFORMER
# ──────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()

        # Step A: Variable selection
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)

        # Step B: Linear projection + LayerNorm before LSTM
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Step C: LSTM — captures local temporal patterns
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                            batch_first=True, num_layers=1)

        # Step D: Transformer with pre-norm (more stable than post-norm)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True   # pre-norm prevents NaN in self-attention
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        # Step E: GRN refinement
        self.post_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout)

        # Step F: Classifier — no Sigmoid (BCEWithLogitsLoss handles it)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Sanitize at entry — kills any stray NaN/Inf in input
        x = sanitize(x)

        # Variable selection
        x_sel, vsn_w = self.vsn(x)

        # Project + normalize
        proj = self.input_norm(self.input_proj(x_sel))

        # LSTM
        lstm_out, _ = self.lstm(proj)

        # Transformer
        attn_out = self.transformer(lstm_out)

        # GRN on transformer output — take last time step
        refined = self.post_grn(attn_out)
        last    = refined[:, -1, :]

        # Logit output (no sigmoid — BCEWithLogitsLoss handles it)
        logit = self.classifier(last).squeeze(-1)
        return logit, vsn_w


# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────
def train_tft(model, train_loader, val_loader, epochs, lr):
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
            torch.save(model.state_dict(), f"{MODEL_DIR}/tft_best.pt")
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
# 5. EVALUATION
# ──────────────────────────────────────────────
def evaluate_tft(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b    = sanitize(X_b).to(DEVICE)
            logits, _ = model(X_b)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    print("\n" + "=" * 40)
    print("  TFT TEST RESULTS")
    print("=" * 40)
    print(f"  ROC-AUC: {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['No Sepsis', 'Sepsis']))
    return np.array(all_probs), np.array(all_labels)


def plot_training_curves(history):
    if not history['train_loss']:
        print("  No training history to plot.")
        return
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('TFT Training History', fontweight='bold')

    axes[0].plot(epochs_range, history['train_loss'],
                 label='Train Loss', color='steelblue')
    axes[0].plot(epochs_range, history['val_loss'],
                 label='Val Loss',   color='tomato')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['val_auc'],
                 label='Val AUC', color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/tft_training_curves.png", dpi=150)
    plt.show()
    print(f"  Plot saved → {MODEL_DIR}/tft_training_curves.png")


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  STEP 3: TFT MODEL TRAINING")
    print("=" * 50)

    # ── Load numpy arrays ──
    X_seq_train = np.load(f"{PROCESSED_DIR}/X_seq_train.npy")
    X_seq_val   = np.load(f"{PROCESSED_DIR}/X_seq_val.npy")
    X_seq_test  = np.load(f"{PROCESSED_DIR}/X_seq_test.npy")
    y_train     = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_val       = np.load(f"{PROCESSED_DIR}/y_val.npy")
    y_test      = np.load(f"{PROCESSED_DIR}/y_test.npy")

    # ── DATA SANITY CHECK & CLEAN ──
    # This is the root cause: stray NaN/Inf in the .npy files
    # cause every single forward pass to output NaN
    print("\n[Data check]")
    for name, arr in [("X_seq_train", X_seq_train),
                      ("X_seq_val",   X_seq_val),
                      ("X_seq_test",  X_seq_test)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"  {name}: shape={arr.shape}  NaN={n_nan}  Inf={n_inf}")

    # Fix: replace NaN/Inf and clip extreme outliers (>5 std after scaling)
    def clean(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
        arr = np.clip(arr, -5.0, 5.0)
        return arr.astype(np.float32)

    X_seq_train = clean(X_seq_train)
    X_seq_val   = clean(X_seq_val)
    X_seq_test  = clean(X_seq_test)

    print(f"\n  After clean — NaN remaining: "
          f"{np.isnan(X_seq_train).sum() + np.isnan(X_seq_val).sum()}")

    # ── Convert to tensors ──
    X_seq_train = torch.FloatTensor(X_seq_train)
    X_seq_val   = torch.FloatTensor(X_seq_val)
    X_seq_test  = torch.FloatTensor(X_seq_test)
    y_train_t   = torch.FloatTensor(y_train)
    y_val_t     = torch.FloatTensor(y_val)
    y_test_t    = torch.FloatTensor(y_test)

    input_dim = X_seq_train.shape[2]
    print(f"\n  Input shape : {X_seq_train.shape}")
    print(f"  Input dim   : {input_dim}")
    print(f"  Sepsis rate : {y_train_t.mean().item()*100:.1f}% (train)")

    # ── DataLoaders ──
    train_loader = DataLoader(TensorDataset(X_seq_train, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_seq_val,   y_val_t),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_seq_test,  y_test_t),
                              batch_size=BATCH_SIZE)

    # ── Build model ──
    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=2,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"  TFT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ──
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = train_tft(model, train_loader, val_loader, EPOCHS, LR)
    plot_training_curves(history)

    # ── Evaluate ──
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/tft_best.pt", map_location=DEVICE))
    tft_test_probs, _ = evaluate_tft(model, test_loader)

    # ── Save val predictions for fusion layer ──
    model.eval()
    val_probs = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            logits, _ = model(sanitize(X_b).to(DEVICE))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy())

    np.save(f"{PROCESSED_DIR}/tft_val_probs.npy",  np.array(val_probs))
    np.save(f"{PROCESSED_DIR}/tft_test_probs.npy", tft_test_probs)
    print("\n✅ TFT complete! Predictions saved to ./processed/")


if __name__ == "__main__":
    main()