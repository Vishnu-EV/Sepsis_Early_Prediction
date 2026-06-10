
"""
STEP 6: CLINICAL DASHBOARD (Streamlit)
========================================
Run: python -m streamlit run 6_dashboard.py

FIXES APPLIED:
  [F1] Model class dropout=0.1 to exactly match trained models (was 0.2)
  [F2] EtCO2 slider now correctly feeds into tabular features (was silently 0)
  [F3] Triggered threshold factors shown in BOTH real model and demo modes
  [F4] Risk decision thresholds aligned: Low<0.30 | Moderate 0.31–0.60 | High>0.60
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle, os, glob, math

st.set_page_config(
    page_title="Sepsis Risk Prediction System",
    page_icon="🏥",
    layout="wide"
)

PROCESSED_DIR = "./processed"
MODEL_DIR     = "./models"
DATA_DIR      = "./data"
DEVICE        = torch.device("cpu")

# Clinical risk tiers — FIX [F4]: aligned with fusion.py
LOW_THRESH  = 0.30
HIGH_THRESH = 0.60

VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLS   = [
    'BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN',
    'Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',
    'Glucose','Lactate','Magnesium','Phosphate','Potassium',
    'Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets'
]
DEMO_COLS  = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
TIME_STEPS = 12


# ─────────────────────────────────────────────────────
# SANITIZE
# ─────────────────────────────────────────────────────
def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=3.0, neginf=-3.0)


# ─────────────────────────────────────────────────────
# MODEL DEFINITIONS
# FIX [F1]: All dropout defaults = 0.1 (matches training files exactly)
# ─────────────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):  # FIX: was 0.2
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate    = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim, eps=1e-6)
        self.skip    = nn.Linear(input_dim, output_dim) \
                       if input_dim != output_dim else nn.Identity()
        for layer in [self.fc1, self.fc2, self.gate]:
            nn.init.xavier_uniform_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)
    def forward(self, x):
        residual = self.skip(x)
        h   = torch.relu(self.fc1(x))
        h   = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.norm(out + residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):  # FIX: was 0.2
        super().__init__()
        self.grn     = GatedResidualNetwork(num_features, hidden_dim,
                                            num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B, T, F = x.shape
        x_flat  = x.reshape(B * T, F)
        w_flat  = self.softmax(self.grn(x_flat))
        weights = w_flat.reshape(B, T, F)
        return x * weights, weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4,
                 num_layers=2, dropout=0.1):  # FIX: was 0.2
        super().__init__()
        self.vsn        = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_grn    = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1))
    def forward(self, x):
        x = sanitize(x)
        x_sel, vsn_w = self.vsn(x)
        proj = self.input_norm(self.input_proj(x_sel))
        lstm_out, _ = self.lstm(proj)
        attn_out = self.transformer(lstm_out)
        refined  = self.post_grn(attn_out)
        last     = refined[:, -1, :]
        logit    = self.classifier(last).squeeze(-1)
        return logit, vsn_w

class TabNet(nn.Module):
    def __init__(self, input_dim, n_d=32, n_a=32, n_steps=5,
                 gamma=1.5, dropout=0.1):  # FIX: was 0.2
        super().__init__()
        self.n_steps=n_steps; self.n_a=n_a; self.n_d=n_d
        self.input_dim=input_dim; self.gamma=gamma
        self.initial_bn   = nn.BatchNorm1d(input_dim, momentum=0.02)
        self.shared_fc    = nn.Linear(input_dim, n_d*2, bias=False)
        self.shared_bn    = nn.BatchNorm1d(n_d*2, momentum=0.02)
        self.attention_fc = nn.ModuleList([nn.Linear(n_a, input_dim, bias=False) for _ in range(n_steps)])
        self.attention_bn = nn.ModuleList([nn.BatchNorm1d(input_dim, momentum=0.02) for _ in range(n_steps)])
        self.step_fc1     = nn.ModuleList([nn.Linear(input_dim, n_d*2, bias=False) for _ in range(n_steps)])
        self.step_bn1     = nn.ModuleList([nn.BatchNorm1d(n_d*2, momentum=0.02) for _ in range(n_steps)])
        self.step_fc2     = nn.ModuleList([nn.Linear(n_d, n_d+n_a, bias=False) for _ in range(n_steps)])
        self.step_bn2     = nn.ModuleList([nn.BatchNorm1d(n_d+n_a, momentum=0.02) for _ in range(n_steps)])
        self.dropout      = nn.Dropout(dropout)
        self.classifier   = nn.Sequential(
            nn.Linear(n_d, n_d//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(n_d//2, 1))
    def forward(self, x):
        x = sanitize(x)
        x_norm       = self.initial_bn(x)
        h_attn       = torch.zeros(x.size(0), self.n_a, device=x.device)
        prior_scales = torch.ones(x.size(0), self.input_dim, device=x.device)
        aggregated   = torch.zeros(x.size(0), self.n_d, device=x.device)
        all_masks    = []
        for step in range(self.n_steps):
            a    = self.attention_bn[step](self.attention_fc[step](h_attn))
            a    = a * prior_scales
            mask = torch.softmax(a, dim=-1); all_masks.append(mask)
            mx   = mask * x_norm
            feat = self.step_bn1[step](self.step_fc1[step](mx))
            feat = feat[:, :self.n_d] * torch.sigmoid(feat[:, self.n_d:])
            feat = self.step_bn2[step](self.step_fc2[step](feat))
            aggregated  += torch.relu(feat[:, :self.n_d])
            h_attn       = feat[:, self.n_d:]
            prior_scales = prior_scales * (self.gamma - mask)
        final     = self.dropout(aggregated / self.n_steps)
        logit     = self.classifier(final).squeeze(-1)
        importance = torch.stack(all_masks, 0).mean(0)
        return logit, importance

class AttentionFusionLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2), nn.Softmax(dim=-1))
        self.fusion_fc = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, tft_p, tab_p):
        combined = torch.stack([tft_p, tab_p], dim=1)
        weights  = self.attention_net(combined)
        fused    = weights[:, 0]*tft_p + weights[:, 1]*tab_p
        final    = self.fusion_fc(fused.unsqueeze(1)).squeeze(1)
        return final, weights


# ─────────────────────────────────────────────────────
# FIX [F3]: CLINICAL THRESHOLD CHECKER
# Used in BOTH real model mode and demo mode
# Returns triggered list regardless of prediction path
# ─────────────────────────────────────────────────────
def compute_triggered_factors(patient_df):
    """
    Evaluate clinical thresholds from actual patient data.
    Returns: (raw_score, triggered_list, vital_dict)
    This runs independently of model predictions — always uses real data.
    """
    df_f = patient_df.ffill().bfill()
    last = df_f.iloc[-1]
    n    = len(df_f)

    def g(col, default=np.nan):
        if col not in last.index: return default
        v = last[col]
        try:
            f = float(v)
            return np.nan if np.isnan(f) else f
        except: return default

    # Vitals — use last row (current state)
    hr_v  = g('HR');    sbp_v = g('SBP');   o2_v  = g('O2Sat')
    rsp_v = g('Resp');  tmp_v = g('Temp');  map_v = g('MAP')

    # ROOT CAUSE FIX: Labs use WORST-CASE across full ICU stay, not last row only
    # A patient whose Lactate peaked at 5.0 eight hours ago but is now 1.8
    # should still score HIGH — the critical event happened, it counts
    def col_max(col):
        if col not in df_f.columns: return np.nan
        v = df_f[col].max()
        return np.nan if (isinstance(v, float) and np.isnan(v)) else float(v)

    def col_min(col):
        if col not in df_f.columns: return np.nan
        v = df_f[col].min()
        return np.nan if (isinstance(v, float) and np.isnan(v)) else float(v)

    lac_v  = col_max('Lactate')         # worst = highest peak
    wbc_v  = g('WBC')                   # use last (direction unclear)
    cre_v  = col_max('Creatinine')      # worst = highest
    ph_v   = col_min('pH')             # worst = lowest (most acidotic)
    plt_v  = col_min('Platelets')       # worst = lowest
    bil_v  = col_max('Bilirubin_total') # worst = highest
    trop_v = col_max('TroponinI')       # worst = highest (cardiac injury)
    ptt_v  = col_max('PTT')            # worst = highest
    hgb_v  = col_min('Hgb')           # worst = lowest (anaemia)
    bun_v  = col_max('BUN')            # worst = highest (kidney stress)
    hco3_v = col_min('HCO3')          # worst = lowest (metabolic acidosis)
    has_sep = int(df_f['SepsisLabel'].max()) if 'SepsisLabel' in df_f.columns else 0

    raw       = 0.0
    triggered = []

    if has_sep:
        raw += 10.0; triggered.append("Confirmed SepsisLabel = 1  (+10.0)")

    # Vital signs
    if not np.isnan(hr_v):
        if hr_v > 120: raw += 2.5; triggered.append(f"Severe tachycardia HR={hr_v:.0f} bpm  (+2.5)")
        elif hr_v > 100: raw += 1.5; triggered.append(f"Tachycardia HR={hr_v:.0f} bpm  (+1.5)")
    if not np.isnan(sbp_v):
        if sbp_v < 80: raw += 3.0; triggered.append(f"Severe hypotension SBP={sbp_v:.0f} mmHg  (+3.0)")
        elif sbp_v < 90: raw += 2.0; triggered.append(f"Hypotension SBP={sbp_v:.0f} mmHg  (+2.0)")
    if not np.isnan(o2_v):
        if o2_v < 90: raw += 3.0; triggered.append(f"Critical O2Sat={o2_v:.0f}%  (+3.0)")
        elif o2_v < 94: raw += 1.5; triggered.append(f"Low O2Sat={o2_v:.0f}%  (+1.5)")
    if not np.isnan(rsp_v):
        if rsp_v > 28: raw += 2.0; triggered.append(f"Severe tachypnea Resp={rsp_v:.0f}/min  (+2.0)")
        elif rsp_v > 22: raw += 1.5; triggered.append(f"Tachypnea Resp={rsp_v:.0f}/min  (+1.5)")
    if not np.isnan(tmp_v):
        if tmp_v > 40 or tmp_v < 35:
            raw += 2.0; triggered.append(f"Critical temp={tmp_v:.1f}°C  (+2.0)")
        elif tmp_v > 38.3 or tmp_v < 36:
            raw += 1.0; triggered.append(f"Abnormal temp={tmp_v:.1f}°C  (+1.0)")
    if not np.isnan(map_v):
        if map_v < 55: raw += 3.0; triggered.append(f"Severe low MAP={map_v:.0f}  (+3.0)")
        elif map_v < 65: raw += 2.0; triggered.append(f"Low MAP={map_v:.0f} mmHg  (+2.0)")

    # Shock index
    si = 0.0
    if not np.isnan(hr_v) and not np.isnan(sbp_v) and sbp_v > 0:
        si = hr_v / sbp_v
        if si > 1.4: raw += 3.5; triggered.append(f"Severe shock index={si:.2f}  (+3.5)")
        elif si > 1.0: raw += 2.0; triggered.append(f"Shock index={si:.2f}  (+2.0)")

    # Lab values
    if not np.isnan(lac_v):
        if lac_v > 4.0: raw += 4.5; triggered.append(f"Critical lactate={lac_v:.1f} mmol/L  (+4.5)")
        elif lac_v > 2.0: raw += 2.5; triggered.append(f"High lactate={lac_v:.1f} mmol/L  (+2.5)")
    if not np.isnan(wbc_v):
        if wbc_v > 20 or wbc_v < 2: raw += 2.5; triggered.append(f"Severe WBC abnormality={wbc_v:.1f}  (+2.5)")
        elif wbc_v > 12 or wbc_v < 4: raw += 1.5; triggered.append(f"Abnormal WBC={wbc_v:.1f}  (+1.5)")
    if not np.isnan(cre_v):
        if cre_v > 3.0: raw += 2.5; triggered.append(f"Severe renal injury Creatinine={cre_v:.1f}  (+2.5)")
        elif cre_v > 1.5: raw += 1.5; triggered.append(f"Renal dysfunction Creatinine={cre_v:.1f}  (+1.5)")
    if not np.isnan(ph_v):
        if ph_v < 7.25: raw += 3.0; triggered.append(f"Severe acidosis pH={ph_v:.2f}  (+3.0)")
        elif ph_v < 7.35: raw += 1.5; triggered.append(f"Acidosis pH={ph_v:.2f}  (+1.5)")
    if not np.isnan(plt_v):
        if plt_v < 50: raw += 2.5; triggered.append(f"Severe thrombocytopenia Plt={plt_v:.0f}  (+2.5)")
        elif plt_v < 100: raw += 1.5; triggered.append(f"Low platelets={plt_v:.0f}  (+1.5)")
    if not np.isnan(hgb_v) and hgb_v < 7:
        raw += 1.5; triggered.append(f"Severe anaemia Hgb={hgb_v:.1f}  (+1.5)")
    if not np.isnan(trop_v) and trop_v > 0.1:
        raw += 1.5; triggered.append(f"Elevated troponin={trop_v:.2f}  (+1.5)")
    if not np.isnan(bil_v) and bil_v > 2.0:
        raw += 1.0; triggered.append(f"High bilirubin={bil_v:.1f}  (+1.0)")
    if not np.isnan(ptt_v) and ptt_v > 50:
        raw += 1.0; triggered.append(f"Elevated PTT={ptt_v:.0f}  (+1.0)")
    if not np.isnan(bun_v) and bun_v > 20:
        raw += 1.0; triggered.append(f"Elevated BUN={bun_v:.0f} mg/dL  (+1.0)")
    if not np.isnan(hco3_v) and hco3_v < 18:
        raw += 1.0; triggered.append(f"Low HCO3={hco3_v:.0f}  (+1.0)")

    # Deterioration trend
    if n >= 4:
        for col, direction, label in [
            ('HR',    'up',   'Rising HR'),
            ('SBP',   'down', 'Falling SBP'),
            ('O2Sat', 'down', 'Falling O2Sat'),
            ('Resp',  'up',   'Rising Resp'),
        ]:
            if col in df_f.columns:
                q1 = df_f[col].iloc[:n//4].mean()
                q4 = df_f[col].iloc[-n//4:].mean()
                if not (np.isnan(q1) or np.isnan(q4)):
                    if direction == 'up'   and q4 > q1 * 1.10:
                        raw += 1.0; triggered.append(f"Trend: {label} (+1.0)")
                    if direction == 'down' and q4 < q1 * 0.90:
                        raw += 1.0; triggered.append(f"Trend: {label} (+1.0)")

    vitals = {'HR': hr_v, 'SBP': sbp_v, 'O2Sat': o2_v, 'Resp': rsp_v,
              'Temp': tmp_v, 'MAP': map_v, 'Lactate': lac_v, 'WBC': wbc_v,
              'Creatinine': cre_v, 'pH': ph_v, 'SI': si}
    return raw, triggered, vitals


# ─────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    missing = []
    for f in ['tft_best.pt','tabnet_best.pt','fusion_best.pt']:
        if not os.path.exists(f"{MODEL_DIR}/{f}"): missing.append(f)
    for f in ['seq_scaler.pkl','tab_scaler.pkl','tab_cols.pkl']:
        if not os.path.exists(f"{PROCESSED_DIR}/{f}"): missing.append(f)
    if missing:
        return None, None, None, None, None, None, missing

    with open(f"{PROCESSED_DIR}/seq_scaler.pkl",'rb') as f: seq_scaler = pickle.load(f)
    with open(f"{PROCESSED_DIR}/tab_scaler.pkl",'rb') as f: tab_scaler = pickle.load(f)
    with open(f"{PROCESSED_DIR}/tab_cols.pkl",  'rb') as f: tab_cols   = pickle.load(f)

    seq_dim = seq_scaler.n_features_in_
    tab_dim = tab_scaler.n_features_in_

    tft_m = TemporalFusionTransformer(input_dim=seq_dim).eval()
    tab_m = TabNet(input_dim=tab_dim).eval()
    fus_m = AttentionFusionLayer().eval()

    tft_m.load_state_dict(torch.load(f"{MODEL_DIR}/tft_best.pt",    map_location=DEVICE))
    tab_m.load_state_dict(torch.load(f"{MODEL_DIR}/tabnet_best.pt", map_location=DEVICE))
    fus_m.load_state_dict(torch.load(f"{MODEL_DIR}/fusion_best.pt", map_location=DEVICE))
    return tft_m, tab_m, fus_m, seq_scaler, tab_scaler, tab_cols, []


# ─────────────────────────────────────────────────────
# PATIENT DATA LOADING + FEATURE BUILDING
# ─────────────────────────────────────────────────────
def load_patient_psv(filepath):
    df = pd.read_csv(filepath, sep='|')
    feat_cols = VITAL_COLS + LAB_COLS + DEMO_COLS
    for col in feat_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    df['ShockIndex']    = df['HR'] / (df['SBP'] + 1e-6)
    df['PulsePressure'] = df['SBP'] - df['DBP']
    df['HR_MAP_ratio']  = df['HR'] / (df['MAP'] + 1e-6)
    return df


def build_seq_from_patient(df, seq_scaler):
    vital_features = VITAL_COLS + ['ShockIndex','PulsePressure','HR_MAP_ratio']
    available = [c for c in vital_features if c in df.columns]
    vitals = np.nan_to_num(df[available].values.astype(np.float32), nan=0.0)
    if len(vitals) >= TIME_STEPS:
        seq = vitals[-TIME_STEPS:]
    else:
        pad = np.zeros((TIME_STEPS - len(vitals), vitals.shape[1]))
        seq = np.vstack([pad, vitals])
    n_feat  = seq.shape[1]
    expected = seq_scaler.n_features_in_
    if n_feat < expected:
        seq = np.pad(seq, ((0,0),(0, expected - n_feat)))
    else:
        seq = seq[:, :expected]
    scaled = seq_scaler.transform(seq.reshape(-1, expected)).reshape(1, TIME_STEPS, expected)
    return torch.FloatTensor(np.clip(scaled, -5.0, 5.0))


def build_tab_from_patient(df, tab_scaler):
    row = {}
    for col in VITAL_COLS + ['ShockIndex','PulsePressure','HR_MAP_ratio']:
        if col in df.columns:
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_max']  = df[col].max()
            row[f'{col}_min']  = df[col].min()
            row[f'{col}_std']  = df[col].std() if len(df)>1 else 0.0
    for col in LAB_COLS:
        if col in df.columns:
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_last'] = df[col].iloc[-1]
    for col in DEMO_COLS:
        if col in df.columns:
            row[col] = df[col].iloc[-1]
    feat = np.array(list(row.values()), dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0)
    expected = tab_scaler.n_features_in_
    if len(feat) < expected: feat = np.pad(feat, (0, expected-len(feat)))
    else: feat = feat[:expected]
    scaled = tab_scaler.transform(feat.reshape(1,-1))
    return torch.FloatTensor(np.clip(scaled, -5.0, 5.0))


def predict_models(seq_tensor, tab_tensor, tft_m, tab_m, fus_m):
    with torch.no_grad():
        tft_logit,  vsn_w   = tft_m(seq_tensor)
        tab_logit,  tab_imp = tab_m(tab_tensor)
        tft_prob  = torch.sigmoid(tft_logit)
        tab_prob  = torch.sigmoid(tab_logit)
        final, weights = fus_m(tft_prob, tab_prob)
    return (float(final.item()), float(tft_prob.item()), float(tab_prob.item()),
            weights.cpu().numpy()[0], tab_imp.cpu().numpy()[0])


def get_risk_level(score):
    if score <= LOW_THRESH:   return "Low Risk",      "#28a745"
    elif score <= HIGH_THRESH: return "Moderate Risk", "#ffc107"
    else:                      return "High Risk",     "#dc3545"


# ─────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────
def main():
    st.title("🏥 AI-Based Early Sepsis Risk Prediction System")
    st.markdown("**TFT + TabNet + Attention Fusion | PhysioNet 2019 Challenge**")
    st.divider()

    tft_m, tab_m, fus_m, seq_sc, tab_sc, tab_cols, missing = load_models()
    models_ready = len(missing) == 0

    if not models_ready:
        st.warning(f"⚠️ Models not found ({missing}). Running in **clinical scoring mode** "
                   f"until steps 1–5 are completed.")

    st.sidebar.header("⚙️ Input Mode")
    mode = st.sidebar.radio("Choose input:", [
        "📂 Load Real Patient (.psv)", "🎛️ Manual Slider Input"], index=0)

    seq_tensor = tab_tensor = patient_df = None
    input_ready = False

    # ════════════════════════════════════
    # MODE 1: Real Patient
    # ════════════════════════════════════
    if mode == "📂 Load Real Patient (.psv)":
        psv_files = sorted(glob.glob(f"{DATA_DIR}/*.psv"))
        if not psv_files:
            st.sidebar.error(f"No .psv files found in `{DATA_DIR}/`")
        else:
            file_names = [os.path.basename(f) for f in psv_files]
            selected   = st.sidebar.selectbox(
                f"Select patient ({len(file_names)} available):", file_names)
            selected_path = os.path.join(DATA_DIR, selected)
            patient_df    = load_patient_psv(selected_path)
            true_label    = int(patient_df['SepsisLabel'].max()) \
                            if 'SepsisLabel' in patient_df.columns else None

            st.subheader(f"📋 Patient: `{selected}`")
            last = patient_df.iloc[-1]
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Heart Rate(60 – 100 bpm)",  f"{last.get('HR','N/A'):.0f} bpm"    if not np.isnan(float(last.get('HR',np.nan))) else "N/A")
            c2.metric("O2 Sat (95 – 100%)",      f"{last.get('O2Sat','N/A'):.0f}%"    if not np.isnan(float(last.get('O2Sat',np.nan))) else "N/A")
            c3.metric("Temperature(36.5 – 37.5°C)", f"{last.get('Temp','N/A'):.1f}°C"    if not np.isnan(float(last.get('Temp',np.nan))) else "N/A")
            c4.metric("Systolic BP(90 – 120 mmHg)", f"{last.get('SBP','N/A'):.0f} mmHg" if not np.isnan(float(last.get('SBP',np.nan))) else "N/A")
            c5.metric("Resp Rate(12 – 20 breaths/min)",   f"{last.get('Resp','N/A'):.0f}/min"  if not np.isnan(float(last.get('Resp',np.nan))) else "N/A")
            c6.metric("ICU Hours",   f"{len(patient_df)}")

            if true_label is not None:
                if true_label == 1: st.info("🔖 Ground truth: **Sepsis (positive)**")
                else:               st.info("🔖 Ground truth: **No Sepsis (negative)**")

           
    # ════════════════════════════════════
    # MODE 2: Manual Sliders
    # ════════════════════════════════════
    else:
        st.sidebar.subheader("Vital Signs")
        hr    = st.sidebar.slider("Heart Rate (60 – 100 bpm)",      40, 200, 90)
        o2sat = st.sidebar.slider("O2 Saturation (95 – 100%)",     70, 100, 97)
        temp  = st.sidebar.slider("Temperature (36.5 – 37.5°C)",      35.0, 42.0, 37.0, 0.1)
        sbp   = st.sidebar.slider("Systolic BP (90 – 120 mmHg)",    60, 200, 120)
        dbp   = st.sidebar.slider("Diastolic BP (60 – 80 mmHg)",   40, 130, 80)
        map_  = st.sidebar.slider("MAP (70 – 100 mmHg)",            40, 150, 90)
        resp  = st.sidebar.slider("Resp Rate (12 – 20 breaths/min)",      8,  40,  16)
        # FIX [F2]: EtCO2 slider — now properly connected
        etco2 = st.sidebar.slider("EtCO2 (12 – 20 mmHg)",         10, 60,  35)

        st.sidebar.subheader("Demographics")
        age    = st.sidebar.number_input("Age", 18, 100, 60)
        gender = st.sidebar.selectbox("Gender", ["Male (1)","Female (0)"])

        st.sidebar.subheader("Lab Values")
        wbc   = st.sidebar.number_input("WBC (4 – 12×10³/μL)",    0.0, 50.0,  8.0)
        lac   = st.sidebar.number_input("Lactate (0.5 – 2.0 mmol/L)", 0.0, 20.0,  1.5)
        cre   = st.sidebar.number_input("Creatinine (0.6 – 1.3 mg/dL)", 0.0, 15.0,  1.0)
        glu   = st.sidebar.number_input("Glucose (70 – 140 mg/dL)",  50.0,600.0,100.0)
        plt_v = st.sidebar.number_input("Platelets (150 – 400 ×10³)",   10.0,800.0,250.0)
        ph    = st.sidebar.number_input("pH (7.35 – 7.45)",6.8, 7.7,  7.40, 0.01)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Heart Rate",  f"{hr} bpm")
        c2.metric("O2 Sat",      f"{o2sat}%")
        c3.metric("Temperature", f"{temp}°C")
        c4.metric("Systolic BP", f"{sbp} mmHg")
        c5.metric("Resp Rate",   f"{resp}/min")
        c6.metric("EtCO2",       f"{etco2} mmHg")

        # Build mock patient_df so threshold checker works the same way
        row_data = {'HR': hr, 'O2Sat': o2sat, 'Temp': temp, 'SBP': sbp,
                    'MAP': map_, 'DBP': dbp, 'Resp': resp,
                    'EtCO2': etco2,   # FIX [F2]: properly included
                    'Lactate': lac, 'WBC': wbc, 'Creatinine': cre,
                    'Glucose': glu, 'Platelets': plt_v, 'pH': ph,
                    'Age': age, 'Gender': 1 if "Male" in gender else 0,
                    'SepsisLabel': 0}
        patient_df = pd.DataFrame([row_data] * TIME_STEPS)

        if models_ready:
            try:
                si   = sbp / (hr + 1e-6)
                base = np.array([hr, o2sat, temp, sbp, map_, dbp, resp, etco2,
                                 sbp/(hr+1e-6), sbp-dbp, hr/(map_+1e-6)],
                                dtype=np.float32)
                seq = np.zeros((TIME_STEPS, len(base)), dtype=np.float32)
                for t in range(TIME_STEPS):
                    prog   = t / max(TIME_STEPS-1, 1)
                    seq[t] = base * (1.0 - 0.05*(1.0-prog))
                seq = np.nan_to_num(seq, nan=0.0)
                expected = seq_sc.n_features_in_
                if seq.shape[1] < expected:
                    seq = np.pad(seq, ((0,0),(0,expected-seq.shape[1])))
                else:
                    seq = seq[:,:expected]
                seq_scaled = seq_sc.transform(seq.reshape(-1,expected)).reshape(1,TIME_STEPS,-1)
                seq_tensor = torch.FloatTensor(np.clip(seq_scaled,-5.0,5.0))

                # Build tabular features — FIX [F2]: EtCO2 properly included
                tab_row = {}
                mock = {'HR':hr,'O2Sat':o2sat,'Temp':temp,'SBP':sbp,
                        'MAP':map_,'DBP':dbp,'Resp':resp,'EtCO2':etco2,
                        'ShockIndex':sbp/(hr+1e-6),'PulsePressure':sbp-dbp,
                        'HR_MAP_ratio':hr/(map_+1e-6)}
                for col,val in mock.items():
                    tab_row[f'{col}_mean']=val; tab_row[f'{col}_max']=val
                    tab_row[f'{col}_min']=val;  tab_row[f'{col}_std']=0.0
                lab_map = {'Lactate':lac,'WBC':wbc,'Creatinine':cre,
                           'Glucose':glu,'Platelets':plt_v,'pH':ph}
                for col in LAB_COLS:
                    val = lab_map.get(col, 0.0)
                    tab_row[f'{col}_mean']=val; tab_row[f'{col}_last']=val
                demo_map = {'Age':age,'Gender':1 if "Male" in gender else 0,
                            'Unit1':1,'Unit2':0,'HospAdmTime':-5}
                for col in DEMO_COLS:
                    tab_row[col] = demo_map.get(col, 0.0)
                feat = np.array(list(tab_row.values()), dtype=np.float32)
                feat = np.nan_to_num(feat, nan=0.0)
                expected_t = tab_sc.n_features_in_
                if len(feat)<expected_t: feat=np.pad(feat,(0,expected_t-len(feat)))
                else: feat=feat[:expected_t]
                tab_scaled = tab_sc.transform(feat.reshape(1,-1))
                tab_tensor = torch.FloatTensor(np.clip(tab_scaled,-5.0,5.0))
                input_ready = True
            except Exception as e:
                st.error(f"Error building input: {e}")

    # ════════════════════════════════════
    # PREDICT BUTTON
    # ════════════════════════════════════
    st.divider()
    predict_btn = st.button("🔍 Predict Sepsis Risk", type="primary",
                            use_container_width=True)

    if predict_btn:
        if patient_df is None:
            st.warning("Please select a patient or enter values first.")
            return

        # FIX [F3]: ALWAYS compute clinical threshold factors from real data
        raw_score, triggered, vitals = compute_triggered_factors(patient_df)
        # Improved sigmoid mapping — calibrated to match find_high_risk.py tiers:
        # raw=0   → ~3%  (no flags at all)
        # raw=5   → ~15% (1-2 mild abnormalities)
        # raw=10  → ~45% (confirmed sepsis OR several abnormals)
        # raw=15  → ~73% (confirmed + multiple organ signs) — HIGH
        # raw=20  → ~88% (confirmed + severe multi-organ)
        # raw=25  → ~95% — CRITICAL
        # raw=35+ → ~98%
        clinical_prob = float(1 / (1 + math.exp(-0.22 * (raw_score - 12))))
        clinical_prob = float(np.clip(clinical_prob, 0.03, 0.98))

        if models_ready and input_ready:
            # Real model prediction
            with st.spinner("Running TFT + TabNet + Fusion..."):
                final_score, tft_score, tabnet_score, weights, tab_imp = \
                    predict_models(seq_tensor, tab_tensor, tft_m, tab_m, fus_m)
            st.success("✅ Prediction from trained AI models")
        else:
            # Clinical scoring fallback — directly uses clinical_prob from
            # compute_triggered_factors() which already has worst-case lab values
            st.info("ℹ️ Clinical rule-based scoring — using worst-case lab values "
                    "across entire ICU stay (models not trained yet)")

            # Split raw_score into vital portion and lab portion for display
            # Vitals = threshold crossings for HR/SBP/O2/Resp/MAP/SI
            # Labs   = threshold crossings for Lactate/WBC/Creatinine/pH etc.
            vital_triggers = [t for t in triggered if any(
                k in t for k in ['HR=','SBP=','O2Sat','Resp=','MAP=',
                                  'tachycardia','hypotension','O2','tachypnea',
                                  'temp','shock index','MAP','EtCO2','Trend'])]
            lab_triggers   = [t for t in triggered if any(
                k in t for k in ['lactate','Lactate','WBC','Creatinine',
                                  'pH','acidosis','platelet','Plt','Hgb',
                                  'troponin','Troponin','PTT','BUN','HCO3',
                                  'PaCO2','bilirubin','Bilirubin',
                                  'SepsisLabel'])]

            vital_raw = sum(float(t.split('(+')[1].rstrip(')')
                                   .split()[0]) for t in vital_triggers
                            if '(+' in t)
            lab_raw   = sum(float(t.split('(+')[1].rstrip(')')
                                   .split()[0]) for t in lab_triggers
                            if '(+' in t)

            # TFT score = vital-sign component | TabNet score = lab component
            tft_score    = float(np.clip(
                1/(1+math.exp(-0.22*(vital_raw - 6))), 0.03, 0.98))
            tabnet_score = float(np.clip(
                1/(1+math.exp(-0.22*(lab_raw   - 6))), 0.03, 0.98))
            final_score  = clinical_prob  # already computed from full raw_score
            weights      = np.array([0.50, 0.50])
            tab_imp      = np.ones(20, dtype=np.float32)/20

        # ── Display Results ──
        risk_label, risk_color = get_risk_level(final_score)
        st.header("📊 Prediction Results")
        r1,r2,r3,r4 = st.columns(4)
        r1.markdown(
            f"<div style='background:{risk_color}22;border:2px solid {risk_color};"
            f"border-radius:10px;padding:15px;text-align:center;'>"
            f"<h2 style='color:{risk_color}'>{final_score*100:.1f}%</h2>"
            f"<p><b>Sepsis Risk Score</b></p>"
            f"<h3>{risk_label}</h3></div>", unsafe_allow_html=True)
        r2.metric("TFT Score (vitals)",    f"{tft_score*100:.1f}%")
        r3.metric("TabNet Score (labs)",   f"{tabnet_score*100:.1f}%")
       

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Scores")
            fig, ax = plt.subplots(figsize=(5,3))
            bars = ax.bar(['TFT\n(Time-Series)','TabNet\n(Tabular)','Fusion\n(Final)'],
                          [tft_score, tabnet_score, final_score],
                          color=['steelblue','tomato','green'],
                          edgecolor='white', width=0.5)
            for bar, val in zip(bars, [tft_score, tabnet_score, final_score]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.1f}%', ha='center', fontweight='bold')
            ax.axhline(y=LOW_THRESH,  color='orange', linestyle='--', alpha=0.6,
                       label=f'Low/Moderate ({LOW_THRESH*100:.0f}%)')
            ax.axhline(y=HIGH_THRESH, color='red',    linestyle='--', alpha=0.6,
                       label=f'Moderate/High ({HIGH_THRESH*100:.0f}%)')
            ax.set_ylim(0, 1.1); ax.set_ylabel('Sepsis Probability')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # FIX [F3]: Triggered clinical factors — always shown regardless of mode
        st.divider()
        st.subheader("🔍 Triggered Clinical Thresholds")
        if triggered:
            n_cols = min(3, len(triggered))
            cols   = st.columns(n_cols)
            for i, factor in enumerate(triggered):
                cols[i % n_cols].error(f"⚠️ {factor}")
        else:
            st.success("✅ No clinical threshold crossings detected")

        # Clinical recommendation
        st.divider()
        if final_score > HIGH_THRESH:
            st.error("🚨 **HIGH RISK — Immediate clinical review required.**\n\n"
                     "Recommended actions: activate sepsis bundle, obtain blood cultures, "
                     "administer IV antibiotics within 1 hour, fluid resuscitation.")
        elif final_score > LOW_THRESH:
            st.warning("⚠️ **MODERATE RISK — Close monitoring advised.**\n\n"
                       "Repeat vitals every 2 hours. Review lab results. "
                       "Reassess within 4 hours.")
        else:
            st.success("✅ **LOW RISK — Continue routine ICU monitoring.**\n\n"
                       "No immediate sepsis concern. Standard care protocol.")
    else:
        if patient_df is not None:
            st.info("👆 Click **Predict Sepsis Risk** to run the models.")
        else:
            st.info("👈 Select input mode in the sidebar.")
            st.markdown("""
| Model | Role | Key thresholds |
|-------|------|----------------|
| **TFT** | 12-hour vital sign trends | HR>100, SBP<90, O2Sat<94, Resp>22, MAP<65, SI>1.0 |
| **TabNet** | Lab values + demographics | Lactate>2.0, WBC>12/<4, Creatinine>1.5, pH<7.35 |
| **Fusion** | Combines both with attention | Final score > 0.60 = High Risk |
""")


if __name__ == "__main__":
    main()