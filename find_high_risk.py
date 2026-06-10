"""
SEPSIS RISK ANALYSER — PhysioNet 2019 Challenge
=================================================
Scans ALL .psv files and ranks every patient by clinical sepsis risk.

Run : python find_high_risk.py
Output :
  → high_risk_patients.txt   (full human-readable report)
  → high_risk_patients.csv   (sortable in Excel/Sheets)
  → results_checkpoint.csv   (saved every 5,000 files — never lose progress)

FIXES APPLIED vs previous version:
  [F1] Added missing lab thresholds: BUN, TroponinI, PTT, Hgb, HCO3, PaCO2
  [F2] Worst-case lab values used: Lactate_max, pH_min, Platelets_min, Hgb_min
       (peak values matter for threshold crossing, not just the last reading)
  [F3] MAP and EtCO2 added to deterioration trend check
  [F4] HIGH_RISK_SCORE lowered to 10.0 (catches all confirmed + unstable patients)
       Added CRITICAL_SCORE=25.0 and three severity tiers in output
  [F5] Progress prints every 1,000 files with elapsed time and ETA
  [F6] Checkpoint CSV saved every 5,000 files — safe for 20,000 file runs
  [F7] CSV output added for Excel/Sheets filtering and sorting
  [F8] Error logging shows which file failed and why
  [F9] MAP recorded in per-patient output dict
  [F10] Console shows top 3 triggered reasons per patient
"""

import os
import time
import pandas as pd
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
DATA_DIR         = "./data"
OUTPUT_TXT       = "high_risk_patients.txt"
OUTPUT_CSV       = "high_risk_patients.csv"
CHECKPOINT_CSV   = "results_checkpoint.csv"

# FIX [F4]: Score tiers aligned with clinical severity
# Low risk     : < 10.0
# Moderate risk: 10.0 – 14.9   (confirmed sepsis or 2–3 abnormal vitals)
# High risk    : 15.0 – 24.9   (confirmed sepsis + multiple organ signs)
# Critical     : ≥ 25.0        (multi-organ failure pattern)
HIGH_RISK_SCORE = 10.0   # include in report (was 15.0 — too high, missed many)
CRITICAL_SCORE  = 25.0   # flag as critical tier
TOP_N_CONSOLE   = 20     # how many to print in terminal


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def safe_float(val, default=np.nan):
    """Safely convert a value to float, returning default if NaN or missing."""
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def severity_tier(score):
    """Convert a raw score to a human-readable severity tier."""
    if score >= CRITICAL_SCORE: return "CRITICAL"
    if score >= HIGH_RISK_SCORE + 5: return "HIGH"
    if score >= HIGH_RISK_SCORE: return "MODERATE-HIGH"
    return "LOW"


# ─────────────────────────────────────────────────────
# CLINICAL RISK SCORING
# All thresholds from clinical breakdown table
# ─────────────────────────────────────────────────────
def compute_risk_score(df):
    """
    Compute a composite sepsis risk score for one patient.

    Threshold reference:
    VITAL SIGNS (TFT model thresholds):
      HR    > 100 bpm           → +1.5  |  > 120 → +2.5 total
      SBP   < 90  mmHg          → +2.0  |  < 80  → +3.0 total
      O2Sat < 94%               → +1.5  |  < 90% → +3.0 total
      Resp  > 22 /min           → +1.5  |  > 28  → +3.5 total
      Temp  > 38.3°C or < 36°C  → +1.0  |  > 40 or < 35 → +2.0 total
      MAP   < 65  mmHg          → +2.0  |  < 55  → +3.0 total
      ShockIndex > 1.0          → +2.0  |  > 1.4 → +3.5 total

    LAB VALUES (TabNet model thresholds):
      Lactate    > 2.0 mmol/L   → +2.5  |  > 4.0 → +4.5 total (uses MAX)
      WBC        > 12 or < 4    → +1.5  |  > 20 or < 2 → +4.0 total
      Creatinine > 1.5 mg/dL    → +1.5  |  > 3.0 → +4.0 total (uses MAX)
      pH         < 7.35         → +1.5  |  < 7.25 → +4.5 total (uses MIN)
      Platelets  < 100          → +1.5  |  < 50  → +4.0 total (uses MIN)
      Bilirubin  > 2.0 mg/dL    → +1.0  |  > 4.0 → +2.0 total
      TroponinI  > 0.1 ng/mL    → +1.5  (cardiac injury)
      PTT        > 50 sec       → +1.0  |  > 70  → +2.5 total
      Hgb        < 9 g/dL       → +1.5  (uses MIN)
      BUN        > 20 mg/dL     → +1.0  (kidney function)
      HCO3       < 18 mEq/L     → +1.0  (metabolic acidosis)
      PaCO2      < 30 mmHg      → +1.0  (respiratory compensation)

    DETERIORATION TREND (+1.0 per vital worsening > 10%):
      HR rising, SBP falling, O2Sat falling, Resp rising,
      MAP falling, EtCO2 falling
    """
    score    = 0.0
    reasons  = []

    # Fill missing values forward/backward within this patient
    df   = df.ffill().bfill()
    last = df.iloc[-1]
    n    = len(df)

    # Helper: get last-row value safely
    def lv(col):
        return safe_float(last[col] if col in last.index else np.nan)

    # FIX [F2]: Worst-case aggregations for critical labs
    def col_max(col):
        return safe_float(df[col].max() if col in df.columns else np.nan)

    def col_min(col):
        return safe_float(df[col].min() if col in df.columns else np.nan)

    # ── Ground truth label ──
    has_sepsis = int(df['SepsisLabel'].max()) \
                 if 'SepsisLabel' in df.columns else 0
    if has_sepsis:
        score += 10.0
        reasons.append("SepsisLabel=1 confirmed  (+10.0)")

    # ── Vital signs (use last-row — current state) ──
    hr   = lv('HR')
    sbp  = lv('SBP')
    o2   = lv('O2Sat')
    resp = lv('Resp')
    tmp  = lv('Temp')
    mp   = lv('MAP')
    etco2 = lv('EtCO2')

    if not np.isnan(hr):
        if hr > 120:   score += 1.0; reasons.append(f"Severe tachycardia HR={hr:.0f} bpm  (+2.5 total)")
        if hr > 100:   score += 1.5; reasons.append(f"Tachycardia HR={hr:.0f} bpm  (+1.5)")

    if not np.isnan(sbp):
        if sbp < 80:   score += 1.0; reasons.append(f"Severe hypotension SBP={sbp:.0f} mmHg  (+3.0 total)")
        if sbp < 90:   score += 2.0; reasons.append(f"Hypotension SBP={sbp:.0f} mmHg  (+2.0)")

    if not np.isnan(o2):
        if o2 < 90:    score += 1.5; reasons.append(f"Critical O2Sat={o2:.0f}%  (+3.0 total)")
        if o2 < 94:    score += 1.5; reasons.append(f"Low O2Sat={o2:.0f}%  (+1.5)")

    if not np.isnan(resp):
        if resp > 28:  score += 2.0; reasons.append(f"Severe tachypnea Resp={resp:.0f}/min  (+3.5 total)")
        elif resp > 22: score += 1.5; reasons.append(f"Tachypnea Resp={resp:.0f}/min  (+1.5)")

    if not np.isnan(tmp):
        if tmp > 40 or tmp < 35:
            score += 1.0; reasons.append(f"Critical temp={tmp:.1f}°C  (+2.0 total)")
        elif tmp > 38.3 or tmp < 36.0:
            score += 1.0; reasons.append(f"Abnormal temp={tmp:.1f}°C  (+1.0)")

    if not np.isnan(mp):
        if mp < 55:    score += 1.0; reasons.append(f"Severe low MAP={mp:.0f} mmHg  (+3.0 total)")
        if mp < 65:    score += 2.0; reasons.append(f"Low MAP={mp:.0f} mmHg  (+2.0)")

    if not np.isnan(etco2) and etco2 < 25:
        score += 1.0; reasons.append(f"Low EtCO2={etco2:.0f} mmHg — acidosis sign  (+1.0)")

    # ── Shock index ──
    si = np.nan
    if not np.isnan(hr) and not np.isnan(sbp) and sbp > 0:
        si = hr / sbp
        if si > 1.4:   score += 1.5; reasons.append(f"Severe shock index={si:.2f}  (+3.5 total)")
        if si > 1.0:   score += 2.0; reasons.append(f"Shock index={si:.2f}  (+2.0)")

    # ── Lab values — FIX [F2]: peak/worst values ──
    lac = col_max('Lactate')       # worst = highest
    ph  = col_min('pH')            # worst = lowest
    plt_min = col_min('Platelets') # worst = lowest
    hgb_min = col_min('Hgb')       # worst = lowest
    wbc = lv('WBC')                # last is fine — direction unclear
    cre = col_max('Creatinine')    # worst = highest
    bil = col_max('Bilirubin_total')
    trop = col_max('TroponinI')    # worst = highest
    ptt  = col_max('PTT')          # worst = highest
    bun  = col_max('BUN')          # worst = highest
    hco3 = col_min('HCO3')         # worst = lowest
    paco2 = col_min('PaCO2')       # worst = lowest

    if not np.isnan(lac):
        if lac > 4.0:  score += 2.0; reasons.append(f"Critical lactate={lac:.1f} mmol/L  (+4.5 total)")
        if lac > 2.0:  score += 2.5; reasons.append(f"High lactate={lac:.1f} mmol/L  (+2.5)")

    if not np.isnan(wbc):
        if wbc > 20 or wbc < 2:
            score += 2.5; reasons.append(f"Severe WBC abnormality={wbc:.1f}  (+4.0 total)")
        elif wbc > 12 or wbc < 4:
            score += 1.5; reasons.append(f"Abnormal WBC={wbc:.1f}  (+1.5)")

    if not np.isnan(cre):
        if cre > 3.0:  score += 2.5; reasons.append(f"Severe renal injury Creatinine={cre:.1f}  (+4.0 total)")
        elif cre > 1.5: score += 1.5; reasons.append(f"Renal dysfunction Creatinine={cre:.1f}  (+1.5)")

    if not np.isnan(ph):
        if ph < 7.25:  score += 3.0; reasons.append(f"Severe acidosis pH={ph:.2f}  (+4.5 total)")
        elif ph < 7.35: score += 1.5; reasons.append(f"Acidosis pH={ph:.2f}  (+1.5)")

    if not np.isnan(plt_min):
        if plt_min < 50:   score += 2.5; reasons.append(f"Severe thrombocytopenia Plt={plt_min:.0f}  (+4.0 total)")
        elif plt_min < 100: score += 1.5; reasons.append(f"Low platelets={plt_min:.0f}  (+1.5)")

    if not np.isnan(bil):
        if bil > 4.0:  score += 1.0; reasons.append(f"Severe high bilirubin={bil:.1f}  (+2.0 total)")
        elif bil > 2.0: score += 1.0; reasons.append(f"High bilirubin={bil:.1f}  (+1.0)")

    # FIX [F1]: Previously missing labs
    if not np.isnan(trop) and trop > 0.1:
        score += 1.5; reasons.append(f"Elevated TroponinI={trop:.2f} — cardiac injury  (+1.5)")

    if not np.isnan(ptt):
        if ptt > 70:   score += 1.5; reasons.append(f"Severe PTT={ptt:.0f} sec — DIC risk  (+2.5 total)")
        elif ptt > 50:  score += 1.0; reasons.append(f"Elevated PTT={ptt:.0f} sec  (+1.0)")

    if not np.isnan(hgb_min) and hgb_min < 9:
        score += 1.5; reasons.append(f"Severe anaemia Hgb={hgb_min:.1f} g/dL  (+1.5)")

    if not np.isnan(bun) and bun > 20:
        score += 1.0; reasons.append(f"Elevated BUN={bun:.0f} mg/dL — kidney stress  (+1.0)")

    if not np.isnan(hco3) and hco3 < 18:
        score += 1.0; reasons.append(f"Low HCO3={hco3:.0f} mEq/L — metabolic acidosis  (+1.0)")

    if not np.isnan(paco2) and paco2 < 30:
        score += 1.0; reasons.append(f"Low PaCO2={paco2:.0f} — respiratory compensation  (+1.0)")

    # ── Deterioration trend ──
    # FIX [F3]: Added MAP and EtCO2 to trend check
    if n >= 4:
        for col, direction, label in [
            ('HR',    'up',   'Rising HR'),
            ('SBP',   'down', 'Falling SBP'),
            ('O2Sat', 'down', 'Falling O2Sat'),
            ('Resp',  'up',   'Rising Resp'),
            ('MAP',   'down', 'Falling MAP'),      # FIX [F3]
            ('EtCO2', 'down', 'Falling EtCO2'),   # FIX [F3]
        ]:
            if col in df.columns:
                q1 = df[col].iloc[:n//4].mean()
                q4 = df[col].iloc[-n//4:].mean()
                if not (np.isnan(q1) or np.isnan(q4)):
                    if direction == 'up'   and q4 > q1 * 1.10:
                        score += 1.0; reasons.append(f"Trend: {label}  (+1.0)")
                    if direction == 'down' and q4 < q1 * 0.90:
                        score += 1.0; reasons.append(f"Trend: {label}  (+1.0)")

    return round(score, 1), has_sepsis, reasons, n, last, si


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.psv')])
    total = len(files)

    if total == 0:
        print(f"❌ No .psv files found in {DATA_DIR}/"); return

    print("=" * 65)
    print(f"  SEPSIS RISK ANALYSER — {total:,} patient files")
    print("=" * 65)
    print(f"  Score tiers:")
    print(f"    CRITICAL      : ≥ {CRITICAL_SCORE}")
    print(f"    HIGH          : ≥ {HIGH_RISK_SCORE+5:.0f}")
    print(f"    MODERATE-HIGH : ≥ {HIGH_RISK_SCORE}")
    print(f"  Output files: {OUTPUT_TXT}  |  {OUTPUT_CSV}")
    print(f"  Checkpoint  : {CHECKPOINT_CSV} (saved every 5,000 files)")
    print("=" * 65)

    all_results    = []
    error_log      = []
    start_time     = time.time()

    for i, fname in enumerate(files):
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path, sep='|')
            score, has_sepsis, reasons, icu_h, last, si = compute_risk_score(df)

            def lv(col):
                return safe_float(last[col] if col in last.index else np.nan)

            rec = {
                'rank':       0,
                'file':       fname.replace('.psv',''),
                'score':      score,
                'tier':       severity_tier(score),
                'has_sepsis': has_sepsis,
                'icu_hours':  icu_h,
                'HR':         lv('HR'),
                'SBP':        lv('SBP'),
                'O2Sat':      lv('O2Sat'),
                'Temp':       lv('Temp'),
                'Resp':       lv('Resp'),
                'MAP':        lv('MAP'),      # FIX [F9]
                'EtCO2':      lv('EtCO2'),
                'Lactate':    safe_float(df['Lactate'].max() if 'Lactate' in df else np.nan),
                'WBC':        lv('WBC'),
                'Creatinine': safe_float(df['Creatinine'].max() if 'Creatinine' in df else np.nan),
                'pH':         safe_float(df['pH'].min() if 'pH' in df else np.nan),
                'Platelets':  safe_float(df['Platelets'].min() if 'Platelets' in df else np.nan),
                'ShockIndex': round(float(si), 3) if not np.isnan(si) else np.nan,
                'n_reasons':  len(reasons),
                'top_reasons': ' | '.join(reasons[:3]),
                'all_reasons': reasons,
            }
            all_results.append(rec)

        except Exception as e:
            # FIX [F8]: Log which file failed and why
            error_log.append(f"{fname}: {str(e)}")

        # FIX [F5]: Progress every 1,000 with elapsed time and ETA
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate    = (i + 1) / elapsed
            eta     = (total - i - 1) / rate
            pct     = (i + 1) / total * 100
            print(f"  {i+1:>6,}/{total:,}  ({pct:.0f}%)  |  "
                  f"{elapsed:.0f}s elapsed  |  ETA ~{eta:.0f}s")

        # FIX [F6]: Checkpoint CSV every 5,000 files
        if (i + 1) % 5000 == 0 and all_results:
            _write_checkpoint(all_results, CHECKPOINT_CSV, i + 1)

    elapsed_total = time.time() - start_time

    # Sort by score descending, assign ranks
    all_results.sort(key=lambda x: x['score'], reverse=True)
    for idx, r in enumerate(all_results, 1):
        r['rank'] = idx

    # Filter by tier
    high_risk  = [r for r in all_results if r['score'] >= HIGH_RISK_SCORE]
    critical   = [r for r in all_results if r['score'] >= CRITICAL_SCORE]
    confirmed  = [r for r in all_results if r['has_sepsis'] == 1]

    print(f"\n{'='*65}")
    print(f"  SCAN COMPLETE — {elapsed_total:.1f}s for {total:,} files")
    print(f"  Errors          : {len(error_log)}")
    print(f"  Confirmed sepsis: {len(confirmed):,}")
    print(f"  Critical (≥{CRITICAL_SCORE:.0f}) : {len(critical):,}")
    print(f"  High risk (≥{HIGH_RISK_SCORE:.0f}) : {len(high_risk):,}")
    print(f"{'='*65}\n")

    # ── Write TXT report ──
    _write_txt_report(all_results, high_risk, critical, confirmed,
                      error_log, total, OUTPUT_TXT)

    # ── Write CSV ── FIX [F7]
    _write_csv(all_results, OUTPUT_CSV)

    # ── Console TOP N ──
    print(f"  TOP {TOP_N_CONSOLE} HIGHEST RISK PATIENTS")
    print(f"  {'#':<5} {'File':<14} {'Score':<8} {'Tier':<16} "
          f"{'Sepsis':<8} {'Top reasons'}")
    print("  " + "-" * 80)
    for r in all_results[:TOP_N_CONSOLE]:
        sep  = "YES" if r['has_sepsis'] else "no"
        # FIX [F10]: Show top 3 reasons
        top3 = "  |  ".join(r['all_reasons'][:3]) if r['all_reasons'] else "—"
        print(f"  {r['rank']:<5} {r['file']:<14} {r['score']:<8} "
              f"{r['tier']:<16} {sep:<8} {top3[:60]}")

    print(f"\n✅ Full report  → {OUTPUT_TXT}")
    print(f"✅ CSV export   → {OUTPUT_CSV}")
    if error_log:
        print(f"⚠️  {len(error_log)} files had errors — see bottom of {OUTPUT_TXT}")

    # ── Dashboard recommendation ──
    print(f"\n  Files to test in dashboard (top confirmed sepsis):")
    for r in confirmed[:10]:
        print(f"    {r['file']}.psv  (score={r['score']}, tier={r['tier']})")


# ─────────────────────────────────────────────────────
# WRITE HELPERS
# ─────────────────────────────────────────────────────
def _fmt(val, fmt='.1f', fallback='N/A'):
    """Format a value or return fallback if NaN."""
    if isinstance(val, float) and np.isnan(val): return fallback
    try: return format(val, fmt)
    except: return fallback


def _write_checkpoint(results, path, n_processed):
    """Save partial results to CSV checkpoint."""
    sorted_r = sorted(results, key=lambda x: x['score'], reverse=True)
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file','score','tier','has_sepsis','icu_hours',
                        'HR','SBP','O2Sat','Temp','Resp','MAP',
                        'Lactate','WBC','Creatinine','pH','ShockIndex',
                        'top_reasons'])
            for r in sorted_r:
                w.writerow([
                    r['file'], r['score'], r['tier'], r['has_sepsis'],
                    r['icu_hours'],
                    _fmt(r['HR'],':.0f'), _fmt(r['SBP'],':.0f'),
                    _fmt(r['O2Sat'],':.0f'), _fmt(r['Temp'],':.1f'),
                    _fmt(r['Resp'],':.0f'), _fmt(r['MAP'],':.0f'),
                    _fmt(r['Lactate'],':.2f'), _fmt(r['WBC'],':.1f'),
                    _fmt(r['Creatinine'],':.2f'), _fmt(r['pH'],':.3f'),
                    _fmt(r['ShockIndex'],':.3f'),
                    r['top_reasons']
                ])
        print(f"  [checkpoint] {n_processed:,} files saved → {path}")
    except Exception as e:
        print(f"  [checkpoint] failed: {e}")


def _write_csv(results, path):
    """Write full CSV — all patients sorted by score."""
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank','file','score','tier','has_sepsis','icu_hours',
                        'HR','SBP','O2Sat','Temp','Resp','MAP','EtCO2',
                        'Lactate_max','WBC','Creatinine_max','pH_min',
                        'Platelets_min','ShockIndex','n_triggers','top_3_reasons'])
            for r in results:
                w.writerow([
                    r['rank'], r['file'], r['score'], r['tier'],
                    'YES' if r['has_sepsis'] else 'no',
                    r['icu_hours'],
                    _fmt(r['HR'],  ':.0f'), _fmt(r['SBP'],  ':.0f'),
                    _fmt(r['O2Sat'],':.0f'), _fmt(r['Temp'], ':.1f'),
                    _fmt(r['Resp'], ':.0f'), _fmt(r['MAP'],  ':.0f'),
                    _fmt(r['EtCO2'],':.0f'),
                    _fmt(r['Lactate'],':.2f'), _fmt(r['WBC'],':.1f'),
                    _fmt(r['Creatinine'],':.2f'), _fmt(r['pH'],':.3f'),
                    _fmt(r['Platelets'],':.0f'),
                    _fmt(r['ShockIndex'],':.3f'),
                    r['n_reasons'],
                    r['top_reasons']
                ])
        print(f"  CSV written: {len(results):,} patients → {path}")
    except Exception as e:
        print(f"  CSV write failed: {e}")


def _write_txt_report(all_results, high_risk, critical, confirmed,
                      error_log, total, path):
    """Write the full human-readable text report."""
    scores = [r['score'] for r in all_results]

    with open(path, 'w', encoding='utf-8') as f:

        # ── Header ──
        f.write("=" * 72 + "\n")
        f.write("  SEPSIS RISK REPORT — PhysioNet Challenge 2019\n")
        f.write(f"  Total patients : {len(all_results):,}\n")
        f.write(f"  Confirmed sepsis: {len(confirmed):,}  "
                f"({len(confirmed)/len(all_results)*100:.1f}%)\n")
        f.write(f"  Critical (≥{CRITICAL_SCORE:.0f}) : {len(critical):,}  "
                f"({len(critical)/len(all_results)*100:.1f}%)\n")
        f.write(f"  High risk (≥{HIGH_RISK_SCORE:.0f}) : {len(high_risk):,}  "
                f"({len(high_risk)/len(all_results)*100:.1f}%)\n")
        f.write(f"  Avg risk score : {np.mean(scores):.2f}\n")
        f.write(f"  Max risk score : {max(scores):.1f}  ({all_results[0]['file']})\n")
        f.write("=" * 72 + "\n\n")

        # ── Section 1: Critical patients ──
        f.write("=" * 72 + "\n")
        f.write(f"  SECTION 1: CRITICAL PATIENTS — score ≥ {CRITICAL_SCORE}  "
                f"({len(critical)} patients)\n")
        f.write("=" * 72 + "\n")
        _write_table(f, critical)

        # ── Section 2: All confirmed sepsis ──
        f.write("\n\n" + "=" * 72 + "\n")
        f.write(f"  SECTION 2: ALL CONFIRMED SEPSIS — SepsisLabel=1  "
                f"({len(confirmed)} patients)\n")
        f.write("=" * 72 + "\n")
        _write_table(f, confirmed)

        # ── Section 3: All high-risk ──
        f.write("\n\n" + "=" * 72 + "\n")
        f.write(f"  SECTION 3: ALL HIGH-RISK PATIENTS — score ≥ {HIGH_RISK_SCORE}  "
                f"({len(high_risk)} patients)\n")
        f.write("=" * 72 + "\n")
        _write_table(f, high_risk)

        # ── Section 4: Detailed top 25 ──
        f.write("\n\n" + "=" * 72 + "\n")
        f.write("  SECTION 4: DETAILED CLINICAL BREAKDOWN — TOP 25\n")
        f.write("=" * 72 + "\n")
        for r in all_results[:25]:
            sep = "CONFIRMED SEPSIS" if r['has_sepsis'] else "clinical risk only"
            f.write(f"\n#{r['rank']:02d}  {r['file']}  |  Score: {r['score']}  "
                    f"|  Tier: {r['tier']}  |  {sep}\n")
            f.write(f"    ICU hours  : {r['icu_hours']}\n")
            f.write(f"    Vitals     : "
                    f"HR={_fmt(r['HR'],':.0f')}  "
                    f"SBP={_fmt(r['SBP'],':.0f')}  "
                    f"O2={_fmt(r['O2Sat'],':.0f')}%  "
                    f"Temp={_fmt(r['Temp'],':.1f')}°C  "
                    f"Resp={_fmt(r['Resp'],':.0f')}  "
                    f"MAP={_fmt(r['MAP'],':.0f')}  "
                    f"ShockIdx={_fmt(r['ShockIndex'],':.2f')}\n")
            f.write(f"    Labs       : "
                    f"Lactate(max)={_fmt(r['Lactate'],':.2f')}  "
                    f"WBC={_fmt(r['WBC'],':.1f')}  "
                    f"Creatinine(max)={_fmt(r['Creatinine'],':.2f')}  "
                    f"pH(min)={_fmt(r['pH'],':.3f')}  "
                    f"Plt(min)={_fmt(r['Platelets'],':.0f')}\n")
            f.write(f"    Triggers   :\n")
            for reason in r['all_reasons']:
                f.write(f"      • {reason}\n")

        # ── Section 5: Summary statistics ──
        f.write("\n\n" + "=" * 72 + "\n")
        f.write("  SECTION 5: DATASET SUMMARY STATISTICS\n")
        f.write("=" * 72 + "\n")
        f.write(f"  Total patients            : {len(all_results):,}\n")
        f.write(f"  Files with errors         : {len(error_log)}\n")
        f.write(f"  Confirmed sepsis          : {len(confirmed):,} "
                f"({len(confirmed)/len(all_results)*100:.1f}%)\n")
        f.write(f"  Critical (≥ {CRITICAL_SCORE:.0f})          : {len(critical):,} "
                f"({len(critical)/len(all_results)*100:.1f}%)\n")
        f.write(f"  High risk (≥ {HIGH_RISK_SCORE:.0f})         : {len(high_risk):,} "
                f"({len(high_risk)/len(all_results)*100:.1f}%)\n")
        f.write(f"  Average risk score        : {np.mean(scores):.2f}\n")
        f.write(f"  Median risk score         : {np.median(scores):.2f}\n")
        f.write(f"  Score std deviation       : {np.std(scores):.2f}\n")
        f.write(f"  Highest score             : {max(scores):.1f} ({all_results[0]['file']})\n")
        f.write(f"  Avg ICU hours             : "
                f"{np.mean([r['icu_hours'] for r in all_results]):.1f}\n")

        # Score distribution
        f.write(f"\n  Score distribution:\n")
        bins = [(0,5),(5,10),(10,15),(15,20),(20,25),(25,35)]
        for lo, hi in bins:
            cnt = sum(1 for r in all_results if lo <= r['score'] < hi)
            bar = '█' * (cnt * 40 // len(all_results))
            f.write(f"    {lo:>3}–{hi:<3}: {cnt:>5}  {bar}\n")

        # ── Error log ──
        if error_log:
            f.write("\n\n" + "=" * 72 + "\n")
            f.write(f"  ERROR LOG — {len(error_log)} files failed to parse\n")
            f.write("=" * 72 + "\n")
            for err in error_log:
                f.write(f"  {err}\n")

    print(f"  TXT report written: {len(all_results):,} patients → {path}")


def _write_table(f, records):
    """Write a compact table of patients to the txt file."""
    hdr = (f"{'Rank':<6}{'File':<14}{'Score':<8}{'Tier':<16}"
           f"{'Sep':<5}{'ICU':<7}{'HR':<6}{'SBP':<6}"
           f"{'O2%':<6}{'Lactate':<10}{'pH':<7}{'MAP':<6}\n")
    f.write(hdr)
    f.write("-" * 80 + "\n")
    for r in records:
        sep = "Y" if r['has_sepsis'] else "n"
        f.write(
            f"{r['rank']:<6}{r['file']:<14}{r['score']:<8}{r['tier']:<16}"
            f"{sep:<5}{r['icu_hours']:<7}"
            f"{_fmt(r['HR'],  ':.0f'):<6}"
            f"{_fmt(r['SBP'], ':.0f'):<6}"
            f"{_fmt(r['O2Sat'],':.0f'):<6}"
            f"{_fmt(r['Lactate'],':.1f'):<10}"
            f"{_fmt(r['pH'],  ':.2f'):<7}"
            f"{_fmt(r['MAP'], ':.0f'):<6}\n"
        )


if __name__ == "__main__":
    main()