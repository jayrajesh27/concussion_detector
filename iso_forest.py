import os, re, zipfile, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from scipy import signal  # install via: pip install scipy
import matplotlib.pyplot as plt

# Author: Jay Rajesh

# -------- CONFIG --------
# Working assumptions:
# - We’re given per-subject .zip bundles (Subject_XXXX.zip), each containing CSV trials.
# - We aggregate trial-level biomarkers and run unsupervised anomaly detection (IsolationForest).
DATA_DIR = Path(".")        # Dummy directory containing the Subject_*.zip files, I inserted GazeBase
OUT_DIR  = Path("./outputs")
CONTAMINATION = 0.05        # model prior: ~5% of trials expected to be anomalous
MIN_VALID_SAMPLES = 10      # ignore ultra-short/empty trials after filtering

OUT_DIR.mkdir(exist_ok=True, parents=True)

# -------- Helpers --------
def parse_ids(fname):
    # Parse subject/session/task from canonical filenames: S_<subj>_S<sess>_<TASK>.csv
    base = Path(fname).stem
    m = re.match(r"S_(\d+)_S(\d+)_([A-Z0-9]+)", base)
    if m:
        subj, sess, task = m.groups()
        return int(subj), int(sess), task
    return None, None, None

def robust_dt_ms(n):
    # Median positive inter-sample interval (ms). Robust against dropped or non-monotonic stamps.
    n = np.asarray(n, float)
    dn = np.diff(n)
    dn_pos = dn[dn > 0]
    if dn_pos.size == 0: return np.nan
    return float(np.median(dn_pos))

def bcea_proxy(x, y):
    # BCEA proxy: 2π σx σy sqrt(1 − ρ²). Captures fixation dispersion in 2D.
    if len(x) < 5: return np.nan
    sx, sy = np.nanstd(x), np.nanstd(y)
    if sx == 0 or sy == 0: return 0.0
    xv = np.nan_to_num(x, nan=np.nanmean(x))
    yv = np.nan_to_num(y, nan=np.nanmean(y))
    rho = np.corrcoef(xv, yv)[0,1]
    rho = np.clip(rho, -0.9999, 0.9999) if np.isfinite(rho) else 0.0
    return float(2*np.pi*sx*sy*np.sqrt(1 - rho**2))

def velocities(x, y, n_ms):
    # Central kinematic readout: per-axis velocities and resultant speed (deg/s).
    if len(x) < 3: return np.array([]), np.array([]), np.array([])
    dt = np.diff(np.asarray(n_ms, float)) / 1_000.0
    dt[dt <= 0] = np.nan        # guard against non-increasing timestamps
    dx, dy = np.diff(x), np.diff(y)
    vx, vy = dx/dt, dy/dt
    v = np.sqrt(vx**2 + vy**2)
    return vx, vy, v

def drift_velocity(x, y, n_ms):
    # Slow positional drift during “fixation”: linear trend magnitude across the trial.
    if len(x) < 5: return np.nan
    t = np.asarray(n_ms, float) - float(n_ms[0])
    try:
        sx, bx = np.polyfit(t, x, 1)
        sy, by = np.polyfit(t, y, 1)
        return float(np.sqrt(sx**2 + sy**2) * 1000.0)  # convert from per-ms to per-s
    except Exception:
        return np.nan

def event_counts(lab):
    # Counts from label stream if provided: 1=fixation, 2=saccade, -1=blink.
    if lab is None or pd.isna(lab).all(): return 0,0,0
    return int((lab==1).sum()), int((lab==2).sum()), int((lab==-1).sum())

def pursuit_metrics(x, y, xT, yT, n_ms):
    # Smooth pursuit quality (if target traces exist): correlation, gain, and temporal lag.
    res = dict(purs_corr_x=np.nan, purs_corr_y=np.nan,
               purs_gain_x=np.nan, purs_gain_y=np.nan, purs_lag_ms=np.nan)
    if pd.isna(xT).all() and pd.isna(yT).all(): return res
    valid = (~pd.isna(x)) & (~pd.isna(y))
    if (~pd.isna(xT)).any(): valid &= (~pd.isna(xT))
    if (~pd.isna(yT)).any(): valid &= (~pd.isna(yT))
    x, y, xT, yT, t = x[valid], y[valid], xT[valid], yT[valid], n_ms[valid]
    if len(x) < 10: return res

    def safe_corr(a,b):
        return np.corrcoef(a,b)[0,1] if (np.std(a)>0 and np.std(b)>0) else np.nan
    res["purs_corr_x"] = float(safe_corr(x.values, xT.values)) if (~pd.isna(xT)).any() else np.nan
    res["purs_corr_y"] = float(safe_corr(y.values, yT.values)) if (~pd.isna(yT)).any() else np.nan

    vx, vy, v = velocities(x.values, y.values, t.values)
    vTx, vTy, vT = velocities(xT.values, yT.values, t.values)
    def ratio_std(a,b):
        # Gain ≈ relative velocity variability; robust to offsets.
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a)<5 or len(b)<5 or np.std(b)==0: return np.nan
        return float(np.std(a)/np.std(b))
    res["purs_gain_x"] = ratio_std(vx, vTx)
    res["purs_gain_y"] = ratio_std(vy, vTy)

    try:
        # Cross-correlation to estimate lag (ms) between eye and target.
        a = x.values - np.nanmean(x.values)
        b = xT.values - np.nanmean(xT.values)
        lags = signal.correlation_lags(len(a), len(b), mode='full')
        corr = signal.correlate(a, b, mode='full')
        best = lags[np.nanargmax(corr)]
        dt = robust_dt_ms(t.values)
        res["purs_lag_ms"] = float(best * dt) if np.isfinite(dt) else np.nan
    except Exception:
        pass
    return res

def saccade_metrics_from_labels(x, y, v, lab):
    # Quick-and-dirty saccade summary from labels: count, mean amplitude, 95th percentile peak speed.
    if lab is None or pd.isna(lab).all() or v is None or len(v)==0:
        return 0, np.nan, np.nan
    idx = np.where(lab.values[:-1] == 2)[0]
    cnt = len(idx)
    if cnt == 0: return 0, np.nan, np.nan
    amp = np.sqrt(np.diff(x.values)[idx]**2 + np.diff(y.values)[idx]**2)
    peak = np.nanpercentile(v[idx], 95) if len(idx) else np.nan
    return int(cnt), (float(np.nanmean(amp)) if len(amp) else np.nan), (float(peak) if np.isfinite(peak) else np.nan)

def compute_trial_features(df_valid, df_orig, source_file):
    # Trial-level feature extractor: condenses thousands of samples into a biomarker vector.
    subj, sess, task = parse_ids(source_file)
    n, x, y = df_valid['n'], df_valid['x'], df_valid['y']
    dP = df_valid['dP'] if 'dP' in df_valid.columns else pd.Series([np.nan]*len(df_valid))
    lab = df_valid['lab'] if 'lab' in df_valid.columns else pd.Series([np.nan]*len(df_valid))
    xT  = df_valid['xT'] if 'xT' in df_valid.columns else pd.Series([np.nan]*len(df_valid))
    yT  = df_valid['yT'] if 'yT' in df_valid.columns else pd.Series([np.nan]*len(df_valid))

    # Temporal structure
    dt_med = robust_dt_ms(n.values)
    duration_s = (float(n.iloc[-1]) - float(n.iloc[0]))/1000.0 if len(n)>1 else 0.0

    # Kinematics
    vx, vy, v = velocities(x.values, y.values, n.values)
    mean_speed = float(np.nanmean(v)) if len(v)>0 else np.nan
    std_speed  = float(np.nanstd(v))  if len(v)>0 else np.nan
    p95_speed  = float(np.nanpercentile(v,95)) if len(v)>0 else np.nan

    # Fixation dispersion and drift
    drift = drift_velocity(x, y, n)
    fix_std_x = float(np.nanstd(x.values))
    fix_std_y = float(np.nanstd(y.values))
    bcea = bcea_proxy(x.values, y.values)

    # Pupil dynamics
    pupil_mean = float(np.nanmean(dP)) if len(dP)>0 else np.nan
    pupil_std  = float(np.nanstd(dP))  if len(dP)>0 else np.nan

    # Event counts and label-derived saccade metrics
    cnt_fix, cnt_sac, cnt_blink = event_counts(lab)
    cnt_sac_lab, sac_amp_mean, sac_peak_v = saccade_metrics_from_labels(x, y, v, lab)

    # Pursuit quality (if target present) + fraction invalid (pre-filter QA)
    purs = pursuit_metrics(x, y, xT, yT, n)
    frac_invalid = float((df_orig['val'] != 0).mean()) if 'val' in df_orig.columns else np.nan

    return {
        "source_file": source_file, "subject_id": subj, "session": sess, "task": task,
        "num_samples": int(len(df_valid)), "dt_med_ms": dt_med, "duration_s": duration_s,
        "fix_std_x": fix_std_x, "fix_std_y": fix_std_y, "bcea_proxy": bcea,
        "mean_speed": mean_speed, "std_speed": std_speed, "p95_speed": p95_speed,
        "drift_vel": drift, "pupil_mean": pupil_mean, "pupil_std": pupil_std,
        "cnt_fix": cnt_fix, "cnt_sac": cnt_sac, "cnt_blink": cnt_blink,
        "cnt_sac_lab": cnt_sac_lab, "sac_amp_mean": sac_amp_mean, "sac_peak_v": sac_peak_v,
        **purs, "frac_invalid": frac_invalid
    }

# -------- Unzip all Subject_*.zip and collect CSVs --------
def unzip_all_and_collect_csvs(data_dir: Path):
    # Convenience: expand each Subject_*.zip once, then crawl for trial CSVs.
    csvs = []
    for zp in data_dir.glob("Subject_*.zip"):
        out = data_dir / zp.stem
        if not out.exists():
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(out)
        for p,_,files in os.walk(out):
            for nm in files:
                if nm.endswith(".csv"):
                    csvs.append(Path(p)/nm)
    return csvs

csv_files = unzip_all_and_collect_csvs(DATA_DIR)
assert len(csv_files) > 0, "No CSV files found. Place Subject_*.zip in the working folder."

# -------- Step 1: Preprocessing by trial --------
# Strategy: strict validity filter (val==0), drop rows missing both x and y, skip trivially short trials.
feature_rows, skipped = [], []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        orig = df.copy()
        df = df[df['val'] == 0].copy()                  # retain only valid samples
        df = df[~(df['x'].isna() & df['y'].isna())]     # drop rows with no gaze position
        if len(df) < MIN_VALID_SAMPLES:
            skipped.append((f.name, "Too few valid samples")); continue
        feats = compute_trial_features(df, orig, f.name)
        feature_rows.append(feats)
    except Exception as e:
        skipped.append((f.name, str(e)))

df_feat = pd.DataFrame(feature_rows)
feat_path = OUT_DIR / "GB_TrialFeatures_UpToAnomaly.csv"
df_feat.to_csv(feat_path, index=False)

# -------- Step 2: Anomaly Detection (IsolationForest) --------
# Model input: fixed set of biomarkers; missing/inf handled via NaN->0 imputation for tree models.
FEATURE_COLS = [
    'num_samples','dt_med_ms','duration_s','fix_std_x','fix_std_y','bcea_proxy',
    'mean_speed','std_speed','p95_speed','drift_vel','pupil_mean','pupil_std',
    'cnt_fix','cnt_sac','cnt_blink','cnt_sac_lab','sac_amp_mean','sac_peak_v',
    'purs_corr_x','purs_corr_y','purs_gain_x','purs_gain_y','purs_lag_ms','frac_invalid'
]

X = df_feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
iso.fit(X)
scores = -iso.score_samples(X)  # increasing with “outlierness”
thr = float(np.quantile(scores, 1.0 - CONTAMINATION))  # data-driven threshold at (1 − contamination)

df_feat['anomaly_score'] = scores
df_feat['anomaly_threshold'] = thr
df_feat['anomaly_flag'] = (scores > thr).astype(int)

anom_path = OUT_DIR / "GB_AnomalyScores_PerTrial.csv"
df_feat.to_csv(anom_path, index=False)

# Visualization: score distribution with the learned cutoff.
plt.figure(figsize=(7,5))
plt.hist(scores, bins=30)
plt.axvline(thr, linestyle="--")
plt.title("Anomaly Score Distribution (IsolationForest, contamination=0.05)")
plt.xlabel("Anomaly score (higher = more anomalous)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "GB_AnomalyScore_Hist.png", dpi=150)
plt.close()

# Provenance + diagnostics: record skipped trials and basic counts.
readme = OUT_DIR / "README_UpToAnomaly.txt"
with open(readme, "w") as f:
    f.write("GazeBase anomaly detection deliverables (up to anomaly stage)\n")
    f.write(f"Trials processed: {len(df_feat)}\n")
    f.write(f"Trials flagged anomalous (>{thr:.4f}): {(df_feat['anomaly_flag']==1).sum()}\n")
    f.write(f"Contamination: {CONTAMINATION}\n\n")
    if skipped:
        f.write("Skipped trials:\n")
        for nm, reason in skipped:
            f.write(f"- {nm}: {reason}\n")

print("DONE")
print("Outputs:")
print(" -", feat_path)
print(" -", anom_path)
print(" -", OUT_DIR / "GB_AnomalyScore_Hist.png")
print(" -", readme)