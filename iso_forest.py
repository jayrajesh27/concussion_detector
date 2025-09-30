"""
gaze_anomaly_isoforest.py

What this script does:
- Unzips GazeBase subject folders (Subject_XXXX.zip) if needed.
- Reads every trial CSV (one trial = one file).
- Keeps only valid gaze samples (val == 0) and drops empty rows.
- Computes a bunch of per-trial features (fixation spread, saccade stats, pupil stuff, pursuit if available).
- Trains an Isolation Forest (unsupervised) to find "weird" trials (set at ~5% anomalies).
- Saves a CSV with trial features + anomaly score + a flag if it's above the threshold.
- Also saves a histogram of anomaly scores and a small README with basic stats.

How to run (from the folder that has Subject_*.zip):
    python gaze_anomaly_isoforest.py

Outputs (go to ./outputs):
    - GB_TrialFeatures_UpToAnomaly.csv
    - GB_AnomalyScores_PerTrial.csv
    - GB_AnomalyScore_Hist.png
    - README_UpToAnomaly.txt
"""

import os, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.ensemble import IsolationForest

# Author: Jay Rajesh


# ----------------- basic config I’m using -----------------
DATA_DIR = Path(".")         # where Subject_*.zip live (or their unzipped folders)
OUT_DIR  = Path("./outputs") # where I save results
CONTAMINATION = 0.05         # IsolationForest “assumes” ~5% are anomalies
MIN_VALID_SAMPLES = 10       # if a trial is teeny after filtering, I skip it

OUT_DIR.mkdir(exist_ok=True, parents=True)


# ----------------- small helper functions -----------------
def parse_ids(fname):
    """Grab subject/session/task from filenames like: S_1001_S2_HSS.csv"""
    m = re.match(r"S_(\d+)_S(\d+)_([A-Z0-9]+)", Path(fname).stem)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (None, None, None)

def robust_dt_ms(n):
    """Median time step (ms). Ignores non-positive jumps just in case timestamps glitch."""
    n = np.asarray(n, float)
    dn = np.diff(n)
    good = dn[dn > 0]
    return float(np.median(good)) if good.size else np.nan

def bcea_proxy(x, y):
    """Fixation dispersion proxy: 2π σx σy sqrt(1-ρ²). If zero std, area ~0."""
    if len(x) < 5: return np.nan
    sx, sy = np.nanstd(x), np.nanstd(y)
    if sx == 0 or sy == 0: return 0.0
    xv = np.nan_to_num(x, nan=np.nanmean(x))
    yv = np.nan_to_num(y, nan=np.nanmean(y))
    rho = np.corrcoef(xv, yv)[0, 1]
    rho = 0.0 if not np.isfinite(rho) else np.clip(rho, -0.9999, 0.9999)
    return float(2*np.pi*sx*sy*np.sqrt(1 - rho**2))

def velocities(x, y, n_ms):
    """Simple finite-difference velocities (deg/s). Returns vx, vy, and resultant speed v."""
    if len(x) < 3: return np.array([]), np.array([]), np.array([])
    t = np.asarray(n_ms, float) / 1000.0
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    dx, dy = np.diff(x), np.diff(y)
    vx, vy = dx/dt, dy/dt
    v = np.sqrt(vx**2 + vy**2)
    return vx, vy, v

def drift_velocity(x, y, n_ms):
    """How much the gaze slowly “drifts” across the trial (deg/s)."""
    if len(x) < 5: return np.nan
    t = np.asarray(n_ms, float) - float(n_ms[0])
    try:
        sx, _ = np.polyfit(t, x, 1)
        sy, _ = np.polyfit(t, y, 1)
        return float(np.sqrt(sx**2 + sy**2) * 1000.0)  # per ms -> per s
    except Exception:
        return np.nan

def event_counts(lab):
    """If labels exist: 1=fix, 2=sac, -1=blink. Otherwise zeros."""
    if lab is None or pd.isna(lab).all(): return 0, 0, 0
    return int((lab==1).sum()), int((lab==2).sum()), int((lab==-1).sum())

def pursuit_metrics(x, y, xT, yT, n_ms):
    """
    If target traces are present (xT,yT):
      - corr (how well eye follows target)
      - gain (eye velocity variability / target velocity variability)
      - lag (ms) via cross-correlation peak
    If any piece is missing, I return NaNs.
    """
    res = dict(purs_corr_x=np.nan, purs_corr_y=np.nan,
               purs_gain_x=np.nan, purs_gain_y=np.nan, purs_lag_ms=np.nan)
    if pd.isna(xT).all() and pd.isna(yT).all(): return res

    valid = (~pd.isna(x)) & (~pd.isna(y))
    if (~pd.isna(xT)).any(): valid &= (~pd.isna(xT))
    if (~pd.isna(yT)).any(): valid &= (~pd.isna(yT))

    x, y, xT, yT, t = x[valid], y[valid], xT[valid], yT[valid], n_ms[valid]
    if len(x) < 10: return res

    def safe_corr(a,b): return np.corrcoef(a,b)[0,1] if (np.std(a)>0 and np.std(b)>0) else np.nan
    res["purs_corr_x"] = float(safe_corr(x.values, xT.values)) if (~pd.isna(xT)).any() else np.nan
    res["purs_corr_y"] = float(safe_corr(y.values, yT.values)) if (~pd.isna(yT)).any() else np.nan

    vx, vy, _ = velocities(x.values, y.values, t.values)
    vTx, vTy, _ = velocities(xT.values, yT.values, t.values)

    def ratio_std(a,b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a)<5 or len(b)<5 or np.std(b)==0: return np.nan
        return float(np.std(a)/np.std(b))
    res["purs_gain_x"] = ratio_std(vx, vTx)
    res["purs_gain_y"] = ratio_std(vy, vTy)

    try:
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
    """
    Quick saccade summary using labels (if lab exists):
    - count (# of frames labeled as saccade)
    - mean amplitude (deg) using position diffs
    - 95th percentile of speed during labeled saccade frames
    """
    if lab is None or pd.isna(lab).all() or v is None or len(v)==0:
        return 0, np.nan, np.nan
    idx = np.where(lab.values[:-1] == 2)[0]
    cnt = len(idx)
    if cnt == 0: return 0, np.nan, np.nan
    amp = np.sqrt(np.diff(x.values)[idx]**2 + np.diff(y.values)[idx]**2)
    peak = np.nanpercentile(v[idx], 95) if len(idx) else np.nan
    return int(cnt), (float(np.nanmean(amp)) if len(amp) else np.nan), (float(peak) if np.isfinite(peak) else np.nan)

def compute_trial_features(df_valid, df_orig, source_file):
    """This is where I compress a whole trial into a single feature row."""
    subj, sess, task = parse_ids(source_file)
    n, x, y = df_valid['n'], df_valid['x'], df_valid['y']
    dP = df_valid.get('dP', pd.Series([np.nan]*len(df_valid)))
    lab = df_valid.get('lab', pd.Series([np.nan]*len(df_valid)))
    xT  = df_valid.get('xT',  pd.Series([np.nan]*len(df_valid)))
    yT  = df_valid.get('yT',  pd.Series([np.nan]*len(df_valid)))

    # time structure of the recording
    dt_med = robust_dt_ms(n.values)
    duration_s = (float(n.iloc[-1]) - float(n.iloc[0]))/1000.0 if len(n)>1 else 0.0

    # kinematics (simple speed summary)
    vx, vy, v = velocities(x.values, y.values, n.values)
    mean_speed = float(np.nanmean(v)) if len(v)>0 else np.nan
    std_speed  = float(np.nanstd(v))  if len(v)>0 else np.nan
    p95_speed  = float(np.nanpercentile(v,95)) if len(v)>0 else np.nan

    # fixations & drift
    drift = drift_velocity(x, y, n)
    fix_std_x = float(np.nanstd(x.values))
    fix_std_y = float(np.nanstd(y.values))
    bcea = bcea_proxy(x.values, y.values)

    # pupil
    pupil_mean = float(np.nanmean(dP)) if len(dP)>0 else np.nan
    pupil_std  = float(np.nanstd(dP))  if len(dP)>0 else np.nan

    # labels (optional)
    cnt_fix, cnt_sac, cnt_blink = event_counts(lab)
    cnt_sac_lab, sac_amp_mean, sac_peak_v = saccade_metrics_from_labels(x, y, v, lab)

    # pursuit stuff (if target exists)
    purs = pursuit_metrics(x, y, xT, yT, n)

    # QA: fraction invalid before we filtered
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


# ----------------- read all the CSVs (unzips first time) -----------------
def unzip_all_and_collect_csvs(data_dir: Path):
    csvs = []
    for zp in data_dir.glob("Subject_*.zip"):
        out = data_dir / zp.stem
        if not out.exists():
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(out)
        for p, _, files in os.walk(out):
            for nm in files:
                if nm.endswith(".csv"):
                    csvs.append(Path(p) / nm)
    return csvs


# ----------------- main: preprocess -> features -> anomalies -----------------
def main():
    csv_files = unzip_all_and_collect_csvs(DATA_DIR)
    assert len(csv_files) > 0, "No CSVs found. Put Subject_*.zip in this folder."

    # 1) per-trial features
    rows, skipped = [], []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            orig = df.copy()
            # keep only valid rows and drop rows that don’t have any (x,y)
            df = df[df['val'] == 0].copy()
            df = df[~(df['x'].isna() & df['y'].isna())]
            if len(df) < MIN_VALID_SAMPLES:
                skipped.append((f.name, "too few valid samples")); continue
            rows.append(compute_trial_features(df, orig, f.name))
        except Exception as e:
            skipped.append((f.name, str(e)))

    df_feat = pd.DataFrame(rows)
    feat_path = OUT_DIR / "GB_TrialFeatures_UpToAnomaly.csv"
    df_feat.to_csv(feat_path, index=False)

    # 2) isolation forest on the feature matrix
    FEATURE_COLS = [
        'num_samples','dt_med_ms','duration_s','fix_std_x','fix_std_y','bcea_proxy',
        'mean_speed','std_speed','p95_speed','drift_vel','pupil_mean','pupil_std',
        'cnt_fix','cnt_sac','cnt_blink','cnt_sac_lab','sac_amp_mean','sac_peak_v',
        'purs_corr_x','purs_corr_y','purs_gain_x','purs_gain_y','purs_lag_ms','frac_invalid'
    ]
    X = df_feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    iso.fit(X)

    # decision_function/score_samples return "more normal = higher".
    # I flip sign so "higher = more anomalous" which feels more intuitive for plotting.
    scores = -iso.score_samples(X)
    thr = float(np.quantile(scores, 1.0 - CONTAMINATION))

    df_feat['anomaly_score'] = scores
    df_feat['anomaly_threshold'] = thr
    df_feat['anomaly_flag'] = (scores > thr).astype(int)

    anom_path = OUT_DIR / "GB_AnomalyScores_PerTrial.csv"
    df_feat.to_csv(anom_path, index=False)

    # 3) simple histogram of scores (with threshold line)
    plt.figure(figsize=(7,5))
    plt.hist(scores, bins=30)
    plt.axvline(thr, linestyle="--")
    plt.title("Anomaly Score Distribution (IsolationForest, contamination=0.05)")
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "GB_AnomalyScore_Hist.png", dpi=200)
    plt.close()

    # 4) tiny README so I remember what happened
    with open(OUT_DIR / "README_UpToAnomaly.txt", "w") as f:
        f.write("GazeBase anomaly detection (up to anomaly stage)\n")
        f.write(f"Trials processed: {len(df_feat)}\n")
        f.write(f"Trials flagged anomalous (>{thr:.4f}): {(df_feat['anomaly_flag']==1).sum()}\n")
        f.write(f"Contamination: {CONTAMINATION}\n\n")
        if skipped:
            f.write("Skipped trials:\n")
            for nm, reason in skipped:
                f.write(f"- {nm}: {reason}\n")

    print("Done. Saved:")
    print(" -", feat_path)
    print(" -", anom_path)
    print(" -", OUT_DIR / "GB_AnomalyScore_Hist.png")
    print(" -", OUT_DIR / "README_UpToAnomaly.txt")


if __name__ == "__main__":
    main()