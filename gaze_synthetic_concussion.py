"""
gaze_synthetic_concussion.py

- Starts from real healthy trials (same CSVs as anomaly script).
- For each valid trial, makes a "synthetic concussed" version by perturbing the time series:
    * lower pursuit gain + add time lag,
    * add low-frequency jitter to x/y (fixation instability),
    * increase saccade latency (if labels exist),
    * increase pupil variability.
- Recomputes the exact same trial features on both healthy and synthetic versions.
- Builds a labeled dataset (0 = healthy, 1 = synthetic).
- Trains a RandomForest to tell them apart using subject-wise CV (no subject leakage).
- Saves metrics (AUC, accuracy) + ROC plot + feature importances.

How to run (from the folder that has Subject_*.zip):
    python gaze_synthetic_concussion.py

Outputs (go to ./outputs_synth):
    - GB_Synth_TrialFeatures.csv
    - ROC.png
    - FeatureImportances.png
    - clf_metrics.json
"""

import os, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal, interpolate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GroupKFold

# Author: Jay Rajesh

DATA_DIR = Path(".")
OUT_DIR  = Path("./outputs_synth")
MIN_VALID_SAMPLES = 10
RANDOM_STATE = 42
N_TREES = 400
N_SPLITS = 5                 # GroupKFold splits by subject
MAX_TRIALS_PER_SUBJECT = None  # set an int (e.g., 30) to cut runtime

# “Concussion-like” perturbation magnitudes (tweak if needed)
PURSUIT_GAIN_RANGE = (0.7, 0.9)   # scale down gaze vs target
PURSUIT_LAG_MS = (50, 100)        # shift in time (ms)
SACCADE_LATENCY_MS = 50           # push saccade onset later (approx)
FIXATION_JITTER_SD = 0.5          # low-frequency jitter amplitude (deg)
PUPIL_STD_SCALE = 1.5             # increase pupil variance


# ----------------- helpers reused from anomaly script -----------------
def parse_ids(fname):
    m = re.match(r"S_(\d+)_S(\d+)_([A-Z0-9]+)", Path(fname).stem)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (None, None, None)

def robust_dt_ms(n):
    n = np.asarray(n, float)
    dn = np.diff(n); good = dn[dn > 0]
    return float(np.median(good)) if good.size else np.nan

def bcea_proxy(x, y):
    if len(x) < 5: return np.nan
    sx, sy = np.nanstd(x), np.nanstd(y)
    if sx == 0 or sy == 0: return 0.0
    xv = np.nan_to_num(x, nan=np.nanmean(x))
    yv = np.nan_to_num(y, nan=np.nanmean(y))
    rho = np.corrcoef(xv, yv)[0,1]
    rho = 0.0 if not np.isfinite(rho) else np.clip(rho, -0.9999, 0.9999)
    return float(2*np.pi*sx*sy*np.sqrt(1 - rho**2))

def velocities(x, y, n_ms):
    if len(x) < 3: return np.array([]), np.array([]), np.array([])
    t = np.asarray(n_ms, float) / 1000.0
    dt = np.diff(t); dt[dt <= 0] = np.nan
    dx, dy = np.diff(x), np.diff(y)
    vx, vy = dx/dt, dy/dt
    v = np.sqrt(vx**2 + vy**2)
    return vx, vy, v

def drift_velocity(x, y, n_ms):
    if len(x) < 5: return np.nan
    t = np.asarray(n_ms, float) - float(n_ms[0])
    try:
        sx, _ = np.polyfit(t, x, 1)
        sy, _ = np.polyfit(t, y, 1)
        return float(np.sqrt(sx**2 + sy**2) * 1000.0)
    except Exception:
        return np.nan

def event_counts(lab):
    if lab is None or pd.isna(lab).all(): return 0,0,0
    return int((lab==1).sum()), int((lab==2).sum()), int((lab==-1).sum())

def pursuit_metrics(x, y, xT, yT, n_ms):
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
    if lab is None or pd.isna(lab).all() or v is None or len(v)==0:
        return 0, np.nan, np.nan
    idx = np.where(lab.values[:-1] == 2)[0]
    cnt = len(idx)
    if cnt == 0: return 0, np.nan, np.nan
    amp = np.sqrt(np.diff(x.values)[idx]**2 + np.diff(y.values)[idx]**2)
    peak = np.nanpercentile(v[idx], 95) if len(idx) else np.nan
    return int(cnt), (float(np.nanmean(amp)) if len(amp) else np.nan), (float(peak) if np.isfinite(peak) else np.nan)

def compute_trial_features(df_valid, df_orig, source_file):
    subj, sess, task = parse_ids(source_file)
    n, x, y = df_valid['n'], df_valid['x'], df_valid['y']
    dP = df_valid.get('dP', pd.Series([np.nan]*len(df_valid)))
    lab = df_valid.get('lab', pd.Series([np.nan]*len(df_valid)))
    xT  = df_valid.get('xT',  pd.Series([np.nan]*len(df_valid)))
    yT  = df_valid.get('yT',  pd.Series([np.nan]*len(df_valid)))

    dt_med = robust_dt_ms(n.values)
    duration_s = (float(n.iloc[-1]) - float(n.iloc[0]))/1000.0 if len(n)>1 else 0.0

    vx, vy, v = velocities(x.values, y.values, n.values)
    mean_speed = float(np.nanmean(v)) if len(v)>0 else np.nan
    std_speed  = float(np.nanstd(v))  if len(v)>0 else np.nan
    p95_speed  = float(np.nanpercentile(v,95)) if len(v)>0 else np.nan

    drift = drift_velocity(x, y, n)
    fix_std_x = float(np.nanstd(x.values))
    fix_std_y = float(np.nanstd(y.values))
    bcea = bcea_proxy(x.values, y.values)

    pupil_mean = float(np.nanmean(dP)) if len(dP)>0 else np.nan
    pupil_std  = float(np.nanstd(dP))  if len(dP)>0 else np.nan

    cnt_fix, cnt_sac, cnt_blink = event_counts(lab)
    cnt_sac_lab, sac_amp_mean, sac_peak_v = saccade_metrics_from_labels(x, y, v, lab)

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


# ----------------- file IO helpers -----------------
def unzip_all_and_collect_csvs(data_dir: Path):
    csvs = []
    for zp in data_dir.glob("Subject_*.zip"):
        out = data_dir / zp.stem
        if not out.exists():
            import zipfile
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(out)
        for p, _, files in os.walk(out):
            for nm in files:
                if nm.endswith(".csv"):
                    csvs.append(Path(p)/nm)
    return csvs


# ----------------- synthetic perturbations (time-series level) -----------------
def low_freq_jitter(n_ms, sd=0.5, seed=RANDOM_STATE):
    """Low-frequency jiggle (like wandering fixations)."""
    rng = np.random.default_rng(seed)
    t = (np.asarray(n_ms, float) - float(n_ms.iloc[0]))/1000.0
    noise = rng.normal(0, sd, size=len(t))
    b, a = signal.butter(2, 0.05)  # super low-pass
    return signal.filtfilt(b, a, noise)

def shift_series_time(x, n_ms, lag_ms):
    """Shift a series forward by lag_ms using linear interpolation."""
    if len(x) < 3: return x.copy()
    t = np.asarray(n_ms, float)
    t_shift = t + lag_ms
    f = interpolate.interp1d(t, x, kind='linear', bounds_error=False, fill_value="extrapolate")
    return pd.Series(f(t_shift), index=x.index)

def make_synthetic_trial(df):
    """Make a synthetic 'concussion-like' copy of a valid trial."""
    df_syn = df.copy()

    # 1) pursuit impairment (gain down + add lag)
    gain = np.random.uniform(*PURSUIT_GAIN_RANGE)
    lag  = np.random.uniform(*PURSUIT_LAG_MS)
    df_syn['x'] = gain * shift_series_time(df_syn['x'].astype(float), df_syn['n'], lag)
    df_syn['y'] = gain * shift_series_time(df_syn['y'].astype(float), df_syn['n'], lag)

    # 2) fixation instability (slow jitter)
    df_syn['x'] = df_syn['x'] + low_freq_jitter(df_syn['n'], sd=FIXATION_JITTER_SD, seed=np.random.randint(1e6))
    df_syn['y'] = df_syn['y'] + low_freq_jitter(df_syn['n'], sd=FIXATION_JITTER_SD, seed=np.random.randint(1e6))

    # 3) saccade latency (push lab==2 later if labels exist)
    if 'lab' in df_syn.columns:
        dt_med = robust_dt_ms(df_syn['n'].values)
        if np.isfinite(dt_med) and dt_med > 0:
            steps = int(round(SACCADE_LATENCY_MS / dt_med))
            lab = df_syn['lab'].copy()
            lab_idx = (lab == 2).values
            lab_shifted = np.zeros_like(lab.values)
            if steps > 0:
                lab_shifted[steps:] = lab_idx[:-steps]
            else:
                lab_shifted = lab_idx
            df_syn['lab'] = np.where(lab_shifted==1, 2, df_syn['lab'])

    # 4) pupil variability increase
    if 'dP' in df_syn.columns:
        dP = df_syn['dP'].astype(float)
        noise = np.random.normal(0, np.std(dP) * (PUPIL_STD_SCALE - 1.0), size=len(dP))
        df_syn['dP'] = dP + noise

    return df_syn


# ----------------- main: build labeled features + train classifier -----------------
def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    csv_files = unzip_all_and_collect_csvs(DATA_DIR)
    if len(csv_files) == 0:
        raise RuntimeError("No CSVs found under Subject_* folders.")

    rows = []
    per_subj_count = {}

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            subj, _, _ = parse_ids(f.name)
            if subj is None: continue

            # cap per-subject trials if I want faster runs
            cur = per_subj_count.get(subj, 0)
            if MAX_TRIALS_PER_SUBJECT is not None and cur >= MAX_TRIALS_PER_SUBJECT:
                continue

            # valid-only (same as anomaly script)
            df_valid = df[df['val'] == 0].copy()
            df_valid = df_valid[~(df_valid['x'].isna() & df_valid['y'].isna())]
            if len(df_valid) < MIN_VALID_SAMPLES:
                continue

            # (A) healthy features (label 0)
            feats_h = compute_trial_features(df_valid, df, f.name)
            feats_h['label'] = 0
            rows.append(feats_h)

            # (B) synthetic perturbed features (label 1)
            df_syn = make_synthetic_trial(df_valid)
            feats_s = compute_trial_features(df_syn, df, "SYNT_" + f.name)
            feats_s['label'] = 1
            rows.append(feats_s)

            per_subj_count[subj] = cur + 1

        except Exception:
            # if something blows up, just skip that file
            continue

    df_feat = pd.DataFrame(rows)
    if len(df_feat) == 0:
        raise RuntimeError("No features computed. Check inputs or filters.")

    # save feature table so I can reuse
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    feat_path = OUT_DIR / "GB_Synth_TrialFeatures.csv"
    df_feat.to_csv(feat_path, index=False)

    # set up classification problem
    feature_cols = [
        'num_samples','dt_med_ms','duration_s','fix_std_x','fix_std_y','bcea_proxy',
        'mean_speed','std_speed','p95_speed','drift_vel','pupil_mean','pupil_std',
        'cnt_fix','cnt_sac','cnt_blink','cnt_sac_lab','sac_amp_mean','sac_peak_v',
        'purs_corr_x','purs_corr_y','purs_gain_x','purs_gain_y','purs_lag_ms','frac_invalid'
    ]
    X = df_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = df_feat['label'].values
    groups = df_feat['subject_id'].values  # so train/test are different subjects

    # random forest (simple + works well out-of-the-box)
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    # subject-wise cross-val (so no leakage)
    gkf = GroupKFold(n_splits=N_SPLITS)
    y_true_all, y_prob_all = [], []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        y_true_all.append(y[te_idx]); y_prob_all.append(probs)

    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)

    # metrics: AUC and accuracy at 0.5 (can tune later)
    auc = roc_auc_score(y_true_all, y_prob_all)
    y_pred = (y_prob_all >= 0.5).astype(int)
    acc = accuracy_score(y_true_all, y_pred)

    # ROC plot
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Healthy vs Synthetic (GroupKFold by Subject)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ROC.png", dpi=200)
    plt.close()

    # Feature importances (refit on all for a clean bar chart)
    clf.fit(X, y)
    importances = clf.feature_importances_
    fi = (pd.DataFrame({"feature": feature_cols, "importance": importances})
          .sort_values("importance", ascending=False))

    plt.figure(figsize=(7,6))
    plt.barh(fi["feature"][:15][::-1], fi["importance"][:15][::-1])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "FeatureImportances.png", dpi=200)
    plt.close()

    # save metrics as json so I can quote numbers easily
    metrics = {"roc_auc": float(auc), "accuracy_at_0.5": float(acc)}
    with open(OUT_DIR / "clf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Saved:")
    print(" -", feat_path)
    print(" -", OUT_DIR / "ROC.png")
    print(" -", OUT_DIR / "FeatureImportances.png")
    print(" -", OUT_DIR / "clf_metrics.json")


if __name__ == "__main__":
    main()