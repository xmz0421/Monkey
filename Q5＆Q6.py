import os
import warnings
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score

# -------------------------
# Robust loader (your improved version)
# -------------------------
def find_dataset_group(root, keywords):
    for k in root.keys():
        lower = k.lower()
        for kw in keywords:
            if kw in lower:
                return root[k]
    for k in root.keys():
        if isinstance(root[k], h5py.Group):
            found = find_dataset_group(root[k], keywords)
            if found is not None:
                return found
    return None

def load_session(path, prefer_finger=True, verbose=True):
    if verbose: print(f"[load_session] opening {path}")
    with h5py.File(path, 'r') as f:
        t_ds = find_dataset_group(f, ['t', 'time', 'timestamps'])
        if t_ds is None:
            raise RuntimeError("Cannot find timestamps dataset.")
        t = np.array(t_ds).squeeze()
        if verbose: print(f"  t shape: {t.shape}")

        pos_ds = None
        if prefer_finger:
            pos_ds = find_dataset_group(f, ['finger_pos', 'fingerpos', 'hand_pos'])
        if pos_ds is None:
            pos_ds = find_dataset_group(f, ['cursor_pos', 'cursorpos', 'cursor', 'pos', 'position'])
        if pos_ds is None:
            raise RuntimeError("Cannot find finger_pos or cursor_pos dataset.")
        pos_arr = np.array(pos_ds)
        # heuristics to shape pos into (T, >=2)
        if pos_arr.ndim == 2:
            if pos_arr.shape[0] == t.size:
                pos = pos_arr
            elif pos_arr.shape[1] == t.size:
                pos = pos_arr.T
            else:
                pos = pos_arr.T if pos_arr.T.shape[0] == t.size else pos_arr
        elif pos_arr.ndim == 1:
            if pos_arr.size % 2 == 0:
                pos = pos_arr.reshape(-1, 2)
            else:
                pos = pos_arr.reshape(-1, 3)[:, :2]
        else:
            pos = pos_arr.reshape(t.size, -1)
        if pos.shape[1] >= 2:
            pos = pos[:, :2]
        if verbose: print(f"  pos shape (T,2): {pos.shape}")

        ch_names_ds = find_dataset_group(f, ['chan_names', 'ch_names', 'channel', 'labels'])
        ch_names = None
        if ch_names_ds is not None:
            try:
                arr = np.array(ch_names_ds)
                ch_list = []
                for s in arr.ravel():
                    if isinstance(s, (bytes, np.bytes_)):
                        ch_list.append(s.decode('utf-8', errors='ignore'))
                    else:
                        ch_list.append(str(s))
                ch_names = ch_list
            except Exception:
                ch_names = None
        if verbose: print(f"  chan names found: {None if ch_names is None else len(ch_names)}")

        spikes_ds = find_dataset_group(f, ['spike', 'spikes', 'unit', 'spike_times', 'times'])
        spike_list = None
        spike_matrix = None
        if spikes_ds is not None:
            ds = spikes_ds
            arr = np.array(ds)
            # detect reference/object array
            if arr.dtype == object or str(arr.dtype).startswith('ref') or arr.dtype.kind == 'O':
                # cell array of references
                if arr.ndim == 2:
                    n_units = arr.shape[1]
                    spike_list = []
                    for i in range(n_units):
                        try:
                            ref = arr[0, i]
                            sp = np.array(f[ref]).squeeze()
                            spike_list.append(np.asarray(sp).ravel())
                        except Exception:
                            spike_list.append(np.array([]))
                elif arr.ndim == 1:
                    spike_list = []
                    for i in range(arr.shape[0]):
                        try:
                            ref = arr[i]
                            sp = np.array(f[ref]).squeeze()
                            spike_list.append(np.asarray(sp).ravel())
                        except Exception:
                            spike_list.append(np.array([]))
                else:
                    spike_list = None
            else:
                # numeric matrix
                mat = np.asarray(arr)
                if mat.ndim == 2 and mat.shape[0] == t.size:
                    spike_matrix = mat  # time x units
                elif mat.ndim == 2 and mat.shape[1] == t.size:
                    spike_matrix = mat.T
                else:
                    spike_matrix = mat
        return {'t': np.asarray(t).ravel(), 'pos': np.asarray(pos), 'spike_list': spike_list,
                'spike_matrix': spike_matrix, 'ch_names': ch_names}

# -------------------------
# Binning & smoothing
# -------------------------
def compute_binned_rates(session, win_s=0.064, smooth_sigma_s=None, verbose=False):
    t = session['t']
    spike_list = session.get('spike_list', None)
    spike_matrix = session.get('spike_matrix', None)
    t_start, t_end = float(np.nanmin(t)), float(np.nanmax(t))
    bins = np.arange(t_start, t_end + 1e-9, win_s)
    bin_centers = bins[:-1] + win_s / 2.0
    n_bins = len(bin_centers)

    if spike_list is not None:
        n_units = len(spike_list)
        rates = np.zeros((n_units, n_bins), dtype=float)
        for i, st in enumerate(spike_list):
            if st is None or len(st) == 0:
                continue
            st = np.asarray(st).ravel()
            counts, _ = np.histogram(st, bins=bins)
            rates[i, :] = counts / win_s
        source = 'list'
    elif spike_matrix is not None:
        mat = np.asarray(spike_matrix)
        if mat.shape[0] == t.size:
            n_units = mat.shape[1]
            rates = np.zeros((n_units, n_bins), dtype=float)
            inds = np.digitize(t, bins) - 1
            for b in range(n_bins):
                sel = inds == b
                if not np.any(sel): continue
                counts = mat[sel, :].sum(axis=0)
                rates[:, b] = counts / win_s
            source = 'matrix_time_rows'
        elif mat.shape[1] == t.size:
            # units x time -> transpose and recurse
            session_local = dict(session)
            session_local['spike_matrix'] = mat.T
            return compute_binned_rates(session_local, win_s, smooth_sigma_s, verbose)
        else:
            # fallback heuristic
            raise RuntimeError("Cannot interpret spike_matrix shape relative to t.")
    else:
        raise RuntimeError("No spike data found in session.")

    # optional smoothing (Gaussian via simple 1D conv with gaussian kernel)
    if smooth_sigma_s is not None and smooth_sigma_s > 0:
        sigma_bins = max(1, int(round(smooth_sigma_s / win_s)))
        from scipy.ndimage import gaussian_filter1d
        rates = gaussian_filter1d(rates, sigma=sigma_bins, axis=1)
    if verbose:
        print(f"  Binned rates: {rates.shape}, source={source}")
    return rates, bin_centers

# -------------------------
# Utilities & analysis helpers
# -------------------------
def safe_cohens_d(a, b):
    # independent samples
    var_a = np.nanvar(a, ddof=1)
    var_b = np.nanvar(b, ddof=1)
    pooled = np.sqrt((var_a + var_b) / 2.0)
    if pooled < 1e-9 or np.isnan(pooled):
        return 0.0, True
    d = (np.nanmean(a) - np.nanmean(b)) / pooled
    return float(d), False

def cos_func(theta, A, PD, b):
    return A * np.cos(theta - PD) + b

# -------------------------
# Q5: detect stopping neurons robustly
# -------------------------
def detect_stopping_neurons(rates, speed, centers, speed_thr=None, min_epoch_len_bins=1, fdr_alpha=0.05):
    # threshold: 10th percentile if not provided
    if speed_thr is None:
        speed_thr = float(np.percentile(speed, 10))
    still_mask = speed <= speed_thr

    # enforce min epoch length (remove runs shorter than min_epoch_len_bins)
    def filter_runs(mask, min_len):
        if min_len <= 1:
            return mask
        N = len(mask)
        out = mask.copy()
        i = 0
        while i < N:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < N and mask[j + 1]:
                j += 1
            if (j - i + 1) < min_len:
                out[i:j+1] = False
            i = j + 1
        return out

    still_mask = filter_runs(still_mask, min_epoch_len_bins)

    results = []
    auc_list = []
    pvals = []
    for i in range(rates.shape[0]):
        fr_still = rates[i, still_mask]
        fr_move = rates[i, ~still_mask]
        # require some minimal samples
        if fr_still.size < 5 or fr_move.size < 5:
            results.append((i, np.nan, np.nan, np.nan, np.nan, np.nan, None))
            pvals.append(np.nan)
            auc_list.append(np.nan)
            continue
        mean_still = float(np.nanmean(fr_still))
        mean_move = float(np.nanmean(fr_move))
        d, varzero = safe_cohens_d(fr_still, fr_move)
        # t-test: may produce nan if constant; catch exceptions
        try:
            _, pval = ttest_ind(fr_still, fr_move, equal_var=False, nan_policy='omit')
            pval = float(pval) if not np.isnan(pval) else 1.0
        except Exception:
            pval = 1.0
        # AUC
        try:
            y_true = np.concatenate([np.ones(fr_still.size), np.zeros(fr_move.size)])
            y_score = np.concatenate([fr_still, fr_move])
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = np.nan
        # label rules
        label = None
        if pval < 0.05 and not varzero:
            if d > 0.35 and mean_still > mean_move:
                label = 'stopping'
            elif d < -0.35 and mean_still < mean_move:
                label = 'suppressed'
        results.append((i, mean_still, mean_move, d, pval, auc, label))
        pvals.append(pval)
        auc_list.append(auc)

    df = pd.DataFrame(results, columns=['neuron','mean_still','mean_move','cohen_d','pval','auc','label'])
    # FDR correction
    pvals_arr = np.array([1.0 if np.isnan(x) else x for x in pvals])
    rej, p_adj, _, _ = multipletests(pvals_arr, alpha=fdr_alpha, method='fdr_bh')
    df['p_adj'] = p_adj
    df['sig'] = rej
    return df, still_mask

# -------------------------
# Q6: tuning and decoding (robust)
# -------------------------
def compute_directional_tuning(rates, vel_binned, n_angle_bins=12, min_speed=1e-4):
    angles = np.arctan2(vel_binned[:,1], vel_binned[:,0])
    speed = np.linalg.norm(vel_binned, axis=1)
    sel = speed > min_speed
    if sel.sum() < 10:
        print("[W] too few movement samples for tuning")
    bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    results = []
    for i in range(rates.shape[0]):
        fr = rates[i, sel]
        angs = angles[sel]
        inds = np.digitize(angs, bins) - 1
        fr_means = np.zeros(n_angle_bins)
        valid_mask = np.zeros(n_angle_bins, dtype=bool)
        for b in range(n_angle_bins):
            idx = np.where(inds == b)[0]
            if idx.size == 0:
                fr_means[b] = np.nan
            else:
                fr_means[b] = fr[idx].mean()
                valid_mask[b] = True
        # If too few non-nan bins skip
        if valid_mask.sum() < max(3, n_angle_bins//3):
            results.append((i, np.nan, np.nan, np.nan, np.nan))
            continue
        # fill nans by circular interpolation (simple)
        if np.any(np.isnan(fr_means)):
            nan_idx = np.isnan(fr_means)
            good_x = centers[~nan_idx]
            good_y = fr_means[~nan_idx]
            try:
                from scipy.interpolate import interp1d
                f = interp1d(good_x, good_y, kind='cubic', fill_value="extrapolate")
                fr_means[nan_idx] = f(centers[nan_idx])
            except Exception:
                fr_means[nan_idx] = np.nanmean(fr_means[~nan_idx])
        # fit cosine
        try:
            A0 = (np.nanmax(fr_means) - np.nanmin(fr_means))/2.0
            PD0 = centers[np.nanargmax(fr_means)]
            b0 = np.nanmean(fr_means)
            popt, _ = curve_fit(cos_func, centers, fr_means, p0=[A0, PD0, b0], maxfev=5000)
            pred = cos_func(centers, *popt)
            denom = np.nansum((fr_means - np.nanmean(fr_means))**2)
            if denom <= 1e-12:
                r2 = 0.0
            else:
                r2 = 1.0 - np.nansum((fr_means - pred)**2) / denom
            results.append((i, float(popt[0]), float(popt[1]), float(popt[2]), float(r2)))
        except Exception:
            results.append((i, np.nan, np.nan, np.nan, np.nan))
    df = pd.DataFrame(results, columns=['neuron','A','PD','b','r2'])
    return df, centers

def decode_velocity_ridge(rates, vel_binned, alpha=1.0, n_splits=5):
    # rates: (units, T) ; vel_binned: (T,2)
    X = rates.T
    y = vel_binned
    if X.shape[1] == 0:
        print("[W] decode: no features (0 units); returning NaN")
        return np.array([np.nan, np.nan])
    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = []
    for tr, te in kf.split(X):
        # ensure not degenerate: if X[tr] has zero columns, skip
        try:
            m = Ridge(alpha=alpha).fit(X[tr], y[tr])
            p = m.predict(X[te])
            r2x = r2_score(y[te][:,0], p[:,0]) if p.shape[0] > 0 else np.nan
            r2y = r2_score(y[te][:,1], p[:,1]) if p.shape[0] > 0 else np.nan
            scores.append([r2x, r2y])
        except Exception as e:
            print("[W] ridge fit failed on a fold:", e)
            scores.append([np.nan, np.nan])
    scores = np.array(scores, dtype=float)
    # mean across folds ignoring NaNs
    return np.nanmean(scores, axis=0)

# -------------------------
# plotting helpers
# -------------------------
def plot_sample_psth(rates, centers, still_mask, neuron_indices, outpath):
    n = len(neuron_indices)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ni in zip(axes, neuron_indices):
        ax.plot(centers, rates[ni,:], lw=0.8)
        ax.fill_between(centers, ax.get_ylim()[0], ax.get_ylim()[1], where=still_mask, color='orange', alpha=0.2)
        ax.set_ylabel(f"unit {ni}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("PSTH (smoothed rates) with still epochs shaded")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_pd_unit_circle(df_tune, regions, outpath):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    for region, marker, alpha in [('M1','o',0.8),('S1','x',0.8),('Other','.',0.5)]:
        sub = df_tune[df_tune['region']==region]
        if sub.shape[0] > 0:
            ax.scatter(sub['PD'].values, np.abs(sub['A'].values), marker=marker, alpha=alpha, label=f"{region} (n={len(sub)})")
    ax.set_title("Preferred directions (radius=|A|)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1))
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_sample_tuning_curves(rates, vel_binned, neurons, centers_ang, outpath):
    angles = np.arctan2(vel_binned[:,1], vel_binned[:,0])
    bins = np.concatenate((centers_ang - (centers_ang[1]-centers_ang[0])/2, [centers_ang[-1]+(centers_ang[1]-centers_ang[0])/2]))
    fig, axes = plt.subplots(len(neurons), 1, figsize=(6, 2*len(neurons)), sharex=True)
    if len(neurons) == 1:
        axes = [axes]
    for ax, ni in zip(axes, neurons):
        inds = np.digitize(angles, bins) - 1
        fr_means = [np.nanmean(rates[ni, inds==b]) if np.any(inds==b) else np.nan for b in range(len(centers_ang))]
        axs = np.linspace(-np.pi, np.pi, 200)
        try:
            popt, _ = curve_fit(cos_func, centers_ang, np.nan_to_num(fr_means, nan=np.nanmean(fr_means)))
            ax.plot(centers_ang, fr_means, 'o-')
            ax.plot(axs, cos_func(axs, *popt), '--')
            ax.set_ylabel(f"unit {ni}")
        except Exception:
            ax.plot(centers_ang, fr_means, 'o-')
            ax.set_ylabel(f"unit {ni} (fit failed)")
    axes[-1].set_xlabel("angle (rad)")
    fig.suptitle("Example tuning curves")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

# -------------------------
# MAIN
# -------------------------
def main():
    # --- parameters ---
    mat_path = r"E:/Various Net/XJTUxch/data/loco_20170301_05.mat"
    outdir = "./analysis_out_v2"
    win_s = 0.064
    smooth_sigma_s = 0.08
    min_fr_keep = 0.5
    min_epoch_dur_s = 0.3
    tuning_bins = 12
    tuning_r2_thr = 0.15
    os.makedirs(outdir, exist_ok=True)

    # load
    print("[I] loading session ...")
    session = load_session(mat_path, prefer_finger=True, verbose=True)
    t = session['t']
    pos = session['pos']  # (T,2)
    print(f"[I] timestamps {t.shape}, pos {pos.shape}")

    # compute rates
    print("[I] computing binned rates ...")
    rates, centers = compute_binned_rates(session, win_s=win_s, smooth_sigma_s=smooth_sigma_s, verbose=True)
    n_units, n_bins = rates.shape
    print(f"[I] rates shape: units {n_units}, bins {n_bins}")

    # align ch_names safely to units
    ch_names_raw = session.get('ch_names', None)
    if ch_names_raw is None:
        ch_names = [f"unit{i}" for i in range(n_units)]
        print("[W] ch_names missing -> generating generic names")
    else:
        if len(ch_names_raw) >= n_units:
            ch_names = list(ch_names_raw[:n_units])
        else:
            # pad
            ch_names = list(ch_names_raw) + [f"unit{i}" for i in range(len(ch_names_raw), n_units)]
            print(f"[W] ch_names length {len(ch_names_raw)} < n_units {n_units} -> padded")

    # compute vel/acc and resample to bin centers
    dt = np.median(np.diff(t))
    # smoothing pos with savgol (ensure window valid)
    win_samples = min(len(pos)-1, int(round(0.15 / dt)))
    if win_samples % 2 == 0:
        win_samples = max(3, win_samples-1)
    if win_samples < 3:
        win_samples = 3
    try:
        pos_smooth = np.vstack([savgol_filter(pos[:,d], win_samples, 3, mode='interp') for d in range(2)]).T
    except Exception:
        pos_smooth = pos.copy()
    vel = np.gradient(pos_smooth, dt, axis=0)
    from scipy.interpolate import interp1d
    fvx = interp1d(t, vel[:,0], bounds_error=False, fill_value="extrapolate")
    fvy = interp1d(t, vel[:,1], bounds_error=False, fill_value="extrapolate")
    vel_binned = np.vstack([fvx(centers), fvy(centers)]).T
    speed = np.linalg.norm(vel_binned, axis=1)

    # drop low-fr units
    mean_fr = np.nanmean(rates, axis=1)
    keep_units = mean_fr >= min_fr_keep
    if keep_units.sum() == 0:
        print("[E] no units survive mean_fr threshold; reducing threshold to 0.1 Hz")
        keep_units = mean_fr >= 0.1
    rates = rates[keep_units, :]
    kept_names = [ch_names[i] for i in np.where(keep_units)[0]]
    print(f"[I] kept {rates.shape[0]} units after FR >= {min_fr_keep} Hz (names length {len(kept_names)})")

    # Q5: stopping neurons
    min_epoch_len_bins = max(1, int(round(min_epoch_dur_s / win_s)))
    df_stop, still_mask = detect_stopping_neurons(rates, speed, centers, speed_thr=None,
                                                  min_epoch_len_bins=min_epoch_len_bins)
    df_stop['unit_name'] = [kept_names[i] for i in df_stop['neuron'].values]
    df_stop.to_csv(os.path.join(outdir, "stopping_neurons_v2.csv"), index=False)
    print("[I] stopping detection done. stopping:", (df_stop['label']=='stopping').sum(),
          "suppressed:", (df_stop['label']=='suppressed').sum())

    # plot top suppressed/stopping examples
    top_stopping = df_stop[df_stop['label']=='stopping'].sort_values('cohen_d', ascending=False).head(6)
    top_suppressed = df_stop[df_stop['label']=='suppressed'].sort_values('cohen_d').head(6)
    # choose indices for plotting (convert to actual indices)
    plot_idxs = []
    for dfsub in (top_stopping, top_suppressed):
        for idx in dfsub['neuron'].values:
            plot_idxs.append(int(idx))
    if len(plot_idxs) > 0:
        plot_sample_psth(rates, centers, still_mask, plot_idxs, os.path.join(outdir, "example_psth_stopping_supp.png"))

    # Q6: tuning
    df_tune, centers_ang = compute_directional_tuning(rates, vel_binned, n_angle_bins=tuning_bins)
    # map region from kept_names (case-insensitive match)
    regions = []
    for name in kept_names:
        lname = name.lower()
        if 'm1' in lname:
            regions.append('M1')
        elif 's1' in lname:
            regions.append('S1')
        else:
            regions.append('Other')
    df_tune['unit_name'] = kept_names
    df_tune['region'] = regions
    df_tune.to_csv(os.path.join(outdir, "tuning_v2.csv"), index=False)

    # summary stats
    for reg in ['M1','S1','Other']:
        sub = df_tune[df_tune['region']==reg]
        if sub.shape[0] == 0:
            print(f"[I] region {reg}: 0 units")
            continue
        frac_tuned = np.nanmean(sub['r2'] > tuning_r2_thr)
        medianA = np.nanmedian(np.abs(sub['A']))
        print(f"[I] region {reg}: n={len(sub)}, tuned_frac(R2>{tuning_r2_thr})={frac_tuned:.3f}, median|A|={medianA:.3f}")

    # PD unit circle
    plot_pd_unit_circle(df_tune, regions, os.path.join(outdir, "pd_unit_circle_v2.png"))

    # sample tuning curves: highest r2 in M1 and S1
    sample_m1 = df_tune[df_tune['region']=='M1'].sort_values('r2', ascending=False).head(6)['neuron'].values.tolist()
    sample_s1 = df_tune[df_tune['region']=='S1'].sort_values('r2', ascending=False).head(6)['neuron'].values.tolist()
    sample_neurons = sample_m1 + sample_s1
    if len(sample_neurons) > 0:
        plot_sample_tuning_curves(rates, vel_binned, sample_neurons, centers_ang, os.path.join(outdir, "sample_tuning_v2.png"))

    # decode comparison
    mask_m1 = np.array([r=='M1' for r in regions])
    mask_s1 = np.array([r=='S1' for r in regions])
    if mask_m1.sum() == 0:
        print("[W] no M1 units detected for decode (mask_m1.sum()==0)")
    if mask_s1.sum() == 0:
        print("[W] no S1 units detected for decode (mask_s1.sum()==0)")
    r2_m1 = decode_velocity_ridge(rates[mask_m1,:], vel_binned) if mask_m1.sum()>0 else np.array([np.nan, np.nan])
    r2_s1 = decode_velocity_ridge(rates[mask_s1,:], vel_binned) if mask_s1.sum()>0 else np.array([np.nan, np.nan])
    r2_both = decode_velocity_ridge(rates, vel_binned)
    df_decode = pd.DataFrame({
        'region': ['M1','S1','M1+S1'],
        'r2_x': [r2_m1[0], r2_s1[0], r2_both[0]],
        'r2_y': [r2_m1[1], r2_s1[1], r2_both[1]]
    })
    df_decode.to_csv(os.path.join(outdir, "decode_v2.csv"), index=False)
    print("[I] decode results saved.")

    print("[I] All done. Outputs in", outdir)

if __name__ == "__main__":
    # suppress some expected runtime warnings for clarity
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

