"""
project_kalman.py

Supervised Kalman Filter decoding for M1 neural data.
- 数据预处理：提取神经元脉冲 -> 发放率 (Hz)，计算手指位置、速度、加速度
- 训练：监督估计卡尔曼参数 (A, C, Q, R)
- 测试：卡尔曼滤波解码
- 评价指标：R², SNR
- 绘图：2×3 网格 (pos/vel/acc × x/y)，和之前完全一致
"""

import math
import numpy as np
import h5py
from scipy import interpolate
from scipy.stats import norm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy.linalg as npl


# Utility

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


# Load session (dereference spikes), prefer finger_pos

def load_session(path, prefer_finger=True, verbose=True):
    if verbose:
        print(f"[load_session] opening {path}")
    with h5py.File(path, 'r') as f:
        # timestamps
        t_ds = find_dataset_group(f, ['t', 'time', 'timestamps'])
        if t_ds is None:
            raise RuntimeError("Cannot find timestamps dataset 't'.")
        t = np.array(t_ds).squeeze()

        # position
        pos_ds = None
        if prefer_finger:
            pos_ds = find_dataset_group(f, ['finger_pos', 'fingerpos', 'hand_pos'])
        if pos_ds is None:
            pos_ds = find_dataset_group(f, ['cursor_pos', 'cursorpos', 'cursor', 'pos', 'position'])
        if pos_ds is None:
            raise RuntimeError("Cannot find finger_pos/cursor_pos.")
        pos_arr = np.array(pos_ds)
        # normalize shape -> (T, >=2)
        if pos_arr.ndim == 2:
            if pos_arr.shape[1] == t.size:
                pos = pos_arr.T
            elif pos_arr.shape[0] == t.size:
                pos = pos_arr
            else:
                pos = pos_arr.T if pos_arr.T.shape[0] == t.size else pos_arr
        elif pos_arr.ndim == 1:
            if pos_arr.size % 2 == 0:
                pos = pos_arr.reshape(-1, 2)
            else:
                pos = pos_arr.reshape(-1, 3)
        else:
            pos = pos_arr.reshape(t.size, -1)
        pos = pos[:, :2]
        if verbose:
            print(f"  t shape: {t.shape}, pos shape: {pos.shape}")

        # spikes (references)
        spikes_ds = find_dataset_group(f, ['spike', 'spikes', 'spiketimes', 'times'])
        if spikes_ds is None:
            raise RuntimeError("Cannot find spikes dataset.")
        arr = np.array(spikes_ds)
        spike_list = []
        if arr.dtype == object or str(arr.dtype).startswith('ref') or arr.dtype.kind == 'O':
            # common format (5, N) referencing arrays per unit
            if arr.ndim == 2:
                n_units = arr.shape[1]
                for i in range(n_units):
                    ref = arr[0, i]
                    if ref is None:
                        spike_list.append(np.array([]))
                        continue
                    sp = np.array(f[ref]).squeeze()
                    spike_list.append(np.asarray(sp).ravel())
            elif arr.ndim == 1:
                for i in range(arr.shape[0]):
                    ref = arr[i]
                    try:
                        sp = np.array(f[ref]).squeeze()
                        spike_list.append(np.asarray(sp).ravel())
                    except Exception:
                        spike_list.append(np.array([]))
            else:
                raise RuntimeError("Unexpected spikes reference shape.")
        else:
            # numeric matrix case: try to turn into spike lists (rare for your files)
            mat = np.array(spikes_ds)
            if mat.ndim == 2 and mat.shape[0] == t.size:
                for u in range(mat.shape[1]):
                    counts = mat[:, u]
                    times = []
                    for idx, c in enumerate(counts):
                        if c > 0:
                            times.extend([t[idx]] * int(c))
                    spike_list.append(np.array(times))
            elif mat.ndim == 2 and mat.shape[1] == t.size:
                mat = mat.T
                for u in range(mat.shape[1]):
                    counts = mat[:, u]
                    times = []
                    for idx, c in enumerate(counts):
                        if c > 0:
                            times.extend([t[idx]] * int(c))
                    spike_list.append(np.array(times))
            else:
                raise RuntimeError("Uninterpretable numeric spikes matrix.")
        if verbose:
            print(f"  loaded spikes for {len(spike_list)} units")
    return {'t': np.asarray(t).ravel(), 'pos': np.asarray(pos), 'spike_list': spike_list}


# Binning & smoothing

def compute_binned_rates(session, win_s=0.064, smooth_sigma_s=None, verbose=True):
    t = session['t']
    spike_list = session['spike_list']
    t_start, t_end = float(np.min(t)), float(np.max(t))
    bins = np.arange(t_start, t_end + 1e-9, win_s)
    bin_centers = bins[:-1] + win_s / 2.0
    n_bins = len(bin_centers)
    n_units = len(spike_list)
    rates = np.zeros((n_units, n_bins), dtype=float)
    for i, st in enumerate(spike_list):
        if st is None or len(st) == 0:
            continue
        counts, _ = np.histogram(st, bins=bins)
        rates[i, :] = counts / win_s
    if smooth_sigma_s is not None and smooth_sigma_s > 0:
        dt = win_s
        kern_half = max(3 * smooth_sigma_s, win_s * 3)
        x = np.arange(-kern_half, kern_half + 1e-12, dt)
        kernel = norm.pdf(x, scale=smooth_sigma_s)
        kernel /= (kernel.sum() + 1e-16)
        for i in range(n_units):
            rates[i, :] = fftconvolve(rates[i, :], kernel, mode='same')
    if verbose:
        print(f"[binning] rates shape: {rates.shape}, bins: {n_bins}, win_s={win_s}")
    return rates, bin_centers


# compute kinematics

def compute_kinematics(t_samples, pos_samples, bin_centers):
    fx = interpolate.interp1d(t_samples, pos_samples[:, 0], bounds_error=False, fill_value="extrapolate")
    fy = interpolate.interp1d(t_samples, pos_samples[:, 1], bounds_error=False, fill_value="extrapolate")
    pos = np.stack([fx(bin_centers), fy(bin_centers)], axis=1)
    dt = np.gradient(bin_centers)
    vel = np.gradient(pos, axis=0) / dt.reshape(-1, 1)
    acc = np.gradient(vel, axis=0) / dt.reshape(-1, 1)
    return pos, vel, acc


# preprocess & save

def process_and_save_session(path, out_path, win_ms=64, min_rate_hz=0.5, smooth_sigma_s=0.02):
    sess = load_session(path, prefer_finger=True, verbose=True)
    rates, bin_centers = compute_binned_rates(sess, win_s=win_ms/1000.0, smooth_sigma_s=smooth_sigma_s, verbose=True)
    mean_rates = rates.mean(axis=1)
    keep_idx = np.where(mean_rates >= min_rate_hz)[0]
    kept_rates = rates[keep_idx, :]
    pos_b, vel_b, acc_b = compute_kinematics(sess['t'], sess['pos'], bin_centers)
    np.savez_compressed(out_path,
                        X=kept_rates.T.astype(np.float32),
                        Y_pos=pos_b.astype(np.float32),
                        Y_vel=vel_b.astype(np.float32),
                        Y_acc=acc_b.astype(np.float32),
                        bin_centers=bin_centers.astype(np.float32),
                        keep_idx=keep_idx.astype(np.int32),
                        win_s=np.float32(win_ms/1000.0))
    print(f"[process] saved {out_path}, kept units={len(keep_idx)}")
    return out_path


# Supervised Kalman (improved with intercept)

def fit_supervised_kalman_with_bias(npzfile, train_frac=0.7, val_frac=0.1, reg=1e-3, verbose=True):
    """
    Fit A (state dynamics), and observation C0 and bias d via ridge regression on training data.
    Return A, C0, d, Q, R, x0, P0 and indices.
    """
    data = np.load(npzfile, allow_pickle=True)
    X = data['X']            # (T, m)
    Y_pos = data['Y_pos']    # (T,2)
    Y_vel = data['Y_vel']    # (T,2)
    Y_acc = data['Y_acc']    # (T,2)
    bin_centers = data['bin_centers']
    win_s = float(data['win_s']) if 'win_s' in data else 0.064

    Z = np.concatenate([Y_pos, Y_vel, Y_acc], axis=1)  # (T, 6)
    Y = X                                             # (T, m)

    T = Z.shape[0]
    i_tr = int(T * train_frac)
    i_val = int(T * (train_frac + val_frac))
    i_test = i_val

    # pairs for training A: Z_prev -> Z_next
    Z_prev = Z[:i_tr-1, :]   # (Ttr-1, n)
    Z_next = Z[1:i_tr, :]    # (Ttr-1, n)
    Y_prev = Y[:i_tr-1, :]   # (Ttr-1, m)

    # Estimate A with ridge: A = (Z_next^T Z_prev) (Z_prev^T Z_prev + reg I)^{-1}
    ZpT_Zp = (Z_prev.T @ Z_prev) + reg * np.eye(Z_prev.shape[1])
    A = (Z_next.T @ Z_prev) @ npl.inv(ZpT_Zp)  # (n,n)

    # Q from residuals
    W = Z_next - (Z_prev @ A.T)
    Q = np.cov(W.T, bias=False) + reg * np.eye(W.shape[1])

    # Estimate C0 and bias d by augmenting Z_prev with constant 1 column:
    ones = np.ones((Z_prev.shape[0], 1))
    Zp_aug = np.concatenate([Z_prev, ones], axis=1)  # (Ttr-1, n+1)
    # closed-form ridge: C_full = (Y_prev^T Zp_aug) (Zp_aug^T Zp_aug + reg I)^{-1}
    Zp_aug_T_Zp_aug = (Zp_aug.T @ Zp_aug) + reg * np.eye(Zp_aug.shape[1])
    C_full = (Y_prev.T @ Zp_aug) @ npl.inv(Zp_aug_T_Zp_aug)  # (m, n+1)
    C0 = C_full[:, :Z_prev.shape[1]]                         # (m, n)
    d = C_full[:, -1]                                        # (m,)

    # R from residuals including bias
    V = Y_prev - (Z_prev @ C0.T + d.reshape(1, -1))  # broadcast d
    R = np.cov(V.T, bias=False) + reg * np.eye(V.shape[1])

    # initial state x0: choose last training state (better than mean)
    x0 = Z[i_tr - 1].copy()
    P0 = np.cov(Z[:i_tr].T, bias=False) + reg * np.eye(Z.shape[1])

    if verbose:
        print("[KF fit] shapes: Z_prev", Z_prev.shape, "Y_prev", Y_prev.shape)
        print("[KF fit] A", A.shape, "C0", C0.shape, "d", d.shape, "Q", Q.shape, "R", R.shape)
        print("[KF fit] x0 idx", i_tr - 1, "P0 shape", P0.shape)

    return {'A': A, 'C0': C0, 'd': d, 'Q': Q, 'R': R,
            'x0': x0, 'P0': P0,
            'i_tr': i_tr, 'i_val': i_val, 'i_test': i_test,
            'Z': Z, 'Y': Y, 'bin_centers': bin_centers, 'win_s': win_s}

def run_kalman_filter_with_bias(A, C0, d, Q, R, x0, P0, Y_obs, reg=1e-8, return_full=False):
    """
    Kalman filter when observation model is y = C0 z + d + v.
    Y_obs: (T_obs, m)
    If return_full True -> return (Xf, x_last, P_last); else return (x_last, P_last)
    """
    T_obs = Y_obs.shape[0]
    n = x0.shape[0]
    x = x0.copy()
    P = P0.copy()
    if return_full:
        Xf = np.zeros((T_obs, n))
    for t in range(T_obs):
        # Predict
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        # Innovation cov
        S = C0 @ P_pred @ C0.T + R + reg * np.eye(R.shape[0])
        # Gain
        try:
            S_inv = npl.inv(S)
        except np.linalg.LinAlgError:
            S_inv = npl.pinv(S)
        K = P_pred @ C0.T @ S_inv
        # Update with bias d
        y = Y_obs[t, :]
        innov = y - (C0 @ x_pred + d)
        x_upd = x_pred + K @ innov
        P_upd = (np.eye(n) - K @ C0) @ P_pred
        if return_full:
            Xf[t, :] = x_upd
        x = x_upd
        P = P_upd
    if return_full:
        return Xf, x, P
    else:
        return x, P

def train_and_eval_kalman_improved(npzfile, train_frac=0.7, val_frac=0.1, reg=1e-3, verbose=True):
    """
    Fits supervised Kalman (with bias), warms-up using validation obs, then evaluates on test set.
    Returns dict with preds/trues/r2s/snrs/bin_centers_test etc.
    """
    fit = fit_supervised_kalman_with_bias(npzfile, train_frac=train_frac, val_frac=val_frac, reg=reg, verbose=verbose)
    A, C0, d, Q, R = fit['A'], fit['C0'], fit['d'], fit['Q'], fit['R']
    x0, P0 = fit['x0'].copy(), fit['P0'].copy()
    i_tr, i_val, i_test = fit['i_tr'], fit['i_val'], fit['i_test']
    bin_centers = fit['bin_centers']
    win_s = fit['win_s']

    data = np.load(npzfile, allow_pickle=True)
    X_all = data['X']
    Z_all = np.concatenate([data['Y_pos'], data['Y_vel'], data['Y_acc']], axis=1)

    # Warm-up on validation observations (if exist)
    if i_val > i_tr:
        Y_warm = X_all[i_tr:i_val, :]
        if Y_warm.shape[0] > 0:
            # Run filter on warm-up and get final x,P
            _, x_warm, P_warm = run_kalman_filter_with_bias(A, C0, d, Q, R, x0, P0, Y_warm, reg=1e-8, return_full=True)
            x_init, P_init = x_warm, P_warm
            if verbose:
                print(f"[KF warmup] used validation obs to advance state from idx {i_tr-1} to {i_val-1}")
        else:
            x_init, P_init = x0, P0
    else:
        x_init, P_init = x0, P0

    # Test filtering
    Y_test = X_all[i_test:, :]
    Z_test = Z_all[i_test:, :]
    if Y_test.shape[0] == 0:
        raise RuntimeError("No test data (check splits).")

    Xf_test, x_last, P_last = run_kalman_filter_with_bias(A, C0, d, Q, R, x_init, P_init, Y_test, reg=1e-8, return_full=True)

    # metrics
    r2s = [r2_score(Z_test[:, d], Xf_test[:, d]) for d in range(Z_test.shape[1])]
    snrs = []
    for r in r2s:
        r_clip = min(max(r, -1 + 1e-9), 1 - 1e-12)
        snr = -10.0 * math.log10(1.0 - r_clip)
        snrs.append(snr)
    if verbose:
        print("[KF improved eval] Test R2 per dim:", r2s)
        print("[KF improved eval] Test SNR(dB) per dim:", snrs)
        # diagnostic: mean comparison for position x
        mean_true_px = float(np.mean(Z_test[:, 0])); mean_pred_px = float(np.mean(Xf_test[:, 0]))
        print(f"[diag] mean pos_x true = {mean_true_px:.4f}, pred = {mean_pred_px:.4f}, ratio pred/true = {mean_pred_px/ (mean_true_px+1e-9):.4f}")

    return {
        'model': {'A': A, 'C0': C0, 'd': d, 'Q': Q, 'R': R},
        'preds': Xf_test,
        'trues': Z_test,
        'r2s': r2s,
        'snrs': snrs,
        'bin_centers_test': bin_centers[i_test:],
        'win_s': win_s,
        'i_test_start': i_test
    }


# plotting same 2x3 layout

def plot_pred_vs_gt_grid(result_dict, time_window_sec=6.0, savepath=None):
    preds = result_dict['preds']
    trues = result_dict['trues']
    times = result_dict['bin_centers_test']
    r2s = result_dict['r2s']
    snrs = result_dict['snrs']

    pred_pos = preds[:, 0:2]; pred_vel = preds[:, 2:4]; pred_acc = preds[:, 4:6]
    true_pos = trues[:, 0:2]; true_vel = trues[:, 2:4]; true_acc = trues[:, 4:6]

    center_time = times[len(times)//2]
    half = time_window_sec / 2.0
    sel = np.where((times >= center_time - half) & (times <= center_time + half))[0]
    if sel.size == 0:
        sel = np.arange(min(1000, len(times)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    targets = [('Position', true_pos, pred_pos, 0), ('Velocity', true_vel, pred_vel, 2), ('Acceleration', true_acc, pred_acc, 4)]
    for col, (title, true_xy, pred_xy, base_idx) in enumerate(targets):
        for row in range(2):
            ax = axes[row, col]
            comp = row
            ax.plot(times[sel], true_xy[sel, comp], 'k-', label='gt', linewidth=1.5)
            ax.plot(times[sel], pred_xy[sel, comp], 'r--', label='pred', linewidth=1.2)
            if row == 0:
                ax.set_title(title, fontsize=14)
            if col == 0:
                ax.set_ylabel('x' if row == 0 else 'y', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            r2_val = r2s[base_idx + comp]
            snr_val = snrs[base_idx + comp]
            ax.text(0.02, 0.88, f"R²={r2_val:.3f}\nSNR={snr_val:.1f} dB", transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")
    plt.show()


# main

if __name__ == "__main__":
    session_file = r"E:/Various Net/XJTUxch/data/loco_20170215_02.mat"
    out_npz = "卡尔曼模型.npz"

    # 1) preprocess
    process_and_save_session(session_file, out_npz, win_ms=64, min_rate_hz=0.5, smooth_sigma_s=0.02)

    # 2) train & evaluate KF (improved)
    res = train_and_eval_kalman_improved(out_npz, train_frac=0.7, val_frac=0.1, reg=1e-3, verbose=True)

    # 3) plot results (same layout)
    plot_pred_vs_gt_grid(res, time_window_sec=6.0, savepath="卡尔曼模型.png")