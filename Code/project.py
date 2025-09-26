"""
功能:
1. 数据预处理 (spike binning, 平滑, kinematics)
2. 线性回归解码手部速度, 计算 R² 和 SNR, 并绘制 Pred vs GT 曲线
3. 余弦调谐分析:
   - 10 个典型神经元的 firing rate 直方图 + 拟合曲线
   - Preferred direction 分布散点图 (cosφ, sinφ)

输入:  indy_20160921_01.mat
输出:  processed_final.npz
"""
import math
import numpy as np
import h5py
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# -------------------------
# Utility: find dataset by keyword (recursive)
# -------------------------
def find_dataset_group(root, keywords):
    """
    Recursively search HDF5 group/dataset keys for any keyword substring (case-insensitive).
    Returns the first matching dataset/group object.
    """
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

# -------------------------
# Robust loader for your .mat (NWB/HDF5) structure
# -------------------------
def load_session(path, prefer_finger=True, verbose=True):
    """
    Load t (timestamps), pos (T,2) (prefer finger_pos), and spikes as list-of-arrays of spike times (seconds).
    Supports cases where spikes dataset contains HDF5 references (common in your .mat).
    Returns dict: {'t': t, 'pos': pos, 'spike_list': spike_list, 'ch_names': ch_names_or_None}
    """
    if verbose:
        print(f"[load_session] opening {path}")
    with h5py.File(path, 'r') as f:
        # timestamps
        t_ds = find_dataset_group(f, ['t', 'time', 'timestamps'])
        if t_ds is None:
            raise RuntimeError("Cannot find timestamps dataset (t).")
        t = np.array(t_ds).squeeze()
        if verbose:
            print(f"  t shape: {t.shape}")

        # position: prefer finger_pos then cursor_pos
        pos_ds = None
        if prefer_finger:
            pos_ds = find_dataset_group(f, ['finger_pos', 'fingerpos', 'hand_pos'])
        if pos_ds is None:
            pos_ds = find_dataset_group(f, ['cursor_pos', 'cursorpos', 'cursor', 'pos', 'position'])
        if pos_ds is None:
            raise RuntimeError("Cannot find finger_pos or cursor_pos dataset.")
        pos_arr = np.array(pos_ds)
        # Normalize pos to (T, >=2)
        if pos_arr.ndim == 2:
            # If (2, T) or (3, T) -> transpose to (T,2/3)
            if pos_arr.shape[1] == t.size:
                pos = pos_arr.T
            elif pos_arr.shape[0] == t.size:
                pos = pos_arr
            else:
                # fallback: try transpose then reshape
                pos = pos_arr.T if pos_arr.T.shape[0] == t.size else pos_arr
        elif pos_arr.ndim == 1:
            # reshape heuristics
            if pos_arr.size % 2 == 0:
                pos = pos_arr.reshape(-1, 2)
            else:
                pos = pos_arr.reshape(-1, 3)
        else:
            pos = pos_arr.reshape(t.size, -1)
        # only take x,y
        if pos.shape[1] >= 2:
            pos = pos[:, :2]
        if verbose:
            print(f"  pos shape (T,2): {pos.shape}")

        # channel names if present
        ch_names_ds = find_dataset_group(f, ['chan_names', 'ch_names', 'channel', 'labels'])
        ch_names = None
        if ch_names_ds is not None:
            try:
                arr = np.array(ch_names_ds)
                # flatten and decode bytes if necessary
                ch_list = []
                for s in arr.ravel():
                    if isinstance(s, bytes) or isinstance(s, np.bytes_):
                        ch_list.append(s.decode('utf-8', errors='ignore'))
                    else:
                        ch_list.append(str(s))
                ch_names = ch_list
            except Exception:
                ch_names = None
        if verbose:
            print(f"  chan names found: {None if ch_names is None else len(ch_names)}")

        # spikes dataset: could be references or numeric
        spikes_ds = find_dataset_group(f, ['spike', 'spikes', 'unit', 'spiketimes', 'spike_times', 'times'])
        spike_list = None
        spike_matrix = None
        if spikes_ds is None:
            if verbose:
                print("  Warning: no spikes dataset found.")
        else:
            ds = spikes_ds
            # If ds dtype is reference/object -> need dereference
            try:
                dt = ds.dtype
                # HDF5 reference array (object/refs) -> np.array(ds) contains references
                arr = np.array(ds)
                if arr.dtype == object or str(arr.dtype).startswith('ref') or arr.dtype.kind == 'O':
                    # many files store shape (5, Nunits) where first dim indexes different arrays
                    # We assume second dim enumerates units: take arr[0, i] references
                    if arr.ndim == 2:
                        n_units = arr.shape[1]
                        spike_list = []
                        for i in range(n_units):
                            ref = arr[0, i]
                            if ref is None:
                                spike_list.append(np.array([]))
                                continue
                            try:
                                sp = np.array(f[ref]).squeeze()
                                spike_list.append(np.asarray(sp).ravel())
                            except Exception:
                                spike_list.append(np.array([]))
                    elif arr.ndim == 1:
                        spike_list = []
                        for i in range(arr.shape[0]):
                            ref = arr[i]
                            try:
                                sp = np.array(f[ref]).squeeze()
                                spike_list.append(np.asarray(sp).ravel())
                            except Exception:
                                spike_list.append(np.array([]))
                    else:
                        spike_list = None
                else:
                    # numeric dataset -> treat as matrix
                    mat = arr
                    # guess orientation relative to t
                    if mat.ndim == 2 and mat.shape[0] == t.size:
                        spike_matrix = mat  # time x units
                    elif mat.ndim == 2 and mat.shape[1] == t.size:
                        spike_matrix = mat.T
                    else:
                        # fallback: store matrix and let compute_binned_rates handle heuristics
                        spike_matrix = mat
            except Exception as e:
                if verbose:
                    print("  Exception handling spikes dataset:", e)
                spike_list = None

    return {'t': np.asarray(t).ravel(), 'pos': np.asarray(pos), 'spike_list': spike_list,
            'spike_matrix': spike_matrix, 'ch_names': ch_names}


# -------------------------
# Binning & smoothing
# -------------------------
def compute_binned_rates(session, win_s=0.064, smooth_sigma_s=None, verbose=True):
    """
    session: dict returned by load_session
    Supports either 'spike_list' (list of spike times arrays) OR 'spike_matrix' (time x units)
    Returns: rates (units, n_bins) in Hz, bin_centers (seconds)
    """
    t = session['t']
    spike_list = session.get('spike_list', None)
    spike_matrix = session.get('spike_matrix', None)

    t_start = float(np.nanmin(t))
    t_end = float(np.nanmax(t))
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
        # try to map rows to time samples
        if mat.shape[0] == t.size:
            # mat shape: time x units
            n_units = mat.shape[1]
            rates = np.zeros((n_units, n_bins), dtype=float)
            inds = np.digitize(t, bins) - 1
            for b in range(n_bins):
                sel = (inds == b)
                if not np.any(sel):
                    continue
                counts = mat[sel, :].sum(axis=0)
                rates[:, b] = counts / win_s
            source = 'matrix_time_rows'
        elif mat.shape[1] == t.size:
            # units x time -> transpose
            session_local = dict(session)
            session_local['spike_matrix'] = mat.T
            return compute_binned_rates(session_local, win_s, smooth_sigma_s, verbose)
        else:
            # fallback heuristics
            # treat columns as units if columns < rows
            if mat.shape[1] < mat.shape[0]:
                sample_times = np.linspace(t_start, t_end, mat.shape[0])
                inds = np.digitize(sample_times, bins) - 1
                n_units = mat.shape[1]
                rates = np.zeros((n_units, n_bins), dtype=float)
                for b in range(n_bins):
                    sel = (inds == b)
                    if not np.any(sel): continue
                    counts = mat[sel, :].sum(axis=0)
                    rates[:, b] = counts / win_s
                source = 'matrix_heuristic'
            else:
                raise RuntimeError("Cannot interpret spike_matrix shape relative to t.")
    else:
        raise RuntimeError("No spike data found in session.")

    # optional Gaussian smoothing (in seconds)
    if smooth_sigma_s is not None and smooth_sigma_s > 0:
        dt = win_s
        kern_half = max(3 * smooth_sigma_s, win_s * 3)
        x = np.arange(-kern_half, kern_half + 1e-12, dt)
        kernel = norm.pdf(x, scale=smooth_sigma_s)
        kernel /= (kernel.sum() + 1e-16)
        for i in range(rates.shape[0]):
            rates[i, :] = fftconvolve(rates[i, :], kernel, mode='same')

    if verbose:
        print(f"  Binned rates shape: {rates.shape}, source={source}, win_s={win_s}s")
    return rates, bin_centers


# -------------------------
# Kinematics: pos, vel, acc aligned to bin centers
# -------------------------
def compute_kinematics(t_samples, pos_samples, bin_centers):
    """
    Interpolate position to bin_centers and compute velocity/acceleration.
    pos_samples expected shape: (T, 2) for x,y
    """
    fx = interpolate.interp1d(t_samples, pos_samples[:, 0], bounds_error=False, fill_value="extrapolate")
    fy = interpolate.interp1d(t_samples, pos_samples[:, 1], bounds_error=False, fill_value="extrapolate")
    pos = np.stack([fx(bin_centers), fy(bin_centers)], axis=1)
    # gradients w.r.t time
    dt = np.gradient(bin_centers)
    vel = np.gradient(pos, axis=0) / dt.reshape(-1, 1)
    acc = np.gradient(vel, axis=0) / dt.reshape(-1, 1)
    return pos, vel, acc


# -------------------------
# process & save preprocessed data (.npz)
# -------------------------
def process_and_save_session(path, out_path, win_ms=64, min_rate_hz=0.5, smooth_sigma_s=0.02, prefer_finger=True, verbose=True):
    """
    Full preprocessing pipeline:
     - load session (spike times, pos, t)
     - compute binned firing rates (Hz)
     - threshold low-rate units
     - compute pos/vel/acc aligned to bins
     - save .npz with metadata
    """
    sess = load_session(path, prefer_finger=prefer_finger, verbose=verbose)
    rates, bin_centers = compute_binned_rates(sess, win_s=win_ms/1000.0, smooth_sigma_s=smooth_sigma_s, verbose=verbose)
    mean_rates = rates.mean(axis=1)
    keep_idx = np.where(mean_rates >= min_rate_hz)[0]
    kept_rates = rates[keep_idx, :]  # (n_kept_units, n_bins)
    pos_b, vel_b, acc_b = compute_kinematics(sess['t'], sess['pos'], bin_centers)
    # Save: X = (n_bins, n_units_kept) as float32 (rates in Hz)
    np.savez_compressed(out_path,
                        X=kept_rates.T.astype(np.float32),
                        Y_pos=pos_b.astype(np.float32),
                        Y_vel=vel_b.astype(np.float32),
                        Y_acc=acc_b.astype(np.float32),
                        bin_centers=bin_centers.astype(np.float32),
                        keep_idx=keep_idx.astype(np.int32),
                        win_s=np.float32(win_ms/1000.0))
    if verbose:
        print(f"[process] saved to {out_path}. units kept={len(keep_idx)}; win_s={win_ms/1000.0}s")
    return out_path


# -------------------------
# PyTorch dataset & model (linear multi-output)
# -------------------------
class NeuralDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class LinearDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.lin(x)


# -------------------------
# Train multi-output linear decoder and return test preds/trues
# -------------------------
def train_multi_decoder(npzfile, train_frac=0.7, val_frac=0.1, epochs=40, batch_size=256, lr=1e-3, device='cpu', verbose=True):
    """
    Train one linear model to predict [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y] (6 outputs).
    Returns: dict with preds (test), trues (test), bin_centers_test, r2s (6 values), snrs (6 values), model.
    """
    data = np.load(npzfile, allow_pickle=True)
    X = data['X']        # (T, N)
    Y_pos = data['Y_pos']
    Y_vel = data['Y_vel']
    Y_acc = data['Y_acc']
    bin_centers = data['bin_centers']
    win_s = float(data['win_s']) if 'win_s' in data else 0.064

    # stack outputs into shape (T, 6)
    Y_all = np.concatenate([Y_pos, Y_vel, Y_acc], axis=1)

    T = X.shape[0]
    i_tr = int(T * train_frac)
    i_val = int(T * (train_frac + val_frac))
    i_test = i_val

    # scalers fit on train
    xsc = StandardScaler().fit(X[:i_tr])
    ysc = StandardScaler().fit(Y_all[:i_tr])
    Xn = xsc.transform(X)
    Yn = ysc.transform(Y_all)

    train_ds = NeuralDataset(Xn[:i_tr], Yn[:i_tr])
    val_ds = NeuralDataset(Xn[i_tr:i_val], Yn[i_tr:i_val])
    test_ds = NeuralDataset(Xn[i_val:], Yn[i_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device(device)
    model = LinearDecoder(X.shape[1], Y_all.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossfn = nn.MSELoss()

    best_val = 1e12
    best_state = None
    for ep in range(epochs):
        model.train()
        total_loss = 0.0; nn_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = lossfn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
            nn_count += xb.shape[0]
        # validation
        model.eval()
        vtot = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss = lossfn(model(xb), yb)
                vtot += vloss.item() * xb.shape[0]
                vn += xb.shape[0]
        vloss_avg = vtot / (vn + 1e-16)
        if vloss_avg < best_val:
            best_val = vloss_avg
            best_state = model.state_dict()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"[train] ep {ep+1}/{epochs} val_loss={vloss_avg:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # evaluate on test set
    preds_list = []; trues_list = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device)).cpu().numpy()
            preds_list.append(out)
            trues_list.append(yb.numpy())
    preds = np.vstack(preds_list)
    trues = np.vstack(trues_list)
    preds_orig = ysc.inverse_transform(preds)
    trues_orig = ysc.inverse_transform(trues)

    # compute R2 per output dimension
    r2s = []
    for d in range(trues_orig.shape[1]):
        r2_d = r2_score(trues_orig[:, d], preds_orig[:, d])
        r2s.append(r2_d)
    # compute SNR (dB) using R^2
    snrs = []
    for r in r2s:
        r_clip = min(max(r, -1 + 1e-9), 1 - 1e-12)
        snr = -10.0 * math.log10(1.0 - r_clip)
        snrs.append(snr)

    # test times
    bin_centers_test = bin_centers[i_test:]

    return {
        'model': model,
        'preds': preds_orig,
        'trues': trues_orig,
        'r2s': r2s,
        'snrs': snrs,
        'bin_centers_test': bin_centers_test,
        'win_s': win_s,
        'i_test_start': i_test
    }


# -------------------------
# Plot Pred vs GT in a 2x3 grid
# -------------------------
def plot_pred_vs_gt_grid(result_dict, time_window_sec=6.0, savepath=None):
    """
    result_dict: returned from train_multi_decoder
    Plot arrangement:
        columns: Position | Velocity | Acceleration
        rows: x (top), y (bottom)

    time_window_sec: width of the plotting window (seconds) centered in test set
    """
    preds = result_dict['preds']   # (T_test, 6)
    trues = result_dict['trues']
    times = result_dict['bin_centers_test']
    r2s = result_dict['r2s']
    snrs = result_dict['snrs']

    # split preds/trues into pos/vel/acc each (2 components)
    pred_pos = preds[:, 0:2]; pred_vel = preds[:, 2:4]; pred_acc = preds[:, 4:6]
    true_pos = trues[:, 0:2]; true_vel = trues[:, 2:4]; true_acc = trues[:, 4:6]

    # choose a time window centered in the middle of test set
    center_time = times[len(times) // 2]
    half = time_window_sec / 2.0
    sel = np.where((times >= center_time - half) & (times <= center_time + half))[0]
    if sel.size == 0:
        # fallback to first plot_len samples
        plot_idx = np.arange(min(1000, len(times)))
    else:
        plot_idx = sel

    # create grid 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    targets = [('Position', true_pos, pred_pos, 0), ('Velocity', true_vel, pred_vel, 2), ('Acceleration', true_acc, pred_acc, 4)]
    for col, (title, true_xy, pred_xy, base_idx) in enumerate(targets):
        # top row -> x (component 0), bottom row -> y (component 1)
        for row in range(2):
            ax = axes[row, col]
            comp = row  # 0 for x, 1 for y
            ax.plot(times[plot_idx], true_xy[plot_idx, comp], label='gt', color='k', linewidth=1.5)
            ax.plot(times[plot_idx], pred_xy[plot_idx, comp], label='pred', color='tab:orange', linestyle='--', linewidth=1.2)
            # axis labels & title
            if row == 0:
                ax.set_title(title, fontsize=14)
            if col == 0:
                ax.set_ylabel('x' if row == 0 else 'y', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            # annotate R2 and SNR for that component
            r2_val = r2s[base_idx + comp]
            snr_val = snrs[base_idx + comp]
            ax.text(0.02, 0.88, f"R²={r2_val:.3f}\nSNR={snr_val:.1f} dB", transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")
    plt.show()


# -------------------------
# Cosine tuning analysis with nice plots
# -------------------------
def cosine_tuning(theta, b, a, phi):
    return b + a * np.cos(theta - phi)

def analyze_cosine_tuning(npzfile, n_show=10, top_k=100, savepath=None):
    """
    - compute cosine fits for each unit (using velocity direction theta)
    - plot n_show histograms (x 10, arranged 2x5) with fitted curve, using degrees on x-axis
    - plot distribution (cosφ, sinφ) scatter for top_k units (blue dots on unit circle)
    """
    data = np.load(npzfile, allow_pickle=True)
    X = data['X']            # (T, N)
    Y_vel = data['Y_vel']    # (T,2)
    win_s = float(data['win_s']) if 'win_s' in data else 0.064

    # --- velocity direction in [0, 2π) ---
    theta = (np.arctan2(Y_vel[:, 1], Y_vel[:, 0]) + 2*np.pi) % (2*np.pi)

    results = []
    for i in range(X.shape[1]):
        r = X[:, i]  # firing rate in Hz
        if np.all(np.isnan(r)) or np.std(r) == 0:
            results.append((i, np.nan, np.nan, np.nan, 0.0))
            continue
        try:
            popt, _ = curve_fit(cosine_tuning, theta, r,
                               p0=[np.mean(r), np.std(r), 0], maxfev=5000)
            b, a, phi = popt
            r_pred = cosine_tuning(theta, b, a, phi)
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum((r - np.mean(r)) ** 2) + 1e-16
            R2 = 1 - ss_res / ss_tot
        except Exception:
            b, a, phi, R2 = np.nan, np.nan, np.nan, 0.0
        results.append((i, b, a, phi, R2))

    # sort by R2 descending
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
    n_show = min(n_show, len(results_sorted))
    top_k = min(top_k, len(results_sorted))

    # ---------- Plot 10 histograms (2 x 5) ----------
    theta_grid = np.linspace(0, 2*np.pi, 400)
    theta_deg_grid = np.degrees(theta_grid)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    bins_deg = np.linspace(0.0, 360.0, 36+1)  # 10-degree bins
    theta_deg_all = np.degrees(theta)  # already in [0,360)

    for j in range(n_show):
        idx, b, a, phi, R2 = results_sorted[j]
        r = X[:, idx]  # Hz
        spikes_per_window = r * win_s
        inds = np.digitize(theta_deg_all, bins_deg) - 1
        bin_means = [np.mean(spikes_per_window[inds == kk]) if np.any(inds == kk) else np.nan
                     for kk in range(len(bins_deg)-1)]
        bin_centers_deg = (bins_deg[:-1] + bins_deg[1:]) / 2.0

        ax = axes[j]
        ax.bar(bin_centers_deg, bin_means,
               width=(bins_deg[1]-bins_deg[0]) * 0.9,
               alpha=0.5, edgecolor='none')

        if not np.isnan(b):
            fit_rate = cosine_tuning(theta_grid, b, a, phi)  # Hz
            fit_spikes_window = fit_rate * win_s
            ax.plot(theta_deg_grid, fit_spikes_window, 'r-', lw=2)
        if not np.isnan(phi):
            phi_deg = np.degrees(phi) % 360
            ax.axvline(phi_deg, color='k', linestyle='--', linewidth=1)

        ax.set_xlim(0, 360)
        ax.set_xlabel("Angle °")
        ax.set_ylabel("Spikes / time window")
        ax.set_title(f"θ = {np.degrees(phi)%360:.2f}°   (Unit {idx})\nR²={R2:.3f}")
        ax.set_xticks(np.arange(0, 361, 60))

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath.replace('.png', '余弦拟合.png'), dpi=200)
    plt.show()

    # ---------- Plot preferred direction distribution ----------
    phis = [entry[3] for entry in results_sorted[:top_k] if not np.isnan(entry[3])]
    if len(phis) == 0:
        print("No valid phi found for preferred direction distribution.")
        return results_sorted
    xs = np.cos(phis)
    ys = np.sin(phis)

    fig, ax = plt.subplots(figsize=(7, 7))
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)
    ax.scatter(xs, ys, s=40, c='tab:blue', label='Preferred direction', zorder=3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('cosθ', fontsize=12)
    ax.set_ylabel('sinθ', fontsize=12)
    ax.set_title('Distribution of preferred directions', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
    if savepath:
        fig.savefig(savepath.replace('.png', '散点图.png'), dpi=200)
    plt.show()

    return results_sorted


# -------------------------
# Main entry: run the full pipeline and produce figures
# -------------------------
if __name__ == "__main__":
    # === User-specific path ===
    session_file = r"E:/Various Net/XJTUxch/data/loco_20170301_05.mat"
    out_npz = "线性模型.npz"

    # === 1) Preprocess & save ===
    process_and_save_session(session_file, out_npz, win_ms=64, min_rate_hz=0.5, smooth_sigma_s=0.02, prefer_finger=True, verbose=True)

    # === 2) Train multi-output linear decoder ===
    res = train_multi_decoder(out_npz, train_frac=0.7, val_frac=0.1, epochs=30, batch_size=256, lr=1e-3, device='cpu', verbose=True)

    # === 3) Plot predictions vs GT in the requested 2x3 layout ===
    plot_pred_vs_gt_grid(res, time_window_sec=6.0, savepath="线性模型.png")

    # === 4) Cosine tuning analysis: 10 histograms + distribution plot ===
    analyze_cosine_tuning(out_npz, n_show=10, top_k=100, savepath=".png")
