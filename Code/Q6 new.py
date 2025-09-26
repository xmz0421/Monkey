# -*- coding: utf-8 -*-
"""
q6_mat_main.py — 一键从 .mat 读取并完成 Q6：方向调谐 + 岭回归解码 + 可视化
输出：与 .mat 同目录的 q6_results/ 下生成 CSV 和 PNG 图
"""

# ==== 只需改这里：你的 .mat 文件路径 ====
MAT_PATH = r"E:/Various Net/XJTUxch/data/loco_20170301_05.mat"

# ==== 常用参数（可按需微调）====
WIN_MS          = 64       # 分箱宽度 (ms)
SMOOTH_SIGMA_S  = 0.02     # 高斯平滑 σ(s); 0 关闭
MIN_RATE_HZ     = 0.5      # 平均放电率阈值 (Hz)；低于此值的单元将被丢弃
TUNING_R2_THR   = 0.15     # tuned_frac 的 R² 阈值
N_ANGLE_BINS    = 12       # 方向分桶数
MIN_SPEED       = 1e-3     # 调谐阶段过滤静止的速度阈值
RIDGE_ALPHA     = 1.0      # 岭回归正则
N_SPLITS        = 5        # 时序交叉验证折数
PSTH_N          = 6        # PSTH 示例单元数
SEED            = 0        # 随机种子
DEBUG_PRINT_CHN = False    # True 可打印前若干通道名以自检

import os, re, json
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal import fftconvolve

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# ================== HDF5/Mat 读取工具 ==================
def find_dataset_group(root, keywords):
    """在 HDF5 层次结构中递归查找名字包含关键字的数据集/组。"""
    for k in root.keys():
        if any(kw in k.lower() for kw in keywords):
            return root[k]
    for k in root.keys():
        if isinstance(root[k], h5py.Group):
            got = find_dataset_group(root[k], keywords)
            if got is not None:
                return got
    return None

def read_matlab_cellstr(f: h5py.File, ds: h5py.Dataset):
    """把 MATLAB 的 cellstr/char 读成 Python 字符串列表（兼容 object refs / uint16 / uint8 / vlen）。"""
    arr = np.array(ds)

    def _to_str(x):
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode('utf-8', errors='ignore')
        if isinstance(x, str):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                return ''.join(map(chr, x.ravel()))
            if x.dtype == np.uint8:
                try:
                    return x.tobytes().decode('utf-8', errors='ignore')
                except Exception:
                    return ''.join(map(chr, x.ravel()))
            if x.dtype.kind in ('S','U'):
                return str(x[()])
            return str(x.ravel())
        return str(x)

    out = []
    # A) cell-array: 对象引用
    if arr.dtype.kind == 'O' or 'ref' in str(arr.dtype).lower():
        for ref in arr.ravel():
            try:
                obj = f[ref]
                data = np.array(obj)
                if data.ndim >= 1 and (data.dtype == np.uint16 or data.dtype == np.uint8):
                    out.append(_to_str(data))
                elif data.dtype.kind in ('S','U'):
                    out.append(_to_str(data))
                else:
                    out.append(_to_str(np.asarray(data)))
            except Exception:
                out.append(str(ref))
    else:
        # B) char 矩阵 / vlen
        if arr.ndim == 2 and (arr.dtype == np.uint16 or arr.dtype == np.uint8):
            for i in range(arr.shape[0]):
                out.append(_to_str(arr[i, :]))
        else:
            for x in arr.ravel():
                out.append(_to_str(x))
    return [s.strip() for s in out]

def align_channel_names(ch_names, n_units):
    """
    把 ch_names 对齐到 n_units 长度：
      - None：生成占位名
      - 过短：用 'Unknown ###' 填充
      - 过长：截断
    返回：长度 == n_units 的 list[str]
    """
    if ch_names is None:
        return [f"Unknown {i:03d}" for i in range(n_units)]
    ch_names = [str(x) for x in ch_names]
    if len(ch_names) < n_units:
        ch_names = ch_names + [f"Unknown {i:03d}" for i in range(len(ch_names), n_units)]
    elif len(ch_names) > n_units:
        ch_names = ch_names[:n_units]
    return ch_names

# ================== 会话读取 ==================
def load_session(path, prefer_finger=True, verbose=True):
    """
    返回：
      t: (T,)
      pos: (T,2)
      spike_list: list[np.ndarray] 每个元素是秒级脉冲时刻（可选）
      spike_matrix: (time x units) 或 None
      ch_names: list[str] 或 None
    """
    if verbose: print(f"[load_session] opening {path}")
    with h5py.File(path, 'r') as f:
        # timestamps
        t_ds = find_dataset_group(f, ['t', 'time', 'timestamps'])
        if t_ds is None:
            raise RuntimeError("Cannot find timestamps dataset (t).")
        t = np.array(t_ds).squeeze()

        # position：优先 finger_pos
        pos_ds = find_dataset_group(f, ['finger_pos', 'fingerpos', 'hand_pos']) if prefer_finger else None
        if pos_ds is None:
            pos_ds = find_dataset_group(f, ['cursor_pos', 'cursorpos', 'cursor', 'pos', 'position'])
        if pos_ds is None:
            raise RuntimeError("Cannot find finger_pos or cursor_pos dataset.")
        pos_arr = np.array(pos_ds)
        if pos_arr.ndim == 2:
            pos = pos_arr if pos_arr.shape[0] == t.size else pos_arr.T
        elif pos_arr.ndim == 1:
            pos = pos_arr.reshape(-1, 2) if pos_arr.size % 2 == 0 else pos_arr.reshape(-1, 3)
        else:
            pos = pos_arr.reshape(t.size, -1)
        pos = pos[:, :2]

        # ch_names
        ch_names = None
        ch_names_ds = find_dataset_group(f, ['chan_names', 'ch_names', 'channel', 'labels'])
        if ch_names_ds is not None:
            try:
                ch_names = read_matlab_cellstr(f, ch_names_ds)
            except Exception as e:
                if verbose: print("[W] read_matlab_cellstr failed:", e)
                ch_names = None

        # spikes
        spikes_ds = find_dataset_group(f, ['spike', 'spikes', 'unit', 'spiketimes', 'spike_times'])
        spike_list, spike_matrix = None, None
        if spikes_ds is not None:
            try:
                arr = np.array(spikes_ds)
                if arr.dtype.kind == 'O' or 'ref' in str(arr.dtype).lower():
                    spike_list = []
                    for ref in arr.ravel():
                        try:
                            sp = np.array(f[ref]).squeeze()
                            spike_list.append(np.asarray(sp).ravel())
                        except Exception:
                            spike_list.append(np.array([]))
                else:
                    mat = np.asarray(arr)
                    if mat.ndim == 2 and mat.shape[0] == t.size:
                        spike_matrix = mat
                    elif mat.ndim == 2 and mat.shape[1] == t.size:
                        spike_matrix = mat.T
                    else:
                        spike_matrix = mat  # 兜底：后续再对齐
            except Exception as e:
                if verbose: print("[W] parsing spikes failed:", e)

    if verbose:
        print(f"  t shape: {t.shape}, pos shape: {pos.shape}, "
              f"ch_names: {None if ch_names is None else len(ch_names)}")
    return {'t': np.asarray(t).ravel(), 'pos': np.asarray(pos),
            'spike_list': spike_list, 'spike_matrix': spike_matrix,
            'ch_names': ch_names}

# ================== 发放率（分箱+平滑） ==================
def compute_binned_rates(session, win_s=0.064, smooth_sigma_s=None, verbose=True):
    t = session['t']
    spike_list = session.get('spike_list', None)
    spike_matrix = session.get('spike_matrix', None)

    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    bins = np.arange(t0, t1 + 1e-9, win_s)
    centers = bins[:-1] + win_s/2.0
    n_bins = len(centers)

    if spike_list is not None:
        n_units = len(spike_list)
        rates = np.zeros((n_units, n_bins), float)
        for i, st in enumerate(spike_list):
            if st is None or len(st) == 0: continue
            cnt, _ = np.histogram(np.asarray(st).ravel(), bins=bins)
            rates[i, :] = cnt / win_s
        source = 'list'
    elif spike_matrix is not None:
        mat = np.asarray(spike_matrix)  # 期望 time x units
        if mat.ndim != 2:
            raise RuntimeError("Invalid spike_matrix shape.")
        # 若时间维不等于 len(t)，做最近邻重采样以对齐
        if mat.shape[0] != len(t):
            tt = np.linspace(t0, t1, mat.shape[0])
            mat_rs = np.zeros((len(t), mat.shape[1]))
            for j in range(mat.shape[1]):
                f = interpolate.interp1d(tt, mat[:,j], bounds_error=False, fill_value="extrapolate")
                mat_rs[:,j] = f(t)
            mat = mat_rs
        inds = np.digitize(t, bins) - 1
        n_units = mat.shape[1]
        rates = np.zeros((n_units, n_bins), float)
        for b in range(n_bins):
            sel = (inds == b)
            if sel.any():
                rates[:, b] = mat[sel, :].sum(axis=0) / win_s
        source = 'matrix'
    else:
        raise RuntimeError("No spike data found.")

    # 高斯平滑
    if smooth_sigma_s and smooth_sigma_s > 0:
        dt = win_s
        half = max(3*smooth_sigma_s, 3*dt)
        x = np.arange(-half, half + 1e-12, dt)
        kern = norm.pdf(x, scale=smooth_sigma_s)
        kern /= (kern.sum() + 1e-16)
        for i in range(rates.shape[0]):
            rates[i,:] = fftconvolve(rates[i,:], kern, mode='same')

    if verbose:
        print(f"  Binned rates: {rates.shape}, source={source}, win_s={win_s}s")
    return rates, centers

# ================== 运动学插值 ==================
def compute_kinematics(t_samples, pos_samples, bin_centers):
    fx = interpolate.interp1d(t_samples, pos_samples[:,0], bounds_error=False, fill_value="extrapolate")
    fy = interpolate.interp1d(t_samples, pos_samples[:,1], bounds_error=False, fill_value="extrapolate")
    pos = np.stack([fx(bin_centers), fy(bin_centers)], axis=1)
    dt = np.gradient(bin_centers)
    vel = np.gradient(pos, axis=0) / dt.reshape(-1,1)
    acc = np.gradient(vel, axis=0) / dt.reshape(-1,1)
    return pos, vel, acc

# ================== 区域解析 ==================
def parse_region_from_name(name: str) -> str:
    if name is None: return 'Other'
    s = str(name).strip().lower()
    tokens = re.split(r'[\s_\-:/\\]+', s)
    token = next((tok for tok in tokens if tok), "")
    if token == 'm1' or s.startswith('m1'): return 'M1'
    if token == 's1' or s.startswith('s1'): return 'S1'
    return 'Other'

# ================== 调谐/解码 ==================
def cos_func(theta, A, PD, b):
    return A * np.cos(theta - PD) + b

def compute_directional_tuning(rates, vel_binned, n_angle_bins=12, min_speed=1e-3):
    from scipy.interpolate import interp1d
    angles = np.arctan2(vel_binned[:,1], vel_binned[:,0])
    speed  = np.linalg.norm(vel_binned, axis=1)
    sel    = speed > min_speed

    bins    = np.linspace(-np.pi, np.pi, n_angle_bins+1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    results = []

    for i in range(rates.shape[0]):
        fr   = rates[i, sel]
        angs = angles[sel]
        inds = np.digitize(angs, bins) - 1
        fr_means = np.full(n_angle_bins, np.nan, float)
        for b in range(n_angle_bins):
            idx = np.where(inds == b)[0]
            if idx.size > 0:
                fr_means[b] = np.nanmean(fr[idx])

        valid = ~np.isnan(fr_means)
        if valid.sum() < max(3, n_angle_bins//3):
            results.append((i, np.nan, np.nan, np.nan, np.nan)); continue

        if np.any(~valid):
            gx, gy = centers[valid], fr_means[valid]
            try:
                f = interp1d(gx, gy, kind='cubic', fill_value="extrapolate")
                fr_means[~valid] = f(centers[~valid])
            except Exception:
                fr_means[~valid] = np.nanmean(gy)

        try:
            A0  = (np.nanmax(fr_means) - np.nanmin(fr_means)) / 2.0
            PD0 = centers[np.nanargmax(fr_means)]
            b0  = np.nanmean(fr_means)
            popt, _ = curve_fit(cos_func, centers, fr_means, p0=[A0, PD0, b0], maxfev=5000)
            pred  = cos_func(centers, *popt)
            denom = np.nansum((fr_means - np.nanmean(fr_means))**2)
            r2    = 0.0 if denom <= 1e-12 else 1.0 - np.nansum((fr_means - pred)**2)/denom
            results.append((i, float(popt[0]), float(popt[1]), float(popt[2]), float(r2)))
        except Exception:
            results.append((i, np.nan, np.nan, np.nan, np.nan))

    df = pd.DataFrame(results, columns=['neuron','A','PD','b','r2'])
    return df, centers

def decode_velocity_ridge(rates, vel_binned, alpha=1.0, n_splits=5):
    X = rates.T; y = vel_binned
    if X.shape[1] == 0: return np.array([np.nan, np.nan])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=alpha))
        p = model.fit(X[tr], y[tr]).predict(X[te])
        if p.shape[0] == 0:
            scores.append([np.nan, np.nan]); continue
        scores.append([r2_score(y[te][:,0], p[:,0]), r2_score(y[te][:,1], p[:,1])])
    return np.nanmean(np.array(scores, float), axis=0)

# ================== 画图 ==================
def plot_sample_psth(rates, centers, still_mask, neuron_indices, outpath):
    n = len(neuron_indices)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2*n), sharex=True)
    if n == 1: axes = [axes]
    for ax, ni in zip(axes, neuron_indices):
        ax.plot(centers, rates[ni,:], lw=0.8)
        if still_mask is not None:
            y0, y1 = ax.get_ylim()
            ax.fill_between(centers, y0, y1, where=still_mask, color='orange', alpha=0.2)
        ax.set_ylabel(f"unit {ni}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("PSTH (smoothed rates) with still epochs shaded")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

def plot_pd_unit_circle(df_tune, outpath):
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, polar=True)
    for region, marker, alpha in [('M1','o',0.85),('S1','x',0.85),('Other','.',0.6)]:
        sub = df_tune[df_tune['region']==region]
        if len(sub)>0:
            ax.scatter(sub['PD'].values, np.abs(sub['A'].values), marker=marker, alpha=alpha, label=f"{region} (n={len(sub)})")
    ax.set_title("Preferred directions (radius = |A|) by region")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1))
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

def plot_sample_tuning_curves(rates, vel_binned, neurons, centers_ang, outpath):
    angles = np.arctan2(vel_binned[:,1], vel_binned[:,0])
    step = (centers_ang[1]-centers_ang[0])
    bins = np.concatenate((centers_ang - step/2, [centers_ang[-1]+step/2]))
    fig, axes = plt.subplots(len(neurons), 1, figsize=(7, 2.3*len(neurons)), sharex=True)
    if len(neurons) == 1: axes = [axes]
    for ax, ni in zip(axes, neurons):
        inds = np.digitize(angles, bins) - 1
        fr_means = [np.nanmean(rates[ni, inds==b]) if np.any(inds==b) else np.nan for b in range(len(centers_ang))]
        axs = np.linspace(-np.pi, np.pi, 300)
        try:
            y_in = np.array(fr_means)
            if np.any(np.isnan(y_in)):
                y_in = np.where(np.isnan(y_in), np.nanmean(y_in[~np.isnan(y_in)]), y_in)
            popt, _ = curve_fit(cos_func, centers_ang, y_in)
            ax.plot(centers_ang, fr_means, 'o-'); ax.plot(axs, cos_func(axs, *popt), '--')
            ax.set_ylabel(f"unit {ni}")
        except Exception:
            ax.plot(centers_ang, fr_means, 'o-'); ax.set_ylabel(f"unit {ni} (fit failed)")
    axes[-1].set_xlabel("angle (rad)")
    fig.suptitle("Example tuning curves")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

# ================== 主流程（一键） ==================
def main():
    assert os.path.exists(MAT_PATH), f"找不到 .mat 文件：{MAT_PATH}"
    outdir = os.path.join(os.path.dirname(MAT_PATH), "q6_results")
    os.makedirs(outdir, exist_ok=True)

    # 1) 读取
    sess = load_session(MAT_PATH, prefer_finger=True, verbose=True)

    # 2) 分箱 + 平滑
    win_s = WIN_MS / 1000.0
    rates, bin_centers = compute_binned_rates(sess, win_s=win_s, smooth_sigma_s=SMOOTH_SIGMA_S, verbose=True)

    # 3) 先计算 keep_idx（不要急着裁剪 rates；以免后续 ch_names 索引错位）
    mean_rates = rates.mean(axis=1)
    keep_idx = np.where(mean_rates >= MIN_RATE_HZ)[0]
    n_units_orig = rates.shape[0]

    # 4) 通道名对齐到原始单位数，再用 keep_idx 取子集（防止越界）
    ch_names_full = align_channel_names(sess.get('ch_names', None), n_units_orig)
    ch_names_kept = [ch_names_full[i] for i in keep_idx]

    if DEBUG_PRINT_CHN:
        print("[debug] first 10 ch_names_full:", ch_names_full[:10])
        prefixes = [re.split(r'[\s_\-:/\\]+', s.lower())[0] for s in ch_names_full]
        print("[debug] prefix counts:", pd.Series(prefixes).value_counts().to_dict())

    # 5) 现在再裁剪 rates
    rates = rates[keep_idx, :]
    print(f"[keep] {len(keep_idx)} units kept (>= {MIN_RATE_HZ} Hz)")

    # 6) 运动学
    pos_b, vel_b, acc_b = compute_kinematics(sess['t'], sess['pos'], bin_centers)
    spd = np.linalg.norm(vel_b, axis=1)
    still = (spd <= np.percentile(spd, 20.0))

    # 7) 区域标签（基于“已保留单元”的通道名）
    regions = np.array([parse_region_from_name(nm) for nm in ch_names_kept], dtype=object)

    # 8) 调谐
    df_tune, centers_ang = compute_directional_tuning(rates, vel_binned=vel_b, n_angle_bins=N_ANGLE_BINS, min_speed=MIN_SPEED)
    df_tune['region'] = regions[df_tune['neuron'].values]
    df_tune.to_csv(os.path.join(outdir, "tuning_by_neuron.csv"), index=False)

    # 区域汇总
    rows = []
    for reg in ['M1','S1','Other']:
        sub = df_tune[df_tune['region']==reg]
        if len(sub)==0:
            rows.append({'region':reg,'n_neurons':0,'tuned_frac':np.nan,'median_abs_A':np.nan,'median_r2':np.nan})
        else:
            rows.append({
                'region': reg,
                'n_neurons': int(len(sub)),
                'tuned_frac': float(np.nanmean(sub['r2'] > TUNING_R2_THR)),
                'median_abs_A': float(np.nanmedian(np.abs(sub['A']))),
                'median_r2': float(np.nanmedian(sub['r2']))
            })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(outdir, "tuning_summary_by_region.csv"), index=False)

    # 9) 解码（M1/S1/Other/All）
    def _decode(mask):
        return decode_velocity_ridge(rates[mask,:], vel_binned=vel_b, alpha=RIDGE_ALPHA, n_splits=N_SPLITS) if mask.sum()>0 else np.array([np.nan, np.nan])
    mask_m1, mask_s1, mask_other = (regions=='M1'), (regions=='S1'), (regions=='Other')
    r2_m1, r2_s1, r2_other = _decode(mask_m1), _decode(mask_s1), _decode(mask_other)
    r2_all = decode_velocity_ridge(rates, vel_binned=vel_b, alpha=RIDGE_ALPHA, n_splits=N_SPLITS)
    df_decode = pd.DataFrame({
        'region':['M1','S1','Other','All'],
        'r2_x':[r2_m1[0], r2_s1[0], r2_other[0], r2_all[0]],
        'r2_y':[r2_m1[1], r2_s1[1], r2_other[1], r2_all[1]],
    })
    df_decode.to_csv(os.path.join(outdir, "decode_by_region.csv"), index=False)

    # 10) 作图
    rng = np.random.default_rng(SEED)
    pick = rng.choice(rates.shape[0], size=min(PSTH_N, rates.shape[0]), replace=False).tolist()
    plot_sample_psth(rates, bin_centers, still, pick, os.path.join(outdir, "psth_with_still.png"))
    plot_pd_unit_circle(df_tune, os.path.join(outdir, "pd_unit_circle_by_region.png"))
    top_units = df_tune.dropna().sort_values(['r2','A'], ascending=[False, False]).head(min(6, len(df_tune)))['neuron'].tolist()
    plot_sample_tuning_curves(rates, vel_binned=vel_b, neurons=top_units, centers_ang=centers_ang, outpath=os.path.join(outdir, "example_tuning_curves.png"))

    # 11) 记录 run 信息
    info = dict(
        mat=MAT_PATH, outdir=outdir, win_ms=WIN_MS, smooth_sigma_s=SMOOTH_SIGMA_S,
        min_rate_hz=MIN_RATE_HZ, tuning_r2_thr=TUNING_R2_THR, n_angle_bins=N_ANGLE_BINS,
        min_speed=MIN_SPEED, alpha=RIDGE_ALPHA, splits=N_SPLITS, psth_n=PSTH_N, seed=SEED,
        n_units=int(rates.shape[0]), n_time=int(rates.shape[1]),
        region_counts={k:int(v) for k,v in pd.Series(regions).value_counts().to_dict().items()}
    )
    with open(os.path.join(outdir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n[Outputs] written to:", outdir)
    print("\n[Tuning summary]\n", df_summary)
    print("\n[Decoding R2]\n", df_decode)

if __name__ == "__main__":
    main()
