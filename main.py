import h5py
import numpy as np
from scipy.stats import stats

f = h5py.File('E:/Various Net/XJTUxch/data/loco_20170301_05.mat', 'r')
g=list(f.keys())
chan_names = f['chan_names'] #脑区＋编号 n为通道数，96只有M1,192既有M1又有S1
t = f['t'] #时间戳 k是时间点个数，采样频率是250Hz
finger_pos = f['finger_pos'] #手指位置 3指的是（z，x，y），后面是时间点个数
spikes = f['spikes'] #神经元放电序列
# print(chan_names,t,finger_pos,spikes)

# 将HDF5数据集转换为NumPy数组
t_array = np.array(t).flatten()  # 时间戳数组
total_duration = t_array[-1] - t_array[0]  # 总记录时长

# 获取spikes数据的形状
spikes_shape = spikes.shape
n_channels = spikes_shape[0]  # 通道数
n_units = spikes_shape[1]  # 每个通道上的单元数

print(f"数据信息: {n_channels}个通道, 每个通道最多{n_units}个单元")
print(f"记录时长: {total_duration:.2f}秒")

# 筛选有效神经元（平均发放率≥0.5Hz）
valid_neurons = []  # 存储有效神经元信息
neuron_counter = 0  # 有效神经元计数器

# 遍历所有通道和单元
for chan_idx in range(n_channels):
    for unit_idx in range(n_units):
        # 获取当前神经元的脉冲时间数据
        spike_times_ref = spikes[chan_idx, unit_idx]
        if isinstance(spike_times_ref, h5py.Reference):
            spike_dataset = f[spike_times_ref]
            spike_times = np.array(spike_dataset).flatten()
        else:
            spike_times = np.array(spike_times_ref).flatten()

        # 计算平均发放率
        n_spikes = len(spike_times)
        avg_firing_rate = n_spikes / total_duration  # 单位: Hz

        # 只保留发放率≥0.5Hz的神经元
        if avg_firing_rate >= 0.5:
            valid_neurons.append({
                'channel_idx': chan_idx,
                'unit_idx': unit_idx,
                'spike_times': spike_times,
                'avg_rate': avg_firing_rate
            })
            neuron_counter += 1

print(f"找到 {neuron_counter} 个有效神经元(发放率≥0.5Hz)")

# 设置时间窗参数
window_width = 0.064  # 64ms时间窗
# 创建时间窗中心点（与行为数据采样点对齐）
time_bins = t_array  # 直接使用原始时间戳作为窗中心

# 初始化发放率矩阵 [时间点数量 × 有效神经元数量]
firing_rates = np.zeros((len(time_bins), len(valid_neurons)))

# 计算每个有效神经元的发放率
for neuron_idx, neuron in enumerate(valid_neurons):
    spike_times = neuron['spike_times']

    # 使用直方图统计每个时间窗内的脉冲数
    # 时间窗边界：每个时间点前后32ms
    window_edges = np.concatenate((
        [time_bins[0] - window_width / 2],  # 第一个时间点之前的边界
        time_bins + window_width / 2,  # 每个时间点之后的边界
    ))

    # 计算每个时间窗内的脉冲数
    spike_counts, _ = np.histogram(spike_times, bins=window_edges)

    # 转换为发放率 (Hz)，并存储
    firing_rates[:, neuron_idx] = spike_counts / window_width

# 可选：对发放率进行高斯平滑（σ=16ms）
sigma = 0.016  # 16ms
# 创建高斯核
kernel_size = int(4 * sigma / (time_bins[1] - time_bins[0]))  # 核大小
if kernel_size % 2 == 0:
    kernel_size += 1  # 确保核大小为奇数
x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
gaussian_kernel = stats.norm.pdf(x, scale=sigma / (time_bins[1] - time_bins[0]))
gaussian_kernel /= np.sum(gaussian_kernel)  # 归一化

# 对每个神经元的发放率进行平滑
smoothed_rates = np.zeros_like(firing_rates)
for i in range(firing_rates.shape[1]):
    smoothed_rates[:, i] = np.convolve(
        firing_rates[:, i], gaussian_kernel, mode='same'
    )

print("发放率计算完成!")
print(f"发放率矩阵形状: {smoothed_rates.shape} (时间点×神经元)")
