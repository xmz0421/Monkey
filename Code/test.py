import h5py, numpy as np

path = r"E:/Various Net/XJTUxch/data/indy_20160921_01.mat"
with h5py.File(path,'r') as f:
    spikes_refs = f['spikes']
    print("spikes shape:", spikes_refs.shape)
    ref = spikes_refs[0,0]  # 第1个神经元
    print("type:", type(ref))
    data = np.array(f[ref])
    print("ref dataset shape:", data.shape, "dtype:", data.dtype, "example:", data[:10])
