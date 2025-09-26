import os, math, time, random
import numpy as np
import h5py
from scipy.signal import savgol_filter, fftconvolve
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------
# Reproducibility
# ------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------
# Dataset & Preprocessing utils
# ------------------------
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
    with h5py.File(path, 'r') as f:
        t_ds = find_dataset_group(f, ['t', 'time', 'timestamps'])
        t = np.array(t_ds).squeeze()

        pos_ds = None
        if prefer_finger:
            pos_ds = find_dataset_group(f, ['finger_pos', 'fingerpos', 'hand_pos'])
        if pos_ds is None:
            pos_ds = find_dataset_group(f, ['cursor_pos', 'cursorpos', 'cursor', 'pos', 'position'])
        pos = np.array(pos_ds).T
        pos = pos[:,:2]

        spikes_ds = find_dataset_group(f, ['spike', 'spikes'])
        arr = np.array(spikes_ds)
        spike_list = []
        if arr.dtype.kind == 'O':  # reference
            n_units = arr.shape[1]
            for i in range(n_units):
                ref = arr[0,i]
                sp = np.array(f[ref]).squeeze() if ref else []
                spike_list.append(np.asarray(sp).ravel())
        else:
            raise RuntimeError("Unexpected spike format")
    return {'t': t, 'pos': pos, 'spike_list': spike_list}

def compute_binned_rates(session, win_s=0.064, smooth_sigma_s=0.02, sg_window=9):
    t = session['t']
    spike_list = session['spike_list']
    bins = np.arange(t.min(), t.max()+1e-9, win_s)
    bin_centers = bins[:-1] + win_s/2
    rates = np.zeros((len(bin_centers), len(spike_list)))
    for i, st in enumerate(spike_list):
        if st is None or len(st)==0: continue
        counts,_ = np.histogram(st,bins=bins)
        rates[:,i] = counts/win_s
    if smooth_sigma_s>0:
        dt=win_s
        x = np.arange(-3*smooth_sigma_s,3*smooth_sigma_s,dt)
        kernel = norm.pdf(x, scale=smooth_sigma_s)
        kernel /= kernel.sum()
        for i in range(rates.shape[1]):
            rates[:,i] = fftconvolve(rates[:,i],kernel,mode='same')
    rates = savgol_filter(rates, window_length=sg_window if sg_window%2==1 else sg_window+1, polyorder=3, axis=0)
    return rates, bin_centers

def compute_kinematics(t_samples,pos_samples,bin_centers):
    fx=np.interp(bin_centers,t_samples,pos_samples[:,0])
    fy=np.interp(bin_centers,t_samples,pos_samples[:,1])
    pos=np.stack([fx,fy],axis=1)
    vel=np.gradient(pos,bin_centers,axis=0)
    acc=np.gradient(vel,bin_centers,axis=0)
    return pos,vel,acc

def process_and_save_session(path, out_path, win_ms=64, min_rate_hz=0.5, smooth_sigma_s=0.02, sg_window=9):
    sess=load_session(path,prefer_finger=True,verbose=True)
    rates,bin_centers=compute_binned_rates(sess,win_s=win_ms/1000.0,smooth_sigma_s=smooth_sigma_s,sg_window=sg_window)
    keep_idx=np.where(rates.mean(0)>=min_rate_hz)[0]
    rates=rates[:,keep_idx]
    pos,vel,acc=compute_kinematics(sess['t'],sess['pos'],bin_centers)
    xsc=StandardScaler().fit(rates)
    Xn=xsc.transform(rates)
    np.savez_compressed(out_path,X=Xn.astype(np.float32),Y_pos=pos.astype(np.float32),
                        Y_vel=vel.astype(np.float32),Y_acc=acc.astype(np.float32),
                        bin_centers=bin_centers.astype(np.float32))
    return out_path

# ------------------------
# Dataset for sequences
# ------------------------
class SeqDataset(Dataset):
    def __init__(self,X,Z,window):
        self.X=X;self.Z=Z;self.window=window;self.half=window//2
        self.valid_len=X.shape[0]-window
    def __len__(self): return max(0,self.valid_len)
    def __getitem__(self,idx):
        start=idx;end=start+self.window
        xseq=self.X[start:end,:].T
        y=self.Z[start+self.half,:]
        return xseq.astype(np.float32),y.astype(np.float32)

def compute_test_times(bin_centers,T_total,i_val,window):
    half=window//2
    return bin_centers[i_val+half:T_total-half]

# ------------------------
# Model: CNN+BiLSTM
# ------------------------
class CNNBiLSTM(nn.Module):
    def __init__(self,n_channels,window,out_dim=6,dropout=0.25):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(n_channels,64,5,padding=2),nn.ReLU(),nn.Dropout(dropout),
            nn.Conv1d(64,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2))
        self.lstm=nn.LSTM(64,64,num_layers=1,bidirectional=True,batch_first=True)
        self.fc=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,out_dim))
    def forward(self,x):
        feat=self.cnn(x)   #(B,64,L)
        feat=feat.permute(0,2,1)
        _,(hn,_) = self.lstm(feat)
        h=torch.cat([hn[0],hn[1]],dim=1)
        return self.fc(h)

# ------------------------
# Training with Δ-loss + smoothing
# ------------------------
def train_deep_decoder(npzfile,window=25,epochs=60,train_frac=0.7,val_frac=0.1,
                       batch_size=128,lr=1e-3,device=None):
    data=np.load(npzfile,allow_pickle=True)
    X=data['X']; Y=np.concatenate([data['Y_pos'],data['Y_vel'],data['Y_acc']],axis=1)
    bin_centers=data['bin_centers']; T=X.shape[0]

    i_tr=int(T*train_frac); i_val=int(T*(train_frac+val_frac))
    ysc=StandardScaler().fit(Y[:i_tr]); Ysc=ysc.transform(Y)
    ds_tr=SeqDataset(X[:i_tr],Ysc[:i_tr],window)
    ds_val=SeqDataset(X[i_tr:i_val],Ysc[i_tr:i_val],window)
    ds_test=SeqDataset(X[i_val:],Ysc[i_val:],window)
    loader_tr=DataLoader(ds_tr,batch_size=batch_size,shuffle=True)
    loader_val=DataLoader(ds_val,batch_size=batch_size)
    loader_test=DataLoader(ds_test,batch_size=batch_size)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    model=CNNBiLSTM(X.shape[1],window).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    crit=nn.MSELoss()

    best=np.inf;best_state=None
    for ep in range(epochs):
        model.train();loss_sum=0
        for xb,yb in loader_tr:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad();out=model(xb)
            # Δ-loss penalty
            delta_out=out[:,1:]-out[:,:-1] if out.shape[1]>1 else out
            delta_yb=yb[:,1:]-yb[:,:-1] if yb.shape[1]>1 else yb
            loss=crit(out,yb)+0.1*crit(delta_out,delta_yb)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();loss_sum+=loss.item()*xb.size(0)
        val_loss=0
        model.eval()
        with torch.no_grad():
            for xb,yb in loader_val:
                xb,yb=xb.to(device),yb.to(device)
                out=model(xb)
                loss=crit(out,yb);val_loss+=loss.item()*xb.size(0)
        if val_loss<best:best=val_loss;best_state=model.state_dict()
        if ep%5==0:print(f"[Epoch {ep}] train_loss={loss_sum/len(ds_tr):.4f} val_loss={val_loss/len(ds_val):.4f}")
    model.load_state_dict(best_state)

    preds=[];trues=[]
    with torch.no_grad():
        for xb,yb in loader_test:
            out=model(xb.to(device)).cpu().numpy()
            preds.append(out);trues.append(yb.numpy())
    preds=np.vstack(preds);trues=np.vstack(trues)
    preds=ysc.inverse_transform(preds);trues=ysc.inverse_transform(trues)
    bin_test=compute_test_times(bin_centers,T,i_val,window)

    # post smoothing
    for d in range(preds.shape[1]):
        preds[:,d]=savgol_filter(preds[:,d],window_length=9,polyorder=2)

    r2s=[r2_score(trues[:,d],preds[:,d]) for d in range(6)]
    mses=[mean_squared_error(trues[:,d],preds[:,d]) for d in range(6)]
    snrs=[-10*math.log10(1-r2) if (r2<1 and r2>-np.inf) else np.nan for r2 in r2s]
    return {'model':model,'preds':preds,'trues':trues,'r2s':r2s,'snrs':snrs,'mses':mses,'bin_centers_test':bin_test}

# ------------------------
# Plot
# ------------------------
def plot_pred_vs_gt_grid(res,time_window_sec=6.0):
    preds,trues=res['preds'],res['trues'];times=res['bin_centers_test']
    pred_pos,pred_vel,pred_acc=preds[:,:2],preds[:,2:4],preds[:,4:6]
    true_pos,true_vel,true_acc=trues[:,:2],trues[:,2:4],trues[:,4:6]
    center=times[len(times)//2];half=time_window_sec/2
    sel=np.where((times>=center-half)&(times<=center+half))[0]
    fig,axes=plt.subplots(2,3,figsize=(15,6),sharex=True)
    groups=[('Position',true_pos,pred_pos,0),('Velocity',true_vel,pred_vel,2),('Acceleration',true_acc,pred_acc,4)]
    for col,(ttl,gt,pd,idx) in enumerate(groups):
        for row in range(2):
            ax=axes[row,col]
            ax.plot(times[sel],gt[sel,row],'k-',label='gt')
            ax.plot(times[sel],pd[sel,row],'r--',label='pred')
            ax.set_title(ttl if row==0 else "");ax.set_ylabel('x' if row==0 else 'y')
            ax.text(0.02,0.85,f"R²={res['r2s'][idx+row]:.3f}\nSNR={res['snrs'][idx+row]:.1f} dB\nMSE={res['mses'][idx+row]:.3f}",
                    transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=0.6))
    plt.tight_layout();plt.show()

# ------------------------
# Main
# ------------------------
if __name__=="__main__":
    session_file=r"E:/Various Net/XJTUxch/data/loco_20170301_05.mat"
    out_npz="深度学习——优化.npz"
    if not os.path.exists(out_npz):
        process_and_save_session(session_file,out_npz)
    res=train_deep_decoder(out_npz,window=25,epochs=60)
    print("Final metrics:")
    for name,r2,snr,mse in zip(['pos_x','pos_y','vel_x','vel_y','acc_x','acc_y'],res['r2s'],res['snrs'],res['mses']):
        print(f"{name:6s} R2={r2:.3f} SNR={snr:.2f}dB MSE={mse:.3f}")
    plot_pred_vs_gt_grid(res)