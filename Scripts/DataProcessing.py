import os
import h5py
import numpy as np
from scipy.stats import pearsonr

def compute_noise_ceiling(resp_mat, stim_idx, n_splits=100):
    """
    Compute noise ceiling using split-half reliability with Pearson correlation.
    
    Parameters
    ----------
    resp_mat : np.ndarray, shape (n_trials, n_units)
        Neural responses. Each row is a trial, each column is a unit.
    stim_idx : np.ndarray, shape (n_trials,)
        Stimulus index (int from 1..n_images).
    n_splits : int
        Number of random split-half repetitions.
    
    Returns
    -------
    noise_ceiling : np.ndarray, shape (n_units,)
        Noise ceiling estimate for each unit.
    """
    n_trials, n_units = resp_mat.shape
    stim_idx = np.array(stim_idx)
    unique_stim = np.unique(stim_idx)
    n_stim = len(unique_stim)

    corr_all = np.zeros((n_splits, n_units))
    
    for split in range(n_splits):
        # --- (1) Split trials into two halves randomly for each stimulus ---
        resp1 = np.zeros((n_stim, n_units))
        resp2 = np.zeros((n_stim, n_units))
        
        for i, stim in enumerate(unique_stim):
            trials = np.where(stim_idx == stim)[0]
            np.random.shuffle(trials)
            
            half = len(trials) // 2
            trials1, trials2 = trials[:half], trials[half:]
            
            if len(trials1) > 0:
                resp1[i] = resp_mat[trials1].mean(axis=0)
            if len(trials2) > 0:
                resp2[i] = resp_mat[trials2].mean(axis=0)
        
        # --- (2) Compute Pearson correlation across stimuli for each unit ---
        for u in range(n_units):
            r, _ = pearsonr(resp1[:, u], resp2[:, u])
            corr_all[split, u] = r
    
    # --- (3) Average across splits ---
    noise_ceiling = np.nanmean(corr_all, axis=0)
    corrected_noise_ceiling = (2*noise_ceiling)/(1+noise_ceiling)
    return corrected_noise_ceiling

def Load_data_from_GoodUnit(file_dir, save_path=None, start_time=60, end_time=220):
    if file_dir.endswith('.mat'):
        file_list = [file_dir]
    else:
        file_list = [os.path.join(file_dir, i) for i in os.listdir(file_dir) if i.endswith('.mat')]
    resp_total, nc_total = np.array([]), np.array([])
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            n_units = len(f['GoodUnitStrc']['response_matrix_img'])
            pre_onset = int(np.ravel(np.array(f[next(k for k in f.keys() if k.lower()=="global_params")]
             [next(k for k in f[next(k for k in f.keys() if k.lower()=="global_params")].keys() if k.lower()=="pre_onset")]))[0])
            cur_start_time, cur_end_time = pre_onset + start_time, pre_onset + end_time
            trial_valid_idx, dataset_valid_idx = np.array(f['meta_data']['trial_valid_idx']).reshape(-1).astype(np.int32), np.array(f['meta_data']['dataset_valid_idx']).reshape(-1).astype(bool)
            trial_idx = trial_valid_idx[dataset_valid_idx]
            response_list, raster_list = [], []
            for unit_idx in range(n_units):
                resp_mat = f[f['GoodUnitStrc']['response_matrix_img'][unit_idx][0]]
                raster = f[f['GoodUnitStrc']['Raster'][unit_idx][0]]
                cur_resp = np.mean(resp_mat[cur_start_time:cur_end_time], axis=0)
                cur_raster = np.mean(raster[cur_start_time:cur_end_time], axis=0) * 1000
                response_list.append(cur_resp)
                raster_list.append(cur_raster)
            response_arr, raster_arr = np.array(response_list), np.array(raster_list)
            noise_ceiling = compute_noise_ceiling(raster_arr.T, trial_idx, n_splits=100)
        if resp_total.size == 0:
            resp_total = response_arr
        else:
            resp_total = np.concatenate([resp_total, response_arr], axis=0)
        if nc_total.size == 0:
            nc_total = noise_ceiling
        else:
            nc_total = np.concatenate([nc_total, noise_ceiling], axis=0)

    merged_data = {'data': resp_total, 'nc': nc_total}
    if save_path:
        np.savez(save_path, data=resp_total, nc=nc_total)
    return merged_data

file_dir = r"D:\Dataset\AxisData_EVC" # One .mat file or the path of data folder
save_path = r"D:\Analysis\NSD_Alignment\Data\AxisData.npz"
merged_data = Load_data_from_GoodUnit(file_dir=file_dir, save_path=save_path)
print(merged_data['data'].shape, merged_data['nc'].shape)