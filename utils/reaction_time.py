#!/usr/bin/env python
# coding: utf-8

#Libraries
from scipy import signal
import numpy as np
from sklearn.linear_model import LinearRegression
from parse_logfile import TextLog
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.metrics import r2_score
from scipy.signal import butter, lfilter
np.random.seed(1)

# Rolling Window Algorithm #
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def extract_trial_arrays(loc, ts, start_states, end_states, tr_id):
    tr_x = loc[np.where(np.logical_and(ts >= start_states[tr_id], ts <= end_states[tr_id,0]))[0],0]
    tr_y = loc[np.where(np.logical_and(ts >= start_states[tr_id], ts <= end_states[tr_id,0]))[0],1]
    tr_t = ts[np.logical_and(ts >= start_states[tr_id], ts <= end_states[tr_id,0])]
    tr_x = tr_x - tr_x[0]
    tr_y = tr_y - tr_y[0]
    tr_t = tr_t - tr_t[0]
    
    return tr_x, tr_y, tr_t
    
# Euclidean Distance #
def euc_dist(x1,y1,x2,y2):
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist
    
# Sess_Data_Maker
def sess_data_maker(filename, species, evt_for_mouse = 3000):
    sess_x = []
    sess_y = []
    sess_t = []

    with TextLog(filename) as log:
        if species=='mouse':
            trinfo=log.get_info_per_trial(return_eventmarkers=True,return_loc=True, start=evt_for_mouse)
        else:
            trinfo=log.get_info_per_trial(return_eventmarkers=True,return_loc=True)
    
    locc = np.array(trinfo["Location"])
    locTs = np.array(trinfo["LocationTs"])

    for tr_id in range(len(locTs)):
        # if np.logical_or(tr_id % 500 == 0, tr_id == len(locTs)-1) :
        #     print("Trial id:", tr_id)

        tr_x = locc[tr_id][:,0] - locc[tr_id][:,0][0]
        tr_y = locc[tr_id][:,1] - locc[tr_id][:,1][0]
        tr_t = locTs[tr_id] - locTs[tr_id][0]
        
        
        sess_x.append(tr_x)
        sess_y.append(tr_y)
        sess_t.append(tr_t)

    sess_data = np.vstack([sess_x,sess_y,sess_t]).T

    return sess_data

def apply_normalized_signal(signal, center_index, target_array, center_index_weights = 0):
    if len(center_index) < 1:
        return target_array

    if type(center_index_weights) is int:
        center_index_weights = list(np.ones(len(center_index)))

    for ii, idx in enumerate(center_index):
        half_size = signal.shape[0] // 2
        start_idx = max(0, idx - half_size)
        end_idx = min(target_array.shape[0], idx + half_size + 1)
        if start_idx == 0:
            signal_start_idx = half_size - idx
        else:
            signal_start_idx = 0
        if end_idx == target_array.shape[0]:
            signal_end_idx = signal_start_idx + end_idx - start_idx
        else:
            signal_end_idx = signal.shape[0] if signal.shape[0] <= end_idx - idx else end_idx - idx + half_size
        section = target_array[start_idx:end_idx]
        signal_section = signal[signal_start_idx:signal_end_idx]*center_index_weights[ii]
        if section.shape[0] != signal_section.shape[0]:
            signal_section = signal_section[:section.shape[0]]
        mask = signal_section > section
        section[mask] = signal_section[mask]
        target_array[start_idx:end_idx] = section
    return target_array

#Find extrema V2 (better results) #
def find_extrema(trajectory, importance = 1, ext_type = "both"):
    dist = int(len(trajectory)/100*1)
    if dist < 1:
        dist = 1

    if ext_type == "both":
        tr_maxima, tr_max_properties = find_peaks(trajectory, prominence = importance,distance = dist, width = len(trajectory)/100*1)
        tr_minima, tr_min_properties  = find_peaks(-trajectory, prominence = importance,distance = dist ,width = len(trajectory)/100*1)
        extrema = np.concatenate([tr_maxima, tr_minima])
        extrema_properties = np.concatenate([tr_max_properties['prominences'], tr_min_properties['prominences']])
    elif ext_type == "minima":
        extrema, ext_prop = find_peaks(-trajectory, prominence = importance,distance = dist ,width = len(trajectory)/100*1)
        if len(extrema) > 0:
            extrema_properties = ext_prop['prominences']
        else:
            extrema_properties = []
    elif ext_type == "maxima":
        extrema, ext_prop = find_peaks(trajectory, prominence = importance,distance = dist, width = len(trajectory)/100*1)
        if len(extrema) > 0:
            extrema_properties = ext_prop['prominences']
        else:
            extrema_properties = []

    return extrema, extrema_properties

def extrema_merge(y1, y1_p, y2, y2_p):
    y1 = np.array(y1) if hasattr(y1, '__iter__') else np.array([y1])
    y1_p = np.array(y1_p) if hasattr(y1_p, '__iter__') else np.array([y1_p])
    y2 = np.array(y2) if hasattr(y2, '__iter__') else np.array([y2])
    y2_p = np.array(y2_p) if hasattr(y2_p, '__iter__') else np.array([y2_p])

    y_grad_ext_all = np.hstack([y1, y2])
    unique_y_grad_ext_all = np.unique(y_grad_ext_all)
    w_vals = []
    for i in unique_y_grad_ext_all:
        y1_p_temp = y1_p[np.where(y1 == i)[0]] if len(y1) > 0 and len(y1_p) > 0 else []
        y2_p_temp = y2_p[np.where(y2 == i)[0]] if len(y2) > 0 and len(y2_p) > 0 else []
        if len(y1_p_temp) == 0 and len(y2_p_temp) == 0:
            w_vals.append(0)
        else:
            w_vals.append(np.max(np.concatenate([y1_p_temp, y2_p_temp])))

    w_vals = np.array(w_vals)
    if len(np.unique(w_vals)) > 1:
        w_vals = 0.5 + 0.5 * (w_vals - min(w_vals)) / (max(w_vals) - min(w_vals))
    elif len(w_vals) >= 1:
        w_vals = np.array(np.ones(len(w_vals)))
    return unique_y_grad_ext_all, w_vals

def weight_normalization(w_vals):
    if len(np.unique(w_vals)) > 1:
        w_vals = np.array(w_vals)
        w_vals = 0.5 + 0.5 * (w_vals - min(w_vals)) / (max(w_vals) - min(w_vals))
    else:
        w_vals = np.ones(len(w_vals))
    return w_vals

def detrend_1d(data, alpha=0.0):

    data = np.asarray(data)
    n = data.shape[0]
    x = np.arange(n)
    x_norm = np.sum(x**2)
    sse = np.sum((data - np.mean(data))**2)
    if alpha == 0.0:
        slope = (n * np.sum(x * data) - np.sum(x) * np.sum(data)) / (n * x_norm - np.sum(x)**2)
    else:
        ridge_coef = alpha * x_norm
        slope = (n * np.sum(x * data) - np.sum(x) * np.sum(data) + ridge_coef * np.mean(data)) / (n * x_norm + ridge_coef)

    intercept = np.mean(data) - slope * np.mean(x)

    detrended_data = data - (slope * x + intercept)

    return detrended_data

# Decay
def exponential_decay(decay, length):

    len_cutoff = int(length*70/100)
    the_len = length-len_cutoff
    x = np.linspace(0, 1, the_len)
    y = -np.exp(decay * x)
    y_normalized = 0.1 + 0.9 * (y - min(y)) / (max(y) - min(y))
    first_part = np.ones(len_cutoff)
    last_y = np.concatenate([first_part, y_normalized])
    return last_y

def decision_detection_V2(sess_data, norm_func):
    
    rt_points, weights, points_of_interest = [], [], []
    
    for tr_id in range(len(sess_data)):
        x = sess_data[tr_id, 0]
        y = sess_data[tr_id, 1]
        t = sess_data[tr_id, 2]

        if len(y) > 30:
            # y grad extrema 
            y_grad = np.gradient(y)
            extrema_grad, extrema_grad_prom = find_extrema(y_grad,0.1)
            extrema_grad_detrend, extrema_grad_detrend_prom = find_extrema(detrend_1d(y_grad,1e-5),0.1)
            y_grad_extrema, y_grad_prom = extrema_merge(extrema_grad, extrema_grad_prom, extrema_grad_detrend, extrema_grad_detrend_prom)
            zero_array = np.zeros(len(t))
            w1 = apply_normalized_signal(norm_func,y_grad_extrema,zero_array, y_grad_prom)

            # cod y 
            y_extrema, y_prom = find_extrema(y)
            zero_array = np.zeros(len(t))
            y_prom = weight_normalization(y_prom)
            w2 = apply_normalized_signal(norm_func,y_extrema,zero_array, y_prom)

            # x grad extrema 
            x_grad = np.gradient(x)
            extrema_grad, extrema_grad_prom = find_extrema(x_grad,0.1,"minima")
            extrema_grad_detrend, extrema_grad_detrend_prom = find_extrema(detrend_1d(x_grad,1e-5),0.1,"minima")
            x_grad_extrema, x_grad_prom = extrema_merge(extrema_grad, extrema_grad_prom, extrema_grad_detrend, extrema_grad_detrend_prom)
            zero_array = np.zeros(len(t))
            w3 = apply_normalized_signal(norm_func,x_grad_extrema,zero_array)

            # cod x
            x_extrema, x_extrema_prom = find_extrema(x)
            zero_array = np.zeros(len(t))
            x_extrema_prom = weight_normalization(x_extrema_prom)
            w4 = apply_normalized_signal(norm_func,x_extrema,zero_array, x_extrema_prom)


            # Surplus extrema
            ideal_y = np.linspace(y[0],y[-1],len(y))
            y_surplus = ideal_y - y
            surplus_extrema, surplus_extrema_prom = find_extrema(y_surplus)
            zero_array = np.zeros(len(t))
            surplus_extrema_prom = weight_normalization(surplus_extrema_prom)
            w5 = apply_normalized_signal(norm_func,surplus_extrema,zero_array, surplus_extrema_prom)


            # Fast R^2
            y_windows = rolling_window(y,9)
            t_val = np.array(list(range(len(y_windows[0]))))
            lr = LinearRegression()
            lr.fit(t_val.reshape(-1, 1), y_windows.T)
            r2_scores = r2_score(y_windows.T, lr.predict(t_val.reshape(-1, 1)), multioutput='raw_values')
            r2_scores[np.where(np.var(y_windows,1) < 1)[0]] = 1
            r2_scores = np.concatenate([r2_scores,np.ones(len(t_val)-1)])
            peaks_r2, _ = find_extrema(r2_scores,0.1)
            peaks_r2 = peaks_r2+4
            zero_array = np.zeros(len(t))
            w6 = apply_normalized_signal(norm_func,peaks_r2,zero_array)

            # Slow R^2
            y_windows = rolling_window(y,18)
            t_val = np.array(list(range(len(y_windows[0]))))
            lr = LinearRegression()
            lr.fit(t_val.reshape(-1, 1), y_windows.T)
            r2_scores = r2_score(y_windows.T, lr.predict(t_val.reshape(-1, 1)), multioutput='raw_values')
            r2_scores[np.where(np.var(y_windows,1) < 1)[0]] = 1
            r2_scores = np.concatenate([r2_scores,np.ones(len(t_val)-1)])
            peaks_r2, _ = find_extrema(r2_scores,0.1)
            peaks_r2 = peaks_r2+8
            zero_array = np.zeros(len(t))
            w6_1 = apply_normalized_signal(norm_func,peaks_r2,zero_array)

            # Stillness
            y_windows = rolling_window(y,22)
            t_val = np.array(list(range(len(y_windows[0]))))
            trial_var = np.var(y_windows,1)
            trial_var = np.concatenate([trial_var,np.ones(len(t_val)-1)*100])
            still_points = np.where(trial_var < 1)[0]
            zero_array = np.zeros(len(t))
            w7 = apply_normalized_signal(norm_func,still_points,zero_array)

            # Decay
            decay_func = exponential_decay(1,len(w1))

            # 5% Rule
            five_percent_len = int(len(y)/100*5)
            zero_array = np.zeros(len(t))
            zero_array[0:five_percent_len] = -50
            w0 = zero_array

            # 
            y_windows = rolling_window(y,9)
            t_val = np.array(list(range(len(y_windows[0]))))
            lr = LinearRegression()
            lr.fit(t_val.reshape(-1, 1), y_windows.T)
            slopes = lr.coef_
            slopes = np.concatenate([slopes.reshape(-1, 1),np.ones(len(t_val)-1).reshape(-1, 1)])

            last_values = y_windows[:, -1]
            last_values = np.concatenate([last_values.reshape(-1, 1),np.ones(len(t_val)-1).reshape(-1, 1)])
            endpoint = y[-1]
            slope_draft = endpoint - last_values
            signs = np.array(np.where(np.sign(slope_draft*slopes)[0] == 1)[0], dtype = int)
            zero_array = np.zeros(len(t))
            w_slope = apply_normalized_signal(norm_func,signs,zero_array)

            weights_combined = (w_slope+w0+w1+w2+w3+w4+w5+w6+w6_1-w7)*decay_func
            weights.append(weights_combined)

            tr_poi,_ =find_extrema(weights_combined,0.5,"maxima")
            points_of_interest.append(tr_poi)
            rt_points.append(int(np.where(weights_combined == np.max(weights_combined))[0][0]))
        else:
            weights = np.zeros(30)*np.nan
            points_of_interest.append(np.array([np.nan]))
            rt_points.append(np.nan)
    
    rt_points = np.array(rt_points)

    return rt_points


def butter_filter(data, fs = 60, order = 1,cutoff = 3): 
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def find_extrema_rt(trajectory, importance = 1, ext_type = "both"):
    dist = 0.001
    if dist < 1:
        dist = 0.001

    if ext_type == "both":
        tr_maxima, tr_max_properties = find_peaks(trajectory, prominence = importance)
        tr_minima, tr_min_properties  = find_peaks(-trajectory, prominence = importance)
        extrema = np.concatenate([tr_maxima, tr_minima])
        extrema_properties = np.concatenate([tr_max_properties['prominences'], tr_min_properties['prominences']])
    elif ext_type == "minima":
        extrema, ext_prop = find_peaks(-trajectory, prominence = importance)
        if len(extrema) > 0:
            extrema_properties = ext_prop['prominences']
        else:
            extrema_properties = []
    elif ext_type == "maxima":
        extrema, ext_prop = find_peaks(trajectory, prominence = importance)
        if len(extrema) > 0:
            extrema_properties = ext_prop['prominences']
        else:
            extrema_properties = []

    return extrema, extrema_properties

def sliding_r2(t,y,winsize, norm_func):
    y_windows = rolling_window(y,winsize)
    t_val = np.linspace(0,winsize-1,winsize,dtype=int)
    
    lr = LinearRegression()
    lr.fit(t_val.reshape(-1, 1), y_windows.T)
    r2_scores = r2_score(y_windows.T, lr.predict(t_val.reshape(-1, 1)), multioutput='raw_values')
    
    detrend_y_windows =  signal.detrend(y_windows)
    low_variance = (np.var(y_windows,1) < 1) * (np.var(detrend_y_windows,1) < 1)
    r2_scores[np.where(low_variance == True)[0]] = 1
    r2_scores = np.concatenate([r2_scores,np.ones(winsize-1)])
    
    peaks_r2, _ = find_extrema_rt(-r2_scores,0.1,"maxima")
    peaks_r2 = peaks_r2 + int(winsize/3)
    #print(peaks_r2)
    weight_array = apply_normalized_signal(norm_func,peaks_r2,np.zeros(len(t)))
    
    return weight_array

def trial_reaction_time(sess_data, tr_id, multiple_windows):
    y = butter_filter(sess_data[tr_id,1])
    t = sess_data[tr_id,2]

    # linear time decay
    time_decay = np.linspace(1,-0.1,len(t))
    
    x = np.linspace(-5, 5, 11)
    norm_func = norm.pdf(x,np.mean(x),np.std(x))
    
    # Multiple Windows sliding R^2
    r2_weights = np.zeros(len(t))
    for i in range(len(multiple_windows)):
        one_w = sliding_r2(t,y,multiple_windows[i], norm_func)
        r2_weights = np.vstack((r2_weights, one_w))
    
    r2_weight_array = np.sum(r2_weights,0)

    combined_w = [r2_weight_array]*time_decay
    combined_w = np.array(combined_w).T

    r_time = np.where(combined_w == np.max(combined_w))[0][0]

    return r_time, combined_w

def reaction_time(sess_data, multiple_windows):
    r_time, combined_w = np.full(sess_data.shape[0], np.nan), []

    for i in range(len(sess_data)):
        if len(sess_data[i,1]) > (np.max(multiple_windows)*1.1):
            tr_r_time, tr_combined_w = trial_reaction_time(sess_data, i, multiple_windows)
            r_time[i] = tr_r_time
            combined_w.append(tr_combined_w)
        else:
            combined_w.append([np.nan]*len(sess_data[i,2]))

    maxes = []
    for i in range(len(combined_w)):
        maxes.append(np.max(combined_w[i]))
    maxes = np.array(maxes)
    maxes[maxes < 0.001] = np.nan
    maxes[maxes < np.nanpercentile(maxes,5)] = np.nan

    r_time = np.array(r_time, dtype = np.float32)
    
    r_time[np.isnan(maxes)] = np.nan

    
    return r_time, combined_w