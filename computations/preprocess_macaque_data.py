# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
import BehavUtils as butils
import acme
import reaction_time as reac
import dask.dataframe as dd
import os

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='RuntimeWarning: invalid value encountered in double_scalars')
np.seterr(over='ignore')

animal = 'macaque'

with open('RawData/Macaque/files_logs_macaque_moreSes.txt') as f:
    files_logs = [line.strip() for line in f.readlines()]
with open('RawData/Macaque/files_flashes_macaque_moreSes.txt') as f:
    files_flashes = [line.strip() for line in f.readlines()]
with open('RawData/Macaque/files_face_macaque_moreSes.txt') as f:
    files_dlc = [line.strip() for line in f.readlines()]
with open('RawData/Macaque/files_eye_macaque_moreSes.txt') as f:
    files_eye = [line.strip() for line in f.readlines()]
with open('RawData/Macaque/files_eyeNet_macaque_moreSes.txt') as f:
    files_eyeNet = [line.strip() for line in f.readlines()]

with open('RawData/Macaque/RelevantColumns_Larger.txt') as f:
    relevant_cols = [line.strip() for line in f.readlines()]
    
varNames = ['PupSize', 'EyeMov', 'Eye_x', 'Eye_y', 'rEar_x', 'rEar_y', 'lEar_x', 'lEar_y', 
            'rEyeBr_x', 'rEyeBr_y', 'lEyeBr_x', 'lEyeBr_y', 'nostrils_x', 'nostrils_y', 
            'uLip_x', 'uLip_y', 'lLip_x', 'lLip_y']

# Define parameters for cross-validation, repetitions, and data splitting
nWindows = 5
nRepetitions = 5
nSplits = 5
btscv = butils.BlockingTimeSeriesSplit(n_splits=nSplits)
shiftStim = -0.25
winSize = 0.25
internal_states = np.arange(2, 13)

# Define how many partitions to use for Dask loading in the data.
nPartitions = os.cpu_count() - 2

# How many trials will Optuna optimize hyperparameters for?
numTrials = 100
# Clean up the cluster and set up a new cluster client
acme.cluster_cleanup()
client = acme.esi_cluster_setup(partition="8GBXS", n_jobs=len(varNames),
                                n_jobs_startup=2, timeout=60000, interactive_wait=1)

subjects = []
for ii in range(len(files_logs)):
    subjects.append(files_logs[ii].split('/')[5])
uniq_subj = np.unique(subjects)

dat_train_list, dat_test_list, y_train_list, y_test_list = [], [], [], []
concentration_list, stickiness_list, scores_cv_list = [], [], []

for rr in range(len(files_logs)):

    r_earCols, l_earCols, r_eyeBrowCols, l_eyeBrowCols = [], [], [], []
    nostrilsCols, u_lipCols, l_lipCols = [], [], []

    flashes = np.load(files_flashes[rr], allow_pickle=True)
    # Parse event markers from log file
    evt, newSamp, nPoints, t_final, idx_start, idx_stim, _ = butils.readLog(files_logs[rr], animal)

    # Process session data, reaction times, and rescale reaction times
    sess_data = reac.sess_data_maker(files_logs[rr], animal, 3000)
    r_time, _ = reac.reaction_time(sess_data, [5,10,15])
    r_time[pd.isna(r_time)] = -1
    r_time = r_time.astype(np.float64)
    r_time = r_time / newSamp
    r_time[r_time > 4] = 4

    dat_eyeNet = pd.read_csv(files_eyeNet[rr])
    times_eyeNet = dat_eyeNet['time_cpu'].values
    evts_eyeNet = dat_eyeNet['data'].values
    random_numbers = evts_eyeNet[evts_eyeNet>30000]

    start = butils.search_sequence(evts_eyeNet, evt[evt>30000])[-1]+1
    end = butils.search_sequence(evts_eyeNet, evts_eyeNet[evts_eyeNet>30000][-3:])[0]-1
    t_tmp = np.nan * np.ones(nPoints)
    t_tmp[idx_start[:-1]] = flashes[:-1]
    frames_dlc = pd.Series(t_tmp).interpolate(method="linear").values
    frames_dlc[np.isnan(frames_dlc)] = 0
    frames_dlc = np.array(frames_dlc, dtype=int)

    t_stim = np.unique(t_final[idx_stim])

    dat_eye = pd.read_csv(files_eye[rr])
    t_tmp = dat_eye['time'].values
    fast_locs = np.searchsorted(t_tmp, [times_eyeNet[start], times_eyeNet[end]])
    t_eye = t_tmp[fast_locs[0]:fast_locs[-1]]

    df_train = dd.read_csv(files_dlc[rr], usecols = relevant_cols, header = 1, low_memory = False)

    df_train=df_train.compute()
    df_train.drop(0, inplace=True)
    dask_dat_dlc = dd.from_pandas(df_train, npartitions=nRepetitions)
    for col in dask_dat_dlc.columns:
        dask_dat_dlc[col] = dd.to_numeric(dask_dat_dlc[col])

    for col in dask_dat_dlc.columns:
        if 'RightEar' in col:
            r_earCols.append(col)
        elif 'LeftEar' in col:
            l_earCols.append(col)
        elif 'RightBrow' in col:
            r_eyeBrowCols.append(col)
        elif 'LeftBrow' in col:
            l_eyeBrowCols.append(col) 

        elif 'Nostrils' in col:
            nostrilsCols.append(col)
        elif 'UpperLip' in col:
            u_lipCols.append(col)
        elif 'LowerLip' in col:
            l_lipCols.append(col)

    dat_dlc = dask_dat_dlc.compute()
    
    rEar_x_fin, rEar_y_fin, _ = butils.dlcCalcs(dat_dlc[r_earCols], nPoints = frames_dlc.shape[0], doSize = 0)
    lEar_x_fin, lEar_y_fin, _  = butils.dlcCalcs(dat_dlc[l_earCols], nPoints = frames_dlc.shape[0], doSize = 0)
    rEyeBr_x_fin, rEyeBr_y_fin, _  = butils.dlcCalcs(dat_dlc[r_eyeBrowCols], nPoints = frames_dlc.shape[0], doSize = 0)
    lEyeBr_x_fin, lEyeBr_y_fin, _  = butils.dlcCalcs(dat_dlc[l_eyeBrowCols], nPoints = frames_dlc.shape[0], doSize = 0)

    eye_x, eye_y, pupSize = butils.eyeCalcs(files_eye[rr], t_eye.shape[0])

    nostrils_x_fin, nostrils_y_fin, _ = butils.dlcCalcs(dat_dlc[nostrilsCols], nPoints = frames_dlc.shape[0], doSize = 0)
    uLip_x_fin, uLip_y_fin, _  = butils.dlcCalcs(dat_dlc[u_lipCols], nPoints = frames_dlc.shape[0], doSize = 0)
    lLip_x_fin, lLip_y_fin, _  = butils.dlcCalcs(dat_dlc[l_lipCols], nPoints = frames_dlc.shape[0], doSize = 0)

    predVar = [eye_x, eye_y, rEar_x_fin, rEar_y_fin, lEar_x_fin, lEar_y_fin, rEyeBr_x_fin, rEyeBr_y_fin, lEyeBr_x_fin, lEyeBr_y_fin,
              nostrils_x_fin, nostrils_y_fin, uLip_x_fin, uLip_y_fin, lLip_x_fin, lLip_y_fin]
    predictors = butils.preprocess_data(predVar,pupSize,eye_x, eye_y,t_final,idx_start,t_stim,animal,shiftStim,winSize,t_eye)

    if predictors.shape[0] > r_time.shape[0]:
        predictors = predictors[:-1,:]
    elif predictors.shape[0] < r_time.shape[0]:
        r_time = r_time[:-1]

    dat_train, dat_test, y_train, y_test = butils.split_impute_scale(predictors, r_time, trainSize = 0.8, shift = 0)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)


    dat_train_list.append(dat_train)
    dat_test_list.append(dat_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    filename = f'RawData/Macaque/Predictors_emissions_macaque_test_session{rr}.npz'
    np.savez(filename, predictors=dat_test, emissions=y_test)

dat_train_final_macaque = butils.pad_concatenate(dat_train_list,doEmissions = False, numPad = 10)
y_train_final_macaque = butils.pad_concatenate(y_train_list,doEmissions = True, numPad = 10)