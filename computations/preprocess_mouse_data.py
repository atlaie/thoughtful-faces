# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
import BehavUtils as butils
import reaction_time as reac
import os
import acme

# Ignore warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='RuntimeWarning: invalid value encountered in double_scalars')
np.seterr(over='ignore')

# Define the animal
animal = 'mouse'

# Load log, flash, face, and eye files
with open('RawData/Mouse/files_logs_mouse_moreSes.txt') as f:
    files_logs = [line.strip() for line in f.readlines()]
with open('RawData/Mouse/files_flashes_mouse_moreSes.txt') as f:
    files_flashes = [line.strip() for line in f.readlines()]
with open('RawData/Mouse/files_face_mouse_moreSes_updated.txt') as f:
    files_face = [line.strip() for line in f.readlines()]
with open('RawData/Mouse/files_eye_mouse_moreSes_updated.txt') as f:
    files_eye = [line.strip() for line in f.readlines()]
    
# Define which column names to read from the face and eye files
with open('RawData/Mouse/cols_nose_updated.txt') as f:
    cols_nose = [line.strip() for line in f.readlines()]
with open('RawData/Mouse/cols_whiskers_updated.txt') as f:
    cols_whiskers = [line.strip() for line in f.readlines()]
with open('RawData/Mouse/cols_eye.txt') as f:
    cols_eye = [line.strip() for line in f.readlines()]

# Define parameters for cross-validation, repetitions, and data splitting
nWindows = 5
nRepetitions = 5
nSplits = 5
btscv = butils.BlockingTimeSeriesSplit(n_splits=nSplits)
shiftStim = -0.25
winSize = 0.25
internal_states = np.arange(2, 16)

# Define how many partitions to use for Dask loading in the data.
nPartitions = os.cpu_count() - 2

# How many trials will Optuna optimize hyperparameters for?
numTrials = 50

# Clean up the cluster and set up a new cluster client
acme.cluster_cleanup()
client = acme.esi_cluster_setup(partition="8GBXS", n_jobs=int(nSplits),
                                n_jobs_startup=2, timeout=60000, interactive_wait=1)

dat_train_list, dat_test_list, y_train_list, y_test_list = [], [], [], []
concentration_list, stickiness_list, scores_cv_list = [], [], []
for rr in range(len(files_logs)):
    # Load flashes data
    flashes = np.load(files_flashes[rr], allow_pickle=True)
    # Parse event markers from log file
    evt, newSamp, nPoints, t_final, idx_start, idx_stim, _ = butils.readLog(files_logs[rr], 'mouse')

    # Process session data, reaction times, and rescale reaction times
    sess_data = reac.sess_data_maker(files_logs[rr], animal, 3000)
    r_time, _ = reac.reaction_time(sess_data, [5,10,15])
    r_time[pd.isna(r_time)] = -1
    r_time = r_time.astype(np.float64)
    r_time = r_time / newSamp
    r_time[r_time > 4] = 4

    t_tmp = np.nan * np.ones(nPoints)
    t_tmp[idx_start[:-1]] = flashes[:-1]
    frames_dlc = pd.Series(t_tmp).interpolate(method="linear").values
    frames_dlc[np.isnan(frames_dlc)] = 0
    frames_dlc = np.array(frames_dlc, dtype=int)

    t_stim = np.unique(t_final[idx_stim])

    dat_face = butils.daskLoadCSV(files_face[rr], cols_nose, nPartitions=nPartitions)
    dat_whis = butils.daskLoadCSV(files_face[rr], cols_whiskers, nPartitions=nPartitions)
    dat_eye = butils.daskLoadCSV(files_eye[rr], cols_eye, nPartitions=nPartitions)
    
    
    nose_x_fin, nose_y_fin, _ = butils.dlcCalcs(dat_face, nPoints=frames_dlc.shape[0], doSize=0)
    eye_x_fin, eye_y_fin, pupSize_t = butils.dlcCalcs(dat_eye, nPoints=frames_dlc.shape[0], doSize=1)
    whisk_x_fin, whisk_y_fin, _ = butils.dlcCalcs(dat_whis, nPoints=frames_dlc.shape[0], doSize=0)

    predVar = [eye_x_fin, eye_y_fin, nose_x_fin, nose_y_fin, whisk_x_fin, whisk_y_fin]
    predictors = butils.preprocess_data(predVar,pupSize_t,eye_x_fin, eye_y_fin, t_final,idx_start,t_stim,animal,shiftStim,winSize)


    if predictors.shape[0] > r_time.shape[0]:
        predictors = predictors[:-1,:]
    elif predictors.shape[0] < r_time.shape[0]:
        r_time = r_time[:-1]
    
    size = int(len(r_time) * 0.8)
    dat_train, dat_test, y_train, y_test = butils.split_impute_scale(predictors, r_time, trainSize = 0.8, shift = 0)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    dat_train_list.append(dat_train)
    dat_test_list.append(dat_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    filename = f'RawData/Mouse/Predictors_emissions_mouse_newDLC_test_session{rr}_nonScaled.npz'
    np.savez(filename, predictors=dat_test, emissions=y_test)
    
dat_train_final = butils.pad_concatenate(dat_train_list,doEmissions = False, numPad = 20)
y_train_final = butils.pad_concatenate(y_train_list,doEmissions = True, numPad = 20)

np.savez('Predictors_emissions_mouse_newDLC_concat_train_20pad_nonScaled.npz', predictors = dat_train_final, emissions = y_train_final)