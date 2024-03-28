import numpy as np
import pandas as pd
from parse_logfile import TextLog
from scipy.signal import butter, sosfiltfilt
import warnings
import acme
import h5py
import re

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='RuntimeWarning: overflow encountered in exp')

#**** FUNCTIONS ****#

def flatten_list(l):
    for i in l:
        if isinstance(i,list):
            yield from flatten_list(i)
        else:
            yield i

def get_matching_paths_ordered(list1, list2):
    pattern = re.compile(r'/[^/]+/[^/]+/[^/]+/([^/]+)/([^/]+)/RWD/([^/]+)/')

    # Store elements of interest and their corresponding paths in dictionaries.
    elements_paths_dict1 = {pattern.search(s).groups(): s for s in list1 if pattern.search(s)}
    elements_paths_dict2 = {pattern.search(s).groups(): s for s in list2 if pattern.search(s)}

    # Find the common keys (i.e., elements of interest) between the two dictionaries.
    common_keys = [key for key in elements_paths_dict1.keys() if key in elements_paths_dict2.keys()]

    # Get the paths corresponding to the common keys in the order they appear in the original lists.
    matching_paths_list1 = [elements_paths_dict1[key] for key in common_keys]
    matching_paths_list2 = [elements_paths_dict2[key] for key in common_keys]

    return matching_paths_list1, matching_paths_list2

def extract_states(file_list):
    state_nums = []
    for filename in file_list:
        match = re.search(r'_([0-9]+)_states_', filename)
        if match:
            state_nums.append(int(match.group(1)))
    return state_nums

def find_consecutive_indices(array, lower_threshold, upper_threshold, consecutive_points):
    # Condition check
    condition = np.logical_and(array >= lower_threshold, array <= upper_threshold)
    
    # Create a filter for the number of consecutive points
    consecutive_filter = np.ones((consecutive_points,))
    
    # Use convolution to find the regions where the condition is met over consecutive points
    # Use 'full' mode to include shorter sequences at the beginning and end of the array
    condition_consecutive = np.convolve(condition, consecutive_filter, mode='full')
    
    # Initialize the result array with False values
    result = np.full(array.shape, False)
    
    # Set the elements at the indices where the condition is met to True
    for i, value in enumerate(condition_consecutive):
        if value >= consecutive_points:
            result[max(0, i - consecutive_points + 1): i + 1] = True
    return ~result

def readLog(logFile, animal):
    # Parse event markers from log file
    with TextLog(logFile) as log:
        evt, _, _, true_ts = log.parse_eventmarkers()
        log.make_id_struct()

        # Find the AnimalCharacter name
        for name in log.all_ids['name']:
            if name.startswith('AnimalCharacter'):
                AC = name
                print(AC)
                break

    # Process screen and log times
    screenTimes = log.log_to_screen_times[:, 1]
    # logTimes = log.log_to_screen_times[:, 0]
    newSamp = (1 / np.nanmedian(np.diff(screenTimes)))
    start = screenTimes[0]
    end = screenTimes[-1]
    nPoints = int(np.abs(start - end) * newSamp)
    t_final = np.linspace(start, end, nPoints)

    # Find indices for various events in the master time array
    idx_start = np.searchsorted(t_final, true_ts[evt == 3000])
    if animal == 'macaque':
        idx_stim = np.searchsorted(t_final, true_ts[(evt == 3011) | (evt == 3021)])
    elif animal == 'mouse':
        idx_stim = np.searchsorted(t_final, true_ts[evt == 3001])
    
    return evt, newSamp, nPoints, t_final, idx_start, idx_stim, true_ts
    

def align_Flashes_Video(flashes, screenTimes, true_ts, evt):
    
    newSamp = (1/np.nanmedian(np.diff(screenTimes)))
    start = screenTimes[0]
    end = screenTimes[-1]
    nPoints = int(np.abs(start - end)*newSamp)
    t_final = np.linspace(start,  end, nPoints)

    idx_start = np.searchsorted(t_final, true_ts[evt == 3000])

    t_tmp = np.nan*np.ones(nPoints)
    t_tmp[idx_start[:-1]] = flashes[:-1]
    frames_dlc = pd.Series(t_tmp).interpolate(method='linear').values
    frames_dlc[np.isnan(frames_dlc)] = 0
    frames_dlc = np.array(frames_dlc, dtype = int)
    
    return t_final, frames_dlc    
    

def align_Flashes_Eye_Video(flashes, screenTimes, true_ts, evt, files_eyeNet, files_eye, doMacaque = False):
    
    newSamp = (1/np.nanmedian(np.diff(screenTimes)))
    start = screenTimes[0]
    end = screenTimes[-1]
    nPoints = int(np.abs(start - end)*newSamp)
    t_final = np.linspace(start,  end, nPoints)

    idx_start = np.searchsorted(t_final, true_ts[evt == 3000])

    t_tmp = np.nan*np.ones(nPoints)
    t_tmp[idx_start[:-1]] = flashes[:-1]
    frames_dlc = pd.Series(t_tmp).interpolate(method='linear').values
    frames_dlc[np.isnan(frames_dlc)] = 0
    frames_dlc = np.array(frames_dlc, dtype = int)
    if doMacaque:
        dat = pd.read_csv(files_eyeNet)
        times_eye = dat['time_cpu'].values
        evts_eye = dat['data'].values

        start = search_sequence(evts_eye, evt[evt>30000])[-1]+1
        end = search_sequence(evts_eye, evts_eye[evts_eye>30000][-3:])[0]-1

        dat_eye = pd.read_csv(files_eye)
        t_tmp = dat_eye['time'].values
        fast_locs = np.searchsorted(t_tmp, [times_eye[start], times_eye[end]])
        t_eye = t_tmp[fast_locs[0]:fast_locs[-1]]


        return t_final, frames_dlc, t_eye
    
    else:
        return t_final, frames_dlc

def fromTimetoTrials(xTime, yTime, t_start, t_of_interest, t_vec, shift1, shift2):
    # Initialize the trialAggregate array with the same length as t_start
    trialAggregate = np.full_like(t_start, np.nan)
    
    # Initialize the index for t_of_interest
    toi_idx = 0

    # Iterate over the trial start times, excluding the last one
    for ii, trial_start in enumerate(t_start[:-1]):
        # Iterate over t_of_interest values within the current trial
        while toi_idx < len(t_of_interest) and t_of_interest[toi_idx] < t_start[ii + 1]:
            # Calculate the start and end indices for the median calculation
            st = np.searchsorted(t_vec, t_of_interest[toi_idx] + shift1)
            end = np.searchsorted(t_vec, t_of_interest[toi_idx] + shift2)

            # Compute the median distance and store it in the trialAggregate array
            trialAggregate[ii] = np.nanmedian(np.sqrt(np.power(xTime[st:end], 2) + np.power(yTime[st:end], 2)))

            # Increment the t_of_interest index
            toi_idx += 1

    # Handle the last trial separately
    while toi_idx < len(t_of_interest):
        # Calculate the start and end indices for the median calculation
        st = np.searchsorted(t_vec, t_of_interest[toi_idx] + shift1)
        end = np.searchsorted(t_vec, t_of_interest[toi_idx] + shift2)

        # Compute the median distance and store it in the trialAggregate array
        trialAggregate[-1] = np.nanmedian(np.sqrt(np.power(xTime[st:end], 2) + np.power(yTime[st:end], 2)))

        # Increment the t_of_interest index
        toi_idx += 1

    return trialAggregate


def medianPos(mean_vec, t_start, t_vec, t_of_interest, shiftStim1, shiftStim2):
    # Initialize the medianPos array with the same length as t_start
    medianPos = np.full_like(t_start, np.nan)
    
    # Initialize the index for t_of_interest
    toi_idx = 0

    # Iterate over the trial start times, excluding the last one
    for ii, trial_start in enumerate(t_start[:-1]):
        # Iterate over t_of_interest values within the current trial
        while toi_idx < len(t_of_interest) and t_of_interest[toi_idx] < t_start[ii + 1]:
            # Calculate the start and end indices for the median calculation
            st = np.searchsorted(t_vec, t_of_interest[toi_idx] + shiftStim1)
            end = np.searchsorted(t_vec, t_of_interest[toi_idx] + shiftStim2)

            # Compute the median and store it in the medianPos array
            medianPos[ii] = np.nanmedian(mean_vec[st:end])

            # Increment the t_of_interest index
            toi_idx += 1

    # Handle the last trial separately
    while toi_idx < len(t_of_interest):
        # Calculate the start and end indices for the median calculation
        st = np.searchsorted(t_vec, t_of_interest[toi_idx] + shiftStim1)
        end = np.searchsorted(t_vec, t_of_interest[toi_idx] + shiftStim2)

        # Compute the median and store it in the medianPos array
        medianPos[-1] = np.nanmedian(mean_vec[st:end])

        # Increment the t_of_interest index
        toi_idx += 1

    return medianPos


def preprocess_data(predVar,pupSize,eye_x, eye_y, t_final,idx_start,t_stim,animal,shiftStim,winSize, t_eye = None):

    pmap = acme.ParallelMap(medianPos, predVar, t_final[idx_start], t_final, t_stim, 
                            shiftStim, shiftStim + winSize, n_inputs = len(predVar), setup_timeout = 1)
    with pmap as p:
        results = p.compute()

    data2 = np.zeros((idx_start.shape[0], len(predVar)))

    for ii, fname in enumerate(results):
        with h5py.File(fname, 'r') as f:
            data2[:,ii] = np.array(f['result_0'])

    if animal == 'mouse':
        pupSize = fromTimetoTrials(pupSize, np.zeros_like(pupSize), t_final[idx_start], t_stim, t_final, shiftStim, shiftStim + winSize)
        eyeMov = fromTimetoTrials(predVar[0], predVar[1], t_final[idx_start], t_stim, t_final, shiftStim, shiftStim + winSize)
        noseMov = fromTimetoTrials(predVar[2], predVar[3], t_final[idx_start], t_stim, t_final, shiftStim, shiftStim + winSize)
        data = np.column_stack([pupSize, eyeMov, noseMov, data2])

    elif animal == 'macaque':
        pupSize = fromTimetoTrials(pupSize, np.zeros_like(pupSize), t_final[idx_start], t_stim, t_eye - t_eye[0], shiftStim, shiftStim + winSize)
        eyeMov = fromTimetoTrials(eye_x, eye_y, t_final[idx_start], t_stim, t_eye - t_eye[0], shiftStim, shiftStim + winSize)
        data = np.column_stack([pupSize, eyeMov, data2])
    else:
        print('WRONG ANIMAL TYPE!')
        data = np.nan*np.ones_like(data2)

    notNan = np.abs(np.sum(np.isnan(data), axis = 0) - data.shape[0]) == 0
    predictors = data.copy()
    predictors[:,notNan] = 0

    return predictors


def decTimetoTrials(t_vec, t_start, t_hit, t_miss, t_err_stim, t_err_move):

    idx_decision = np.zeros(len(t_start))
    for i in range(len(t_start)):
        if i < len(t_start)-1:
            trial_idx = np.where(np.logical_and(t_vec>t_start[i],t_vec<t_start[i+1]))[0]
            if len(trial_idx)>1:
                idx_correct = np.where(np.logical_and(t_hit>t_start[i],t_hit<t_start[i+1]))[0]
                idx_incorrect = np.where(np.logical_and(t_miss>t_start[i],t_miss<t_start[i+1]))[0]
                idx_err_stim = np.where(np.logical_and(t_err_stim>t_start[i],t_err_stim<t_start[i+1]))[0]
                idx_err_move = np.where(np.logical_and(t_err_move>t_start[i],t_err_move<t_start[i+1]))[0]
                if len(idx_correct):
                    idx_decision[i] = 1
                elif len(idx_incorrect):
                    idx_decision[i] = 2
                elif len(idx_err_stim):
                    idx_decision[i] = 3
                elif len(idx_err_move):
                    idx_decision[i] = 4
        else:
            trial_idx = np.where(t_vec>t_start[i])[0]
            if len(trial_idx)>1:
                idx_correct = np.where(t_hit>t_start[i])[0]
                idx_incorrect = np.where(t_miss>t_start[i])[0]
                idx_err_stim = np.where(t_err_stim>t_start[i])[0]
                idx_err_move = np.where(t_err_move>t_start[i])[0]
                if len(idx_correct):
                    idx_decision[i] = 1
                elif len(idx_incorrect):
                    idx_decision[i] = 2
                elif len(idx_err_stim):
                    idx_decision[i] = 3
                elif len(idx_err_move):
                    idx_decision[i] = 4
    # decision = idx_decision[(idx_decision==1) | (idx_decision==2) | (idx_decision==3)| (idx_decision==4)]
    return idx_decision

def search_sequence(arr,seq):
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

def reSample(tSeries, newLen):
    original = np.array(tSeries, dtype=float)
    index_arr = np.linspace(0, len(original)-1, num=newLen, dtype=float)
    index_floor = np.array(index_arr, dtype=int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem

    return interp

def getDistances(data):
    # Extract X, Y, and likelihood columns for each feature and store in a list
    num_features = data.shape[1] // 3
    ptlist = [data.iloc[:, i*3:i*3+3].values for i in range(num_features)]

    # Remove outliers and interpolate missing values for X and Y coordinates
    for elem in ptlist:
        filters = np.logical_or.reduce([
            elem[:, 0] > np.nanpercentile(elem[:, 0], 99.9),
            elem[:, 0] < np.nanpercentile(elem[:, 0], 0.1),
            elem[:, 1] > np.nanpercentile(elem[:, 1], 99.9),
            elem[:, 1] < np.nanpercentile(elem[:, 1], 0.1),
            elem[:, 2] < 0.95
        ])
        elem[:, :2][filters] = np.nan
        elem[:, :2] = pd.DataFrame(elem[:, :2]).interpolate('linear').values

    # Calculate pairwise distances between features and store in a 3D array
    ptlist_stacked = np.stack(ptlist)[:, :, :2]
    dists = np.sqrt(np.sum(np.square(ptlist_stacked[:, np.newaxis, :, :] - ptlist_stacked[np.newaxis, :, :, :]), axis=-1))

    # Calculate mean locations per feature
    meanLoc = np.nanmean(np.concatenate([elem[:, :2] for elem in ptlist], axis=1).reshape(-1, num_features, 2), axis=1)

    return dists, meanLoc

def dlcCalcs(data, nPoints, doSize):
    if doSize:
        dists_tmp_tmp, meanLocs = getDistances(data)
        dists_tmp = np.nanmedian(np.nanmedian(dists_tmp_tmp, axis = 0), axis = 0)
        # print(dists_tmp.shape)
        dists = reSample(dists_tmp, nPoints)
    else:
        _, meanLocs = getDistances(data)
        dists = []

    mean_x_fin = reSample(meanLocs[:,0], nPoints)
    mean_y_fin = reSample(meanLocs[:,1], nPoints)

    return mean_x_fin, mean_y_fin, dists


def daskLoadCSV(filename, relevant_cols, nPartitions):
    import dask.dataframe as dd

    df_train = dd.read_csv(filename, header = 1, usecols = relevant_cols,low_memory = False)
    df_train=df_train.compute()
    df_train.drop(0, inplace=True)

    dask_dat = dd.from_pandas(df_train, npartitions=nPartitions)
    for col in dask_dat.columns:
        dask_dat[col] = dd.to_numeric(dask_dat[col])

    return dask_dat.compute()


def eyeCalcs(file, nPoints):
    Data_pupil = pd.read_csv(file, low_memory=False)
    mean_x = reSample(Data_pupil['x'].values, nPoints)
    mean_x = np.concatenate([np.nanmedian(mean_x)*np.ones(1), mean_x])

    mean_y = reSample(Data_pupil['y'].values, nPoints)
    mean_y = np.concatenate([np.nanmedian(mean_y)*np.ones(1), mean_y])

    mean_pupil = reSample(Data_pupil['pupil'].values, nPoints)
    mean_pupil = np.concatenate([np.nanmedian(mean_pupil)*np.ones(1), mean_pupil])

    mean_t = reSample(Data_pupil['time'].values, nPoints)
    mean_x = pd.DataFrame(mean_x).interpolate('linear').values.reshape(mean_x.shape[0],)
    mean_y = pd.DataFrame(mean_y).interpolate('linear').values.reshape(mean_y.shape[0],)
    mean_pupil = pd.DataFrame(mean_pupil).interpolate('linear').values.reshape(mean_pupil.shape[0],)
    print(np.sum(np.isnan(mean_pupil)))
    samplingTS = 1/np.nanmedian(np.diff(mean_t))
    slowfreq = samplingTS/60
    bu_slow = butter(N = 4, Wn = slowfreq, btype = 'low', output = 'sos', analog = False, fs = samplingTS)    
    pupSize_slow = sosfiltfilt(bu_slow, mean_pupil)

    return mean_x, mean_y, pupSize_slow

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def split_impute_scale(predictors, outcomes, trainSize = 0.7, shift = 2, randomState = 1121218):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    estimator = BayesianRidge()

    size = int(len(outcomes) * trainSize)
    if shift:
        out2 = np.concatenate([np.nan*np.ones(shift), outcomes])
        dat_train, dat_test, y_train, y_test = predictors[shift:size+shift,:], predictors[size+shift:,:], out2[shift:size+shift], out2[size+shift:-shift]
    else:
        dat_train, dat_test, y_train, y_test = predictors[:size,:], predictors[size:,:], outcomes[:size], outcomes[size:]

    imp = IterativeImputer(estimator=estimator, random_state=randomState, max_iter = 100, sample_posterior = True, skip_complete = True)
    dat_train = imp.fit_transform(dat_train)
    dat_test = imp.transform(dat_test)

    scaler = StandardScaler()
    dat_train = scaler.fit_transform(dat_train)
    dat_test = scaler.transform(dat_test)

    return dat_train, dat_test, y_train, y_test#, idx_test


def pad_concatenate(list_mats, emission_dim = 1, doEmissions = False, numPad = 50):
    newLists = []
    if doEmissions:
        for ii in range(len(list_mats)):
            newLists.append(np.concatenate([list_mats[ii], 0*np.ones((numPad,1)) + 1e-3*np.random.rand(numPad,1)], axis = 0))
    else:
        for ii in range(len(list_mats)):
            newLists.append(np.concatenate([list_mats[ii], 0*np.ones((numPad,list_mats[ii].shape[1])) + 1e-3*np.random.rand(numPad,list_mats[ii].shape[1])], axis = 0))
            
    return np.concatenate(newLists)

def finalModel_MSLR(dat_train, y_train, dat_test, y_test, numStates, emission_dim, stickiness, concentration, params, param_props):

    from jax import vmap
    from dynamax.hidden_markov_model import LinearRegressionHMM
    from itertools import count
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score

    # First set some global constants
    covariate_dim = dat_train.shape[1]

    # Initialize our MSLR.
    mslr = LinearRegressionHMM(numStates, covariate_dim, emission_dim,transition_matrix_stickiness=stickiness, 
                              transition_matrix_concentration = concentration)

    # To fit the model, give it a batch of emissions and a batch of corresponding inputs
    test_params, lps = mslr.fit_em(params, param_props, y_train, inputs=dat_train, num_iters = 50)
    if np.sum(np.isnan(lps)):
        numIters = np.where(~np.isnan(lps))[0][-1]
        # Initialize our MSLR.
        mslr = LinearRegressionHMM(numStates, covariate_dim, emission_dim,transition_matrix_stickiness=stickiness, 
                                transition_matrix_concentration = concentration)
        # To fit the model, give it a batch of emissions and a batch of corresponding inputs
        test_params, lps = mslr.fit_em(params, param_props, y_train, inputs=dat_train, num_iters = numIters)
    else:
        # Initialize our MSLR.
        mslr = LinearRegressionHMM(numStates, covariate_dim, emission_dim,transition_matrix_stickiness=stickiness, 
                                transition_matrix_concentration = concentration)
        # To fit the model, give it a batch of emissions and a batch of corresponding inputs
        test_params, lps = mslr.fit_sgd(params, param_props, y_train, inputs=dat_train, num_epochs = 5000)
    
    if len(lps) == 0:
        lps = np.array([0, 0])

    # Compute the most likely states
    most_likely_states = mslr.most_likely_states(test_params, y_test, inputs = dat_test)
    most_likely_states = np.array(most_likely_states)

    # Predict the emissions given the true states
    As = test_params.emissions.weights[most_likely_states]
    bs = test_params.emissions.biases[most_likely_states]
    y_pred= vmap(lambda x, A, b: A @ x + b)(dat_test, As, bs)
    
    if np.sum(np.isnan(y_pred)):
        y_pred = np.random.permutation(y_test)
    
    return r2_score(y_pred, y_test), lps[-1]