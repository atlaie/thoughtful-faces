import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import BehavUtils as butils
import acme
import glob
import time
import reaction_time as reac
import dask.dataframe as dd
import optuna
import pipeline_MSLR as pipln
import os
import argparse
from sklearn.metrics import r2_score

sns.set_style('white')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='RuntimeWarning: invalid value encountered in double_scalars')
np.seterr(over='ignore')

animal = 'macaque'

with open('RawData/Macaque/files_logs_macaque_moreSes.txt') as f:
    files_logs = [line.strip() for line in f.readlines()]
    
# Define parameters for cross-validation, repetitions, and data splitting
nWindows = 5
nRepetitions = 5
nSplits = 5
btscv = butils.BlockingTimeSeriesSplit(n_splits=nSplits)
shiftStim = -0.25
winSize = 0.25
internal_states = np.arange(8, 16)

# How many trials will Optuna optimize hyperparameters for?
numTrials = 100

# Create the parser
parser = argparse.ArgumentParser(description='Process some arguments.')
# Add the arguments
parser.add_argument('--doCV', type=str, required=True,
                    help='an required argument to do CV, input should be "True" or "False"')
parser.add_argument('--filename', type=str, required=False, 
                    help='a required argument for filename')
parser.add_argument('--date', type=str, required=True, 
                    help='a required argument for the current date')
# Parse the arguments
args = parser.parse_args()

# Perform a check for doCV value
if args.doCV.lower() not in ['true', 'false']:
    raise ValueError('Invalid value for doCV. Please enter "True" or "False".')
else:
    args.doCV = args.doCV.lower() == 'true'

dats_train = np.load('Predictors_emissions_macaque_concat_train.npz', allow_pickle = True)
dat_train_final = dats_train['predictors']

varNames_macaque = np.array(['PupSize', 'EyeMov', 'Eye_x', 'Eye_y', 'rEar_x', 'rEar_y', 'lEar_x', 'lEar_y', 
            'rEyeBr_x', 'rEyeBr_y', 'lEyeBr_x', 'lEyeBr_y', 'nostrils_x', 'nostrils_y', 
            'uLip_x', 'uLip_y', 'lLip_x', 'lLip_y'])

dat_train_final = dat_train_final[:,np.where(varNames_macaque!='lEyeBr_y')[0]]

# # No pupil:
# dat_train_final = dat_train_final[:,1:]

# # No pupil and no Eye Mov (after the correlation analysis):
dat_train_final = dat_train_final[:,2:]

# # Only pupil:
# dat_train_final = dat_train_final[:,0].reshape(-1,1)
## No eye quantities:
# dat_train_final = dat_train_final[:,4:]
# No ear quantities:
# dat_train_final = dat_train_final[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [4,5,6,7])]
# # No eyebrows quantities:
# dat_train_final = dat_train_final[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [8,9,10])]
# # No nostrils quantities:
# dat_train_final = dat_train_final[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [11,12])]
# # No lips quantities:
dat_train_final = dat_train_final[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [13,14,15,16])]

y_train_final = dats_train['emissions']
y_train_final[y_train_final>4] = 4

start1 = time.time()
if args.doCV:
    # Clean up the cluster and set up a new cluster client
    acme.cluster_cleanup()
    client = acme.esi_cluster_setup(partition="32GBS", n_jobs=int(nSplits),
                                    n_jobs_startup=2, timeout=60000, interactive_wait=1)

    scores_cv = np.full((internal_states.shape[0], numTrials), np.nan)
    selected_concentration = np.full(internal_states.shape[0], np.nan)
    selected_stickiness = np.full(internal_states.shape[0], np.nan)

    start2 = time.time()

    for num_state in internal_states:
        start3 = time.time()
        def objective(trial):
            """Objective function to be optimized by Optuna. 

            Hyperparameters chosen to be optimized: 

            concentration: how dense the transition matrix is.
            stickiness: how self-biased the states are.
            To read more about the last one: https://arxiv.org/pdf/0905.2592.pdf

            Inputs:
                - trial (optuna.trial._trial.Trial): Optuna trial
            Returns:
                - RMSE(torch.Tensor): The test RMSE. Parameter to be minimized.
            """

            # Define range of values to be tested for the model hyperparameters.
            concentration = trial.suggest_float("transition_matrix_concentration", 0, 100)            
            stickiness = trial.suggest_float("transition_matrix_stickiness", 0, 100)              

            siz = int(0.8*y_train_final.shape[0])
            macaque_mslr_instance = pipln.model_MSLR_concat(y_train_final[:siz,:],dat_train_final[:siz,:],y_train_final[siz:,:],dat_train_final[siz:,:],
                                                          num_state,concentration,stickiness,btscv,nRepetitions)
            score_folds = macaque_mslr_instance.train_mslr()

            # Init tracking experiment hyper-parameters, trial id are stored.
            config = dict(trial.params)
            config["trial.number"] = trial.number

            return np.nanmedian(score_folds)


        if __name__ == '__main__':

            # --- Parameters ----------------------------------------------------------

            number_of_trials = numTrials                  # Number of Optuna trials

            # Create an Optuna study to minimize the RMSE, using CV.
            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(direction = "maximize", sampler = sampler)
            study.optimize(objective, n_trials = number_of_trials, gc_after_trial=True)

            # Save results to csv file
            df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
            df = df.loc[df['state'] == 'COMPLETE']          # Keep only results that did not prune
            df = df.drop('state', axis=1)                   # Exclude state column
            df = df.sort_values('value')                    # Sort based on performance

            # Find the most important hyperparameters
            most_important_parameters = optuna.importance.get_param_importances(study, target=None)

            scores_cv[num_state - internal_states[0], :] = df['value'].values
            selected_concentration[num_state - internal_states[0]], selected_stickiness[num_state - internal_states[0]] = list(study.best_trial.params.values())

            print('--'*20+'States '+str(num_state)+' out of '+str(np.max(internal_states))+' finished. It took '+str(np.round(time.time() - start3, 3))+' seconds to finish'+'--'*20)
            
            np.savez("Results/Macaque/Results_CV_sLDS_Optuna_macaque_"+str(numTrials)+"trials_"+str(num_state)+"_states_"+args.date+"_RT_noPupilnoEyeMovModel_AllSubjects_R2score_CVOnly.npz", 
                     scores = scores_cv[num_state - internal_states[0], :], concentration = selected_concentration[num_state - internal_states[0]],
                     stickiness = selected_stickiness[num_state - internal_states[0]])
            np.savez("Results/Macaque/Results_ParamImportance_sLDS_Optuna_macaque_"+str(numTrials)+"trials_"+str(num_state)+"_states_"+args.date+"_RT_noPupilnoEyeMovModel_AllSubjects_R2score_CVOnly.npz",    dict(most_important_parameters.items()))
            
else:
    # Clean up the cluster and set up a new cluster client
    acme.cluster_cleanup()
    client = acme.esi_cluster_setup(partition="8GBXS", n_jobs=int(nSplits),
                                    n_jobs_startup=2, timeout=60000, interactive_wait=1)

    dat = np.load(args.filename, allow_pickle = True)
    selected_states = butils.extract_states([args.filename])[0]
    selected_concentration = dat['concentration']
    selected_stickiness = dat['stickiness']
    
    for rr in range(len(files_logs)):
        start2 = time.time()
        filename = f'RawData/Macaque/Predictors_emissions_macaque_test_session{rr}.npz'
        dats = np.load(filename, allow_pickle = True)
        dat_test = dats['predictors']
        dat_test = dat_test[:,np.where(varNames_macaque!='lEyeBr_y')[0]]
        # # No pupil:
        # dat_test = dat_test[:,1:]

        # # No pupil and no Eye Mov (after the correlation analysis):
        dat_test = dat_test[:,2:]

        # # Only pupil:
        # dat_test = dat_test[:,0].reshape(-1,1)
        ## No eye quantities:
        # dat_test = dat_test[:,4:]
        # No ear quantities:
        # dat_test = dat_test[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [4,5,6,7])]
        # # No eyebrows quantities:
        # dat_test = dat_test[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [8,9,10])]
        # # No nostrils quantities:
        # dat_test = dat_test[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [11,12])]
        # # No lips quantities:
        dat_test = dat_test[:,np.setdiff1d(np.arange(len(varNames_macaque)-1), [13,14,15,16])]


        y_test = dats['emissions']
        y_test[y_test>4] = 4

        macaque_mslr_instance_test = pipln.model_MSLR_concat(y_train_final,dat_train_final,y_test,dat_test,
                                                           selected_states, selected_concentration, selected_stickiness,btscv,nRepetitions)
        test_mse, y_pred, y_test, dat_test, filtered_state_probabilities, most_likely_states, params = macaque_mslr_instance_test.test_mslr()


        results = {
                'predictions': y_pred,
                'X_test': dat_test,
                'y_test': y_test,
                'numStates': selected_states,
                'stickiness': selected_stickiness,
                'concentration': selected_concentration,
                'predicted_states': most_likely_states,
                'predicted_statesProbs': filtered_state_probabilities,
                'transitionMatrices': params.transitions.transition_matrix,
                'weights': params.emissions.weights,
                'biases': params.emissions.biases
            }

        np.savez("Results/Macaque/Results_Test_sLDS_Optuna_macaque_"+str(numTrials)+"trials_"+args.date+"_RT_noPupilnoEyeMovModel_AllSubjects_session"+str(rr), results)
        #print(r2_score(y_test, y_pred))

        print('--'*20+'Session number '+str(rr+1)+' out of '+str(len(files_logs))+' finished. It took '+str(np.round(time.time() - start2, 3))+' seconds to finish'+'--'*20)
    print('--'*20+'It took '+str(np.round(time.time() - start1, 3))+' seconds for all subjects to finish'+'--'*20)