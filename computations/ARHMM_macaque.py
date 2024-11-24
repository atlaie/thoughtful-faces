import optuna
import BehavUtils as butils
import time
import acme
import h5py
import numpy as np
import jax.random as jr
from dynamax.hidden_markov_model import LinearRegressionHMM
from itertools import count
from jax import vmap
from sklearn.metrics import r2_score


nSplits = 5
# How many trials will Optuna optimize hyperparameters for?
numTrials = 50

btscv = butils.BlockingTimeSeriesSplit(n_splits=nSplits)
internal_states = np.arange(9, 16)

scores_cv = np.full((internal_states.shape[0], numTrials), np.nan)
selected_concentration = np.full(internal_states.shape[0], np.nan)
selected_stickiness = np.full(internal_states.shape[0], np.nan)

dats_train = np.load('Predictors_emissions_macaque_concat_train_10pad.npz', allow_pickle = True)
dat_train_final = dats_train['predictors']
y_train_final = dats_train['emissions']
y_train_final[y_train_final>4] = 4

start2 = time.time()

# Clean up the cluster and set up a new cluster client
acme.cluster_cleanup()
client = acme.esi_cluster_setup(partition="8GBL", n_jobs=int(nSplits),
                                n_jobs_startup=2, timeout=60000, interactive_wait=1)
input_dim = 1
emission_dim = 1
shift = 1

def fit_ARHMM(numStates, input_dim, emission_dim, stickiness, concentration, y_train, y_test, shift, params, param_props):

    arhmm = LinearRegressionHMM(num_states = numStates, input_dim = input_dim, emission_dim = emission_dim, 
                                transition_matrix_stickiness=stickiness, transition_matrix_concentration = concentration)

    test_params, _ = arhmm.fit_sgd(params = params, props = param_props, emissions = y_train[:-shift], inputs = y_train[shift:], num_epochs = 5000)
    most_likely_states = arhmm.most_likely_states(params = test_params, emissions = y_test[:-shift], inputs = y_test[shift:])
    most_likely_states = np.array(most_likely_states)

    # Predict the emissions given the true states
    As = test_params.emissions.weights[most_likely_states]
    bs = test_params.emissions.biases[most_likely_states]
    y_pred = vmap(lambda x, A, b: A @ x + b)(y_test[shift:], As, bs)
    return r2_score(y_pred[shift:], y_test[shift:-shift])

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
        concentration = trial.suggest_float("transition_matrix_concentration", 0, 10)            
        stickiness = trial.suggest_float("transition_matrix_stickiness", 0, 100)              

        # Initialize keys for random number generation
        keys = map(jr.PRNGKey, count())    

        # Set emission dimension
        emission_dim = y_train_final.shape[1]

        # Initialize cross-validation and validation data lists and parameter lists

        y_cv_list = []
        y_val_list = []
        paramss = []
        param_propss = []

        # Loop through cross-validation splits
        for cv_index, val_index in btscv.split(y_train_final):

            # Split data into cross-validation and validation sets
            y_cv, y_val = y_train_final[cv_index], y_train_final[val_index]
            y_cv = y_cv.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)

            arhmm = LinearRegressionHMM(num_state, input_dim, emission_dim, transition_matrix_stickiness=stickiness, 
                                transition_matrix_concentration = concentration)

            params, param_props = arhmm.initialize(next(keys))
            paramss.append(params)
            param_propss.append(param_props)

            # Append cross-validation and validation data to lists
            y_cv_list.append(y_cv)
            y_val_list.append(y_val)

        # Evaluate sLDS models in parallel

        pmap = acme.ParallelMap(fit_ARHMM, num_state, input_dim, emission_dim, stickiness, concentration, y_cv_list, y_val_list, shift,
            paramss, param_propss, n_inputs=len(paramss), setup_timeout=1)
        with pmap as p:
            results = p.compute()
        
        score_folds = np.zeros(len(paramss))
        for ii, fname in enumerate(results):
            with h5py.File(fname, 'r') as f:
                score_folds[ii] = float(np.array(f['result_0']))


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
        np.save('ARHMM_CVscores_macaque_'+str(num_state)+'06112023.npy', scores_cv[num_state - internal_states[0], :])