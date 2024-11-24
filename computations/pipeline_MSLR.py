# Import necessary libraries
import numpy as np
import BehavUtils as butils
import h5py
import acme
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import LinearRegressionHMM
from itertools import count
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from jax.config import config
config.update("jax_enable_x64", True)

class model_MSLR_concat:
    def __init__(self,emissions_train,predictors_train,emissions_test,predictors_test,num_state,concentration,stickiness,btscv,nRepetitions):
        self.emissions_train = emissions_train
        self.predictors_train = predictors_train
        self.emissions_test = emissions_test
        self.predictors_test = predictors_test
        self.num_state = num_state
        self.concentration = concentration
        self.stickiness = stickiness
        self.btscv = btscv
        self.nRepetitions = nRepetitions

    def train_mslr(self):
        
        # Initialize keys for random number generation
        keys = map(jr.PRNGKey, count())    

        # Set emission dimension
        emission_dim = self.emissions_train.shape[1]

        # Initialize cross-validation and validation data lists and parameter lists
        X_cv_list = []
        X_val_list = []
        y_cv_list = []
        y_val_list = []
        paramss = []
        param_propss = []

        # Loop through cross-validation splits
        for cv_index, val_index in self.btscv.split(self.predictors_train, self.emissions_train):

            # Split data into cross-validation and validation sets
            X_cv, X_val = self.predictors_train[cv_index, :], self.predictors_train[val_index, :]
            y_cv, y_val = self.emissions_train[cv_index], self.emissions_train[val_index]
            y_cv = y_cv.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)

            mslr = LinearRegressionHMM(self.num_state, X_cv.shape[1], emission_dim,transition_matrix_stickiness=self.stickiness, 
                              transition_matrix_concentration = self.concentration)
            
            params, param_props = mslr.initialize(next(keys))
            paramss.append(params)
            param_propss.append(param_props)
            scaler = MinMaxScaler()
            # Append cross-validation and validation data to lists
            X_cv_list.append(scaler.fit_transform(X_cv))
            X_val_list.append(scaler.transform(X_val))
            scaler = MinMaxScaler()
            y_cv_list.append(scaler.fit_transform(y_cv))
            y_val_list.append(scaler.transform(y_val))

        # Evaluate MSLR models in parallel
        pmap = acme.ParallelMap(butils.finalModel_MSLR, X_cv_list, y_cv_list, X_val_list, y_val_list, self.num_state, emission_dim, self.stickiness, self.concentration,
            paramss, param_propss, n_inputs=len(paramss), setup_timeout=1)
        with pmap as p:
            results = p.compute()
        
        score_folds = np.zeros(len(paramss))
        lps_folds = np.zeros(len(paramss))
        for ii, fname in enumerate(results):
            with h5py.File(fname, 'r') as f:
                if f['result_0'] is not None:
                    score_folds[ii] = np.float(np.array(f['result_0']))
                    lps_folds[ii] = np.float(np.array(f['result_1']))

        return score_folds#, lps_folds
    
    def test_mslr(self):

        # Initialize keys for random number generation
        keys = map(jr.PRNGKey, count())    

        # Set emission dimension
        emission_dim = self.emissions_train.shape[1]
        scaler = MinMaxScaler()
        # Append cross-validation and validation data to lists
        self.predictors_train = scaler.fit_transform(self.predictors_train)
        self.predictors_test = scaler.transform(self.predictors_test)
        scaler = MinMaxScaler()
        self.emissions_train = scaler.fit_transform(self.emissions_train)
        self.emissions_test = scaler.transform(self.emissions_test)

        # Initialize our MSLR.
        mslr = LinearRegressionHMM(self.num_state, self.predictors_train.shape[1], emission_dim,transition_matrix_stickiness=self.stickiness, 
                                transition_matrix_concentration = self.concentration)
        params, param_props = mslr.initialize(next(keys))
        # To fit the model, give it a batch of emissions and a batch of corresponding inputs
        test_params, lps = mslr.fit_em(params, param_props, self.emissions_train, inputs=self.predictors_train, num_iters = 50)
        if np.sum(np.isnan(lps)):
            numIters = np.where(~np.isnan(lps))[0][-1]
            # Initialize our MSLR.
            mslr = LinearRegressionHMM(self.num_state, self.predictors_train.shape[1], emission_dim,transition_matrix_stickiness=self.stickiness, 
                                    transition_matrix_concentration = self.concentration)
            # To fit the model, give it a batch of emissions and a batch of corresponding inputs
            test_params, lps = mslr.fit_em(self.params, param_props, self.emissions_train, inputs=self.predictors_train, num_iters = numIters)

        # Compute the most likely states
        most_likely_states = mslr.most_likely_states(test_params, self.emissions_test, inputs = self.predictors_test)
        most_likely_states = np.array(most_likely_states)
        filter_model = mslr.filter(test_params, self.emissions_test, inputs = self.predictors_test)
        filtered_state_probabilities = np.array(filter_model[1])

        # Predict the emissions_train given the true states
        As = test_params.emissions.weights[most_likely_states]
        bs = test_params.emissions.biases[most_likely_states]
        print(self.predictors_test.shape, As.shape, bs.shape)
        y_pred= vmap(lambda x, A, b: A @ x + b)(self.predictors_test, As, bs)
        
        test_score = metrics.r2_score(self.emissions_test, y_pred)
        
        return test_score, y_pred, self.emissions_test, self.predictors_test, filtered_state_probabilities, most_likely_states, test_params