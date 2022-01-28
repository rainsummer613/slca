import json
import logging
from matplotlib.image import imread
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import seaborn as sns
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
    
class SLCA():
    """
    The leaky competing accumulator class in numpy
    """
    def __init__(
            self, stimulus, params_range, dt_t=0.01, leak=0.2, competition=0, self_excit=0,
            w_input=1, w_cross=0, offset=0, noise_sd=0, threshold=5.0):
        """
        Initialize a leaky competing accumulator.
        
        Parameters
        ----------
        stimulus : np.ndarray
            the stimulus image
        params_range : dict
            range of parameters
        dt_t : float
            the dt / tao term, representing the time step size
        leak : float
            the leak term
        competition : float
            lateral inhibition across accumulators (i vs. j)
        self_excit : float
            self excitation of the accumulator (i vs. i)
        w_input : float
            input strengh of the feedforward weights
        w_cross : float
            cross talk of the feedforward weights
        offset : float
            the additive drift term of the LCA process
        noise_sd : float
            the sd of the noise term of the LCA process
        threshold : float
            the activation threshold  
        """
        self.stimulus = stimulus
        self.threshold = threshold
        self.h, self.w = stimulus.shape
        self.n_units = self.h * self.w
        self.dt_t = dt_t
        self.leak = leak
        self.competition = competition
        self.self_excit = self_excit
        self.w_input = w_input
        self.w_cross = w_cross
        self.offset = offset
        self.noise_sd = noise_sd
        # check params
        self._check_model_config(params_range)
        
    def _check_model_config(self, params_range):
        if params_range['dt_t'][0] > self.dt_t: 
            self.dt_t = params_range['dt_t'][0]
        elif self.dt_t > params_range['dt_t'][1]:
            self.dt_t = params_range['dt_t'][1]
            
        if params_range['leak'][0] > self.leak: 
            self.leak = params_range['leak'][0]
        elif self.leak > params_range['leak'][1]:
            self.leak = params_range['leak'][1]
            
        if params_range['competition'][0] > self.competition: 
            self.competition = params_range['competition'][0]
        elif self.competition > params_range['competition'][1]:
            self.competition = params_range['competition'][1]        
            
        if params_range['self_excit'][0] > self.self_excit: 
            self.self_excit = params_range['self_excit'][0]
        elif self.self_excit > params_range['self_excit'][1]:
            self.self_excit = params_range['self_excit'][1]  
            
        if params_range['w_input'][0] > self.w_input: 
            self.w_input = params_range['w_input'][0]
        elif self.w_input > params_range['w_input'][1]:
            self.w_input = params_range['w_input'][1] 
            
        if params_range['w_cross'][0] > self.w_cross: 
            self.w_cross = params_range['w_cross'][0]
        elif self.w_cross > params_range['w_cross'][1]:
            self.w_cross = params_range['w_cross'][1] 
            
        if params_range['noise_sd'][0] > self.noise_sd: 
            self.noise_sd = params_range['noise_sd'][0]
        elif self.noise_sd > params_range['noise_sd'][1]:
            self.noise_sd = params_range['noise_sd'][1] 
            
        if params_range['offset'][0] > self.offset: 
            self.offset = params_range['offset'][0]
        elif self.offset > params_range['offset'][1]:
            self.offset = params_range['offset'][1] 

        if params_range['threshold'][0] > self.threshold: 
            self.threshold = params_range['threshold'][0]
        elif self.threshold > params_range['threshold'][1]:
            self.threshold = params_range['threshold'][1]             
            
    def make_weights(self, diag_val, offdiag_val):
        """
        Get a connection weight matrix with "diag-offdial structure"
        e.g.
            | x, y, y |
            | y, x, y |
            | y, y, x |
        where x = diag_val, and y = offdiag_val
        
        Parameters
        ----------
        diag_val : float
            the value of the diag entries
        offdiag_val : float
            the value of the off-diag entries
        n_units : int
            the number of LCA nodes
            
        Returns
        -------
        2d array
            the weight matrix with "diag-offdial structure"
        """
        diag_mask = np.eye(self.n_units, dtype=np.float32)
        offdiag_mask = np.ones((self.n_units, self.n_units), dtype=np.float32) - np.eye(self.n_units, dtype=np.float32)
        weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
        return weight_matrix
        
    def _init_2dim(self, n_trials=40, trial_length=750):
        input_seq = np.tile(self.stimulus.flatten(), (n_trials, 1))
        # precompute the transformed input, for all time points
        input_transformed = input_seq[0] @ self.W_i
        input_transformed = np.repeat(input_transformed[np.newaxis, ...], trial_length, axis=0)
        # inp = np.transpose(inp, (1,0,2))
        sqrt_dt_t = np.sqrt(self.dt_t)
        # precompute offset for all units
        offset = self.offset * np.ones(self.n_units, )
        # precompute sqrt(dt/tao)
        trial_coord = np.zeros(n_trials, dtype=np.float32) - 1
        trial_rt = np.zeros(n_trials, dtype=np.float32) - 1
        V = np.zeros(shape=(n_trials, trial_length, self.n_units), dtype=np.float32)
        init_val = np.ones(self.n_units, dtype=np.float32)
        return input_transformed, sqrt_dt_t, offset, trial_coord, trial_rt, V, init_val

class SLCA_global(SLCA):
    """
    The SLCA with global inhibition
    """

    def __init__(
            self, stimulus, params_range, dt_t=0.01, leak=0.2, competition=1, self_excit=1,
            w_input=1, w_cross=0, offset=0, noise_sd=0, threshold=1):
        super().__init__(
            stimulus=stimulus, params_range=params_range, dt_t=dt_t, leak=leak, competition=competition, self_excit=self_excit,
            w_input=w_input, w_cross=w_cross, offset=offset, noise_sd=noise_sd, threshold=threshold)
        # the input / recurrent weights
        self.W_i = self.make_weights(self.w_input, self.w_cross)
        self.W_r = self.make_weights(self.self_excit, -self.competition)

    def run_2dim(self, n_trials=40, trial_length=750):
        inp, sqrt_dt_t, offset, trial_coord, trial_rt, V, init_val = self._init_2dim(n_trials=40, trial_length=750)

        for trial in range(n_trials):
            # the LCA computation at time t
            for time_moment in range(trial_length):
                V_prev = init_val if time_moment == 0 else V[trial, time_moment - 1, :]
                V[trial, time_moment, :] = V_prev + offset + noise[time_moment] * sqrt_dt_t + (inp[trial] - \
                                                                                               self.leak * V_prev + self.W_r @ V_prev) * self.dt_t
                V[trial, time_moment, :][V[trial, time_moment, :] < 0] = 0

                if np.any(V[trial, time_moment, :] > self.threshold):
                    coords = np.where(V[trial, time_moment, :] > self.threshold)[0]
                    trial_coord[trial] = coords[np.argmin(abs(coords - trial_coord[trial - 1]))]
                    trial_rt[trial] = time_moment
                    break

            if trial_coord[trial] == -1:
                trial_coord[trial] = trial_coord[trial - 1]

        return trial_coord, trial_rt  

class SLCA_local(SLCA):
    """
    The SLCA with local lateral inhibition
    """

    def __init__(
            self, stimulus, params_range, dt_t=0.01, leak=0.2, competition=1, self_excit=1,
            w_input=1, w_cross=0, offset=0, noise_sd=0, threshold=1):
        super().__init__(
            stimulus=stimulus, params_range=params_range, dt_t=dt_t, leak=leak, competition=competition, self_excit=self_excit,
            w_input=w_input, w_cross=w_cross, offset=offset, noise_sd=noise_sd, threshold=threshold)
        # the input / recurrent weights
        self.W_i = self.make_weights(self.w_input, self.w_cross)
        self.W_r = self.make_weights(self.self_excit, 0)
        
    def local_inhibition(self, vector, pad=1):
        filt = [[-pad,-pad,-pad],[-pad,1,-pad],[-pad,-pad,-pad]]
        vector = np.reshape(vector, [self.h, self.w])
        m = np.pad(vector, (1,1), 'constant', constant_values=np.median(vector))
        return np.reshape(signal.convolve(m, filt, mode='same')[1:-1, 1:-1], [self.w*self.h,])
    
    def run_2dim(self, n_trials=40, trial_length=750):
        inp, sqrt_dt_t, offset, trial_coord, trial_rt, V, init_val = self._init_2dim(n_trials=40, trial_length=750)

        for trial in range(n_trials):
            noise = np.random.normal(scale=self.noise_sd, size=(trial_length, self.n_units))

            # the LCA computation at time t
            for time_moment in range(trial_length):
                
                V_prev = init_val if time_moment == 0 else V[trial, time_moment - 1, :]
                V[trial, time_moment, :] = self.local_inhibition(V_prev) + self.offset + noise[time_moment] * sqrt_dt_t + (inp[trial] - self.leak * V_prev + self.W_r @ V_prev) * self.dt_t
                    
                V[trial, time_moment, :][V[trial, time_moment, :] < 0] = 0
                    
                if np.any(V[trial, time_moment, :] > self.threshold):
                    coords = np.where(V[trial, time_moment, :] > self.threshold)[0]
                    trial_coord[trial] = coords[np.argmin(abs(coords - trial_coord[trial - 1]))]
                    trial_rt[trial] = time_moment
                    break              
                
            if trial_coord[trial] == -1:
                trial_coord[trial] = trial_coord[trial - 1]

        return trial_coord, trial_rt