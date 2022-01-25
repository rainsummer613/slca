import json
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import scipy.signal as signal
import seaborn as sns

from matplotlib.image import imread
from scipy.ndimage import gaussian_filter

from src.utils import *

class GA():
    """
    The genetic algorithm class
    """
    def __init__(self, data, participants, metrics, n_metrics, lca_model, n_trials, trial_length, 
                 param_to_index, index_to_param, params_range, fixed, gen_size=12):
        """
        Parameters
        ----------
        metrics : dictionary
            a 3-level dictionary with various types of metrics
        n_metrics : int
            a precomputed total amount of metrics
        gen_size : int
            the amount of the descendants in one generation
        lca_model
            a model class for the data simulation
        n_trials : int
            number of trials in the single data simulation
        trial_length : int
            duration of a single trial
        param_to_index : dict
            a dictionary for mapping the parameter names to the indices
        index_to_param : dict
            a dictionary for reverse mapping the indices to the parameter names
        params_range
            a list or tuple with the parameters range for the data simulation
        fixed
            fixed parameters with indices
        data
            object of the class DataLoaser with all rt and coordinates data loaded 
            path to the dictionary with human fixation times data by images and participants
        participants : list
            list of indices of the participants to train the model on
        """
        self.metric_spatial = metrics['spatial']
        self.metric_temporal = metrics['temporal']
        self.n_metrics = n_metrics
        
        self.gen_size = gen_size
        self.n_desc = int(gen_size / 4)
        self.participants = participants
        self.lca_model = lca_model       
        
        self.params_range = params_range
        self.fixed = fixed
        self.unfixed_inds = tuple(set(range(len(self.params_range))) - set(fixed.keys()))
        self.param_to_index = param_to_index
        self.index_to_param = index_to_param
        
        self.data = data
        self.n_trials = n_trials
        self.trial_length = trial_length

    def calculate_fitness(self, parameters):
        """
        Run one epoch of the gradient descent:
            - simulate data for all the images and participants.
            - calculate how similar it is to the ground truth human data.
        
        Parameters
        ----------
        parameters : np.ndarray
            an array with the model parameters
        """
        all_rt, all_human_rt = [], []
        res = {}

        #iterate over stimulus images
        for stim in list(self.data.stim.keys()):
            salient = self.data.stim[stim]['salient']
            
            #slca requires a single treshold parameter, but we also want to keep both threshold and threshold_change for optimization
            parameters_lca = {self.index_to_param[i]:v for i,v in enumerate(parameters)}
            parameters_lca["threshold"] += salient * parameters[self.param_to_index["threshold_change"]]
            parameters_lca.pop("threshold_change")
            print("new threshold", parameters_lca["threshold"])

            #initialize an SLCA model
            lca = self.lca_model(stimulus=self.data.stim[stim]['map'], params_range=self.params_range, **parameters_lca)
             
            print(mp.current_process().name + ' LCA started ' + stim + ' ' + ', '.join([str(v) for v in parameters]))
            #run the SLCA model and save the results
            coords, rt = lca.run_2dim(self.n_trials, self.trial_length)
            print(mp.current_process().name + ' LCA finished ' + stim + ' ' + ', '.join([str(p) for p in rt]))

            trial_coord = np.array(coords, dtype=np.float32)
            trials_h = list(trial_coord // lca.w)
            trials_w = list(trial_coord % lca.w)

            lca_map = np.zeros(shape=(lca.h, lca.w), dtype=np.float32)
            for i in range(len(rt)):
                lca_map[int(trials_h[i]), int(trials_w[i])] = rt[i]
                
            if len(self.metric_spatial['sal']) > 0:
                lca_smap = gaussian_filter(lca_map, sigma=1.5)
                lca_smap = (lca_smap - np.min(lca_smap)) / (np.max(lca_smap) - np.min(lca_smap))

            if len(self.metric_temporal['all']) > 0:
                all_rt.extend(rt)

            #iterate over participants
            for participant in self.data.human_rt[stim]:
                if participant in self.participants and len(self.data.human_rt[stim][participant]) == 0:
                    print(f'NO STIM {stim} for participant {participant}')
                elif participant in self.participants:
                
                    #iterate over spatial metrics for simulated saliency maps
                    for metric in self.metric_spatial['sal']:
                        metric_full = metric + '_sal'
                        if metric_full not in res:
                            res[metric_full] = []
                        res[metric_full].append(self.metric_spatial['sal'][metric](lca_smap, self.human_coords[stim][participant]).run())
                        print(f'METRIC {metric_full} for stim {stim}, participant {participant}:: {res[metric_full]}')
                    
                    #iterate over spatial metrics for simulated scanpaths
                    for metric in self.metric_spatial['raw']:
                        metric_full = metric + '_raw'
                        if metric_full not in res:
                            res[metric_full] = []
                        res[metric_full].append(self.metric_spatial['raw'][metric](lca_map, self.human_coords[stim][participant]).run())
                        print(f'METRIC {metric_full} for stim {stim}, participant {participant}:: {res[metric_full]}')
                            
                    #iterate over temporal metrics for individual data (by image and participant)
                    for metric in self.metric_temporal['ind']:
                        metric_full = metric + '_ind'
                        if metric_full not in res:
                            res[metric_full] = []
                        res[metric_full].append(self.metric_temporal['ind'][metric_full](rt, self.data.human_rt[stim][participant]).run())
                        print(f'METRIC {metric_full} for stim {stim}, participant {participant}:: {res[metric_full]}')
                            
                    #iterate over temporal metrics for all data combined
                    if len(self.metric_temporal['all']) > 0:
                        try:
                            all_human_rt.extend(self.human_rt[stim][participant])
                        except:
                            pass                   
        
        if len(all_human_rt) > 0 and len(all_rt) > 0:
            for metric in self.metric_temporal['all']:
                metric_full = metric + '_all'
                res[metric_full] = [self.metric_temporal['all'][metric](all_rt, all_human_rt).run()]
                print(f'METRIC {metric_full} for stim {stim} :: {res[metric_full][0]}')
         
        #find averaged metric values + remove non-existent values from calculations
        fitness = {metric: np.mean([val for val in res[metric] if val]) for metric in res}
        return {'params': parameters, 'fitness': fitness.values}

    def process_fit(self, param_set, output, idx):
        print(f'{mp.current_process().name} STARTED params_set {idx}')
        res = self.calculate_fitness(param_set)
        print(f'{mp.current_process().name} FINISHED params_set {idx}')
        output.put(res)
        return res
        
    def find_pareto_efficient(self, costs):
        p = costs[:].argsort(0)
        best_inds = np.ones(len(p))
        for i in range(len(p)):
            best_inds[i] = sum([list(p[:,j]).index(i) for j in len(costs[0])], [])   
        best_inds = best_inds.argsort()
        
        # fill with other variants if there are <3 optimal
        if len(best_inds) < self.n_desc():
            fits_sorted = np.array([ind for ind in np.mean(costs, axis=1).argsort() if ind not in best_inds])
            best_inds = np.r_[best_inds, fits_sorted[:self.n_desc - len(best_inds)]]
        return best_inds

    def select_best(self, sets):
        output = mp.Queue()
        processes = []

        for i in range(len(sets)):
            p = mp.Process(target=self.process_fit, args=(sets[i], output, i))
            processes.append(p)
            print(f'{p.name} started')
            p.start()
        for p in processes:
            p.join()

        results = []
        while not output.empty():
            o = output.get()
            results.append(o)

        fits = np.array([el['fitness'] for el in results])
        print(f'fits {len(fits)} {fits}')

        # find best descendants
        if self.n_metrics == 1:
            best_inds = fits.argsort()[-self.n_desc:][::-1]
        else:
            print('started computing pareto')
            best_inds = self.find_pareto_efficient(fits)[:self.n_desc]

        results = np.array(results)
        print(f'BEST RES: {results[best_inds]}')
        return np.array([res['params'] for res in results[best_inds]])

    def random_gen(self, sets_num):
        gen = np.array([np.random.uniform(low=self.params_range[d][0], high=self.params_range[d][1], size=sets_num) for d in self.params_range]).T
        gen[:,list(self.fixed.keys())] = list(self.fixed.values())
        return gen

    def first_gen(self):
        return self.random_gen(self.gen_size)

    def mutate(self, set_params):
        param_num = np.random.choice(self.unfixed_inds)
        new_params = set_params.copy()
        new_params[param_num] *= np.random.choice([0.95, 1.05])
        if new_params[param_num] < self.params_range[param_num][0]:
            new_params[param_num] = self.params_range[param_num][0]
        elif new_params[param_num] > self.params_range[param_num][1]:
            new_params[param_num] = self.params_range[param_num][1]
        return new_params

    def crossover(self, parent0, parent1):
        cross_ind = np.random.choice(self.unfixed_inds)
        child = parent0.copy()
        child[cross_ind] = parent1[cross_ind]
        return child

    def next_gen(self, g, all_sets):
        print(f'GENERATION {g}')
        print('REMOVING DUPLICATES')
        unq, count = np.unique(all_sets, axis=0, return_counts=True)
        repeated_groups = unq[count > 1]
        for repeated_group in repeated_groups:
            repeated_idx = np.argwhere(np.all(all_sets == repeated_group, axis=1))
            repeated_idx = repeated_idx.ravel()[1:]
            all_sets[repeated_idx] = np.apply_along_axis(self.mutate, 1, all_sets[repeated_idx])

        print('SELECTION')
        gens_best = self.select_best(all_sets)
        print(f'gens_best shape, {gens_best.shape}')

        print('MUTATION')
        gens_mutated = np.apply_along_axis(self.mutate, 1, gens_best)
        print(f'mutated shape, {gens_mutated.shape}')

        print('CROSSOVER')
        parents0 = gens_best.copy()
        parents1_inds = [np.random.choice(np.delete(np.arange(len(gens_best)), [i]))
                         for i in range(len(gens_best))]
        parents1 = gens_best[parents1_inds]

        gens_crossover = np.array([self.crossover(parents0[i], parents1[i]) for i in range(len(parents0))])
        print(f'crossover shape, {gens_crossover.shape}')

        print('RANDOM')
        gens_random = self.random_gen(self.n_desc)
        print(f'random shape, {gens_random.shape}')

        gens = np.concatenate([gens_best, gens_mutated, gens_crossover, gens_random])
        print(f'all shape, {gens.shape}')
        return gens
