import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
import json
import logging
import os
from src.slca import *

from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns

class GA():

    def __init__(self, lca_model, n_units, time, params_diap, human_rt_path, human_coords_path, stim_path, part):
        self.params_diap = params_diap
        self.time = time
        self.n_units = n_units

        self.dt_t = 0.01
        self.threshold = 5.0

        self.human_rt = self.load_human_data(human_rt_path)
        self.human_coords = self.get_human_gaze(human_coords_path)
        self.stim = self.load_stim(stim_path)

        self.part = part
        self.lca_model = lca_model
        self.output = None

    def load_human_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def get_human_gaze(self, human_coords_path):
        human_coords = self.load_human_data(human_coords_path)
        data = {}
        for stim in human_coords:
            data[stim] = {}

            for participant in human_coords[stim]:
                gazepoints = human_coords[stim][participant]
                if len(gazepoints) > 0:
                    lens = self.human_rt[stim][participant]
                    gaze = np.zeros((68, 120))
                    for i in range(len(gazepoints)):
                        y, x = gazepoints[i]
                        try:
                            gaze[int(x / 16)][int(y / 16)] += lens[i]
                        except:
                            pass
                    gaze = (gaze - np.min(gaze)) / (np.max(gaze) - np.min(gaze))
                    data[stim][participant] = np.array(gaze, dtype=np.float32)
        return data

    def load_stim(self, stim_folder):
        pics = {}
        for pic in self.human_coords:
            im_path = f'{stim_folder}/{pic[:-4]}_smap.png'
            if os.path.isfile(im_path):
                eml_net = imread(im_path)
                eml_net = eml_net[::16, ::16]
                eml_net = (eml_net - np.min(eml_net)) / (np.max(eml_net) - np.min(eml_net))
                pics[pic] = {'flat': eml_net.flatten(), 'map': eml_net, 'salient': len(eml_net[eml_net > 0.6])}
        return pics

    def find_pareto_efficient(self, costs):
        p = costs[:].argsort(0)
        efficient = np.ones(len(p))
        for i in range(len(p)):
            efficient[i] = list(p[:,0]).index(i) + list(p[:,1]).index(i) + list(p[:,2]).index(i)
        return efficient.argsort()

    def calculate_fitness(self, parameters):
        all_coords, all_rt, all_ks, all_aj, all_human_rt = [], [], [], [], []

        for stim in list(self.stim.keys()):
            salient = self.stim[stim]['salient']

            T = 40
            input_seq = np.tile(self.stim[stim]['flat'], (T, 1))

            threshold = self.threshold + salient * parameters[7]
            print('new thresh ' + str(threshold))

            if self.lca_model == 'local':
                lca = SLCA_local(n_units=self.n_units, dt_t=self.dt_t, leak=parameters[0], competition=parameters[1], self_excit=parameters[2], w_input=parameters[3], w_cross=parameters[4], offset=parameters[5], noise_sd=parameters[6], threshold=threshold)
            elif self.lca_model == 'global':
                lca = SLCA_global(n_units=self.n_units, dt_t=self.dt_t, leak=parameters[0], competition=parameters[1], self_excit=parameters[2], w_input=parameters[3], w_cross=parameters[4], offset=parameters[5], noise_sd=parameters[6], threshold=threshold)
                
            print(mp.current_process().name + ' LCA started ' + stim + ' ' + ', '.join(
                [str(p) for p in parameters]) + ' ' + str(threshold))
            coords, rt = lca.run_2dim(stimuli=input_seq)
            print(mp.current_process().name + ' LCA finished ' + stim + ' ' + ', '.join([str(p) for p in rt]))

            trial_coord = np.array(coords, dtype=np.float32)
            trials_h = list(trial_coord // 120)
            trials_w = list(trial_coord % 120)

            lca_map = np.zeros(shape=(68, 120), dtype=np.float32)
            for i in range(len(rt)):
                lca_map[int(trials_h[i]), int(trials_w[i])] = rt[i]
            lca_smap = gaussian_filter(lca_map, sigma=1.5)
            lca_smap = (lca_smap - np.min(lca_smap)) / (np.max(lca_smap) - np.min(lca_smap))

            all_coords.append(coords)
            all_rt.extend(rt)

            participant = list(self.human_rt[stim].keys())[self.part]
            #for participant in self.human_rt[stim]:
            if 2 > 1:
                if len(self.human_rt[stim][participant]) == 0:
                    print(f'NO STIM {stim} for participant {participant}')
                else:
                    try:
                        ks = stats.ks_2samp(self.human_rt[stim][participant], rt)
                    except Exception as e:
                        ks = None
                        print(f'ks stim {stim}, participant {participant}:: {str(e)}. RT {rt}, human RT {self.human_rt[stim][participant]}')

                    if ks and ks.statistic:
                        all_ks.append(ks.statistic)
                    else:
                        all_ks.append(0.99999)

                    all_human_rt.extend(self.human_rt[stim][participant])
                    # gaze_smap = gaussian_filter(gaze, sigma=1.5)
                    # gaze_smap = (gaze_smap - np.min(gaze_smap))/(np.max(gaze_smap) - np.min(gaze_smap))

                    if participant in self.human_coords[stim]:
                        try:
                            aj = auc_judd(lca_smap, self.human_coords[stim][participant])
                            print(f'aj GOOD stim {stim}, participant {participant}:: SM max {np.max(lca_smap)}, min: {np.min(lca_smap)}, mean {np.mean(lca_smap)}')
                        except Exception as e:
                            aj = 0.00001
                            print(f'aj stim {stim}, participant {participant}:: {str(e)}. SM max {np.max(lca_smap)}, min {np.min(lca_smap)}, mean {np.mean(lca_smap)}')

                    print(f'stim {stim}, participant {participant}, params {parameters}:: ks {ks}, 1-aj {1 - aj}')
                    all_aj.append(1 - aj)  
                    
        all_human_rt = np.random.choice([el for el in all_human_rt if 50 <= el <= 750], 50)
        if len(all_rt) > 50:
            all_rt = np.random.choice(all_rt, 50)

        ks_fin = stats.ks_2samp(all_human_rt, all_rt).statistic
        res = {'params': parameters, 'fitness': (np.mean(all_ks), np.mean(all_aj))}
        return res

    def process_fit(self, param_set, output, idx):
        print(f'{mp.current_process().name} STARTED params_set {idx}')
        res = self.calculate_fitness(param_set)
        print(f'{mp.current_process().name} FINISHED params_set {idx}')
        output.put(res)
        return res

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

        # find pareto optimal
        print('started computing pareto')
        best_inds = self.find_pareto_efficient(fits)[:3]
        print('best_pareto ' + ' '.join([str(i) for i in best_inds]))

        # fill with other variants if there are <3 optimal
        if len(best_inds) < 3:
            fits_sorted = np.array([ind for ind in np.mean(fits, axis=1).argsort() if ind not in best_inds])
            best_inds = np.r_[best_inds, fits_sorted[:3 - len(best_inds)]]

        results = np.array(results)
        print(f'BEST RES: {results[best_inds]}')
        return np.array([res['params'] for res in results[best_inds]])

    def random_gen(self, sets_num):
        return np.array([np.random.uniform(low=d[0], high=d[1], size=sets_num) for d in self.params_diap]).T

    def first_gen(self):
        return self.random_gen(12)

    def mutate(self, set_params):
        param_num = np.random.choice((1, 7))
        new_params = set_params.copy()
        new_params[param_num] *= np.random.choice([0.95, 1.05])
        if new_params[param_num] < self.params_diap[param_num][0]:
            new_params[param_num] = self.params_diap[param_num][0]
        elif new_params[param_num] > self.params_diap[param_num][1]:
            new_params[param_num] = self.params_diap[param_num][1]
        return new_params

    def crossover(self, parent0, parent1):
        cross_ind = np.random.choice((1, 7))
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
        gens_random = self.random_gen(3)
        print(f'random shape, {gens_random.shape}')

        gens = np.concatenate([gens_best, gens_mutated, gens_crossover, gens_random])
        print(f'all shape, {gens.shape}')
        return gens
