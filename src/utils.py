import json
from matplotlib.image import imread
import numpy as np
import os

from collections import Counter

from src.measure import *
from src.slca import *

class DataLoader:
    def __init__(self, data_folder, desired_res):
        self.data_folder = data_folder
        
        self.desired_res = desired_res
        self.human_rt = self.load_human_data('all_rts')
        self.stim = self.load_stim()
        self.human_coords = self.get_human_gaze()

    def load_human_data(self, file):
        with open(f'{self.data_folder}/{file}.json', 'r') as f:
            data = json.load(f)
        return data

    def get_human_gaze(self):
        human_coords = self.load_human_data('all_coords')
        data = {}
        for stim in human_coords:
            data[stim] = {}

            for participant in human_coords[stim]:
                if stim in self.stim:
                    gazepoints = human_coords[stim][participant]
                    if len(gazepoints) > 0:
                        lens = self.human_rt[stim][participant]
                        gaze = np.zeros(self.stim[stim]['map'].shape)
                        for i in range(len(gazepoints)):
                            y, x = gazepoints[i]
                            try:
                                gaze[int(x / self.x_decrease)][int(y / self.y_decrease)] += lens[i]
                            except:
                                pass
                        gaze = (gaze - np.min(gaze)) / (np.max(gaze) - np.min(gaze))
                        data[stim][participant] = np.array(gaze, dtype=np.float32)
        return data

    def load_stim(self):
        pics = {}
        for pic in self.human_rt:
            im_path = f'{self.data_folder}/smaps/eml_net/{pic[:-4]}_smap.png'
            if os.path.isfile(im_path):
                eml_net = imread(im_path)

                if self.desired_res[0] > eml_net.shape[1] or self.desired_res[1] > eml_net.shape[0]:
                    raise Exception('Desired saliency map resolution should be less than original resolution. Please change the parameter desired_res in the file slca_static_parameters.json')
                
                self.y_decrease = round(eml_net.shape[0] / self.desired_res[1])
                self.x_decrease = round(eml_net.shape[1] / self.desired_res[0])
                
                eml_net = eml_net[::self.y_decrease, ::self.x_decrease]
                eml_net = (eml_net - np.min(eml_net)) / (np.max(eml_net) - np.min(eml_net))
                pics[pic] = {'map': eml_net, 'salient': len(eml_net[eml_net > 0.6])}
        return pics          
           
class ParamsLoader:
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.all_params = {'dt_t': 0, 'leak': 1, 'competition': 2, 'self_excit': 3, 'w_cross': 4, 'w_input': 5, 'offset': 6, 'noise_sd': 7, 
                           'base_threshold': 8, 'threshold_change': 9}
        
    def _parse_params(self, params):
        return params
        
    def load(self, filename):
        self.filename = filename
        file_path = f'{self.parameter_folder}/{filename}.json'
        if not os.path.isfile(file_path):
            raise Exception(f'Please check the parameters folder {self.parameter_folder}! It should contain file {self.filename}.json. Either file is absent, or the name of the file is incorrect, or there is no such file in the folder.')
        else:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                    return self._parse_params(params)
            except:
                raise Exception(f'Error while reading file {file_path} in the folder {self.parameter_folder}. Please check whether the file follows correct json standards.')
           
class GA_ParamsLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.available_metrics = {'ks': {'descr': 'Kolmogorov-Smirnov, a temporal metric', 'method': KolmogorovSmirnov}, 
                                  'aj': {'descr': 'AUC-Judd, a spatial metric', 'method': AucJudd}
                                  }
    
    def _parse_params(self, params):
        n_generations, gen_size, model_type, participants, metrics = params.values()
        if gen_size % 4 != 0:
            raise Exception(f'The amount of the descendants in one generation should be a multiple of 4. Please check the parameter gen_size in the file {self.filename}.json')
        
        if model_type == 'local':
            lca_model = SLCA_local
        elif model_type == 'global':
            lca_model = SLCA_global
        else:
            raise Exception(f'Please specify one of the available types of the SLCA model: local or global (file {self.filename}.json, parameter model_type')
        
        n_metrics = 0
        metric_methods = {}
        for metric_group_1 in metrics:
            metric_methods[metric_group_1] = {}
            for metric_group_2 in metrics[metric_group_1]:
                metric_methods[metric_group_1][metric_group_2] = {}
                #if user specified an unknown metric in the {self.filename}.json file
                for metric in metrics[metric_group_1][metric_group_2]:
                    if metric not in self.available_metrics:
                        raise Exception(get_metric_exception_row(self.available_metrics, f'Only the following metrics are available for the parameter metrics in the file {self.filename}.json:'))
                    
                    metric_methods[metric_group_1][metric_group_2][metric] = self.available_metrics[metric]['method']                
                    n_metrics += 1
        
        #if user did not specify any metric in the {self.filename}.json file
        if n_metrics == 0:
            raise Exception(get_metric_exception_row(self.available_metrics, f'Please specify at least one of the following metrics in the file {self.filename}.json:'))
        
        if len(participants) == 0:
            participants = list(range(1,47))
            
        return n_generations, gen_size, model_type, participants, metric_methods, n_metrics, lca_model
        
class SLCA_ParamsInitLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        
    def _parse_params(self, param_sets):
        for _, params in param_sets.items():
            extra_params = set(params.keys()) - set(self.all_params.keys())
            
            if len(params) > len(set(params.keys())):
                param_counter = Counter(list(params.leys()))
                param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
                raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
            elif len(extra_params) > 0:
                raise Exception(f'Extra unknown parameters in file {self.filename}.json: {extra_params}')
    
        return [{self.all_params[p]: param_set[p] for p in param_set} for param_set in param_sets.values()]
        
class SLCA_ParamsRangeLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        
    def _parse_params(self, params):
        extra_params = set(params.keys()) - set(self.all_params.keys())
        missed_params = set(self.all_params.keys()) - set(params.keys())
        
        if len(params) > len(set(params.keys())):
            param_counter = Counter(list(params.leys()))
            param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
            raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
        elif len(extra_params) > 0:
            raise Exception(f'Extra unknown parameters in file {self.filename}.json: {extra_params}')
        elif len(missed_params) > 0:
            raise Exception(f'Please specify missing parameter(s) {missed_params} in file {self.filename}.json')
        
        return params
        
class SLCA_ParamsFixedLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        
    def _parse_params(self, params):
        extra_params = set(params.keys()) - set(self.all_params.keys())
        
        if len(params) > len(set(params.keys())):
            param_counter = Counter(list(params.leys()))
            param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
            raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
        elif len(extra_params) > 0:
            raise Exception(f'Extra unknown parameters in file {self.filename}.json: {extra_params}')
        
        return {self.all_params[p]: params[p] for p in params}
        
def get_metric_exception_row(available_metrics, exception_start):
    exception_row = exception_start
    for metric in self.available_metrics:
        exception_row += f"\n{metric} - {self.available_metrics[metric]['descr']}"
    return exception_row