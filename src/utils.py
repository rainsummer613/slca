import json
from matplotlib.image import imread
import numpy as np
import os

from src.slca import *

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        
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
                                gaze[int(x / 16)][int(y / 16)] += lens[i]
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
                eml_net = eml_net[::16, ::16]
                eml_net = (eml_net - np.min(eml_net)) / (np.max(eml_net) - np.min(eml_net))
                pics[pic] = {'map': eml_net, 'salient': len(eml_net[eml_net > 0.6])}
        return pics          
           
class ParamsLoader:
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.filename = 'ga_parameters'
        
    def _parse_params(self, params):
        return params.values()
        
    def load(self):
        file_path = f'{self.parameter_folder}/{self.filename}.json'
        if not os.path.isfile(file_path):
            raise Exception(f'Please check the parameters folder {self.parameter_folder}! It should contain file {self.filename}.json. Either file is absent, or the name of the file is incorrect, or there is no such file in the folder.')
        else:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                    return self._parse_params(params)
            except:
                raise Exception(f'Error while reading file {file_path} in the folder {self.parameter_folder}. Please check whether the file follows correct json standards.')
           
def get_metric_exception_row(available_metrics, exception_start):
    exception_row = exception_start
    for metric in self.available_metrics:
        exception_row += f'\n{metric} - {self.available_metrics[metric]}'
    return exception_row
           
class GA_ParamsLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.filename = 'ga_parameters'
        self.available_metrics = {'ks': 'Kolmogorov-Smirnov, a temporal metric', 'aj': 'AUC-Judd, a spatial metric'}
    
    def _parse_params(self, params):
        n_generations, gen_size, model_type, participants, metrics = params.values()
        if gen_size % 4 != 0:
            raise Exception('The amount of the descendants in one generation should be a multiple of 4. Please check the parameter gen_size in the file {self.filename}.json')
        
        if model_type == 'local':
            lca_model = SLCA_local
        elif model_type == 'global':
            lca_model = SLCA_global
        else:
            raise Exception(f'Please specify one of the available types of the SLCA model: local or global (file {self.filename}.json, parameter model_type')
        
        metric_names = []
        for metric_group_1 in metrics:
            for metric_group_2 in metrics[metric_group_1]:

                #if user specified an unknown metric in the ga_parameters.json file
                for metric in metrics[metric_group_1][metric_group_2]:
                    if metric not in self.available_metrics:
                        raise Exception(get_metric_exception_row(self.available_metrics, 'Only the following metrics are available for the parameter metrics in the file {self.filename}.json:'))
                
                    metric_names.append(metric)
        
        #if user did not specify any metric in the ga_parameters.json file
        if len(metric_names) == 0:
            raise Exception(get_metric_exception_row(self.available_metrics, 'Please specify at least one of the following metrics in the file {self.filename}.json:'))
        
        if len(participants) == 0:
            participants = list(range(1,47))
            
        return n_generations, gen_size, model_type, participants, metrics, lca_model
        
class SLCA_StaticParamsLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.filename = 'slca_static_parameters'
    
    def _parse_params(self, params):
        dt_t, threshold, trial_length, n_trials = params.values()
        return dt_t, threshold, trial_length, n_trials
        
class SLCA_ParamsRangeLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.filename = 'slca_parameters_range'    
        
    def _parse_params(self, params):
        params_range = list(params.values())
        return params_range
        
class SLCA_ParamsInitLoader(ParamsLoader):
    def __init__(self, parameter_folder, model_type):
        super().__init__(parameter_folder)
        self.filename = f'slca_parameters_init_{model_type}'    
        
    def _parse_params(self, params):
        init_params_array = [list(params[param_set].values()) for param_set in params]
        return np.array(init_params_array)