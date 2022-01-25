import argparse

from src.genetic import *
from src.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-p', default='config', help='path to the folder with parameters and configurations') 
    parser.add_argument('--data', '-d', type=str, default='data', help='path to the data folder')
    
    config = parser.parse_args()
    data_folder = config.data
    parameter_folder = config.parameters
    
    basic_loader = ParamsLoader(parameter_folder)
    #parse the parameters of the genetic algorithm
    gens_num, gen_size, model_type, participants, metrics, n_metrics, lca_model = GA_ParamsLoader(parameter_folder).load('ga_parameters')
    #parameters of simulation
    trial_length, n_trials, desired_res = basic_loader.load('slca_parameters_sim').values()
    data = DataLoader(data_folder, desired_res)
    #parse the parameters range of the SLCA
    params_range = SLCA_ParamsRangeLoader(parameter_folder).load('slca_parameters_range')
    
    #parse the initial parameters of the SLCA
    params_init = SLCA_ParamsInitLoader(parameter_folder).load('slca_parameters_init')
    #parse fixed parameters of the SLCA
    fixed_parameters = SLCA_ParamsFixedLoader(parameter_folder).load('slca_parameters_fixed')
    
    #initialize the genetic algorithm
    ga = GA(gen_size=gen_size, params_range=params_range, fixed=fixed_parameters,
            param_to_index=basic_loader.all_params, index_to_param={v:k for k,v in basic_loader.all_params.items()},
            data=data, participants=participants, metrics=metrics, n_metrics=n_metrics, lca_model=lca_model,
            trial_length=trial_length, n_trials=n_trials)
    #initial parameter sets
    params = np.zeros(shape=(gens_num, gen_size, len(params_range)))
    params[0] = ga.first_gen()
    
    for i, param_set in enumerate(params_init):
        params[0, i, list(param_set.keys())] = list(param_set.values())
                
    #start optimization
    for g in range(1, gens_num):
        print(f'GENERATION {g}/{gens_num}')
        try:
            params[g] = ga.next_gen(g, params[g-1])
            print(f'GENERATION {g} RES {list(params[g])}')
        except Exception as e:
            print("!!! Exception", str(e))
    