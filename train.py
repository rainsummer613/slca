import argparse

from src.genetic import *
from src.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', '-p', default='args', help='path to the parameters folder') 
    parser.add_argument('--data', '-d', type=str, default='data', help='path to the data folder')
    
    args = parser.parse_args()
    data_folder = args.data
    parameter_folder = args.parameters
    
    #parse the parameters of the genetic algorithm
    gens_num, gen_size, model_type, participants, metrics, n_metrics, lca_model = GA_ParamsLoader(parameter_folder).load()
    #parse static (fixed) parameters of the SLCA
    dt_t, threshold, trial_length, n_trials = SLCA_StaticParamsLoader(parameter_folder).load()
    #parse the parameters range of the SLCA
    params_range = SLCA_ParamsRangeLoader(parameter_folder).load()
    #parse the initial parameters of the SLCA
    params_init = SLCA_ParamsInitLoader(parameter_folder, model_type).load()
    
    data = DataLoader(data_folder)
    
    #initialize the genetic algorithm
    ga = GA(gen_size=gen_size, n_trials=n_trials, trial_length=trial_length, dt_t=dt_t, threshold=threshold, params_range=params_range, 
            data=data, participants=participants, metrics=metrics, n_metrics=n_metrics, lca_model=lca_model)
    #initial parameter sets
    params = np.zeros(shape=(gens_num, gen_size, params_init.shape[1]))
    params[0] = ga.first_gen()
    params[0][:len(params_init)] = params_init
                
    #start optimization
    for g in range(1, gens_num):
        print(f'GENERATION {g}/{gens_num}')
        try:
            params[g] = ga.next_gen(g, params[g - 1])
            print(f'GENERATION {g} RES {list(params[g])}')
        except Exception as e:
            print("!!! Exception", str(e))
    