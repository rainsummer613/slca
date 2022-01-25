# Spatial Leaky Competing Accumulator Model
The modification of Leaky Competing Accumulator Model by Usher & McClelland (2001). 

## The original model
A model of spiking neural network, where each neuron accumulates input over time. The values of the neurons are updated with use of biologically inspired mechanisms:
- information leakage,
- recurrent self-excitation,
- non-linearity,
- random noise.

If you have eye-tracking data with coordinates and durations of human gaze, this model can simulate it. Our model can do it too 😉\
The model has many parameters, which are stated in the description of the class `SLCA` in the file `src/slca.py` and further in *How to use*.

## What we added
1) **Local lateral inhibition** - active neurons inhibit only their immediate neighbors. The original **global inhibition** is also implemented: each neuron inhibits all other neurons.
2) **Saliency map as the input**. Each pixel of the stimulus image correspond to a pixel of the saliency map and to a neuron-like accumulator unit. For creating saliency maps from the raw images we used pretrained EML-NET model (2020), but any other saliency model will suffice.
3) **Genetic algorithm** for finding the optimal model parameters.

## Our data
We trained and tested our model on the following experimental data: 46 participants had to search real photos of scenes for multiple instances of either cup or picture. These images were taken from the LabelMe dataset (Russell et al., 2008), which provides images of indoor and outdoor scenes. Data were collected using an eye tracker EyeLink 1000+, with the sampling rate  1000 Hz. Fixation detection was set to a velocity threshold of 35 degrees per second. In addition to that, the fixations with RT <100 and >750 ms were dropped as outliers. The detailed description of the data collection process was provided in (Merzon et al, 2019)\
<br>
🌸 **In simple words**: we recorded the gaze of the participants, while they were searching for a cup or a picture on the image from the LabelMe dataset. When they fixated at some point, we recorded the spatial coordinates of this point and how long the participant was looking at it.

![image](https://user-images.githubusercontent.com/32509711/150928717-d543b1bb-091c-4c06-9a4d-143f11604eea.png)

## How to use
Model can be used with the parameters that we found during optimization, or with any other parameters. If you want to try to find more optimal parameters - you need to complete the next step and retrain the model.

If you just want to check it out - please run the `example.ipynb` file. **What you need**:
- a pretrained saliency model,
- a picture to simulate the fixations on it,
- maybe some human RT data to evaluate the model performance.
    
**What you get**:
- a saliency map,
- a sequence of the fixation durations,
- a simple graph which shows their distribution,
- if you provided the human data, then also its comparison to the simulated data.

## How to train
1) Install all required libraries from the `requirements.txt`.
2) Prepare your **training data**. Create a `data` folder which contains 3 things: files `all_coords.json` (`data/all_coords.json`) and `all_rts.json` (`data/all_rts.json`), and directory `smaps` (`data/smaps`). 
    * File `all_coords.json` should contain the coordinates of the human gaze over time. We used the data from multiple participants and multiple images, but you can have data from only one participant and/or one image. Please follow the structure of the file:

        ```
        {
        "image1.jpg": 
            {"participant1": 
                [
                [10.10, 10.10], 
                [14.14, 15.15]
                ]
            }
        }
        ```
    * File `all_rts.json` should contain the durations of gaze fixations over time. We used the data from multiple participants and multiple images, but you can have data from only one participant and/or one image. Please follow the structure of the file:

        ```
        {
        "image1.jpg": 
            {"participant1": 
                [100.0, 115.0, 120.0, 200.0]
            }
        }
        ```
4) Create a subdirectory `smaps/eml_net` which should contain saliency maps pre-generated by EML-NET. If you use a different saliency model and want to rename or remove the subdirectory, don't forget to change the path in class method `DataLoader.load_stim` in the file `utils.py`. We used the saliency maps of size 1920 x 1080 but you can use images of different size.\
**Important!** Names of the files with saliency maps should be derived from the names of the original images, e.g. `image1.jpg` > `image1_smap.png`.
5) Prepare configuration files for training a model. There are 4 config files in total: `ga_parameters.json`, `slca_parameters_sim.json`, `slca_parameters_fixed.json`,  `slca_parameters_init.json`, `slca_parameters_range.json`. Let's describe all of them! 
    * In `ga_parameters.json` we specified main parameters for running the entire training procedure. 
    
        ```
        {
            "n_generations": 100,
            "gen_size": 12,
            "model_type": "local",
            "participants": [],
            "metrics": 
               {
               "spatial":
                        {
                        "raw": [],
                        "sal": []
                        },
               "temporal":
                        {
                        "all": [],
                        "ind": ["ks"]
                        }
                }
        }
        ```
        
        - **n_generations**: *int*, the amount of epochs/generations for the genetic algorithm to train on.
        - **gen_size**: *int*, the amount of the descendants in one generation.
        - **model type**: *string*, type of lateral inhibition, can be **global** or **local**.
        - **participants**: *list*, list of the indices of the participants whose data you want to use. If no indices are specified, then the entire dataset is used.
        - **metrics**: a 3-level *dictionary*, contains names of metrics to evaluate the performance.\
            Available metrics: 
            + **ks** - Kolmogorov-Smirnov non-parametric test, a temporal metric, can be either used on *all* data, i.e. first data is simulated for all images and participants, then combined, and after that 500 random samples are taken from this combined data and from human combined data. These samples are compared to each other. Or it can be used as *ind*, i.e. calculated for each image and participant separately and then averaged.
            + **aj** - AUC-Judd test for evaluating the saliency map, a sptial metric, better use on *sal* data, i.e. on simulated saliency maps.
        
    * In `slca_parameters_sim.json` we specified parameters for simulation which don't participate in the accumulator values update.
        
        ```
        {
            "trial_length": 750,
            "n_trials": 40,
            "desired_res": [120, 68]
        }
        ```
        - **trial_length**: *int*, length of a single simulated trial.
        - **n_trials**: *int*, the amount of trials simulated for one image.
        - **desired_res**: *tuple*, the expected resolution for a saliency map.
            
            
    * In `slca_parameters_range.json` we should specify range of values for the parameters. If values of any of these parameters are changed during the optimization, they should stay in their range.\
    **Imporant!** The default values for all parameter ranges are pre-defined in the code. You can use them if you want. Only the ranges for parameters that you specified in the config file will override the default ones.
          
        ```
        {
             "dt_t": 0.01
             "leak": [0.1, 0.5],
             "competition": [0.0005, 1.0],
             "self_excit": [0.05, 0.5],
             "w_input": [0.1, 0.9],
             "w_cross": [0.2, 1.0],
             "offset": [0.0, 10.0],
             "noise_sd": [0.2, 5.0],
             "threshold": [2.0, 10.0],
             "threshold_change": [0.0, 5.0]
         }
         ```
        - **dt_t**: the time step size,
        - **leak**: the leakage term,
        - **competition**: the strength of lateral inhibition across accumulators,
        - **self_excit**: self excitation of the accumulator,
        - **w_input**: input strengh of the feedforward weights,
        - **w_cross**: cross talk of the feedforward weights,
        - **offset**: the additive drift term of the SLCA process,
        - **noise_sd**: the sd of the noise term of the LCA process,
        - **threshold**: the activation threshold,
        - **threshold_change**: how much the activation threshold depends on overall image saliency.
         
    * In `slca_parameters_init.json` we should initialize the arbitrary amount of values for some of the parameters. If some parameter is missing in this config file, it will be initialized randomly (in a given range).
        
        ```
        {
           "0": {
               "leak": 0.256, 
               "competition": 0.5, 
               "self_excit": 0.37, 
               "w_input": 0.64, 
               "w_cross": 0.097, 
               "offset": 0.31, 
               "noise_sd": 1.043, 
               "threshold_change": 0.465
               }
         }
         ```
         
    * In `slca_parameters_fixed.json` there values of the parameters which we don't want to optimize. They will stay the same through the entire training process. Scientists usually fix the parameters `dt_t` and `threshold` in models of this kind.
        
        ```
        {
            "dt_t": 0.01,
            "threshold": 5.0
        }
        ```
  6. 💪 Good job! Now you can run `train.py` and wait until optimization is complete.
        
## References
* Jia, S., & Bruce, N. D. (2020). EML-NET: An expandable multi-layer network for saliency prediction. *Image and Vision Computing*, 103887. https://doi.org/10.1016/j.imavis.2020.103887.
* Merzon, L., Malevich, T., Zhulikov, G., Krasovskaya, S., & MacInnes, W. J. (2020). Temporal Limitations of the Standard Leaky Integrate and Fire Model. *Brain Sciences*, 10(1), 16.	 https://doi.org/10.3390/brainsci10010016.
* Russell, B. C., Torralba, A., Murphy, K. P., & Freeman, W. T. (2008). LabelMe: a database and web-based tool for image annotation. *International Journal of Computer Vision*, 77(1–3), 157–173. https://doi.org/10.1007/s11263-007-0090-8.
* Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. *Psychological Review*, 108(3), 550. http://doi.org/10.1037/0033-295X.108.3.550
* We also used some code from https://github.com/qihongl/pylca
