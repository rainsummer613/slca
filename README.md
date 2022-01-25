# Spatial Leaky Competing Accumulator Model
The modification of Leaky Competing Accumulator Model by Usher & McClelland (2001). 

## The original model
A model of spiking neural network, where each neuron accumulates input over time. The values of the neurons are updated with use of biologically inspired mechanisms:
- information leakage
- recurrent self-excitation
- non-linearity
- random noise

## What we added
1) **Local lateral inhibition** - active neurons inhibit only their immediate neighbors. The original **global inhibition** is also implemented: each neuron inhibits all other neurons.
2) **Saliency map as the input**. Each pixel of the stimulus image correspond to a pixel of the saliency map and to a neuron-like accumulator unit.
3) **Genetic algorithm** for finding the optimal parameters.

## How to use
Run `train.py` file for training procedure which i

## References
- Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. *Psychological Review*, 108(3), 550. http://doi.org/10.1037/0033-295X.108.3.550
- We also used some code from https://github.com/qihongl/pylca
