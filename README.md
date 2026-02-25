# Minimal_computation
Code and data for "Direct dependencies between neurons explain activity" by Christopher W. Lynn.

The neuronal activity datasets are included in the following files:

(1) data_mouse_hippocampus.mat: Mouse hippocampus data from Gauthier & Tank, "A dedicated population for reward coding in the hippocampus", Neuron (2018).

(2) data_mouse_visual_responding.mat: Mouse visual cortex data during responses to natural images from Stringer, Pachitariu, Steinmetz, Carandini, & Harris, "High-dimensional geometry of population responses in visual cortex", Nature (2019).

(3) data_mouse_visual_spontaneous.mat: Same neurons as (2) recorded during spontaneous activity.

(4) data_C_elegans.mat: C. elegans data from Dag et al. "Dissecting the functional organization of the C. elegans serotonergic system at whole-brain scale", Cell (2023).

For each time-series X, X(i,t) represents activity (1) or silence (0) for neuron i at time t.

To fit a single maximum entropy model with a specified set of inputs, use "maxEnt_neuron.m".

To fit a sequence of maximum entropy models and find the optimal set of inputs, use "minimax_entropy.m". For a given output neuron, this function greedily adds inputs that minimize the model entropy until the model predicts the pairwise correlations with all neurons (within experimental models). This greedy algorithm is used to compute the "complete" models in the paper.

Note: When switching between different datasets, one should update (1) the gradient descent threshold in "maxEnt_neuron.m" and (2) the numbers of inputs to sweep over in "minimax_entropy.m".
