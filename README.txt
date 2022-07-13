#    References:
#    R. Meattini, A. Bernardini, G. Palli and C. Melchiorri 
#    sEMG-Based Minimally Supervised Regression Using Soft-DTW Neural Networks for Robot Hand Grasping Control
#
#    Copyright (c) 2022. R. Meattini, A. Bernardini, G. Palli and C. Melchiorri
#
#    *Necessary disclaimer*: The files of this package are distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

This package contains the nested cross-validation to evaluate the performances of a sEMG-based minimally supervised regression using soft-DTW neural networks for robot hand grasping control.

## Usage

Run the file 'NestedCrossValidation.m' -> This file computes a nested cross-validation, using the file 'RMS.mat' (i.e., the root mean square value of simulated surface electromyography signals) and 'ref.mat' (trapezoidal reference signal) as input and target datasets of the softDTW neural network. The outputs of the overall nested cross validation are the performances of the network over the test sets.

or, alternatively,

Run ResultsPlot.m to visualize the results of the pre-computed workspace ws_cross_validation.mat with the output of a cross-validation process.

## Folders structure

In the main folder:
- 'NestedCrossValidation.m' -> main file for the computation of nested cross-validation using the file 'RMS.mat' 
- 'sdtw_D.m' -> Computes softDTW discrepancy
- 'sdtw_grad_D.m' -> Computes the gradient of softDTW discrepancy
- 'soft_min_argmin.m' -> Computes the soft min and softargmin 
- 'divergence.m' -> Computes the softDTW divergence value and gradient
- 'myperformance.m' -> Custom performance function used in training the network
- 'ws_cross_validation.mat' -> Matlab workspace of an already computed nested cross-validation, with the target signal shrunk to 1/3
- 'ResultsPlot.m' -> Plots the output of the softDTW network for training and test sets (for each outer loop of the nested cross validation)

In the '+myperformance' folder there are the specific functions used to implement the forward and backward propagation of the softDTW neural network
