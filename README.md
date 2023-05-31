# Bayesian Optimization for Optimal Learning Rate Estimation
This repository provides an implementation of Bayesian Optimization (BO) to find the optimal learning rate for a deep neural network for computer vision tasks. It utilizes a Gaussian Process (GP) as a predictive model and employs the Expected Improvement (EI) as an acquisition function. This task was completed as a part of my application in the AutoML Group at the Leibnitz University Hanover.

The objective of this project is to automatize the search for the optimal learning rate that maximizes the claissification performance of a deep residual network on the Fashion-MNIST dataset. The task included the following implementation requirements:
## Requirements
- __Deep neural network__: The residual neural network should be rather small and implemented from scratch.
- __Optimizer__: Stochastic Gradient Descent (SGD) should be used as an optimizer to train the deep neural network.
- __Deep learning framework__: The Framework should be implemented using either PyTorch or JAX. Additionally, PEP8 standards should be adhered to.
- __Optimization steps__: Use an optimization budget of 10 function evaluations.
- __Plotting__: Starting with the second iteration of BO, plot all observations, the posterior mean, the uncertainty estimate, and the acquisition function after each iteration.

The residual network was implemented based on this blogpost:
## Bayesian Optimization Process
1. Initialization: In order to reduce computational time needed, I started with a single learning rate to initlialize and evaluate the model. The learning rate is optimized over the interval [0.0001,0.1].
2. Model Training: Train the deep neural network using the selected learning rates and evaluate its performance. The accuracy of the classification was implemented as an objective function. Again, to reduce computatinal time, the network is only trained for one epoch each.
3. GP Modeling: Build a GP model to fit a function that returns the accuracy of the model for an given learning rate. A Matern52- kernel was used.
4. Acquisition Function: Employ the EI function as the acquisition function to determine the next learning rate to evaluate.
5. Update Model: Update the Gaussian Process model using the newly evaluated learning rate and its corresponding accuracy.
6. Repeat: This process is iterated for ten iterations.
7. Plotting: Starting from the second iteration, where the initialization is the first iteration, of Bayesian Optimization, plot the observations, posterior mean, uncertainty estimate, and acquisition function after each iteration. 

Note that the ResNet9 was implemented following this implementation: https://github.com/ksw2000/ML-Notebook/blob/main/ResNet/ResNet_PyTorch.ipynb.

To run the code, install the necessary dependencies specified in the requierements.txt file. Run the notebook to execute the BO process for learning rate estimation. View the generated plots to analyse the optimization progress.

## Results
These are the generated plot that display the results of the BO, starting after the model fitting with the initialization parameter. There are 10 iterations in total plus the initialization.

<div>
    <img src="plot results/bo_plot_0.png" alt=Slide 1" width="700" height="200" />
    <img src="plot results/bo_plot_1.png" alt=Slide 2" width="700" height="200" />
    <img src="plot results/bo_plot_2.png" alt=Slide 3" width="700" height="200" />
    <img src="plot results/bo_plot_3.png" alt=Slide 4" width="700" height="200" />
    <img src="plot results/bo_plot_4.png" alt=Slide 5" width="700" height="200" />
    <img src="plot results/bo_plot_5.png" alt=Slide 6" width="700" height="200" />
    <img src="plot results/bo_plot_6.png" alt=Slide 7" width="700" height="200" />
    <img src="plot results/bo_plot_7.png" alt=Slide 8" width="700" height="200" />
    <img src="plot results/bo_plot_8.png" alt=Slide 9" width="700" height="200" />
    <img src="plot results/bo_plot_9.png" alt=Slide 10" width="700" height="200" />                                                                           
</div>

The plot includes 1 initilization learning rate as well as ten learning rates used in the ten iterations of BO. The predictive mean of the GP is fitted to these observations. A 95% confidence interval was also generated. Additionally, the acquisition function is plotted over the interval to optimize on. One can see that with each iteration, the varying learning rates lead to an improved fit of the accuracy function. Furthermore, the acquistition function develops more significant peaks, allowing for an increasingly confident prediction of the optimal learning rate.

## Next Steps
To improve the implementation provided here, several steps could still be taken to improve the training, including:
- __The choice of kernel function__: For the GP regressor a Matern52 kernel was used. It is a generalization of the simpler RBF kernel and can handle quite some compexity in the data. However, its parameters, such as the length scale, could be tuned as well as the choice of the kernel function itself.
- __Duration of training__: Due to limited resources, the ResNet is only trained for one epoch each iteration. To increase the accuracy of the prediction, the number of epochs could be increased as well.
- __Evaluation of the acquisition function__: Since the observations/ next learning rates are generated via the acquisition function, finding its global maxima contributes to the performance of the BO. In this implementation, maxima are located by evaluating the EI function on a discrete number of evaluation points ofer the search optimization interval. This is an easy wway to ensure that one does not get stuck in local extrema/ flat ares with e.g. a Gradient Descent algorithm. Still, there should be a way to determine the maximum values via contiunous search.
