Energy Based Generative Model
=============================
Energy-based generative models operate on the principle of assigning an energy score to each data sample, 
where lower energy values correspond to more plausible samples. By learning an energy function that assigns low energy 
to observed data points and high energy to others, these models can effectively generate novel samples that resemble 
the training data.

Unlike probabilistic generative models that directly model the probability distribution of the data, EBMs define the 
probability of a data sample as an exponential function of its energy. This formulation allows for greater flexibility 
in modeling complex data distributions, as it does not rely on explicitly defining a probability density function.

We are building a deep energy based model to estimate the density distribution of the MNIST dataset. The model is trained
by formulating the negative energy function to a CNN model. It is trained in a contrastive manner, where the model is 
minimizing the energy (maximizing the log likelihood) towards observed data points and maximizing the energy towards 
random sampled data points. 

When generating the samples, we apply a Markov Chain Monte Carlo using Langevin Dynamic as below.

```commandline
sample starting point from Guassian
for sample step k=1 to K:
    x_k = x_{k-1} + eta * grad_x energy(x_{k-1}) + epsilon * N(0, sigma)
x_sample = x_K
```
