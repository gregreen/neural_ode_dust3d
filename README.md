# neural_ode_dust3d

3D dust mapping, using neural ODEs to integrate line-of-sight dust.

# Background

In 3D dust mapping, we use stars to constrain the distribution of dust (typically in the Milky Way).
We first measure (or probabilistically infer) the distance and extinction of each star. In a
somewhat simplified picture (omitting variation in the dust extinction "law" and bandpass effects),
the extinction of a star is proportional to the integrated dust column density from our position in
the Solar System to the star. Each observed star therefore puts a constraint on the dust density
field.

If we are given a model of the dust density field and a set of stellar positions, we can calculate
the expected extinction of each star by integrating through the dust density field. By comparison
with the measured stellar extinctions, we obtain a *likelihood* for our model. Combined with a
prior on the model parameters, we obtain a *posterior* on the dust density model. Most 3D dust
mapping methods take this as their basic approach, with differences in how the model of dust density
is parameterized. This is the approach that this project takes as well.

# Neural ODEs

This project takes a very straightforward approach to finding the maximum-posterior model of the
dust density field. Neural ordinary differential equations (ODEs) are a general approach to
integrating ODEs that involve auto-differentiable functions (such as neural networks), and then
calculating the gradient of the results with respect to free parameters in the ODE (such as
neural network weights).

In this project, neural ODEs are used to calculate line integrals through
the dust density field. In this way, we can calculate the extinctions of a set of stars, and their
gradients with respect to the model parameters. This allows us to calculate the gradient of the
likelihood with respect to the model parameters. If the prior is also formulated as an
auto-differentiable function, then we can calculate the gradients of the posterior distribution as
well. We can begin with a guess of dust density model paramters, and then use standard
gradient-descent algorithms to maximize the posterior density. Given that we often work with very
large numbers of stars, we can use *batched* stochastic gradient descent.

# Implementation

This approach can be applied to a wide range of dust density models. Currently, we use a Fourier
series to represent the dust density field. An even more straightforward - but less physically
interpretable - representation would be a feed-forward neural network.

We currently implement this project in [Tensorflow 2.x](https://www.tensorflow.org/), using the
Dormand-Prince (adaptive 4th/5th-order Runga-Kutta) integrator from
[Tensorflow Probability](https://www.tensorflow.org/probability).

The core of the implementation is quite short: a few hundred lines of code. Indeed, the simplicity
and directness of this approach is one of its attractions.
