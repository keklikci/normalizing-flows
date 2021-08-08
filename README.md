# Normalizing Flows

### Overview

* Notebooks 
  * n-dimensional flow implementations in jupyter notebook

* Scripts
  * modular python implementations of flows

### Implementation 

* Parameters
  * number of hidden layers
  * base distribution -- [always use Gaussian]
  * bijector count
  * neuron size
  * optimizers 
  * iteration count
  
### Performance Evaluation
* Performance evaluation will be done at the end of the project.
  * convergence time
  * correctness
  * robustness

### References
* Rezende, D. J., & Mohamed, S. (2015). [Variational Inference with Normalizing Flows.](https://arxiv.org/abs/1505.05770v6)
* Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2019). [Normalizing Flows: An Introduction and Review of Current Methods.](https://arxiv.org/abs/1908.09257v4)
* This repository is inspired by [Eric Jang](https://github.com/ericjang/normalizing-flows-tutorial) and [TensorFlow probability docs.](https://www.tensorflow.org/probability)
