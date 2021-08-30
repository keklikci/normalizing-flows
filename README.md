# Normalizing Flows

### Overview

* Notebooks 
  * n-dimensional flow implementations in jupyter notebook

* Scripts
  * modular python implementations of flows

* Noisy moons
  * initial test data; [see usage.](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

### Implementation [subject to project scope]

* Parameters [subject to project scope]
  * number of hidden layers
  * base distribution -- [gaussian converges better than uniform in most experiments]
  * bijector count
  * neuron size
  * optimizers 
  * iteration count
  
### Performance Evaluation [subject to project scope]
* Performance evaluation will be done at the end of the project.
  * convergence time
  * correctness
  * robustness

### Updates 
* [Discard experiments directory.](https://github.com/kaanguney/normalizing_flows/tree/main/notebooks/experiments)
* [Preprocessing currently supports `prostate.xls`.](https://github.com/kaanguney/normalizing_flows/tree/main/scripts/preprocessing)
* [Refer to noisy-moons directory]() for the most recent, most visual implementation.

### References
* Rezende, D. J., & Mohamed, S. (2015). [Variational Inference with Normalizing Flows.](https://arxiv.org/abs/1505.05770v6)
* Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2019). [Normalizing Flows: An Introduction and Review of Current Methods.](https://arxiv.org/abs/1908.09257v4)
* [Probabilistic Deep Learning with TensorFlow 2 by Imperial College London](https://www.coursera.org/learn/probabilistic-deep-learning-with-tensorflow2)
* Blog posts
 * [Eric Jang](https://github.com/ericjang/normalizing-flows-tutorial)
 * [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
