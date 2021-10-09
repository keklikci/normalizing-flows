# normalizing flows

### Overview
* Beta-VAE + normalizing flows
  * Beta VAE piped with a normalizing flow
  * this is essentially the source code of full project
  * refer to other directories first if you don't want to take a fast-track

* Noisy moons
  * initial test data - [see usage](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

* Thoracic surgery 
  * current test data - [find out more](https://www.kaggle.com/sid321axn/thoraric-surgery)

* Beta-VAE 
  * variational autoencoder algorithm extended with a beta parameter to put implicit pressure on the learnt posterior
  * [find out more](https://paperswithcode.com/method/beta-vae)

### Updates 
* [Discard experiments directory](https://github.com/kaanguney/normalizing_flows/tree/main/notebooks/experiments).
* [Preprocessing](https://github.com/kaanguney/normalizing_flows/tree/main/scripts/preprocessing) currently supports a dataset called `prostate.xls`. Now supports `ThoracicSurgery.csv` as well.
* [Refer to noisy-moons directory](https://github.com/kaanguney/normalizing_flows/tree/main/noisy-moons) for noisy moons.
* [Refer to beta-vae-normalizing-flows](https://github.com/kaanguney/normalizing_flows/tree/main/beta-vae-normalizing-flows) for latest results as of date of this commit.
  
### Performance Evaluation 
  * KL Divergence
  * Poisson
  * MAE
  * Cross Entropy

### References
* Rezende, D. J., & Mohamed, S. (2015). [Variational Inference with Normalizing Flows.](https://arxiv.org/abs/1505.05770v6)
* Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2019). [Normalizing Flows: An Introduction and Review of Current Methods.](https://arxiv.org/abs/1908.09257v4)
* [Probabilistic Deep Learning with TensorFlow 2 by Imperial College London](https://www.coursera.org/learn/probabilistic-deep-learning-with-tensorflow2)
* Blog posts
  * [Eric Jang](https://github.com/ericjang/normalizing-flows-tutorial)
  * [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
