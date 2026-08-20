"""Canonical script generated from beta-vae-maf-thoracic-surgery_v1.0.ipynb."""

%run 'preprocessing/preprocessingv1.0.ipynb'

#
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import pandas as pd
import seaborn as sns 
sns.set_style('white')
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # for gradient warning (variables are still updated, therefore supressing is fine)
import tensorflow_probability as tfp 
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.losses import Loss

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

color_list = ['#bcad', '#dacb']

#
# ### Define prior distribution

#
def get_prior(num_modes, latent_dim):
    
    mixture_distribution = tfd.Categorical(probs=[1./num_modes] * num_modes)
    components_distribution = tfd.MultivariateNormalDiag(loc=tf.Variable(tf.random.normal((num_modes, latent_dim))),
                                                        scale_diag=tfp.util.TransformedVariable(tf.ones((num_modes,latent_dim)),
                                                                                               bijector=tfb.Softplus())
                                                        )
    prior = tfd.MixtureSameFamily(mixture_distribution,
                                  components_distribution
                                 )
    return prior

#
latent_dim = len(X_train.columns)
input_shape = len(X_train.columns)

prior = get_prior(num_modes=latent_dim, latent_dim=input_shape) 
print(f'Prior event shape: {prior.event_shape[0]}')
print(f'# of Gaussions: {prior.components_distribution.batch_shape[0]}') 
print(f'Covariance matrix: {prior.components_distribution.name}')

#
# ### Define KL divergence

#
# set weight for more emphasis on KLDivergence term rather than reconstruction loss
# average over both samples and batches

def get_KL_regularizer(prior, weight=4.):
    
    regularizer = tfpl.KLDivergenceRegularizer(prior, 
                                        use_exact_kl=False,
                                        test_points_reduce_axis=(),
                                        test_points_fn=lambda q: q.sample(10),
                                        weight=weight
                                        )
    return regularizer

#
KLDivergence_regularizer = get_KL_regularizer(prior)

#
# ### Define the encoder

#
def get_encoder(input_shape, latent_dim, KL_regularizer):
    
    encoder = Sequential([
        Dense(input_shape=input_shape, units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(latent_dim, 
                                   activity_regularizer=KL_regularizer),
    ])
    
    return encoder

#
encoder = get_encoder(input_shape=(input_shape,), latent_dim=latent_dim, KL_regularizer=KLDivergence_regularizer)
encoder.summary()

#
# ### Define the decoder 

#
def get_decoder(latent_dim):
    
    decoder = Sequential([
        Dense(input_shape=(latent_dim,), units=5, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(latent_dim)
    ])
    
    return decoder

#
decoder = get_decoder(latent_dim)
decoder.summary()

#
# ### Connect encoder to decoder 

#
vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
vae.summary()

#
# ### Specify the loss function 

#
# KL divergence is implicity incorporated to the loss function before
# add reconstruction error to loss function 
def reconstruction_error(decoding_dist, x_true):
    return -tf.reduce_mean(decoding_dist.log_prob(x_true))

class custom_reconstruction_error(Loss):
    def call(self, decoding_dist, x_true):
        return -tf.reduce_mean(decoding_dist.log_prob(x_true))

#
# ### Selection process

#
print(f'# of training samples: {X_train.shape[0]}')
print(f'# of test samples: {X_test.shape[0]}')

#
X_train_prior_visual = pd.DataFrame(scaler.inverse_transform(X_train), columns=X_train.columns).astype('int64')
y_train.reset_index(drop=True, inplace=True)
X_train_prior_visual['Risk1Yr'] = y_train

#
g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=X_train_prior_visual, inner=None, legend_out=True, doge=True, palette='Set3', aspect=2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=X_train_prior_visual, ax=g.ax)
sns.despine()

#
g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=X_train_prior_visual, inner=None, doge=True, palette='Set3', legend_out=True, aspect=2, bw=0.2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=X_train_prior_visual, ax=g.ax)
sns.despine()

#
# ### Compile and fit the model 

#
optimizer = Adam(learning_rate=3e-4)
epochs = 1000 
epoch_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print('\n Epoch {}/{}'.format(epoch+1, epochs, logs),
                                                                       '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                if epoch % 100 == 0 else False
                               )


vae.compile(optimizer=optimizer, loss=reconstruction_error)
history = vae.fit(X_train, 
                  validation_data=(X_test,),
                  epochs=epochs, 
                  batch_size=32,
                  verbose=0,
                  shuffle=True,
                  callbacks=[epoch_callback]
                 )

#
# ### Plot training and validation losses 

#
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.tight_layout()
plt.show()

# Loss function is ELBO maximization
# ELBO maximization is equivalent to KL divergence minimization

#
# ### Sample from the generative model

#
X_train_sample = encoder(X_train.to_numpy()).sample()
X_train_sample = pd.DataFrame(X_train_sample.numpy(), columns=X_test.columns)

#
X_train_sample.head()

#
# propogate back from the pipeline
# training data -- reverse standardization

X_train_sample = pd.DataFrame(scaler.inverse_transform(X_train_sample.to_numpy()), columns=X_train.columns)
X_train_sample = X_train_sample.astype('int64')
X_train_sample.head()

#
# propogate back from the pipeline
# testing data -- reverse standardization

X_test_sample = encoder(X_test.to_numpy()).sample()
X_test_sample = pd.DataFrame(X_test_sample.numpy(), columns=X_train.columns)

X_test_sample = pd.DataFrame(scaler.inverse_transform(X_test_sample.to_numpy()), columns=X_test.columns)
X_test_sample = X_test_sample.astype('int64')
X_test_sample.head()

#
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True) # for concatenation

X_train_sample['Risk1Yr'] = y_train

#
g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=X_train_sample, inner=None, legend_out=True, doge=True, palette='Set3', aspect=2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=X_train_sample, ax=g.ax)
sns.despine()

#
g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=X_train_sample, inner=None, doge=True, palette='Set3', legend_out=True, aspect=2, bw=0.2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=X_train_sample, ax=g.ax)
sns.despine()

#
# decode 
X_train_sample.drop('Risk1Yr',axis=1,inplace=True)
X_train_sample = pd.DataFrame(scaler.inverse_transform(decoder(X_train_sample.to_numpy()).sample().numpy()), columns=X_train.columns)
X_train_sample = X_train_sample.astype('int64')
X_train_sample.head()

#
X_train_sample['Risk1Yr'] = y_train

g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=X_train_sample, inner=None, doge=True, palette='Set3', legend_out=True, aspect=2, bw=0.2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=X_train_sample, ax=g.ax)
sns.despine()

#
# ### Masked Autoregressive Flow

#
X_train_sample.drop('Risk1Yr', axis=1, inplace=True) # drop target, was only intended for VAE sampling

#
loc = [X_train_sample[i].mean().astype('float32') for i in list(X_train.columns)]
scale_diag = [X_train_sample[i].std().astype('float32') for i in list(X_train.columns)]

#
mvn = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
mvn

#
def masked_autoregressive_flow(hidden_units=[16,16], event_shape=[16], activation='relu'):
    network = tfb.AutoregressiveNetwork(params=2, 
                                    hidden_units=hidden_units,
                                    event_shape=event_shape,
                                    activation=activation
                                   )
    return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=network)

#
trainable_dist = tfd.TransformedDistribution(distribution=mvn,
                                             bijector=masked_autoregressive_flow(
                                             activation='sigmoid'))
trainable_dist

#
n_samples = 300
x = mvn.sample(sample_shape=n_samples)
names = [mvn.name, trainable_dist.bijector.name]
samples = [x, trainable_dist.bijector.forward(x)]

#
X_train_np = X_train_sample.to_numpy()
X_test_np = X_test_sample.to_numpy()

#
# standardize once again before feeding into network 
scaler.fit(X_train_np)
X_train_np = scaler.transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

#
X_train = X_train_np.astype(np.float32)
X_train = tf.data.Dataset.from_tensor_slices(X_train)
X_train = X_train.batch(128)

X_valid = X_test_np.astype(np.float32)
X_valid = tf.data.Dataset.from_tensor_slices(X_valid)
X_valid = X_valid.batch(128)

#
num_epochs = 600
opt = tf.keras.optimizers.Adam(3e-4)
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    if epoch % 100 == 0:
        print("Epoch {}...".format(epoch))
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    for train_batch in X_train:
        with tf.GradientTape() as tape:
            tape.watch(trainable_dist.bijector.trainable_variables)
            loss = -trainable_dist.log_prob(train_batch)
        train_loss(loss)
        grads = tape.gradient(loss, trainable_dist.bijector.trainable_variables)
        opt.apply_gradients(zip(grads, trainable_dist.bijector.trainable_variables))
    train_losses.append(train_loss.result().numpy())
        
    # Validation
    for valid_batch in X_valid:
        loss = -trainable_dist.log_prob(valid_batch)
        val_loss(loss)
    valid_losses.append(val_loss.result().numpy())

#
train_losses = history.history['loss']
valid_losses = history.history['val_loss']

plt.plot(train_losses, label='train', c=color_list[0])
plt.plot(valid_losses, label='valid', c=color_list[1])
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Negative log likelihood")
plt.title("Training and validation loss curves")
plt.show()

#
x = mvn.sample(sample_shape=n_samples)
names = [mvn.name, trainable_dist.bijector.name]
samples = [x, trainable_dist.bijector.forward(x)]

#
num_layers = 4
flow_bijector = []

# set trainable=False for once and for all initialization 
def init_once(x):
    return tf.Variable(x, name='permutation', trainable=False)


for i in range(num_layers):
    flow_i = masked_autoregressive_flow(hidden_units=[256,256])
    flow_bijector.append(flow_i) 
    flow_bijector.append(tfb.Permute(init_once(x=np.random.permutation(16).astype('int'))))
# discard the last permute layer 
flow_bijector = tfb.Chain(list(reversed(flow_bijector[:-1])))

#
trainable_dist = tfd.TransformedDistribution(distribution=mvn,
                                            bijector=flow_bijector)

#
def make_samples():
    x = mvn.sample(n_samples)
    samples = [x]
    names = [mvn.name]
    for bijector in reversed(trainable_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)
    return names, samples

names, samples = make_samples()

#
num_epochs = 300
opt = tf.keras.optimizers.Adam(3e-4)
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    if epoch % 100 == 0:
        print("Epoch {}...".format(epoch))
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    for train_batch in X_train:
        with tf.GradientTape() as tape:
            tape.watch(trainable_dist.bijector.trainable_variables)
            loss = -trainable_dist.log_prob(train_batch)
        train_loss(loss)
        grads = tape.gradient(loss, trainable_dist.bijector.trainable_variables)
        opt.apply_gradients(zip(grads, trainable_dist.bijector.trainable_variables))
    train_losses.append(train_loss.result().numpy())
        
    # Validation
    for valid_batch in X_valid:
        loss = -trainable_dist.log_prob(valid_batch)
        val_loss(loss)
    valid_losses.append(val_loss.result().numpy())

#
train_losses = history.history['loss']
valid_losses = history.history['val_loss']

plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Negative log likelihood")
plt.title("Training and validation loss curves")
plt.show()

#
names, samples = make_samples()

#
samples_nf = pd.DataFrame(scaler.inverse_transform(samples[0]).astype(np.int64),columns=X_test.columns)
samples_nf['Risk1Yr'] = y_train

#
g = sns.catplot(x='DGN', y='PRE4', kind="violin", hue='Risk1Yr', data=samples_nf, inner=None, doge=True, palette='Set3', legend_out=True, aspect=2, bw=0.2, ci=95)
sns.swarmplot(x="DGN", y='PRE4', color="k", size=3, data=samples_nf, ax=g.ax)
sns.despine()

#
