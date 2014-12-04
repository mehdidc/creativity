import numpy as np
from utils import sigmoid

def sample(model, v=None, steps=20, nb_samples=10):
  if v is None:
    nb_vis = model.components_.shape[1]
    v = np.random.randint(0, 1, size=(nb_samples, nb_vis))
  
  for i in xrange(steps):
    v = model.gibbs(v)
  return v

def dbn_sample(top_layer_sample, models, only_forward=False):
  sample = top_layer_sample
  for model in reversed(models):
    sample = sigmoid(np.dot(top_layer_sample, model.components_) + model.intercept_visible_)
    if only_forward==False:
      unif = np.random.uniform(size=sample.shape)
      sample = 1.*(unif <= sample)
  return sample
