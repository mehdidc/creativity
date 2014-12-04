import matplotlib.pyplot as plt
import numpy as np

def show_representation(representation, nb=None):
    if nb is None:
        nb = int(np.sqrt(representation.shape[0]))
    nb_col = (representation.shape[0])/nb + (1 if (representation.shape[0]%nb!=0 ) else 0)
    for i in xrange(min(nb*nb_col, representation.shape[0])):
        plt.subplot(nb, nb_col, i + 1)
        plt.axis('off')
        plt.imshow(representation[i], cmap='gray')
    plt.show()

def get_representation(model, prev_model_representation=None, imsize=None, top=4):
    if prev_model_representation is None:
        assert imsize is not None
        r = []
        for ihidden in xrange(model.components_.shape[0]):
            r.append(model.components_[ihidden].reshape(imsize).tolist())
        return np.array(r)
    else:
        r = []
        for ihidden in xrange(model.components_.shape[0]):
            weights = model.components_[ihidden]
            indexes = range(weights.shape[0])
            indexes = sorted(indexes, key=lambda i:abs(weights[i]), reverse=True)
            indexes = indexes[0:top]
            weights = weights[indexes]
            weights /= np.sum(weights)
            
            representation =(np.sum(prev_model_representation[indexes] * weights[:, np.newaxis, np.newaxis], 
                                    axis=0) + model.intercept_hidden_[ihidden])
            r.append(representation.tolist())
        return np.array(r)
