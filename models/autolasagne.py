
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

from lasagne import easy
import lasagne
import lasagne.data
import theano.tensor as T
import theano
import os
import numpy as np
import time



from lasagne import easy
import copy
import lasagne
import lasagne.data
import theano.tensor as T
import theano
import os
import numpy as np
import time

from lasagne.layers.dense import DenseLayer
from lasagne.layers.base import Layer
import lasagne.layers
from pylearn2.scripts import plot_weights


#get_ipython().magic(u'matplotlib inline')


# In[80]:



# In[81]:

def build_model(x_dim, y_dim, num_hidden):
    l_x_in = lasagne.layers.InputLayer(
            shape=(None, x_dim),
        )
    l_y_in = lasagne.layers.InputLayer(
            shape=(None, y_dim),
        )


    l_hidden_x_intermediate = lasagne.layers.DenseLayer(
        l_x_in,
        num_units=num_hidden,
        nonlinearity=lasagne.nonlinearities.rectify,
    )


    l_hidden_x = lasagne.layers.DenseLayer(
        l_hidden_x_intermediate,
        num_units=num_hidden,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_x_hat = lasagne.layers.DenseLayer(
        l_hidden_x,
        num_units=x_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,
    )
    
    l_y_hat = lasagne.layers.DenseLayer(
        l_hidden_x,
        num_units = y_dim,
        nonlinearity=lasagne.nonlinearities.tanh
    )

    return l_x_in, l_y_in, l_hidden_x, l_x_hat, l_y_hat

def objective(X_batch, y_batch, x_hat, y_hat):
   # y_err =  - T.mean(T.log(y_hat)[T.arange(y_batch.shape[0]), y_batch])
    y_err = T.mean((y_batch - y_hat)**2)
    #y_err = T.mean(y_batch)**2
    
    
    x_err = -T.mean(X_batch * T.log(x_hat) + (1 - X_batch) * T.log(1 - x_hat))
    #x_err = T.mean((X_batch - x_hat)**2) 
    return x_err + y_err


def corrupted(rng, x, corruption_level):
    return rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

# In[97]:
from theano.sandbox.rng_mrg import MRG_RandomStreams

class Experiment(lasagne.easy.Experiment):

    def __init__(self):
        pass

    def load_data(self):
        datasets = lasagne.data.Mnist()
        datasets.load(ratio_valid=0.2)

        for d in datasets.values():
            d.y = theano.shared(lasagne.utils.floatX(lasagne.easy.to_hamming(d.y.get_value(), presence=1, absence=-1)))
        self.train, self.valid, self.test =  datasets["train"], datasets["valid"], datasets["test"]
    
    def set_hp(self):
        
        self.learning_rate, self.momentum = 1., 0.8

        self.input_dim = self.train.X.get_value().shape[1]
        self.output_dim  = 10
        self.batch_size = 128
        self.x_hidden_units = 200
        self.nb_batches = self.train.X.get_value().shape[0] // self.batch_size
        self.nb_epochs = 15
        self.corruption = 0.4
        self.rng = MRG_RandomStreams(2014*5 +  27)
   
    def build_model(self):
        
        self.batch_index, self.X_batch, self.y_batch, self.batch_slice = easy.get_theano_batch_variables(self.batch_size, 
                                                                                                         y_softmax=False)
        

        # build model
        self.l_x_in, self.l_y_in, self.l_hidden_x, self.l_x_hat, self.l_y_hat = build_model(self.input_dim, self.output_dim, self.x_hidden_units)
        self.corrupted_X_batch = corrupted(self.rng, self.X_batch, self.corruption)

        # get the loss
        self.loss = objective(self.X_batch,
                              self.y_batch,
                              self.l_x_hat.get_output(self.corrupted_X_batch),
                              self.l_y_hat.get_output({self.l_x_in : self.corrupted_X_batch, self.l_y_in : self.y_batch}))
        self.get_loss = theano.function([self.X_batch, self.y_batch], self.loss)
        self.get_reconstruction = theano.function([self.X_batch],
                                                   self.l_x_hat.get_output(self.X_batch))
        self.accuracy = lasagne.easy.get_accuracy(self.l_y_hat.get_output({self.l_x_in : self.X_batch, self.l_y_in : self.y_batch}),
                                                  self.y_batch, softmax=False)

        self.get_accuracy = theano.function([self.X_batch, self.y_batch], self.accuracy)

        from theano.sandbox.rng_mrg import MRG_RandomStreams
        
        random_stream = MRG_RandomStreams(2014 * 5 + 27)
        X_batch_prime = T.matrix('xprime')

        """
        nb = T.iscalar('nb')
        weights = (random_stream.uniform(low=0, high=1, size=(X_batch_prime.shape[0], nb) ) *
                   random_stream.binomial(n=1, p=0.0001, size=(X_batch_prime.shape[0], nb)) )
        weights = weights / weights.sum(axis=0).reshape( (1, nb) )
        self.generate = theano.function([X_batch_prime, nb],
            self.l_x_hat.get_output({self.l_hidden_x : T.dot(self.l_hidden_x.get_output(X_batch_prime).T, weights).T})
        )
        """

        # get the gradient updates
        all_params = lasagne.layers.get_all_params(self.l_x_hat)
        #updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate, momentum)
        self.updates = lasagne.updates.adadelta(self.loss, all_params, learning_rate=self.learning_rate)
        #updates = lasagne.updates.Adam(loss, all_params, learning_rate=learning_rate)

        # get the iteration update
        self.iter_update_batch = easy.get_iter_update_supervision(self.train.X, self.train.y, self.X_batch, self.y_batch,
                                                                  self.loss, self.updates,
                                                                  self.batch_index, self.batch_slice)
        
        
    def run(self):
        # Load data
        train, valid, test = self.train, self.valid, self.test
        self.iter = 0
        def iter_update():
            global iter
            for i in xrange(self.nb_batches):
                self.iter_update_batch(i)


            stats = {}    
            for label, data in zip(("train", "valid", "test"), (self.train, self.valid, self.test)):
                loss = self.get_loss(data.X.get_value(), data.y.get_value())
                accuracy = self.get_accuracy(data.X.get_value(), data.y.get_value())
                stats["loss_%s" % (label,) ] = loss
                stats["accuracy_%s" % (label,)] = accuracy
            
            self.iter += 1
            return stats

        def quitter(update_status):
            return False

        def monitor(update_status):
            return update_status

        def observer(monitor_output):
            for k, v in monitor_output.items():
                print("%s : %f" % (k, v))

        lasagne.easy.main_loop(self.nb_epochs, iter_update, quitter, monitor, observer)

    def post(self):

        import matplotlib.pyplot as plt
        
        # reconstruction
        shape = (28, 28)

        x_rec = self.get_reconstruction(self.test.X.get_value())
        
        nb = 20
        k = 1
        for i in xrange(nb):

            ind = np.random.randint(0, self.test.X.get_value().shape[0] - 1)
            x_i = self.test.X.get_value()[ind].reshape(shape)
            x_rec_i = x_rec[ind].reshape(shape)
            plt.axis('Off')
            plt.subplot(nb, 2, k)
            plt.imshow(x_i, cmap='gray')
            k += 1
            plt.axis('off')
            plt.subplot(nb, 2, k)
            plt.imshow(x_rec_i, cmap='gray')
            k += 1
        plt.savefig("mnist-rec.png")
        
        # weights

        layers = lasagne.layers.helper.get_all_layers(self.l_x_hat)
        layers = filter(lambda l:hasattr(l, 'W'), layers)
        weights = [l.W.get_value() for l in layers]
        vis = plot_weights.build_visualizations_by_weighted_combinations(weights, until_layer=len(weights) - 1, top=0.001)

        for i in xrange(len(layers)):
            plt.clf()
            plt.axis('off')
            plot_weights.grid_plot(vis[i], imshow_options={"cmap": "gray"})
            plt.savefig("l%d.png" % (i,))
        #plt.show()
        
        # generation
        """
        from pylearn2.scripts.plot_weights import grid_plot
        x_gen = self.generate(train.X.get_value(), 30)
        x_gen = x_gen.reshape( (x_gen.shape[0], shape[0], shape[1])  )
        grid_plot(x_gen, imshow_options={"cmap": "gray"})

        plt.savefig("mnist-gen.png")

        fd = open("model.pkl", "w")
        self.save(fd)
        fd.close()
        """



if __name__ == "__main__":
    # In[98]:

    e = Experiment()


    # In[99]:

    e.load_data()


    # In[100]:

    e.set_hp()


    # In[101]:

    e.build_model()


    # In[102]:

    e.run()


    # In[103]:

    e.post()

