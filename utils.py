#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
#import _pickle as cPickle
import pickle
import logging

import sys;
sys.setrecursionlimit(40000)

def init_log(filename):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.level = logging.DEBUG

    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

# set use gpu programatically
import theano.sandbox.cuda
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def norm_weight(shape, scale=0.01):
    return scale * numpy.random.randn(shape)

def ortho_weight(n):
    W = np.random.randn(n, n)
    u, s, v = np.linalg.svd(W)
    return u

def np_init_weights(shape):
 return floatX(np.random.randn(*shape))

def xavier_weight(shape):
    return np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)

def init_weights(shape, name, sample = "xavier", scale = 0.01):
    if sample == "norm":
        values = norm_weight(shape, scale)
    elif sample == "xavier":
        values = xavier_weight(shape)
    elif sample == "ortho":
        values = ortho_weight(shape[0])
    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)
    return theano.shared(floatX(values), name)

def init_weights_2(shape, name, sample = "xavier", scale = 0.01, couple_axis = 1):
    if couple_axis in [0, 1]:
        if sample == "norm":
            values = np.concatenate([norm_weight(shape, scale),
                                     norm_weight(shape, scale)], couple_axis)
        elif sample == "xavier":
            values = np.concatenate([xavier_weight(shape),
                                     xavier_weight(shape)], couple_axis)
        elif sample == "ortho":
            values = np.concatenate([ortho_weight(shape[0]),
                                     ortho_weight(shape[0])], couple_axis)
        else:
            raise ValueError("Unsupported initialization scheme: %s" % sample)
    return theano.shared(floatX(values), name)

def init_weights_4(shape, name, sample = "xavier", scale = 0.01, couple_axis = 1):
    if couple_axis in [0, 1]:
        if sample == "norm":
            values = np.concatenate([norm_weight(shape, scale),
                                     norm_weight(shape, scale),
                                     norm_weight(shape, scale),
                                     norm_weight(shape, scale)], couple_axis)
        elif sample == "xavier":
            values = np.concatenate([xavier_weight(shape),
                                     xavier_weight(shape),
                                     xavier_weight(shape),
                                     xavier_weight(shape)], couple_axis)
        elif sample == "ortho":
            values = np.concatenate([ortho_weight(shape[0]),
                                     ortho_weight(shape[0]),
                                     ortho_weight(shape[0]),
                                     ortho_weight(shape[0])], couple_axis)
        else:
            raise ValueError("Unsupported initialization scheme: %s" % sample)
    return theano.shared(floatX(values), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size,))), name)

def save_function(f, func):
    pickle.dump(func, open(f, "wb"))

def load_function(f):
    func = pickle.load(open(f, "rb"))
    return func

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
        pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    if sys.version_info[0] < 3:
        ps = pickle.load(open(f, "rb"))
    else:
        u = pickle._Unpickler(open(f, "rb"))
        u.encoding = 'latin1'
        ps = u.load()
    for p in model.params:
        p.set_value(ps[p.name])
    return model