import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import tensorflow as tf
import time
from tensorflow import keras

from lstm_network import LSTM_network

### from lstm_network_update import LSTM_network

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout

import nsl_kdd_processed_data as norm_data


    ## Get the normalized  NSL KDD dataset into x-axis and y-axis from nsl_kdd_processed_data.py file
    ## x_tr = training data set into x-axis
    ## x_te = test set into x-axis
    ## y_tr = training set into y-axis
    ## y_te = test set into y-axis

x_tr, x_te, y_tr, y_te = norm_data.get_data()

def get_my_model(units, embedding_dim, n_classes):
    
    batch_size = embedding_dim
    epochs=1200

    model = Sequential()
    model.add(Bidirectional(LSTM(units), input_shape=(None, embedding_dim)))
    model.add(Dropout(.4))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])
    
    history =  model.fit(x_tr, y_tr, batch_size=batch_size, epochs = epochs, validation_data=(x_te, y_te) )
    loss, accuracy = model.evaluate(x_te, y_te)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

    return model


def test_forwrad_pass():
    # just something
    T, n_hidden, n_embedding, n_classes, batch_size, total = 10, 64, 122, 5, 122, 125973
    orig_model = get_my_model(n_hidden, n_embedding, n_classes)
    print(orig_model.summary())
    
    num_wts = orig_model.get_weights()

    net = LSTM_network(n_hidden, n_embedding, n_classes, orig_model.get_weights())
    
    input_keras = x_tr
    net_output = np.vstack([net.full_pass(input_keras[i:i + batch_size])[0] for i in range(0, total, batch_size)])
    model_output = orig_model.predict(input_keras, batch_size=batch_size)
    

    # Saving The model 
    orig_model.save('trained_model_lstm_LRP')
    reconstructed_model = keras.models.load_model("trained_model_lstm_LRP")
    # Check the loaded Model
    np.testing.assert_allclose(orig_model.predict(input_keras), reconstructed_model.predict(input_keras))

    res = np.allclose(net_output, model_output, atol=1e-6)
    np.set_printoptions(precision=5)
    if res:
        print('Forward pass of model is correct!')
    else:
        diff = np.sum(np.abs(net_output-model_output))
        print('Error in forward pass. Total abs difference : {}'.format(diff))


def test_lrp():
    T, n_hidden, n_embedding, n_classes, batch_size = 10, 64, 122, 5, 122
    eps = 0.
    bias_factor = 1.0
    debug = False
    np.random.seed(42)
    
    net = LSTM_network(n_hidden, n_embedding, n_classes, debug=debug)
    
    start_index = 0
    end_index = 13000
    number_of_item = 13000
    for x in range(10):
        
        end_index = (x+1) * number_of_item

        if x==9:
            input = x_tr[start_index:]
        else:
            input = x_tr[start_index:end_index]
        
        Rx, rest = net.lrp(input, eps=eps, bias_factor=bias_factor)
        
        """ print("The result and Shape of Rx")
        print(Rx)
        print(Rx.shape)

        print("The result of rest: ")
        print(rest)
        """ 

        R_in, R_out = (tf.reduce_sum(tf.reduce_max(net.y_hat, axis=1)).numpy(), tf.reduce_sum(Rx).numpy() + tf.reduce_sum(rest).numpy())
        
        if np.isclose(R_in, R_out):
            print('LRP pass is correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))
        else:
            print('LRP pass is not correct: Relevance in: {0:.5f}, Relevance out: {1:.5f}'.format(R_in, R_out))

        ## change the starting value for the next cycle
        start_index = end_index


def test_runtime():
    T, n_hidden, n_embedding, n_classes, batch_size = 20, 64, 122, 5, 122
    n_samples = 2500
    eps = 1e-3
    bias_factor = 0.
    
    net = LSTM_network(n_hidden, n_embedding, n_classes) 
    input = x_tr

    start = time.time()
    for i in tqdm(range(0, n_samples, batch_size)):
        net.lrp(input[i:i+batch_size], eps=eps, bias_factor=bias_factor)
    end = time.time()
    total = end-start
    per_sample = total / n_samples
    print('Processing {} samples took {} s. {} seconds per sample'.format(n_samples, total, per_sample))


if __name__ == '__main__':
    test_forwrad_pass()
    test_lrp()
    test_runtime()
