import os
import numpy as npa
import tensorflow as tf
import os
import pprint

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

###from scripts.loss_funcs import (unhinged, sigmoid, ramp, savage, boot_soft)

'''
All the loss functions for CNN class, import them into one file while investigating some import
issue
'''

def unhinged(y_true, y_pred):
    return K.mean(1. - y_true * y_pred, axis=-1)


def sigmoid(y_true, y_pred):
    beta = 1.0
    return K.mean(K.sigmoid(-beta * y_true * y_pred), axis=-1)


def ramp(y_true, y_pred):
    beta = 1.0
    return K.mean(K.minimum(1., K.maximum(0., 1. - beta * y_true * y_pred)),
                  axis=-1)


def savage(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(1. / K.square(1. + K.exp(2 * y_true * y_pred)),
                  axis=-1)


def boot_soft(y_true, y_pred):
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum((beta * y_true + (1. - beta) * y_pred) *
                  K.log(y_pred), axis=-1)

## CNN class
class CNN(object):
    # initialize class value for later processing purpose
    def __init__(self, batch_size=128, epochs=10, num_classes=10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.input_shape = (28, 28, 1)
        self.CNN_model = None
        self.TPU = False
    
    # function to check if current enviroment has set to TPU 
    def checkEnv(self):
        if 'COLAB_TPU_ADDR' not in os.environ:
            self.TPU = False # the current environment is not connected to TPU
            print('Note : No TPU detected : The model will run on local machine with cpu...')
            print('Upload to Colab and change runtime type to TPU if you want faster training time.\n')
        else:
            self.TPU = True # the current environment is connected to TPU
            tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            print ('TPU address is', tpu_address)

            # print out the list of divices
            with tf.Session(tpu_address) as session:
                devices = session.list_devices()

            print('TPU devices:')
            pprint.pprint(devices)
    
    # num_classes
    def createCNN(self, output_shape, loss='categorical_crossentropy'):
        # run a check if this is connected to a TPU
        self.checkEnv()
        
        # create  CNN with 2 max pool and 3 dropout layer 
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(0.4))
        
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        if self.TPU != False:
			device_name = os.environ['COLAB_TPU_ADDR']
			TPU_ADDRESS = 'grpc://' + device_name
            # convert the model for tpu usage if the env is connected to TPU
            model = tf.contrib.tpu.keras_to_tpu_model(
                model,
				strategy=tf.contrib.tpu.TPUDistributionStrategy(
					tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS))
            )
        
        # compile the final model
        model.compile(
            optimizer = SGD(lr=0.1, decay=0, momentum=0, nesterov=False),
            loss = savage,
            metrics = ['accuracy']
        )
    
        # set the final model
        self.CNN_model = model
      
    def trainByBatch(self, batch_size, x_train, y_train, cv):  
        model_history = [] 
        i = 1
        sss = StratifiedShuffleSplit(n_splits=cv, random_state=0, test_size=0.1)
        train_index, valid_index = next(sss.split(x_train, y_train))
        # set learning rate scheduler 
        learning_rate_scheduler=ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

        # start of cross validation 
        for train_index, valid_index in sss.split(x_train, y_train):
            t_x, t_y = x_train[train_index], y_train[train_index]
            val_x, val_y = x_train[valid_index], y_train[valid_index]
        
            print("\nTraining on Fold: ",i)
            i += 1
            '''model_history.append(self.CNN_model.fit_generator(          
                  self.train_gen(batch_size, t_x, t_y),
                  epochs=self.epochs,
                  steps_per_epoch=100,
                  verbose=0,
                  validation_data=(val_x, val_y),
                  callbacks=[learning_rate_scheduler]
            ))'''
			model_history.append(self.CNN_model.fit(
				  t_x, t_y, 
				  epochs=self.epochs,
                  steps_per_epoch=100,
				  verbose=0,
				  validation_data=(val_x, val_y),
                  callbacks=[learning_rate_scheduler]
			))
   
            print("Val Score: ", self.CNN_model.evaluate(val_x, val_y))
            print("======="*12, end="\n\n\n")
        
        return model_history
      
    def train_gen(self, batch_size, data, label):
        while True:
            offset = np.random.randint(0, data.shape[0] - batch_size)
            yield data[offset:offset+batch_size], label[offset:offset + batch_size]    
