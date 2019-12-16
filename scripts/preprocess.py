from keras.datasets import fashion_mnist, mnist
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import numpy as np

def load_data(dataset='mnist', reshape=True):
	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	if dataset == 'mnist':
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	elif dataset == 'fashion_mnist':
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
		
	# cast as float 
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	if reshape:
		# reshape loaded dataset back to original image size for later usage
		x_train = x_train.reshape(-1, img_rows, img_cols, 1)
		x_test = x_test.reshape(-1, img_rows, img_cols, 1)
		
	return x_train, y_train, x_test, y_test

def normalize(data):
	# return normalized data
	return (data - np.min(data))/(np.max(data) - np.min(data))

def one_hot(y):
	# get unique labels from numpy array 
	class_label = np.unique(y)
	# get total length of unique class labels
	num_classes = len(class_label)
	# perform one-hot-encoding to the passed data label 
	categorical_label = to_categorical(y, num_classes)
	
	# return the proposeed label 
	return categorical_label