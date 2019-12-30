import numpy as np
import matplotlib.pyplot as plt

class NaiveBayes(object):
    # initialize class value for later processing purpose
    def __init__(self, dataSet, label):
        self.dataSet    = dataSet
        self.label      = label
        self.rowCount, self.n_feature  = dataSet.shape
        self.classCount = len(np.unique(label))
        self.cFreq      = np.ones(self.classCount, dtype=int)
        self.pFreq      = np.ones((self.n_feature, self.classCount), dtype=int )
        self.prior      = self.cFreq / self.rowCount
        self.cond_prob  = np.zeros( (self.n_feature, self.classCount), dtype=float)
        self.iThreshold = 0.1 # pixel intensity threshold
        
    # setter for pixel intensity threshold
    def setIntensityThreshold(self, nValue):
        self.iThreshold = nValue
        
    # calculate class frequency and pixel frequency from training data
    def GenImageFreq(self):
        # reset the following
        self.cFreq      = np.ones(self.classCount, dtype=int)
        self.pFreq      = np.ones((self.n_feature, self.classCount), dtype=int )
        
        # calculate classifier frequency and pixel frequency
        for i in range(self.rowCount):
            # get current class label for current row
            clabel = self.label[i]
            # count the total occurance of each class in the training dataset
            self.cFreq[clabel] = self.cFreq[clabel] + 1
            # loop through all feature in dataset
            for j in range(self.n_feature):
                # if feature(pixel) has higher than 0.1, we will consider this
                # as a part of the image and include in the total pixel 
                # frequency in each class
                if self.dataSet[i, j] > self.iThreshold:
                    self.pFreq[j,clabel] = self.pFreq[j,clabel] + 1
    
    # calculate Posterior mean from training data
    def GenPosteriorMean(self):
        # computing posterior means
        # reset prior of each class 
        self.prior = self.cFreq / self.rowCount
        self.cond_prob  = np.zeros( (self.n_feature, self.classCount), dtype=float)
        
        # loop through all classes to generate posterior mean
        for cls in range(self.classCount):
            # calculate condition probablity based on the pixels freq for each class
            self.cond_prob[:,cls] = self.cond_prob[:,cls] + self.pFreq[:,cls] / self.cFreq[cls]
    
    # calculate posterior probability of an image 
    def posterior(self, pixels):
        np.seterr(divide='ignore', invalid='ignore')
        
        prior_log = np.log(self.prior)
        cond_prob_log = np.log(self.cond_prob)
        one_minus_cond_prob_log = np.log(1 - self.cond_prob)
        
        #given an image array, calculate posterior distrubution
        lp = np.zeros(self.classCount)
        
        for cls in range(self.classCount):
            # class log probability 
            lp[cls] = lp[cls] + prior_log[cls]
            for j in range (self.n_feature):
                if pixels[j] > self.iThreshold:
                    # probability of pixel j in class c 
                    lp[cls] = lp[cls] + cond_prob_log[j,cls]
                else:
                    # probability of pixel j not in class c 
                    lp[cls] = lp[cls] + one_minus_cond_prob_log[j,cls]
        # return the log likelihood of all classes
        return (np.exp(lp) / sum(np.exp(lp)))
    
    #compute argmax for the posterior distrubution and the corresponding probability
    def predict(self, pixels):
        # find the highest probability of the 
        guess = np.argmax(self.posterior(pixels))
        return guess

    # get predictions for the dataset
    def getPredictions(self, dataSet, label):
        counter = 0
        t0 = datetime.now()
        pred = np.zeros(dataSet.shape[0], dtype=int)
        for index in range(dataSet.shape[0]):
            guess = self.predict(dataSet[index])
            try:
                if label[index] == guess:
                    counter = counter + 1
            except Exception:
                pass
            pred[index] = guess
        # calculate accuracy only for those that with label
        accucracy = counter / label.shape[0] * 100
        # round of the accuracy 
        accucracy = np.round(accucracy, 2)
        print("Accuracy : ", accucracy)
        print("Time to compute prediction :", (datetime.now() - t0), " Dataset size:", len(dataSet))
        
        # return the accuracy and predictions
        return accucracy, pred

    def plot_condition_prob(self):
      # plot out the condition probability of each class
      f, axes = plt.subplots(1, self.classCount, figsize=(15,15))
      for c, i in enumerate(axes):
          axes[c].matshow(self.cond_prob[:,c].reshape(28, 28))
          plt.title("Original image", y=-0.15)

      plt.tight_layout()