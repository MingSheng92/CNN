import numpy as np
from sklearn.model_selection import KFold

def softmax(Z):
    #Z = softmax(X.dot(self.W))
    #z1 = np.add(Z, -Z.max(axis=0))
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def get_accuracy(y_pre,y):
    count = y_pre == y
    accuracy = count.sum()/len(count)
    return accuracy
    
class LogisticRegression(object):
    # initialize class value for later processing purpose
    def __init__(self, dataset, label):
        self.num_inputs  = dataset.shape[1]
        self.num_classes = len(set(label))
        self.X           = dataset
        self.y           = label
        self.W           = np.random.randn(self.num_inputs, self.num_classes)
        self.b           = np.zeros([self.num_classes,1], dtype=float)

    # calculate gradient 
    def softmax_grad(self, X, y):
        A = softmax(X.dot(self.W))     # shape of (N, C)
        id0 = range(X.shape[0])  # number of train data
        A[id0, y] -= 1           # A - Y, shape of (N, C)
        return X.T.dot(A)/X.shape[0] 

    # cost or loss function  
    # removed Transition matrix dot here because 
    def softmax_loss(self):
        A = softmax(self.X.dot(self.W)) 
        id0  = range(self.X.shape[0])
        loss = -np.mean(np.log(A[id0, self.y]))
        
        return loss
    
    def eval(self, val_x, val_y):
        # calculate loss
        A = softmax(val_x.dot(self.W))
        id0 = range(val_x.shape[0])
        val_loss = -np.mean(np.log(A[id0, val_y]))
        
        # calculate accuracy 
        y_pred  = self.predict(val_x)
        val_acc = get_accuracy(y_pred, val_y)
        
        return val_loss, val_acc
        
    # train softmax logistic regression
    def train(self, train_x, train_y, val_x, val_y, lr = 0.01, n_epoches = 150, tol = 1e-5, batch_size = 10):
        # keep a copy of weights to for weight update later
        W_old = self.W.copy()
        ep = 0 
        # store history of loss
        loss_hist = [self.softmax_loss()] 
        #loss_hist = []
        N = train_x.shape[0]
        nbatches = int(np.ceil(float(N)/batch_size))
        while ep < n_epoches: 
            ep += 1 
            mix_ids = np.random.permutation(N) # mix data 
            
            # run by batch
            for i in range(nbatches):
                # get the i-th batch
                batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)] 
                X_batch, y_batch = train_x[batch_ids], train_y[batch_ids]
                self.W -= lr * self.softmax_grad(X_batch, y_batch)
                
            # evaluate current model
            if ep % 10 == 0 or ep == 1:
                val_loss, val_acc = self.eval(val_x, val_y)
                message = 'Epoch %d, val Loss %.4f, val Acc %.4f' % (ep, val_loss, val_acc)
                print(message)
            
            # append history
            loss_hist.append(self.softmax_loss())
            
            # stop the looping process if the improvement rate is too low
            if np.linalg.norm(self.W - W_old)/self.W.size < tol:
                print('reached tolerance level.')
                break 
                
            # update previous W to new W for next interation
            W_old = self.W.copy()

        return loss_hist 
    
    # predict function
    def predict(self, X):
        A = softmax(X.dot(self.W))
        return np.argmax(A, axis = 1)
    
    # return probability of classes
    def predict_proba(self, X):
        A = softmax(X.dot(self.W))
        return A 
    
    def cross_fold(self, lr=0.01, K=10, n_epoches=50):
        # create K fold on the current dataset
        fold_count = 1
        k_fold = KFold(n_splits=K, random_state=None, shuffle=False)

        # perform K-fold cv
        for train_idx, val_idx in k_fold.split(self.X):
            print("Fold :", fold_count)
            self.train(self.X[train_idx], self.y[train_idx], self.X[val_idx], self.y[val_idx],
                       lr, n_epoches, tol = 1e-5, batch_size = 300)
            print("--------------------------------------------------------")
            fold_count += 1