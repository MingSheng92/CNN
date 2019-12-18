import numpy as np
from scipy.linalg import svd
# Class PCA for dimension reduction

class PCA(object):
    def __init__(self):
        self.n_samples = 0
        self.n_features = 0
        self.n_components = 0
        self.components = []
        self.explained_variance = []
        self.explained_variance_ratio = []

    def _calc_explained_variance(self, n_components = 1.0):
        # Get variance explained by singular values
        explained_variance = (self.S ** 2) / (self.n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        if( n_components < 1.0 ):
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(ratio_cumsum, n_components) + 1
        else:
            n_components = self.n_features
        
        self.n_components = n_components
        self.components = self.V[:n_components]
        
        if( len(self.explained_variance) == 0):
            self.explained_variance = explained_variance[:n_components]
            self.explained_variance_ratio = explained_variance_ratio[:n_components]
        
        return;
    
    def _sign_correct(self, u,v):
        """
        Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        """
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u,v
    
    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        # Center data
        mean_ = np.mean(X, axis=0)
        X -= mean_

        U, S, V = svd(X, full_matrices=False)
        print(U.shape, S.shape, V.shape)
        # flip eigenvectors' sign to enforce deterministic output
        self.U, self.V = self._sign_correct(U, V)
        self.S = S.copy()
        print(self.S.shape)
        
        self.n_components = self.n_features
        self._calc_explained_variance()  
        
        return self
    
    def fit_components(self, n_components):
        if( n_components < 1.0 ):
            self._calc_explained_variance(n_components)
    
    def transform(self, X):
        X_t = np.dot(X, self.components.T)
        return X_t