cimport numpy as np
import numpy as np
from tqdm import tqdm


cdef class ElasticNet:
    cdef public float alpha
    cdef public float l1_ratio
    cdef public int max_iter
    cdef public float tol
    cdef public np.ndarray coef_
    cdef public float intercept_

    def __init__(self, alpha=1, l1_ratio=0.5, max_iter=100, tol=0.00000000001):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y):
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        self.coef_ = np.random.rand(n_features)
        self.intercept_ = 0.0

        # Compute Lipschitz constant for efficient gradient descent
        cdef float L = np.max(np.sum(X**2, axis=0)) + self.alpha * self.l1_ratio * n_samples / 2

        # Gradient descent
        cdef int iteration
        cdef np.ndarray[np.float64_t, ndim=1] grad
        cdef float l1_norm
        cdef float l2_norm
        cdef float l1_penalty
        cdef float l2_penalty

        for iteration in tqdm(range(self.max_iter)):
            # Compute gradient
            grad = (1/n_samples) * np.dot(X.T, np.dot(X, self.coef_) - y)
            l1_norm = np.linalg.norm(self.coef_, ord=1)
            l2_norm = np.linalg.norm(self.coef_, ord=2)
            l1_penalty = self.alpha * self.l1_ratio * l1_norm
            l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * l2_norm**2
            grad += (l1_penalty + l2_penalty) * np.sign(self.coef_)

            # Update coefficients
            coef_new = self.coef_ - grad / L
            if np.allclose(coef_new, self.coef_, rtol=self.tol, atol=self.tol):
                tqdm.write("Converged in {} iterations".format(iteration+1))
                break
            else:
                self.coef_ = coef_new
            self.intercept_ -= np.mean(y - np.dot(X, self.coef_))
        

    def predict(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, bint intercept=False):        
        if intercept == False:
            return np.dot(X, self.coef_), (np.mean(((np.dot(X, self.coef_)) - (y))))**2
        else:
            return np.dot(X, self.coef_) + self.intercept_, (np.mean(((np.dot(X, self.coef_)) - (y))))**2
