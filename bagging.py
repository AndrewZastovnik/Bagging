import numpy as np
import time

class bagging():
    """
    Parameters
    ----------
    clf : the classification or regression function you wish to run you bagging algorithm on

    bootstraps : the number of bootstrap samples to run

    aggmethod : the method of aggregating the results.
        either voting or average.

    oob : if true estimates and returns the Out Of Bag Error Rate(oober)

    Attributes
    ----------
    bootstraps : a dictionary of the fitted models for the bootstrap samples

    oober : the Out Of Bag Error Rate
    """
    def __init__(self,clf,bootstraps,aggmethod="voting",oober=False):
        self.clf = clf
        self.n = bootstraps
        self.method = aggmethod
        self.oob = oober
        self.bootstraps = {}

    def fit(self,X,y):
        self.start_time = time.time()
        self.elapsed_time = -20
        for i in np.arange(self.n):
            clf1 = sklearn.base.clone(self.clf)
            index = np.random.randint(0, X.shape[0], size=X.shape[0])
            self.bootstraps[i] = clf1.fit(X[index, 0:],y[index])
            if self.oob:
                np.in1d(np.arange(X.shape[0]),index)
            self.timit(i)
        self.labels = np.unique(y)

    def predict(self,X):
            if self.method =="voting":
                pred = np.zeros((X.shape[0],self.labels.shape[0]))
                for i in np.arange(self.n):
                    pred += np.tile(self.bootstraps[i].predict(X).reshape(-1,1),(1,self.labels.shape[0])) ==\
                           np.tile(self.labels, (X.shape[0], 1))
                pred = self.labels[np.argmax(pred,axis = 1)]
            elif self.method == "average":
                pred = np.zeros(X.shape[0])
                for i in np.arange(self.n):
                    pred += self.bootstraps[i].predict(X)/self.n
            else:
                print("Select a valid aggregation method")
            return pred

    def timit(self,i):
        if time.time() - self.elapsed_time - self.start_time > 30:
            self.elapsed_time = (time.time() - self.start_time)
            avg = (self.elapsed_time / i) * (self.n - i)
            print("This has taken ",
                  np.round(self.elapsed_time, 1),
                  "seconds and is expected to take ",
                  np.round(avg, 1),
                  "seconds more ")
