import numpy as np

class knn_classifier():
    '''
    A classifier using k-nearest-neighbor.
    '''
    def __init__(self):
        pass
    
    def train(self, X_train, y_train):
        '''
        Train the classifier
        '''
        self.X = X_train.reshape((X_train.shape[0], -1))
        self.y = y_train
        
    def distance(self, X):
        '''
        Compute the distance between each image in X and each point in self.X
        
        Arguments:
        - X: A numpy array of N * D containing N images, each with D pixels
        
        Return:
        - dists: A numpy array of N * num_train. The dists[i, j] = the L2 distance
              between X[i] and self.X[j]
        '''

        X = X.reshape((X.shape[0], -1))
        return -2 * X.dot(self.X.T) + np.sum(X ** 2, axis=1, keepdims=True) + np.sum(self.X.T ** 2, axis=0, keepdims=True)
        
    
    def predict(self, X, k):
        '''
        Predict the label of images in X
        
        Arguments:
        - X: A numpy array of N * D containing N images, each with D pixels
        - k: An odd number of nearest neighbors
        
        Return:
        - labels: A numpy array of N * 1 containing the labels of N images
        '''
        X = X.reshape((X.shape[0], -1))
        dists = self.distance(X)
        k_closest_y = self.y[dists.argsort()[:, :k]]
        
        labels = []
        for k_votes in k_closest_y:
            appear_time = []
            for i in range(k):
                appear_time.append(np.count_nonzero(k_votes == i))
            labels.append(k_votes[np.argmax(appear_time)])
        return labels