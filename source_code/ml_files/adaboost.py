from sklearn.tree import DecisionTreeRegressor
import numpy as np
# from lovely_numpy import lo


class AdaBoostR2:
    def __init__(self,X,y, n_estimators=100, loss_function='linear', max_depth=4,random_state=None):
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.max_depth = max_depth
        self.estimators = []
        self.N, self.D = X.shape
        self.out_dim = y.shape[1]
        self.fitted_values = np.empty((self.N,self.n_estimators,self.out_dim))
        self.betas = []
        ## weights are initialized to 1/num_samples
        self.weights = np.full(self.N, 1 / self.N)
        self.is_fitted = False
        np.random.seed(random_state)

    def get_loss(self, y_pred):
        if self.loss_function == 'linear':
            return self.linear_loss(y_pred)
        elif self.loss_function == 'square':
            return self.square_loss(y_pred)
        elif self.loss_function == 'exponential':
            return self.exponential_loss(y_pred)

    def linear_loss(self, y_pred):
        return np.mean(np.abs(self.y - y_pred))
    
    def square_loss(self, y_pred):
        return np.mean(np.square(self.y - y_pred))

    def exponential_loss(self, y_pred):
        return 1 - np.mean(np.exp(self.y - y_pred))

    def weighted_median(self,values):    
        num_samples = values.shape[0]

        ## Sort the predictions
        sorted_indices = values.argsort(axis=1)
        values = values[sorted_indices]
        weights = self.model_weights[sorted_indices]

        ## Find index of median prediction for each sample
        weights_cumulative_sum = weights.cumsum(axis=1)
        median_or_above = weights_cumulative_sum > 0.5 * weights_cumulative_sum[:, -1][:,np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_indices[np.arange(num_samples), median_idx]
        ## Return median predictions

        return values[np.arange(num_samples), median_estimators]

    def weighted_mean(self,values):    
        return np.dot(self.model_weights, values)

    # get id of leaf a sample is in
    def apply(self, X):
        leafes = []
        for x in X:
        # for x in self.X:
            leafes.append([t.apply(x.reshape(1, -1)) for t in self.estimators])
        return np.array(leafes)

    def fit(self):
        for t in range(self.n_estimators):
            ## fit tree with sample weights and get predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(self.X, self.y, sample_weight=self.weights)
            self.estimators.append(tree)
            y_pred = tree.predict(self.X)
            self.fitted_values[:,t] = y_pred
            
            ## Calculate observation error
            loss = self.get_loss(y_pred)
            D_t = np.max(loss)
            L_ts = loss/D_t

            ## Calculate model error, break if error >= .5
            L_bar_t = np.sum(loss * self.weights)
            if L_bar_t >= 0.5:
                self.n_estimators = t-1
                self.fitted_values = self.fitted_values[:,:t-1]
                self.estimators = self.estimators[:t-1]
                break
            
            ## Calculate beta
            beta_t = L_bar_t/(1-L_bar_t)
            self.betas.append(beta_t)

            ## Update weights
            Z_t = np.sum(self.weights*beta_t**(1-L_ts))
            self.weights *= beta_t**(1-L_ts)/Z_t
        
        ## Get median 
        print("Shape of fitted values: ", self.fitted_values.shape)
        self.model_weights = np.log(1/np.array(self.betas))
        # self.y_train_hat = np.array([self.weighted_median(self.fitted_values[n]) for n in range(self.N)])
        self.y_train_hat = np.array([self.weighted_mean(self.fitted_values[n]) for n in range(self.N)])
        self.is_fitted = True

    ## The predicted regression value of an input sample is computed as the weighted median prediction of the regressors in the ensemble.
    def predict(self, X_test):
        if self.is_fitted:
            N_test = len(X_test)
            fitted_values = np.empty((N_test, self.n_estimators,self.out_dim))
            for t, tree in enumerate(self.estimators):
                fitted_values[:,t] = tree.predict(X_test)
            # return np.array([self.weighted_median(fitted_values[n]) for n in range(N_test)]) 
            return np.array([self.weighted_mean(fitted_values[n]) for n in range(N_test)])
        else:
            raise Exception('Model not fitted yet')