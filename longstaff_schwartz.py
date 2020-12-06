import numpy as np
import scipy.stats as ss

class Longstaff_Schwartz:
    def __init__(self, S0, K, r, sigma, T, N = 100, paths = 50000, order = 2):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.paths = paths
        self.order = order
        self.dt = self.T /(self.N - 1)
        self.df = np.exp(-self.r * self.dt)
        
    def gbm(self):
        X0 = np.zeros((self.paths, 1))
        increments = ss.norm.rvs(loc=(self.r - self.sigma**2/2)*self.dt, scale = np.sqrt(self.dt)*self.sigma, size = (self.paths, self.N - 1))
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)
        return S
    
    def putprice(self):
        S = self.gbm()
        H = np.maximum(self.K - S, 0)
        V = np.zeros_like(H)
        V[:, -1] = H[:, -1]
        
        for t in range(self.N - 2, 0, -1):
            good_paths = H[:, t] > 0
            rg = np.polyfit(S[good_paths, t], V[good_paths, t+1] * self.df, self.order)
            C = np.polyval(rg, S[good_paths, t])
            
            exercise = np.zeros(len(good_paths), dtype = bool)
            exercise[good_paths] = H[good_paths, t] > C
            
            V[exercise, t] = H[exercise, t]
            V[exercise, t+1:] = 0
            discount_path = (V[:, t] == 0)
            V[discount_path, t] = V[discount_path, t+1] * self.df
            
        V0 = np.mean(V[:, 1]) * self.df
        return V0, V, H
        
if __name__ == '__main__':
    S0, K, r, sigma, T = 36, 40, 0.06, 0.4, 2
    model = Longstaff_Schwartz(S0, K, r, sigma, T)
    putValue, _, _ = model.putprice()
    print('Price of an American Option: ', round(putValue, 3))