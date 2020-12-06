import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class BinomialTree:
    def __init__(self, S0, r, sigma, T, n):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n = n
        self.dt = self.T / self.n
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1./self.u
        self.p = (np.exp(self.r *self.dt) - self.d) / (self.u - self.d)
        
    def buildtree(self):
        S = np.zeros((self.n + 1, self.n + 1))
        S[0, 0] = self.S0
        for i in range(1, self.n + 1):
            S[i, 0] = S[i - 1, 0] * self.u
            for j in range(1, i + 1):
                S[i, j] = S[i - 1, j - 1] * self.d
        return S
    
    def tree2matrix(self):
        a = np.array(list(product((0, 1), repeat = self.n)))
        a = np.c_[[0] * 2 ** self.n, a]
        a = a.cumsum(axis = 1)
        return np.choose(a, self.buildtree().T)
    
    def putprice(self, K):
        putvalue = np.zeros((self.n + 1, self.n + 1))
        tree = self.buildtree()
        putvalue[self.n, :] = np.maximum(0, K - tree[self.n, :])
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                putvalue[i, j] = max(0, K - tree[i, j], np.exp(-self.r * self.dt)*(self.p*putvalue[i+1, j] + (1 - self.p)*putvalue[i+1, j+1]))
        return putvalue

if __name__ == '__main__':
    print('This is a code for generate binomial tree and covert the paths to matrix')
    print("Let's see a basic example: ")
    S0, r, sigma, T, n = 36, 0.06, 0.2, 2, 15
    model = BinomialTree(S0, r, sigma, T, n)
    print('Now we are going to build a binomail tree:')
    tree = model.buildtree()
    print('Now we are going to convert to trajectories')
    trajectories = model.tree2matrix()
    plt.plot(trajectories.T)
    plt.show()
    print("let's calculate the american put option price from binomial tree")
    print('The American Put Option price is: ', model.putprice(40)[0,0])
