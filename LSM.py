import numpy as np
import scipy.stats as ss
import pandas as pd
from binomialtree import BinomialTree

def LSM(S0, K, r, sig, T, N=100, paths=50000, order=2):
        """
        Longstaff-Schwartz Method for pricing American options
        
        N = number of time steps
        paths = number of generated paths
        order = order of the polynomial for the regression 
        """
        
        dt = T/(N-1)          # time interval
        df = np.exp(-r * dt)  # discount factor per time time interval
        
        X0 = np.zeros((paths,1))
        increments = ss.norm.rvs(loc=(r-sig**2/2)*dt, scale=np.sqrt(dt)*sig, size=(paths,N-1))
        X = np.concatenate((X0,increments), axis=1).cumsum(1)
        S = S0 * np.exp(X)
        
        H = np.maximum(K - S, 0)   # intrinsic values for put option
        V = np.zeros_like(H)            # value matrix
        V[:,-1] = H[:,-1]

        # Valuation by LS Method
        for t in range(N-2, 0, -1):
            good_paths = H[:,t] > 0    
            rg = np.polyfit( S[good_paths, t], V[good_paths, t+1] * df, 2)    # polynomial regression
            C = np.polyval( rg, S[good_paths,t] )                             # evaluation of regression  
    
            exercise = np.zeros( len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths,t] > C
    
            V[exercise,t] = H[exercise,t]
            V[exercise,t+1:] = 0
            discount_path = (V[:,t] == 0)
            V[discount_path,t] = V[discount_path,t+1] * df
    
        V0 = np.mean(V[:,1]) * df  # 
        return V0, V, H

S0V = np.array([])
sigV = np.array([])
TV = np.array([])
LSMV = np.array([])
putV = np.array([])

print(' \tLSM results:')
for S0 in (36., 38., 40., 42., 44.):
    for sig in (0.2, 0.4):
        for T in (1.0, 2.0):
            S0V = np.append(S0V, S0)
            sigV = np.append(sigV, sig)
            TV = np.append(TV, T)
            putValue, _, _ = LSM(S0, 40, 0.06, sig, T, N = int(T*50))
            LSMV = np.append(LSMV, putValue)
            print("S0 %4.1f | vol %4.2f | T %2.1f | Option Value %8.3f" % (S0, sig, T, putValue))
            

print('\n \tBinomial Model results:')
for S0 in (36., 38., 40., 42., 44.):
    for sig in (0.2, 0.4):
        for T in (1.0, 2.0):
            model = BinomialTree(S0, 0.06, sig, int(T), int(T*1000))
            putValue = model.putprice(40)[0,0]
            putV = np.append(putV, putValue)
            print("S0 %4.1f | vol %4.2f | T %2.1f | Option Value %8.3f" % (S0, sig, T, putValue))
           
diff = putV - LSMV
df = pd.DataFrame(data= {'S':S0V,
                         'Sigma': sigV,
                         'T': TV,
                         'Binom_put': putV,
                         'LSM_put': LSMV,
                         'Difference': diff})

LSM_df = pd.DataFrame()
for i in range(100):
    LSM_arr = np.array([])
    for S0 in (36., 38., 40., 42., 44.):
        for sig in (0.2, 0.4):
            for T in (1.0, 2.0):
                putValue, _, _ = LSM(S0, 40, 0.06, sig, T, N = int(T*50))
                LSM_arr = np.append(LSM_arr, putValue)
    LSM_df[str(i)] = LSM_arr
    
LSM_des = LSM_df.T.describe()

diffi = putV - LSM_des.T['mean']

df1 = pd.DataFrame(data = {'S': S0V[:20], 'Sigma': sigV[:20], 'T': TV[:20], 'Binomial Value': putV[:20], 'LSM Value': LSM_des.T['mean'], 'LSM Std': LSM_des.T['std'], 'Differencia': diffi})