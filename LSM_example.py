import numpy as np
import matplotlib.pyplot as plt


N = 4          # number of time steps
r = 0.06       # interest rate
K = 1.1        # strike 
T = 3          # Maturity

dt = T/(N-1)          # time interval
df = np.exp(-r * dt)  # discount factor per time interval

S = np.array([
            [1.00, 1.09, 1.08, 1.34],
            [1.00, 1.16, 1.26, 1.54],
            [1.00, 1.22, 1.07, 1.03],
            [1.00, 0.93, 0.97, 0.92],
            [1.00, 1.11, 1.56, 1.52],
            [1.00, 0.76, 0.77, 0.90],
            [1.00, 0.92, 0.84, 1.01],
            [1.00, 0.88, 1.22, 1.34]])

print('Stock price paths: \n', S)
H = np.maximum(K - S, 0)           # intrinsic values for put option
V = np.zeros_like(H)               # value matrix
V[:,-1] = H[:,-1]
print('Cash-flow matrix at time 3: \n',V)

# Valuation by LS Method
for t in range(N-2, 0, -1):
    
    good_paths = H[:,t] > 0        # paths where the intrinsic value is positive 
                                   # the regression is performed only on these paths 
    
    rg = np.polyfit( S[good_paths, t], V[good_paths, t+1] * df, 2)    # polynomial regression
    C = np.polyval( rg, S[good_paths,t] )                             # evaluation of regression
    print('\n Regression at time', t)
    print('X: ', S[good_paths, t])
    print('Y: ', V[good_paths, t+1] * df)
    print('\nPolynomial regression:')
    print('Conditional expactation function is E[ Y | X ] = ', round(rg[-1], 3), ' + ', round(rg[-2], 3),'* X', ' + ', round(rg[-3], 3), '* X^2' )
    print('\nOptimal early exercise decision at time ',t,' countinuation values: ', C)
    
    exercise = np.zeros( len(good_paths), dtype=bool)    # initialize
    exercise[good_paths] = H[good_paths,t] > C           # paths where it is optimal to exercise
    
    V[exercise,t] = H[exercise,t]                        # set V equal to H where it is optimal to exercise 
    V[exercise,t+1:] = 0                                 # set future cash flows, for that path, equal to zero  
    discount_path = (V[:,t] == 0)                        # paths where we didn't exercise 
    V[discount_path,t] = V[discount_path,t+1] * df       # set V[t] in continuation region
    
V0 = np.mean(V[:,1]) * df  # discounted expectation of V[t=1]

print('\nOption cash flow matrix: \n', V)

print("\n \t\tExample price= ", V0)

