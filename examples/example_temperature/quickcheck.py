import numpy as np
import matplotlib.pyplot as plt

results = np.load('output.npz', allow_pickle=True)

L12 = results['L12']
L12q = results['L12q']
L11 = results['L11']


T = results['Ts']

L12_boltz = results['L12_boltz']
L11_boltz = results['L11_boltz']
S_boltz = -L12_boltz/L11_boltz/T
S = -L12/L11/T
Sq = -L12q/L11/T

L12_0 = results['L12_0']
L12q_0 = results['L12q_0']
L11_0 = results['L11_0']
L12_corr = results['L12_corr']
L12q_corr = results['L12q_corr']
L11_corr = results['L11_corr']

S0 = -L12_0/L11_0/T
Sq0 = -L12q_0/L11_0/T
S_corr = -L12_corr/L11_corr/T
Sq_corr = -L12q_corr/L11_corr/T


#plt.plot(T, S_boltz)
plt.plot(T, S, color='red')
plt.plot(T, S0, ls='dashed', color='black')
plt.plot(T, S + Sq, color='blue')
plt.plot(T, S0 + Sq0, ls='dashed', color='black')
plt.plot(T, S_corr + Sq_corr, ls='dotted', color='green')

plt.ylim(-3,3)
plt.xlim(0,0.1)
plt.show()