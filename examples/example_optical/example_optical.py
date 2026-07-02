import os, sys, json5
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to obtain optical conductivity with and without vertex corrections, two ways:
## (1) using RPA equations
## (2) performing a tdHF simulation.
####################################################################################################################

print("Which computer are you on? (ana for Apple, anast for Lenovo)")
while True:
    user_input = input("Enter 'ana' or 'anast': ").strip().lower()
    if user_input in ('ana', 'anast'):
        computer = user_input
        break
    print("Invalid input. Please type 'ana' or 'anast' or change path")

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/takarada-ana/'
else:  # ana
    DIR = '/Users/ana/Desktop/takarada-ana/'

print(f"Selected computer: {computer}")
print(f"Directory: {DIR}")

sys.path.insert(0, DIR + 'main-programs/')
import takarada_module as module
import takarada_tokovi as tokovi

input_file = DIR + 'examples/example_optical/input.json5'
print('hello')
####################################################################################################################
## (1) Initialize system, find ground state
s = module.model(input_file)

''' look at effective bands '''
fig, ax = plt.subplots(figsize=(6,4))
K = s.K
E_bare_0 = s.hk0[0,0,:]
E_bare_1 = s.hk0[1,1,:]
E_dressed_minus = s.energije[0]
E_dressed_plus = s.energije[1]
ax.plot(K, E_bare_0, label='bare bands', color='black')
ax.plot(K, E_bare_1, color='black')
ax.plot(K, E_dressed_minus, label='effective bands', color='red')
ax.plot(K, E_dressed_plus, color='red')
ax.set_xlabel(r'$k$', fontsize=15)
ax.set_ylabel(r'$\epsilon_k$', fontsize=15)
fig.suptitle('Band structure')
plt.tight_layout()
plt.show()

## (2) Reach a finite temperature (RPA equations will need a finite beta=1/T)
s.run_Tdependence()

## (3) Optical conductivity two ways, with and without vertex corrections
    ## (3.1) Compute optical conductivity using RPA equations

results = s.optical_response()
omegas0 = results["omegas"]
chi_jj0 = results['chi_jj0']
dchi_jj = results['dchi_jj']
sigma0 = -chi_jj0.imag / omegas0
dsigma = -dchi_jj.imag / omegas0

c1, c2 = 'red', 'blue'
lab1, lab2 = 'frozen', 'dynamic'
lab1_rpa, lab2_rpa = 'bubble', 'bubble + corrections'

fig, ax = plt.subplots(figsize=(6,4))
plt.plot(omegas0, sigma0, color=c1, label=lab1_rpa)
plt.plot(omegas0, sigma0 + dsigma, color=c2, label=lab2_rpa)
plt.xlabel(r'$\omega$', fontsize=15)
plt.ylabel(r'$\sigma$', fontsize=15)
plt.legend(frameon=False, fontsize=13)
plt.title(r'Conductivity with RPA')
plt.legend()
plt.tight_layout()
plt.show()

    ## (3.2) Obtain optical conductivity via tdHF
results_dynamic = s.simulate_perturbation(do_freeze=False)
results_frozen = s.simulate_perturbation(do_freeze=True)
t = results_dynamic['time']                         # same for dynamic, frozen
perturbation = results_dynamic['pulz'].real         # same for dynamic, frozen

norm_dynamic = results_dynamic['norma'].real
norm_frozen = results_frozen['norma'].real

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

# order parameters delta_0 (called delta_b in code) as a function of time (SHIFTED for clarity)
ax[0,0].plot(t, results_dynamic['delta_bs'].real, label=lab2, color=c2)
shift = np.max(results_dynamic['delta_bs'].real) - np.min(results_dynamic['delta_bs'])
ax[0,0].plot(t, shift + results_frozen['delta_bs'].real, label=lab1, color=c1)
ax[0,0].set_ylabel(r'$\text{Re}\Delta_0(t)$', fontsize=15)

# order parameters delta_1 (called delta_c in code) as a function of time (SHIFTED for clarity)
ax[0,1].plot(t, results_dynamic['delta_cs'], label=lab2, color=c2)
shift = np.max(results_dynamic['delta_cs']) - np.min(results_dynamic['delta_cs'])
ax[0,1].plot(t, shift + results_frozen['delta_cs'], label=lab1, color=c1)
ax[0,1].set_ylabel(r'$\text{Re}\Delta_1(t)$', fontsize=15)

# normalization of density matrix
ax[0,2].plot(t, np.abs(norm_frozen - 1.0), label=lab1, color=c1)
ax[0,2].plot(t, np.abs(norm_dynamic - 1.0), label=lab2, color=c2)
ax[0,2].set_ylabel(r'$|\frac{1}{N_k}\sum_k\text{Tr}\rho_k(t) - 1|$')
ax[0,2].set_yscale('log')

for n in range(3):
    ax[0,n].set_xlabel(r'time', fontsize=15)
    ax[0,n].legend(frameon=False, fontsize=13)

# current (frozen is SHIFTED for clarity)
current_frozen = results_frozen['measurement'].real[0]
current_dynamic = results_dynamic['measurement'].real[0]

ax[1,0].plot(t, current_dynamic, label=lab2, color=c2)
ax[1,0].plot(t, np.max(current_dynamic) + current_frozen, label=lab1, color=c1)
ax[1,0].set_xlabel(r'time', fontsize=15)
ax[1,0].set_ylabel(r'$<j(t)>$', fontsize=15)

# optical conductivity (real component)
omegas = results_frozen['omegas']       # same for dynamic, frozen

omega_cut = max(omegas0)                # maximum frequency
eta = 1/ t[-1] * 2                      # damping factor in Fourier transform

omegas, Re_sigma_frozen = tokovi.optical_conductivity(t, current_frozen, perturbation, eta, omega_cut, s.Nk)
omegas, Re_sigma_dynamic = tokovi.optical_conductivity(t, current_dynamic, perturbation, eta, omega_cut, s.Nk)

for j in [1,2]:
    ax[j,1].plot(omegas, Re_sigma_frozen, label=lab1, color=c1)
    ax[j,1].plot(omegas, Re_sigma_dynamic, label=lab2, color=c2)

    ax[j,2].plot(omegas0, sigma0, label=lab1_rpa, color=c1)
    ax[j,2].plot(omegas0, sigma0 + dsigma, label=lab2_rpa, color=c2)

for j in [1,2]:
    for n in [1,2]:
        ax[n,j].set_xlabel(r'$\omega$', fontsize=15)
        ax[n,j].set_ylabel(r'$\sigma$', fontsize=15)

for j in range(3):
    for n in [1,2]:
        if (j,n) != (2,0): ax[n,j].legend(frameon=False, fontsize=13)

ax[1,2].set_title(r'RPA results (for comparison)', fontsize=13)
fig.suptitle(r'tdHF imulation results', fontsize=15)

ax[2,0].axis('off')

ax[2,1].set_ylim(0,1.0)
ax[2,2].set_ylim(0,1.0)

plt.tight_layout()
plt.show()

plt.plot(omegas, Re_sigma_dynamic)
plt.plot(omegas, sigma0 + dsigma)#sss
plt.show()