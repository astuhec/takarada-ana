import os, sys, json5
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to run 
####################################################################################################################

#''' Add path to programs and parameters '''
#computer = 'ana' # if on mac, else 'anast' if on lenovo

print("Which computer are you on? (ana for Apple, anast for Lenovo)")

while True:
    user_input = input("Enter 'ana' or 'anast': ").strip().lower()
    if user_input in ('ana', 'anast'):
        computer = user_input
        break
    print("Invalid input. Please type 'ana' or 'anast'.")

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
####################################################################################################################

s = module.model(input_file)
s.run_Tdependence()


results = s.optical_response()
omegas = results["omegas"]
chi_jj0 = results['chi_jj0']
dchi_jj = results['dchi_jj']
sigma0 = -chi_jj0.imag / omegas
dsigma = -dchi_jj.imag / omegas
plt.plot(omegas, sigma0)
plt.plot(omegas, sigma0 + dsigma)
'''
results_dynamic = s.simulate_perturbation(do_freeze=False)
results_frozen = s.simulate_perturbation(do_freeze=True)
t = results_dynamic['time']                 # same for dynamic, frozen
perturbation = results_dynamic['pulz']      # same for dynamic, frozen

norm_dynamic = results_dynamic['norma'].real
norm_frozen = results_frozen['norma'].real

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12,10))

# order parameters delta_0 as a function of time
ax[0,0].plot(t, results_frozen['delta_bs'], label='HF frozen')
ax[0,0].plot(t, results_dynamic['delta_bs'], label='HF dynamic')

# order parameters delta_1 as a function of time
ax[0,1].plot(t, results_frozen['delta_cs'], label='HF frozen')
ax[0,1].plot(t, results_dynamic['delta_cs'], label='HF dynamic')

# normalization of density matrix
ax[0,2].plot(t, norm_frozen)
ax[0,2].plot(t, norm_dynamic)

# current
current_frozen = results_frozen['measurement'].real[0]
current_dynamic = results_dynamic['measurement'].real[0]

ax[1,0].plot(t, current_frozen, label='HF frozen')
ax[1,0].plot(t, current_dynamic, label='HF dynamic')


# optical conductivity (real component)
omegas = results_frozen['omegas']       # same for dynamic, frozen
Re_sigma_frozen = results_frozen['Re_sigma']
Re_sigma_dynamic = results_dynamic['Re_sigma']
ax[1,1].plot(omegas, Re_sigma_frozen)
ax[1,1].plot(omegas, Re_sigma_dynamic)
'''
plt.show()