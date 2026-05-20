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
import takarada_helpers as helpers

input_file = DIR + 'examples/example_temperature/input.json5'
####################################################################################################################

## (1) Initialize model
s = module.model(input_file)

## (2) Get T dependence of order parameters, chemical potential, and transport coefficients
s.run_Tdependence()

s.correction()

Ts = s.merge(s.Ts)
mus = s.merge(s.mus)

plt.figure(figsize=(10, 6))
plt.plot(s.Ts,s.mus, '.-', linewidth=1, markersize=4, alpha=0.6, label='Raw μ(T)')
plt.plot(Ts, mus, color='black')
plt.xlabel('Temperature (eV)', fontsize=12)
plt.ylabel('Chemical Potential (eV)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.ion()
plt.show()