import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to run 
####################################################################################################################

''' Add path to programs and parameters '''
computer = 'anast' # if on mac, else 'anast' if on lenovo

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/takarada-ana/'
elif computer == 'ana':
    DIR = '/Users/ana/Desktop/takarada-ana/'

print('Running on computer: ' + DIR)
sys.path.insert(0, DIR + 'main-programs/')
import takarada_module as module
import takarada_helpers as helpers

input_file = DIR + 'examples/example_temperature/input.json'
input_temperature = DIR + 'examples/example_temperature/input_temperature.json'
####################################################################################################################

s = module.model(input_file)
mu0 = 0.5 * (np.min(s.energije[1]) + np.max(s.energije[0]))
s.run_Tdependence(input_temperature)

Nbeta_intial = 200
scale_initial = 1/1.005
s.correction(input_temperature, Nbeta_intial, scale_initial, safety=np.argmin(np.abs(s.mus - mu0)))

Ts = s.merge(s.Ts)
mus = s.merge(s.mus)

plt.figure(figsize=(10, 6))
plt.plot(s.Ts,s.mus, '.-', linewidth=1, markersize=4, alpha=0.6, label='Raw μ(T)')
plt.plot(Ts, mus, color='black')
plt.axhline(mu0, color='r', linestyle='--', alpha=0.7, label=f'μ_GS={mu0:.4f}')
plt.xlabel('Temperature (eV)', fontsize=12)
plt.ylabel('Chemical Potential (eV)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.ion()
plt.show()