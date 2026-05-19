import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

####################################################################################################################
## This example shows how to calculate optical conductivity, with and without vertex corrections, two ways
####################################################################################################################

''' Add path to programs and parameters '''
computer = 'ana' # if on mac, else 'anast' if on lenovo

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/takarada/repo/takarada-ana/'
elif computer == 'ana':
    DIR = '/Users/ana/Desktop/takarada-ana/'

print('Running on computer: ' + DIR)
sys.path.insert(0, DIR + 'main-programs/')
import takarada_module as module
import takarada_helpers as helpers

input_file = DIR + 'examples/example_optical/input.json'
input_temperature = DIR + 'examples/example_optical/input_temperature.json'
input_perturbation = DIR + 'examples/example_optical/input_perturbation.json'
####################################################################################################################

s = module.model(input_file)
s.run_Tdependence(input_temperature)

s.simulate_perturbation(input_perturbation, )