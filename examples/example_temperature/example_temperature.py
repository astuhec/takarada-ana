import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
import argparse
import json

####################################################################################################################
## This example shows how to run 
####################################################################################################################
DIR = '/Users/ana/Desktop/takarada-ana/'
sys.path.insert(0, DIR + 'main-programs/')
import takarada_module as module


def main():
        # Parse command line: input_file required, output_file optional
    parser = argparse.ArgumentParser(description="Run takarada module with input and optional output.")
    parser.add_argument('input_file', help="Path to input JSON5 file (e.g., input.json5)")
    parser.add_argument('output_file', nargs='?', default=None,
                        help="Optional path to output file (e.g., output.txt)")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    if not os.path.isabs(input_file):
        input_file_path = os.path.join(DIR + 'examples/example_temperature/', input_file)
    else:
        input_file_path = input_file

    ## (1) Initialize model
    s = module.model(input_file_path)
    mu0 = 0.5*(np.min(s.energije[1]) + np.max(s.energije[0]))
    ## (2) Get T dependence of order parameters, chemical potential, and transport coefficients
    s.run_Tdependence()

    #T_start = 0.02
    #T_end = 0.06
    ## (3) Get low-T dependence of order parameters, chemical potential, and transport coefficients
    #s.run_lowT_dependence(T_stable=T_start, T_start=T_start, T_end=T_end, )

    Ts = np.array(s.Ts)
    mus = np.array(s.mus)
    #L11 = np.array(s.L11)
    #L12 = np.array(s.L12)
    #L12q = np.array(s.L12q)
    #L11_boltz = np.array(s.L11_boltz)
    #L12_boltz = np.array(s.L12_boltz)
    #plt.plot(Ts, -L12_boltz/L11_boltz/Ts)
    #plt.plot(Ts, -L12/L11/Ts)
    #plt.plot(Ts, -L12/L11/Ts -L12q/L11/Ts)
    plt.plot(Ts, mus)
    plt.axhline(mu0)

    #plt.plot(s.merge(Ts), s.merge(mus), color='black', ls='dashed')
    #plt.axvline(Ts[s.stable_index])

    plt.show()

    data = s.collect_data(evaluate_transport_DC=True, evaluate_vertex_DC=True, merge=True)
    np.savez(output_file + '.npz', **data)
    
if __name__ == "__main__":
    main()