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

    ## (2) Get T dependence of order parameters, chemical potential, and transport coefficients
    s.run_Tdependence()

    ## (3) Get low-T dependence of order parameters, chemical potential, and transport coefficients
    s.run_lowT_dependence()

    with open(input_file_path, 'r') as f:
        data = json.load(f)
    eval_DC=True if data['evaluate_transport_DC'] == 1 else False
    eval_vertex=True if data['evaluate_vertex_DC'] == 1 else False

    data = s.collect_data(evaluate_transport_DC=eval_DC, evaluate_vertex_DC=eval_vertex)
    np.savez(output_file + '.npz', **data)

if __name__ == "__main__":
    main()