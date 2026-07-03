import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import argparse

####################################################################################################################
## This example calculates S(T)
####################################################################################################################
#DIR = '/Users/ana/Desktop/takarada-ana/'
#sys.path.insert(0, DIR + 'main-programs/')

DIR = 'C:\\Users\\anast\\OneDrive\\Namizje\\takarada-ana\\'
sys.path.insert(0, DIR + 'main-programs')
import takarada_module as module

def main(T_start = 0.03, T_end = 0.06):
        # Parse command line: input_file required, output_file optional
    parser = argparse.ArgumentParser(description="Run takarada module with input and optional output.")
    parser.add_argument('suffix', help='suffix for input and output files')
    
    args = parser.parse_args()
    suffix = args.suffix

    input_file_path = os.path.join(DIR + f'examples/example_temperature/input_{suffix}.json')
    output_file_path = os.path.join(DIR + f'examples/example_temperature/output/output_{suffix}.npz')

    ## (1) Initialize model
    s = module.model(input_file_path)
    mu0 = 0.5*(np.min(s.energije[1]) + np.max(s.energije[0]))

    ## (2) Get T dependence of order parameters, chemical potential, and transport coefficients
    s.run_Tdependence()

    ## (3) Get low-T dependence of order parameters, chemical potential, and transport coefficients
    s.run_lowT_dependence(T_stable=T_start, T_start=T_start, T_end=T_end, )

    ## (4) Save results
    evaluate_transport_DC = s.config['evaluate_transport_DC']
    evaluate_vertex_DC = s.config['evaluate_vertex_DC']
    data = s.collect_data(evaluate_transport_DC=evaluate_transport_DC, evaluate_vertex_DC=evaluate_vertex_DC, merge=False)
    np.savez(output_file_path, **data)

if __name__ == "__main__":
    main()