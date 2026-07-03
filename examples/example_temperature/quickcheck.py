import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(Vc, tperp):
    suffix = f'Vc{Vc}_tperp{tperp}'
    results = np.load(f'output/output_{suffix}.npz')

    T = results['Ts']
    mu = results['mus']

    '''    L12_boltz = results['L12_boltz']
    L11_boltz = results['L11_boltz']

    L12 = results['L12']
    L12q = results['L12q']
    L11 = results['L11']

    L12_0 = results['L12_0']
    L12q_0 = results['L12q_0']
    L11_0 = results['L11_0']
    L12_corr = results['L12_corr']
    L12q_corr = results['L12q_corr']
    L11_corr = results['L11_corr']

    S_boltz = -L12_boltz/L11_boltz/T
    S = -L12/L11/T
    Sq = -L12q/L11/T
    S0 = -L12_0/L11_0/T
    Sq0 = -L12q_0/L11_0/T
    S_corr = -L12_corr/L11_corr/T
    Sq_corr = -L12q_corr/L11_corr/T

    fig, ax = plt.subplots(ncols=2,figsize=(8,4))

    ax[1].plot(T, S, color='red')
    ax[1].plot(T, S0, color='maroon', ls='dashed', lw=2)

    ax[1].plot(T, S+Sq, color='blue')
    ax[1].plot(T, S0+Sq0, color='navy', ls='dashed', lw=2)

    ax[1].plot(T, S_corr+Sq_corr, color='magenta', lw=2)'''
    plt.plot(T, mu)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot results Seebeck(temperature)")
    parser.add_argument('Vc', help='Vc')
    parser.add_argument('tperp', help='tperp')
    args = parser.parse_args()
    main(args.Vc, args.tperp)