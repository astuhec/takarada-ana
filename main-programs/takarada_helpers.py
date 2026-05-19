import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.optimize import brentq
from scipy.linalg import expm
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import linalg as LA

''' this function is called when I create current operators '''
@njit(cache=True)
def parameters(b, t, t_, t12, Vb, Vc, delta=0):
    ''' delta, orb1, orb2, hopping '''
    kinetic = np.array([
        (1, 0, 0, t),
        (-1, 0, 0, t),
        (1, 1, 1, -t_),
        (-1, 1, 1, -t_),
        (0, 0, 1, t12 + delta),
        (0, 1, 0, t12 + delta),
        # onsite energies don't matter for current operators !
        #(0, 0, 0, epsilon - mu),
        #(0, 1, 1, epsilon_ - mu) 
    ])

    ''' x, orb1, orb2, interaction '''
    interaction = np.array([
        (0, 0, 1, Vb / 2),
        (0, 1, 0, Vb / 2),
        (1, 0, 1, Vc / 2),
        (-1, 1, 0, Vc / 2)
    ])

    ''' intracell orbital positions. I set r0=0 '''
    pos = np.array([0.0, b])
    return pos, kinetic, interaction

''' zeroth approximation for density matrix: lower band fully occupied '''
def rho0(Nk):
    rho = np.zeros((2, 2, Nk))
    rho[0,0,:] = 1.
    return rho

''' helper functions to calculate gap, <n0>, <n1> '''
def Gap(energije, delta_b, delta_c, Vb, Vc, epsilon_threshold, gap_infty):
    condition = False
    if Vb != 0 and Vc != 0:
        if np.abs(delta_b) < epsilon_threshold and np.abs(delta_c) < epsilon_threshold:
            condition = True
    elif Vb != 0 and Vc == 0:
        if np.abs(delta_b) < epsilon_threshold:
            condition = True
    elif Vb == 0 and Vc != 0:
        if np.abs(delta_c) == 0:
            condition = True
    elif Vb == 0 and Vc == 0:
        condition = True
    if condition == True:
        gap = gap_infty
    else:
        gap = np.min(energije[1]) - np.max(energije[0])
    return gap

def Ns(rho, energije_infty, delta_b, delta_c, Vb, Vc, epsilon_threshold, T, mu):
    Nk = rho.shape[-1]
    condition = False
    if Vb != 0 and Vc != 0:
        if np.abs(delta_b) < epsilon_threshold and np.abs(delta_c) < epsilon_threshold:
            condition = True
    elif Vb != 0 and Vc == 0:
        if np.abs(delta_b) < epsilon_threshold:
            condition = True
    elif Vb == 0 and Vc != 0:
        if np.abs(delta_c) == 0:
            condition = True
    elif Vb == 0 and Vc == 0:
        condition = True
    if condition == True:
        rho1 = np.zeros_like(rho)
        rho1[0,0] = fd(energije_infty[0], mu, T)
        rho1[1,1] = fd(energije_infty[1], mu, T)
        n0 = np.sum(rho1[0,0]).real / Nk
        n1 = np.sum(rho1[1,1]).real / Nk
    else:
        n0 = np.sum(rho[0,0]).real / Nk
        n1 = np.sum(rho[1,1]).real / Nk
    return n0, n1

# subtract Hartree shift
def Gap_tilde(rho, phys_parameters):
    Nk = rho.shape[-1]
    n0 = np.sum(rho[0,0,:]).real / Nk
    n1 = np.sum(rho[1,1,:]).real / Nk
    _, t, t_, _, epsilon, epsilon_, Vb, Vc, _ = phys_parameters
    gap0 = epsilon + epsilon_ - 2*(t + t_)
    gaptilde = gap0 + (Vb + Vc) * (n0 - n1)
    return gaptilde

def Gap_infty(Nk, phys_parameters, parameters1, include_hartree):
    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters
    new_parameters1 = parameters1.copy()
    new_parameters1["eps0"] = 0.
    m = mt.model(Nk, 0., phys_parameters, new_parameters1, new_parameters1, include_hartree, True)
    m.GS()
    hk = m.hk0.copy()
    if include_hartree == True:
        hk[0,0,:] += (Vb + Vc) * np.sum(m.rho[1,1,:]) / Nk
        hk[1,1,:] += (Vb + Vc) * np.sum(m.rho[0,0,:]) / Nk
    energy_infty = np.zeros((2, Nk))
    energy_infty[0] = hk[0,0].real
    energy_infty[1] = hk[1,1].real
    mu_infty = m.mu
    gap_infty = np.min(hk[1,1]) - np.max(hk[0,0])
    return energy_infty, mu_infty, gap_infty

''' kinetic (fixed) part of the Hamiltonian; for me delta is always zero '''
def h_k0(K, phys_parameters):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    
    Nk = len(K)
    hk = np.zeros((2, 2, Nk), dtype=np.complex128)

    _, kinetic, _ = parameters(b, t, t_, t12, Vb, Vc, delta)

    for line in kinetic:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        hk[orb1,orb2] += t * np.exp(-1j*K*x)

    # add onsite energies, which are not included in kinetic
    hk[0,0] += -epsilon
    hk[1,1] += epsilon_

    return hk

''' amplitudes delta_b, delta_c of the order parameter ( delta_k = delta_b + delta_c * exp(ik) ) '''
@njit
def Delta(K, rho, Vb, Vc):
    Nk = len(K)
    deltas = [0., 1.]
    phi_b = np.sum(rho[1,0] * np.exp(-1j*K * deltas[0]))
    phi_c = np.sum(rho[1,0]  * np.exp(-1j*K * deltas[1]))
    return - np.array([Vb * phi_b, Vc * phi_c]) / Nk

''' full hamiltonian, built from kinetic part hk0 and the self-energy
if include_hartree=True, I add also Hartree self-energy,
if include_hartree=False, I have only Fock (off-diagonal) self-energy.
a small eps0 is used in first couple of iterations to produce an excitonic state '''
def h_k(K, hk0, rho, Vb, Vc, eps0, include_hartree):
    delta_b, delta_c = Delta(K, rho, Vb, Vc)

    Nk = rho.shape[-1]
    hk = hk0.copy()

    # Fock term:
    delta_k = delta_b + delta_c * np.exp(1j*K)
    hk[1,0,:] += delta_k
    hk[0,1,:] += delta_k.conj()

    # Hartree term:
    if include_hartree == True:
        hk[0,0,:] += (Vb + Vc) * np.sum(rho[1,1,:]) / Nk
        hk[1,1,:] += (Vb + Vc) * np.sum(rho[0,0,:]) / Nk

    # simulate a perturbation to break symmetry
    if eps0 != 0:
        hk[0,1,:] += eps0 * np.exp(-1j*K)
        hk[1,0,:] += - eps0 * np.exp(1j*K)
    return hk

''' diagonalization of the hamiltonian '''
def H_diagonalize(hamiltonian, K, T, mu):
    Nk = len(K)

    # ── Batch diagonalize unique k-points: i = 0, 1, ..., Nk//2 ──────
    # hamiltonian: (2, 2, Nk) -> np.linalg.eigh expects (batch, 2, 2)
    H_batch = hamiltonian.transpose(2, 0, 1)                # (Nk, 2, 2)
    n_unique = Nk // 2 + 1                                  # indices 0..Nk//2

    en_batch, v_batch = np.linalg.eigh(H_batch[:n_unique])  # (n_unique, 2), (n_unique, 2, 2)

    # ── Fill positive/unique half ──────────────────────────────────────
    energije = np.zeros((2, Nk))
    vecs     = np.zeros((2, 2, Nk), dtype=np.complex128)

    energije[:, :n_unique] = en_batch.T                     # (2, n_unique)
    vecs[:, :, :n_unique]  = v_batch.transpose(1, 2, 0)     # (2, 2, n_unique)

    # ── Fill negative half by conjugate symmetry: i -> -i ─────────────
    # indices 1..Nk//2-1 map to -1..-( Nk//2-1), i.e. Nk-1..Nk//2+1
    if Nk // 2 - 1 > 0:
        energije[:, Nk//2+1:] = energije[:, 1:Nk//2][:, ::-1]
        vecs[:, :, Nk//2+1:]  = vecs[:, :, 1:Nk//2][:, :, ::-1].conj()

    # ── Fermi-Dirac occupation matrices ───────────────────────────────
    fs = np.zeros((2, 2, Nk))

    if T == 0:
        fs[0, 0, :] = 1.0
        fs[1, 1, :] = 0.0
    else:
        fs[0, 0, :] = fd(energije[0, :], mu, T)
        fs[1, 1, :] = fd(energije[1, :], mu, T)

    return energije, vecs, fs

''' a single iteration in the self-consistency equation. rho --> rho_new  '''
def F(hamiltonian, rho, K, T, mu):
    _, vecs, fs = H_diagonalize(hamiltonian, K, T, mu)
    rho_new = np.einsum('ijk,jmk,mnk->ink', vecs, fs, np.swapaxes(vecs.conj(),0,1))
    return rho_new, np.max(np.abs(rho - rho_new))

''' occupation, which should be 1.0 for undoped case '''
def zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijk->ij', rho)))/(np.prod(rho.shape[-1]))).real

''' various functions for converging the self-consistnecy equation '''
def Rho_next(hk0, rho, K, T, mu, Vb, Vc, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5):
    err, N_iters = 1.0, 0
    while err > epsilon_threshold and N_iters < maxiter:
        eps = eps0 if N_iters < N_epsilon else 0.0
        rho_new, err = F(h_k(K, hk0, rho, Vb, Vc, eps, include_hartree), rho, K, T, mu)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(h_k(K, hk0, rho, Vb, Vc, 0., include_hartree), rho, K, T, mu)
    energije, vecs, fs = H_diagonalize(h_k(K, hk0, rho, Vb, Vc, 0., include_hartree), K, T, mu)
    n = zasedenost(rho)
    return rho, err, energije, vecs, fs, n

''' functions for determining chemical potential for half-filling '''
def f_newmu(mu, hk0, rho, K, T, Vb, Vc, eps0,
            epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.50, n_target=1.0):
    _, _, _, _, _, n = Rho_next(hk0, rho, K, T, mu, Vb, Vc, eps0, epsilon_threshold, N_epsilon, maxiter, include_hartree, mix)
    return n - n_target

def find_bracket(mu1, mu2, hk0, rho, K, T, phys_parameters, eps0,
                 epsilon_threshold, N_epsilon, maxiter, include_hartree, mix,
                 max_expand=20, expand_factor=2.0):
    """
    Expand [mu1, mu2] outward until f(mu1) and f(mu2) have opposite signs.
    """
    args = (hk0, rho, K, T, phys_parameters, eps0,
            epsilon_threshold, N_epsilon, maxiter, include_hartree, mix)
    
    f1 = f_newmu(mu1, *args)
    f2 = f_newmu(mu2, *args)
    
    center = (mu1 + mu2) / 2.0
    half_width = (mu2 - mu1) / 2.0

    for i in range(max_expand):
        if f1 * f2 < 0:
            return mu1, mu2  # valid bracket found
        
        # Expand symmetrically
        half_width *= expand_factor
        mu1 = center - half_width
        mu2 = center + half_width
        
        f1 = f_newmu(mu1, *args)
        f2 = f_newmu(mu2, *args)
            
    raise ValueError(
        f"Could not bracket root after {max_expand} expansions. "
        f"Last: mu1={mu1:.4f}, f(mu1)={f1:.4f}, mu2={mu2:.4f}, f(mu2)={f2:.4f}"
    )

def NewMu2(mu1, mu2, hk0, rho, K, T, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.5, xtol=1e-4, rtol=1e-4, maxiterbrentq=50, n_target=1.0):
    # Auto-fix bracket if needed
    try:
        mu1, mu2 = find_bracket(mu1, mu2, hk0, rho, K, T, phys_parameters, eps0,
                                epsilon_threshold, N_epsilon, maxiter, include_hartree, mix)
    except ValueError as e:
        print(f"Warning: {e}")
        raise
    mu_star = brentq(f_newmu, mu1, mu2, args=(hk0, rho, K, T, phys_parameters, eps0,
                                              epsilon_threshold, N_epsilon, maxiter, include_hartree, mix, n_target),
                     xtol=xtol, rtol=rtol, maxiter=maxiterbrentq)
    rho_final, err, energije, vecs, fs, n = Rho_next(hk0, rho, K, T, mu_star, phys_parameters[6], phys_parameters[7], eps0, epsilon_threshold, N_epsilon,
                                                          maxiter, include_hartree, mix=mix)
    return mu_star, rho_final, err, energije, vecs, fs, n

''' expectation value of Hamiltonian. I need this for specific heat and entropy '''
@njit(parallel=True, cache=True)
def energy_average(K, rho, phys_parameters, energije, mu, T):
    Nk = len(K)
    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters
    en = 0.
    ''' first add average of MF Hamiltonian'''
    for i in [0, Nk//2]:
        for orb in range(2):
            en += fd(energije[orb,i], mu, T) * energije[orb,i]
    for i in prange(1,Nk//2):
        for orb in range(2):
            en += 2 * fd(energije[orb,i], mu, T) * energije[orb,i]
    ''' then add also constant terms which are discarded in MF '''
    delta_b, delta_c = Delta(K, rho, Vb, Vc)
    en += -(Vb + Vc) / Nk * np.sum(rho[0,0]) * np.sum(rho[1,1])
    if Vb != 0:
        en += Nk * 1/Vb * np.abs(delta_b)**2
    if Vc != 0:
        en += Nk * 1/Vc * np.abs(delta_c)**2
    return en.real / Nk

''' density of states '''
def DoS(K, energije, epsilons, mu, tok_tilde, faktor, shape='Gaussian'):
    Nk = len(K)
    v_max = np.max(np.abs(tok_tilde))
    sigma = np.sqrt(v_max * (epsilons[1] - epsilons[0]) * (K[1] - K[0])) * faktor
    dos = np.zeros((2, len(epsilons)))
    for k in prange(Nk):
        for alpha in range(2):
            dos[alpha] += delta_approximation(epsilons - energije[alpha,k] + mu, sigma, shape) 
    return dos / Nk

''' approximation for Dirac delta function '''
def delta_approximation(x, width, shape='Gaussian'):
    if shape == 'Gaussian':
        return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))
    elif shape == 'Lorentzian':
        return 1/np.pi * width/(x**2 + width**2)