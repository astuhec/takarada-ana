import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.linalg import expm
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import linalg as LA
from scipy.special import digamma, expit
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

''' this function is called when I create free Hamiltonian and current operators '''
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
    # factors 1/2 because I split n_a*n_b into 0.5*(n_a*n_b + n_b*n_a)
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

''' kinetic (fixed) part of the Hamiltonian; for me delta is always zero
    mazza parameters are only relevant to imitate Denis' program lattice_1d_2b_optical_abinitio with mean field  '''
def h_k0(K, phys_parameters, mazza=None, delta_mazza=None):
    b, t, t_, t12, epsilon, epsilon_, Vb, Vc, delta = phys_parameters
    
    Nk = len(K)
    hk = np.zeros((2, 2, Nk), dtype=np.complex128)

    if mazza==None:
        _, kinetic, _ = parameters(b, t, t_, t12, Vb, Vc, delta)

        for line in kinetic:
            x, orb1, orb2, t = line
            x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
            hk[orb1,orb2] += t * np.exp(-1j*K*x)

        # add onsite energies, which are not included in kinetic
        hk[0,0] += -epsilon
        hk[1,1] += epsilon_

    elif mazza:
        tTa=-0.72/0.3
        tNi=1.0
        epsTa=1.35/0.3
        epsNi=-0.36/0.3
        hk[0,0] += epsNi + 2.0*tNi*np.cos(K) - 0.5*delta_mazza
        hk[1,1] += epsTa + 2.0*tTa*np.cos(K) + 0.5*delta_mazza
        hk[0,1] += (1-np.exp(1j*K)) * 0.116/0.3*np.sqrt(2)
        hk[1,0] += (1-np.exp(-1j*K)) * 0.116/0.3*np.sqrt(2)
    return hk

''' amplitudes delta_b, delta_c of the order parameter ( delta_k = delta_b + delta_c * exp(ik) )
    don't forget that rho[0,1]=<c_1^dag c_0>, with indices interchanged '''
@njit
def Delta(K, rho, Vb, Vc):
    Nk = len(K)
    deltas = [0., 1.]
    phi_b = np.sum(rho[0,1] * np.exp(+1j*K * deltas[0]))
    phi_c = np.sum(rho[0,1]  * np.exp(+1j*K * deltas[1]))
    return - np.array([Vb * phi_b, Vc * phi_c], dtype=np.complex128) / np.float64(Nk)

''' full hamiltonian, built from kinetic part hk0 and the self-energy
if include_hartree=True, I add also Hartree self-energy. in practice I do use Hartree
if include_hartree=False, I have only Fock (off-diagonal) self-energy.
a small eps0 is used in first couple of iterations to produce an excitonic state '''
def h_k(K, hk0, rho, Vb, Vc, eps0, include_hartree, mazza=None):
    if mazza==None:
        delta_b, delta_c = Delta(K, rho, Vb, Vc)

        Nk = rho.shape[-1]
        hk = hk0.copy()

        # Fock term:
        delta_k = delta_b + delta_c * np.exp(-1j*K)
        hk[0,1,:] += delta_k
        hk[1,0,:] += delta_k.conj()

        # Hartree term:
        if include_hartree:
            hk[0,0,:] += (Vb + Vc) * np.sum(rho[1,1,:]) / Nk
            hk[1,1,:] += (Vb + Vc) * np.sum(rho[0,0,:]) / Nk

        # simulate a perturbation to break symmetry
        if eps0 != 0:
            hk[0,1,:] += eps0 * np.exp(-1j*K)
            hk[1,0,:] += - eps0 * np.exp(1j*K)

    elif mazza:
        delta_b, delta_c = Delta(K, rho, Vb, Vc)

        Nk = rho.shape[-1]
        hk = hk0.copy()

        # Fock term:
        delta_k = delta_b * (Vb+Vc)/Vb#+ delta_c * np.exp(1j*K)
        hk[1,0,:] += delta_k
        hk[0,1,:] += delta_k.conj()

        # Hartree term:
        if include_hartree:
            hk[0,0,:] += (Vb) * np.sum(rho[1,1,:]) / Nk
            hk[1,1,:] += (Vb) * np.sum(rho[0,0,:]) / Nk

        # simulate a perturbation to break symmetry
        if eps0 != 0:
            hk[0,1,:] += eps0 * np.exp(-1j*K)
            hk[1,0,:] += - eps0 * np.exp(1j*K)

    return hk

''' helper functions to calculate gap '''
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

''' helper function to calculate <n0>,<n1> '''
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

''' Fermi-Dirac function '''
@njit
def fd(eps, mu, T):
    return 1.0 / (np.exp((eps - mu) / T) + 1.0)

''' diagonalization of the hamiltonian '''
def zero_T_filling(energije, n_target):
    """Fill the lowest states, sharing partial filling across degenerate states."""
    if not np.isfinite(n_target) or not 0 < n_target < 2:
        raise ValueError("n_target must be strictly between 0 and 2")
    levels = np.sort(energije.ravel())
    count = n_target * energije.shape[-1]
    nearest = round(count)
    if abs(count - nearest) < 1e-12:
        count = float(nearest)
    index = min(max(int(np.ceil(count)) - 1, 0), len(levels) - 1)
    fermi = levels[index]
    tied = np.isclose(energije, fermi, rtol=0, atol=1e-12)
    below = (energije < fermi) & ~tied
    occupations = below.astype(float)
    fraction = (count - np.count_nonzero(below)) / np.count_nonzero(tied)
    occupations[tied] = fraction
    above = energije[(energije > fermi) & ~tied]
    mu = 0.5 * (fermi + above.min()) if fraction == 1.0 and above.size else fermi
    return occupations, mu

def H_diagonalize(hamiltonian, K, T, mu, Gamma, n_target=1.0):
    Gamma = 0.0 if Gamma is None else float(Gamma)
    if not np.isfinite(Gamma) or Gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")
    if not np.isfinite(T) or T < 0:
        raise ValueError("T must be finite and nonnegative")
    Nk = len(K)

    # ── batch diagonalize unique k-points: i = 0, 1, ..., Nk//2 ──────
    H_batch = hamiltonian.transpose(2, 0, 1)
    n_unique = Nk // 2 + 1

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
        if Gamma is None or Gamma == 0:
            occupations, _ = zero_T_filling(energije, n_target)
        else:
            occupations = 0.5 - np.arctan((energije-mu) / Gamma) / np.pi
        fs[0, 0, :] = occupations[0]
        fs[1, 1, :] = occupations[1]
    elif Gamma == 0:
        fs[0, 0, :] = expit(-(energije[0] - mu) / T)
        fs[1, 1, :] = expit(-(energije[1] - mu) / T)
    else:
        z = 0.5 + (Gamma + 1j*(energije-mu)) / (2*np.pi*T)
        occupations = 0.5 - np.imag(digamma(z)) / np.pi
        fs[0, 0, :] = occupations[0]#fd(energije[0, :], mu, T)
        fs[1, 1, :] = occupations[1]#fd(energije[1, :], mu, T)

    return energije, vecs, fs

''' a single iteration in the self-consistency equation. rho --> rho_new  '''
def F(hamiltonian, rho, K, T, mu, Gamma, n_target=1.0):
    _, vecs, fs = H_diagonalize(hamiltonian, K, T, mu, Gamma, n_target=n_target)
    rho_new = np.einsum('ijk,jmk,mnk->ink', vecs, fs, np.swapaxes(vecs.conj(),0,1))
    return rho_new, np.max(np.abs(rho - rho_new))

''' occupation, which should be 1.0 for undoped case '''
def zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijk->ij', rho)))/(np.prod(rho.shape[-1]))).real

''' occupation with broadened spectral functions '''
@njit(cache=True)
def fermi(x,beta):
    return 0.5 * (1.0 - np.tanh(0.5*beta*x))

def zasedenost_Gamma(energije, mu, Gamma, beta):
    # z is the argument of digamma function
    Nk = energije.shape[1]
    z = 0.5 + beta/(2.0*np.pi) * (Gamma + 1j*(energije - mu))
    occ = 0.5 - np.imag(digamma(z))/np.pi
    return np.sum(occ) / Nk

def zasedenost_Gamma_lowT(energije, mu, Gamma, beta):
    Nk = energije.shape[1]
    d = energije - mu
    r2 = Gamma**2 + d**2
    occ0 = 0.5 - np.arctan(d / Gamma) / np.pi
    occ2 = np.pi / (3.0 * beta**2)* Gamma * d / r2**2
    return np.sum(occ0 + occ2) / Nk

''' various functions for converging the self-consistnecy equation '''
def Rho_next(hk0, rho, K, T, mu, Vb, Vc, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, Gamma, mix=0.5, mazza=None, n_target=1.0):
    err, N_iters = 1.0, 0
    while err > epsilon_threshold and N_iters < maxiter:
        eps = eps0 if N_iters < N_epsilon else 0.0
        rho_new, err = F(h_k(K, hk0, rho, Vb, Vc, eps, include_hartree, mazza), rho, K, T, mu, Gamma, n_target=n_target)
        rho = rho_new * mix + rho * (1 - mix)
        N_iters += 1
    rho, _ = F(h_k(K, hk0, rho, Vb, Vc, 0., include_hartree, mazza), rho, K, T, mu, Gamma, n_target=n_target)
    energije, vecs, fs = H_diagonalize(h_k(K, hk0, rho, Vb, Vc, 0., include_hartree, mazza), K, T, mu, Gamma, n_target=n_target)
    n = zasedenost(rho)
    return rho, err, energije, vecs, fs, n

''' functions for determining chemical potential at the target filling '''
def f_newmu(mu, hk0, rho, K, T, Vb, Vc, eps0,
            epsilon_threshold, N_epsilon, maxiter, include_hartree, mix=0.50, n_target=1.0, Gamma=None):
    rhonew, _, energies, _, _, n = Rho_next(
        hk0, rho, K, T, mu, Vb, Vc, eps0, epsilon_threshold,
        N_epsilon, maxiter, include_hartree, Gamma, mix, n_target=n_target)
    if Gamma is not None and Gamma > 0:
        return zasedenost_Gamma(energies, mu, Gamma, 1 / T) - n_target
    elif Gamma == 0.0:
        return zasedenost(rhonew) - n_target
    if n_target == 1.0 and T > 0:
        # n - 1 = upper-band electrons - lower-band holes. Evaluate
        # their logarithms directly: subtracting from a filled band loses
        # the exponentially small hole density in an insulator.
        log_electrons = logsumexp(-np.logaddexp(0.0, (energies[1] - mu) / T))
        log_holes = logsumexp(-np.logaddexp(0.0, (mu - energies[0]) / T))
        return log_electrons - log_holes
    return n - n_target

def find_bracket(mu1, mu2, hk0, rho, K, T, Vb, Vc, eps0,
                 epsilon_threshold, N_epsilon, maxiter, include_hartree, mix,
                 max_expand=20, expand_factor=2.0, n_target=1.0, Gamma=None):
    """
    Expand [mu1, mu2] outward until f(mu1) and f(mu2) have opposite signs.
    """
    args = (hk0, rho, K, T, Vb, Vc, eps0,
            epsilon_threshold, N_epsilon, maxiter, include_hartree, mix, n_target, Gamma)
    
    f1 = f_newmu(mu1, *args)
    f2 = f_newmu(mu2, *args)
    
    center = (mu1 + mu2) / 2.0
    half_width = (mu2 - mu1) / 2.0

    for i in range(max_expand):
        if f1 * f2 < 0 or (n_target != 1.0 and (f1 == 0 or f2 == 0)):
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

def NewMu(rho, K, hk0, Vb, Vc, T, mu, dmu, maxiter, epsilon_threshold, eps_last, Gamma, mix, mix2, mix3, n_pass, max_trials, faktor1=0.001, include_hartree=True, n_target=1.0):
    _, err_a, _, _, _, n_a = Rho_next(hk0, rho, K, T, mu, Vb, Vc, 0.0, epsilon_threshold, 0, maxiter, include_hartree, Gamma, mix)
    _, err_b, _, _, _, n_b = Rho_next(hk0, rho, K, T, mu + dmu, Vb, Vc, 0.0, epsilon_threshold, 0, maxiter, include_hartree, Gamma, mix)

    chi = (n_b - n_a)/dmu

    if abs(chi) < 1e-5:
        step_direction = np.sign(n_a - n_target)
        mu = mu - 0.1 * dmu * step_direction
    elif chi != 0:
        mu = mu - mix2 * (n_a - n_target)/np.abs(chi)

    if np.abs(chi) > 0:
        faktor = (n_a - n_target)/chi * mix3
    else:
        faktor = faktor1
    if chi >= 0:
        if n_a >= n_target:
            sign = -1
        elif n_a < n_target: sign = +1
    elif chi < 0:
        if n_a >= n_target: sign = +1
        elif n_a < n_target: sign = -1
    
    pogoj = False
    steps = 0
    enough = False

    sgns = np.ones(2) * np.sign(n_a - n_target)
    ns = np.array([0, n_a])
    mus = [0.0, mu]

    while sgns[0] == sgns[1]:
        if np.abs(n_a - n_target) < n_pass and err_a < eps_last:
            enough = True
            break
        _, err_b, _, _, _, n_b = Rho_next(hk0, rho, K, T, mu + faktor*steps*sign, Vb, Vc, 0.0, epsilon_threshold, 0, maxiter, include_hartree, Gamma, mix)

        ns[0] = n_b
        mus[0] = mu + faktor*steps*sign
        sgns[1] = np.sign(n_b - n_target)
        if sgns[0] != sgns[1]: break
        if n_b < n_target and n_b < ns[1]:
            sign *= -1
        if n_b > n_target and n_b > ns[1]:
            sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - n_target)
        steps +=1
        if np.abs(n_b - n_target) < n_pass and err_b < eps_last:
            enough = True
            mu_mid = mu + faktor*steps*sign
            break
        
    mus = np.sort(np.array([mu + faktor*steps*sign, mu + faktor*(steps-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        mu_mid = (mus[0] + mus[1])/2
        if enough == True:
            break   
        n_mid = Rho_next(hk0, rho, K, T, mu_mid, Vb, Vc, 0.0, epsilon_threshold, 0, maxiter, include_hartree, Gamma, mix)[-1]
        if n_mid > n_target: mus[1] = mu_mid
        elif n_mid < n_target: mus[0] = mu_mid
        if np.abs(n_mid - n_target) < n_pass:
            break
        trials += 1 
        if trials > max_trials:
            break
    rho, err, energije, vecs, fs, n = Rho_next(hk0, rho, K, T, mu_mid, Vb, Vc, 0.0, epsilon_threshold, 0, maxiter, include_hartree, Gamma, mix)
    return rho, err, energije, vecs, fs, n, mu_mid

def NewMu2(mu1, mu2, hk0, rho, K, T, Vb, Vc, eps0,
             epsilon_threshold, N_epsilon, maxiter, include_hartree, Gamma, mix=0.5, xtol=1e-4, rtol=1e-4, maxiterbrentq=50, n_target=1.0):
    if Gamma is not None and (not np.isfinite(Gamma) or Gamma < 0):
        raise ValueError("Gamma must be finite and nonnegative")
    if not np.isfinite(n_target) or not 0 < n_target < 2:
        raise ValueError("n_target must be strictly between 0 and 2")
    if T <= 0:
        raise ValueError("NewMu2 requires T > 0; use fixed-filling Rho_next for T == 0")
    # Auto-fix bracket if needed
    try:
        mu1, mu2 = find_bracket(mu1, mu2, hk0, rho, K, T, Vb, Vc, eps0,
                                epsilon_threshold, N_epsilon, maxiter, include_hartree, mix, n_target=n_target, Gamma=Gamma)
    except ValueError as e:
        print(f"Warning: {e}")
        raise
    mu_star = brentq(f_newmu, mu1, mu2, args=(hk0, rho, K, T, Vb, Vc, eps0,
                                              epsilon_threshold, N_epsilon, maxiter, include_hartree, mix, n_target, Gamma),
                     xtol=xtol, rtol=rtol, maxiter=maxiterbrentq)
    rho_final, err, energije, vecs, fs, n = Rho_next(hk0, rho, K, T, mu_star, Vb, Vc, eps0, epsilon_threshold, N_epsilon,
                                                          maxiter, include_hartree, Gamma, mix=mix, n_target=n_target)
    if Gamma is not None and Gamma > 0:
        n = zasedenost_Gamma(energije, mu_star, Gamma, 1 / T)
    return mu_star, rho_final, err, energije, vecs, fs, n

def ground_state_fixed_filling(hk0, rho, K, mu, Vb, Vc, eps0,
                               epsilon_threshold, N_epsilon, maxiter,
                               include_hartree, Gamma, n_target=1.0,
                               mazza=None, mix=0.5):
    """Converge the zero-temperature density and chemical potential together."""
    if not np.isfinite(n_target) or not 0 < n_target < 2:
        raise ValueError("n_target must be strictly between 0 and 2")
    Gamma = 0.0 if Gamma is None else Gamma
    if not np.isfinite(Gamma) or Gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")

    def solve(trial_mu):
        result = Rho_next(
            hk0, rho, K, 0., trial_mu, Vb, Vc, eps0,
            epsilon_threshold, N_epsilon, maxiter, include_hartree,
            Gamma, mix=mix, mazza=mazza, n_target=n_target)
        if not np.isfinite(result[1]) or result[1] > epsilon_threshold:
            raise RuntimeError("Ground-state Hartree-Fock iteration did not converge")
        return result

    if Gamma == 0:
        result = solve(mu)
        _, mu = zero_T_filling(result[2], n_target)
        return (mu, *result)

    def residual(trial_mu):
        return solve(trial_mu)[-1] - n_target

    width = max(float(np.max(np.abs(hk0))), Gamma, abs(Vb), abs(Vc), 1.)
    for _ in range(60):
        lo, hi = mu - width, mu + width
        if residual(lo) <= 0 <= residual(hi):
            break
        width *= 2
    else:
        raise RuntimeError("Could not bracket the ground-state chemical potential")
    mu = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-12)
    result = solve(mu)
    if abs(result[-1] - n_target) > max(1e-9, 10 * epsilon_threshold):
        raise RuntimeError("Ground-state filling did not converge to n_target")
    return (mu, *result)

def mu_fixed_bands(energije, T, Gamma=None, n_target=1.0):
    Gamma = 0.0 if Gamma is None else float(Gamma)
    if not np.isfinite(Gamma) or Gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")
    if not np.isfinite(n_target) or not 0 < n_target < 2:
        raise ValueError("n_target must be between 0 and 2")
    if not np.isfinite(T) or T < 0:
        raise ValueError("T must be finite and nonnegative")
    if T == 0 and Gamma == 0:
        return zero_T_filling(energije, n_target)[1]
    Nk = energije.shape[-1]

    def residual(mu):
        if T == 0:
            return (0.5 - np.arctan((energije - mu) / Gamma) / np.pi).sum() / Nk - n_target
        if n_target == 1.0 and Gamma == 0:
            log_e = logsumexp(-np.logaddexp(0.0, (energije[1] - mu) / T))
            log_h = logsumexp(-np.logaddexp(0.0, (mu - energije[0]) / T))
            return log_e - log_h
        if Gamma == 0:
            return expit(-(energije - mu) / T).sum() / Nk - n_target
        return zasedenost_Gamma(energije, mu, Gamma, 1 / T) - n_target

    width = max(float(np.ptp(energije)), T, Gamma, 1e-6)
    for _ in range(60):
        lo, hi = energije.min() - width, energije.max() + width
        if residual(lo) <= 0 <= residual(hi):
            return brentq(residual, lo, hi, xtol=1e-12, rtol=1e-12)
        width *= 2
    raise RuntimeError("Could not bracket the fixed-band chemical potential")

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
def DoS(K, energije, epsilons, mu, tok_tilde, faktor, shape='Gaussian', Gamma=None):
    Nk = len(K)
    v_max = np.max(np.abs(tok_tilde))
    sigma = np.sqrt(v_max * (epsilons[1] - epsilons[0]) * (K[1] - K[0])) * faktor
    dos = np.zeros((2, len(epsilons)))
    for k in prange(Nk):
        for alpha in range(2):
            if Gamma==None:
                dos[alpha] += delta_approximation(epsilons - energije[alpha,k] + mu, sigma, shape) 
            else:
                dos[alpha] += 1/np.pi * Gamma / ((epsilons - energije[alpha,k])**2 + Gamma**2)
    return dos / Nk

''' approximation for Dirac delta function '''
def delta_approximation(x, width, shape='Gaussian'):
    if shape == 'Gaussian':
        return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))
    elif shape == 'Lorentzian':
        return 1/np.pi * width/(x**2 + width**2)
    
def local_maxima(arr):
    n = len(arr)
    indices, vals = [], []
    for i in range(n):
        if i > 0 and arr[i] <= arr[i - 1]:
            continue
        if i < n - 1 and arr[i] <= arr[i + 1]:
            continue
        indices.append(i)
        vals.append(arr[i])
    return np.array(indices), np.array(vals)

def to_scalar_if_single(x):
    x = np.asarray(x)
    if x.size == 1:
        return float(x.item())
    return x

def is_stable(Ts, mus, threshold=0.02, window=5):
    stable_from_idx = 0
    for i in range(len(Ts) - window):
        local_mus = mus[i:i+window]
        variation = np.max(local_mus) - np.min(local_mus)
        if variation < threshold:
            stable_from_idx = i
            break
    stable_from_idx = stable_from_idx + window
    return stable_from_idx

def find_linear_region(Ts, mus, stable_idx, window=10, r2_threshold=0.99):
    """
    Slide a window over T > T_stable, fit linear, check R^2.
    Linear region = consecutive windows with high R^2.
    Stop when R^2 drops below threshold.
    """
    from scipy.stats import linregress
        
    # Only look above stable index
    T_sub  = Ts[stable_idx:]
    mu_sub = mus[stable_idx:]
    
    r2_values = []
    T_centers = []
    
    for i in range(len(T_sub) - window):
        T_win  = T_sub[i:i+window]
        mu_win = mu_sub[i:i+window]
        
        slope, intercept, r, p, se = linregress(T_win, mu_win)
        r2 = r**2
        r2_values.append(r2)
        T_centers.append(T_sub[i])
    
    r2_values = np.array(r2_values)
    T_centers = np.array(T_centers)
    
    # Find last index where R^2 is still above threshold
    linear_mask = r2_values >= r2_threshold
    if not np.any(linear_mask):
        return None, None
    
    last_linear_idx = np.where(linear_mask)[0][-1]
    T_linear_end    = T_centers[last_linear_idx] + (T_sub[1]-T_sub[0])*window
    T_linear_start  = T_sub[0] # starts at T_stable

    return T_linear_start, T_linear_end
