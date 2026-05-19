import numpy as np
import scipy.linalg as LA
from numba import njit, prange
from scipy.linalg import expm
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import linalg as LA

from takarada_helpers import parameters, h_k, Delta, delta_approximation

''' current operator '''
@njit(cache=True)
def j_tok(K, phys_parameters):
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, _ = parameters(b, t, t_, t12, Vb, Vc, delta)
    Nk = len(K)
    j = np.zeros((2, 2, Nk), dtype=np.complex128)
    for line in kinetic:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        ad = - 1j * t * np.exp(-1j * K * x) * (pos[orb1] - pos[orb2] + x)
        j[orb1, orb2] += ad
    return j

''' below is stuff for mean-field approximation of the non-local interaction current operator '''
def input_data(K, phys_parameters):
    Nk = len(K)
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12,  Vb, Vc, delta)
    geom = dict()
    geom["kinetic"] = kinetic
    geom["interaction"] = interaction
    geom["pos"] = pos

    phases_kin = np.zeros((len(kinetic), Nk), dtype=np.complex128)
    phases_int = np.zeros((len(interaction), Nk), dtype=np.complex128)
    for l in range(len(kinetic)):
        x = kinetic[l][0]
        phases_kin[l] = np.exp(-1j*K*x)
    for l in range(len(interaction)):
        x = interaction[l][0]
        phases_int[l] = np.exp(-1j*K*x)
    phases = dict()
    phases["int"] = phases_int
    phases["kin"] = phases_kin
    return geom, phases

def convolution(g, h):
    return np.fft.fftshift(np.fft.ifft(g * h))

def G_ffts(phases, Nk):
    L = len(phases["kin"])
    M = len(phases["int"])
    g_ffts_M4a1 = np.zeros((L, M, Nk), dtype=np.complex128)
    g_ffts_M4a2 = np.copy(g_ffts_M4a1)
    g_ffts_M4b1 = np.copy(g_ffts_M4a1)
    g_ffts_M4b2 = np.copy(g_ffts_M4a1)

    for l in range(L):
        for m in range(M):
            g = np.conj(phases["kin"][l]) * phases["int"][m]
            g_ffts_M4a1[l,m,:] = np.fft.fft(np.fft.ifftshift(g))
            
            g = phases["kin"][l] * np.conj(phases["int"][m])
            g_ffts_M4b1[l,m,:] = np.fft.fft(np.fft.ifftshift(g))

            g = phases["int"][m]
            g_ffts_M4a2[l,m,:] = np.fft.fft(np.fft.ifftshift(g))

            g = np.conj(phases["int"][m])
            g_ffts_M4b2[l,m,:] = np.fft.fft(np.fft.ifftshift(g))
    return g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2

def compute_all_mf_matrices(K, rho, geom, phases, g_ffts):
    Nk = len(K)

    M3  = np.zeros((2,2,Nk), dtype=np.complex128)
    M6  = np.zeros((2,2,Nk), dtype=np.complex128)
    M4a = np.zeros((2,2,Nk), dtype=np.complex128)
    M4b = np.zeros((2,2,Nk), dtype=np.complex128)

    # --- prepare densities ---
    rho00 = rho[0,0,:]
    rho11 = rho[1,1,:]
    rho01 = rho[0,1,:]
    rho10 = rho[1,0,:]

    n = [rho00.sum(), rho11.sum()]

    rho_fft = {
        (0,0): np.fft.fft(np.fft.ifftshift(rho00)),
        (1,1): np.fft.fft(np.fft.ifftshift(rho11)),
        (0,1): np.fft.fft(np.fft.ifftshift(rho01)),
        (1,0): np.fft.fft(np.fft.ifftshift(rho10)),
    }

    g_ffts_M4a1, g_ffts_M4a2, g_ffts_M4b1, g_ffts_M4b2 = g_ffts

    # --- loop over geometry ---
    for l, (x, orb1, orb2, t) in enumerate(geom["kinetic"]):
        orb1, orb2 = int(orb1), int(orb2)
        phase_k = phases["kin"][l]
        fk = t * phase_k / Nk

        for m, (x_, orb1_, orb2_, V_) in enumerate(geom["interaction"]):
            orb1_, orb2_ = int(orb1_), int(orb2_)
            if orb2 == orb2_:
                lega = geom["pos"][orb2] - geom["pos"][orb1_] - x_

                # ---------- M3 ----------
                M3[orb1,orb2] += -1j * t * V_ * lega * phase_k * n[orb1_] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb2,orb1,:] * phase_k)
                M6[orb1_,orb1_] += -1j * t * V_ * lega * suma / Nk

                # ---------- M4a ---------- (matrix3) - fourth term in equation
                #g = -1j * V_ * lega * np.conj(phase_k) * phases["int"][m]
                g_fft = -1j * V_ * lega * g_ffts_M4a1[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                gh = np.fft.fftshift(
                        np.fft.ifft(g_fft * rho_fft[(orb1_,orb1)]))
                M4a[orb1_,orb2] += fk * gh

                # ---------- M4b ---------- (matrix4) - third term in equation
                h = fk * rho[orb2,orb1_,:]
                h_fft = np.fft.fft(np.fft.ifftshift(h))
                g_fft = -1j * V_ * lega * g_ffts_M4b1[l,m,:]
                #g_fft = np.fft.fft(np.fft.ifftshift(
                #            -1j * V_ * lega * np.conj(phases["int"][m]) * phases["kin"][l]
                #        ))
                gh = np.fft.fftshift(np.fft.ifft(g_fft * h_fft))
                M4b[orb1,orb1_] += gh

            if orb1 == orb2_:
                lega = geom["pos"][orb1] - geom["pos"][orb1_] - x_
                
                # ---------- M3 ----------
                M3[orb1,orb2] += +1j * t * V_ * lega * phase_k * n[orb1_] / Nk

                # ---------- M6 ----------
                suma = np.sum(rho[orb2,orb1,:] * phase_k)
                M6[orb1_,orb1_] += +1j * t * V_ * lega * suma / Nk

                # ---------- M4a ---------- fourth term in equation
                #g = +1j * V_ * lega * phases["int"][m]
                g_fft = +1j * V_ * lega * g_ffts_M4a2[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                gh = np.fft.fftshift(
                        np.fft.ifft(g_fft * rho_fft[(orb1_,orb1)]))
                M4a[orb1_,orb2] += fk * gh

                # ---------- M4b ---------- third term in equation
                #g = +1j * V_ * lega * np.conj(phases["int"][m])
                g_fft = +1j * V_ * lega * g_ffts_M4b2[l,m,:]#np.fft.fft(np.fft.ifftshift(g))
                h = fk * rho[orb2,orb1_,:]
                h_fft = np.fft.fft(np.fft.ifftshift(h))
                gh = np.fft.fftshift(
                            np.fft.ifft(g_fft * h_fft)
                )
                M4b[orb1,orb1_] += gh

    return M3, M6, -M4a, -M4b

def compute_together_mf_matrices(K, rho, geom, phases, g_ffts):
    m1,m2,m3,m4 = compute_all_mf_matrices(K, rho, geom, phases, g_ffts)
    return m1 + m2 + m3 + m4

''' functions mf_matrix1,2,3,4 give exactly the same as function compute_all_mf_matrices,
but the latter is more convenient (faster if used for multiple calls) because it uses precomputed elements '''
def mf_matrix1(K, rho, phys_parameters):
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, Vb, Vc, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(2):
        for beta in range(2):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb1_, orb1_, :])
                            lega = pos[orb2 ] - pos[orb1_ ] - x_
                            matrix[orb1, orb2 ] += -1j * t * V_ * lega * np.exp(-1j * K * x) / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb1_, orb1_, :])
                            lega = pos[orb1] - pos[orb1_] - x_
                            matrix[orb1, orb2] += 1j * t * V_ * lega * np.exp(-1j * K * x) / Nk * suma_n
    return matrix

def mf_matrix2(K, rho, phys_parameters):
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, Vb, Vc, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(2):
        for beta in range(2):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            suma_n = np.sum(rho[orb2, orb1, :] * np.exp(-1j*K*x))
                            lega = pos[orb2] - pos[orb1_] - x_
                            matrix[orb1_, orb1_] += -1j * t * V_ * lega / Nk * suma_n

                        if orb1 == orb2_:
                            suma_n = np.sum(rho[orb2, orb1, :] * np.exp(-1j*K*x))
                            lega = pos[orb1] - pos[orb1_] - x_
                            matrix[orb1_, orb1_] += 1j * t * V_ * lega  / Nk * suma_n
    return matrix


def mf_matrix3(K, rho, phys_parameters):
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, Vb, Vc, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)

    for alpha in range(2):
        for beta in range(2):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    f_k = t * np.exp(-1j * K * x) / Nk

                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - x_
                            g = -1j * V_ * lega * np.exp(1j * K * x) * np.exp(-1j * K * x_)
                            h = rho[orb1_, orb1, :]

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))

                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)

                            matrix[orb1_, orb2] +=  f_k * gh

                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - x_
                            g = 1j * V_ * lega * np.exp(-1j * K * x_)
                            h = rho[orb1_, orb1, :]

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))
                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)
                            matrix[orb1_, orb2] += f_k * gh
    return -matrix

def mf_matrix4(K, rho, phys_parameters):
    b, t, t_, t12, _, _, Vb, Vc, delta = phys_parameters
    pos, kinetic, interaction = parameters(b, t, t_, t12, Vb, Vc, delta)
    Nk = len(K)
    matrix = np.zeros((2, 2, Nk), dtype=np.complex128)
    for alpha in range(2):
        for beta in range(2):
            for line in kinetic:
                x, orb1, orb2, t = line
                x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
                if orb1 == orb2 and x == 0: pass
                if orb1 == alpha and orb2 == beta:
                    for line_ in interaction:
                        x_, orb1_, orb2_, V_ = line_
                        x_, orb1_, orb2_, V_ = float(x_), int(orb1_), int(orb2_), float(V_)

                        if orb2 == orb2_:
                            lega = pos[orb2] - pos[orb1_] - x_
                            g = -1j * V_ * lega * np.exp(-1j * K * x) * np.exp(1j * K * x_)
                            h = t * np.exp(-1j*K*x) * rho[orb2, orb1_, :] / Nk

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))

                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)

                            matrix[orb1, orb1_] +=  gh

                        if orb1 == orb2_:
                            lega = pos[orb1] - pos[orb1_] - x_
                            g = 1j * V_ * lega * np.exp(1j * K * x_)
                            h = t * np.exp(-1j*K*x) * rho[orb2, orb1_, :] / Nk

                            g_fft = np.fft.fft(np.fft.ifftshift(g))
                            h_fft = np.fft.fft(np.fft.ifftshift(h))
                            gh = np.fft.ifft(g_fft * h_fft)
                            gh = np.fft.fftshift(gh)
                            matrix[orb1, orb1_] += gh
    return -matrix

''' Lorentzian spectral function at fix momentum '''
@njit
def spektralna_k(epsilon, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros(N_orb)
    for orb in range(N_orb):
        A[orb] = 1/np.pi * Gamma / ( (epsilon - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

''' spectral function for all momenta and an array of epsilons '''
@njit(parallel=True, cache=True)
def Spektralka(epsilons, mu, energije, Gamma):
    Nk = energije.shape[1]
    Nepsilons = len(epsilons)
    A = np.zeros((Nepsilons, 2, Nk))
    for i in prange(Nepsilons):
        eps = epsilons[i]
        for m in range(Nk):
            A_k = spektralna_k(eps, mu, energije[:,m], Gamma)
            A[i,:,m] = A_k
    return A

''' Kubo transport function. I will input mat1=mat2=current '''
@njit(parallel=True, cache=True)
def phi_Kubo(K, mat1, mat2, spektralka, epsilons):
    Nk = len(K)
    phi = np.zeros(len(epsilons), dtype=np.complex128)
    A = spektralka

    for m in [0, Nk//2]:
        for a in range(2):
            for b in range(2):
                phi += (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m])
    for m in prange(1,Nk//2):
        for a in range(2):
            for b in range(2):
                phi += 2 * (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m])
    return phi / Nk

''' same as phi_Kubo, but neglecting overlap of spectral function in different bands '''
@njit(parallel=True, cache=True)
def phi_Kubo_diagonal(K, mat1, mat2, spektralka, omegas):
    Nk = len(K)
    phi = np.zeros(len(omegas), dtype=np.complex128)
    A = spektralka

    for m in [0, Nk//2]:
        for a in range(2):
            phi += (mat1[a,a,m] * A[:,a,m] * mat2[a,a,m] * A[:,a,m]).real
    for m in prange(1,Nk//2):
        for a in range(2):
            phi += 2 * (mat1[a,a,m] * A[:,a,m] * mat2[a,a,m] * A[:,a,m]).real
    return phi / Nk

''' operator in band basis obtained from operator in orbital basis '''
def operator_tilde(op_bare, vecs):
    op_tilde = np.empty_like(op_bare, dtype=np.complex128)
    if len(op_tilde.shape) > 3:
        # this means there are multiple operators
        n_ops = op_tilde.shape[0]
        for n in range(n_ops):
            op_tilde[n] = np.einsum('jix, jlx, lmx -> imx', vecs.conj(), op_bare[n], vecs)
    else:
        op_tilde = np.einsum('jix, jlx, lmx -> imx', vecs.conj(), op_bare, vecs)
    return op_tilde

''' df/domega, f is Fermi-Dirac distribution function '''
@njit(cache=True)
def fd_1(eps_or_omega, T, mu=0.0):
    eps = eps_or_omega - mu
    x = eps/T
    x_clipped = np.clip(x,-100,100)
    exp_x = np.exp(x_clipped)
    return - exp_x / (T * (1 + exp_x)**2)

''' Boltzmann transport function'''
def phi_Boltzmann(K, energije, mu, omegas, faktor=0.2, shape='Gaussian'):
    Nk = len(K)
    dK = K[1] - K[0]
    phi = np.zeros(len(omegas))
    vel1 = np.diff(energije[0]) / dK
    vel2 = np.diff(energije[1]) / dK

    vel = np.zeros((2, Nk))
    vel[0] = np.hstack([[0], vel1])
    vel[1] = np.hstack([[0], vel2])

    v_max = np.max(np.abs(vel))
    sigma = np.sqrt(v_max * (omegas[1] - omegas[0]) * dK) * faktor

    for i, omega in enumerate(omegas):
        for j in range(0,Nk//2+1):
            multiply = 1 if j in [0,Nk//2] else 2
            for alpha in [0,1]:
                phi[i] += multiply * delta_approximation(omega - energije[alpha,j] + mu, sigma, shape) * vel[alpha,j]**2
    return phi / Nk

def kahan_sum(vals):
    total = 0.0
    c = 0.0
    for x in vals:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def Kn_boltz(K, energije, mu, T):
    Nk = len(K)
    dK = K[1] - K[0]

    energije1 = energije[0]
    energije2 = energije[1]

    vel1 = np.diff(energije1) / dK
    vel2 = np.diff(energije2) / dK

    energije1 = energije1 - mu
    energije2 = energije2 - mu

    fd1 = -fd_1(energije1, T)
    fd2 = -fd_1(energije2, T)

    K0_terms = []
    K1_terms = []

    for j in [0, Nk//2]:
        K0_terms.append(fd1[j] * vel1[j]**2)
        K0_terms.append(fd2[j] * vel2[j]**2)
        K1_terms.append(fd1[j] * vel1[j]**2 * energije1[j])
        K1_terms.append(fd2[j] * vel2[j]**2 * energije2[j])

    for j in range(1, Nk//2):
        K0_terms.append(2 * (fd1[j] * vel1[j]**2 + fd2[j] * vel2[j]**2))
        K1_terms.append(2 * (fd1[j] * vel1[j]**2 * energije1[j]
                              + fd2[j] * vel2[j]**2 * energije2[j]))

    K0 = kahan_sum(K0_terms)
    K1 = kahan_sum(K1_terms)
    return K0 / Nk, K1 / Nk

# ====== response and susceptibility obtained from simulation of a pulse ======

''' Gaussian pulse modulated by a cosine (in practice, however, I choose Omega=0, i.e. the pulse is a Gaussian ''' 
def A_pulz(t, A0, t0, sigma, Omega0):
    return A0 * np.cos(Omega0 * t) * np.exp(-(t-t0)**2/(2*sigma**2))

def build_U(H, dt):
    Nk = H.shape[-1]
    U = np.empty_like(H)
    for m in range(Nk):
        H_m = H[:,:,m]
        U[:,:,m] = expm(-1j * H_m * dt)
    return U

@njit(parallel=True)
def evolve_rho_kernel2(U, rho):
    Nk = rho.shape[-1]
    rho_next = np.empty_like(rho)

    for m in prange(Nk):
        U_m = U[:,:,m]
        rho_next[:,:,m] = U_m @ rho[:,:,m] @ U_m.conj().T

    return rho_next 

''' evolving rho(t) into rho(t+dt). for 2-band system, this can be done analytically using decomposition of h_k into a linear combination of Pauli matrices '''
@njit(parallel=True)
def evolve_rho_kernel(Hk, rho, dt):
    Nk = rho.shape[2]

    rho_next = np.empty_like(rho)

    for j in prange(Nk):
        hk = Hk[:,:,j]

        # Decompose H = ε I + d·σ, ε I do not actually need
        dx  = 0.5 * (hk[0,1] + hk[1,0]).real
        dy  = -0.5 * (hk[0,1] - hk[1,0]).imag
        dz  = 0.5 * (hk[0,0] - hk[1,1]).real

        norm_d = np.sqrt(dx*dx + dy*dy + dz*dz)

        if norm_d < 1e-14:
            c = 1.0
            s = dt
        else:
            c = np.cos(norm_d * dt)
            s = np.sin(norm_d * dt) / norm_d

        #c = np.cos(norm_d * dt)
        #s = np.sin(norm_d * dt) / norm_d

        # Build U explicitly, using: U = cos(|d_k|dt) - isin(|d_k|dt) d_k*sigma / |d_k|
        U00 = c - 1j * s * dz
        U11 = c + 1j * s * dz
        U01 = -1j * s * (dx - 1j*dy)
        U10 = -1j * s * (dx + 1j*dy)

        # Apply U rho
        r00 = rho[0,0,j]
        r01 = rho[0,1,j]
        r10 = rho[1,0,j]
        r11 = rho[1,1,j]

        a00 = U00*r00 + U01*r10
        a01 = U00*r01 + U01*r11
        a10 = U10*r00 + U11*r10
        a11 = U10*r01 + U11*r11

        # Apply U† to U rho --> this is rho_next
        rho_next[0,0,j] = a00*U00.conjugate() + a01*U01.conjugate()
        rho_next[0,1,j] = a00*U10.conjugate() + a01*U11.conjugate()
        rho_next[1,0,j] = a10*U00.conjugate() + a11*U01.conjugate()
        rho_next[1,1,j] = a10*U10.conjugate() + a11*U11.conjugate()

    return rho_next

def relax_rho(rho, rho_eq, dt, Gamma):
    decay = np.exp(-Gamma * dt)
    return rho_eq + decay * (rho - rho_eq)

''' expectation value of measure_operators when system is described by density matrix rho'''
@njit(parallel=True)
def measure(Nk, Nop, measure_operators, rho):
    measurements_k = np.zeros((Nop, Nk), dtype=np.complex128)
    for j in prange(Nk):
        for n in range(Nop):
            # Tr[rho_k O_k]
            measurements_k[n, j] = (
                rho[0,0,j]*measure_operators[n,0,0,j] +
                rho[0,1,j]*measure_operators[n,1,0,j] +
                rho[1,0,j]*measure_operators[n,0,1,j] +
                rho[1,1,j]*measure_operators[n,1,1,j]
            )
    measurements = measurements_k.sum(axis=1)
    return measurements

@njit(parallel=True)
def norm(rho):
    Nk = rho.shape[-1]
    n = 0.0
    for m in prange(Nk):
        n += np.trace(rho[:,:,m])
    return n.real / Nk

''' main function which propagates the system and measures observables (measure_provider)
upon application of a perturbation (generated by perturbation_operator)
* do_freeze=True means that Hartree-Fock is frozen to its equilibrium value, hence we observe no corrections, e.g., in current-current response
* do_freeze=False is the opposite; Hartree-Fock is dynamic, i.e. densities respond to perturbations, and in this response the vertex corrections are captured
'''

def compile_measure_provider(measure_provider):

    if not isinstance(measure_provider, (list, tuple)):
        measure_provider = [measure_provider]

    static_ops = []
    dynamic_providers = []

    for item in measure_provider:

        if callable(item):
            dynamic_providers.append(item)

        else:
            ops = item
            if ops.ndim == 3:
                ops = ops[np.newaxis, ...]
            static_ops.append(ops)

    return static_ops, dynamic_providers

def simulate_pulz(K, hk0, rho, Vb, Vc, include_hartree,
                  perturbation_operator, measure_provider,
                  A0, t0, sigma, Omega, dt, t_max,
                  do_freeze, Ncorr, tol, geom, phases, g_ffts, Gamma=0.0, verbose=True, freq_verbose=50):
    N_points = int(t_max/dt)
    Nk = len(K)
    rho_eq = np.copy(rho)

    static_ops, dynamic_providers = compile_measure_provider(measure_provider)

    ops_list = []

    if len(static_ops) > 0:
        ops_list.append(np.concatenate(static_ops, axis=0))

    if do_freeze:
        # evaluate dynamic operators once
        for p in dynamic_providers:
            ops = p(K, rho, geom, phases, g_ffts)
            if ops.ndim == 3:
                ops = ops[np.newaxis, ...]
            ops_list.append(ops)

        measure_operators_fixed = np.concatenate(ops_list, axis=0)
        Nop = measure_operators_fixed.shape[0]

    else:
        # dynamic operators will be recomputed
        for p in dynamic_providers:
            ops = p(K, rho, geom, phases, g_ffts)
            if ops.ndim == 3:
                ops = ops[np.newaxis, ...]
            ops_list.append(ops)

        measure_operators = np.concatenate(ops_list, axis=0)
        Nop = measure_operators.shape[0]

    rho0 = np.copy(rho)
    H0 = h_k(K, hk0, rho0, Vb, Vc, 0.0, include_hartree)

    rho_expvals = np.zeros((Nop, N_points), dtype=np.complex128)
    rho_norms = np.zeros(N_points)
    Delta_bs = np.zeros(N_points, dtype=np.complex128)
    Delta_cs = np.zeros(N_points, dtype=np.complex128)
    ns0 = np.zeros(N_points)
    ns1 = np.zeros(N_points)

    ts = dt * np.arange(N_points)

    for i in range(N_points):

        if verbose and i % freq_verbose == 0:
            print(f'Progress: {i/N_points}', flush=True)

        A_t = A_pulz(i * dt, A0, t0, sigma, Omega)
        A_half = A_pulz(i * dt + dt/2, A0, t0, sigma, Omega)

        # Hamiltonian at current density
        if do_freeze:
            H_k0 = H0
        else:
            H_k0 = h_k(K, hk0, rho,Vb, Vc, 0., include_hartree)
        # ------------------
        # Predictor
        # ------------------

        if Gamma != 0.0:
            rho_half = relax_rho(rho, rho_eq, dt/2, Gamma)
            rho_pred = evolve_rho_kernel(H_k0 - A_t * perturbation_operator, rho_half, dt)
            rho_pred = relax_rho(rho_pred, rho_eq, dt/2, Gamma)
        else:
            rho_pred = evolve_rho_kernel(H_k0 - A_t * perturbation_operator, rho, dt)

        if do_freeze:
            H_k1 = H0
        else:
            H_k1 = h_k(K, hk0, rho_pred, Vb, Vc, 0., include_hartree)

        rho_guess = rho_pred

        # ------------------
        # Corrector iteration
        # ------------------

        for _ in range(Ncorr):

            H_mid = 0.5 * (H_k0 + H_k1) - A_half * perturbation_operator

            if Gamma != 0.0:
                rho_half = relax_rho(rho, rho_eq, dt/2, Gamma)
                rho_new = evolve_rho_kernel(H_mid, rho_half, dt)
                rho_new = relax_rho(rho_new, rho_eq, dt/2, Gamma)
            else:
                rho_new = evolve_rho_kernel(H_mid, rho, dt)

            err = np.max(np.abs(rho_new - rho_guess))
            rho_guess = rho_new

            if err < tol:
                break

            if not do_freeze:
                H_k1 = h_k(K, hk0, rho_guess, Vb, Vc, 0., include_hartree)

        rho = rho_guess

        # ------------------
        # Measurements
        # ------------------

        if do_freeze:
            measure_operators = measure_operators_fixed

        else:
            ops_list = []

            if len(static_ops) > 0:
                ops_list.append(np.concatenate(static_ops, axis=0))

            for p in dynamic_providers:
                ops = p(K, rho, geom, phases, g_ffts)
                if ops.ndim == 3:
                    ops = ops[np.newaxis, ...]
                ops_list.append(ops)

            measure_operators = np.concatenate(ops_list, axis=0)

        measurement_t = measure(Nk, Nop, measure_operators, rho)
        rho_expvals[:,i] = measurement_t
        rho_norms[i] = norm(rho)
        Delta_bs[i], Delta_cs[i] = Delta(K, rho, Vb, Vc)
        ns0[i] = np.sum(rho[0,0]).real / Nk
        ns1[i] = np.sum(rho[1,1]).real / Nk
        
    return ts, rho_expvals, rho_norms, Delta_bs, Delta_cs, ns0, ns1

''' susceptibility obtained from temporal response, using Fourier transform. window exp(-eta*t) is applied '''
def susceptibility(time, signal, probe, eta, omega_cut, Nk):
    dt = time[1] - time[0]
    window = np.exp(- eta * time)
    
    signal_omega = np.fft.fft((signal - signal[0]) * window * dt) / Nk
    probe_omega = np.fft.fft(probe * window * dt)

    omega = 2*np.pi*np.fft.fftfreq(len(time), d=dt)

    pos = (omega > 0) * (omega < omega_cut)
    omega = omega[pos]
    signal_omega = signal_omega[pos]
    probe_omega = probe_omega[pos]

    return omega, signal_omega, probe_omega

''' optical conductivity calculated from susceptibility obtained from temporal response'''
def optical_conductivity(time, signal, probe, eta, omega_cut, Nk):
    omega, signal_omega, probe_omega = susceptibility(time, signal, probe, eta, omega_cut, Nk)
    sigma_omega = signal_omega / (-1j * omega * probe_omega)

    return omega, sigma_omega.real

def drude_weight(K, vecs, energije, mu, T, kinetic):
    Nk = len(K)
    Norb = energije.shape[0]
    #build first and second derivatives of kinetic Hamiltonian
    dH_dk = np.zeros_like(vecs, dtype=np.complex128)
    d2H_dk2 = np.zeros_like(vecs, dtype=np.complex128)
    for line in kinetic:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        dH_dk[orb1,orb2] += -1j * t * x * np.exp(-1j*K*x)
        d2H_dk2[orb1,orb2] += - t * x**2 * np.exp(-1j*K*x)
    dH_dk_rotate = operator_tilde(dH_dk, vecs)
    d2H_dk2_rotate = operator_tilde(d2H_dk2, vecs)
    drude = 0.0
    for k in range(Nk):
        for orb in range(Norb):
            suma_k = 0.0
            suma_k += d2H_dk2_rotate[orb,orb,k]
            for orb_ in range(Norb):
                if orb != orb_:
                    suma_k += np.abs(dH_dk_rotate[orb_,orb,k])**2 / (energije[orb,k] - energije[orb_,k])**2
            drude += suma_k * fd(energije[orb,k], mu, T)
    return drude

def integral_omega(integrand, omega):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(integrand.real, omega)
    else:
        return np.trapz(integrand.real, omega)

''' this is just helpers to get local maxima and local minima. I use this to get the envelope of the response '''
def local_minima(arr):
    n = len(arr)
    indices, vals = [], []
    for i in range(n):
        if i > 0 and arr[i] > arr[i - 1]:
            continue
        if i < n - 1 and arr[i] >= arr[i + 1]:
            continue
        indices.append(i)
        vals.append(arr[i])
    return np.array(indices), np.array(vals)

''' Pauli matrices '''
sigmas = np.zeros((4, 2, 2), dtype=np.complex128)
sigmas[0] = np.eye(2)
sigmas[1] = np.array([[0,1],[1,0]])
sigmas[2] = np.array([[0,-1j], [1j,0]])
sigmas[3] = np.diag([1,-1])

def rho_operators(K, phys_parameters, include_hartree):

    _, _, _, _, _, _, Vb, Vc, _ = phys_parameters

    if include_hartree == True:
        thetas = np.array([Vb/2, -Vb/2, -Vb/2, -Vb/2,
                            Vc/2, -Vc/2, -Vc/2, -Vc/2])
        if Vc == 0:
            thetas = thetas[:4]
        nus = [0, 1, 2, 3]
    else:
        thetas = np.array([Vb/2, -Vb/2, -Vc/2, -Vc/2])
        if Vc == 0:
            thetas = thetas[:2]
        nus = [1, 2]

    if Vc == 0:
        deltas = [0]
    else: deltas = [0, 1]

    rhos = np.zeros((len(thetas), 2, 2, len(K)), dtype=np.complex128)
    for i, delta in enumerate(deltas):
        for j, nu in enumerate(nus):
            ind = len(nus) * i + j
            U_kdelta = np.zeros((2, 2, len(K)), dtype=np.complex128)
            U_kdelta[0,0] = np.exp(-1j*K*delta/2)
            U_kdelta[1,1] = np.exp(1j*K*delta/2)
            Rho = np.einsum('ijx, jl, klx -> ikx', U_kdelta, sigmas[nu], U_kdelta.conj())
            rhos[ind] = Rho
    return rhos, thetas

''' Fermi-Dirac function '''
@njit
def fd(eps, mu, T):
    return 1.0 / (np.exp((eps - mu) / T) + 1.0)

@njit(cache=True)
def Pi_bubble_tilde(omega, E_mk, E_nk, Gamma, mu_, invt, nodes, weights, eps=1e-5, n_eps=1.0):
    w    = omega / Gamma
    e_mk = E_mk  / Gamma
    e_nk = E_nk  / Gamma
    
    T = Gamma / invt
    
    invpi = 1.0 / np.pi

    # Single, T-independent cutoff based on Lorentzian tail
    # A(e) ~ 1/(pi * e^2) < eps  =>  e > 1/(pi*eps)
    epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T) / Gamma * n_eps

    # Three integration intervals, one per peak
    centers = np.array([e_mk, e_nk - w, mu_])

    # Build, sort, merge intervals (same as your new code)
    raw = np.empty((3, 2), dtype=np.float64)
    for c in range(3):
        raw[c, 0] = centers[c] - epsilon_max
        raw[c, 1] = centers[c] + epsilon_max

    # Sort by left endpoint
    for i in range(3):
        for j in range(i + 1, 3):
            if raw[j, 0] < raw[i, 0]:
                raw[i, 0], raw[j, 0] = raw[j, 0], raw[i, 0]
                raw[i, 1], raw[j, 1] = raw[j, 1], raw[i, 1]

    # Merge overlapping intervals
    merged  = np.empty((3, 2), dtype=np.float64)
    merged[0, 0] = raw[0, 0]
    merged[0, 1] = raw[0, 1]
    n_merged = 1
    for i in range(1, 3):
        if raw[i, 0] <= merged[n_merged - 1, 1]:
            merged[n_merged - 1, 1] = max(raw[i, 1], merged[n_merged - 1, 1])
        else:
            merged[n_merged, 0] = raw[i, 0]
            merged[n_merged, 1] = raw[i, 1]
            n_merged += 1

    # Integrate over merged intervals
    n_nodes  = len(nodes)

    res_mn_r = 0.0; res_mn_i = 0.0
    res_nm_r = 0.0; res_nm_i = 0.0
    res_w_mn_r = 0.0; res_w_mn_i = 0.0
    res_w_nm_r = 0.0; res_w_nm_i = 0.0

    for s in range(n_merged):
        a    = merged[s, 0]
        b    = merged[s, 1]
        mid  = 0.5 * (a + b)
        half = 0.5 * (b - a)

        for i in range(n_nodes):
            e   = mid + half * nodes[i]
            ew  = e + w
            dm  = e - e_mk
            dn  = e - e_nk
            dmw = dm + w
            dnw = dn + w
            pref = e - mu_ + 0.5 * w
            wi   = weights[i] * half

            f  = 1.0 / (np.exp((e  - mu_) * invt) + 1.0)
            fw = 1.0 / (np.exp((ew - mu_) * invt) + 1.0)

            A_mk  = invpi / (dm  * dm  + 1.0)
            A_nk  = invpi / (dn  * dn  + 1.0)
            A_mkw = invpi / (dmw * dmw + 1.0)
            A_nkw = invpi / (dnw * dnw + 1.0)

            Grnw_r =  dnw / (dnw * dnw + 1.0)
            Grnw_i = -1.0 / (dnw * dnw + 1.0)
            Grmw_r =  dmw / (dmw * dmw + 1.0)
            Grmw_i = -1.0 / (dmw * dmw + 1.0)
            Gam_r  =  dm  / (dm  * dm  + 1.0)
            Gam_i  =  1.0 / (dm  * dm  + 1.0)
            Gan_r  =  dn  / (dn  * dn  + 1.0)
            Gan_i  =  1.0 / (dn  * dn  + 1.0)

            mn_r = A_mk * Grnw_r * f + A_nkw * Gam_r * fw
            mn_i = A_mk * Grnw_i * f + A_nkw * Gam_i * fw
            nm_r = A_nk * Grmw_r * f + A_mkw * Gan_r * fw
            nm_i = A_nk * Grmw_i * f + A_mkw * Gan_i * fw

            res_mn_r   += wi * mn_r
            res_mn_i   += wi * mn_i
            res_nm_r   += wi * nm_r
            res_nm_i   += wi * nm_i
            res_w_mn_r += wi * pref * mn_r
            res_w_mn_i += wi * pref * mn_i
            res_w_nm_r += wi * pref * nm_r
            res_w_nm_i += wi * pref * nm_i

    res_mn   = (res_mn_r   + 1j * res_mn_i)   / Gamma
    res_nm   = (res_nm_r   + 1j * res_nm_i)   / Gamma
    res_w_mn =  res_w_mn_r + 1j * res_w_mn_i
    res_w_nm =  res_w_nm_r + 1j * res_w_nm_i

    return res_mn, res_nm, res_w_mn, res_w_nm

@njit(parallel=True, cache=True)
def precompute_Pi_all(omega, energije, Gamma, mu_, invt, nodes, weights, eps=1e-5):
    Norb, Nk = energije.shape

    pi_mn  = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    pi_nm  = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    piw_mn = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    piw_nm = np.zeros((Norb, Norb, Nk), dtype=np.complex128)

    for j in prange(Nk):
        for m in range(Norb):
            for n in range(m, Norb):

                pi_mnk, pi_nmk, pie_mnk, pie_nmk = Pi_bubble_tilde(omega, energije[m,j], energije[n,j], Gamma, mu_, invt, nodes, weights, eps)

                pi_mn [m, n, j] = pi_mnk
                pi_nm [m, n, j] = pi_nmk
                piw_mn[m, n, j] = pie_mnk
                piw_nm[m, n, j] = pie_nmk
    return pi_mn, pi_nm, piw_mn, piw_nm

@njit(parallel=True, cache=True)
def chi_UV(Nk, U, V, pi_mn, pi_nm):
    Norb = U.shape[-2]
    chi = 0.0 + 0.0j

    for j in prange(Nk):
        chi_j = 0.0 + 0.0j
        for m in range(Norb):
            for n in range(m, Norb):

                U_mn = U[m, n, j]
                U_nm = U[n, m, j]
                V_mn = V[m, n, j]
                V_nm = V[n, m, j]

                p_mn = pi_mn[m, n, j]  
                p_nm = pi_nm[m, n, j]

                if m == n:
                    chi_j += 0.5 * (U_mn * V_nm * p_nm + U_nm * V_mn * p_mn)
                else:
                    chi_j += U_mn * V_nm * p_nm + U_nm * V_mn * p_mn

        chi += chi_j

    return chi / Nk

def compute_single_om_fused(
    om,
    Nk, Gamma, mu_, invt, nodes, weights,
    thetas, tok_tilde, mat_tilde,
    energije,
    rhos_tilde,
    gbx=None, omega_bx=None,
    gby=None, omega_by=None,
    gcx=None, omega_cx=None,
    gcy=None, omega_cy=None,
    eps=1e-5, phonon=None, include_hartree=True, Vb=None, Vc=None, Gamma_ph=None
):
    Nop = len(thetas)
    
    if not phonon:
        thetas_diag = np.diag(thetas)
    else:
        thetas_diag = np.diag(thetas_phonon(om, include_hartree, Vb, Vc, gbx, omega_bx, gby, omega_by, gcx, omega_cx, gcy, omega_cy, Gamma_ph))
    I = np.eye(Nop)

    # ── ONE precomputation pass for this omega ──────────────────────────
    pi_mn, pi_nm, piw_mn, piw_nm = precompute_Pi_all(
        om, energije, Gamma, mu_, invt, nodes, weights, eps
    )

    # ── chi0 matrix  (Nop x Nop calls, but now cheap) ──────────────────
    chi0 = np.zeros((Nop, Nop), dtype=np.complex128)
    for i in range(Nop):
        for j in range(Nop):
            chi0[i, j] = chi_UV(Nk, rhos_tilde[i], rhos_tilde[j], pi_mn, pi_nm)

    # ── chi_jj0 ────────────────────────────────────────────────────────
    chi_jj0 = chi_UV(Nk, tok_tilde, tok_tilde, pi_mn, pi_nm)
    chi_jEj0 = chi_UV(Nk, tok_tilde, tok_tilde, piw_mn, piw_nm)
    chi_matj0 = chi_UV(Nk, mat_tilde, tok_tilde, pi_mn, pi_nm)

    # ── chi_jrho0 / chi_rhoj0 ──────────────────────────────────────────
    chi_jrho0 = np.zeros(Nop, dtype=np.complex128)
    chi_rhoj0 = np.zeros(Nop, dtype=np.complex128)
    chi_jErho0 = np.zeros(Nop, dtype=np.complex128)
    chi_matrho0 = np.zeros(Nop, dtype=np.complex128)
    for i in range(Nop):
        chi_jrho0[i] = chi_UV(Nk, tok_tilde,    rhos_tilde[i], pi_mn, pi_nm)
        chi_rhoj0[i] = chi_UV(Nk, rhos_tilde[i], tok_tilde,    pi_mn, pi_nm)
        chi_jErho0[i] = chi_UV(Nk, tok_tilde,    rhos_tilde[i], piw_mn, piw_nm)
        chi_matrho0[i] = chi_UV(Nk, mat_tilde, rhos_tilde[i], pi_mn, pi_nm)

    # ── RPA ────────────────────────────────────────────────────────────
    mat     = I - chi0 @ thetas_diag
    inv     = LA.inv(mat)
    chi_rpa = inv @ chi0
    dchi_jj = chi_jrho0 @ thetas_diag @ inv @ chi_rhoj0
    dchi_jEj = chi_jErho0 @ thetas_diag @ inv @ chi_rhoj0
    dchi_matj = chi_matrho0 @ thetas_diag @ inv @ chi_rhoj0

    return om, chi0, chi_rpa, chi_jj0, dchi_jj, chi_jEj0, dchi_jEj, chi_matj0, dchi_matj, chi_rhoj0

def compute_chi(
    omegas,
    Nk, Gamma, mu_, invt, nodes, weights,
    thetas, tok_tilde, mat_tilde,
    energije,
    rhos_tilde,
    verbose=True,
    gbx=None, omega_bx=None,
    gby=None, omega_by=None,
    gcx=None, omega_cx=None,
    gcy=None, omega_cy=None,
    phonon=False, include_hartree=True,
    n_workers=None, #None: number of CPU cores, or specify an integer
    eps=1e-5, Vb=None, Vc=None, Gamma_ph=None
):
    omegas = np.asarray(omegas)
    N_om   = len(omegas)
    Nop    = len(thetas)

    chi0_arr      = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_rpa_arr   = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_jj0_arr   = np.zeros(N_om,             dtype=np.complex128)
    dchi_jj_arr   = np.zeros(N_om,             dtype=np.complex128)
    chi_matj0_arr = np.zeros(N_om, dtype=np.complex128)
    chi_jEj0_arr = np.zeros(N_om, dtype=np.complex128)
    dchi_matj_arr = np.zeros(N_om, dtype=np.complex128)
    dchi_jEj_arr = np.zeros(N_om, dtype=np.complex128)

    chi_rhoj0_arr = np.zeros((N_om, Nop), dtype=np.complex128)

    t_total = time.time()

    def _worker(om_idx, om):
        result = compute_single_om_fused(
            om,        # om = frequency value, omegas = full array
            Nk, Gamma, mu_, invt, nodes, weights,
            thetas, tok_tilde, mat_tilde,
            energije, rhos_tilde,
            gbx=gbx, omega_bx=omega_bx, gby=gby, omega_by=omega_by, gcx=gcx, omega_cx=omega_cx, gcy=gcy, omega_cy=omega_cy,
            eps=eps, phonon=phonon, include_hartree=include_hartree, Vb=Vb, Vc=Vc, Gamma_ph=Gamma_ph
        )
        return om_idx, result


    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_worker, om_idx, om): om_idx
            for om_idx, om in enumerate(omegas)
        }

        with tqdm(total=N_om, desc="omegas", disable=not verbose) as pbar:
            for future in as_completed(futures):
                om_idx, result = future.result()
                om, chi0, chi_rpa, chi_jj0, dchi_jj, chi_jEj0, dchi_jEj, chi_matj0, dchi_matj, chi_rhoj0 = result

                chi0_arr[om_idx]      = chi0
                chi_rpa_arr[om_idx]   = chi_rpa
                chi_jj0_arr[om_idx]   = chi_jj0
                dchi_jj_arr[om_idx]   = dchi_jj
                chi_jEj0_arr[om_idx] = chi_jEj0
                dchi_jEj_arr[om_idx] = dchi_jEj
                chi_matj0_arr[om_idx] = chi_matj0
                dchi_matj_arr[om_idx] = dchi_matj
                chi_rhoj0_arr[om_idx] = chi_rhoj0

                if verbose:
                    print(
                        f"  [om {om_idx+1}/{N_om}]",
                        flush=True
                    )
                pbar.update(1)

    if verbose:
        print(f"\nTotal time: {time.time() - t_total:.2f}s")

    results = {'chi0' : chi0_arr,
               'chi' : chi_rpa_arr,

               'chi_jj0' : chi_jj0_arr,
               'dchi_jj' : dchi_jj_arr,

               'chi_jEj0' : chi_jEj0_arr,
               'chi_matj0' : chi_matj0_arr,

               'dchi_matj' : dchi_matj_arr,
               'dchi_jEj' : dchi_jEj_arr,

               'chi_rhoj0' : chi_rhoj0_arr
               }
    
    return results

def find_flat_regime(omegas, chi_omega, window=10):
    """
    Find flattest window in dchi_domega, using sliding window in LOG omega space.
    Relative std (std/|mean|) is minimized in the flat region.
    """
    dchi_domega = np.gradient(chi_omega, omegas)
    n          = len(omegas)
    rel_std    = np.full(n, np.nan)

    for i in range(n - window):
        chunk      = dchi_domega[i : i + window]
        mean       = np.mean(chunk)
        std        = np.std(chunk)
        rel_std[i] = std / np.abs(mean)

    # Best window = minimum relative std
    best_start = np.nanargmin(rel_std)
    best_end   = best_start + window

    # Expand window outward as long as rel_std stays low
    threshold = rel_std[best_start] * 3   # allow 3x the minimum rel_std
    
    # expand left
    left = best_start
    while left > 0 and rel_std[left - 1] < threshold:
        left -= 1
    
    # expand right
    right = best_end
    while right < n - window and rel_std[right] < threshold:
        right += 1

    return left, right

def get_dc_coefficient(omegas, chi_imag, omega_cutoff=None):
    """
    Get DC coefficient (alpha = chi_imag / omega as omega -> 0)
    for log-spaced omega arrays using weighted linear regression.
    
    Fits: chi_imag = alpha * omega + beta * omega^3
    Weights = 1/omega to ensure equal contribution per decade.
    """
    
    # Select low-frequency window
    if omega_cutoff is None:
        log_min = np.log10(omegas.min())
        log_max = np.log10(omegas.max())
        omega_cutoff = 10 ** (log_min + 0.2 * (log_max - log_min))
    
    mask = omegas <= omega_cutoff
    w    = 1.0 / omegas[mask]          # weights: uniform per decade
    x    = omegas[mask]
    y    = chi_imag[mask]

    # Weighted least squares: chi_imag = alpha * omega + beta * omega^3
    # Design matrix
    A  = np.column_stack([x, x**3])
    Aw = A * w[:, None]                # apply weights to rows
    yw = y * w

    # Solve weighted normal equations
    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    alpha, beta = coeffs

    return alpha, beta

def find_DC_limit(omega0, chi_imag):
    left, right = find_flat_regime(omega0, chi_imag)
    return get_dc_coefficient(omega0[left:right], chi_imag[left:right])[0]
    
# adding phonon
def D0(omega, omega_ph, Gamma_ph):
    return 2 * omega_ph / ((omega + 1j * Gamma_ph)**2 - omega_ph**2)

def thetas_phonon(omega, include_hartree, Vb, Vc,
                  gbx, omega_bx, gby, omega_by, gcx, omega_cx, gcy, omega_cy, Gamma_ph):
    if include_hartree:
        thetas = np.zeros(8, dtype=np.complex128)
        thetas[0] = Vb/2 + Vc/2
        thetas[3] = Vb + Vc/2
    else:
        thetas = np.zeros(4, dtype=np.complex128)

    Dbx = D0(omega, omega_bx, Gamma_ph)
    Dby = D0(omega, omega_by, Gamma_ph)
    Dcx = D0(omega, omega_cx, Gamma_ph)
    Dcy = D0(omega, omega_cy, Gamma_ph)

    if include_hartree:
        thetas[1] = gbx**2 * Dbx
        thetas[2] = gby**2 * Dby
        thetas[5] = gcx**2 * Dcx
        thetas[6] = gcy**2 * Dcy
    else:
        thetas[0] = gbx**2 * Dbx
        thetas[1] = gby**2 * Dby
        thetas[2] = gcx**2 * Dcx
        thetas[3] = gcy**2 * Dcy

    return thetas