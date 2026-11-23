import numpy as np
import scipy.linalg as LA
import numba
from numba import njit, prange
from scipy.linalg import expm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import linalg as LA
from scipy.special import psi as _psi

from takarada_helpers import h_k, Delta, delta_approximation

''' current operator '''
@njit(cache=True)
def j_tok(K, pos, kinetic):
    Nk = len(K)
    j = np.zeros((2, 2, Nk), dtype=np.complex128)
    for line in kinetic:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        ad = - 1j * t * np.exp(-1j * K * x) * (pos[orb1] - pos[orb2] + x)
        j[orb1, orb2] += ad
    return j

''' kinetic energy current operator '''
@njit(cache=True)
def jK_tok(K, pos, kinetic, epsilon, epsilon_, mu, hartree_shift0, hartree_shift1):
    # Keep hopping and onsite terms in one homogeneous array for Numba.
    kinetic_full = np.zeros((len(kinetic) + 2, 4), dtype=np.float64)
    kinetic_full[:len(kinetic)] = kinetic
    kinetic_full[-2, 3] = -epsilon - mu + hartree_shift0
    kinetic_full[-1, 1] = 1.0
    kinetic_full[-1, 2] = 1.0
    kinetic_full[-1, 3] = epsilon_ - mu + hartree_shift1
    Nk = len(K)
    jK = np.zeros((2, 2, Nk), dtype=np.complex128)
    for line in kinetic_full:
        x, orb1, orb2, t = line
        x, orb1, orb2, t = float(x), int(orb1), int(orb2), float(t)
        for line_ in kinetic_full:
            x_, orb1_, orb2_, t_ = line_
            x_, orb1_, orb2_, t_ = float(x_), int(orb1_), int(orb2_), float(t_)
            if orb2==orb1_:
                ad = -1j * 0.5 * t * t_ * np.exp(-1j*K*(x+x_)) * (pos[orb1] - pos[orb2_] + x + x_)
                jK[orb1,orb2_] += ad
    return jK

def kinetic_current_provider(epsilon, epsilon_, mu, Vb, Vc, include_hartree):
    """Build a pulse measurement provider using the instantaneous Hartree shifts."""
    def provider(K, rho, geom, phases, g_ffts, impose_deltas=False, impose_additional=True):
        shift0 = shift1 = 0.0
        if include_hartree:
            shift0 = (Vb + Vc) * np.sum(rho[1, 1]).real / len(K)
            shift1 = (Vb + Vc) * np.sum(rho[0, 0]).real / len(K)
        return jK_tok(K, geom["pos"], geom["kinetic"],
                      epsilon, epsilon_, mu, shift0, shift1)

    return provider

#@njit(cache=True)
def input_data(K, pos, kinetic, interaction):
    Nk = len(K)
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

''' below is stuff for mean-field approximation of the non-local interaction current operator '''
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

''' I thought imposing deltas by hand is necessary but it doesn't make a change (so impose_deltas=True,False give the same) '''
def compute_all_mf_matrices(K, rho, geom, phases, g_ffts, impose_deltas=False, impose_additional=True):
    Nk = len(K)
    lege = geom["pos"]
    M3  = np.zeros((2,2,Nk), dtype=np.complex128)
    M6  = np.zeros((2,2,Nk), dtype=np.complex128)
    M4a = np.zeros((2,2,Nk), dtype=np.complex128)
    M4b = np.zeros((2,2,Nk), dtype=np.complex128)

    M3_nonlocI  = np.zeros((2,2,Nk), dtype=np.complex128)
    M6_nonlocI  = np.zeros((2,2,Nk), dtype=np.complex128)
    M4a_nonlocI = np.zeros((2,2,Nk), dtype=np.complex128)
    M4b_nonlocI = np.zeros((2,2,Nk), dtype=np.complex128)

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

            # first term in square bracket --> t_{alpha beta} * V_{alpha' beta}
            if orb2 == orb2_:

                deltas=1.0
                if impose_deltas:
                    # check if the delta functions coming from CAR prohibit these MF terms
                    if orb1==orb1_ and x==x_: deltas=0.0 # correct
                    elif orb1_==orb2 and x_==0: deltas=0.0 # correct

                if deltas!=0.0:
                    lega = lege[orb2] - lege[orb1_] - x_
                    lega_nonlocI = lege[orb1] - lege[orb2] + x

                    # ---------- M3 ---------- first term in equation (ell=1 term)
                    M3[orb1,orb2] += -1j * t * V_ * lega * phase_k * n[orb1_] / Nk
                    M3_nonlocI[orb1,orb2] += -1j * t * V_ * lega_nonlocI * phase_k * n[orb1_] / Nk

                    # ---------- M6 ---------- second term in equation (ell=2 term)
                    suma = np.sum(rho[orb2,orb1,:] * phase_k)
                    M6[orb1_,orb1_] += -1j * t * V_ * lega * suma / Nk
                    M6_nonlocI[orb1_,orb1_] += -1j * t * V_ * lega_nonlocI * suma / Nk

                    # ---------- M4a ---------- fourth term in equation (ell=3 term)
                    if impose_additional and orb1==orb1_ and x==x_ and x==0.0:
                        faktor=0.0
                    else:
                        faktor=1.0
                    #g = -1j * V_ * lega * np.conj(phase_k) * phases["int"][m]
                    g_fft = -1j * V_ * lega * g_ffts_M4a1[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                    g_fft_nonlocI = -1j * V_ * lega_nonlocI * g_ffts_M4a1[l,m,:]
                    gh = np.fft.fftshift(
                            np.fft.ifft(g_fft * rho_fft[(orb1_,orb1)]))
                    gh_nonlocI = np.fft.fftshift(
                            np.fft.ifft(g_fft_nonlocI * rho_fft[(orb1_,orb1)]))
                    M4a[orb1_,orb2] += fk * gh * faktor
                    M4a_nonlocI[orb1_,orb2] += fk * gh_nonlocI * faktor

                    # ---------- M4b ---------- third term in equation (ell=4 term)
                    if impose_additional and orb1_==orb2 and x_==0.0 and x==0.0:
                        faktor=0.0
                    else:
                        faktor=1.0
                    h = fk * rho[orb2,orb1_,:]
                    h_fft = np.fft.fft(np.fft.ifftshift(h))
                    g_fft = -1j * V_ * lega * g_ffts_M4b1[l,m,:]
                    g_fft_nonlocI = -1j * V_ * lega_nonlocI * g_ffts_M4b1[l,m,:]
                    #g_fft = np.fft.fft(np.fft.ifftshift(
                    #            -1j * V_ * lega * np.conj(phases["int"][m]) * phases["kin"][l]
                    #        ))
                    gh = np.fft.fftshift(np.fft.ifft(g_fft * h_fft))
                    gh_nonlocI = np.fft.fftshift(np.fft.ifft(g_fft_nonlocI * h_fft))
                    M4b[orb1,orb1_] += gh * faktor
                    M4b_nonlocI[orb1,orb1_] += gh_nonlocI * faktor

            # first term in square bracket --> t_{alpha beta} * V_{alpha' alpha}
            if orb1 == orb2_:
                deltas=1.0
                if impose_deltas:
                    # check if the delta functions coming from CAR prohibit these MF terms
                    if orb1==orb1_ and x_==0.0: deltas=0.0 # correct
                    if orb1_==orb2 and x+x_==0.0: deltas=0.0 # correct
                    
                if deltas!=0.0:
                    lega = lege[orb1] - lege[orb1_] - x_
                    lega_nonlocI = lege[orb2] - lege[orb1] - x
                    
                    # ---------- M3 ---------- first term in equation (ell=1 term)
                    M3[orb1,orb2] += +1j * t * V_ * lega * phase_k * n[orb1_] / Nk
                    M3_nonlocI[orb1,orb2] += +1j * t * V_ * lega_nonlocI * phase_k * n[orb1_] / Nk

                    # ---------- M6 ---------- second term in equation (ell=2 term)
                    suma = np.sum(rho[orb2,orb1,:] * phase_k)
                    M6[orb1_,orb1_] += +1j * t * V_ * lega * suma / Nk
                    M6_nonlocI[orb1_,orb1_] += +1j * t * V_ * lega_nonlocI * suma / Nk

                    # ---------- M4a ---------- fourth term in equation (ell=3 term)
                    if impose_additional and orb1_==orb1 and x_==0.0 and x==0.0:
                        faktor=0.0
                    else:
                        faktor=1.0
                    #g = +1j * V_ * lega * phases["int"][m]
                    g_fft = +1j * V_ * lega * g_ffts_M4a2[l,m,:] #np.fft.fft(np.fft.ifftshift(g))
                    g_fft_nonlocI = +1j * V_ * lega_nonlocI * g_ffts_M4a2[l,m,:]
                    gh = np.fft.fftshift(
                            np.fft.ifft(g_fft * rho_fft[(orb1_,orb1)]))
                    gh_nonlocI = np.fft.fftshift(
                            np.fft.ifft(g_fft_nonlocI * rho_fft[(orb1_,orb1)]))
                    M4a[orb1_,orb2] += fk * gh * faktor
                    M4a_nonlocI[orb1_,orb2] += fk * gh_nonlocI * faktor

                    # ---------- M4b ---------- third term in equation (ell=4 term)
                    #g = +1j * V_ * lega * np.conj(phases["int"][m])
                    if impose_additional and orb1_==orb2 and x+x_==0.0 and x==0.0:
                        faktor=0.0
                    else:
                        faktor=1.0
                    g_fft = +1j * V_ * lega * g_ffts_M4b2[l,m,:]#np.fft.fft(np.fft.ifftshift(g))
                    g_fft_nonlocI = +1j * V_ * lega_nonlocI * g_ffts_M4b2[l,m,:]#np.fft.fft(np.fft.ifftshift(g))
                    h = fk * rho[orb2,orb1_,:]
                    h_fft = np.fft.fft(np.fft.ifftshift(h))
                    gh = np.fft.fftshift(
                                np.fft.ifft(g_fft * h_fft)
                    )
                    gh_nonlocI = np.fft.fftshift(
                                np.fft.ifft(g_fft_nonlocI * h_fft)
                    )
                    M4b[orb1,orb1_] += gh * faktor
                    M4b_nonlocI[orb1,orb1_] += gh_nonlocI * faktor

    return M3, M6, -M4a, -M4b, M3_nonlocI, M6_nonlocI, -M4a_nonlocI, -M4b_nonlocI


def compute_equilibrium_interaction_current(K, rho, geom, phases, g_ffts,
                                            impose_deltas=False,
                                            impose_additional=True,
                                            add_delta_J2=False, rho_eq=None):
    """Return the full one-body interaction-current vertex used by RPA.
    add_delta_J2=True adds delta_J2_tok (the <n>-subtraction in H_int, P_int), with <n> from rho_eq
    (default: rho). Recommended: impose_additional=False, add_delta_J2=True; the two together would
    subtract the Hartree part twice and are refused."""
    if add_delta_J2 and impose_additional:
        raise ValueError("add_delta_J2=True with impose_additional=True subtracts the Hartree part twice")
    m3, m6, m4a, m4b, m3_nonlocI, m6_nonlocI, m4a_nonlocI, m4b_nonlocI = compute_all_mf_matrices(
        K, rho, geom, phases, g_ffts,
        impose_deltas=impose_deltas,
        impose_additional=impose_additional)
    if impose_additional:
        return m6 + m4a + m4b, m6_nonlocI + m4a_nonlocI + m4b_nonlocI
    mat = m3 + m6 + m4a + m4b
    mat_ = m3_nonlocI + m6_nonlocI + m4a_nonlocI + m4b_nonlocI
    if add_delta_J2:
        dJ, dJ_ = delta_J2_tok(K, rho if rho_eq is None else rho_eq,
                               geom["pos"], geom["kinetic"], geom["interaction"])
        mat, mat_ = mat + dJ, mat_ + dJ_
    return mat, mat_

def compute_together_mf_matrices(K, rho, geom, phases, g_ffts,
                                 impose_deltas=False, impose_additional=True):
    interaction_current, interaction_current_ = compute_equilibrium_interaction_current(
        K, rho, geom, phases, g_ffts,
        impose_deltas=impose_deltas,
        impose_additional=impose_additional)
    return 0.5 * np.stack((interaction_current, interaction_current_), axis=0)

def delta_J2_tok(K, rho, pos, kinetic, interaction, exclude_same_bond=True):
    ''' Change of J2 under n -> n - <n> in H_int and P_int: dJ2 (p-terms, interaction-bond vector, goes
        with mat) and dJ2_ (mu-terms, kinetic-bond vector, goes with mat_); dJ2 = -m3, dJ2_ = -m3_.
        exclude_same_bond: for each hopping line (x,a,b) leave out the interaction lines on the same bond,
        (x,a,b) and (-x,b,a). In J2 their Hartree terms (m3) cancel exactly against their exchange terms
        (m4a+m4b), as Pauli requires ([c_a^+ c_b, n_a n_b] = 0); subtracting only the Hartree half would leave
        the exchange half uncompensated. dJ2 + dJ2_ is unchanged (X_int = X_kin for these pairs). '''
    Nk = len(K)
    n = np.array([np.sum(rho[0, 0]).real, np.sum(rho[1, 1]).real]) / Nk

    dJ2 = np.zeros((2, 2, Nk), dtype=np.complex128)
    dJ2_ = np.zeros((2, 2, Nk), dtype=np.complex128)
    for x, orb1, orb2, t in kinetic:
        orb1, orb2 = int(orb1), int(orb2)
        ms = np.zeros(2)
        ps = np.zeros(2)
        for x_, orb1_, orb2_, V_ in interaction:
            orb1_, orb2_ = int(orb1_), int(orb2_)
            if exclude_same_bond and ((orb1_ == orb1 and orb2_ == orb2 and x_ == x) or
                                      (orb1_ == orb2 and orb2_ == orb1 and x_ == -x)):
                continue
            ms[orb1_] += V_ * n[orb2_]
            ps[orb1_] += V_ * n[orb2_] * (x_ + pos[orb1_] - pos[orb2_])
        phase = np.exp(-1j * K * x)
        lega_kin = x + pos[orb1] - pos[orb2]
        dJ2[orb1,orb2] += 1j * t * (- ps[orb1] + ps[orb2]) * phase
        dJ2_[orb1,orb2] += 1j * t * (ms[orb1] + ms[orb2]) * lega_kin  * phase
    return dJ2, dJ2_

''' functions mf_matrix1,2,3,4 give exactly the same as function compute_all_mf_matrices,
but the latter is more convenient (faster if used for multiple calls) because it uses precomputed elements '''
def mf_matrix1(K, rho, pos, kinetic, interaction):
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

def mf_matrix2(K, rho, pos, kinetic, interaction):
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


def mf_matrix3(K, rho, pos, kinetic, interaction):
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

def mf_matrix4(K, rho, pos, kinetic, interaction):
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

''' Lorentzian spectral function at fixed momentum '''
@njit
def spektralna_k(epsilon, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros(N_orb)
    for orb in range(N_orb):
        A[orb] = 1/np.pi * Gamma / ( (epsilon - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

''' spectral function for all momenta '''
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

''' Kubo transport function, coming from bubble between two single-body operators
    e.g. for K0 I will input mat1=mat2=current  '''
@njit(parallel=True, cache=True)
def phi_Kubo(K, mat1, mat2, spektralka, epsilons):
    Nk = len(K)
    phi = np.zeros(len(epsilons), dtype=np.complex128)
    A = spektralka

    '''for m in [0, Nk//2]:
        for a in range(2):
            for b in range(2):
                phi += (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m])
    for m in prange(1,Nk//2):
        for a in range(2):
            for b in range(2):
                phi += 2 * (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m])
    '''

    for m in prange(Nk):
        for a in range(2):
            for b in range(2):
                phi += (mat1[a,b,m] * A[:,b,m] * mat2[b,a,m] * A[:,a,m])

    return phi / Nk

''' same as phi_Kubo, but neglecting overlap of spectral function in different bands due to Gamma << gap
    in practice I just use phi_Kubo '''
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

''' operator in orbital basis obtained from operator in band basis '''
def operator_bare(op_tilde, vecs):
    op_bare = np.empty_like(op_tilde, dtype=np.complex128)
    if len(op_tilde.shape) > 3:
        n_ops = op_tilde.shape[0]
        for n in range(n_ops):
            op_bare[n] = np.einsum('jix,imx,lmx->jlx', vecs, op_tilde[n], vecs.conj())
    else:
        op_bare = np.einsum('jix,imx,lmx->jlx',vecs, op_tilde, vecs.conj())
    return op_bare

''' hermitian part (X + X^dag)/2 of an operator of shape (..., Norb, Norb, Nk), in band or orbital basis.
    operator_tilde returns hermitian operators only up to roundoff (~1e-16 relative; ~5e-14 for mat_tilde,
    where m4a and m4b cancel). In chi_UV the anti-hermitian roundoff (e.g. Im of diagonal elements)
    multiplies Re(pi), which stays finite as omega -> 0, so Im chi gets an omega-independent error and
    -Im chi/omega one that grows like 1/omega (it spoils e.g. chi_matj = 0 at the PH point).
    Use only for operators that are hermitian by construction (not for the separate pieces m4a, m4b, ...). '''
def hermitian_part(op):
    return 0.5 * (op + np.conj(np.swapaxes(op, -3, -2)))

''' df/domega, f is Fermi-Dirac distribution function '''
@njit(cache=True)
def fd_1(eps_or_omega, T, mu=0.0):
    eps = eps_or_omega - mu
    x = eps/T
    x_clipped = np.clip(x,-100,100)
    exp_x = np.exp(x_clipped)
    return - exp_x / (T * (1 + exp_x)**2)

''' Boltzmann transport function'''
def phi_Boltzmann(K, energije, mu, epsilons, faktor=0.2, shape='Gaussian'):
    Nk = len(K)
    dK = K[1] - K[0]
    depsilon = epsilons[1] - epsilons[0]
    phi = np.zeros(len(epsilons))
    vel1 = np.diff(energije[0]) / dK
    vel2 = np.diff(energije[1]) / dK

    vel = np.zeros((2, Nk))
    vel[0] = np.hstack([[0], vel1])
    vel[1] = np.hstack([[0], vel2])

    v_max = np.max(np.abs(vel))
    sigma = np.sqrt(v_max * depsilon * dK) * faktor

    for i, epsilon in enumerate(epsilons):
        for j in range(0,Nk//2+1):
            multiply = 1 if j in [0,Nk//2] else 2
            for alpha in [0,1]:
                phi[i] += multiply * delta_approximation(epsilon - energije[alpha,j] + mu, sigma, shape) * vel[alpha,j]**2
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

''' Boltzmann transport coefficients '''
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

def dephase_rho(rho, vecs, dt, Gamma, add_diagonals=None, fs=None):
    rho_band = operator_tilde(rho, vecs)
    decay = np.exp(-Gamma*dt)
    rho_band[0,1,:] *= decay
    rho_band[1,0,:] *= decay
    if add_diagonals!=None:
        rho_band[0,0,:] = fs[0,:] + decay*(rho_band[0,0,:] - fs[0,:])
        rho_band[1,1,:] = fs[1,:] + decay*(rho_band[1,1,:] - fs[1,:])
    return operator_bare(rho_band, vecs)

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
    return measurements#, measurements_k

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

''' Dynamic providers may return one operator or a stack of measurement operators. '''
def compile_measure_provider(measure_provider):
    """Group static operators first, preserving the supplied dynamic-provider order."""

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

def _measurement_operator_stack(ops, Nk):
    if ops.ndim == 3:
        ops = ops[np.newaxis, ...]
    if ops.ndim != 4 or ops.shape[1:] != (2, 2, Nk):
        raise ValueError(
            "A dynamic measurement provider must return an operator with "
            "shape (2, 2, Nk) or a stack with shape (Nop, 2, 2, Nk).")
    return ops

def simulate_pulz(K, hk0, rho, vecs, Vb, Vc, include_hartree,
                  perturbation_operator, measure_provider,
                  A0, t0, sigma, Omega, dt, t_max,
                  do_freeze, Ncorr, tol, geom, phases, g_ffts, add_diagonals=None, fs=None, Gamma=0.0, verbose=True, freq_verbose=50,
                  impose_deltas=False, impose_additional=True):
    N_points = int(t_max/dt)
    Nk = len(K)
    rho0 = np.copy(rho)

    static_ops, dynamic_providers = compile_measure_provider(measure_provider)

    ops_list = []

    if len(static_ops) > 0:
        ops_list.append(np.concatenate(static_ops, axis=0))

    for p in dynamic_providers:
        provider_rho = rho0 if do_freeze else rho
        ops = p(K, provider_rho, geom, phases, g_ffts,
                impose_deltas, impose_additional)
        ops_list.append(_measurement_operator_stack(ops, Nk))

    measure_operators = np.concatenate(ops_list, axis=0)
    Nop = measure_operators.shape[0]

    H0 = h_k(K, hk0, rho0, Vb, Vc, 0.0, include_hartree)

    rho_expvals = np.zeros((Nop, N_points), dtype=np.complex128)
    #rho_expvals_k = np.zeros((Nop, N_points, Nk), dtype=np.complex128)
    rho_norms = np.zeros(N_points)
    Delta_bs = np.zeros(N_points, dtype=np.complex128)
    Delta_cs = np.zeros(N_points, dtype=np.complex128)
    ns0 = np.zeros(N_points)
    ns1 = np.zeros(N_points)

    ts = dt * (np.arange(N_points) + 1)

    for i in range(N_points):

        ''' dynamics of rho:
            \partial_t \rho = -i[H[\rhorho] -  A(t)*P, \rho] + D[rho]
                               ---- Hamiltonian substep ---   ---- dissipator ---
        '''
        if verbose and i % freq_verbose == 0:
            msg = f'Progress: {i/N_points}'
            print('\r' + msg + ' ' * (80 - len(msg)), end='', flush=True)

        t = i*dt
        A_t = A_pulz(t, A0, t0, sigma, Omega)
        A_half = A_pulz(t + dt/2, A0, t0, sigma, Omega)

        ''' 1. first dissipative half-step '''
        if Gamma != 0.0:
            rho_a = dephase_rho(rho, vecs, dt/2, Gamma, add_diagonals, fs)
        else:
            rho_a = rho

        ''' 2. Hamiltonian at START of the Hamiltonian substep '''
        if do_freeze:
            H_k0 = H0
        else:
            H_k0 = h_k(K, hk0, rho_a, Vb, Vc, 0., include_hartree)

        ''' 3. predictor '''
        rho_pred = evolve_rho_kernel(H_k0 - A_t*perturbation_operator, rho_a, dt)

        if do_freeze:
            H_k1 = H0
        else:
            H_k1 = h_k(K, hk0, rho_pred, Vb, Vc, 0.0, include_hartree)

        rho_guess = rho_pred

        ''' 4. corrector iteration: Hamiltonian substep only '''

        for _ in range(Ncorr):

            H_mid = 0.5 * (H_k0 + H_k1) - A_half*perturbation_operator

            rho_new = evolve_rho_kernel(H_mid, rho_a, dt)

            err = np.max(np.abs(rho_new - rho_guess))

            rho_guess = rho_new

            if err < tol:
                break

            if not do_freeze:
                H_k1 = h_k(K, hk0, rho_guess, Vb, Vc, 0.0, include_hartree)

        # endpoint of the Hamiltonian tdhf substep
        rho_b = rho_guess

        ''' 5. final dissipative half-step'''
        if Gamma != 0.0:
            rho = dephase_rho(rho_b, vecs, dt/2, Gamma, add_diagonals, fs)
        else:
            rho = rho_b


        ''' 6. measurements '''
        ops_list = []

        if len(static_ops) > 0:
            ops_list.append(np.concatenate(static_ops, axis=0))

        for p in dynamic_providers:
            provider_rho = rho0 if do_freeze else rho
            ops = p(K, provider_rho, geom, phases, g_ffts,
                    impose_deltas, impose_additional)
            ops_list.append(_measurement_operator_stack(ops, Nk))

        measure_operators = np.concatenate(ops_list, axis=0)

        measurement_t = measure(Nk, Nop, measure_operators, rho)
        rho_expvals[:,i] = measurement_t
        #rho_expvals_k[:,i,:] = measurement_t_k
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

''' this is just a helper to get local maxima and local minima. I use this to get the envelope of the response '''
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

''' interaction kernel Theta for RPA equations '''
def rho_operators(K, Vb, Vc, include_hartree):
    if include_hartree:
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
    ''' matrix element of pseudo-density operators in orbital basis '''
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

''' Pi(omega) bubble integral. I rescale energies by Gamma and use Gauss-Legendre quadrature
    because I need to integrate from -infty to +infty, but integrand is big only at poles of Green's functions)'''
@njit(cache=True)
def Pi_bubble_tilde(omega, E_mk, E_nk, Gamma, mu_, invt, nodes, weights, eps=1e-5, n_eps=1.0, faktor=1.0):
    w    = omega / Gamma
    e_mk = E_mk  / Gamma
    e_nk = E_nk  / Gamma
    
    T = Gamma / invt
    
    invpi = 1.0 / np.pi

    # Single, T-independent cutoff based on Lorentzian tail
    # A(e) ~ 1/(pi * e^2) < eps  =>  e > 1/(pi*eps)
    epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T) / Gamma * n_eps

    # Cover spectral poles from both mn and nm channels, plus the Fermi region.
    #centers = np.array([e_mk, e_nk - w, e_nk, e_mk - w, mu_])
    centers = np.array([e_mk, e_nk - w, e_nk, e_mk - w, mu_, mu_ - w])
    n_centers = len(centers)

    # Build, sort, and merge intervals to avoid integrating overlaps twice.
    raw = np.empty((n_centers, 2), dtype=np.float64)
    for c in range(n_centers):
        raw[c, 0] = centers[c] - epsilon_max
        raw[c, 1] = centers[c] + epsilon_max

    # Sort by left endpoint
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            if raw[j, 0] < raw[i, 0]:
                raw[i, 0], raw[j, 0] = raw[j, 0], raw[i, 0]
                raw[i, 1], raw[j, 1] = raw[j, 1], raw[i, 1]

    # Merge overlapping intervals
    merged  = np.empty((n_centers, 2), dtype=np.float64)
    merged[0, 0] = raw[0, 0]
    merged[0, 1] = raw[0, 1]
    n_merged = 1
    for i in range(1, n_centers):
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
            pref = e - mu_ + 0.5 * w * faktor
            wi   = weights[i] * half

            # Evaluate f and its complement independently: avoid overflow
            # and loss of the small hole population deep below the Fermi level.
            x = (e - mu_) * invt
            xw = (ew - mu_) * invt
            ex = np.exp(-abs(x))
            exw = np.exp(-abs(xw))
            if x >= 0.0:
                f, fc = ex / (1.0 + ex), 1.0 / (1.0 + ex)
            else:
                f, fc = 1.0 / (1.0 + ex), ex / (1.0 + ex)
            if xw >= 0.0:
                fw, fwc = exw / (1.0 + exw), 1.0 / (1.0 + exw)
            else:
                fw, fwc = 1.0 / (1.0 + exw), exw / (1.0 + exw)
            # Exact Fermi difference, stable for either sign of omega.
            if w >= 0.0:
                df = f * fwc * (-np.expm1(-w * invt))
            else:
                df = -fw * fc * (-np.expm1(w * invt))

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
            mn_i = -np.pi * A_mk * A_nkw * df
            nm_r = A_nk * Grmw_r * f + A_mkw * Gan_r * fw
            nm_i = -np.pi * A_nk * A_mkw * df

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

''' I precompute Pi_mn for all m,n,k for a fixed omega. this object is then used for all bubbles with different vertices attached '''
@njit(parallel=True, cache=True)
def precompute_Pi_all(omega, energije, Gamma, mu_, invt, nodes, weights, eps=1e-5, faktor=1.0, n_eps=1.0):
    Norb, Nk = energije.shape

    pi_mn  = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    pi_nm  = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    piw_mn = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    piw_nm = np.zeros((Norb, Norb, Nk), dtype=np.complex128)

    for j in prange(Nk):
        for m in range(Norb):
            for n in range(m, Norb):

                pi_mnk, pi_nmk, pie_mnk, pie_nmk = Pi_bubble_tilde(omega, energije[m,j], energije[n,j], Gamma, mu_, invt, nodes, weights, eps, faktor=faktor, n_eps=n_eps)

                pi_mn [m, n, j] = pi_mnk
                pi_nm [m, n, j] = pi_nmk
                piw_mn[m, n, j] = pie_mnk
                piw_nm[m, n, j] = pie_nmk
    return pi_mn, pi_nm, piw_mn, piw_nm

''' accurate Lorentzian bubbles, same output as precompute_Pi_all (default in compute_chi: bubble="exact").
    Why: Pi_bubble_tilde gets the imaginary parts right, but its real (static) parts are off by
    0.1-10 % and much more in some cases, because the Gauss-Legendre nodes do not resolve the
    Lorentzians over the wide windows. The real parts enter the RPA denominator
    (e.g. the Goldstone zero mode at t12=0 is gapped by ~1e-3 instead of ~1e-10).
    See takarada/latex/tekst.tex, Sec. 7.

    Real parts: closed form. With the integrand written as sum_j alpha_j/(e - z_j), sum_j alpha_j = 0,
        int de f(e) sum_j alpha_j/(e - z_j) = sum_j alpha_j [psi(1/2 + s_j (z_j-mu)/(2 pi i T)) + i pi s_j/2],
    s_j = sign(Im z_j), psi = digamma.
    Imaginary parts: -pi int de A_m(e) A_n(e+w) [f(e) - f(e+w)] (x (e - mu + w/2) for the weighted bubble)
    on a uniform grid with spacing h_over * min(Gamma, T) over the window where f(e) - f(e+w) is nonzero.
    (The closed form loses precision for these tiny tail contributions in a gapped state.) '''
def _Psi_digamma(z, mu, T):
    s = np.sign(z.imag)
    return _psi(0.5 + s * (z - mu) / (2j * np.pi * T)) + 0.5j * np.pi * s

def _int_f_three_poles(poles, n0, n1, mu, T):
    ''' int de f(e) (n0 + n1 e) / prod_j (e - poles[j]) for three simple poles '''
    out = 0.0
    for j in range(3):
        den = 1.0
        for l in range(3):
            if l != j:
                den = den * (poles[j] - poles[l])
        out = out + (n0 + n1 * poles[j]) / den * _Psi_digamma(poles[j], mu, T)
    return out

def _bubble_half_closed_form(Em, En, w, G, mu, T, faktor=1.0):
    ''' int de [A_m(e) G^R_n(e+w) f(e) + A_n(e+w) G^A_m(e) f(e+w)], plain and weighted with
        (e - mu + 0.5*w*faktor); Em, En arrays over k. Only the real parts are used. '''
    c = G / np.pi
    # coinciding poles when Em - En + w = 0: shift by a negligible amount (the result is continuous)
    tiny = 1e-9 * G
    d = Em - En + w
    Em = np.where(np.abs(d) < tiny, En - w + tiny, Em)
    p1 = (Em + 1j * G, Em - 1j * G, En - w - 1j * G)      # A_m(e) G^R_n(e+w) f(e)
    p2 = (En + 1j * G, En - 1j * G, Em + w + 1j * G)      # A_n(e') G^A_m(e'-w) f(e'), e' = e+w
    t1 = _int_f_three_poles(p1, c, 0.0, mu, T)
    t2 = _int_f_three_poles(p2, c, 0.0, mu, T)
    t1w = _int_f_three_poles(p1, c * (-mu + 0.5 * w * faktor), c, mu, T)
    t2w = _int_f_three_poles(p2, c * (-mu - w + 0.5 * w * faktor), c, mu, T)
    return t1 + t2, t1w + t2w

@njit(parallel=True, cache=True)
def _bubble_imag_window(omega, energije, Gamma, mu, T, lo, h, npts, faktor=1.0):
    Norb, Nk = energije.shape
    im_mn = np.zeros((Norb, Norb, Nk)); im_nm = np.zeros((Norb, Norb, Nk))
    imw_mn = np.zeros((Norb, Norb, Nk)); imw_nm = np.zeros((Norb, Norb, Nk))
    es = np.empty(npts); df = np.empty(npts); pref = np.empty(npts)
    for i in range(npts):
        e = lo + i * h
        es[i] = e
        x = (e - mu) / T
        xw = (e + omega - mu) / T
        ex = np.exp(-abs(x)); exw = np.exp(-abs(xw))
        if x >= 0.0:
            f, fc = ex / (1.0 + ex), 1.0 / (1.0 + ex)
        else:
            f, fc = 1.0 / (1.0 + ex), ex / (1.0 + ex)
        if xw >= 0.0:
            fw, fwc = exw / (1.0 + exw), 1.0 / (1.0 + exw)
        else:
            fw, fwc = 1.0 / (1.0 + exw), exw / (1.0 + exw)
        if omega >= 0.0:
            df[i] = f * fwc * (-np.expm1(-omega / T))      # f(e) - f(e+w), stable
        else:
            df[i] = -fw * fc * (-np.expm1(omega / T))
        pref[i] = e - mu + 0.5 * omega * faktor
    c = Gamma / np.pi
    g2 = Gamma * Gamma
    for j in prange(Nk):
        for m in range(Norb):
            for n in range(m, Norb):
                Em = energije[m, j]; En = energije[n, j]
                s_mn = 0.0; s_nm = 0.0; sw_mn = 0.0; sw_nm = 0.0
                for i in range(npts):
                    e = es[i]
                    Am = c / ((e - Em) ** 2 + g2)
                    An = c / ((e - En) ** 2 + g2)
                    Amw = c / ((e + omega - Em) ** 2 + g2)
                    Anw = c / ((e + omega - En) ** 2 + g2)
                    v_mn = Am * Anw * df[i]
                    v_nm = An * Amw * df[i]
                    s_mn += v_mn; s_nm += v_nm
                    sw_mn += v_mn * pref[i]; sw_nm += v_nm * pref[i]
                im_mn[m, n, j] = -np.pi * h * s_mn
                im_nm[m, n, j] = -np.pi * h * s_nm
                imw_mn[m, n, j] = -np.pi * h * sw_mn
                imw_nm[m, n, j] = -np.pi * h * sw_nm
    return im_mn, im_nm, imw_mn, imw_nm

def precompute_Pi_all_exact(omega, energije, Gamma, mu, T, faktor=1.0, n_T=40.0, h_over=0.05):
    ''' same output as precompute_Pi_all, but mu and T in physical units (not scaled by Gamma) '''
    if T <= 0.0:
        raise ValueError("precompute_Pi_all_exact needs T > 0")
    Norb, Nk = energije.shape
    pi_mn = np.zeros((Norb, Norb, Nk), dtype=np.complex128)
    pi_nm = np.zeros_like(pi_mn); piw_mn = np.zeros_like(pi_mn); piw_nm = np.zeros_like(pi_mn)
    h = h_over * min(Gamma, T)
    lo = mu - max(omega, 0.0) - n_T * T
    hi = mu - min(omega, 0.0) + n_T * T
    npts = int(np.ceil((hi - lo) / h)) + 1
    im_mn, im_nm, imw_mn, imw_nm = _bubble_imag_window(omega, energije, Gamma, mu, T, lo, h, npts, faktor)
    for m in range(Norb):
        for n in range(m, Norb):
            re_mn, rew_mn = _bubble_half_closed_form(energije[m], energije[n], omega, Gamma, mu, T, faktor)
            re_nm, rew_nm = _bubble_half_closed_form(energije[n], energije[m], omega, Gamma, mu, T, faktor)
            pi_mn[m, n] = re_mn.real + 1j * im_mn[m, n]
            pi_nm[m, n] = re_nm.real + 1j * im_nm[m, n]
            piw_mn[m, n] = rew_mn.real + 1j * imw_mn[m, n]
            piw_nm[m, n] = rew_nm.real + 1j * imw_nm[m, n]
    return pi_mn, pi_nm, piw_mn, piw_nm

''' bubble susceptibility between U and V (need to be in band basis,i.e. tilde versions) '''
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

''' for fixed omega this calculates all susceptibilities '''
def compute_single_om_fused(
    om,
    Nk, Gamma, mu_, invt, nodes, weights,
    thetas, tok_tilde, tokK_tilde, mat_tilde, mat_tilde_,
    energije,
    rhos_tilde,
    eps=1e-5,
    include_hartree=True,
    include_phonon=False, n_eps=1.0,
    lam_b=None, om_b=None, lam_c=None, om_c=None, Vb=None, Vc=None, Gamma_ph=None, faktor=1.0,
    bubble="exact"):
    Nop = len(thetas)

    if not include_phonon:
        thetas_om = thetas
    elif include_phonon:
        thetas_om = thetas_ph(Vb, Vc, include_hartree, lam_b, om_b, lam_c, om_c, om, Gamma_ph)

    thetas_diag = np.diag(thetas_om)

    I = np.eye(Nop)

    # ── single precomputation of bubble for this omega ──────────────────────────
    if bubble == "exact":
        pi_mn, pi_nm, piw_mn, piw_nm = precompute_Pi_all_exact(
            om, energije, Gamma, mu_ * Gamma, Gamma / invt, faktor=faktor)
    elif bubble == "quadrature":
        pi_mn, pi_nm, piw_mn, piw_nm = precompute_Pi_all(
            om, energije, Gamma, mu_, invt, nodes, weights, eps, faktor=faktor, n_eps=n_eps
        )
    else:
        raise ValueError("bubble must be 'exact' or 'quadrature'")

    # ── chi0 matrix ──────────────────
    chi0 = np.zeros((Nop, Nop), dtype=np.complex128)
    for i in range(Nop):
        for j in range(Nop):
            chi0[i, j] = chi_UV(Nk, rhos_tilde[i], rhos_tilde[j], pi_mn, pi_nm)

    # ── chi_jj0 ────────────────────────────────────────────────────────
    chi_jj0 = chi_UV(Nk, tok_tilde, tok_tilde, pi_mn, pi_nm)
    chi_jjK0 = chi_UV(Nk, tok_tilde, tokK_tilde, pi_mn, pi_nm)
    chi_jjE0 = chi_UV(Nk, tok_tilde, tok_tilde, piw_mn, piw_nm)
    chi_jmat0 = chi_UV(Nk, tok_tilde, mat_tilde, pi_mn, pi_nm)
    chi_jmat0_ = chi_UV(Nk, tok_tilde, mat_tilde_, pi_mn, pi_nm)

    # ── chi_jrho0 / chi_rhoj0 ──────────────────────────────────────────
    chi_rhoj0 = np.zeros(Nop, dtype=np.complex128)
    chi_jrho0 = np.zeros(Nop, dtype=np.complex128)
    chi_rhojK0 = np.zeros(Nop, dtype=np.complex128)
    chi_rhojE0 = np.zeros(Nop, dtype=np.complex128)
    chi_rhomat0 = np.zeros(Nop, dtype=np.complex128)
    chi_rhomat0_ = np.zeros(Nop, dtype=np.complex128)
    for i in range(Nop):
        chi_rhoj0[i] = chi_UV(Nk, rhos_tilde[i], tok_tilde, pi_mn, pi_nm)
        chi_jrho0[i] = chi_UV(Nk, tok_tilde, rhos_tilde[i], pi_mn, pi_nm)
        chi_rhojK0[i] = chi_UV(Nk, rhos_tilde[i], tokK_tilde, pi_mn, pi_nm)
        chi_rhojE0[i] = chi_UV(Nk, rhos_tilde[i], tok_tilde, piw_mn, piw_nm)
        chi_rhomat0[i] = chi_UV(Nk, rhos_tilde[i], mat_tilde, pi_mn, pi_nm)
        chi_rhomat0_[i] = chi_UV(Nk, rhos_tilde[i], mat_tilde_, pi_mn, pi_nm)

 # ── RPA corrections ────────────────────────────────────────────────────────────
    mat     = I - chi0 @ thetas_diag
    inv     = LA.inv(mat)
    chi_rpa = inv @ chi0
    dchi_jj = chi_jrho0 @ thetas_diag @ inv @ chi_rhoj0
    dchi_jjK = chi_jrho0 @ thetas_diag @ inv @ chi_rhojK0
    dchi_jmat = chi_jrho0 @ thetas_diag @ inv @ chi_rhomat0
    dchi_jmat_ = chi_jrho0 @ thetas_diag @ inv @ chi_rhomat0_
    dchi_jjE = chi_jrho0 @ thetas_diag @ inv @ chi_rhojE0

    return om, chi0, chi_rpa, chi_jj0, dchi_jj, chi_jjK0, dchi_jjK, chi_jjE0, dchi_jjE, chi_jmat0, dchi_jmat, chi_jmat0_, dchi_jmat_, chi_rhoj0

''' susceptibilities for an array of omegas '''
def compute_chi(
    omegas,
    Nk, Gamma, mu_, invt, nodes, weights,
    thetas, tok_tilde, tokK_tilde, mat_tilde, mat_tilde_,
    energije,
    rhos_tilde,
    verbose=True,
    n_workers=None, #None: number of CPU cores, or specify an integer
    eps=1e-5, n_eps=1.0,
    include_hartree=True,
    include_phonon=False,
    lam_b=None, om_b=None, lam_c=None, om_c=None, Vb=None, Vc=None, Gamma_ph=None, faktor=1.0,
    bubble="exact"
):
    ''' bubble="exact": accurate bubbles (precompute_Pi_all_exact, needs T > 0); "quadrature": previous
        Gauss-Legendre bubbles (nodes, weights, eps, n_eps are only used then).
        'dchi_jEj' is the SB vertex correction in full RPA with the energy weight on the absorptive pair;
        'dchi_jEj_lastbubble' is the previous formula (weight on the last bubble), for comparison. '''
    if include_phonon:
        missing = []
        if lam_b is None:
            missing.append("lam_b")
        if om_b is None:
            missing.append("om_b")
        if Gamma_ph is None:
            missing.append("Gamma_ph")
        if Vc is None:
            missing.append("Vc")
        if include_hartree and Vb is None:
            missing.append("Vb")
        if Vc not in (None, 0):
            if lam_c is None:
                missing.append("lam_c")
            if om_c is None:
                missing.append("om_c")

        if missing:
            raise ValueError(
                "include_phonon=True requires the following parameters: "
                + ", ".join(missing)
            )
        if om_b == 0:
            raise ValueError("om_b must be nonzero when include_phonon=True")
        if Vc != 0 and om_c == 0:
            raise ValueError(
                "om_c must be nonzero when include_phonon=True and Vc != 0"
            )

    omegas = np.asarray(omegas)
    N_om   = len(omegas)
    Nop    = len(thetas)

    chi0_arr      = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_rpa_arr   = np.zeros((N_om, Nop, Nop), dtype=np.complex128)
    chi_jj0_arr   = np.zeros(N_om,             dtype=np.complex128)
    dchi_jj_arr   = np.zeros(N_om,             dtype=np.complex128)
    chi_jjK0_arr   = np.zeros(N_om,             dtype=np.complex128)
    dchi_jjK_arr   = np.zeros(N_om,             dtype=np.complex128)
    chi_jmat0_arr = np.zeros(N_om, dtype=np.complex128)
    chi_jmat0_arr_ = np.zeros(N_om, dtype=np.complex128)
    chi_jjE0_arr = np.zeros(N_om, dtype=np.complex128)
    dchi_jmat_arr = np.zeros(N_om, dtype=np.complex128)
    dchi_jmat_arr_ = np.zeros(N_om, dtype=np.complex128)
    dchi_jjE_arr = np.zeros(N_om, dtype=np.complex128)

    chi_rhoj0_arr = np.zeros((N_om, Nop), dtype=np.complex128)

    def _worker(om_idx, om):
        result = compute_single_om_fused(
            om,        # om = frequency value, omegas = full array
            Nk, Gamma, mu_, invt, nodes, weights,
            thetas, tok_tilde, tokK_tilde, mat_tilde, mat_tilde_,
            energije, rhos_tilde,
            eps=eps,
            include_hartree=include_hartree, include_phonon=include_phonon, n_eps=n_eps,
            lam_b=lam_b, om_b=om_b, lam_c=lam_c, om_c=om_c, Vb=Vb, Vc=Vc, Gamma_ph=Gamma_ph, faktor=faktor,
            bubble=bubble
        )
        return om_idx, result

    # Every omega runs parallel=True numba kernels (bubble, chi_UV). numba's 'workqueue' threading
    # layer (the only one without TBB/OpenMP installed) aborts when such kernels are launched from
    # several Python threads at once, so with it the omegas run one after another; the prange loops
    # over k inside already use all cores. Threads over omega are used only with 'tbb' or 'omp'.
    numba.get_num_threads()     # starts numba's thread pool, so that threading_layer() is known
    if numba.threading_layer() == 'workqueue':
        n_workers = 1

    def _results():
        if n_workers == 1:
            for om_idx, om in enumerate(omegas):
                yield _worker(om_idx, om)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_worker, om_idx, om) for om_idx, om in enumerate(omegas)]
                for future in as_completed(futures):
                    yield future.result()

    for om_idx, result in _results():
        om, chi0, chi_rpa, chi_jj0, dchi_jj, chi_jjK0, dchi_jjK, chi_jjE0, dchi_jjE, chi_jmat0, dchi_jmat, chi_jmat0_, dchi_jmat_, chi_rhoj0 = result

        chi0_arr[om_idx]      = chi0
        chi_rpa_arr[om_idx]   = chi_rpa
        chi_jj0_arr[om_idx]   = chi_jj0
        dchi_jj_arr[om_idx]   = dchi_jj
        chi_jjK0_arr[om_idx] = chi_jjK0
        dchi_jjK_arr[om_idx] = dchi_jjK
        chi_jjE0_arr[om_idx] = chi_jjE0
        dchi_jjE_arr[om_idx] = dchi_jjE
        chi_jmat0_arr[om_idx] = chi_jmat0
        dchi_jmat_arr[om_idx] = dchi_jmat
        chi_jmat0_arr_[om_idx] = chi_jmat0_
        dchi_jmat_arr_[om_idx] = dchi_jmat_
        chi_rhoj0_arr[om_idx] = chi_rhoj0

        if verbose:
            msg = f'Progress: {(om_idx+1)/N_om}'
            print('\r' + msg + ' ' * (80 - len(msg)), end='', flush=True)

    results = {'chi0' : chi0_arr,
               'chi' : chi_rpa_arr,

               'chi_jj0' : chi_jj0_arr,
               'dchi_jj' : dchi_jj_arr,

                'chi_jKj0' : chi_jjK0_arr,
                'dchi_jKj' : dchi_jjK_arr,

               'chi_jEj0' : chi_jjE0_arr,
               'chi_matj0' : chi_jmat0_arr,
               'chi_matj0_' : chi_jmat0_arr_,

               'dchi_matj' : dchi_jmat_arr,
               'dchi_jEj' : dchi_jjE_arr,
               'dchi_matj_' : dchi_jmat_arr_,

               'chi_rhoj0' : chi_rhoj0_arr,
               }
    
    return results

''' the following few functions are used to get DC limit of coefficients '''
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

''' phonon propagator '''
def D(lam, om, omega, Gamma_ph):
    return -lam / (1 - (omega/om + 1j*Gamma_ph)**2)

''' interaction kernel entering RPA equations in case of interaction + phonon '''
def thetas_ph(Vb, Vc, include_hartree, lam_b, om_b, lam_c, om_c, omega, Gamma_ph):
    Db = D(lam_b, om_b, omega, Gamma_ph)
    if include_hartree:
        if Vc == 0:
            theta = np.array([Vb/2, Db, Db, -Vb/2])
        else:
            Dc = D(lam_c, om_c, omega, Gamma_ph)
            theta = np.array([Vb/2, Db, Db, -Vb/2,
                              Vc/2, Dc, Dc, -Vc/2])
    else:
        if Vc == 0:
            theta = np.array([Db, Db])
        else:
            Dc = D(lam_c, om_c, omega, Gamma_ph)
            theta = np.array([Db, Db, Dc, Dc])
    return theta
