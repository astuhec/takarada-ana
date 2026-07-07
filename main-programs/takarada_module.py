import numpy as np
import json5
from scipy.special import roots_legendre

import takarada_helpers as helpers
import takarada_tokovi as tokovi

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json5.load(f)
            
''' Takarada model '''
class model:
    def __init__(self, input_file, compute_gap_infty=True, verbose=True,
                 b=None, t=None, t_=None, t12=None, epsilon=None, epsilon_=None, Vb=None, Vc=None, delta=None):
        
        ''' read input parameter and initialize the system '''
    
        config = load_config(input_file)
        self.config = config

        self.Nk = config.get("Nk")
        if verbose:
            print(f'=' * 80 + '\n' + 'Started 2-orbital calculation' + '\n' + f'=' * 80, flush=True)
            print(f'Initialized 1d lattice with Nk={self.Nk} unit cells.', flush=True)
     
        self.K = 2*np.pi * np.arange(-self.Nk//2, self.Nk//2) / self.Nk
     
        self.parameters = config.get("parameters")
        self.eps0 = self.parameters['eps0']
        self.dmu = self.parameters['dmu']
        self.epsilon_threshold = self.parameters['epsilon_threshold']
        self.N_epsilon = self.parameters['N_epsilon']
        self.maxiter = self.parameters['maxiter']
        self.n_pass = self.parameters['n_pass']
        self.eps_last = self.parameters['eps_last']
        self.mix2 = self.parameters['mix2']
        self.mix3 = self.parameters['mix3']
        self.max_trials = self.parameters['max_trials']

        self.mu = config.get("mu0")
        self.include_hartree = config.get("include_hartree")
        self.n_target = config.get("n_target")
        
        self.phys_parameters = config.get("phys_parameters")
        self.b = self.phys_parameters["b"] if b==None else b
        self.t = self.phys_parameters["t"] if t==None else t
        self.t_ = self.phys_parameters["t_"] if t_==None else t_
        self.t12 = self.phys_parameters["t12"] if t12==None else t12
        self.epsilon = self.phys_parameters["epsilon"] if epsilon==None else epsilon
        self.epsilon_ = self.phys_parameters["epsilon_"] if epsilon_==None else epsilon_
        self.Vb = self.phys_parameters["Vb"] if Vb==None else Vb
        self.Vc = self.phys_parameters["Vc"] if Vc==None else Vc
        self.delta = self.phys_parameters["delta"] if delta==None else delta
        #self.phys_parameters = list(self.phys_parameters.values())
        
        self.phys_parameters = [self.b, self.t, self.t_, self.t12, self.epsilon, self.epsilon_, self.Vb, self.Vc, self.delta]
        self.phys_parameters = [float(u) for u in self.phys_parameters]
        
        if verbose:
            print('Physical parameters are:' + '\n' + \
                f'b={self.b}' + '\n' + \
                f't0={self.t}' + '\n' + \
                f't1={self.t_}' + '\n' + \
                f'tperp={self.t12}' + '\n' + \
                f'epsilon={self.epsilon}' + '\n' + \
                f'V0={self.Vb}' + '\n' + \
                f'V1={self.Vc}' + '\n' + '=' * 80, flush=True)
                
        self.hk0 = helpers.h_k0(self.K, self.phys_parameters)

        if compute_gap_infty:
            self.energy_infty, self.mu_infty, self.gap_infty = Gap_infty(input_file, self.parameters, self.include_hartree)
        
        ''' Find ground state '''
        self.GS()
        if verbose:
            print(f'Found ground state.' + '\n' + 'Order parameter components are:' + '\n' + \
                f'delta_0 = {np.round(self.delta_b.real, 5)}' + '\n' + \
                f'delta_1 = {np.round(self.delta_c.real, 5)}' + '\n' + \
                f'Chemical potential is {np.round(self.mu, 5)} eV' + '\n' + \
                f'Occupation is {np.round(self.n, 5)}' + '\n' + '=' * 80, flush=True
                )
        self.current = tokovi.j_tok(self.K, self.phys_parameters)
        self.rhos, self.thetas = tokovi.rho_operators(self.K, self.phys_parameters, self.include_hartree)
        self.geom, self.phases = tokovi.input_data(self.K, self.phys_parameters)
        self.g_ffts = tokovi.G_ffts(self.phases, self.Nk)

        self.delta_bs = []
        self.delta_cs = []
        self.gaps = []
        self.mus = []
        self.errors = []
        self.occupations = []
        self.Ts = []
        self.mean_energies = []
        self.ns0 = []
        self.ns1 = []

        self.L11 = []
        self.L12 = []
        self.L22 = []
        self.L12q = []
        self.L22q = []

        self.L11_boltz = []
        self.L12_boltz  = []
        self.L22_boltz = []

        self.L11_0 = []
        self.L11_corr = []
        self.L12_0 = []
        self.L12_corr = []
        self.L12q_0 = []
        self.L12q_corr = []

    def GS(self):
        rho0 = helpers.rho0(self.Nk)
        rho, err, energije, vecs, fs, n = helpers.Rho_next(self.hk0, rho0, self.K, 0, self.mu, self.Vb, self.Vc, self.eps0,
                                                  self.epsilon_threshold, self.N_epsilon, self.maxiter, self.include_hartree, mix=0.5)
        self.rho = rho
        self.energije = energije
        self.vecs = vecs
        self.err = err
        self.n = n
        self.delta_b, self.delta_c = helpers.Delta(self.K, self.rho, self.Vb, self.Vc)
        self.gap = np.min(self.energije[1]) - np.max(self.energije[0])
        self.mu = 0.5 * (np.min(self.energije[1]) + np.max(self.energije[0]))
        self.mu_GS = self.mu

        self.rho_GS = self.rho
        self.vecs_GS = self.vecs
        self.energije_GS = self.energije
        self.delta_b_GS = self.delta_b
        self.delta_c_GS = self.delta_c

    def next_T(self, maxbrentq=50, mu_initial=None) -> None:
        if mu_initial==None:
            #rho, err, energije, vecs, _, n, mu = helpers.NewMu(self.rho, self.K, self.hk0, self.Vb, self.Vc, self.T, self.mu, self.dmu, self.maxiter, self.epsilon_threshold, self.eps_last, 0.5, self.mix2, self.mix3, self.n_pass, self.max_trials, faktor1=0.001, include_hartree=self.include_hartree)

            mu, rho, err, energije, vecs, _, n = helpers.NewMu2(self.mu - self.dmu, self.mu + self.dmu, self.hk0, self.rho, self.K, self.T, self.Vb, self.Vc, self.eps0, self.epsilon_threshold, self.N_epsilon, self.maxiter, self.include_hartree, mix=0.5, xtol=self.n_pass, rtol=self.n_pass, maxiterbrentq=maxbrentq, n_target=self.n_target)
        else:
            rho, err, energije, vecs, _, n = helpers.Rho_next(self.hk0, self.rho, self.K, self.T, mu_initial, self.Vb, self.Vc, self.eps0, self.epsilon_threshold, self.N_epsilon, self.maxiter, self.include_hartree)
            mu = mu_initial

        self.rho = rho
        self.energije = energije
        self.vecs = vecs
        self.err = err
        self.n = n
        self.mu = mu
        #print(self.mu)
        self.delta_b, self.delta_c = helpers.Delta(self.K, self.rho, self.Vb, self.Vc)

    def run_Tdependence(self, input_phonon=None, k=None, n=None,
                        beta=None, scale=None, Nbeta=None):
                        
        ''' read input parameter and initialize the system '''
        config = self.config

        evaluate_transport_DC = config.get("evaluate_transport_DC")
        evaluate_vertex_DC = config.get("evaluate_vertex_DC")

        print('Started to find temperature dependence of transport coefficients.', flush=True)
        if evaluate_transport_DC == True:
            print('Will calculate Boltzmann and Kubo bubble DC coefficients.', flush=True)
        if evaluate_vertex_DC == True:
            print('Will calculate Kubo bubble DC coefficients and vertex corrections.', flush=True)
        if evaluate_transport_DC == False and evaluate_vertex_DC == False:
            print('Will not calculate transport coefficients, but will find self-consistent rho(T) and mu(T).', flush=True)

        params = config.get("params")
        Nomega = params['Nomega']
        eps = params['eps']
        omega0_low = params['omega0_low']
        omega0_high = params['omega0_high']
        omega0_len = params['omega0_len']
        omega0 = np.logspace(omega0_low, omega0_high, omega0_len)
        eps2 = params['eps2']
        deg = params['deg']
        n_workers = params['n_workers']

        maxbrentq = config.get("maxbrentq")
        Gammas = config.get("Gammas")
        eps_ns0 = config.get("eps_ns0")
        betas0 = config.get("beta0") if beta==None else beta
        scale = config.get("scale") if scale==None else scale
        Nbetas = config.get("Nbetas") if Nbeta==None else Nbeta
        freq_betas = config.get("freq_betas")
        betas = betas0 / scale**np.arange(1, Nbetas+1)
        max_stop = int(Nbetas // freq_betas)
        stops = [freq_betas*i for i in range(1, max_stop+1)]
        phonon = config.get("phonon")

        if phonon:

            config_phonon = load_config(input_phonon)

            gbx = config_phonon.get("gbx")
            gby = config_phonon.get("gby")
            gcx = config_phonon.get("gcx")
            gcy = config_phonon.get("gcy")
            omega_bx = config_phonon.get("omega_bx")
            omega_by = config_phonon.get("omega_by")
            omega_cx = config_phonon.get("omega_cx")
            omega_cy = config_phonon.get("omega_cy")
            Gamma_ph = config_phonon.get("Gamma_ph")

        if evaluate_vertex_DC:
            nodes, weights = roots_legendre(deg)
            
        self.Gammas = Gammas
        print('started', flush=True)
        for i, beta in enumerate(betas):
            eps_ns = max(self.errors) if len(self.errors) > 0 else eps_ns0
            self.T = 1/beta
            if i not in stops:
                if (i+1) in stops:
                    rho_save = self.rho
                    energije_save = self.energije
                    vecs_save = self.vecs
                    mu_save = self.mu
                    err_save = self.err
                    n_save = self.n
                if (k,n) == (None,None):
                    self.next_T(maxbrentq=maxbrentq)
                else:
                    self.next_T(maxbrentq=maxbrentq, mu_initial=k*self.T+n)
            else:
                if (k,n) == (None,None):
                    self.next_T(maxbrentq=maxbrentq)
                else:
                    self.next_T(maxbrentq=maxbrentq, mu_initial=k*self.T+n)
                self.Ts.append(self.T)
                self.delta_bs.append(self.delta_b)
                self.delta_cs.append(self.delta_c)
                self.mus.append(self.mu)
                self.errors.append(self.err)
                self.occupations.append(self.n)

                n0, n1 = helpers.Ns(self.rho, self.energy_infty, self.delta_b, self.delta_c, self.Vb, self.Vc, eps_ns, self.T, self.mu)
                self.ns0.append(n0)
                self.ns1.append(n1)

                self.gap = helpers.Gap(self.energije, self.delta_bs[-1], self.delta_cs[-1], self.Vb, self.Vc, eps_ns, self.gap_infty)
                self.gaps.append(self.gap)

                energy = helpers.energy_average(self.K, self.rho, self.phys_parameters, self.energije, self.mu, self.T)
                self.mean_energies.append(energy)

                self.velocities()

                if evaluate_transport_DC:
                    ''' DC coefficients: Boltzmann's and Kubo's, evaluating the bubble diagram (coefficients as integrals of transport functions) '''
                    self.DC_coefficients(eps, Nomega, Gammas)
                
                if evaluate_vertex_DC:
                    ''' Kubo's DC coefficients, bubble and corrections '''
                    if not phonon:
                        self.DC_bubble_corr(nodes, weights, Gammas, omega0, eps2, n_workers=n_workers)
                    else:
                        self.DC_bubble_corr(nodes, weights, Gammas, omega0, eps2,
                                            gbx=gbx, omega_bx=omega_bx,
                                            gby=gby, omega_by=omega_by,
                                            gcx=gcx, omega_cx=omega_cx,
                                            gcy=gcy, omega_cy=omega_cy,
                                            Gamma_ph=Gamma_ph, phonon=phonon, n_workers=n_workers)
                        
                if i > 0:
                    self.rho = rho_save
                    self.energije = energije_save
                    self.vecs = vecs_save
                    self.mu = mu_save
                    self.err = err_save
                    self.n = n_save

            msg = f'Progress {np.round(i/len(betas), 3)}. beta={np.round(1/self.T, 1)}, n={np.round(self.n)}, delta_b={np.round(self.delta_b.real, 5)}, delta_c={np.round(self.delta_c.real, 5)}'
            print(msg, flush=True)
    
    def run_lowT_dependence(self, input_phonon=None, T_start=None, T_end=None, T_stable=None,
                        threshold=0.02, window=5, safety=10, window0=20, r2_threshold=0.99):
        # find interval where mu(T) makes sense (linear) or provide region by putting T_start, T_end, T_stable

        Nbeta_correction = self.config.get("Nbeta_correction")
        scale_correction = 1/self.config.get("scale")
        stable_index = np.argmin(np.abs(np.array(self.Ts) - T_stable))  if T_stable is not None else helpers.is_stable(self.Ts, self.mus, threshold, window) + safety

        # find interval above T_stable where mu(T) is linear. this will serve as approximation for mu(T) at T < T_stable

        while T_start==None or T_end==None:
            T_start, T_end = helpers.find_linear_region(self.Ts, self.mus, stable_index, window0, r2_threshold=r2_threshold)
            window0 -= 5
        ind_start = np.argmin(np.abs(np.array(self.Ts) - T_start))
        ind_end = np.argmin(np.abs(np.array(self.Ts) - T_end))
        k, n = np.polyfit(np.array(self.Ts)[ind_start:ind_end], np.array(self.mus)[ind_start:ind_end], 1)

        self.coefficients = (k,n)
        n = self.mu_GS
        k = (self.mus[stable_index] - self.mu_GS) / self.Ts[stable_index]
        
        beta_initial = 1/self.Ts[stable_index]
        mu_initial = self.mus[stable_index]

        self.mu = self.mu_GS
        self.rho = self.rho_GS
        self.vecs = self.vecs_GS
        self.energije = self.energije_GS
        self.delta_b = self.delta_b_GS
        self.delta_c = self.delta_c_GS

        count = len(self.Ts)
        self.run_Tdependence(input_phonon=input_phonon, k=k, n=n,
                             beta=beta_initial, Nbeta=Nbeta_correction, scale=scale_correction)
        self.stable_index = stable_index
        self.Ncorrection = len(self.Ts) - count

    def reset(self, mu0) -> None:
        self.GS()
        self.mu = mu0
    
    def velocities(self) -> None:
        self.current_tilde = tokovi.operator_tilde(self.current, self.vecs)
        m3, m6, m4a, m4b = tokovi.compute_all_mf_matrices(self.K, self.rho, self.geom, self.phases, self.g_ffts)
        mat = m3 + m6 + m4a + m4b
        self.mat_tilde = tokovi.operator_tilde(mat, self.vecs)
        self.rhos_tilde = tokovi.operator_tilde(self.rhos, self.vecs)

    def transport_functions(self, epsilons, Gamma, dict_form=None):
        spektralka = tokovi.Spektralka(epsilons, self.mu, self.energije, Gamma)
        phi = tokovi.phi_Kubo(self.K, self.current_tilde, self.current_tilde, spektralka, epsilons)
        phiQ = tokovi.phi_Kubo(self.K, self.mat_tilde, self.current_tilde, spektralka, epsilons)
        phiQ2 = tokovi.phi_Kubo(self.K, self.mat_tilde, self.mat_tilde, spektralka, epsilons)
        if dict_form == None:
            return phi, phiQ, phiQ2
        elif dict_form == True:
            phi_boltz = tokovi.phi_Boltzmann(self.K, self.energije, self.mu, epsilons)
            results = {'phi' : phi,
                       'phiQ' : phiQ,
                       'epsilons' : epsilons,
                       'phi_Boltz' : phi_boltz}
            return results

    def ls_Kubo(self, epsilons, Gamma, mfd1):
        phi, phiQ, phiQ2 = self.transport_functions(epsilons, Gamma)

        l11 = np.pi * tokovi.integral_omega(phi * mfd1, epsilons)
        l12 = np.pi * tokovi.integral_omega(epsilons * phi * mfd1, epsilons)
        l22 = np.pi * tokovi.integral_omega(epsilons**2 * phi * mfd1, epsilons)
        l12q = np.pi * tokovi.integral_omega(phiQ * mfd1, epsilons)
        l22q = np.pi * tokovi.integral_omega(phiQ2 * mfd1, epsilons) + 2 * np.pi * tokovi.integral_omega(phiQ * epsilons * mfd1, epsilons)        

        return l11, l12, l22, l12q, l22q

    def DC_coefficients(self, eps, Nomega, Gammas):
        epsilon_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*self.T))) * 2 * self.T)
        epsilons = np.linspace(-epsilon_max, epsilon_max, Nomega, dtype=np.float64)

        K0b, K1b = tokovi.Kn_boltz(self.K, self.energije, self.mu, self.T)
        mfd1 = -tokovi.fd_1(epsilons, self.T)

        Ngamma = len(Gammas)

        l11 = np.zeros(Ngamma)
        l12 = np.zeros(Ngamma)
        l22 = np.zeros(Ngamma)
        l12q = np.zeros(Ngamma)
        l22q = np.zeros(Ngamma)

        l11_boltz = np.zeros(Ngamma)
        l22_boltz = np.zeros(Ngamma)
        l12_boltz = np.zeros(Ngamma)

        for g, Gamma in enumerate(Gammas):
            l11_, l12_, l22_, l12q_, l22q_ = self.ls_Kubo(epsilons, Gamma, mfd1)
            l11[g] = l11_.real
            l22[g] = l22_.real
            l12[g] = l12_.real
            l12q[g] = l12q_.real
            l22q[g] = l22q_.real

            l11_boltz[g] = K0b / (2 * Gamma)
            l22_boltz[g] = K0b / (2 * Gamma)
            l12_boltz[g] = K1b / (2 * Gamma)

        self.L11.append(helpers.to_scalar_if_single(l11))
        self.L12.append(helpers.to_scalar_if_single(l12))
        self.L22.append(helpers.to_scalar_if_single(l22))
        self.L12q.append(helpers.to_scalar_if_single(l12q))
        self.L22q.append(helpers.to_scalar_if_single(l22q))

        self.L11_boltz.append(helpers.to_scalar_if_single(l11_boltz))
        self.L22_boltz.append(helpers.to_scalar_if_single(l22_boltz))
        self.L12_boltz.append(helpers.to_scalar_if_single(l12_boltz))

    def DC_bubble_corr(self, nodes, weights, Gammas, omega0, eps,
                       gbx=None, omega_bx=None,
                       gby=None, omega_by=None,
                       gcx=None, omega_cx=None,
                       gcy=None, omega_cy=None,
                       Gamma_ph=None, phonon=False, n_workers=None):
        Ngamma = len(Gammas)

        l11_0 = np.zeros(Ngamma)
        l12_0 = np.zeros_like(l11_0)
        l12q_0 = np.zeros_like(l11_0)

        l11 = np.zeros_like(l11_0)
        l12 = np.zeros_like(l11_0)
        l12q = np.zeros_like(l11_0)

        for g, Gamma in enumerate(Gammas):
            mu_ = self.mu / Gamma
            invt = Gamma / self.T
            
            if not phonon:
                results = tokovi.compute_chi(omega0, self.Nk, Gamma, mu_, invt, nodes, weights, self.thetas, self.current_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=True, eps=eps, n_workers=n_workers)
            else:
                results = tokovi.compute_chi(omega0, self.Nk, Gamma, mu_, invt, nodes, weights, self.thetas, self.current_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=True, eps=eps, n_workers=n_workers,
                                         gbx=gbx, omega_bx=omega_bx,
                                         gby=gby, omega_by=omega_by,
                                         gcx=gcx, omega_cx=omega_cx,
                                         gcy=gcy, omega_cy=omega_cy,
                                         Gamma_ph=Gamma_ph, phonon=True, include_hartree=self.include_hartree, Vb=self.Vb, Vc=self.Vc)

            ##np.savez(f'results{len(self.Ts)}.npz', **results)
            
            Chi_jj0 = - results['chi_jj0'].imag
            dChi_jj  = - results['dchi_jj'].imag
            Chi_jj = Chi_jj0 + dChi_jj
            
            l11_0[g] = tokovi.find_DC_limit(omega0, Chi_jj0)
            l11[g] = tokovi.find_DC_limit(omega0, Chi_jj)

            Chi_jEj0 = - results['chi_jEj0'].imag
            dChi_jEj = - results['dchi_jEj'].imag
            Chi_jEj = Chi_jEj0 + dChi_jEj
            l12_0[g] = tokovi.find_DC_limit(omega0, Chi_jEj0)
            l12[g] = tokovi.find_DC_limit(omega0, Chi_jEj)

            Chi_matj0 = - results['chi_matj0'].imag
            dChi_matj = - results['dchi_matj'].imag
            Chi_matj = Chi_matj0 + dChi_matj
            
            if np.max(np.abs(Chi_matj0)) < 1e-14:
                l12q_0 = 0.0
            else:
                l12q_0 = tokovi.find_DC_limit(omega0, Chi_matj0)

            if np.max(np.abs(Chi_matj)) < 1e-14:
                l12q = 0.0
            else:
                l12q = tokovi.find_DC_limit(omega0, Chi_matj)

        self.L11_0.append(helpers.to_scalar_if_single(l11_0))
        self.L11_corr.append(helpers.to_scalar_if_single(l11))
        self.L12_0.append(helpers.to_scalar_if_single(l12_0))
        self.L12_corr.append(helpers.to_scalar_if_single(l12))
        self.L12q_0.append(helpers.to_scalar_if_single(l12q_0))
        self.L12q_corr.append(helpers.to_scalar_if_single(l12q))

    def optical_response(self):
        print('\n' + '-' * 80 + '\n' + \
              'Started calculation of RPA responses.', flush=True)
        params = self.config.get("params_RPA")
        deg = params["deg"]
        Gamma = params["Gamma"]
        eps = params["eps"]
        n_workers = params["n_workers"]

        omega_low = params["omega_low"]
        omega_high = params["omega_high"]
        omega_len = params["omega_len"]
        space = params["space"]
        if space == "lin":
            omegas = np.linspace(omega_low, omega_high, omega_len)
        elif space == "log":
            omegas = np.logspace(omega_low, omega_high, omega_len)

        nodes, weights = roots_legendre(deg)
        mu_ = self.mu / Gamma
        invt = Gamma / self.T
        results = tokovi.compute_chi(omegas, self.Nk, Gamma, mu_, invt, nodes, weights, self.thetas, self.current_tilde, self.mat_tilde, self.energije, self.rhos_tilde, verbose=True, n_workers=n_workers, eps=eps)
        results["Gamma"] = Gamma
        results["T"] = self.T
        results["mu"] = self.mu
        results["omegas"] = omegas
        print('\n' + 'Finished calculation of RPA responses.', flush=True)
        return results

    def simulate_perturbation(self, do_freeze=None):
        print('\n' + '-' * 80 + '\n' + \
              'Started simulation of perturbation.', flush=True)
        params = self.config.get("params_perturbation")
        A0 = params['A0']
        t0 = params['t0']
        sigma = params['sigma']
        Omega0 = params['Omega0']
        dt = params['dt']
        t_max = params['t_max']
        Ncorr = params['Ncorr']
        tol = params['tol']
        Gamma_ = params['Gamma_']
        verbose = params['verbose']
        freq_verbose = params['freq_verbose']
        eta = params['eta']
        omega_cut = params['omega_cut']

        if params["perturbation_operator"] == 'current':
            perturbation_operator = self.current
        else:
            print("Choose a valid perturbation operator. Currently available : current", flush=True)
        
        if params["measure_provider"] == 'current':
            measure_provider = self.current
        else:
            print("Choose a valid measure provider. Currently available : current", flush=True)

        times, measurement, norma, delta_bs, delta_cs, ns0, ns1 = tokovi.simulate_pulz(self.K, self.hk0, self.rho, self.Vb, self.Vc, self.include_hartree,
                                                                                       perturbation_operator, measure_provider,
                                                                                       A0, t0, sigma, Omega0, dt, t_max,
                                                                                       do_freeze, Ncorr, tol, self.geom, self.phases, self.g_ffts, Gamma=Gamma_, verbose=verbose, freq_verbose=freq_verbose)
        
        pulz = tokovi.A_pulz(times, A0, t0, sigma, Omega0)
        results = {"time" : times,
                   "measurement" : measurement,
                   "pulz" : pulz,
                   "norma" : norma,
                   "delta_bs" : delta_bs,
                   "delta_cs" : delta_cs,
                   "ns0" : ns0,
                   "ns1" : ns1,
                   "T" : self.T,
                   "mu" : self.mu}
        
        omegas, Re_sigma = tokovi.optical_conductivity(times, measurement[0], pulz, eta, omega_cut, self.Nk)
        results["omegas"] = omegas
        results["Re_sigma"] = Re_sigma
        print('\n' + 'Finished simulation of perturbation.', flush=True)
        return results        

    def merge(self, arr):
        arr1 = arr[self.stable_index:-self.Ncorrection]
        arr2 = arr[-self.Ncorrection+1:][::-1]
        arr = np.concatenate([arr2, arr1], axis=0)
        return arr

    def collect_data(self, evaluate_transport_DC, evaluate_vertex_DC, merge=True):
        if merge:
            data = {"delta_bs": self.merge(self.delta_bs),
                    "delta_cs": self.merge(self.delta_cs),
                    "ns0": self.merge(self.ns0),
                    "ns1": self.merge(self.ns1),
                    "gaps": self.merge(self.gaps),
                    "mus": self.merge(self.mus),
                    "errors": self.merge(self.errors),
                    "occupations": self.merge(self.occupations),
                    "Ts": self.merge(self.Ts),
                    "mean_energies": self.merge(self.mean_energies),
                    "phys_parameters" : np.array(self.phys_parameters),
                    "Gammas": self.Gammas,
                    "include_hartree" : self.include_hartree}
            
            if evaluate_transport_DC:
                data["L11"] = self.merge(self.L11)
                data["L12"] = self.merge(self.L12)
                data["L22"] = self.merge(self.L22)
                data["L12q"] = self.merge(self.L12q)
                data["L22q"] = self.merge(self.L22q)
                data["L11_boltz"] = self.merge(self.L11_boltz)
                data["L12_boltz"] = self.merge(self.L12_boltz)
                data["L22_boltz"] = self.merge(self.L22_boltz)

            if evaluate_vertex_DC:
                data["L11_0"] = self.merge(self.L11_0)
                data["L11_corr"] = self.merge(self.L11_corr)
                data["L12_0"] = self.merge(self.L12_0)
                data["L12_corr"] = self.merge(self.L12_corr)
                data["L12q_0"] = self.merge(self.L12q_0)
                data["L12q_corr"] = self.merge(self.L12q_corr)
        else:
            data = {"delta_bs": self.delta_bs,
                    "delta_cs": self.delta_cs,
                    "ns0": self.ns0,
                    "ns1": self.ns1,
                    "gaps": self.gaps,
                    "mus": self.mus,
                    "errors": self.errors,
                    "occupations": self.occupations,
                    "Ts": self.Ts,
                    "mean_energies": self.mean_energies,
                    "phys_parameters" : np.array(self.phys_parameters),
                    "Gammas": self.Gammas,
                    "include_hartree" : self.include_hartree}
            
            if evaluate_transport_DC:
                data["L11"] = self.L11
                data["L12"] = self.L12
                data["L22"] = self.L22
                data["L12q"] = self.L12q
                data["L22q"] = self.L22q
                data["L11_boltz"] = self.L11_boltz
                data["L12_boltz"] = self.L12_boltz
                data["L22_boltz"] = self.L22_boltz

            if evaluate_vertex_DC:
                data["L11_0"] =self.L11_0
                data["L11_corr"] = self.L11_corr
                data["L12_0"] = self.L12_0
                data["L12_corr"] = self.L12_corr
                data["L12q_0"] = self.L12q_0
                data["L12q_corr"] = self.L12q_corr
            
        return data
    
def Gap_infty(input_file, parameters, include_hartree):
    new_parameters1 = parameters.copy()
    new_parameters1["eps0"] = 0.
    m = model(input_file, compute_gap_infty=False, verbose=False)
    m.GS()  # Compute ground state to populate m.rho
    hk = m.hk0.copy()
    if include_hartree == True:
        hk[0,0,:] += (m.Vb + m.Vc) * np.sum(m.rho[1,1,:]) / m.Nk
        hk[1,1,:] += (m.Vb + m.Vc) * np.sum(m.rho[0,0,:]) / m.Nk
    energy_infty = np.zeros((2, m.Nk))
    energy_infty[0] = hk[0,0].real
    energy_infty[1] = hk[1,1].real
    mu_infty = m.mu
    gap_infty = np.min(hk[1,1]) - np.max(hk[0,0])
    return energy_infty, mu_infty, gap_infty