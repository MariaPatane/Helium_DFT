import numpy as np
import logging

logger = logging.getLogger("DFT_solver")  

class DFTHelium:
    """
    Initialization solver DFT for an atom
    Args:
        Z (int): Atomic number (default=2 for Helium).
        r_max (float): maximum radius  (a.u.).
        h (float): Step of  the radial grid (a.u.).
    """

    def __init__(self, Z=2, r_max=30.0, h=0.001):
        if Z <= 0:
           raise ValueError("Atomic number (Z) must be > 0.")
        self.Z = Z
        if r_max <= 0:
           raise ValueError(" Grid endpoint (r_max) must be > 0.")
        self.r_max = r_max
        #h has a small value to avoid division for zero 
        if h <= 0:
           raise ValueError("Step of the grid (h) must be > 0 .")
        self.h = h   
        self.N = int(r_max / h)
        self.x = np.linspace(h, r_max, self.N)

        # Array potential and density
        self.Vn = np.zeros(self.N)
        self.Vh = np.zeros(self.N)
        self.Vx = np.zeros(self.N)
        self.Vc = np.zeros(self.N)
        self.den = np.zeros(self.N)
        self.u_eff = np.zeros(self.N)
        self.V_eff = np.zeros(self.N)

        # Parameter LDA correlation
        self.lda_params = {
        "A": 0.0311, "B": -0.048, "C": 0.0020, "D": -0.0116, "gamma": -0.1423,
         "beta1": 1.0529, "beta2": 0.3334
        }
    
    # ------------------------
    # Potentials 
    # ------------------------
    def nucl_potential(self):
        """
            Nuclear potential -Z/r
            Parameters
            ---------
            Z (int) : Atomic number
                    
            Output
            -------
            Vn (np.ndarray): nuclear potential 
        """
        self.Vn = -self.Z / self.x
        return self.Vn

    def hartree(self, nn: int, l: int) -> np.ndarray:
        """
        Semplified Hartree potential (using Verlet Algorithm)
        Parameters
        ---------
        nn : int
                Principal quantum number 
        l : int
                Angular quantum number
        Output
        -------
        Vh : np.ndarray
                Hartree potential on the radial grid
        """
        N, h, den, x, Z, r_max = self.N, self.h, self.den, self.x, self.Z, self.r_max
        U = np.zeros(N, dtype=float)

        # Initial condition (U(0)=0, U(h)=2h)
        U[1] = 2*h
        coeff = h**2
        for i in range(1, N -1):
            U[i + 1] = 2.0 * U[i] - U[i - 1] - coeff * den[i] / x[i]
        alpha = (Z - U[N-1]) / r_max
        #Hartree potential 
        Vh = U / x + alpha
        #hartree energy (needed to perform a test)
        U = U + alpha*x
        self.U = U
        self.Vh = Vh
        return Vh
    
    def exchange(self, nn, l):
        """
            Exchange potential computed using LDA scheme.

            Parameters
            ----------
            nn : int
                Principal quantum number
            l : int
                Angular momentum quantum number

            Returns
            -------
            Vc : np.ndarray
                Exchange potential on the radial grid
        """
        #to avoid division for too small value of the grid
        mask = self.x > 1e-12  
        Vx = np.zeros_like(self.x)
        P = self.den
        # costant
        c = (3.0 / (4.0 * np.pi**2))**(1.0/3.0)  

        Vx[mask] = -c * (P[mask] / (self.x[mask]**2))**(1.0/3.0)
        self.Vx = Vx
        return self.Vx

    def correlation(self, nn, l):
        """
        Compute the correlation potential using the LDA scheme.

        Parameters
        ----------
        nn : int
            Principal quantum number
        l : int
            Angular momentum quantum number

        Returns
        -------
        Vc : np.ndarray
            Correlation potential on the radial grid
        """
        p = self.lda_params
        den = self.den

        #to avoid calculation for meaningless value of density
        mask = den >= 1e-10

        rs = np.zeros_like(den)
        rs[mask] = (3 / (4 * np.pi * den[mask]))**(1/3)

        # to avoid calculation for invalid value of density
        mask_low = mask & (rs > 1)
        mask_high = mask & (rs <= 1)

        Vc = np.zeros_like(den)

        # Low-density case
        if np.any(mask_low):
            rs_h = rs[mask_low]
            eps_h = p["gamma"] / (1 + p["beta1"]*np.sqrt(rs_h) + p["beta2"]*rs_h)
            Vc[mask_low] = eps_h * (1 + (7/6)*p["beta1"]*np.sqrt(rs_h) + (4/3)*p["beta2"]*rs_h) \
                                / (1 + p["beta1"]*np.sqrt(rs_h) + p["beta2"]*rs_h)

        # High-density case
        if np.any(mask_high):
            rs_l = rs[mask_high]
            Vc[mask_high] = p["A"] * np.log(rs_l) + p["B"] - p["A"]/3 \
                        + (2/3)*p["C"]*rs_l*np.log(rs_l) \
                        + (2*p["D"] - p["C"])*rs_l/3

        self.Vc = Vc
        return Vc
    
    # ------------------------
    # Radial solver
    # ------------------------
    def eigenvalue_Eff_iterative(self, l, E_eff, eps=1e-10):
        """
        Solver of radial SE using backward iteration.

        Parameters
        ----------
        l : int
            Angular momentum quantum number
        E_eff : float 
            Effective energy in radial SE
        Returns
        -------
        u_eff : np.ndarray
            radial wavefunction u_eff
        nodes : int
            number of nodes of u_eff   
        """
        N = self.N
        h = self.h
        x = self.x
        V = self.V_eff

        u_eff = np.zeros(N)
        u_eff[-2] = eps
        nodes = 0

        for i in range(N-2, 0, -1):
            k2 = -2*E_eff + 2*V[i] + l*(l+1)/x[i]**2
            u_eff[i-1] = 2*u_eff[i] - u_eff[i+1] + h**2 * k2 * u_eff[i]
            
            if u_eff[i-1]*u_eff[i] < 0:
                nodes += 1
            
        u_eff[0] = 0.0
        
        return u_eff, nodes
        
           
    # ------------------------
    # Bisection eigenvalue solver
    # ------------------------
    def bisection_Eff(self, nn, l, tol=1e-11, Emin=-10.0, Emax=0.0):
        """
        Compute the effective energy eigenvalue using the bisection method.
        `E_eff` for a given set of quantum numbers, 
        by counting the number of nodes in the radial wavefunction until 
        the expected number of nodes is matched.

        Parameters
        ----------
        nn : int
            Principal quantum number 
        l : int
            Angular momentum quantum number 
        tol : float
            Convergence tolerance for the energy interval (default: 1e-11).
        Emin : float
            Lower bound for the energy search window (default: -10.0).
        Emax : float
            Upper bound for the energy search window (default: 0.0).

        Returns
        -------
        E_eff : float
            Effective energy eigenvalue that satisfies the node condition.
        u_eff : np.ndarray
            Corresponding radial wavefunction computed at `E_eff`.
        """
        nodes_expected = nn - l - 1

        while (Emax - Emin) > tol:
            E_eff = 0.5 * (Emin + Emax)
            u_eff, nodes = self.eigenvalue_Eff_iterative(l, E_eff)

            #update of energy range considering the number of expected nodes
            if nodes > nodes_expected:
                Emax = E_eff
            else:
                Emin = E_eff

        # update of the last effective energy value
        u_eff, _ = self.eigenvalue_Eff_iterative(l, E_eff)
        return E_eff, u_eff


    # ------------------------
    # Normalization and energy
    # ------------------------
    def norm(self, u):
        """
        Normalize the radial wavefunction and compute the electron density.

        Parameters
        ----------
        u : np.ndarray
            Radial wavefunction (not normalized).

        Returns
        -------
        den : np.ndarray
            Radial electron density, defined as: den(r) = Z * |u_norm(r)|^2
            where `u_norm` is the normalized radial wavefunction and `Z` is 
            the atomic number.
        """
        norm_factor = np.sqrt(np.sum(u**2) * self.h)
        u_norm = u / norm_factor if norm_factor > 0 else u
        self.den = self.Z * u_norm**2
        return self.den
 
    def energy_integral(self, v):
        return 0.5 * self.h * np.sum(v * self.den)

    # ------------------------
    # Main DFT iteration
    # ------------------------
    def run(self, max_iteration=30, tol=1e-5):
        """
        Perform the self-consistent field (SCF) loop for the radial DFT problem.

        The method iteratively updates the effective potential and solves the 
        radial Schrödinger equation until the total energy converges within a 
        given tolerance or the maximum number of iterations is reached.

        Parameters
        ----------
        max_iteration : int
            Maximum number of self-consistent iterations (default: 30).
        tol : float, optional
            Convergence tolerance for the total energy difference between 
            successive iterations (default: 1e-5).

        Returns
        -------
        E_total : float
            Final converged total energy.
        E_corr : float
            Correlation energy contribution 
        iteration : int
            Number of iterations performed until convergence or until 
            `max_iteration` was reached.
        Notes
        -----
        Notes
        -----
        The total energy is evaluated as::

            E_total = 2*E_eff - E_H + 0.5*(-E_x + E_c)

        where the terms correspond to the orbital energy (with spin 
        degeneracy), Hartree correction, and exchange–correlation 
        contributions as defined in this implementation.
        """
        #header of .txt file to save values
        header = f"{"Iter":>5} {"E_total":>12} {"E_eff":>12} {"E_corr":>12} {"E_x":>12} {"E_H":>12}(per atom)"
        logger.info(header)

        # Initialize nuclear potential
        self.nucl_potential()

        E_old, E_new = np.inf, 0.0

        for iteration in range(1, max_iteration + 1):
            if abs(E_old - E_new) <= tol:
                logger.warning(f" DFT loop converged after {iteration-1} iterations with final energy {E_new:10f} a.u.")
               
                break
            E_old = E_new

            # Compute potentials for nn=1 l=0
            self.Vh = self.hartree(1, 0)        
            self.Vx = self.exchange(1, 0)      
            self.Vc = self.correlation(1, 0)    
            self.V_eff = self.Vn + self.Vh + self.Vx + self.Vc

            # Solve radial equation
            E_eff, self.u_eff = self.bisection_Eff(1, 0)
            self.den = self.norm(self.u_eff)

            # Energy contributions
            hartree_energy = self.energy_integral(self.Vh)
            exchange_energy = self.energy_integral(self.Vx)
            correlation_energy = self.energy_integral(self.Vc)
    
            # Total energy from simplified KS-DFT:
            # 2*E_eff (two electrons in 1s) 
            # - hartree_energy (remove double counting) 
            # + 0.5*(-exchange_energy + correlation_energy) (XC correction as defined here)
            E_new = 2 * E_eff - hartree_energy + 0.5 * (-exchange_energy + correlation_energy)
            line = f"{iteration:5d} {E_new:12.6f} {E_eff:12.6f} {0.5*correlation_energy:12.6f} {0.5*exchange_energy:12.6f} {-hartree_energy:12.6f}"
            logger.info(line)

        return E_new , iteration #,  correlation_energy, exchange_energy , hartree_energy ,iteration
        

