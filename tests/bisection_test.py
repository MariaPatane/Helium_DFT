import pytest
import numpy as np
from DFT.DFT_solver import DFTHelium


# Bisection tests

def test_bisection_converges_to_first_threshold(monkeypatch, dft_solver):
    """
        nn=1 => nodes_expected=0. 
    """
    E1 = -3.0
    def fake_iter(self, l, E_eff, eps=1e-10, normalize=True):
        nodes = 0 if E_eff < E1 else 1
        u = np.ones(self.N)
        return u, nodes

    monkeypatch.setattr(DFTHelium, "eigenvalue_Eff_iterative", fake_iter)

    E_eff, u = dft_solver.bisection_Eff(nn=1, l=0, tol=1e-8, Emin=-10.0, Emax=0.0)
    assert -10.0 < E_eff < 0.0
    assert abs(E_eff - E1) <= 1e-8
    assert u.shape == (dft_solver.N,)

def test_bisection_converges_to_second_threshold(monkeypatch, dft_solver):
    """
       nn=2 => nodes_expected=1. 
    """
    E1, E2 = -5.0, -1.0
    def fake_iter(self, l, E_eff, eps=1e-10, normalize=True):
        if E_eff < E1:
            nodes = 0
        elif E_eff < E2:
            nodes = 1
        else:
            nodes = 2
        u = np.full(self.N, 2.0)
        return u, nodes

    monkeypatch.setattr(DFTHelium, "eigenvalue_Eff_iterative", fake_iter)

    E_eff, u = dft_solver.bisection_Eff(nn=2, l=0, tol=1e-8, Emin=-10.0, Emax=0.0)
    assert -10.0 < E_eff < 0.0
    assert abs(E_eff - E2) <= 1e-8
    assert u.shape == (dft_solver.N,)


def test_hydrogen_coulomb_bisection_matches_analytic(hydrogen_solver_physical_test):
    """
       The test verify that applying just bisection for level 
       1s of hydrogen atom, neglecting hartree and XC 
       correction, Eeff is near to -0.5 a.u
    """ 
    #coulom potential
    hydrogen_solver_physical_test.nucl_potential()
    # no Hartree, exchange and correlation
    hydrogen_solver_physical_test.Vh[:] = 0.0
    hydrogen_solver_physical_test.Vx[:] = 0.0
    hydrogen_solver_physical_test.Vc[:] = 0.0
    hydrogen_solver_physical_test.V_eff = hydrogen_solver_physical_test.Vn

    #ground state nn=1 l=0
    E_eff, u = hydrogen_solver_physical_test.bisection_Eff(nn=1, l=0, tol=1e-8, Emin=-1.0, Emax=0.0)

    assert np.isfinite(E_eff)
    assert abs(E_eff + 0.5) < 2e-2, f"E={E_eff:.4f} differs from -0.5 Ha"
    assert u.shape == (hydrogen_solver_physical_test.N,)