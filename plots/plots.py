import matplotlib.pyplot as plt
import numpy as np
from DFT.DFT_solver import DFTHelium

'''
   Plots of the hydrogen 1s radial wavefunction.
   Computed via bisection in the pure Coulomb potential (no SCF: V_eff = V_n).
   Expected ground-state energy: E = -0.5 a.u.
'''
#parameters
m = DFTHelium(Z=1, r_max=20.0, h=0.001)
m.nucl_potential()
m.Vh[:] = 0.0
m.Vx[:] = 0.0
m.Vc[:] = 0.0
m.V_eff = m.Vn
#bisection
E_eff, u = m.bisection_Eff(nn=1, l=0, tol=1e-8, Emin=-1.0, Emax=0.0)
#normalization 
norm = np.sqrt(np.sum(u**2) * m.h)
u_n = u / norm if norm > 0 else u

# plot of  u(r) normalized
plt.figure()
plt.plot(m.x, u_n, label="u(r) normalized")
plt.xlabel("r [a.u.]"); plt.ylabel("u(r)")
plt.title(f"H 1s via bisection E ≈ {E_eff:.4f} a.u")
plt.grid(True)
plt.savefig("hydrogen_radial_wavefunction_1s.png")

'''
   Plots of the numerical solution U(r) for hartree
   energy and confront with analitical solution 
   U(r) = 1 - (r+1) * exp(-2r)
'''

#density for 1s hydrogen 
m.den = 4 * (m.x**2) * np.exp(-2*m.x) 

#hartree energy for nn=1 l=0
m.hartree(1, 0)

# Analitical solution
U_exact = 1 - (m.x + 1) * np.exp(-2 * m.x)

plt.figure()
plt.plot(m.x, m.U,label="Numerical solution")
plt.plot(m.x, U_exact, color='r',linestyle = '--',label ="Analitical solution")
plt.xlabel("r [a.u.]")
plt.ylabel("U(r)")
plt.title("Potential and eigenenergy")
plt.grid(True)
plt.legend()
plt.savefig("hydrogen_hartree_energy.png")

plt.show()


