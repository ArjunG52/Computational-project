import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Setup ---
xmax = 6.0      # Box size
Ng = 200        # Number of grid points
Nn = 4          # Number of states

x_g = np.linspace(-xmax, xmax, Ng)
dx = x_g[1] - x_g[0]

vext_g = 0.5 * x_g**2  # External potential

# --- 2. Kinetic Energy Operator ---
T_gg = np.zeros((Ng, Ng))  
for i in range(Ng):
    T_gg[i, i] = -2.0
    if i > 0:
        T_gg[i, i - 1] = 1.0
        T_gg[i - 1, i] = 1.0
T_gg *= -0.5 / dx**2

# --- 3. Initializations ---
# Initialize density as even
n_g = 2.0 * Nn / (Ng * dx) * np.ones(Ng)
print('Initial charge', n_g.sum() * dx)

# Nn states, each one doubly occupied
vhartree_g = np.zeros(Ng)
vx_g = np.zeros(Ng)

# --- 4. Helper Functions ---
def soft_poisson_solve(n_g):
    vhartree_g = np.zeros(Ng)
    for i in range(Ng):
        for j in range(Ng):
            vhartree_g[i] += n_g[j] / np.sqrt(1.0 + (x_g[i] - x_g[j])**2)
    vhartree_g *= dx
    return vhartree_g

# --- 5. Self-Consistent Field (SCF) Loop ---
density_change_integral = 1.0

while density_change_integral > 1e-6:
    
    # Calculate Hamiltonian
    veff_g = vext_g + vhartree_g + vx_g
    H_gg = T_gg + np.diag(veff_g)  
    
    # Solve KS equations
    eps_n, psi_gn = np.linalg.eigh(H_gg)
    print('Energies', ' '.join('{:4f}'.format(eps) for eps in eps_n[:Nn]))
    
    # Normalize states (states are normalized, but not in our dx metric)
    psi_gn /= np.sqrt(dx)
    
    # Update density
    nold_g = n_g
    n_g = 2.0 * (psi_gn[:, :Nn]**2).sum(axis=1)
    density_change_integral = np.abs(nold_g - n_g).sum() * dx
    
    # Charge conservation check
    charge = n_g.sum() * dx
    print('Number of electrons', charge)
    print('Convergence err', density_change_integral)
    assert abs(charge - 2.0 * Nn) < 1e-14
    
    # Calculate Hartree potential
    vhartree_g = soft_poisson_solve(n_g)
    Ehartree = 0.5 * (vhartree_g * n_g).sum() * dx
    print('Electrostatic energy', Ehartree)
    
    # Calculate exchange potential (we won't bother with correlation!)
    vx_g = -(3. / np.pi * n_g)**(1. / 3.)
    Ex = -3. / 4. * (3. / np.pi)**(1. / 3.) * (n_g**(4. / 3.)).sum() * dx
    print('Exchange energy', Ex)
    
    # Calculate Energies
    Eks = 2.0 * eps_n[:Nn].sum()  # Band structure energy
    Ekin = Eks - (veff_g * n_g).sum() * dx
    print('Ekin', Ekin)
    
    Epot = Ehartree + Ex + (vext_g * n_g).sum() * dx
    print('Epot', Epot)
    
    Etot = Ekin + Epot
    print('Energy', Etot)
    print('-' * 40) # Added a small visual separator for console output readability

# --- 6. Plotting ---
for i in range(Nn):
    plt.plot(x_g, psi_gn[:, i], label='n={}, e={:3f}'.format(i + 1, eps_n[i]))
    plt.legend(loc='lower right')

plt.show()