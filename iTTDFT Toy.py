import numpy as np
import matplotlib.pyplot as plt 
import math as mt

def Kinetic_operator(lim,pts):
    psi_x = np.linspace(-lim,lim,pts)
    a = psi_x[1]-psi_x[0]
    KE = np.zeros((len(psi_x),len(psi_x)))
    KE[0,0],KE[0,1] = 1/a**2, -1/(2*a**2) 
    KE[pts-1,pts-1],KE[pts-1,pts-2] =  1/a**2, -1/(2*a**2)

    count_j = 1
    for i in range(1,pts-1):
        KE[i,count_j-1], KE[i,count_j],KE[i,count_j+1] = -1/(2*a**2), 1/a**2 , -1/(2*a**2)
        count_j = count_j+1
    return KE

    def calculate_exchange(dx,density):
    Ex_potential = -(3/mt.pi)*density**(1/3)
    sum = 0
    for i in range(len(density)):
        sum = sum + dx*(density[i])**4/3
    Ex_Energy = sum*(-3/4)*(3/mt.pi)
    return Ex_potential,Ex_Energy

        
def Calculate_ColoumbE(dx, density,lim, pts):
    
    x_c = np.linspace(-lim,lim, pts)
    dx,dx_p = x_c[1]-x_c[0],x_c[1]-x_c[0]
    nxsum = 0
    for nx,x in zip(density,x_c):
        npsum = 0
        for nx_p,xp in zip(density,x_c):
            npsum = npsum + (nx*nx_p/np.sqrt((x-xp)**2+1))*dx_p
        nxsum = nxsum+ npsum*dx
    ColumbE = 0.5*nxsum
    Coloumb_P = []
    for x in x_c:
        npsum = 0
        for nx_p,xp in zip(density,x_c):
            npsum = npsum + (nx_p/np.sqrt((x-xp)**2+1))*dx_p
        Coloumb_P.append(npsum)
    return Coloumb_P,ColumbE

        
def SCF_loop(lim,points,occup,NOE):
    a = np.linspace(-lim,lim,points)
    H_o = Kinetic_operator(lim,points)
    v_ext = 0.5 * a**2
    dx = a[1]-a[0]
    iD = np.zeros(a.shape[0])
    num_states_to_fill = mt.ceil(NOE / occup)
    n_x = iD
    Convergence = 1
    Dens_tresh = 1e-12
    while(Convergence>Dens_tresh):
        Ex_pot1D, Ex_energy = calculate_exchange(dx,n_x)
        Clmb_pot1D, Clmb_energy = Calculate_ColoumbE(dx,n_x,lim,points)
        Vex = np.diag(Ex_pot1D)
        Vcb = np.diag(Clmb_pot1D)
        V_ext = np.diag(v_ext) 
        Comp_Hamil = H_o+Vex+Vcb+V_ext
        H_eig, H_wavfn = np.linalg.eigh(Comp_Hamil)
        new_density = np.zeros(points)
        Orbital_E = 0.0
        prev_n_x = n_x
        for i in range(num_states_to_fill): 
            current_occup = min(occup, NOE - (i * occup))
            psi_squared = np.abs(H_wavfn[:, i])**2
            new_density += (current_occup / dx) * psi_squared 
            Orbital_E += current_occup * H_eig[i]
        n_x = new_density
        total_v_x = Ex_pot1D + Clmb_pot1D + v_ext
        
        pot_integral = np.sum(n_x * total_v_x) * dx
        Kinetic_Energy = Orbital_E - pot_integral
        ext_pot_energy = np.sum(n_x * v_ext) * dx
        Total_Energy = Kinetic_Energy + Clmb_energy + Ex_energy + ext_pot_energy

        print(f"Final Kinetic Energy: {Kinetic_Energy:.6f}")
        print(f"Final Total Energy: {Total_Energy:.6f}")
        Convergence = np.sqrt(np.sum((n_x - prev_n_x)**2) / n_x.shape[0])
        print(Convergence)
SCF_loop(6,500,2,8)
    

