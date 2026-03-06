import numpy as np
import matplotlib.pyplot as plt 
import math as mt
from scipy.linalg import expm
from numpy.linalg import norm

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

def QR_decomp(A):
  A = np.array(A,dtype=float)
  m, n = A.shape
  Q,R = np.zeros((m,n)),np.zeros((n,n))
  tA = A.copy()
  if n > 0:
    Q[:,0] = tA[:,0]/norm(tA[:,0])
    R[0,0] = norm(tA[:,0])
    for i in range(1,n):
      Q[:,i] = tA[:,i].copy()
      for j in range(i-1,-1,-1):
        R[j,i] = np.dot(Q[:,j],tA[:,i])
        Q[:,i] = Q[:,i] - np.dot(Q[:,j],tA[:,i])*Q[:,j]
      R[i,i] = norm(Q[:,i])
      Q[:,i] = Q[:,i]/norm(Q[:,i])
  return Q,R

lim = 6
points = 200
NOE = 3
occup = 2

a = np.linspace(-lim,lim,points)
H_o = Kinetic_operator(lim,points)
v_ext = 0.5 * a**2
dx = a[1]-a[0]
iD = np.zeros(a.shape[0])
I = np.eye(len(a))



#############################################
x = np.linspace(-lim,lim,points)
alpha = 0.5
phi = np.exp(-alpha*x**2)
phi_1 = x*np.exp(-alpha*x**2)
dt = 0.8 * dx**2 / 2
#############################################
phi_n = phi/norm(phi)
phi_1n = phi_1/norm(phi_1)
c0,c1 = 0.7,0.3
Q = np.column_stack((c0*phi_n,c1*phi_1n))
Qtemp = np.zeros(Q.shape)
Dens_tresh = 1e-12
Convergence = 1
n_x = iD




num_states_to_fill = mt.ceil(NOE / occup)
Ex_pot1D, Ex_energy = calculate_exchange(dx,n_x)
Clmb_pot1D, Clmb_energy = Calculate_ColoumbE(dx,n_x,lim,points)
Vex = np.diag(Ex_pot1D)
Vcb = np.diag(Clmb_pot1D)
V_ext = np.diag(v_ext) 
Comp_Hamil = H_o+Vex+Vcb+V_ext
H_eig, H_wavfn = np.linalg.eigh(Comp_Hamil)

prev_n_x = n_x





while(Convergence>Dens_tresh):
    Orbital_E = 0
    prev_n_x = n_x
    new_density = np.zeros(points)

    O = expm(-dt  * Comp_Hamil)

    for i in range(Q.shape[1]):
        Qtemp[:,i] = O@(Q[:,i])
    NQ,R =  QR_decomp(Qtemp)
    Q = NQ.copy()
    for i in range(num_states_to_fill): 
        current_occup = min(occup, NOE - (i * occup))
        psi_squared = np.abs(Q[:,i])**2
        new_density += (current_occup / dx) * psi_squared 
    n_x = new_density

    for i in range(num_states_to_fill):
        current_occup = min(occup, NOE - (i * occup))
        Hpsi = Comp_Hamil @ Q[:,i]
        normalisation = np.dot(Q[:,i],Q[:,i])
        Orbital_E += current_occup * np.dot(Q[:,i],Hpsi)/normalisation
        



    Ex_pot1D, Ex_energy = calculate_exchange(dx,n_x)
    Clmb_pot1D, Clmb_energy = Calculate_ColoumbE(dx,n_x,lim,points)
    Vex = np.diag(Ex_pot1D)
    Vcb = np.diag(Clmb_pot1D)
    V_ext = np.diag(v_ext) 
    Comp_Hamil = H_o+Vex+Vcb+V_ext


    total_v_x = Ex_pot1D + Clmb_pot1D + v_ext
        
    pot_integral = np.sum(n_x * total_v_x) * dx
    Kinetic_Energy = Orbital_E - pot_integral
    ext_pot_energy = np.sum(n_x * v_ext) * dx
    Total_Energy = Kinetic_Energy + Clmb_energy + Ex_energy + ext_pot_energy

    print(f"Final Kinetic Energy: {Kinetic_Energy:.6f}")
    print(f"Final Total Energy: {Total_Energy:.6f}")
    Convergence = np.sqrt(np.sum((n_x - prev_n_x)**2) / n_x.shape[0])
    print(Convergence)

    

