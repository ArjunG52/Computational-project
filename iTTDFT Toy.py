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

a = np.linspace(-lim,lim,points)
H_o = Kinetic_operator(lim,points)
v_ext = 0.5 * a**2
dx = a[1]-a[0]
iD = np.zeros(a.shape[0])
I = np.eye(len(a))


x = np.linspace(-lim,lim,points)
alpha = 0.5
phi = np.exp(-alpha*x**2)
phi_1 = x*np.exp(-alpha*x**2)


































num_states_to_fill = mt.ceil(NOE / occup)
n_x = iD
Ex_pot1D, Ex_energy = calculate_exchange(dx,n_x)
Clmb_pot1D, Clmb_energy = Calculate_ColoumbE(dx,n_x,lim,points)
Vex = np.diag(Ex_pot1D)
Vcb = np.diag(Clmb_pot1D)
V_ext = np.diag(v_ext) 
Comp_Hamil = H_o+Vex+Vcb+V_ext
        

SCF_loop(6,500,2,8)
    

