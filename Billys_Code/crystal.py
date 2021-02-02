import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import leastsq, minimize

import os
import pickle
import fnmatch
import traceback
import time
import itertools  
import pylab

from make_lattice import make_lattice



################################## SET PARAMETERS ##################################

pottrap     = -400 # trap voltag in volt
freqRW1     = 165e3 # crystal rotation frequency in Hz
potRW       = 0 


Bfield      = 1.998 # Tesla
C2          = -4682 # 1/m**2, trap voltage scaling, from trap geometry
scaleRW     = 309.1 # 1/m**2, rotating wall scaling, from trap geometry
freqRW      = 2*np.pi*freqRW1 

charge      = 1.602e-19 # C
mass        = 9.00652*1.6607E-27 # kg
coulconst   = 8.987551787e9 # Nm**2C**-2
epsilon0    = 8.854187817e-12 # F/m vacuum permittivity


# cyclotron frequency
wc = charge/mass*Bfield
print('cyclotron frequency /MHz: ', wc/(2*np.pi)/1e6)

# axial frequency
wz = np.sqrt(2*charge/mass*pottrap*C2)
print('axial frequency /kHz: ', wz/(2*np.pi)/1e3)

# w1
w1 = np.sqrt(wc**2/4-wz**2/2)

# magnetron frequency
wm = wc/2 - w1
print('magnetron frequency /kHz: ', wm/(2*np.pi)/1e3)

# reduced cyclotron frequency
wp = wc/2 + w1
print('reduced cyclotron frequency /MHz: ', wp/(2*np.pi)/1e6)

# rotating wall "strength", equal to axial potential swing
ww = np.sqrt(2*charge/mass*potRW*scaleRW)
print('rotating wall "strength" /kHz: ', ww/(2*np.pi)/1e3)

# normalized axial trapping frequency wz/(sqrt(2))/wc/2, tha max value is 1
wzn = wz/np.sqrt(2)/(wc/2)
print('normalized axial trapping frequency wz/(sqrt(2))/wc/2: ', wzn)

# effective trapping frequency
weff = np.sqrt(freqRW*wc-freqRW**2-0.5*wz**2)
print('effective trapping frequency /kHz: ', weff/(2*np.pi)/1e3)

axialterm = pottrap*C2
radialterm = Bfield*freqRW - mass/charge*freqRW**2 - pottrap*C2
rotwallterm = potRW*scaleRW
coulconstcharge = coulconst*charge

print("axialterm", axialterm)
print("radialterm", radialterm)
print("rotwallterm", rotwallterm)
print("coulconstcharge", coulconstcharge)



################################## POTENTIAL ENERGY ##################################

def Epot_total(popt, numberions=19, axialterm=axialterm, radialterm=radialterm, 
    rotwallterm=rotwallterm, coulconstcharge=coulconstcharge
    ) -> float:
    """calculate total potential enery in corotating frame, see Wang2013 eq. 5 
    but without extra elementary charge"""
    
    # sum all ion distances for Coulomb potential calculation
    # instead of j!=i, use j>i to avoid double summing (i-j + j-i)
    Coulombdist = 0
    count_coulomb = 0
    for i in range(0,numberions-1):
        for j in range(i+1,numberions):
            Coulombdist = Coulombdist + 1/np.sqrt((popt[i*3]-popt[j*3])**2+
                        (popt[i*3+1]-popt[j*3+1])**2+(popt[i*3+2]-popt[j*3+2])**2)
            count_coulomb = count_coulomb + 1

    # iterate over all ion coordinates for trap potential calculation
    U = 0
    for i in range(0,numberions):
        U = (U + axialterm*popt[i*3+2]**2 
             + 0.5*(radialterm)*(popt[i*3]**2+popt[i*3+1]**2) 
             + rotwallterm*(popt[i*3]**2-popt[i*3+1]**2))
            
    Utot = U + coulconstcharge*Coulombdist

    return Utot


def get_ions(
    shells=3, elongation=0, axialterm=axialterm, radialterm=radialterm, 
    rotwallterm=rotwallterm, coulconstcharge=coulconstcharge
    ) -> np.array:

    ################################## INITIAL ION POSITIONS ##################################
    # hexagonal 2D grid: every shell has 6 ions more than the previous: 1, 6, 12, 18, ...
    # total ion number: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271, 331

    # intial positions
    # shelled hexagonal lattice
    xpos, ypos = make_lattice(shells, elong=elongation)
    numberions = len(xpos)
    zpos = np.zeros((1, numberions)).tolist()[0]

    # convert to vector for optimization: (x1,y1,z1,x2,y2,z2,....)

    popt = np.concatenate(list(zip(xpos, ypos, zpos)))*0.005/1000

    print(popt)

    ################################## MINIMISATION ##################################
    print("Beginning energy minimisation 🤞")
    res = minimize(Epot_total, popt, args=(numberions, axialterm, radialterm, rotwallterm, coulconstcharge),  method='Powell', options={'ftol': 1e-7})#, options={'ftol': 1e-13})#, tol=1e-10)#,options={'xatol': 1e-10, 'maxfev': 50000,'disp': True})

    if not res["success"]:
        print("❌ Failed to minimise energy")
        exit()
    else:
        print("✅ Successfully minimised energy")

    x = res['x'][0::3]*1e6
    y = res['x'][1::3]*1e6
    z = res['x'][2::3]*1e6

    return (x, y, z)