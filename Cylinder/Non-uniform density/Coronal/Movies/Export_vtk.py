# Import the required modules
import matplotlib; matplotlib.use('agg') ##comment out to show figs
import numpy as np
import scipy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sym
import math
from scipy.optimize import fsolve
import cmath
from matplotlib import animation
from scipy import interpolate
import pickle
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.polynomial as poly
import types
from matplotlib.animation import FuncAnimation
import os
import struct
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy import interpolate


#mpl.use('agg')

###################    REGULAR SPACED GRID    ###########
#def makeDumpVTK2(origen,dx,dy,dz,variables,varList,dumpFile):
#
#    ax = x.shape[0]
#    ay = x.shape[1]
#    az = x.shape[2]
#
#    fid = open(dumpFile + '.vtk','w')
#
#    fid.write("# vtk DataFile Version 3.0 \n")
#    fid.write("vtk chanput \n")
#    fid.write("BINARY \n")
#    fid.write("DATASET STRUCTURED_POINTS \n")
#    fid.write("DIMENSIONS  %s %s %s  \n" % (ax,ay,az))
#    fid.write("ORIGIN  %f %f %f  \n" % (origen[0],origen[1],origen[2]))
#    fid.write("SPACING %f %f %f  \n" % (dx,dy,dz))
#
#    fid.close()
#
#    fid = open(dumpFile + '.vtk','a')
#    fid.write("\nPOINT_DATA %s  " % (ax*ay*az) )
#    fid.close()
#   
#    n = 0
#    for var in varList:
#        fid = open(dumpFile + '.vtk','a')
#        fid.write("\nSCALARS %s float \n" % var)
#        fid.write("LOOKUP_TABLE default \n")
#        fid.close()
#        fid = open(dumpFile + '.vtk','ab')
#        data =  variables[n]
#        for k in range(az):
#            for j in range(ay):
#                for i in range(ax):
#                    fid.write(struct.pack(">f", data[i,j,k]))
#        n = n+1
#        fid.close()    
#

#####################################################


###############     IRREGULAR SPACED GRID    ######################
def makeDumpVTK(x,y,z,variables,varList,dumpFile):

    ax = x.shape[0]
    ay = x.shape[1]
    az = x.shape[2]

    fid = open(dumpFile + '.vtk','w')

    fid.write("# vtk DataFile Version 3.0 \n")
    fid.write("vtk output \n")
    fid.write("BINARY \n")
    fid.write("DATASET STRUCTURED_GRID \n")
    fid.write("DIMENSIONS  %s %s %s  \n" % (ax,ay,az))
    fid.write("POINTS %s float  \n" % (ax*ay*az))

    fid.close()
    fid = open(dumpFile + '.vtk','ab')
   
    for k in range(az):
        for j in range(ay):
            for i in range(ax):
                fid.write(struct.pack(">f", x[i,j,k]))
                fid.write(struct.pack(">f", y[i,j,k]))
                fid.write(struct.pack(">f", z[i,j,k]))
    fid.close()

    fid = open(dumpFile + '.vtk','a')
    fid.write("\nPOINT_DATA %s  " % (ax*ay*az) )
    fid.close()
   
    n = 0
    for var in varList:
        fid = open(dumpFile + '.vtk','a')
        fid.write("\nSCALARS %s float \n" % var)
        fid.write("LOOKUP_TABLE default \n")
        fid.close()
        fid = open(dumpFile + '.vtk','ab')
        data =  variables[n]
        for k in range(az):
            for j in range(ay):
                for i in range(ax):
                    fid.write(struct.pack(">f", data[i,j,k]))
        n = n+1
        fid.close()        
        
#####################################################

      

# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
        

########################################
c_i0 = 1.
vA_e = 5.*c_i0     #5.*c_i #-coronal        #0.7*c_i -photospheric
vA_i0 = 2.*c_i0     #2*c_i #-coronal        #1.7*c_i  -photospheric
c_e = 0.5*c_i0      #0.5*c_i #- coronal          #1.5*c_i  -photospheric

cT_i0 = np.sqrt((c_i0**2 * vA_i0**2)/(c_i0**2 + vA_i0**2))
cT_e = np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)

print('rho_e    =', rho_e)


c_kink = np.sqrt(((rho_i0*vA_i0**2)+(rho_e*vA_e**2))/(rho_i0+rho_e))

v_phi = 0.
v_z = 0.


def v_iz(r):
  return v_z*r


P_0 = c_i0**2*rho_i0/gamma
P_e = c_e**2*rho_e/gamma

T_0 = P_0/rho_i0
T_e = P_e/rho_e

B_0 = vA_i0*np.sqrt(rho_i0)
B_e = vA_e*np.sqrt(rho_e)
B_phi = 0.

B_tot_i = np.sqrt(B_0**2+B_phi**2)

P_tot_0 = P_0 + B_0**2/2.
P_tot_e = P_e + B_e**2/2.


print('PT_i   =', P_tot_0)
print('PT_e   =', P_tot_e)

Kmax = 4.5

ix = np.linspace(1., 0.001, 500.)  # 1e3   # inside slab x values


r0=0.  #mean
dr=0.9 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density
A = 1.


def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   


#################
dr1e5=1e5 #standard dev
def profile_1e5(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr1e5**2)))   
profile1e5_np=sym.lambdify(rr,profile_1e5(rr),"numpy")   #In order to evaluate we need to switch to numpy

dr3=3. #standard dev
def profile_3(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr3**2)))   
profile3_np=sym.lambdify(rr,profile_3(rr),"numpy")   #In order to evaluate we need to switch to numpy

dr15=1.5 #standard dev
def profile_15(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr15**2)))   
profile15_np=sym.lambdify(rr,profile_15(rr),"numpy")   #In order to evaluate we need to switch to numpy

dr09=0.9 #standard dev
def profile_09(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr09**2)))   
profile09_np=sym.lambdify(rr,profile_09(rr),"numpy")   #In order to evaluate we need to switch to numpy




################################################
def rho_i(r):                  # Define the internal density profile
    return rho_A*profile(r)


#############################################

def P_i(r):
    return ((c_i(r))**2*rho_i(r)/gamma)


#############################################

def T_i(r):     #use this for constant P
    return (P_i(r)/rho_i(r))


##########################################

def vA_i(r):                  # consatnt temp
    return (B_i(r)+B_phi)/(sym.sqrt(rho_i(r)))

##########################################

def B_i(r):  #constant rho
    return B_0


###################################

def PT_i(r):
    return P_i(r) + B_i(r)**2/2.

###################################

def c_i(r):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(r)) - 0.5*gamma*vA_i(r)**2)

###################################################
def cT_i(r):                 # Define the internal tube speed
    return sym.sqrt(((c_i(r))**2 * (vA_i(r))**2) / ((c_i(r))**2 + (vA_i(r))**2))

#####################################################

rho_i_np=sym.lambdify(rr,rho_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
cT_i_np=sym.lambdify(rr,cT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
c_i_np=sym.lambdify(rr,c_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
vA_i_np=sym.lambdify(rr,vA_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
P_i_np=sym.lambdify(rr,P_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
T_i_np=sym.lambdify(rr,T_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
B_i_np=sym.lambdify(rr,B_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
PT_i_np=sym.lambdify(rr,PT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy

#####   SPEEDS AT BOUNDARY    ###

rho_bound = rho_i_np(-1)
c_bound = c_i_np(-1)
vA_bound = vA_i_np(-1)
cT_bound = np.sqrt(c_bound**2 * vA_bound**2 / (c_bound**2 + vA_bound**2))
c_kink_bound = np.sqrt(((rho_bound*vA_bound**2)+(rho_e*vA_e**2))/(rho_bound+rho_e))

P_bound = P_i_np(-1)
print(P_bound)
internal_pressure = P_i_np(ix)
print('internal gas pressure  =', internal_pressure)

print('c_i0 ratio   =',   c_e/c_i0)
print('c_i ratio   =',   c_e/c_bound)
print('c_i boundary   =', c_bound)
print('vA_i boundary   =', vA_bound)
print('rho_i boundary   =', rho_bound)

print('temp = ', T_i_np(ix))

wavenumber = np.linspace(0.01,3.5,20)      #(1.5, 1.8), 5     


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_width1e5.pickle', 'rb') as f:
    sol_omegas1e5, sol_ks1e5, sol_omegas_kink1e5, sol_ks_kink1e5 = pickle.load(f)

with open('Cylindrical_coronal_width3.pickle', 'rb') as f:
    sol_omegas3, sol_ks3, sol_omegas_kink3, sol_ks_kink3 = pickle.load(f)

with open('Cylindrical_coronal_width15.pickle', 'rb') as f:
    sol_omegas15, sol_ks15, sol_omegas_kink15, sol_ks_kink15 = pickle.load(f)

with open('Cylindrical_coronal_width09.pickle', 'rb') as f:
    sol_omegas09, sol_ks09, sol_omegas_kink09, sol_ks_kink09 = pickle.load(f)


### SORT ARRAYS IN ORDER OF wavenumber ###
sol_omegas1e5 = [x for _,x in sorted(zip(sol_ks1e5,sol_omegas1e5))]
sol_ks1e5 = np.sort(sol_ks1e5)

sol_omegas1e5 = np.array(sol_omegas1e5)
sol_ks1e5 = np.array(sol_ks1e5)

sol_omegas_kink1e5 = [x for _,x in sorted(zip(sol_ks_kink1e5,sol_omegas_kink1e5))]
sol_ks_kink1e5 = np.sort(sol_ks_kink1e5)

sol_omegas_kink1e5 = np.array(sol_omegas_kink1e5)
sol_ks_kink1e5 = np.array(sol_ks_kink1e5)


### SORT ARRAYS IN ORDER OF wavenumber ###
sol_omegas3 = [x for _,x in sorted(zip(sol_ks3,sol_omegas3))]
sol_ks3 = np.sort(sol_ks3)

sol_omegas3 = np.array(sol_omegas3)
sol_ks3 = np.array(sol_ks3)

sol_omegas_kink3 = [x for _,x in sorted(zip(sol_ks_kink3,sol_omegas_kink3))]
sol_ks_kink3 = np.sort(sol_ks_kink3)

sol_omegas_kink3 = np.array(sol_omegas_kink3)
sol_ks_kink3 = np.array(sol_ks_kink3)


### SORT ARRAYS IN ORDER OF wavenumber ###
sol_omegas15 = [x for _,x in sorted(zip(sol_ks15,sol_omegas15))]
sol_ks15 = np.sort(sol_ks15)

sol_omegas15 = np.array(sol_omegas15)
sol_ks15 = np.array(sol_ks15)

sol_omegas_kink15 = [x for _,x in sorted(zip(sol_ks_kink15,sol_omegas_kink15))]
sol_ks_kink15 = np.sort(sol_ks_kink15)

sol_omegas_kink15 = np.array(sol_omegas_kink15)
sol_ks_kink15 = np.array(sol_ks_kink15)


### SORT ARRAYS IN ORDER OF wavenumber ###
sol_omegas09 = [x for _,x in sorted(zip(sol_ks09,sol_omegas09))]
sol_ks09 = np.sort(sol_ks09)

sol_omegas09 = np.array(sol_omegas09)
sol_ks09 = np.array(sol_ks09)

sol_omegas_kink09 = [x for _,x in sorted(zip(sol_ks_kink09,sol_omegas_kink09))]
sol_ks_kink09 = np.sort(sol_ks_kink09)

sol_omegas_kink09 = np.array(sol_omegas_kink09)
sol_ks_kink09 = np.array(sol_ks_kink09)
##########################################################################


test_arr_k = []
test_arr_omega = []

#############################################################

for i in range(len(sol_omegas_kink09)):
    v_ph = sol_omegas_kink09[i]/sol_ks_kink09[i]
    
    #slow body
#    if v_ph > cT_i0 and v_ph < c_i0 and sol_ks_kinkv0025[i] < 1.5:
#        test_arr_k.append(sol_ks_kinkv0025[i])
#        test_arr_omega.append(sol_omegas_kinkv0025[i])


#    if v_ph < c_kink and v_ph > vA_i0: 
#        test_arr_k.append(sol_ks_kink1e5[i])
#        test_arr_omega.append(sol_omegas_kink1e5[i])

#    if v_ph < c_kink_bound and v_ph > vA_i0: 
#        test_arr_k.append(sol_ks_kink15[i])
#        test_arr_omega.append(sol_omegas_kink15[i])

#    if v_ph < c_kink and v_ph > vA_i0: 
#        test_arr_k.append(sol_ks_kink3[i])
#        test_arr_omega.append(sol_omegas_kink3[i])

    if v_ph < c_kink_bound and v_ph > vA_i0: 
        test_arr_k.append(sol_ks_kink09[i])
        test_arr_omega.append(sol_omegas_kink09[i])


    #fast kink branch
#    if v_ph < c_kink and v_ph > vA_i0: 
#        test_arr_k.append(sol_ks_kinkv0025[i])
#        test_arr_omega.append(sol_omegas_kinkv0025[i])

         
test_arr_k = np.array(test_arr_k)
test_arr_omega = np.array(test_arr_omega)     
         
test_k = test_arr_k[20]     #25 for roughly k=1          # 0 for k=0.1
test_w = test_arr_omega[20]    # 25 for 1.5   #20 for 3   # 20 for 0.9    # 25 for 1e5

#test_k = sol_ksv00[121]      #101 good
#test_w = sol_omegasv00[121]  

#test_k = sol_ks_kinkv0025[121]     
#test_w = sol_omegas_kinkv0025[121]  



#####   PLOT DISPERSION DIAGRAM


plt.figure()
#plt.title("$ v_{\varphi} = 0.01$")
ax = plt.subplot(111)
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'b.', markersize=4.)

#ax.plot(sol_ks15, (sol_omegas15/sol_ks15), 'r.', markersize=4.)
#ax.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'b.', markersize=4.)

#ax.plot(sol_ks3, (sol_omegas3/sol_ks3), 'r.', markersize=4.)
#ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), 'b.', markersize=4.)

ax.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)
ax.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'b.', markersize=4.)


#ax.plot(sol_ksv001, (sol_omegasv001/sol_ksv001), 'r.', markersize=4.)
#ax.plot(sol_ks_kinkv001, (sol_omegas_kinkv001/sol_ks_kinkv001), 'b.', markersize=4.)


#ax.plot(sol_ksv0025, (sol_omegasv0025/sol_ksv0025), 'r.', markersize=4.)
#ax.plot(sol_ks_kinkv0025, (sol_omegas_kinkv0025/sol_ks_kinkv0025), 'b.', markersize=4.)

ax.plot(test_k, (test_w/test_k), 'b.', markersize=15.)


#ax.plot(sol_ksv005, (sol_omegasv005/sol_ksv005), 'r.', markersize=4.)
#ax.plot(sol_ks_kinkv005, (sol_omegas_kinkv005/sol_ks_kinkv005), 'b.', markersize=4.)


#ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega/fast_sausage_branch1_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega/fast_sausage_branch2_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega/fast_sausage_branch3_k, 'r.', markersize=4.)

#ax.plot(k_new, ffit, color='r')
#ax.plot(k_new_2, ffit_2, color='r')
#ax.plot(k_new_3, ffit_3, color='r')

#ax.plot(fast_kink_branch1_k, fast_kink_branch1_omega/fast_kink_branch1_k, 'y.', markersize=4.)
#ax.plot(fast_kink_branch2_k, fast_kink_branch2_omega/fast_kink_branch2_k, 'g.', markersize=4.)
#ax.plot(fast_kink_branch3_k, fast_kink_branch3_omega/fast_kink_branch3_k, 'k.', markersize=4.)

#ax.plot(k_kink_new, ffit_kink, color='b')
#ax.plot(k_kink_new_2, ffit_2_kink, color='b')

#ax.plot(slow_body_sausage_k, (slow_body_sausage_omega/slow_body_sausage_k), 'r.', markersize=4.)
#ax.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)

#ax.plot(slow_body_sausage_branch1_k, slow_body_sausage_branch1_omega/slow_body_sausage_branch1_k, 'y.', markersize=4.)   # body sausage
#ax.plot(slow_body_kink_branch1_k, slow_body_kink_branch1_omega/slow_body_kink_branch1_k, 'y.', markersize=4.)   # body kink
#ax.plot(slow_body_sausage_branch2_k, slow_body_sausage_branch2_omega/slow_body_sausage_branch2_k, 'g.', markersize=4.)   # body sausage
#ax.plot(slow_body_kink_branch2_k, slow_body_kink_branch2_omega/slow_body_kink_branch2_k, 'g.', markersize=4.)   # body kink

#ax.plot(sb_k_new, sb_ffit, color='r')
#ax.plot(sbk_k_new, sbk_ffit, color='b')

#ax.plot(sbb_k_new, sbb_ffit, color='r')
#ax.plot(sbbk_k_new, sbbk_ffit, color='b')

#ax.plot(new_s_k_new, new_s_ffit, color='r')
#ax.plot(newK_k_new, newK_ffit, color='b')

ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
#ax.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax.annotate( ' $c_{e}, c_{Te}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)

ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

#ax.axhline(y=-cT_i0+v_twist, color='k', linestyle='solid', label='_nolegend_')
#ax.annotate( ' $-c_{Ti}+U_{\phi}$', xy=(Kmax, -cT_i0+v_twist), fontsize=20)

#ax.axhline(y=c_i0+v_twist, color='k', linestyle='solid', label='_nolegend_')
#ax.annotate( ' $c_{i}+U_{\phi}$', xy=(Kmax, c_i0+v_twist), fontsize=20)


ax.annotate( ' $-c_{Ti}$', xy=(Kmax, -cT_i0), fontsize=20)
#ax.annotate( ' $-c_{Te}$', xy=(Kmax, -cT_e), fontsize=20)
ax.annotate( ' $-c_{e}, -c_{Te}$', xy=(Kmax, -c_e), fontsize=20)
ax.annotate( ' $-c_{i}$', xy=(Kmax, -c_i0), fontsize=20)
ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)
ax.annotate( ' $-c_{k}$', xy=(Kmax, -c_kink), fontsize=20)



box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.92, box.height])

ax.set_ylim(-5, 5)   # whole

#ax.set_ylim(0.85, 1.05)  #slow body

#ax.set_ylim(1.9, 5)
#ax.set_ylim(-1.05, -0.8)  #neg slow body


#plt.grid(color='r', linestyle='-', linewidth=2)


#plt.grid(True)
#plt.savefig("cylinder_coronal_vphi_001_neg_slowbody.png")

#plt.show()
#exit()


k = test_k
w = test_w  #test_W*test_k


#######################################################################################
###############    BEGIN CYLINDER FROM SHOOTING METHOD        #########################
#######################################################################################

d = 1.

#assume x0=1
k = test_k
w = test_w  #test_W*test_k

print('k =', k)
print('w =', w)


###########################################################  


osc_image = []

B = 1.
      
rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

B_twist = 0.
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


def B_i(r):  
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))

B_i_np=sym.lambdify(rr,B_i(rr),"numpy")


v_twist = 0.
def v_iphi(r):  
    return v_twist*r   #0.

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy

m = 1.

lx = np.linspace(3.*2.*np.pi/k, 1., 700.)  # 200       # Number of wavelengths/2*pi accomodated in the domain      
                   
m_e = ((((k**2*vA_e**2)-w**2)*((k**2*c_e**2)-w**2))/((vA_e**2+c_e**2)*((k**2*cT_e**2)-w**2)))

xi_e_const = -1/(rho_e*((k**2*vA_e**2)-w**2))       

               ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def f_B(r): 
  return (m*B_iphi(r)/r + k*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + k*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")


def shift_freq(r):
  return (w - (m*(v_iphi(r))/r) - k*v_z)
 
shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
 
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(k*B_i(r))/(sym.sqrt(rho_i(r))))

alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2 + vA_i(r)**2)))

cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")
 
def D(r):  
  return (rho_i(r)*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")

def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*(v_iphi(r))**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*(v_iphi(r))*((m*B_iphi(r)/r)+(k*B_i(r)))/r))

Q_np=sym.lambdify(rr,Q(rr),"numpy")

def T(r):
  return ((((m*B_iphi(r)/r)+(k*B_i(r)))*B_iphi(r)) + rho_i(r)*(v_iphi(r))*shift_freq(r))

T_np=sym.lambdify(rr,T(rr),"numpy")

def C1(r):
  return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
    
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + k**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
   
def C3_diff(r):
  return ((B_iphi(r)/r)**2 - (rho_i(r)*((v_iphi(r))/r)**2))

def C3(r):
  return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
  
C3_np=sym.lambdify(rr,C3(rr),"numpy")
                              
def F(r):  
  return ((r*D(r))/C3(r))

F_np=sym.lambdify(rr,F(rr),"numpy")  
          
def dF(r):   #First derivative of profile in symbols    
  return sym.diff(F(r), r)  

dF_np=sym.lambdify(rr,dF(rr),"numpy")


def g(r):            
  return ( -(sym.diff((r*C1(r)/C3(r)), r)) - (r*(C2(r)-(C1(r)**2/C3(r)))/D(r)))
  
g_np=sym.lambdify(rr,g(rr),"numpy")
  
###################################################### 

                          
def dP_dr_e(P_e, r_e):
       return [P_e[1], (-P_e[1]/r_e + (m_e+((m**2)/(r_e**2)))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)    #was lx 
left_P_solution = Ls[:,0]     # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 
          

def dP_dr_i(P_i, r_i):
       return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]

           
def objective_dPi(dPi):   
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]  #0 for kink 1 for sausage
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)  #Pressure perturbation solution for left hand side

normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))




###############      GATHER VARIABLES    ##################


### Make arrays time dependant here [time, radial]

spatial = np.concatenate((ix[::-1], lx[::-1]), axis=None)     

radial_displacement = np.concatenate((inside_xi_solution[::-1], left_xi_solution[::-1]), axis=None)  
radial_PT = np.concatenate((inside_P_solution[::-1], left_P_solution[::-1]), axis=None)  

normalised_radial_displacement = np.concatenate((normalised_inside_xi_solution[::-1], normalised_left_xi_solution[::-1]), axis=None)  
normalised_radial_PT = np.concatenate((normalised_inside_P_solution[::-1], normalised_left_P_solution[::-1]), axis=None)  




fig, (ax, ax2) = plt.subplots(2,1, sharex=False)
ax.set_title('kink') 
ax.axvline(x=B, color='r', linestyle='--')
ax.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
#ax.set_ylim(-1.2,1.2)
#ax.set_xlim(0.001, 2.)

ax.plot(spatial, radial_PT, 'k')
#ax.plot(spatial, normalised_radial_PT, 'k')

ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
#ax2.set_ylim(-1.5,1.)
#ax2.set_xlim(0.001, 2.)

ax2.plot(spatial, radial_displacement, 'k')
#ax2.plot(spatial, normalised_radial_displacement, 'k')

#plt.show()
#exit()



##   calculate inside and outside variables

time = np.linspace(0.01, 2.*np.pi, 80.)  #125   #160 5 pi
step = (max(time)-min(time))/len(time)


######  v_r   #######

outside_v_r = -w*left_xi_solution[::-1]
inside_v_r = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]

radial_vr = np.concatenate((inside_v_r, outside_v_r), axis=None)    


normalised_outside_v_r = -w*normalised_left_xi_solution[::-1]
normalised_inside_v_r = -shift_freq_np(ix[::-1])*normalised_inside_xi_solution[::-1]

normalised_radial_vr = np.concatenate((normalised_inside_v_r, normalised_outside_v_r), axis=None)    


############

inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-k**2*cT_e**2)*(c_e**2+vA_e**2))
radial_xi_z = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 



inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - k**2*vA_e**2))
radial_xi_phi = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    


normalised_inside_xi_phi = inside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_outside_xi_phi = outside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_radial_xi_phi = np.concatenate((normalised_inside_xi_phi, normalised_outside_xi_phi), axis=None)    


######  v_phi   #######

def dv_phi(r):
  return sym.diff(v_iphi(r)/r)

dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")

outside_v_phi = -w*outside_xi_phi
inside_v_phi = -(shift_freq_np(ix[::-1])*inside_xi_phi) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1])

radial_v_phi = np.concatenate((inside_v_phi, outside_v_phi), axis=None)    



######  v_z   #######

def dv_z(r):
  return sym.diff(v_iz(r)/r)

dv_z_np=sym.lambdify(rr,dv_z(rr),"numpy")

outside_v_z = -w*outside_xi_z
inside_v_z = -(shift_freq_np(ix[::-1])*inside_xi_z) - (dv_z_np(ix[::-1])*inside_xi_solution[::-1])

radial_v_z = np.concatenate((inside_v_z, outside_v_z), axis=None)    

######################


fig, (ax, ax2) = plt.subplots(2,1, sharex=False)
ax.set_title('kink') 
ax.axvline(x=B, color='r', linestyle='--')
ax.set_ylabel("$\hat{\u03be}_{phi}$", fontsize=18, rotation=0, labelpad=15)
#ax.set_ylim(0.,1.2)
#ax.set_xlim(0.001, 2.)

ax.plot(spatial, radial_xi_phi, 'k')
#ax.plot(spatial, normalised_radial_xi_phi, 'k')

#ax.plot(spatial, radial_v_phi[0,:], 'r')
#ax.plot(ix[::-1], inside_xi_phi2, 'b--')

ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_z$", fontsize=18, rotation=0, labelpad=15)
#ax2.set_ylim(-20.,0.)
ax2.set_xlim(0.001, 2.)

ax2.plot(spatial, radial_xi_z, 'k')
#ax2.plot(spatial, radial_v_z[0,:], 'r')

#plt.show()

#exit()


########  rho   #######

outside_step = (max(lx[::-1])-min(lx[::-1]))/len(lx[::-1])
inside_step = (max(ix[::-1])-min(ix[::-1]))/len(ix[::-1])

dvr_outside_dr = np.gradient(outside_v_r, outside_step)
dvr_inside_dr = np.gradient(inside_v_r, inside_step)

outside_rho_perturbed = (rho_e*((dvr_outside_dr/lx[::-1]) + (m*outside_v_phi/lx[::-1]) + k*outside_v_z))/w
inside_rho_perturbed = (rho_i0*((dvr_inside_dr/ix[::-1]) + (m*inside_v_phi/ix[::-1]) + k*inside_v_z))/w

radial_rho_perturbed = np.concatenate((inside_rho_perturbed, outside_rho_perturbed), axis=None)    


B=1.
fig, (ax, ax2) = plt.subplots(2,1, sharex=False)
ax.axvline(x=B, color='r', linestyle='--')
ax.set_xlim(0.001, 2.)
ax.set_ylim(-500., 200.)
ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel(r"$\hat{\rho}$", fontsize=18, rotation=0, labelpad=15)
ax.plot(spatial, radial_rho_perturbed, 'k')

ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlim(0.001, 2.)
ax2.set_ylim(-2., 0.5)
ax2.set_ylabel(r"$\hat{\rho}$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlabel("$r$", fontsize=18)
ax2.plot(spatial, radial_rho_perturbed, 'k')

#plt.savefig("Density_pert_test_plot.png")

#plt.show()
#exit()

#########################

########################################

wavenum = k

z = np.linspace(0.01, 5., 16.)   #21  #31

print('z[6]  =', z[6])
#exit()
THETA = np.linspace(0., 2.*np.pi, 60) #50

radii, thetas, Z = np.meshgrid(spatial,THETA,z,sparse=False, indexing='ij')

print(Z.shape)

#exit()
###########

xi_r = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_phi = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_z = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
PT = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_r = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_phi =  np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_z = np.zeros(((len(time), len(spatial), len(THETA), len(z))))

xi_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
P_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
P_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))

density = np.zeros(((len(time), len(spatial), len(THETA), len(z))))

boundary_point_r = np.zeros(((len(spatial), len(THETA), len(z))))

bound_index = np.where(radii[:,0,0] == 1.)

#exit()


for k in range(len(z)):
  for j in range(len(THETA)):
    for i in range(len(spatial)):
      for t in range(len(time)):
        xi_r[t,i,j,k] = radial_displacement[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t])  
        xi_phi[t,i,j,k] = radial_xi_phi[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t])        
        #xi_z[t,i,j,k] = radial_xi_z[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.sin(w*time[t])
        PT[t,i,j,k] = radial_PT[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]) 
        v_r[t,i,j,k] = radial_vr[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]) 
        v_phi[t,i,j,k] = radial_v_phi[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]) 
        v_z[t,i,j,k] = 400.*radial_v_z[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t])   

       
        xi_x[t,i,j,k] = (xi_r[t,i,j,k]*np.cos(thetas[i,j,k]) - xi_phi[t,i,j,k]*np.sin(thetas[i,j,k]))
        xi_y[t,i,j,k] = (xi_r[t,i,j,k]*np.sin(thetas[i,j,k]) + xi_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        v_x[t,i,j,k] = 400.*(v_r[t,i,j,k]*np.cos(thetas[i,j,k]) - v_phi[t,i,j,k]*np.sin(thetas[i,j,k]))   # 100 for 1e5, 3, 1.5   try 400 for 0.9
        v_y[t,i,j,k] = 400.*(v_r[t,i,j,k]*np.sin(thetas[i,j,k]) + v_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        P_x[t,i,j,k] = PT[t,i,j,k]*np.cos(thetas[i,j,k])
        P_y[t,i,j,k] = PT[t,i,j,k]*np.sin(thetas[i,j,k])
               
        density[t,i,j,k] = (radial_rho_perturbed[i]/time[t])*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]) 
               

normalised_v_x = v_x/np.amax(v_x)
normalised_v_y = v_y/np.amax(v_y)

print(np.amax(v_x[:,:,0]))
print(np.amax(v_y[:,:,0]))

#exit()
           
boundary_point_r = radii[bound_index] 
boundary_point_z = Z[bound_index]        

bound_element = bound_index[0][0] 

boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[0,bound_element,:,:]*step
boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[0,bound_element,:,:]*step  # for small k
#boundary_point_z = boundary_point_z + v_z[bound_index]

boundary_point_x = boundary_point_x[0,:,:]
boundary_point_y = boundary_point_y[0,:,:]
boundary_point_z = boundary_point_z[0,:,:]


print(boundary_point_x.shape)
print(boundary_point_z.shape)
print(boundary_point_x.shape)

x = radii*np.cos(thetas)
y = radii*np.sin(thetas)
z = Z


######   EXPORT SOLUTIONS TO .VTK FILES   #######

varlist = ['xi_r', 'xi_phi', 'P_T', 'v_r', 'v_phi', 'xi_x', 'xi_y', 'v_x', 'v_y', 'v_z', 'density']

for t in range(len(time)):
    makeDumpVTK(x,y,z,[ xi_r[t,:,:,:], xi_phi[t,:,:,:], PT[t,:,:,:], v_r[t,:,:,:], v_phi[t,:,:,:], xi_x[t,:,:,:], xi_y[t,:,:,:], v_x[t,:,:,:], v_y[t,:,:,:], v_z[t,:,:,:], density[t,:,:,:]],varlist,'gaussian09_density_slow_kink_0'+ str(t))

exit()


#################################################
print(x.shape)
print(y.shape)
print(z.shape)

el = 6
snapshot = 14.

#fig = plt.figure(figsize=(22,8))
#ax = plt.subplot(121)
#ax2 = plt.subplot(122)
#
#circle_rad =1.
#a = circle_rad*np.cos(THETA) #circle x
#b = circle_rad*np.sin(THETA) #circle y
#
#ax.plot(a,b, 'r--')
#
#ax.set_title('Velocity')
#ax.set_xlim(-1.5, 1.5)
#ax.set_ylim(-1.5, 1.5)
#ax.set_xlabel('$x$', fontsize=18)
#ax.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
#
#new_x = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 650)
#new_y = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 650)  # 100
#
#new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')
#
#flat_x = x[:,:,el].flatten('C')
#flat_y = y[:,:,el].flatten('C')
#    
#points = np.transpose(np.vstack((flat_x, flat_y)))
#
#boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[snapshot,bound_element,:,:]*step    
#boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[snapshot,bound_element,:,:]*step    
#
#boundary_point_x = boundary_point_x[0,:,:]
#boundary_point_y = boundary_point_y[0,:,:]
#
#flat_vx = v_x[snapshot,:,:,el].flatten('C')
#flat_vy = v_y[snapshot,:,:,el].flatten('C')
#
#vx_interp = griddata(points, flat_vx, (new_xx, new_yy), method='cubic')
#vy_interp = griddata(points, flat_vy, (new_xx, new_yy), method='cubic')
#
##ax.plot(new_xx, new_yy, 'k.', markersize=5)
#
#ax.contourf(x[:,:,el], y[:,:,el], PT[snapshot,:,:,el], extend='both', cmap='bwr', alpha=0.3)  
#ax.scatter(boundary_point_x[:,el], boundary_point_y[:,el], s=4., c='blue')
#ax.plot(boundary_point_x[:,el], boundary_point_y[:,el])     
#ax.quiver(new_xx[::5,::5], new_yy[::5,::5], vx_interp[::5,::5], vy_interp[::5,::5], pivot='tail', color='black', scale_units='inches', scale=7., width=0.003)
#
#
#ax2.plot(a,b, 'r--')
#
#ax2.set_title('Vorticity')
#ax2.set_xlim(-1.5, 1.5)
#ax2.set_ylim(-1.5, 1.5)
#ax2.set_xlabel('$x$', fontsize=18)
#ax2.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
#
#
##plt.show()
##exit()
#

#########################

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, bitrate=4000)   #9fps if skipping 5 in time step   15 good

#fig = plt.figure()  # use for triple plot

#ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax2 = fig.add_subplot(2, 3, 3)
#ax3 = fig.add_subplot(2, 3, 6)
#
#box = ax.get_position()
#ax.set_position([box.x0-0.1, box.y0, box.width*1.4, box.height])
#
#box2 = ax2.get_position()
#ax2.set_position([box2.x0-0.05, box2.y0, box2.width*1.4, box2.height])
#
#box3 = ax3.get_position()
#ax3.set_position([box3.x0-0.05, box3.y0, box3.width*1.4, box3.height])


######   interpolate

fig = plt.figure(figsize=(22,8))
ax = plt.subplot(121)
ax2 = plt.subplot(122)

circle_rad =1.
a = circle_rad*np.cos(THETA) #circle x
b = circle_rad*np.sin(THETA) #circle y


#plt.show()
#exit()

def animate(i):     
   ax.clear()
   ax.set_title('$W = 1.5  - kink - k = 1.5$  -  z = 6  -  t = %d' %i)
   ax2.set_title('Vorticity')
   ax.plot(a,b, 'r--')
   ax2.plot(a,b, 'r--')
   
   ax.set_xlim(-1.5, 1.5)
   ax.set_ylim(-1.5, 1.5)
   ax.set_xlabel('$x$', fontsize=18)
   ax.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)

   ax2.set_xlim(-1.5, 1.5)
   ax2.set_ylim(-1.5, 1.5)
   ax2.set_xlabel('$x$', fontsize=18)
   ax2.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)

   boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[i,bound_element,:,:]*step    
   boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[i,bound_element,:,:]*step    
 
   boundary_point_x = boundary_point_x[0,:,:]
   boundary_point_y = boundary_point_y[0,:,:]

   new_x = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 2100)
   new_y = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 2100)  # 100
   
   new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')
   
   flat_x = x[:,:,18].flatten('C')
   flat_y = y[:,:,18].flatten('C')
       
   points = np.transpose(np.vstack((flat_x, flat_y)))
     
   flat_vx = v_x[i,:,:,18].flatten('C')
   flat_vy = v_y[i,:,:,18].flatten('C')
   
   vx_interp = griddata(points, flat_vx, (new_xx, new_yy), method='cubic')
   vy_interp = griddata(points, flat_vy, (new_xx, new_yy), method='cubic')
   
   #ax.plot(new_xx, new_yy, 'k.', markersize=5)
   
   ax.contourf(x[:,:,18], y[:,:,18], PT[i,:,:,18], extend='both', cmap='bwr', alpha=0.2)  
   ax.scatter(boundary_point_x[:,18], boundary_point_y[:,18], s=4., c='blue')
   ax.plot(boundary_point_x[:,18], boundary_point_y[:,18])     
   #ax.quiver(new_xx, new_yy, vx_interp, vy_interp, pivot='tail', color='black', scale_units='inches', scale=7., width=0.003)
   ax.quiver(new_xx[::14,::14], new_yy[::14,::14], vx_interp[::14,::14], vy_interp[::14,::14], pivot='tail', color='black', scale_units='inches', scale=7., width=0.003)

#########################

   el = 18

   flat_dvx = np.gradient(v_x[snapshot,:,:,el].flatten('C'))
   flat_dvy = np.gradient(v_y[snapshot,:,:,el].flatten('C'))
   
   new_x = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 2100)
   new_y = np.linspace(-3.*2.*np.pi/wavenum, 3.*2.*np.pi/wavenum, 2100)  # 100
   new_z = np.linspace(0.01, 10., 31.)   #21  #31
   new_xx, new_yy, new_zz = np.meshgrid(new_x,new_y, new_z, sparse=False, indexing='ij')
   
   [dvx_x, dvx_y, dvx_z] = np.gradient(v_x[i,:,:,:])
   [dvy_x, dvy_y, dvy_z] = np.gradient(v_y[i,:,:,:])
   [dvz_x, dvz_y, dvz_z] = np.gradient(v_z[i,:,:,:])
   
   vort_x_comp = dvz_y - dvy_z
   vort_y_comp = dvx_z - dvz_x
   vort_z_comp = dvy_x - dvx_y
     
   flat_x = x[:,:,:].flatten('C')
   flat_y = y[:,:,:].flatten('C')
   flat_z = z[:,:,:].flatten('C')
       
   points = np.transpose(np.vstack(((flat_x, flat_y, flat_z))))
      
   flat_dvx_y = dvx_y[:,:,:].flatten('C')
   flat_dvx_z = dvx_z[:,:,:].flatten('C')
   flat_dvy_x = dvy_x[:,:,:].flatten('C')
   flat_dvy_z = dvy_z[:,:,:].flatten('C')
   flat_dvz_x = dvz_x[:,:,:].flatten('C')
   flat_dvz_y = dvz_y[:,:,:].flatten('C')
   
   dvx_y_interp = griddata(points, flat_dvx_y, (new_xx, new_yy, new_zz), method='nearest')
   dvx_z_interp = griddata(points, flat_dvx_z, (new_xx, new_yy, new_zz), method='nearest')
   dvy_x_interp = griddata(points, flat_dvy_x, (new_xx, new_yy, new_zz), method='nearest')
   dvy_z_interp = griddata(points, flat_dvy_z, (new_xx, new_yy, new_zz), method='nearest')
   dvz_x_interp = griddata(points, flat_dvz_x, (new_xx, new_yy, new_zz), method='nearest')
   dvz_y_interp = griddata(points, flat_dvz_y, (new_xx, new_yy, new_zz), method='nearest')
   
   vort_x_comp_interp = dvz_y_interp - dvy_z_interp
   vort_y_comp_interp = dvx_z_interp - dvz_x_interp
   vort_z_comp_interp = dvy_x_interp - dvx_y_interp
   
   el_2 = 18
   
   ax2.quiver(new_xx[::14,::14,el_2], new_yy[::14,::14,el_2], vort_x_comp_interp[::14,::14,el_2], vort_y_comp_interp[::14,::14,el_2], pivot='tail', color='black', scale_units='inches', scale=7., width=0.003)
   ax2.contourf(new_xx[:,:,el_2], new_yy[:,:,el_2], vort_z_comp_interp[:,:,el_2], extend='both', cmap='bwr', alpha=0.2)  


anim = FuncAnimation(fig, animate, interval=100, frames=len(time)-1).save('Gaussian_density_15_kink_fast_k15_vorticity.mp4', writer=writer)


#plt.show()

exit()


