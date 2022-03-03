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
        
        
##############      BEGIN SYMMETRIC SLAB DISPERSION DIAGRAM       ###############



c_i0 = 1.
vA_e = 5.*c_i0     #5.*c_i #-coronal        #0.7*c_i -photospheric
vA_i0 = 2.*c_i0     #2*c_i #-coronal        #1.7*c_i  -photospheric
c_e = 0.5*c_i0      #0.5*c_i #- coronal          #1.5*c_i  -photospheric

cT_i0 = np.sqrt(c_i0**2 * vA_i0**2 / (c_i0**2 + vA_i0**2))
cT_e = np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)

#rho_e = rho_i*vA_i**2*B_e**2/(vA_e**2*B_i**2)
#print('external density    =', rho_e)

R1 = rho_e/rho_i0
#R1 = 0.8

print('Density ration external/internal    =', R1)


c_kink = np.sqrt(((rho_i0*vA_i0**2)+(rho_e*vA_e**2))/(rho_i0+rho_e))

# variables (reference only):
# W = omega/k
# K = k*x_0

v_phi = 0.
v_z = 0.

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

Kmax = 4.

ix = np.linspace(1., 0.001, 180.)  # inside slab x values


rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

r0=0.  #mean

dr=1e5


U_i0 = 0.05*c_i0     #0.35*vA_i  coronal    
U_e = 0.          #-0.15*vA_i   photospheric      0 - coronal

def v_z(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2))) 

################################################

def rho_i(r):                  
    return rho_i0

#############################################

def P_i(r):
    return ((c_i(r))**2*rho_i(r)/gamma)

#############################################

def T_i(r):     #use this for constant P
    return (P_i(r)/rho_i(r))

##########################################

def vA_i(r):                  # consatnt temp
    return (B_i(r)+B_iphi(r))/(sym.sqrt(rho_i(r)))

##########################################

def B_i(r):  #constant rho
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))


B_twist = 0.01

def B_iphi(r):  #constant rho
    return 0.    #B_twist*r   #0.   #B_twist*r


v_twist = 0.1
def v_iphi(r):  #constant rho
    return 0.     #v_twist*r    #0.   #v_twist*r   #0.

###################################

def PT_i(r):   # USE THIS ONE
    return (sym.diff((P_i(r) + (B_i(r)**2 + B_iphi(r)**2)/2.), r) + B_iphi(r)**2/r)

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
v_z_np=sym.lambdify(rr,v_z(rr),"numpy")   #In order to evaluate we need to switch to numpy


#####################################################

########   READ IN VARIABLES    v_phi = 0.0   #########

#################################################

########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_1e5.pickle', 'rb') as f:
    sol_omegas1e5, sol_ks1e5, sol_omegas_kink1e5, sol_ks_kink1e5 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas1e5 = [x for _,x in sorted(zip(sol_ks1e5,sol_omegas1e5))]
sol_ks1e5 = np.sort(sol_ks1e5)

sol_omegas1e5 = np.array(sol_omegas1e5)
sol_ks1e5 = np.array(sol_ks1e5)

sol_omegas_kink1e5 = [x for _,x in sorted(zip(sol_ks_kink1e5,sol_omegas_kink1e5))]
sol_ks_kink1e5 = np.sort(sol_ks_kink1e5)

sol_omegas_kink1e5 = np.array(sol_omegas_kink1e5)
sol_ks_kink1e5 = np.array(sol_ks_kink1e5)


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_3.pickle', 'rb') as f:
    sol_omegas3, sol_ks3, sol_omegas_kink3, sol_ks_kink3 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas3 = [x for _,x in sorted(zip(sol_ks3,sol_omegas3))]
sol_ks3 = np.sort(sol_ks3)

sol_omegas3 = np.array(sol_omegas3)
sol_ks3 = np.array(sol_ks3)

sol_omegas_kink3 = [x for _,x in sorted(zip(sol_ks_kink3,sol_omegas_kink3))]
sol_ks_kink3 = np.sort(sol_ks_kink3)

sol_omegas_kink3 = np.array(sol_omegas_kink3)
sol_ks_kink3 = np.array(sol_ks_kink3)



########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_15.pickle', 'rb') as f:
    sol_omegas15, sol_ks15, sol_omegas_kink15, sol_ks_kink15 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas15 = [x for _,x in sorted(zip(sol_ks15,sol_omegas15))]
sol_ks15 = np.sort(sol_ks15)

sol_omegas15 = np.array(sol_omegas15)
sol_ks15 = np.array(sol_ks15)

sol_omegas_kink15 = [x for _,x in sorted(zip(sol_ks_kink15,sol_omegas_kink15))]
sol_ks_kink15 = np.sort(sol_ks_kink15)

sol_omegas_kink15 = np.array(sol_omegas_kink15)
sol_ks_kink15 = np.array(sol_ks_kink15)


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_1.pickle', 'rb') as f:
    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas1 = [x for _,x in sorted(zip(sol_ks1,sol_omegas1))]
sol_ks1 = np.sort(sol_ks1)

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)

sol_omegas_kink1 = [x for _,x in sorted(zip(sol_ks_kink1,sol_omegas_kink1))]
sol_ks_kink1 = np.sort(sol_ks_kink1)

sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_06.pickle', 'rb') as f:
    sol_omegas06, sol_ks06, sol_omegas_kink06, sol_ks_kink06 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas06 = [x for _,x in sorted(zip(sol_ks06,sol_omegas06))]
sol_ks06 = np.sort(sol_ks06)

sol_omegas06 = np.array(sol_omegas06)
sol_ks06 = np.array(sol_ks06)

sol_omegas_kink06 = [x for _,x in sorted(zip(sol_ks_kink06,sol_omegas_kink06))]
sol_ks_kink06 = np.sort(sol_ks_kink06)

sol_omegas_kink06 = np.array(sol_omegas_kink06)
sol_ks_kink06 = np.array(sol_ks_kink06)

#####################################################


test_arr_k = []
test_arr_omega = []

for i in range(len(sol_omegas_kink1e5)):
    v_ph = sol_omegas_kink1e5[i]/sol_ks_kink1e5[i]
    
    if v_ph > cT_i0+U_i0 and v_ph < c_i0+U_i0 and sol_ks_kink1e5[i] < 2.5 and sol_ks_kink1e5[i] > 1.5:
        test_arr_k.append(sol_ks_kink1e5[i])
        test_arr_omega.append(sol_omegas_kink1e5[i])

#for i in range(len(sol_omegas_kink06)):
#    v_ph = sol_omegas_kink06[i]/sol_ks_kink06[i]
#    
#    if v_ph > cT_i0+U_i0 and v_ph < c_i0+U_i0 and sol_ks_kink06[i] < 2.5 and sol_ks_kink06[i] > 1.5:
#        test_arr_k.append(sol_ks_kink06[i])
#        test_arr_omega.append(sol_omegas_kink06[i])


#for i in range(len(sol_omegas_kink3)):
#    v_ph = sol_omegas_kink3[i]/sol_ks_kink3[i]
#    
#    if v_ph > cT_i0+U_i0 and v_ph < c_i0+U_i0 and sol_ks_kink3[i] < 1.9 and sol_ks_kink3[i] > 1.75:
#        test_arr_k.append(sol_ks_kink3[i])
#        test_arr_omega.append(sol_omegas_kink3[i])

         
test_arr_k = np.array(test_arr_k)
test_arr_omega = np.array(test_arr_omega)     
         
test_k = test_arr_k[7]
test_w = test_arr_omega[7]



#test_k = sol_ksv00[121]      #101 good
#test_w = sol_omegasv00[121]  

#test_k = sol_ks_kinkv00[121]     
#test_w = sol_omegas_kinkv00[121]  

#####   PLOT DISPERSION DIAGRAM

plt.figure()
#plt.title("$ v_{\varphi} = 0.01$")
ax = plt.subplot(111)
plt.xlabel("$ka$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'r.', markersize=4.)
ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'b.', markersize=4.)


#ax.plot(sol_ks3, (sol_omegas3/sol_ks3), 'r.', markersize=4.)
#ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), 'b.', markersize=4.)

#ax.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
#ax.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'b.', markersize=4.)

ax.plot(test_k, (test_w/test_k), 'b.', markersize=15.)



ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
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

ax.annotate( ' $-c_{Ti}$', xy=(Kmax, -cT_i0), fontsize=20)
ax.annotate( ' $-c_{Te}$', xy=(Kmax, -cT_e), fontsize=20)
ax.annotate( ' $-c_{e}$', xy=(Kmax, -c_e), fontsize=20)
ax.annotate( ' $-c_{i}$', xy=(Kmax, -c_i0), fontsize=20)
ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)
ax.annotate( ' $-c_{k}$', xy=(Kmax, -c_kink), fontsize=20)



box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.92, box.height])

#ax.set_ylim(0, 5)   # whole
ax.set_ylim(-5, 5)   # whole



#plt.grid(color='r', linestyle='-', linewidth=2)


#plt.grid(True)
#plt.savefig("cylinder_coronal_vphi_001_neg_slowbody.png")


#plt.show()
#exit()


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

B_twist = 0.01
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


def B_i(r):  
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))

B_i_np=sym.lambdify(rr,B_i(rr),"numpy")


v_twist = 0.025
def v_iphi(r):  
    return 0.  #v_twist*r   #0.

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


dr = 1e5
def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  



m = 1.

lx = np.linspace(3.*2.*np.pi/k, 1., 2000.)  # Number of wavelengths/2*pi accomodated in the domain      
                   
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
  return (w - (m*(v_iphi(r))/r) - k*v_z(r))
 
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
ax.set_xlim(0.001, 2.)

ax.plot(spatial, radial_PT, 'k')

ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
#ax2.set_ylim(-1.5,1.)
ax2.set_xlim(0.001, 2.)

ax2.plot(spatial, radial_displacement, 'k')

#plt.show()
#exit()


##   calculate inside and outside variables

time = np.linspace(0.01, 6.*np.pi, 100)  #125
step = (max(time)-min(time))/len(time)


######  v_r   #######

inside_v_r = np.zeros((len(time), len(ix)))
outside_v_r = np.zeros((len(time), len(lx)))

for t in range(len(time)):
   outside_v_r[t,:] = -w*left_xi_solution[::-1]*np.cos(w*time[t])
   inside_v_r[t,:] = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]*np.cos(w*time[t])


print(outside_v_r.shape)
print(inside_v_r.shape)

radial_vr = np.concatenate((inside_v_r, outside_v_r), axis=1)    

print(radial_vr.shape)


inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-k**2*cT_e**2)*(c_e**2+vA_e**2))
radial_xi_z = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 


#inside_xi_phi2 = ((m*inside_P_solution[::-1]/ix[::-1])-(2.*rho_i0*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - k**2*vA_i0**2))

inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - k**2*vA_e**2))
radial_xi_phi = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    


######  v_phi   #######

inside_v_phi = np.zeros((len(time), len(ix)))
outside_v_phi = np.zeros((len(time), len(lx)))

def dv_phi(r):
  return sym.diff(v_iphi(r)/r)

dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")

for t in range(len(time)):
   outside_v_phi[t,:] = -w*outside_xi_phi*np.cos(w*time[t])
   inside_v_phi[t,:] = -(shift_freq_np(ix[::-1])*inside_xi_phi*np.cos(w*time[t])) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1]*np.cos(w*time[t]))

radial_v_phi = np.concatenate((inside_v_phi, outside_v_phi), axis=1)    

######  v_z   #######

inside_v_z = np.zeros((len(time), len(ix)))
outside_v_z = np.zeros((len(time), len(lx)))

def dv_z(r):
  return sym.diff(v_z(r)/r)

dv_z_np=sym.lambdify(rr,dv_z(rr),"numpy")

for t in range(len(time)):
   outside_v_z[t,:] = -w*outside_xi_z*np.cos(w*time[t])
   inside_v_z[t,:] = -(shift_freq_np(ix[::-1])*inside_xi_z*np.cos(w*time[t])) - (dv_z_np(ix[::-1])*inside_xi_solution[::-1]*np.cos(w*time[t]))
# xi_z is multiplied by i such that first term here is multiplied by cos(w*t) and now positive

radial_v_z = np.concatenate((inside_v_z, outside_v_z), axis=1)    

######################


fig, (ax, ax2) = plt.subplots(2,1, sharex=False)
ax.set_title('kink') 
ax.axvline(x=B, color='r', linestyle='--')
ax.set_ylabel("$\hat{\u03be}_{phi}$", fontsize=18, rotation=0, labelpad=15)
#ax.set_ylim(0.,1.2)
ax.set_xlim(0.001, 2.)

ax.plot(spatial, radial_xi_phi, 'k')
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
########################################

wavenum = k

z = np.linspace(0.01, 6., 19.)   #350  #19


THETA = np.linspace(0, 2.*np.pi, 50) #100

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


boundary_point_r = np.zeros(((len(spatial), len(THETA), len(z))))

bound_index = np.where(radii[:,0,0] == 1.)

#exit()


for k in range(len(z)):
  for j in range(len(THETA)):
    for i in range(len(spatial)):
      for t in range(len(time)):
        xi_r[t,i,j,k] = radial_displacement[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.cos(w*time[t])  
        xi_phi[t,i,j,k] = radial_xi_phi[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.cos(w*time[t])       
        #xi_z[t,i,j,k] = radial_xi_z[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.sin(w*time[t])
        PT[t,i,j,k] = radial_PT[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.cos(w*time[t])
        v_r[t,i,j,k] = radial_vr[t,i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k]) 
        v_phi[t,i,j,k] = radial_v_phi[t,i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])
        #v_z[t,i,j,k] = radial_v_z[t,i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])   

       
        xi_x[t,i,j,k] = (xi_r[t,i,j,k]*np.cos(thetas[i,j,k]) - xi_phi[t,i,j,k]*np.sin(thetas[i,j,k]))
        xi_y[t,i,j,k] = (xi_r[t,i,j,k]*np.sin(thetas[i,j,k]) + xi_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        v_x[t,i,j,k] = (v_r[t,i,j,k]*np.cos(thetas[i,j,k]) - v_phi[t,i,j,k]*np.sin(thetas[i,j,k]))
        v_y[t,i,j,k] = (v_r[t,i,j,k]*np.sin(thetas[i,j,k]) + v_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        P_x[t,i,j,k] = PT[t,i,j,k]*np.cos(thetas[i,j,k])
        P_y[t,i,j,k] = PT[t,i,j,k]*np.sin(thetas[i,j,k])
               
               

normalised_v_x = v_x/np.amax(v_x)
normalised_v_y = v_y/np.amax(v_y)

print(np.amax(v_x[:,:,0]))
print(np.amax(v_y[:,:,0]))

#exit()
           
boundary_point_r = radii[bound_index] 
boundary_point_z = Z[bound_index]        

bound_element = bound_index[0][0] 

boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[0,bound_element,:,:]*step*10.
boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[0,bound_element,:,:]*step*10.  # for small k
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

print(x.shape)
print(y.shape)
print(z.shape)



varlist = ['xi_r', 'xi_phi', 'P_T', 'v_r', 'v_phi', 'xi_x', 'xi_y', 'v_x', 'v_y', 'P_x', 'P_y', 'boundary_x', 'boundary_y', 'boundary_z']

#makeDumpVTK(x,y,z,[ xi_r, xi_phi, PT, v_r, v_phi, xi_x, xi_y, v_x, v_y, P_x, P_y],varlist,'test_sausage')


#for t in range(len(time)):
#    makeDumpVTK(x,y,z,[ xi_r*np.cos(-w*time[t]), xi_phi*np.sin(-w*time[t]), PT*np.cos(-w*time[t]), v_r*np.sin(-w*time[t]), v_phi*np.sin(-w*time[t]), xi_x*np.cos(-w*time[t]), xi_y*np.cos(-w*time[t]), v_x*np.sin(-w*time[t]), v_y*np.sin(-w*time[t]), P_x*np.cos(-w*time[t]), P_y*np.cos(-w*time[t]), boundary_point_x*np.cos(-w*time[t]), boundary_point_y*np.cos(-w*time[t]), boundary_point_z*np.cos(-w*time[t])],varlist,'test_sausage_0'+ str(t))

#exit()


#varlist_polar = ['xi_r', 'xi_phi', 'P_T', 'v_r', 'v_phi']
#makeDumpVTK(radii, thetas, Z,[ xi_r, xi_phi, PT, v_r, v_phi],varlist_polar,'test_sausage_polar')

#exit()


##############################

##############################
fig = plt.figure()
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax.set_title('displacement')
ax2.set_title('pressure')
ax.axvline(x=B, color='r', linestyle='--')
ax.axvline(x=-B, color='r', linestyle='--')
ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0., 2.5)
ax2.axvline(x=B, color='r', linestyle='--')
ax2.axvline(x=-B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0., 2.5)
#ax.set_ylim([-1.5,1.5])    #-2, 2 for slow
#ax2.set_ylim([-7.,7.])   # -2, 2 for slow   -7, 7 for fast sausage
ax3 = plt.subplot(223, polar=True)
ax4 = plt.subplot(224, polar=True)
ax3.set_ylim([0,2.5])
ax4.set_ylim([0,2.5]) 
ax3.set_xticks([])
ax4.set_xticks([])
#####
xi_plot, = ax.plot(spatial, xi_r[0,:,0,0], 'k')
PT_plot, = ax2.plot(spatial, PT[0,:,0,0], 'k')  
xi_plot_im = ax3.contourf(thetas[:,:,0], radii[:,:,0], xi_r[0,:,:,0], extend='both', cmap='bwr' ) 
#ax3.scatter(boundary_point_x, boundary_point_y, boundary_point_z) #, z[bound_index]) #, boundary_point_z)

PT_plot_im = ax4.contourf(thetas[:,:,0], radii[:,:,0], PT[0,:,:,0], extend='both', cmap='bwr') 
fig.colorbar(xi_plot_im, ax=ax3)#, cmap='bwr', boundaries=np.linspace(xi_min,xi_max,5), norm=xi_norm)
fig.colorbar(PT_plot_im, ax=ax4)#, cmap='bwr', boundaries=np.linspace(PT_min,PT_max,5), norm=PT_norm)

#plt.show()
#exit()


##############################
fig = plt.figure()
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax.set_title('$\hat{v}_r$')
ax2.set_title('$\hat{v}_{phi}$')
ax.axvline(x=B, color='r', linestyle='--')
ax.axvline(x=-B, color='r', linestyle='--')
ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$\hat{v}_r$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0., 2.5)
ax2.axvline(x=B, color='r', linestyle='--')
ax2.axvline(x=-B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{v}_{phi}$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0., 2.5)
#ax.set_ylim([-1.5,1.5])    #-2, 2 for slow
#ax2.set_ylim([-7.,7.])   # -2, 2 for slow   -7, 7 for fast sausage
ax3 = plt.subplot(223, polar=True)
ax4 = plt.subplot(224, polar=True)
ax3.set_ylim([0,2.5])
ax4.set_ylim([0,2.5]) 

ax3.set_xticks([])
ax4.set_xticks([])

#circle_rad =1.
#a = circle_rad*np.cos(THETA) #circle x
#b = circle_rad*np.sin(THETA) #circle y

#ax3.plot(a,b, 'k--')
#ax4.plot(a,b, 'k--')

#ax3.polar(THETA,1, 'k--')
#ax4.polar(THETA,1, 'k--')

ax3.set_title('v_r')
ax4.set_title('v_{phi}')

#####
xi_plot, = ax.plot(spatial, v_r[0,:,0,0], 'k')
PT_plot, = ax2.plot(spatial, v_phi[0,:,0,0], 'k')  
xi_plot_im = ax3.contourf(thetas[:,:,0], radii[:,:,0], v_r[0,:,:,0], extend='both', cmap='bwr' ) 
PT_plot_im = ax4.contourf(thetas[:,:,0], radii[:,:,0], v_phi[0,:,:,0], extend='both', cmap='bwr') 
fig.colorbar(xi_plot_im, ax=ax3)#, cmap='bwr', boundaries=np.linspace(xi_min,xi_max,5), norm=xi_norm)
fig.colorbar(PT_plot_im, ax=ax4)#, cmap='bwr', boundaries=np.linspace(PT_min,PT_max,5), norm=PT_norm)


#plt.show()
#exit()
#plt.savefig("Uniform_cylinder_kink_velocity_polar_components.png")

################################

fig = plt.figure()
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax.set_title('v_x')
ax2.set_title('v_y')

ax.axvline(x=B, color='r', linestyle='--')
ax.axvline(x=-B, color='r', linestyle='--')
ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$v_x$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(-1.5, 1.5)
ax2.axvline(x=B, color='r', linestyle='--')
ax2.axvline(x=-B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$v_y$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(-1.5, 1.5)

ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
ax3.set_ylim([-1.5,1.5])
ax4.set_ylim([-1.5,1.5]) 
ax3.set_xlim([-1.5,1.5])    
ax4.set_xlim([-1.5,1.5]) 
ax3.set_title('v_x')
ax4.set_title('v_y')

circle_rad =1.
a = circle_rad*np.cos(THETA) #circle x
b = circle_rad*np.sin(THETA) #circle y

ax3.plot(a,b, 'k--')
ax4.plot(a,b, 'k--')

#####
xi_plot, = ax.plot(x[:,0,0], v_x[0,:,0,0], 'k')
PT_plot, = ax2.plot(x[:,0,0], v_y[0,:,0,0], 'k')  
xi_plot_im = ax3.contourf(x[:,:,0], y[:,:,0], v_x[0,:,:,0], np.linspace(-1.,1.,25), extend='both', cmap='bwr') 
PT_plot_im = ax4.contourf(x[:,:,0], y[:,:,0], v_y[0,:,:,0], np.linspace(-1.,1.,25), extend='both', cmap='bwr') #, np.linspace(-7.,7.,25)
fig.colorbar(xi_plot_im, ax=ax3)#, cmap='bwr', boundaries=np.linspace(xi_min,xi_max,5), norm=xi_norm)
fig.colorbar(PT_plot_im, ax=ax4)#, cmap='bwr', boundaries=np.linspace(PT_min,PT_max,5), norm=PT_norm)

#plt.savefig("Uniform_cylinder_kink_velocity_contour.png")

#plt.show()
#exit()


###############################



fig = plt.figure()
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax.set_title('displacement')
ax2.set_title('pressure')
ax.axvline(x=B, color='r', linestyle='--')
ax.axvline(x=-B, color='r', linestyle='--')
ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0., 2.5)
ax2.axvline(x=B, color='r', linestyle='--')
ax2.axvline(x=-B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0., 2.5)
#ax.set_ylim([-1.5,1.5])    #-2, 2 for slow
#ax2.set_ylim([-7.,7.])   # -2, 2 for slow   -7, 7 for fast sausage
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
ax3.set_ylim([-2.5,2.5])
ax4.set_ylim([-2.5,2.5]) 
ax3.set_xlim([-2.5,2.5])
ax4.set_xlim([-2.5,2.5]) 
ax3.set_title('xi_x')
ax4.set_title('P_y')
#####
xi_plot, = ax.plot(x[:,0,0], xi_x[0,:,0,0], 'k')
PT_plot, = ax2.plot(x[:,0,0], P_x[0,:,0,0], 'k')  
xi_plot_im = ax3.contourf(x[:,:,0], y[:,:,0], xi_x[0,:,:,0], extend='both', cmap='bwr') #, np.linspace(-1.5,1.5,25)
PT_plot_im = ax4.contourf(x[:,:,0], y[:,:,0], P_x[0,:,:,0], extend='both', cmap='bwr') #, np.linspace(-7.,7.,25)
fig.colorbar(xi_plot_im, ax=ax3)#, cmap='bwr', boundaries=np.linspace(xi_min,xi_max,5), norm=xi_norm)
fig.colorbar(PT_plot_im, ax=ax4)#, cmap='bwr', boundaries=np.linspace(PT_min,PT_max,5), norm=PT_norm)



###############################################

fig=plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 3)
ax3 = fig.add_subplot(2, 3, 6)

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)

ax2.set_title('z = 0')
ax3.set_title('z = 3.05')

box = ax.get_position()
ax.set_position([box.x0-0.15, box.y0, box.width*1.8, box.height])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width*1.2, box2.height])

box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width*1.2, box3.height])

ax.set_xlabel('$x$', fontsize=18)
#ax2.set_xlabel('$x$', fontsize=18)
ax3.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax2.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
ax3.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
ax.set_zlabel('$z$', fontsize=18)


circle_rad =1.
a = circle_rad*np.cos(THETA) #circle x
b = circle_rad*np.sin(THETA) #circle y

ax2.plot(a,b, 'r--')
ax3.plot(a,b, 'r--')

#ax2.Circle((0., 0.), 1., fill=False)
#ax3.Circle((0., 0.), 1., fill=False)

ax.scatter(boundary_point_x, boundary_point_y, boundary_point_z) #, z[bound_index]) #, boundary_point_z)
ax.plot_surface(boundary_point_x, boundary_point_y, boundary_point_z, color='b', alpha=0.5)

#ax2.scatter(boundary_point_x[:,0], boundary_point_y[:,0])
ax2.plot(boundary_point_x[:,0], boundary_point_y[:,0])
#ax.quiver(x[:,:,0], y[:,:,0], v_x[:,:,0]/np.amax(v_x[:,:,0]), v_y[:,:,0]/np.amax(v_y[:,:,0]), PT[:,:,0]/np.amax(PT[:,:,0]), pivot='mid')
ax2.quiver(x[::18,::2,0], y[::18,::2,0], v_x[0,::18,::2,0], v_y[0,::18,::2,0], PT[0,::18,::2,0], pivot='mid', cmap='bwr', scale_units='inches', scale=0.5)
#ax2.contourf(x[:,:,0], y[:,:,0], PT[:,:,0], pivot='mid', cmap='bwr')

#ax3.scatter(boundary_point_x[:,3], boundary_point_y[:,3])
ax3.plot(boundary_point_x[:,3], boundary_point_y[:,3])
ax3.quiver(x[::35,::2,3], y[::35,::2,3], v_x[0,::35,::2,3], v_y[0,::35,::2,3], PT[0,::35,::2,3], pivot='mid', cmap='bwr', scale_units='inches', scale=0.25)
#ax3.contourf(x[:,:,3], y[:,:,3], PT[:,:,3], pivot='mid', cmap='bwr')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

#ax2.plot(x[::15,::3,0], y[::15,::3,0], 'g.', markersize=2)
ax3.plot(x[::18,::2,3], y[::18,::2,3], 'g.', markersize=1)


#fig, ax = plt.subplots(1,1)
#ax.set_xlim(-2.5, 2.5)
#ax.set_ylim(-2.5, 2.5)
#ax.scatter(boundary_point_x[:,0], boundary_point_y[:,0])
##ax.quiver(x[:,:,0], y[:,:,0], v_x[:,:,0], v_y[:,:,0], PT[:,:,0], pivot='mid', cmap='bwr')
#ax.contourf(x[:,:,0], y[:,:,0], PT[:,:,0], pivot='mid', cmap='bwr')

#plt.show()
#exit()

#########################

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, bitrate=4000)   #9fps if skipping 5 in time step   15 good


#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 3)
ax3 = fig.add_subplot(2, 3, 6)


box = ax.get_position()
ax.set_position([box.x0-0.15, box.y0, box.width*1.8, box.height])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width*1.2, box2.height])

box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width*1.2, box3.height])

def animate(i):  
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax.set_title('$W = 0.9  - kink - k=1.5$')
    ax2.set_title('z = 0')
    ax3.set_title('z = 3.05')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$', fontsize=18)
    ax3.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    ax2.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
    ax3.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
    ax.set_zlabel('$z$', fontsize=18)
    ax2.plot(a,b)
    ax3.plot(a,b)
  
    boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[i,bound_element,:,:]*step*10
    boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[i,bound_element,:,:]*step*10
   
    boundary_point_x = boundary_point_x[0,:,:]
    boundary_point_y = boundary_point_y[0,:,:]
    #boundary_point_z = boundary_point_z[0,:,:]
    
    ax2.plot(x[::35,::2,0], y[::35,::2,0], 'g.', markersize=0.5)
    ax3.plot(x[::35,::2,9], y[::35,::2,9], 'g.', markersize=0.5)

    ax.scatter(boundary_point_x, boundary_point_y, boundary_point_z) 
    ax.plot_surface(boundary_point_x, boundary_point_y, boundary_point_z, color='b', alpha=0.5) #, linewidth=0)
    
    ax2.contourf(x[:,:,0], y[:,:,0], PT[i,:,:,0], extend='both', cmap='bwr', alpha=0.2) 
    ax2.scatter(boundary_point_x[:,0], boundary_point_y[:,0])
    ax2.quiver(x[::35,::2,0], y[::35,::2,0], v_x[i,::35,::2,0], v_y[i,::35,::2,0], PT[i,::35,::2,0], pivot='mid', cmap='bwr', scale_units='inches', scale=0.25)  

    ax3.contourf(x[:,:,9], y[:,:,9], PT[i,:,:,9], extend='both', cmap='bwr', alpha=0.2) 
    ax3.scatter(boundary_point_x[:,3], boundary_point_y[:,3])
    ax3.quiver(x[::35,::2,9], y[::35,::2,9], v_x[i,::35,::2,9], v_y[i,::35,::2,9], PT[i,::35,::2,9], pivot='mid', cmap='bwr', scale_units='inches', scale=0.25)
 
    

anim = FuncAnimation(fig, animate, interval=100, frames=len(time)-1).save('Gaussian_flow_1e5_slow_kink_k2.mp4', writer=writer)

exit()
