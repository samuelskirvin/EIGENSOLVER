# Import the required modules
import numpy as np
import scipy as sc
import matplotlib; matplotlib.use('agg') ##comment out to show figs
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import special
import sympy as sym
import math
from math import log10, floor
from scipy.optimize import fsolve
#import cmath
from matplotlib import animation
from scipy import interpolate
import matplotlib.gridspec as gridspec
import time
import multiprocessing 
import itertools
import pickle

###   Setup to suppress unwanted outputs and reduce size of output file
import os
import sys
import contextlib
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
###

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd
    
@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

sys.setrecursionlimit(25000)
 
    
    
########################################


def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z
        
############################################        
            
    ############################################################
##############      BEGIN SYMMETRIC CYLINDER DISPERSION DIAGRAM       ###############

# INITIALISE VALUES FOR SPECIFIC REGIME (E.G. CORONA / PHOTOSPHERE)

c_i0 = 1.
vA_e = 0.5*c_i0     #5.*c_i #-coronal        #0.5*c_i -photospheric
vA_i0 = 2.*c_i0     #2*c_i #-coronal        #2.*c_i  -photospheric
c_e = 1.5*c_i0      #0.5*c_i #- coronal          #1.5*c_i  -photospheric

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

ix = np.linspace(1., 0.01, 2e3)  # inside slab x values


r0=0.  #mean
dr=1e5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols



def B_i(r):  #constant rho
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))


B_twist = 0.1
def B_iphi(r):  #constant rho
    return 0.  #B_twist*r   #0.


v_twist = 0.15
power = 1.25   ## tis sets the linearity of the rotational flow  i.e. p=1 = linear, p=2 = quadratic, p=0.5 = square rooted etc.

def v_iphi(r):  #constant rho
    return v_twist*(r**power)   #0.


print('v phi   =', v_twist)

def P_i(r):
    #return rho_i(r)*v_iphi(r)**2/2. + P_0    #  if v_phi is linear in r
    return rho_i(r)*v_twist**2.*(r**(2.*power)/(2.*power)) + P_0    #  if v_phi is nonlinear in r


def c_i(r):
    return sym.sqrt((P_i(r)*gamma)/rho_i(r))
      
###################################

def PT_i(r):   # USE THIS ONE
    return (sym.diff((P_i(r) + (B_i(r)**2 + B_iphi(r)**2)/2.), r) + B_iphi(r)**2/r) - (rho_i(r)*v_iphi(r)**2/r)

###################################

################################################

def rho_i(r):                  
    return rho_i0

#############################################

def T_i(r):     #use this for constant P
    return (P_i(r)/rho_i(r))

##########################################

def vA_i(r):                  # consatnt temp
    return (B_i(r)+B_iphi(r))/(sym.sqrt(rho_i(r)))

##########################################

###################################################
def cT_i(r):                 # Define the internal tube speed
    return sym.sqrt(((c_i(r))**2 * (vA_i(r))**2) / ((c_i(r))**2 + (vA_i(r))**2))


#####################################################

#####################################################

#speeds = [5., 4.5, 4., 3.5, 3., 2.75, 2.5]   #3 works 

#speeds = [c_i0, cT_i0, 0.95, 0.925]  #added -c_e and cT_i0
#speeds = [vA_i0, vA_e, c_kink, 0.8]  #added -c_e and cT_i0

#speeds = [-c_i0, -cT_i0, -0.95, -0.925]  #added -c_e and cT_i0
speeds = [1., 0.98, 0.96, 0.94, 0.92, 0.9, 0.88]  #-0.94 ok


print('speeds  =', speeds)

speeds.sort()
print('sorted speeds  =', speeds)

print(len(speeds))
speed_diff = [0]

for i in range(len(speeds)-1):
    speed_diff.append(speeds[i+1] - speeds[i])


print('speed diff    =', speed_diff)
print(len(speed_diff))
#####################################################

rho_i_np=sym.lambdify(rr,rho_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
cT_i_np=sym.lambdify(rr,cT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
c_i_np=sym.lambdify(rr,c_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
vA_i_np=sym.lambdify(rr,vA_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
P_i_np=sym.lambdify(rr,P_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
T_i_np=sym.lambdify(rr,T_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
B_i_np=sym.lambdify(rr,B_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
PT_i_np=sym.lambdify(rr,PT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy
B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy

#####   SPEEDS AT BOUNDARY    ###

rho_bound = rho_i_np(1)
c_bound = c_i_np(1)
vA_bound = vA_i_np(1)
cT_bound = np.sqrt(c_bound**2 * vA_bound**2 / (c_bound**2 + vA_bound**2))

P_bound = P_i_np(1)
print(P_bound)
internal_pressure = P_i_np(ix)


#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################

##
##plt.figure()
##plt.title("width = 1.5")
###plt.xlabel("x")
###plt.ylabel("$\u03C1_{i}$",fontsize=25)
##ax = plt.subplot(331)
##ax.title.set_text("Density")
###ax.plot(ix,rho_i_np(ix));
##ax.annotate( '$\u03C1_{e}$', xy=(1, rho_e),fontsize=25)
##ax.annotate( '$\u03C1_{i}$', xy=(1, rho_i0),fontsize=25)
##ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot')
##ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot')
##ax.axhline(y=rho_i_np(ix), color='b', label='$\u03C1_{i}$', linestyle='solid')
##ax.set_ylim(0., 1.1)
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$c_{i}$")
##ax2 = plt.subplot(332)
##ax2.title.set_text("$c_{i}$")
##ax2.plot(ix,c_i_np(ix));
##ax2.annotate( '$c_e$', xy=(1, c_e),fontsize=25)
##ax2.annotate( '$c_{i0}$', xy=(1, c_i0),fontsize=25)
##ax2.annotate( '$c_{B}$', xy=(1, c_bound), fontsize=20)
##ax2.axhline(y=c_i0, color='k', label='$c_{i0}$', linestyle='dashdot')
##ax2.axhline(y=c_e, color='k', label='$c_{e}$', linestyle='dashdot')
##ax2.fill_between(ix, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
##
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$vA_{i}$")
##ax3 = plt.subplot(333)
##ax3.title.set_text("$v_{Ai}$")
##ax3.plot(ix,vA_i_np(ix));
##ax3.annotate( '$v_{Ae}$', xy=(1, vA_e),fontsize=25)
##ax3.annotate( '$v_{Ai}$', xy=(1, vA_i0),fontsize=25)
##ax3.annotate( '$v_{AB}$', xy=(1, vA_bound), fontsize=20)
##ax3.axhline(y=vA_i0, color='k', label='$v_{Ai}$', linestyle='dashdot')
##ax3.axhline(y=vA_e, color='k', label='$v_{Ae}$', linestyle='dashdot')
##ax3.fill_between(ix, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
##
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$cT_{i}$")
##ax4 = plt.subplot(334)
##ax4.title.set_text("$c_{Ti}$")
##ax4.plot(ix,cT_i_np(ix));
##ax4.annotate( '$c_{Te}$', xy=(1, cT_e),fontsize=25)
##ax4.annotate( '$c_{Ti}$', xy=(1, cT_i0),fontsize=25)
##ax4.annotate( '$c_{TB}$', xy=(1, cT_bound), fontsize=20)
##ax4.axhline(y=cT_i0, color='k', label='$c_{Ti}$', linestyle='dashdot')
##ax4.axhline(y=cT_e, color='k', label='$c_{Te}$', linestyle='dashdot')
##ax4.fill_between(ix, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
##
##
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$P$")
##ax5 = plt.subplot(335)
##ax5.title.set_text("Gas Pressure")
##ax5.plot(ix,P_i_np(ix));
##ax5.annotate( '$P_{0}$', xy=(1, P_0),fontsize=25)
##ax5.annotate( '$P_{e}$', xy=(1, P_e),fontsize=25)
##ax5.axhline(y=P_0, color='k', label='$P_{i}$', linestyle='dashdot')
##ax5.axhline(y=P_e, color='k', label='$P_{e}$', linestyle='dashdot')
###ax5.axhline(y=P_i_np(ix), color='b', label='$P_{e}$', linestyle='solid')
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$T$")
##ax6 = plt.subplot(336)
##ax6.title.set_text("Temperature")
##ax6.plot(ix,T_i_np(ix));
##ax6.annotate( '$T_{0}$', xy=(1, T_0),fontsize=25)
##ax6.annotate( '$T_{e}$', xy=(1, T_e),fontsize=25)
##ax6.axhline(y=T_0, color='k', label='$T_{i}$', linestyle='dashdot')
##ax6.axhline(y=T_e, color='k', label='$T_{e}$', linestyle='dashdot')
###ax6.axhline(y=T_i_np(ix), color='b', linestyle='solid')
###ax6.set_ylim(0., 1.1)
##
###plt.figure()
###plt.xlabel("x")
###plt.ylabel("$B$")
##ax7 = plt.subplot(337)
##ax7.title.set_text("Mag Field")
##ax7.plot(ix,B_i_np(ix));
##ax7.annotate( '$B_{0}$', xy=(1, B_0),fontsize=25)
##ax7.annotate( '$B_{e}$', xy=(1, B_e),fontsize=25)
##ax7.axhline(y=B_0, color='k', label='$B_{i}$', linestyle='dashdot')
##ax7.axhline(y=B_e, color='k', label='$B_{e}$', linestyle='dashdot')
###ax7.axhline(y=B_i_np(ix), color='b', label='$B_{e}$', linestyle='solid')
##
##
###plt.xlabel("x")
###plt.ylabel("$P_T$")
##ax8 = plt.subplot(338)
##ax8.title.set_text("Tot Pressure (uniform)")
##ax8.annotate( '$P_{T0}$', xy=(1, P_tot_0),fontsize=25,color='b')
##ax8.annotate( '$P_{Te}$', xy=(1, P_tot_e),fontsize=25,color='r')
##ax8.axhline(y=P_tot_0, color='b', label='$P_{Ti0}$', linestyle='dashdot')
##ax8.axhline(y=P_tot_e, color='r', label='$P_{Te}$', linestyle='dashdot')
##
###ax8.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
###ax8.axhline(y=B_i_np(ix), color='b', label='$P_{Te}$', linestyle='solid')
##
##ax9 = plt.subplot(339)
##ax9.title.set_text("Tot Pressure  (Modified)")
##ax9.annotate( '$P_{T0},  P_{Te}$', xy=(1, P_tot_0),fontsize=25,color='b')
###ax9.plot(ix,PT_i_np(ix));
##
##ax9.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
##
##plt.suptitle("W = 1e5", fontsize=14)
##plt.tight_layout()

#plt.show()
#exit()


################     BEGIN NEW METHOD FOR DISPERSION       ###########################
def round_to_1_sf(x):
   return round(x, -int(floor(log10(abs(x)))))


## INITIALLY HAVE ONE PAIR OF W AND K AND WORK THROUGH WITH JUST THAT PAIR - IF NO MATCH MOVE ONTO NEXT PAIR

### At the moment test for kink by requiring pressure to be zero at zero

d = 1.
a = 1.

#lx = np.linspace(-25., -1., 500)  #left hand side of slab x values   -25
#ix = np.linspace(0, -1, 1000)  # inside slab x values

#ix = np.linspace(-1., -0.01, 500.)  # inside slab x values

#left_P_solution = np.zeros(len(lx))
#inside_P_solution = np.zeros(len(ix))


xi_tol = 4.5

sol_omegas_kink = []
sol_ks_kink = []

P_diff_check = [0]
P_diff_loop_check = [0]


test_sols_inside = []
test_sols_outside = []
sol_omegas = []
sol_ks = []

xi_diff_check = [0]
xi_diff_loop_check = [0]


all_ws = []
all_ks = []

loop_ws = []

###########    DEFINE A FUNCTION WHICH LOCATES SOLUTION    #####################

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
      ##########       SAUSAGE         ##########

sol_omegas = []
sol_ks = []
P_diff_check_sausage = [0]
P_diff_loop_check_sausage = [0]
loop_ws_sausage = []

all_ws_sausage = []
all_ks_sausage = []

loop_sign_check_sausage = [0]
sign_check_sausage = [0]

p_tol = 4.5


def sausage(wavenumber, sausage_ws, sausage_ks, freq):
  m = 0.
  def locate_sausage(omega, wavenum,itt_num):
      itt_num = itt_num
      m = 0.
      
      lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
   
                
      for k in range(len(omega)):
      
         if itt_num > 250:
            break
            
         m_e = ((((wavenum**2*vA_e**2)-omega[k]**2)*((wavenum**2*c_e**2)-omega[k]**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-omega[k]**2)))
            
                  ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
         
         def shift_freq(r):
           return (omega[k] - (m*v_iphi(r)/r) - wavenum*v_z)
           
         def alfven_freq(r):
           return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))
         
         def cusp_freq(r):
           return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))
         
         def D(r):  
           return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))
         
         D_np=sym.lambdify(rr,D(rr),"numpy")
         
         def Q(r):
           return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_iphi(r)**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*v_iphi(r)*((m*B_iphi(r)/r)+(wavenum*B_i(r)))/r))
         
         def T(r):
           return ((((m*B_iphi(r)/r)+(wavenum*B_i(r)))*B_iphi(r)) + rho_i(r)*v_iphi(r)*shift_freq(r))
         
         def C1(r):
           return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))
         
         C1_np=sym.lambdify(rr,C1(rr),"numpy")
            
         def C2(r):   
           return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))

         C2_np=sym.lambdify(rr,C2(rr),"numpy")

###
         def C2_e(r):   
           return (omega[k]**4 - ((c_e**2 + vA_e**2)*(m**2/r**2 + wavenum**2)*(omega[k]**2 - wavenum**2*cT_e**2)))
            
         def C3_diff(r):
           return ((B_iphi(r)/r)**2 - (rho_i(r)*(v_iphi(r)/r)**2))
         
         
         def C3(r):
           return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
           
         C3_np=sym.lambdify(rr,C3(rr),"numpy")

         ##########################################
                                       
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
                
         if m_e < 0:
             pass
         
         else:
             loop_ws.append(omega[k])
            
             def dP_dr_e(P_e, r_e):
                    return [P_e[1], (-P_e[1]/r_e + (m_e+((0.**2)/(r_e**2)))*P_e[0])]
               
             P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
             Ls = odeintz(dP_dr_e, P0, lx)
             left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side
             
             left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side
           
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_xi_solution = left_xi_solution/np.amax(abs(left_xi_solution))
             left_bound_P = left_P_solution[-1] 
                                
             
             def dP_dr_i(P_i, r_i):
                    return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]                     
                        
             def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
                   u1 = U[:,1] #- left_bound_vx    #was plus (assuming pressure is symmetric for sausage)
                   return u1[-1] 
                   
             dPi, = fsolve(objective_dPi, 0.001) 
     
             # now solve with optimal dvx
             
             Is = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
             inside_P_solution = Is[:,0]
            
             #inside_xi_solution = xi_i_const*Is[:,1] #+ Is[:,0]/ix)   # Pressure perturbation solution for left hand side
             inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side
 
             normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
             normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))
             
             xi_diff_loop_check.append((left_xi_solution[-1]-inside_xi_solution[0]))
             loop_sign_check_sausage.append(xi_diff_loop_check[-1]*xi_diff_loop_check[-2]) 

             if (abs(left_xi_solution[-1] - inside_xi_solution[0])*100/max(abs(left_xi_solution[-1]), abs(inside_xi_solution[0]))) < xi_tol:
                   sol_omegas.append(omega[k])
                   sol_ks.append(wavenum)
                   loop_ws[:] = []
                   loop_sign_check_sausage[:] = [0]
                   break
            
            
             elif loop_sign_check_sausage[-1] < 0. and len(loop_ws)>2:  
                      omega = np.linspace(loop_ws[-2], loop_ws[-1], 3)
                      wavenum = wavenum
             #        #now repeat exact same process but in foccused omega range
                      itt_num = itt_num+1
                      loop_ws[:] = []
                      locate_sausage(omega, wavenum, itt_num)              
                   
  
      
  ##############################################################################
  
            
  with stdout_redirected():
      for j in range(len(freq)):
          m = 0.
          lx = np.linspace(3.*2.*np.pi/wavenumber, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain   #1.75 before
            
          m_e = ((((wavenumber**2*vA_e**2)-freq[j]**2)*((wavenumber**2*c_e**2)-freq[j]**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e**2)-freq[j]**2)))
             
          xi_e_const = -1/(rho_e*((wavenumber**2*vA_e**2)-freq[j]**2))

                   ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
          
          def shift_freq(r):
            return (freq[j] - (m*v_iphi(r)/r) - wavenumber*v_z)
            
          def alfven_freq(r):
            return ((m*B_iphi(r)/r)+(wavenumber*B_i(r))/(sym.sqrt(rho_i(r))))
          
          def cusp_freq(r):
            return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))
          
          def D(r):  
            return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))
          
          D_np=sym.lambdify(rr,D(rr),"numpy")
          
          def Q(r):
            return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_iphi(r)**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*v_iphi(r)*((m*B_iphi(r)/r)+(wavenumber*B_i(r)))/r))
          
          def T(r):
            return ((((m*B_iphi(r)/r)+(wavenumber*B_i(r)))*B_iphi(r)) + rho_i(r)*v_iphi(r)*shift_freq(r))
          
          def C1(r):
            return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))
          
          C1_np=sym.lambdify(rr,C1(rr),"numpy")
             
          def C2(r):   
            return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenumber**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))

          C2_np=sym.lambdify(rr,C2(rr),"numpy")  
             
          def C3_diff(r):
            return ((B_iphi(r)/r)**2 - (rho_i(r)*(v_iphi(r)/r)**2))           
          
          def C3(r):
            return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
            
          C3_np=sym.lambdify(rr,C3(rr),"numpy")


          ##########################################
                                       
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
                 
          if m_e < 0:
              pass
          
          else:
              all_ws.append(freq[j])
             
              def dP_dr_e(P_e, r_e):
                     return [P_e[1], (-P_e[1]/r_e + (m_e+((0.**2)/(r_e**2)))*P_e[0])]
                
              P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
              Ls = odeintz(dP_dr_e, P0, lx)
              left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side
                  
              left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side
            
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_xi_solution = left_xi_solution/np.amax(abs(left_xi_solution))
              left_bound_P = left_P_solution[-1] 
                                  

              def dP_dr_i(P_i, r_i):
                    return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
                    
                        
              def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
                    u1 = U[:,1] #for sausage mode we solve P' = 0 at centre of cylinder
                    return u1[-1] 
                    
              dPi, = fsolve(objective_dPi, 0.001) 
      
              # now solve with optimal dvx
              
              Is = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
              inside_P_solution = Is[:,0]
             
              #inside_xi_solution = xi_i_const*Is[:,1] #+ Is[:,0]/ix)   # Pressure perturbation solution for left hand side
              inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

              normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
              normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))
              
              xi_diff_check.append((left_xi_solution[-1]-inside_xi_solution[0]))
              sign_check_sausage.append(xi_diff_check[-1]*xi_diff_check[-2])  
              
              all_ks.append(wavenumber)
              all_ws.append(freq[j])
              
              if (abs(left_xi_solution[-1] - inside_xi_solution[0])*100/max(abs(left_xi_solution[-1]), abs(inside_xi_solution[0]))) < xi_tol:
                    sol_omegas.append(freq[j])
                    sol_ks.append(wavenumber)
                    all_ws[:] = []
             
             
              elif sign_check_sausage[-1] < 0:# and abs(inside_P_solution[-1]) < 1e-5:   #xi_diff_check[-1] > 0. 
                   if len(all_ws)>2:         
                       omega = np.linspace(all_ws[-2], all_ws[-1], 3)
                       wavenum = all_ks[-2]
                       #now repeat exact same process but in foccused omega range
                       itt_num = 0
                       all_ws[:] = []
                       locate_sausage(omega, wavenum, itt_num)              

  sausage_ks.put(sol_ks)
  sausage_ws.put(sol_omegas) 



wavenumber = np.linspace(0.25,4.,100.)    #0.5 works
#wavenumber = np.linspace(0.001,1.,40.)    #101


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    
    sausage_ws = multiprocessing.Queue()
    sausage_ks = multiprocessing.Queue()
    
    
    for k in wavenumber:
      for i in range(len(speeds)-1):
     
         test_freq = np.linspace(speeds[i]*k, speeds[i+1]*k, 40.)   #was 40
         
         task = multiprocessing.Process(target=sausage, args=(k, sausage_ws, sausage_ks, test_freq))
         
         processes.append(task)
         task.start()

    for p in processes:
        p.join()
        

    sol_ks1 = [sausage_ks.get() for p in processes]
    sol_omegas1 = [sausage_ws.get() for p in processes]
    
    sol_ks1 = list(itertools.chain(*sol_ks1))   #flatten out the list of lists into one single list
    sol_omegas1 = list(itertools.chain(*sol_omegas1))
    

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)


with open('Cylindrical_photospheric_vtwist015_power125_sausage_slow.pickle', 'wb') as f:     # EXPORT SOLUTIONS TO A PICKLE FILE
    pickle.dump([sol_omegas1, sol_ks1], f)
 

print('ks = ', sol_ks, 'ws  =', sol_omegas)
## If last element of left_solutions is within tolerance to last element of inside solution then plot w-k


plt.figure()
ax = plt.subplot(111)
plt.xlabel("$kr$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')
#ax.plot(test_k,test_W, 'x') #test plot

ax.annotate( '$c_{Ti}$', xy=(Kmax, -cT_i0), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, -cT_e), fontsize=20)
ax.annotate( '$c_{e}$', xy=(Kmax, -c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, -c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)
ax.annotate( '$c_{k}$', xy=(Kmax, -c_kink), fontsize=20)

#ax.set_ylim(0, 5)
#ax.set_ylim(-vA_e-0.1, vA_e+0.1)
ax.set_ylim(cT_i0-0.1, c_i0+0.1)
#ax.set_ylim(1.9, 5.)
#ax.set_ylim(0.8, 1.1)
#ax.set_ylim(-5., -1.8)
#ax.set_ylim(-0.8, -1.1)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.9, box.height])


plt.savefig("Cylindrical_photospheric_vtwist015_power125_sausage_slow.png")


#plt.show()
exit()
