# Import the required modules
import numpy as np
import scipy as sc
import sympy as sym
import matplotlib; matplotlib.use('agg') ##comment out to show figures
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pickle
import itertools
import numpy.polynomial.polynomial as poly
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from scipy.interpolate import griddata
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable


###   Setup to suppress unwanted outputs and reduce size of output file
####  '' The user can ignore this section - this section speeds ''
####  '' up simulation by surpressung text output and warnings ''
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


## see   --- https://link.springer.com/content/pdf/10.1023/B:SOLA.0000006901.22169.59.pdf

c_i0 = 1.
vA_e = 0.5*c_i0     #5.*c_i #-coronal        #0.5*c_i -photospheric
vA_i0 = 2.*c_i0     #2*c_i #-coronal        #2.*c_i  -photospheric
c_e = 1.5*c_i0      #0.5*c_i #- coronal          #1.5*c_i  -photospheric

cT_i0 = np.sqrt(c_i0**2 * vA_i0**2 / (c_i0**2 + vA_i0**2))
cT_e = np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))


gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)


print('rho_e  =', rho_e)


R1 = rho_e/rho_i0
#R1 = 0.8

print('Density ration external/internal    =', R1)


c_kink = np.sqrt(((rho_i0*vA_i0**2)+(rho_e*vA_e**2))/(rho_i0+rho_e))


v_phi = 0.
v_z = 0.

P_0 = c_i0**2*rho_i0/gamma
P_e = c_e**2*rho_e/gamma

print('P_0  =', P_0)
print('P_e  =', P_e)

T_0 = P_0/rho_i0
T_e = P_e/rho_e

B_0 = vA_i0*np.sqrt(rho_i0)
B_e = vA_e*np.sqrt(rho_e)
#B_iphi(r) = 0.
twist = 0.0
A = 0.0
B = 1.
#B_tot_i = np.sqrt(B_0**2+B_iphi(r)**2)

P_tot_0 = P_0 + (B_0**2 + A**2)/2. + A**2/2. 
P_tot_e = P_e + B_e**2/2.

#desired_A = np.sqrt(2*(P_tot_e - P_0)-B_0**2)

#print('desired A   =', desired_A)
#exit()

print('PT_i   =', P_tot_0)
print('PT_e   =', P_tot_e)

Kmax = 4.

ix = np.linspace(1., 0.001, 2e3)  # inside slab x values

print('B_0   =', B_0)

r0=0.  #mean
dr=1e5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols



def B_i(r):  #constant rho
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))


B_twist = 0.
def B_iphi(r):  #constant rho
    return 0.  #B_twist*r   #0.


v_twist = 0.1

power = 1.

def v_iphi(r):  
    return v_twist*(r**power)   #0.

#def P_i(r):   # linear twist
#    return rho_i(r)*v_iphi(r)**2/2. + P_0

#def P_i(r):   # quadratic twist
#    return (rho_i(r)*v_twist**2.*(r**2.)/2.) + P_0


def P_i(r):
    #return rho_i(r)*v_iphi(r)**2/2. + P_0    #  if v_phi is linear in r
    return rho_i(r)*v_twist**2.*(r**(2.*power)/(2.*power)) + P_0    #  if v_phi is nonlinear in r



#def P_i(r):   # general twist
#    return rho_i(r)*(v_twist**2.*(r**(2.*power)))/(2.*power) + P_0

##############
#def P_i(r):   # quadratic twist
#    return rho_i(r)*v_iphi(r)**2/4. + P_0
##############

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


rho_i_np=sym.lambdify(rr,rho_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
cT_i_np=sym.lambdify(rr,cT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
c_i_np=sym.lambdify(rr,c_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
vA_i_np=sym.lambdify(rr,vA_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
P_i_np=sym.lambdify(rr,P_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
T_i_np=sym.lambdify(rr,T_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
B_i_np=sym.lambdify(rr,B_i(rr),"numpy")   #In order to evaluate we need to switch to numpy
PT_i_np=sym.lambdify(rr,PT_i(rr),"numpy")   #In order to evaluate we need to switch to numpy

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


#####################################################



########   READ IN VARIABLES    power = 0.8   #########

#################################################    v = 0.01

with open('Cylindrical_photospheric_vtwist001_power08_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v001_p08_pos_slow, sol_ks_v001_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power08_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v001_p08_pos_fast, sol_ks_v001_p08_pos_fast = pickle.load(f)

sol_omegas_v001_p08 = np.concatenate((sol_omegas_v001_p08_pos_slow, sol_omegas_v001_p08_pos_fast), axis=None)  
sol_ks_v001_p08 = np.concatenate((sol_ks_v001_p08_pos_slow, sol_ks_v001_p08_pos_fast), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v001_p08 = [x for _,x in sorted(zip(sol_ks_v001_p08,sol_omegas_v001_p08))]
sol_ks_v001_p08 = np.sort(sol_ks_v001_p08)

sol_omegas_v001_p08 = np.array(sol_omegas_v001_p08)
sol_ks_v001_p08 = np.array(sol_ks_v001_p08)

#############


with open('Cylindrical_photospheric_vtwist001_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p08_pos_slow, sol_ks_kink_v001_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p08_pos_fast, sol_ks_kink_v001_p08_pos_fast = pickle.load(f)


sol_omegas_kink_v001_p08 = np.concatenate((sol_omegas_kink_v001_p08_pos_slow, sol_omegas_kink_v001_p08_pos_fast), axis=None)  
sol_ks_kink_v001_p08 = np.concatenate((sol_ks_kink_v001_p08_pos_slow, sol_ks_kink_v001_p08_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v001_p08 = [x for _,x in sorted(zip(sol_ks_kink_v001_p08,sol_omegas_kink_v001_p08))]
sol_ks_kink_v001_p08 = np.sort(sol_ks_kink_v001_p08)

sol_omegas_kink_v001_p08 = np.array(sol_omegas_kink_v001_p08)
sol_ks_kink_v001_p08 = np.array(sol_ks_kink_v001_p08)



#################################################    v = 0.05

with open('Cylindrical_photospheric_vtwist005_power08_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v005_p08_pos_slow, sol_ks_v005_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power08_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v005_p08_pos_fast, sol_ks_v005_p08_pos_fast = pickle.load(f)


sol_omegas_v005_p08 = np.concatenate((sol_omegas_v005_p08_pos_slow, sol_omegas_v005_p08_pos_fast), axis=None)  
sol_ks_v005_p08 = np.concatenate((sol_ks_v005_p08_pos_slow, sol_ks_v005_p08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v005_p08 = [x for _,x in sorted(zip(sol_ks_v005_p08,sol_omegas_v005_p08))]
sol_ks_v005_p08 = np.sort(sol_ks_v005_p08)

sol_omegas_v005_p08 = np.array(sol_omegas_v005_p08)
sol_ks_v005_p08 = np.array(sol_ks_v005_p08)

#####

with open('Cylindrical_photospheric_vtwist005_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p08_pos_slow, sol_ks_kink_v005_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p08_pos_fast, sol_ks_kink_v005_p08_pos_fast = pickle.load(f)


sol_omegas_kink_v005_p08 = np.concatenate((sol_omegas_kink_v005_p08_pos_slow, sol_omegas_kink_v005_p08_pos_fast), axis=None)  
sol_ks_kink_v005_p08 = np.concatenate((sol_ks_kink_v005_p08_pos_slow, sol_ks_kink_v005_p08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v005_p08 = [x for _,x in sorted(zip(sol_ks_kink_v005_p08,sol_omegas_kink_v005_p08))]
sol_ks_kink_v005_p08 = np.sort(sol_ks_kink_v005_p08)

sol_omegas_kink_v005_p08 = np.array(sol_omegas_kink_v005_p08)
sol_ks_kink_v005_p08 = np.array(sol_ks_kink_v005_p08)


#################################################    v = 0.1

with open('Cylindrical_photospheric_vtwist01_power08_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v01_p08_pos_slow, sol_ks_v01_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power08_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v01_p08_pos_fast, sol_ks_v01_p08_pos_fast = pickle.load(f)

sol_omegas_v01_p08 = np.concatenate((sol_omegas_v01_p08_pos_fast, sol_omegas_v01_p08_pos_slow), axis=None)  
sol_ks_v01_p08 = np.concatenate((sol_ks_v01_p08_pos_fast, sol_ks_v01_p08_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v01_p08 = [x for _,x in sorted(zip(sol_ks_v01_p08,sol_omegas_v01_p08))]
sol_ks_v01_p08 = np.sort(sol_ks_v01_p08)

sol_omegas_v01_p08 = np.array(sol_omegas_v01_p08)
sol_ks_v01_p08 = np.array(sol_ks_v01_p08)

#############


with open('Cylindrical_photospheric_vtwist01_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p08_pos_slow, sol_ks_kink_v01_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power08_slow_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v01_p08_pos_slow2, sol_ks_kink_v01_p08_pos_slow2 = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p08_pos_fast, sol_ks_kink_v01_p08_pos_fast = pickle.load(f)


sol_omegas_kink_v01_p08 = np.concatenate((sol_omegas_kink_v01_p08_pos_slow, sol_omegas_kink_v01_p08_pos_slow2, sol_omegas_kink_v01_p08_pos_fast), axis=None)  
sol_ks_kink_v01_p08 = np.concatenate((sol_ks_kink_v01_p08_pos_slow, sol_ks_kink_v01_p08_pos_slow2, sol_ks_kink_v01_p08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v01_p08 = [x for _,x in sorted(zip(sol_ks_kink_v01_p08,sol_omegas_kink_v01_p08))]
sol_ks_kink_v01_p08 = np.sort(sol_ks_kink_v01_p08)

sol_omegas_kink_v01_p08 = np.array(sol_omegas_kink_v01_p08)
sol_ks_kink_v01_p08 = np.array(sol_ks_kink_v01_p08)



##################################################    v = 0.15

with open('Cylindrical_photospheric_vtwist015_power08_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v015_p08_pos_slow, sol_ks_v015_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power08_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v015_p08_pos_fast, sol_ks_v015_p08_pos_fast = pickle.load(f)

sol_omegas_v015_p08 = np.concatenate((sol_omegas_v015_p08_pos_fast, sol_omegas_v015_p08_pos_slow), axis=None)  
sol_ks_v015_p08 = np.concatenate((sol_ks_v015_p08_pos_fast, sol_ks_v015_p08_pos_slow), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v015_p08 = [x for _,x in sorted(zip(sol_ks_v015_p08,sol_omegas_v015_p08))]
sol_ks_v015_p08 = np.sort(sol_ks_v015_p08)

sol_omegas_v015_p08 = np.array(sol_omegas_v015_p08)
sol_ks_v015_p08 = np.array(sol_ks_v015_p08)

#############

with open('Cylindrical_photospheric_vtwist015_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p08_pos_slow, sol_ks_kink_v015_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p08_pos_fast, sol_ks_kink_v015_p08_pos_fast = pickle.load(f)


sol_omegas_kink_v015_p08 = np.concatenate((sol_omegas_kink_v015_p08_pos_slow, sol_omegas_kink_v015_p08_pos_fast), axis=None)  
sol_ks_kink_v015_p08 = np.concatenate((sol_ks_kink_v015_p08_pos_slow, sol_ks_kink_v015_p08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v015_p08 = [x for _,x in sorted(zip(sol_ks_kink_v015_p08,sol_omegas_kink_v015_p08))]
sol_ks_kink_v015_p08 = np.sort(sol_ks_kink_v015_p08)

sol_omegas_kink_v015_p08 = np.array(sol_omegas_kink_v015_p08)
sol_ks_kink_v015_p08 = np.array(sol_ks_kink_v015_p08)



##################################################    v = 0.25

with open('Cylindrical_photospheric_vtwist025_power08_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v025_p08_pos_slow, sol_ks_v025_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power08_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v025_p08_pos_fast, sol_ks_v025_p08_pos_fast = pickle.load(f)

sol_omegas_v025_p08 = np.concatenate((sol_omegas_v025_p08_pos_fast, sol_omegas_v025_p08_pos_slow), axis=None)  
sol_ks_v025_p08 = np.concatenate((sol_ks_v025_p08_pos_fast, sol_ks_v025_p08_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v025_p08 = [x for _,x in sorted(zip(sol_ks_v025_p08,sol_omegas_v025_p08))]
sol_ks_v025_p08 = np.sort(sol_ks_v025_p08)

sol_omegas_v025_p08 = np.array(sol_omegas_v025_p08)
sol_ks_v025_p08 = np.array(sol_ks_v025_p08)

#############


with open('Cylindrical_photospheric_vtwist025_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p08_pos_slow, sol_ks_kink_v025_p08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p08_pos_fast, sol_ks_kink_v025_p08_pos_fast = pickle.load(f)


sol_omegas_kink_v025_p08 = np.concatenate((sol_omegas_kink_v025_p08_pos_slow, sol_omegas_kink_v025_p08_pos_fast), axis=None)  
sol_ks_kink_v025_p08 = np.concatenate((sol_ks_kink_v025_p08_pos_slow, sol_ks_kink_v025_p08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v025_p08 = [x for _,x in sorted(zip(sol_ks_kink_v025_p08,sol_omegas_kink_v025_p08))]
sol_ks_kink_v025_p08 = np.sort(sol_ks_kink_v025_p08)

sol_omegas_kink_v025_p08 = np.array(sol_omegas_kink_v025_p08)
sol_ks_kink_v025_p08 = np.array(sol_ks_kink_v025_p08)



####################################################################




########   READ IN VARIABLES    power = 0.9   #########

#################################################

##################################################    v = 0.01

with open('Cylindrical_photospheric_vtwist001_power09_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v001_p09_pos_slow, sol_ks_v001_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power09_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v001_p09_pos_fast, sol_ks_v001_p09_pos_fast = pickle.load(f)

sol_omegas_v001_p09 = np.concatenate((sol_omegas_v001_p09_pos_fast, sol_omegas_v001_p09_pos_slow), axis=None)  
sol_ks_v001_p09 = np.concatenate((sol_ks_v001_p09_pos_fast, sol_ks_v001_p09_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v001_p09 = [x for _,x in sorted(zip(sol_ks_v001_p09,sol_omegas_v001_p09))]
sol_ks_v001_p09 = np.sort(sol_ks_v001_p09)

sol_omegas_v001_p09 = np.array(sol_omegas_v001_p09)
sol_ks_v001_p09 = np.array(sol_ks_v001_p09)

#############


with open('Cylindrical_photospheric_vtwist001_power09_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p09_pos_slow, sol_ks_kink_v001_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power09_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p09_pos_fast, sol_ks_kink_v001_p09_pos_fast = pickle.load(f)


sol_omegas_kink_v001_p09 = np.concatenate((sol_omegas_kink_v001_p09_pos_fast, sol_omegas_kink_v001_p09_pos_slow), axis=None)  
sol_ks_kink_v001_p09 = np.concatenate((sol_ks_kink_v001_p09_pos_fast, sol_ks_kink_v001_p09_pos_slow), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v001_p09 = [x for _,x in sorted(zip(sol_ks_kink_v001_p09,sol_omegas_kink_v001_p09))]
sol_ks_kink_v001_p09 = np.sort(sol_ks_kink_v001_p09)

sol_omegas_kink_v001_p09 = np.array(sol_omegas_kink_v001_p09)
sol_ks_kink_v001_p09 = np.array(sol_ks_kink_v001_p09)



##################################################    v = 0.05

with open('Cylindrical_photospheric_vtwist005_power09_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v005_p09_pos_slow, sol_ks_v005_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power09_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v005_p09_pos_fast, sol_ks_v005_p09_pos_fast = pickle.load(f)

sol_omegas_v005_p09 = np.concatenate((sol_omegas_v005_p09_pos_slow, sol_omegas_v005_p09_pos_fast), axis=None)  
sol_ks_v005_p09 = np.concatenate((sol_ks_v005_p09_pos_slow, sol_ks_v005_p09_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v005_p09 = [x for _,x in sorted(zip(sol_ks_v005_p09,sol_omegas_v005_p09))]
sol_ks_v005_p09 = np.sort(sol_ks_v005_p09)

sol_omegas_v005_p09 = np.array(sol_omegas_v005_p09)
sol_ks_v005_p09 = np.array(sol_ks_v005_p09)

#############


with open('Cylindrical_photospheric_vtwist005_power09_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p09_pos_slow, sol_ks_kink_v005_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power09_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p09_pos_fast, sol_ks_kink_v005_p09_pos_fast = pickle.load(f)


sol_omegas_kink_v005_p09 = np.concatenate((sol_omegas_kink_v005_p09_pos_slow, sol_omegas_kink_v005_p09_pos_fast), axis=None)  
sol_ks_kink_v005_p09 = np.concatenate((sol_ks_kink_v005_p09_pos_slow, sol_ks_kink_v005_p09_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v005_p09 = [x for _,x in sorted(zip(sol_ks_kink_v005_p09,sol_omegas_kink_v005_p09))]
sol_ks_kink_v005_p09 = np.sort(sol_ks_kink_v005_p09)

sol_omegas_kink_v005_p09 = np.array(sol_omegas_kink_v005_p09)
sol_ks_kink_v005_p09 = np.array(sol_ks_kink_v005_p09)



##################################################    v = 0.1

with open('Cylindrical_photospheric_vtwist01_power09_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v01_p09_pos_slow, sol_ks_v01_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power09_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v01_p09_pos_fast, sol_ks_v01_p09_pos_fast = pickle.load(f)

sol_omegas_v01_p09 = np.concatenate((sol_omegas_v01_p09_pos_slow, sol_omegas_v01_p09_pos_fast), axis=None)  
sol_ks_v01_p09 = np.concatenate((sol_ks_v01_p09_pos_slow, sol_ks_v01_p09_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v01_p09 = [x for _,x in sorted(zip(sol_ks_v01_p09,sol_omegas_v01_p09))]
sol_ks_v01_p09 = np.sort(sol_ks_v01_p09)

sol_omegas_v01_p09 = np.array(sol_omegas_v01_p09)
sol_ks_v01_p09 = np.array(sol_ks_v01_p09)

#############


with open('Cylindrical_photospheric_vtwist01_power09_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p09_pos_slow, sol_ks_kink_v01_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power09_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p09_pos_fast, sol_ks_kink_v01_p09_pos_fast = pickle.load(f)


sol_omegas_kink_v01_p09 = np.concatenate((sol_omegas_kink_v01_p09_pos_slow, sol_omegas_kink_v01_p09_pos_fast), axis=None)  
sol_ks_kink_v01_p09 = np.concatenate((sol_ks_kink_v01_p09_pos_slow, sol_ks_kink_v01_p09_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v01_p09 = [x for _,x in sorted(zip(sol_ks_kink_v01_p09,sol_omegas_kink_v01_p09))]
sol_ks_kink_v01_p09 = np.sort(sol_ks_kink_v01_p09)

sol_omegas_kink_v01_p09 = np.array(sol_omegas_kink_v01_p09)
sol_ks_kink_v01_p09 = np.array(sol_ks_kink_v01_p09)




##################################################    v = 0.15

with open('Cylindrical_photospheric_vtwist015_power09_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v015_p09_pos_slow, sol_ks_v015_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power09_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v015_p09_pos_fast, sol_ks_v015_p09_pos_fast = pickle.load(f)

sol_omegas_v015_p09 = np.concatenate((sol_omegas_v015_p09_pos_slow, sol_omegas_v015_p09_pos_fast), axis=None)  
sol_ks_v015_p09 = np.concatenate((sol_ks_v015_p09_pos_slow, sol_ks_v015_p09_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v015_p09 = [x for _,x in sorted(zip(sol_ks_v015_p09,sol_omegas_v015_p09))]
sol_ks_v015_p09 = np.sort(sol_ks_v015_p09)

sol_omegas_v015_p09 = np.array(sol_omegas_v015_p09)
sol_ks_v015_p09 = np.array(sol_ks_v015_p09)

#############


with open('Cylindrical_photospheric_vtwist015_power09_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p09_pos_slow, sol_ks_kink_v015_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power09_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p09_pos_fast, sol_ks_kink_v015_p09_pos_fast = pickle.load(f)


sol_omegas_kink_v015_p09 = np.concatenate((sol_omegas_kink_v015_p09_pos_slow, sol_omegas_kink_v015_p09_pos_fast), axis=None)  
sol_ks_kink_v015_p09 = np.concatenate((sol_ks_kink_v015_p09_pos_slow, sol_ks_kink_v015_p09_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v015_p09 = [x for _,x in sorted(zip(sol_ks_kink_v015_p09,sol_omegas_kink_v015_p09))]
sol_ks_kink_v015_p09 = np.sort(sol_ks_kink_v015_p09)

sol_omegas_kink_v015_p09 = np.array(sol_omegas_kink_v015_p09)
sol_ks_kink_v015_p09 = np.array(sol_ks_kink_v015_p09)



####################################################################


##################################################    v = 0.25

with open('Cylindrical_photospheric_vtwist025_power09_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v025_p09_pos_slow, sol_ks_v025_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power09_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v025_p09_pos_fast, sol_ks_v025_p09_pos_fast = pickle.load(f)

sol_omegas_v025_p09 = np.concatenate((sol_omegas_v025_p09_pos_slow, sol_omegas_v025_p09_pos_fast), axis=None)  
sol_ks_v025_p09 = np.concatenate((sol_ks_v025_p09_pos_slow, sol_ks_v025_p09_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v025_p09 = [x for _,x in sorted(zip(sol_ks_v025_p09,sol_omegas_v025_p09))]
sol_ks_v025_p09 = np.sort(sol_ks_v025_p09)

sol_omegas_v025_p09 = np.array(sol_omegas_v025_p09)
sol_ks_v025_p09 = np.array(sol_ks_v025_p09)

#############

with open('Cylindrical_photospheric_vtwist025_power09_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p09_pos_slow, sol_ks_kink_v025_p09_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power09_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p09_pos_fast, sol_ks_kink_v025_p09_pos_fast = pickle.load(f)


sol_omegas_kink_v025_p09 = np.concatenate((sol_omegas_kink_v025_p09_pos_slow, sol_omegas_kink_v025_p09_pos_fast), axis=None)  
sol_ks_kink_v025_p09 = np.concatenate((sol_ks_kink_v025_p09_pos_slow, sol_ks_kink_v025_p09_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v025_p09 = [x for _,x in sorted(zip(sol_ks_kink_v025_p09,sol_omegas_kink_v025_p09))]
sol_ks_kink_v025_p09 = np.sort(sol_ks_kink_v025_p09)

sol_omegas_kink_v025_p09 = np.array(sol_omegas_kink_v025_p09)
sol_ks_kink_v025_p09 = np.array(sol_ks_kink_v025_p09)



####################################################################



########   READ IN VARIABLES    power = 1   #########


#################################################

##################################################    v = 0.01

with open('Cylindrical_photospheric_vtwist001_pos_slow_sausage.pickle', 'rb') as f:
    sol_omegas_v001_p1_pos_slow, sol_ks_v001_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_pos_fast_sausage.pickle', 'rb') as f:
    sol_omegas_v001_p1_pos_fast, sol_ks_v001_p1_pos_fast = pickle.load(f)

sol_omegas_v001_p1 = np.concatenate((sol_omegas_v001_p1_pos_slow, sol_omegas_v001_p1_pos_fast), axis=None)  
sol_ks_v001_p1 = np.concatenate((sol_ks_v001_p1_pos_slow, sol_ks_v001_p1_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v001_p1 = [x for _,x in sorted(zip(sol_ks_v001_p1,sol_omegas_v001_p1))]
sol_ks_v001_p1 = np.sort(sol_ks_v001_p1)

sol_omegas_v001_p1 = np.array(sol_omegas_v001_p1)
sol_ks_v001_p1 = np.array(sol_ks_v001_p1)


######


with open('testing_cylinder_photospheric_vtwist001_pos_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p1_pos_slow, sol_ks_kink_v001_p1_pos_slow = pickle.load(f)

with open('testing_cylinder_photospheric_vtwist001_pos_slow2_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p1_pos_slow2, sol_ks_kink_v001_p1_pos_slow2 = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p1_pos_fast, sol_ks_kink_v001_p1_pos_fast = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power1_fund_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v001_p1_pos_fast2, sol_ks_kink_v001_p1_pos_fast2 = pickle.load(f)


sol_omegas_kink_v001_p1 = np.concatenate((sol_omegas_kink_v001_p1_pos_slow, sol_omegas_kink_v001_p1_pos_slow2, sol_omegas_kink_v001_p1_pos_fast, sol_omegas_kink_v001_p1_pos_fast2), axis=None)  
sol_ks_kink_v001_p1 = np.concatenate((sol_ks_kink_v001_p1_pos_slow, sol_ks_kink_v001_p1_pos_slow2, sol_ks_kink_v001_p1_pos_fast, sol_ks_kink_v001_p1_pos_fast2), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas_kink_v001_p1 = [x for _,x in sorted(zip(sol_ks_kink_v001_p1,sol_omegas_kink_v001_p1))]
sol_ks_kink_v001_p1 = np.sort(sol_ks_kink_v001_p1)

sol_omegas_kink_v001_p1 = np.array(sol_omegas_kink_v001_p1)
sol_ks_kink_v001_p1 = np.array(sol_ks_kink_v001_p1)



##################################################    v = 0.05

with open('Cylindrical_photospheric_vtwist005_pos_slow_sausage.pickle', 'rb') as f:
    sol_omegas_v005_p1_pos_slow, sol_ks_v005_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_pos_fast_sausage.pickle', 'rb') as f:
    sol_omegas_v005_p1_pos_fast, sol_ks_v005_p1_pos_fast = pickle.load(f)

sol_omegas_v005_p1 = np.concatenate((sol_omegas_v005_p1_pos_slow, sol_omegas_v005_p1_pos_fast), axis=None)  
sol_ks_v005_p1 = np.concatenate((sol_ks_v005_p1_pos_slow, sol_ks_v005_p1_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v005_p1 = [x for _,x in sorted(zip(sol_ks_v005_p1,sol_omegas_v005_p1))]
sol_ks_v005_p1 = np.sort(sol_ks_v005_p1)

sol_omegas_v005_p1 = np.array(sol_omegas_v005_p1)
sol_ks_v005_p1 = np.array(sol_ks_v005_p1)

#####

with open('testing_cylinder_photospheric_vtwist005_pos_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p1_pos_slow, sol_ks_kink_v005_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power1_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p1_pos_slow2, sol_ks_kink_v005_p1_pos_slow2 = pickle.load(f)

with open('testing_cylinder_photospheric_vtwist005_pos_fast_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p1_pos_fast, sol_ks_kink_v005_p1_pos_fast = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power1_fund_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v005_p1_pos_fast2, sol_ks_kink_v005_p1_pos_fast2 = pickle.load(f)


sol_omegas_kink_v005_p1 = np.concatenate((sol_omegas_kink_v005_p1_pos_slow, sol_omegas_kink_v005_p1_pos_slow2, sol_omegas_kink_v005_p1_pos_fast, sol_omegas_kink_v005_p1_pos_fast2), axis=None)  
sol_ks_kink_v005_p1 = np.concatenate((sol_ks_kink_v005_p1_pos_slow, sol_ks_kink_v005_p1_pos_slow2, sol_ks_kink_v005_p1_pos_fast, sol_ks_kink_v005_p1_pos_fast2), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas_kink_v005_p1 = [x for _,x in sorted(zip(sol_ks_kink_v005_p1,sol_omegas_kink_v005_p1))]
sol_ks_kink_v005_p1 = np.sort(sol_ks_kink_v005_p1)

sol_omegas_kink_v005_p1 = np.array(sol_omegas_kink_v005_p1)
sol_ks_kink_v005_p1 = np.array(sol_ks_kink_v005_p1)


##################################################    v = 0.1

with open('Cylindrical_photospheric_vtwist01_power1_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v01_p1_pos_slow, sol_ks_v01_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power1_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v01_p1_pos_fast, sol_ks_v01_p1_pos_fast = pickle.load(f)

sol_omegas_v01_p1 = np.concatenate((sol_omegas_v01_p1_pos_fast, sol_omegas_v01_p1_pos_slow), axis=None)  
sol_ks_v01_p1 = np.concatenate((sol_ks_v01_p1_pos_fast, sol_ks_v01_p1_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v01_p1 = [x for _,x in sorted(zip(sol_ks_v01_p1,sol_omegas_v01_p1))]
sol_ks_v01_p1 = np.sort(sol_ks_v01_p1)

sol_omegas_v01_p1 = np.array(sol_omegas_v01_p1)
sol_ks_v01_p1 = np.array(sol_ks_v01_p1)

#####

with open('testing_cylinder_photospheric_vtwist01_pos_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p1_pos_slow, sol_ks_kink_v01_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p1_pos_fast, sol_ks_kink_v01_p1_pos_fast = pickle.load(f)


sol_omegas_kink_v01_p1 = np.concatenate((sol_omegas_kink_v01_p1_pos_slow, sol_omegas_kink_v01_p1_pos_fast), axis=None)  
sol_ks_kink_v01_p1 = np.concatenate((sol_ks_kink_v01_p1_pos_slow, sol_ks_kink_v01_p1_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas_kink_v01_p1 = [x for _,x in sorted(zip(sol_ks_kink_v01_p1,sol_omegas_kink_v01_p1))]
sol_ks_kink_v01_p1 = np.sort(sol_ks_kink_v01_p1)

sol_omegas_kink_v01_p1 = np.array(sol_omegas_kink_v01_p1)
sol_ks_kink_v01_p1 = np.array(sol_ks_kink_v01_p1)

#####

with open('Cylindrical_photospheric_vtwist01_power1_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p1_pos_slow, sol_ks_kink_v01_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p1_pos_fast, sol_ks_kink_v01_p1_pos_fast = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power1_fund_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v01_p1_pos_fast2, sol_ks_kink_v01_p1_pos_fast2 = pickle.load(f)

sol_omegas_kink_v01_p1 = np.concatenate((sol_omegas_kink_v01_p1_pos_slow, sol_omegas_kink_v01_p1_pos_fast, sol_omegas_kink_v01_p1_pos_fast2), axis=None)  
sol_ks_kink_v01_p1 = np.concatenate((sol_ks_kink_v01_p1_pos_slow, sol_ks_kink_v01_p1_pos_fast, sol_ks_kink_v01_p1_pos_fast2), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v01_p1 = [x for _,x in sorted(zip(sol_ks_kink_v01_p1,sol_omegas_kink_v01_p1))]
sol_ks_kink_v01_p1 = np.sort(sol_ks_kink_v01_p1)

sol_omegas_kink_v01_p1 = np.array(sol_omegas_kink_v01_p1)
sol_ks_kink_v01_p1 = np.array(sol_ks_kink_v01_p1)



##################################################    v = 0.15

with open('Cylindrical_photospheric_vtwist015_power1_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v015_p1_pos_slow, sol_ks_v015_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power1_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v015_p1_pos_fast, sol_ks_v015_p1_pos_fast = pickle.load(f)

sol_omegas_v015_p1 = np.concatenate((sol_omegas_v015_p1_pos_fast, sol_omegas_v015_p1_pos_slow), axis=None)  
sol_ks_v015_p1 = np.concatenate((sol_ks_v015_p1_pos_fast, sol_ks_v015_p1_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v015_p1 = [x for _,x in sorted(zip(sol_ks_v015_p1,sol_omegas_v015_p1))]
sol_ks_v015_p1 = np.sort(sol_ks_v015_p1)

sol_omegas_v015_p1 = np.array(sol_omegas_v015_p1)
sol_ks_v015_p1 = np.array(sol_ks_v015_p1)

#####

with open('Cylindrical_photospheric_vtwist015_power1_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p1_pos_slow, sol_ks_kink_v015_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p1_pos_fast, sol_ks_kink_v015_p1_pos_fast = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power1_fund_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v015_p1_pos_fast2, sol_ks_kink_v015_p1_pos_fast2 = pickle.load(f)

sol_omegas_kink_v015_p1 = np.concatenate((sol_omegas_kink_v015_p1_pos_slow, sol_omegas_kink_v015_p1_pos_fast, sol_omegas_kink_v015_p1_pos_fast2), axis=None)  
sol_ks_kink_v015_p1 = np.concatenate((sol_ks_kink_v015_p1_pos_slow, sol_ks_kink_v015_p1_pos_fast, sol_ks_kink_v015_p1_pos_fast2), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v015_p1 = [x for _,x in sorted(zip(sol_ks_kink_v015_p1,sol_omegas_kink_v015_p1))]
sol_ks_kink_v015_p1 = np.sort(sol_ks_kink_v015_p1)

sol_omegas_kink_v015_p1 = np.array(sol_omegas_kink_v015_p1)
sol_ks_kink_v015_p1 = np.array(sol_ks_kink_v015_p1)



############################################################



##################################################    v = 0.25


#################################################

with open('Cylindrical_photospheric_vtwist025_power1_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v025_p1_pos_slow, sol_ks_v025_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power1_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v025_p1_pos_fast, sol_ks_v025_p1_pos_fast = pickle.load(f)

sol_omegas_v025_p1 = np.concatenate((sol_omegas_v025_p1_pos_slow, sol_omegas_v025_p1_pos_fast), axis=None)  
sol_ks_v025_p1 = np.concatenate((sol_ks_v025_p1_pos_slow, sol_ks_v025_p1_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v025_p1 = [x for _,x in sorted(zip(sol_ks_v025_p1,sol_omegas_v025_p1))]
sol_ks_v025_p1 = np.sort(sol_ks_v025_p1)

sol_omegas_v025_p1 = np.array(sol_omegas_v025_p1)
sol_ks_v025_p1 = np.array(sol_ks_v025_p1)


#####


with open('Cylindrical_photospheric_vtwist025_power1_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p1_pos_slow, sol_ks_kink_v025_p1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p1_pos_fast, sol_ks_kink_v025_p1_pos_fast = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power1_fund_kink2.pickle', 'rb') as f:
    sol_omegas_kink_v025_p1_pos_fast2, sol_ks_kink_v025_p1_pos_fast2 = pickle.load(f)

sol_omegas_kink_v025_p1 = np.concatenate((sol_omegas_kink_v025_p1_pos_slow, sol_omegas_kink_v025_p1_pos_fast, sol_omegas_kink_v025_p1_pos_fast2), axis=None)  
sol_ks_kink_v025_p1 = np.concatenate((sol_ks_kink_v025_p1_pos_slow, sol_ks_kink_v025_p1_pos_fast, sol_ks_kink_v025_p1_pos_fast2), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v025_p1 = [x for _,x in sorted(zip(sol_ks_kink_v025_p1,sol_omegas_kink_v025_p1))]
sol_ks_kink_v025_p1 = np.sort(sol_ks_kink_v025_p1)

sol_omegas_kink_v025_p1 = np.array(sol_omegas_kink_v025_p1)
sol_ks_kink_v025_p1 = np.array(sol_ks_kink_v025_p1)
############################################################





########   READ IN VARIABLES    power = 1.05   #########

#################################################    v = 0.01

with open('Cylindrical_photospheric_vtwist001_power105_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v001_p105_pos_slow, sol_ks_v001_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power105_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v001_p105_pos_fast, sol_ks_v001_p105_pos_fast = pickle.load(f)

sol_omegas_v001_p105 = np.concatenate((sol_omegas_v001_p105_pos_slow, sol_omegas_v001_p105_pos_fast), axis=None)  
sol_ks_v001_p105 = np.concatenate((sol_ks_v001_p105_pos_slow, sol_ks_v001_p105_pos_fast), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v001_p105 = [x for _,x in sorted(zip(sol_ks_v001_p105,sol_omegas_v001_p105))]
sol_ks_v001_p105 = np.sort(sol_ks_v001_p105)

sol_omegas_v001_p105 = np.array(sol_omegas_v001_p105)
sol_ks_v001_p105 = np.array(sol_ks_v001_p105)

#############


with open('Cylindrical_photospheric_vtwist001_power105_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p105_pos_slow, sol_ks_kink_v001_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power105_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p105_pos_fast, sol_ks_kink_v001_p105_pos_fast = pickle.load(f)


sol_omegas_kink_v001_p105 = np.concatenate((sol_omegas_kink_v001_p105_pos_slow, sol_omegas_kink_v001_p105_pos_fast), axis=None)  
sol_ks_kink_v001_p105 = np.concatenate((sol_ks_kink_v001_p105_pos_slow, sol_ks_kink_v001_p105_pos_fast), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v001_p105 = [x for _,x in sorted(zip(sol_ks_kink_v001_p105,sol_omegas_kink_v001_p105))]
sol_ks_kink_v001_p105 = np.sort(sol_ks_kink_v001_p105)

sol_omegas_kink_v001_p105 = np.array(sol_omegas_kink_v001_p105)
sol_ks_kink_v001_p105 = np.array(sol_ks_kink_v001_p105)



#################################################    v = 0.05

with open('Cylindrical_photospheric_vtwist005_power105_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v005_p105_pos_slow, sol_ks_v005_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power105_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v005_p105_pos_fast, sol_ks_v005_p105_pos_fast = pickle.load(f)


sol_omegas_v005_p105 = np.concatenate((sol_omegas_v005_p105_pos_slow, sol_omegas_v005_p105_pos_fast), axis=None)  
sol_ks_v005_p105 = np.concatenate((sol_ks_v005_p105_pos_slow, sol_ks_v005_p105_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v005_p105 = [x for _,x in sorted(zip(sol_ks_v005_p105,sol_omegas_v005_p105))]
sol_ks_v005_p105 = np.sort(sol_ks_v005_p105)

sol_omegas_v005_p105 = np.array(sol_omegas_v005_p105)
sol_ks_v005_p105 = np.array(sol_ks_v005_p105)

#####

with open('Cylindrical_photospheric_vtwist005_power105_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p105_pos_slow, sol_ks_kink_v005_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power105_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p105_pos_fast, sol_ks_kink_v005_p105_pos_fast = pickle.load(f)


sol_omegas_kink_v005_p105 = np.concatenate((sol_omegas_kink_v005_p105_pos_slow, sol_omegas_kink_v005_p105_pos_fast), axis=None)  
sol_ks_kink_v005_p105 = np.concatenate((sol_ks_kink_v005_p105_pos_slow, sol_ks_kink_v005_p105_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v005_p105 = [x for _,x in sorted(zip(sol_ks_kink_v005_p105,sol_omegas_kink_v005_p105))]
sol_ks_kink_v005_p105 = np.sort(sol_ks_kink_v005_p105)

sol_omegas_kink_v005_p105 = np.array(sol_omegas_kink_v005_p105)
sol_ks_kink_v005_p105 = np.array(sol_ks_kink_v005_p105)


#################################################    v = 0.1

with open('Cylindrical_photospheric_vtwist01_power105_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v01_p105_pos_slow, sol_ks_v01_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power105_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v01_p105_pos_fast, sol_ks_v01_p105_pos_fast = pickle.load(f)

sol_omegas_v01_p105 = np.concatenate((sol_omegas_v01_p105_pos_fast, sol_omegas_v01_p105_pos_slow), axis=None)  
sol_ks_v01_p105 = np.concatenate((sol_ks_v01_p105_pos_fast, sol_ks_v01_p105_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v01_p105 = [x for _,x in sorted(zip(sol_ks_v01_p105,sol_omegas_v01_p105))]
sol_ks_v01_p105 = np.sort(sol_ks_v01_p105)

sol_omegas_v01_p105 = np.array(sol_omegas_v01_p105)
sol_ks_v01_p105 = np.array(sol_ks_v01_p105)

#############


with open('Cylindrical_photospheric_vtwist01_power105_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p105_pos_slow, sol_ks_kink_v01_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power105_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p105_pos_fast, sol_ks_kink_v01_p105_pos_fast = pickle.load(f)


sol_omegas_kink_v01_p105 = np.concatenate((sol_omegas_kink_v01_p105_pos_slow, sol_omegas_kink_v01_p105_pos_fast), axis=None)  
sol_ks_kink_v01_p105 = np.concatenate((sol_ks_kink_v01_p105_pos_slow, sol_ks_kink_v01_p105_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v01_p105 = [x for _,x in sorted(zip(sol_ks_kink_v01_p105,sol_omegas_kink_v01_p105))]
sol_ks_kink_v01_p105 = np.sort(sol_ks_kink_v01_p105)

sol_omegas_kink_v01_p105 = np.array(sol_omegas_kink_v01_p105)
sol_ks_kink_v01_p105 = np.array(sol_ks_kink_v01_p105)



##################################################    v = 0.15

with open('Cylindrical_photospheric_vtwist015_power105_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v015_p105_pos_slow, sol_ks_v015_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power105_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v015_p105_pos_fast, sol_ks_v015_p105_pos_fast = pickle.load(f)

sol_omegas_v015_p105 = np.concatenate((sol_omegas_v015_p105_pos_fast, sol_omegas_v015_p105_pos_slow), axis=None)  
sol_ks_v015_p105 = np.concatenate((sol_ks_v015_p105_pos_fast, sol_ks_v015_p105_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v015_p105 = [x for _,x in sorted(zip(sol_ks_v015_p105,sol_omegas_v015_p105))]
sol_ks_v015_p105 = np.sort(sol_ks_v015_p105)

sol_omegas_v015_p105 = np.array(sol_omegas_v015_p105)
sol_ks_v015_p105 = np.array(sol_ks_v015_p105)

#############

with open('Cylindrical_photospheric_vtwist015_power105_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p105_pos_slow, sol_ks_kink_v015_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power105_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p105_pos_fast, sol_ks_kink_v015_p105_pos_fast = pickle.load(f)


sol_omegas_kink_v015_p105 = np.concatenate((sol_omegas_kink_v015_p105_pos_fast, sol_omegas_kink_v015_p105_pos_slow), axis=None)  
sol_ks_kink_v015_p105 = np.concatenate((sol_ks_kink_v015_p105_pos_fast, sol_ks_kink_v015_p105_pos_slow), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v015_p105 = [x for _,x in sorted(zip(sol_ks_kink_v015_p105,sol_omegas_kink_v015_p105))]
sol_ks_kink_v015_p105 = np.sort(sol_ks_kink_v015_p105)

sol_omegas_kink_v015_p105 = np.array(sol_omegas_kink_v015_p105)
sol_ks_kink_v015_p105 = np.array(sol_ks_kink_v015_p105)



##################################################    v = 0.25

with open('Cylindrical_photospheric_vtwist025_power105_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v025_p105_pos_slow, sol_ks_v025_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power105_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v025_p105_pos_fast, sol_ks_v025_p105_pos_fast = pickle.load(f)

sol_omegas_v025_p105 = np.concatenate((sol_omegas_v025_p105_pos_fast, sol_omegas_v025_p105_pos_slow), axis=None)  
sol_ks_v025_p105 = np.concatenate((sol_ks_v025_p105_pos_fast, sol_ks_v025_p105_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v025_p105 = [x for _,x in sorted(zip(sol_ks_v025_p105,sol_omegas_v025_p105))]
sol_ks_v025_p105 = np.sort(sol_ks_v025_p105)

sol_omegas_v025_p105 = np.array(sol_omegas_v025_p105)
sol_ks_v025_p105 = np.array(sol_ks_v025_p105)

#############


with open('Cylindrical_photospheric_vtwist025_power105_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p105_pos_slow, sol_ks_kink_v025_p105_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power105_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p105_pos_fast, sol_ks_kink_v025_p105_pos_fast = pickle.load(f)


sol_omegas_kink_v025_p105 = np.concatenate((sol_omegas_kink_v025_p105_pos_fast, sol_omegas_kink_v025_p105_pos_slow), axis=None)  
sol_ks_kink_v025_p105 = np.concatenate((sol_ks_kink_v025_p105_pos_fast, sol_ks_kink_v025_p105_pos_slow), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v025_p105 = [x for _,x in sorted(zip(sol_ks_kink_v025_p105,sol_omegas_kink_v025_p105))]
sol_ks_kink_v025_p105 = np.sort(sol_ks_kink_v025_p105)

sol_omegas_kink_v025_p105 = np.array(sol_omegas_kink_v025_p105)
sol_ks_kink_v025_p105 = np.array(sol_ks_kink_v025_p105)



####################################################################



########   READ IN VARIABLES    power = 1.1   #########

#################################################

with open('Cylindrical_photospheric_vtwist005_power11_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p11_pos_slow, sol_ks_kink_v005_p11_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power11_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p11_pos_fast, sol_ks_kink_v005_p11_pos_fast = pickle.load(f)


sol_omegas_kink_v005_p11 = np.concatenate((sol_omegas_kink_v005_p11_pos_slow, sol_omegas_kink_v005_p11_pos_fast), axis=None)  
sol_ks_kink_v005_p11 = np.concatenate((sol_ks_kink_v005_p11_pos_slow, sol_ks_kink_v005_p11_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v005_p11 = [x for _,x in sorted(zip(sol_ks_kink_v005_p11,sol_omegas_kink_v005_p11))]
sol_ks_kink_v005_p11 = np.sort(sol_ks_kink_v005_p11)

sol_omegas_kink_v005_p11 = np.array(sol_omegas_kink_v005_p11)
sol_ks_kink_v005_p11 = np.array(sol_ks_kink_v005_p11)



####################################################################


with open('Cylindrical_photospheric_vtwist025_power11_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p11_pos_slow, sol_ks_kink_v025_p11_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power11_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p11_pos_fast, sol_ks_kink_v025_p11_pos_fast = pickle.load(f)


sol_omegas_kink_v025_p11 = np.concatenate((sol_omegas_kink_v025_p11_pos_slow, sol_omegas_kink_v025_p11_pos_fast), axis=None)  
sol_ks_kink_v025_p11 = np.concatenate((sol_ks_kink_v025_p11_pos_slow, sol_ks_kink_v025_p11_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v025_p11 = [x for _,x in sorted(zip(sol_ks_kink_v025_p11,sol_omegas_kink_v025_p11))]
sol_ks_kink_v025_p11 = np.sort(sol_ks_kink_v025_p11)

sol_omegas_kink_v025_p11 = np.array(sol_omegas_kink_v025_p11)
sol_ks_kink_v025_p11 = np.array(sol_ks_kink_v025_p11)



####################################################################



########   READ IN VARIABLES    power = 1.25   #########

##################################################    v = 0.01

with open('Cylindrical_photospheric_vtwist001_power125_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v001_p125_pos_slow, sol_ks_v001_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power125_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v001_p125_pos_fast, sol_ks_v001_p125_pos_fast = pickle.load(f)

sol_omegas_v001_p125 = np.concatenate((sol_omegas_v001_p125_pos_fast, sol_omegas_v001_p125_pos_slow), axis=None)  
sol_ks_v001_p125 = np.concatenate((sol_ks_v001_p125_pos_fast, sol_ks_v001_p125_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v001_p125 = [x for _,x in sorted(zip(sol_ks_v001_p125,sol_omegas_v001_p125))]
sol_ks_v001_p125 = np.sort(sol_ks_v001_p125)

sol_omegas_v001_p125 = np.array(sol_omegas_v001_p125)
sol_ks_v001_p125 = np.array(sol_ks_v001_p125)

#############


with open('Cylindrical_photospheric_vtwist001_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p125_pos_slow, sol_ks_kink_v001_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist001_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v001_p125_pos_fast, sol_ks_kink_v001_p125_pos_fast = pickle.load(f)


sol_omegas_kink_v001_p125 = np.concatenate((sol_omegas_kink_v001_p125_pos_slow, sol_omegas_kink_v001_p125_pos_fast), axis=None)  
sol_ks_kink_v001_p125 = np.concatenate((sol_ks_kink_v001_p125_pos_slow, sol_ks_kink_v001_p125_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v001_p125 = [x for _,x in sorted(zip(sol_ks_kink_v001_p125,sol_omegas_kink_v001_p125))]
sol_ks_kink_v001_p125 = np.sort(sol_ks_kink_v001_p125)

sol_omegas_kink_v001_p125 = np.array(sol_omegas_kink_v001_p125)
sol_ks_kink_v001_p125 = np.array(sol_ks_kink_v001_p125)



##################################################    v = 0.05

with open('Cylindrical_photospheric_vtwist005_power125_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v005_p125_pos_slow, sol_ks_v005_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power125_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v005_p125_pos_fast, sol_ks_v005_p125_pos_fast = pickle.load(f)

sol_omegas_v005_p125 = np.concatenate((sol_omegas_v005_p125_pos_fast, sol_omegas_v005_p125_pos_slow), axis=None)  
sol_ks_v005_p125 = np.concatenate((sol_ks_v005_p125_pos_fast, sol_ks_v005_p125_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v005_p125 = [x for _,x in sorted(zip(sol_ks_v005_p125,sol_omegas_v005_p125))]
sol_ks_v005_p125 = np.sort(sol_ks_v005_p125)

sol_omegas_v005_p125 = np.array(sol_omegas_v005_p125)
sol_ks_v005_p125 = np.array(sol_ks_v005_p125)

#############


with open('Cylindrical_photospheric_vtwist005_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p125_pos_slow, sol_ks_kink_v005_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist005_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v005_p125_pos_fast, sol_ks_kink_v005_p125_pos_fast = pickle.load(f)


sol_omegas_kink_v005_p125 = np.concatenate((sol_omegas_kink_v005_p125_pos_fast, sol_omegas_kink_v005_p125_pos_slow), axis=None)  
sol_ks_kink_v005_p125 = np.concatenate((sol_ks_kink_v005_p125_pos_fast, sol_ks_kink_v005_p125_pos_slow), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v005_p125 = [x for _,x in sorted(zip(sol_ks_kink_v005_p125,sol_omegas_kink_v005_p125))]
sol_ks_kink_v005_p125 = np.sort(sol_ks_kink_v005_p125)

sol_omegas_kink_v005_p125 = np.array(sol_omegas_kink_v005_p125)
sol_ks_kink_v005_p125 = np.array(sol_ks_kink_v005_p125)



##################################################    v = 0.1

with open('Cylindrical_photospheric_vtwist01_power125_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v01_p125_pos_slow, sol_ks_v01_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power125_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v01_p125_pos_fast, sol_ks_v01_p125_pos_fast = pickle.load(f)

sol_omegas_v01_p125 = np.concatenate((sol_omegas_v01_p125_pos_fast, sol_omegas_v01_p125_pos_slow), axis=None)  
sol_ks_v01_p125 = np.concatenate((sol_ks_v01_p125_pos_fast, sol_ks_v01_p125_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v01_p125 = [x for _,x in sorted(zip(sol_ks_v01_p125,sol_omegas_v01_p125))]
sol_ks_v01_p125 = np.sort(sol_ks_v01_p125)

sol_omegas_v01_p125 = np.array(sol_omegas_v01_p125)
sol_ks_v01_p125 = np.array(sol_ks_v01_p125)

#############


with open('Cylindrical_photospheric_vtwist01_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p125_pos_slow, sol_ks_kink_v01_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist01_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v01_p125_pos_fast, sol_ks_kink_v01_p125_pos_fast = pickle.load(f)


sol_omegas_kink_v01_p125 = np.concatenate((sol_omegas_kink_v01_p125_pos_fast, sol_omegas_kink_v01_p125_pos_slow), axis=None)  
sol_ks_kink_v01_p125 = np.concatenate((sol_ks_kink_v01_p125_pos_fast, sol_ks_kink_v01_p125_pos_slow), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v01_p125 = [x for _,x in sorted(zip(sol_ks_kink_v01_p125,sol_omegas_kink_v01_p125))]
sol_ks_kink_v01_p125 = np.sort(sol_ks_kink_v01_p125)

sol_omegas_kink_v01_p125 = np.array(sol_omegas_kink_v01_p125)
sol_ks_kink_v01_p125 = np.array(sol_ks_kink_v01_p125)



##################################################    v = 0.15

with open('Cylindrical_photospheric_vtwist015_power125_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v015_p125_pos_slow, sol_ks_v015_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power125_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v015_p125_pos_fast, sol_ks_v015_p125_pos_fast = pickle.load(f)

sol_omegas_v015_p125 = np.concatenate((sol_omegas_v015_p125_pos_fast, sol_omegas_v015_p125_pos_slow), axis=None)  
sol_ks_v015_p125 = np.concatenate((sol_ks_v015_p125_pos_fast, sol_ks_v015_p125_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v015_p125 = [x for _,x in sorted(zip(sol_ks_v015_p125,sol_omegas_v015_p125))]
sol_ks_v015_p125 = np.sort(sol_ks_v015_p125)

sol_omegas_v015_p125 = np.array(sol_omegas_v015_p125)
sol_ks_v015_p125 = np.array(sol_ks_v015_p125)

#############


with open('Cylindrical_photospheric_vtwist015_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p125_pos_slow, sol_ks_kink_v015_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist015_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v015_p125_pos_fast, sol_ks_kink_v015_p125_pos_fast = pickle.load(f)


sol_omegas_kink_v015_p125 = np.concatenate((sol_omegas_kink_v015_p125_pos_slow, sol_omegas_kink_v015_p125_pos_fast), axis=None)  
sol_ks_kink_v015_p125 = np.concatenate((sol_ks_kink_v015_p125_pos_slow, sol_ks_kink_v015_p125_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kink_v015_p125 = [x for _,x in sorted(zip(sol_ks_kink_v015_p125,sol_omegas_kink_v015_p125))]
sol_ks_kink_v015_p125 = np.sort(sol_ks_kink_v015_p125)

sol_omegas_kink_v015_p125 = np.array(sol_omegas_kink_v015_p125)
sol_ks_kink_v015_p125 = np.array(sol_ks_kink_v015_p125)



##################################################    v = 0.25

with open('Cylindrical_photospheric_vtwist025_power125_sausage_slow.pickle', 'rb') as f:
    sol_omegas_v025_p125_pos_slow, sol_ks_v025_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power125_sausage_fast.pickle', 'rb') as f:
    sol_omegas_v025_p125_pos_fast, sol_ks_v025_p125_pos_fast = pickle.load(f)

sol_omegas_v025_p125 = np.concatenate((sol_omegas_v025_p125_pos_fast, sol_omegas_v025_p125_pos_slow), axis=None)  
sol_ks_v025_p125 = np.concatenate((sol_ks_v025_p125_pos_fast, sol_ks_v025_p125_pos_slow), axis=None)


## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_v025_p125 = [x for _,x in sorted(zip(sol_ks_v025_p125,sol_omegas_v025_p125))]
sol_ks_v025_p125 = np.sort(sol_ks_v025_p125)

sol_omegas_v025_p125 = np.array(sol_omegas_v025_p125)
sol_ks_v025_p125 = np.array(sol_ks_v025_p125)

#############


with open('Cylindrical_photospheric_vtwist025_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p125_pos_slow, sol_ks_kink_v025_p125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kink_v025_p125_pos_fast, sol_ks_kink_v025_p125_pos_fast = pickle.load(f)

sol_omegas_kink_v025_p125 = np.concatenate((sol_omegas_kink_v025_p125_pos_fast, sol_omegas_kink_v025_p125_pos_slow), axis=None)  
sol_ks_kink_v025_p125 = np.concatenate((sol_ks_kink_v025_p125_pos_fast, sol_ks_kink_v025_p125_pos_slow), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas_kink_v025_p125 = [x for _,x in sorted(zip(sol_ks_kink_v025_p125,sol_omegas_kink_v025_p125))]
sol_ks_kink_v025_p125 = np.sort(sol_ks_kink_v025_p125)

sol_omegas_kink_v025_p125 = np.array(sol_omegas_kink_v025_p125)
sol_ks_kink_v025_p125 = np.array(sol_ks_kink_v025_p125)





########   READ IN VARIABLES    power = 1.5   #########

#################################################

with open('Cylindrical_photospheric_vtwist025_power15_slow_kink.pickle', 'rb') as f:
    sol_omegas_kinkp15_pos_slow, sol_ks_kinkp15_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power15_fund_kink.pickle', 'rb') as f:
    sol_omegas_kinkp15_pos_fast, sol_ks_kinkp15_pos_fast = pickle.load(f)


sol_omegas_kinkp15 = np.concatenate((sol_omegas_kinkp15_pos_slow, sol_omegas_kinkp15_pos_fast), axis=None)  
sol_ks_kinkp15 = np.concatenate((sol_ks_kinkp15_pos_slow, sol_ks_kinkp15_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas_kinkp15 = [x for _,x in sorted(zip(sol_ks_kinkp15,sol_omegas_kinkp15))]
sol_ks_kinkp15 = np.sort(sol_ks_kinkp15)

sol_omegas_kinkp15 = np.array(sol_omegas_kinkp15)
sol_ks_kinkp15 = np.array(sol_ks_kinkp15)



#######################################################################
#######################################################################

test_k = np.linspace(0.001, 4., 500)
test_c_i0 = c_i0*test_k
test_cT_i0 = cT_i0*test_k 
test_c_e = c_e*test_k
test_c_kink = c_kink*test_k


######################################

plt.figure()
ax = plt.subplot(111)
ax.set_title(r'$\propto 0.1*r^{1.25}$')
ax.set_xlabel("$ka$", fontsize=16)
ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.set_xlim(0., 4.)  
ax.set_ylim(0.85, c_e)  

ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')

ax.plot(sol_ks_v01_p125, (sol_omegas_v01_p125/sol_ks_v01_p125), 'r.', markersize=4.)
ax.plot(sol_ks_kink_v01_p125, (sol_omegas_kink_v01_p125/sol_ks_kink_v01_p125), 'b.', markersize=4.)

amp =0.25
test_r=1.
alpha=1.
ax.fill_between(test_k, cT_i0+((amp*(0.002**(alpha-1)))/(test_k)), cT_i0+((amp*(test_r**(alpha-1)))/(test_k)), color='green', alpha=0.4)  

#plt.savefig("v025_p1_photospheric_curves_KINKONLY.png")
#plt.show()
#exit()





#########################################################################################################################################
#
###########     power  1    #######
#
#########################################################################################################################################

sol_ksv001_p1_singleplot_fast_sausage = []
sol_omegasv001_p1_singleplot_fast_sausage = []
sol_ksv001_p1_singleplot_slow_surf_sausage = []
sol_omegasv001_p1_singleplot_slow_surf_sausage = []
sol_ksv001_p1_singleplot_slow_body_sausage = []
sol_omegasv001_p1_singleplot_slow_body_sausage = []


for i in range(len(sol_ks_v001_p1)):
    if sol_ks_v001_p1[i] > 1.48 and sol_ks_v001_p1[i] < 1.51:        
       if sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] > c_kink and sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] < c_e:
         sol_ksv001_p1_singleplot_fast_sausage.append(sol_ks_v001_p1[i])
         sol_omegasv001_p1_singleplot_fast_sausage.append(sol_omegas_v001_p1[i])        

    if sol_ks_v001_p1[i] > 1.55 and sol_ks_v001_p1[i] < 1.6:         
       if sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] > 0.88 and sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] < cT_i0: 
         sol_ksv001_p1_singleplot_slow_surf_sausage.append(sol_ks_v001_p1[i])
         sol_omegasv001_p1_singleplot_slow_surf_sausage.append(sol_omegas_v001_p1[i])

    if sol_ks_v001_p1[i] > 1.29 and sol_ks_v001_p1[i] < 1.45:         
       if sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] > 0.905 and sol_omegas_v001_p1[i]/sol_ks_v001_p1[i] < 0.92: 
         sol_ksv001_p1_singleplot_slow_body_sausage.append(sol_ks_v001_p1[i])
         sol_omegasv001_p1_singleplot_slow_body_sausage.append(sol_omegas_v001_p1[i])

    
sol_ksv001_p1_singleplot_fast_sausage = np.array(sol_ksv001_p1_singleplot_fast_sausage)
sol_omegasv001_p1_singleplot_fast_sausage = np.array(sol_omegasv001_p1_singleplot_fast_sausage)
sol_ksv001_p1_singleplot_slow_surf_sausage = np.array(sol_ksv001_p1_singleplot_slow_surf_sausage)
sol_omegasv001_p1_singleplot_slow_surf_sausage = np.array(sol_omegasv001_p1_singleplot_slow_surf_sausage)
sol_ksv001_p1_singleplot_slow_body_sausage = np.array(sol_ksv001_p1_singleplot_slow_body_sausage)
sol_omegasv001_p1_singleplot_slow_body_sausage = np.array(sol_omegasv001_p1_singleplot_slow_body_sausage)


#####

sol_ksv001_p1_singleplot_fast_kink = []
sol_omegasv001_p1_singleplot_fast_kink = []
sol_ksv001_p1_singleplot_slow_surf_kink = []
sol_omegasv001_p1_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kink_v001_p1)):
    if sol_ks_kink_v001_p1[i] > 0.68 and sol_ks_kink_v001_p1[i] < 0.7:        
       if sol_omegas_kink_v001_p1[i]/sol_ks_kink_v001_p1[i] > c_kink and sol_omegas_kink_v001_p1[i]/sol_ks_kink_v001_p1[i] < c_e:
         sol_ksv001_p1_singleplot_fast_kink.append(sol_ks_kink_v001_p1[i])
         sol_omegasv001_p1_singleplot_fast_kink.append(sol_omegas_kink_v001_p1[i])        

    if sol_ks_kink_v001_p1[i] > 0.56 and sol_ks_kink_v001_p1[i] < 0.62:         
       if sol_omegas_kink_v001_p1[i]/sol_ks_kink_v001_p1[i] > cT_i0 and sol_omegas_kink_v001_p1[i]/sol_ks_kink_v001_p1[i] < c_i0: 
         sol_ksv001_p1_singleplot_slow_surf_kink.append(sol_ks_kink_v001_p1[i])
         sol_omegasv001_p1_singleplot_slow_surf_kink.append(sol_omegas_kink_v001_p1[i])

    
sol_ksv001_p1_singleplot_fast_kink = np.array(sol_ksv001_p1_singleplot_fast_kink)
sol_omegasv001_p1_singleplot_fast_kink = np.array(sol_omegasv001_p1_singleplot_fast_kink)
sol_ksv001_p1_singleplot_slow_surf_kink = np.array(sol_ksv001_p1_singleplot_slow_surf_kink)
sol_omegasv001_p1_singleplot_slow_surf_kink = np.array(sol_omegasv001_p1_singleplot_slow_surf_kink)



##################################
sol_ksv005_p1_singleplot_fast_sausage = []
sol_omegasv005_p1_singleplot_fast_sausage = []
sol_ksv005_p1_singleplot_slow_surf_sausage = []
sol_omegasv005_p1_singleplot_slow_surf_sausage = []
sol_ksv005_p1_singleplot_slow_body_sausage = []
sol_omegasv005_p1_singleplot_slow_body_sausage = []


for i in range(len(sol_ks_v005_p1)):
    if sol_ks_v005_p1[i] > 1.48 and sol_ks_v005_p1[i] < 1.51:        
       if sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] > c_kink and sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] < c_e:
         sol_ksv005_p1_singleplot_fast_sausage.append(sol_ks_v005_p1[i])
         sol_omegasv005_p1_singleplot_fast_sausage.append(sol_omegas_v005_p1[i])        

    if sol_ks_v005_p1[i] > 1.55 and sol_ks_v005_p1[i] < 1.6:         
       if sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] > 0.88 and sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] < cT_i0: 
         sol_ksv005_p1_singleplot_slow_surf_sausage.append(sol_ks_v005_p1[i])
         sol_omegasv005_p1_singleplot_slow_surf_sausage.append(sol_omegas_v005_p1[i])

    if sol_ks_v005_p1[i] > 1.4 and sol_ks_v005_p1[i] < 1.45:         
       if sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] > 0.908 and sol_omegas_v005_p1[i]/sol_ks_v005_p1[i] < 0.92: 
         sol_ksv005_p1_singleplot_slow_body_sausage.append(sol_ks_v005_p1[i])
         sol_omegasv005_p1_singleplot_slow_body_sausage.append(sol_omegas_v005_p1[i])

    
sol_ksv005_p1_singleplot_fast_sausage = np.array(sol_ksv005_p1_singleplot_fast_sausage)
sol_omegasv005_p1_singleplot_fast_sausage = np.array(sol_omegasv005_p1_singleplot_fast_sausage)
sol_ksv005_p1_singleplot_slow_surf_sausage = np.array(sol_ksv005_p1_singleplot_slow_surf_sausage)
sol_omegasv005_p1_singleplot_slow_surf_sausage = np.array(sol_omegasv005_p1_singleplot_slow_surf_sausage)
sol_ksv005_p1_singleplot_slow_body_sausage = np.array(sol_ksv005_p1_singleplot_slow_body_sausage)
sol_omegasv005_p1_singleplot_slow_body_sausage = np.array(sol_omegasv005_p1_singleplot_slow_body_sausage)


#####

sol_ksv005_p1_singleplot_fast_kink = []
sol_omegasv005_p1_singleplot_fast_kink = []
sol_ksv005_p1_singleplot_slow_surf_kink = []
sol_omegasv005_p1_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kink_v005_p1)):
    if sol_ks_kink_v005_p1[i] > 0.68 and sol_ks_kink_v005_p1[i] < 0.7:        
       if sol_omegas_kink_v005_p1[i]/sol_ks_kink_v005_p1[i] > c_kink and sol_omegas_kink_v005_p1[i]/sol_ks_kink_v005_p1[i] < c_e:
         sol_ksv005_p1_singleplot_fast_kink.append(sol_ks_kink_v005_p1[i])
         sol_omegasv005_p1_singleplot_fast_kink.append(sol_omegas_kink_v005_p1[i])        

    if sol_ks_kink_v005_p1[i] > 0.56 and sol_ks_kink_v005_p1[i] < 0.62:         
       if sol_omegas_kink_v005_p1[i]/sol_ks_kink_v005_p1[i] > cT_i0 and sol_omegas_kink_v005_p1[i]/sol_ks_kink_v005_p1[i] < c_i0: 
         sol_ksv005_p1_singleplot_slow_surf_kink.append(sol_ks_kink_v005_p1[i])
         sol_omegasv005_p1_singleplot_slow_surf_kink.append(sol_omegas_kink_v005_p1[i])

    
sol_ksv005_p1_singleplot_fast_kink = np.array(sol_ksv005_p1_singleplot_fast_kink)
sol_omegasv005_p1_singleplot_fast_kink = np.array(sol_omegasv005_p1_singleplot_fast_kink)
sol_ksv005_p1_singleplot_slow_surf_kink = np.array(sol_ksv005_p1_singleplot_slow_surf_kink)
sol_omegasv005_p1_singleplot_slow_surf_kink = np.array(sol_omegasv005_p1_singleplot_slow_surf_kink)


##################################

sol_ksv01_p1_singleplot_fast_sausage = []
sol_omegasv01_p1_singleplot_fast_sausage = []
sol_ksv01_p1_singleplot_slow_surf_sausage = []
sol_omegasv01_p1_singleplot_slow_surf_sausage = []
sol_ksv01_p1_singleplot_slow_body_sausage = []
sol_omegasv01_p1_singleplot_slow_body_sausage = []


for i in range(len(sol_ks_v01_p1)):
    if sol_ks_v01_p1[i] > 1.48 and sol_ks_v01_p1[i] < 1.51:        
       if sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] > c_kink and sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] < c_e:
         sol_ksv01_p1_singleplot_fast_sausage.append(sol_ks_v01_p1[i])
         sol_omegasv01_p1_singleplot_fast_sausage.append(sol_omegas_v01_p1[i])        

    if sol_ks_v01_p1[i] > 1.55 and sol_ks_v01_p1[i] < 1.6:         
       if sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] > 0.88 and sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] < cT_i0: 
         sol_ksv01_p1_singleplot_slow_surf_sausage.append(sol_ks_v01_p1[i])
         sol_omegasv01_p1_singleplot_slow_surf_sausage.append(sol_omegas_v01_p1[i])

    if sol_ks_v01_p1[i] > 1.4 and sol_ks_v01_p1[i] < 1.45:         
       if sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] > 0.908 and sol_omegas_v01_p1[i]/sol_ks_v01_p1[i] < 0.92: 
         sol_ksv01_p1_singleplot_slow_body_sausage.append(sol_ks_v01_p1[i])
         sol_omegasv01_p1_singleplot_slow_body_sausage.append(sol_omegas_v01_p1[i])

    
sol_ksv01_p1_singleplot_fast_sausage = np.array(sol_ksv01_p1_singleplot_fast_sausage)
sol_omegasv01_p1_singleplot_fast_sausage = np.array(sol_omegasv01_p1_singleplot_fast_sausage)
sol_ksv01_p1_singleplot_slow_surf_sausage = np.array(sol_ksv01_p1_singleplot_slow_surf_sausage)
sol_omegasv01_p1_singleplot_slow_surf_sausage = np.array(sol_omegasv01_p1_singleplot_slow_surf_sausage)
sol_ksv01_p1_singleplot_slow_body_sausage = np.array(sol_ksv01_p1_singleplot_slow_body_sausage)
sol_omegasv01_p1_singleplot_slow_body_sausage = np.array(sol_omegasv01_p1_singleplot_slow_body_sausage)

#####


sol_ksv01_p1_singleplot_fast_kink = []
sol_omegasv01_p1_singleplot_fast_kink = []
sol_ksv01_p1_singleplot_slow_surf_kink = []
sol_omegasv01_p1_singleplot_slow_surf_kink = []
sol_ksv01_p1_singleplot_body_kink = []
sol_omegasv01_p1_singleplot_body_kink = []


for i in range(len(sol_ks_kink_v01_p1)):
    if sol_ks_kink_v01_p1[i] > 0.68 and sol_ks_kink_v01_p1[i] < 0.7:        
       if sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] > c_kink and sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] < c_e:
         sol_ksv01_p1_singleplot_fast_kink.append(sol_ks_kink_v01_p1[i])
         sol_omegasv01_p1_singleplot_fast_kink.append(sol_omegas_kink_v01_p1[i])        

    if sol_ks_kink_v01_p1[i] > 1.65 and sol_ks_kink_v01_p1[i] < 1.7:         
       if sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] > cT_i0 and sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] < 0.96: 
         sol_ksv01_p1_singleplot_slow_surf_kink.append(sol_ks_kink_v01_p1[i])
         sol_omegasv01_p1_singleplot_slow_surf_kink.append(sol_omegas_kink_v01_p1[i])

    if sol_ks_kink_v01_p1[i] > 1.65 and sol_ks_kink_v01_p1[i] < 1.7:         
       if sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] > 0.96 and sol_omegas_kink_v01_p1[i]/sol_ks_kink_v01_p1[i] < 0.97: 
         sol_ksv01_p1_singleplot_body_kink.append(sol_ks_kink_v01_p1[i])
         sol_omegasv01_p1_singleplot_body_kink.append(sol_omegas_kink_v01_p1[i])

    
sol_ksv01_p1_singleplot_fast_kink = np.array(sol_ksv01_p1_singleplot_fast_kink)
sol_omegasv01_p1_singleplot_fast_kink = np.array(sol_omegasv01_p1_singleplot_fast_kink)
sol_ksv01_p1_singleplot_slow_surf_kink = np.array(sol_ksv01_p1_singleplot_slow_surf_kink)
sol_omegasv01_p1_singleplot_slow_surf_kink = np.array(sol_omegasv01_p1_singleplot_slow_surf_kink)
sol_ksv01_p1_singleplot_body_kink = np.array(sol_ksv01_p1_singleplot_body_kink)
sol_omegasv01_p1_singleplot_body_kink = np.array(sol_omegasv01_p1_singleplot_body_kink)


##################################

sol_ksv015_p1_singleplot_fast_sausage = []
sol_omegasv015_p1_singleplot_fast_sausage = []
sol_ksv015_p1_singleplot_slow_surf_sausage = []
sol_omegasv015_p1_singleplot_slow_surf_sausage = []
sol_ksv015_p1_singleplot_slow_body_sausage = []
sol_omegasv015_p1_singleplot_slow_body_sausage = []


for i in range(len(sol_ks_v015_p1)):
    if sol_ks_v015_p1[i] > 1.48 and sol_ks_v015_p1[i] < 1.51:        
       if sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] > c_kink and sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] < c_e:
         sol_ksv015_p1_singleplot_fast_sausage.append(sol_ks_v015_p1[i])
         sol_omegasv015_p1_singleplot_fast_sausage.append(sol_omegas_v015_p1[i])        

    if sol_ks_v015_p1[i] > 1.55 and sol_ks_v015_p1[i] < 1.6:         
       if sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] > 0.88 and sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] < cT_i0: 
         sol_ksv015_p1_singleplot_slow_surf_sausage.append(sol_ks_v015_p1[i])
         sol_omegasv015_p1_singleplot_slow_surf_sausage.append(sol_omegas_v015_p1[i])

    if sol_ks_v015_p1[i] > 1.4 and sol_ks_v015_p1[i] < 1.45:         
       if sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] > 0.908 and sol_omegas_v015_p1[i]/sol_ks_v015_p1[i] < 0.92: 
         sol_ksv015_p1_singleplot_slow_body_sausage.append(sol_ks_v015_p1[i])
         sol_omegasv015_p1_singleplot_slow_body_sausage.append(sol_omegas_v015_p1[i])

    
sol_ksv015_p1_singleplot_fast_sausage = np.array(sol_ksv015_p1_singleplot_fast_sausage)
sol_omegasv015_p1_singleplot_fast_sausage = np.array(sol_omegasv015_p1_singleplot_fast_sausage)
sol_ksv015_p1_singleplot_slow_surf_sausage = np.array(sol_ksv015_p1_singleplot_slow_surf_sausage)
sol_omegasv015_p1_singleplot_slow_surf_sausage = np.array(sol_omegasv015_p1_singleplot_slow_surf_sausage)
sol_ksv015_p1_singleplot_slow_body_sausage = np.array(sol_ksv015_p1_singleplot_slow_body_sausage)
sol_omegasv015_p1_singleplot_slow_body_sausage = np.array(sol_omegasv015_p1_singleplot_slow_body_sausage)


#####

sol_ksv015_p1_singleplot_fast_kink = []
sol_omegasv015_p1_singleplot_fast_kink = []
sol_ksv015_p1_singleplot_slow_surf_kink = []
sol_omegasv015_p1_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kink_v015_p1)):
    if sol_ks_kink_v015_p1[i] > 0.68 and sol_ks_kink_v015_p1[i] < 0.7:        
       if sol_omegas_kink_v015_p1[i]/sol_ks_kink_v015_p1[i] > c_kink and sol_omegas_kink_v015_p1[i]/sol_ks_kink_v015_p1[i] < c_e:
         sol_ksv015_p1_singleplot_fast_kink.append(sol_ks_kink_v015_p1[i])
         sol_omegasv015_p1_singleplot_fast_kink.append(sol_omegas_kink_v015_p1[i])        

    if sol_ks_kink_v015_p1[i] > 0.56 and sol_ks_kink_v015_p1[i] < 0.62:         
       if sol_omegas_kink_v015_p1[i]/sol_ks_kink_v015_p1[i] > c_i0 and sol_omegas_kink_v015_p1[i]/sol_ks_kink_v015_p1[i] < c_kink: 
         sol_ksv015_p1_singleplot_slow_surf_kink.append(sol_ks_kink_v015_p1[i])
         sol_omegasv015_p1_singleplot_slow_surf_kink.append(sol_omegas_kink_v015_p1[i])

    
sol_ksv015_p1_singleplot_fast_kink = np.array(sol_ksv015_p1_singleplot_fast_kink)
sol_omegasv015_p1_singleplot_fast_kink = np.array(sol_omegasv015_p1_singleplot_fast_kink)
sol_ksv015_p1_singleplot_slow_surf_kink = np.array(sol_ksv015_p1_singleplot_slow_surf_kink)
sol_omegasv015_p1_singleplot_slow_surf_kink = np.array(sol_omegasv015_p1_singleplot_slow_surf_kink)


##################################

sol_ksv025_p1_singleplot_fast_sausage = []
sol_omegasv025_p1_singleplot_fast_sausage = []
sol_ksv025_p1_singleplot_slow_surf_sausage = []
sol_omegasv025_p1_singleplot_slow_surf_sausage = []
sol_ksv025_p1_singleplot_slow_body_sausage = []
sol_omegasv025_p1_singleplot_slow_body_sausage = []


for i in range(len(sol_ks_v025_p1)):
    if sol_ks_v025_p1[i] > 1.48 and sol_ks_v025_p1[i] < 1.51:        
       if sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] > c_kink and sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] < c_e:
         sol_ksv025_p1_singleplot_fast_sausage.append(sol_ks_v025_p1[i])
         sol_omegasv025_p1_singleplot_fast_sausage.append(sol_omegas_v025_p1[i])        

    if sol_ks_v025_p1[i] > 1.55 and sol_ks_v025_p1[i] < 1.6:         
       if sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] > 0.88 and sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] < cT_i0: 
         sol_ksv025_p1_singleplot_slow_surf_sausage.append(sol_ks_v025_p1[i])
         sol_omegasv025_p1_singleplot_slow_surf_sausage.append(sol_omegas_v025_p1[i])

    if sol_ks_v025_p1[i] > 1.4 and sol_ks_v025_p1[i] < 1.45:         
       if sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] > 0.908 and sol_omegas_v025_p1[i]/sol_ks_v025_p1[i] < 0.92: 
         sol_ksv025_p1_singleplot_slow_body_sausage.append(sol_ks_v025_p1[i])
         sol_omegasv025_p1_singleplot_slow_body_sausage.append(sol_omegas_v025_p1[i])

    
sol_ksv025_p1_singleplot_fast_sausage = np.array(sol_ksv025_p1_singleplot_fast_sausage)
sol_omegasv025_p1_singleplot_fast_sausage = np.array(sol_omegasv025_p1_singleplot_fast_sausage)
sol_ksv025_p1_singleplot_slow_surf_sausage = np.array(sol_ksv025_p1_singleplot_slow_surf_sausage)
sol_omegasv025_p1_singleplot_slow_surf_sausage = np.array(sol_omegasv025_p1_singleplot_slow_surf_sausage)
sol_ksv025_p1_singleplot_slow_body_sausage = np.array(sol_ksv025_p1_singleplot_slow_body_sausage)
sol_omegasv025_p1_singleplot_slow_body_sausage = np.array(sol_omegasv025_p1_singleplot_slow_body_sausage)


#####

sol_ksv025_p1_singleplot_fast_kink = []
sol_omegasv025_p1_singleplot_fast_kink = []
sol_ksv025_p1_singleplot_slow_surf_kink = []
sol_omegasv025_p1_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kink_v025_p1)):
    if sol_ks_kink_v025_p1[i] > 0.68 and sol_ks_kink_v025_p1[i] < 0.7:        
       if sol_omegas_kink_v025_p1[i]/sol_ks_kink_v025_p1[i] > c_kink and sol_omegas_kink_v025_p1[i]/sol_ks_kink_v025_p1[i] < c_e:
         sol_ksv025_p1_singleplot_fast_kink.append(sol_ks_kink_v025_p1[i])
         sol_omegasv025_p1_singleplot_fast_kink.append(sol_omegas_kink_v025_p1[i])        

    if sol_ks_kink_v025_p1[i] > 0.56 and sol_ks_kink_v025_p1[i] < 0.62:         
       if sol_omegas_kink_v025_p1[i]/sol_ks_kink_v025_p1[i] > c_i0 and sol_omegas_kink_v025_p1[i]/sol_ks_kink_v025_p1[i] < c_kink: 
         sol_ksv025_p1_singleplot_slow_surf_kink.append(sol_ks_kink_v025_p1[i])
         sol_omegasv025_p1_singleplot_slow_surf_kink.append(sol_omegas_kink_v025_p1[i])

    
sol_ksv025_p1_singleplot_fast_kink = np.array(sol_ksv025_p1_singleplot_fast_kink)
sol_omegasv025_p1_singleplot_fast_kink = np.array(sol_omegasv025_p1_singleplot_fast_kink)
sol_ksv025_p1_singleplot_slow_surf_kink = np.array(sol_ksv025_p1_singleplot_slow_surf_kink)
sol_omegasv025_p1_singleplot_slow_surf_kink = np.array(sol_omegasv025_p1_singleplot_slow_surf_kink)



###      kink

#plt.figure(figsize=(14,18))
plt.figure()
ax = plt.subplot(111)
#ax.set_title(r'$\propto r^{0.8}$')
ax.set_title("sausage")

ax.set_xlabel("$ka$", fontsize=16)
ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.set_xlim(0., 4.)  
ax.set_ylim(0.85, c_e+0.01)  


### power 1

#ax.plot(sol_ks_kink_v001_p1, (sol_omegas_kink_v001_p1/sol_ks_kink_v001_p1), 'g.', markersize=4.)
#ax.plot(sol_ks_kink_v005_p1, (sol_omegas_kink_v005_p1/sol_ks_kink_v005_p1), 'k.', markersize=4.)
ax.plot(sol_ks_kink_v01_p1, (sol_omegas_kink_v01_p1/sol_ks_kink_v01_p1), 'b.', markersize=4.)
#ax.plot(sol_ks_kink_v015_p1, (sol_omegas_kink_v015_p1/sol_ks_kink_v015_p1), 'r.', markersize=4.)
#ax.plot(sol_ks_kink_v025_p1, (sol_omegas_kink_v025_p1/sol_ks_kink_v025_p1), 'b.', markersize=4.)

#ax.plot(sol_ksv001_p1_singleplot_fast_kink, (sol_omegasv001_p1_singleplot_fast_kink/sol_ksv001_p1_singleplot_fast_kink), 'g.', markersize=15.)
#ax.plot(sol_ksv005_p1_singleplot_fast_kink, (sol_omegasv005_p1_singleplot_fast_kink/sol_ksv005_p1_singleplot_fast_kink), 'k.', markersize=15.)
#ax.plot(sol_ksv01_p1_singleplot_fast_kink, (sol_omegasv01_p1_singleplot_fast_kink/sol_ksv01_p1_singleplot_fast_kink), 'y.', markersize=15.)
#ax.plot(sol_ksv015_p1_singleplot_fast_kink, (sol_omegasv015_p1_singleplot_fast_kink/sol_ksv015_p1_singleplot_fast_kink), 'r.', markersize=15.)
#ax.plot(sol_ksv025_p1_singleplot_fast_kink, (sol_omegasv025_p1_singleplot_fast_kink/sol_ksv025_p1_singleplot_fast_kink), 'b.', markersize=15.)
#
#ax.plot(sol_ksv001_p1_singleplot_slow_surf_kink, (sol_omegasv001_p1_singleplot_slow_surf_kink/sol_ksv001_p1_singleplot_slow_surf_kink), 'g.', markersize=15.)
#ax.plot(sol_ksv005_p1_singleplot_slow_surf_kink, (sol_omegasv005_p1_singleplot_slow_surf_kink/sol_ksv005_p1_singleplot_slow_surf_kink), 'k.', markersize=15.)
ax.plot(sol_ksv01_p1_singleplot_slow_surf_kink, (sol_omegasv01_p1_singleplot_slow_surf_kink/sol_ksv01_p1_singleplot_slow_surf_kink), 'k.', markersize=15.)
#ax.plot(sol_ksv015_p1_singleplot_slow_surf_kink, (sol_omegasv015_p1_singleplot_slow_surf_kink/sol_ksv015_p1_singleplot_slow_surf_kink), 'r.', markersize=15.)
#ax.plot(sol_ksv025_p1_singleplot_slow_surf_kink, (sol_omegasv025_p1_singleplot_slow_surf_kink/sol_ksv025_p1_singleplot_slow_surf_kink), 'b.', markersize=15.)

#ax.plot(sol_ksv001_p1_singleplot_slow_body_kink, (sol_omegasv001_p1_singleplot_slow_body_kink/sol_ksv001_p1_singleplot_slow_body_kink), 'g.', markersize=15.)
#ax.plot(sol_ksv005_p1_singleplot_slow_body_kink, (sol_omegasv005_p1_singleplot_slow_body_kink/sol_ksv005_p1_singleplot_slow_body_kink), 'k.', markersize=15.)
#ax.plot(sol_ksv01_p1_singleplot_slow_body_kink, (sol_omegasv01_p1_singleplot_slow_body_kink/sol_ksv01_p1_singleplot_slow_body_kink), 'y.', markersize=15.)
#ax.plot(sol_ksv015_p1_singleplot_slow_body_kink, (sol_omegasv015_p1_singleplot_slow_body_kink/sol_ksv015_p1_singleplot_slow_body_kink), 'r.', markersize=15.)
#ax.plot(sol_ksv025_p1_singleplot_slow_body_kink, (sol_omegasv025_p1_singleplot_slow_body_kink/sol_ksv025_p1_singleplot_slow_body_kink), 'b.', markersize=15.)

test_k = np.linspace(0.001, 4., 500.)
amp =0.1
test_r=1.
alpha=1.
ax.fill_between(test_k, cT_i0+((amp*(0.002**(alpha-1)))/(test_k)), cT_i0+((amp*(test_r**(alpha-1)))/(test_k)), color='green', alpha=0.4)  


#plt.show()
#exit()

###################     KINK     ###############################


####################################################

#wavenum = sol_ksv001_p1_singleplot_fast_kink[0]
#frequency = sol_omegasv001_p1_singleplot_fast_kink[0]
 
wavenum = sol_ksv01_p1_singleplot_slow_surf_kink[0]
frequency = sol_omegasv01_p1_singleplot_slow_surf_kink[0]

k = wavenum
w = frequency
#####################################################

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

m = 1.

B_twist = 0.
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy

def B_i(r):  
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))

B_i_np=sym.lambdify(rr,B_i(rr),"numpy")


def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")


v_twist = 0.1
def v_iphi(r):  
    return v_twist*(r**1.) 

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


lx = np.linspace(2.*2.*np.pi/wavenum, 1., 3500.)  # Number of wavelengths/2*pi accomodated in the domain   #1.75 before
  
m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
   
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z)
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

def D(r):  
  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")


def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_iphi(r)**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*v_iphi(r)*((m*B_iphi(r)/r)+(wavenum*B_i(r)))/r))

Q_np=sym.lambdify(rr,Q(rr),"numpy")

def T(r):
  return ((((m*B_iphi(r)/r)+(wavenum*B_i(r)))*B_iphi(r)) + rho_i(r)*v_iphi(r)*shift_freq(r))

T_np=sym.lambdify(rr,T(rr),"numpy")

def C1(r):
  return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
   
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))

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

def dP_dr_e(P_e, r_e):
       return [P_e[1], (-P_e[1]/r_e + (m_e+((1.**2)/(r_e**2)))*P_e[0])]
  
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
      u1 = U[:,0] + ((B_iphi_np(1)**2) - (rho_i_np(1)*v_iphi_np(1)**2))*left_xi_solution[-1]
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

#inside_xi_solution = xi_i_const*Is[:,1] #+ Is[:,0]/ix)   # Pressure perturbation solution for left hand side
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))


#########

spatial_v025_p1 = np.concatenate((ix[::-1], lx[::-1]), axis=None)     

radial_displacement_v025_p1 = np.concatenate((inside_xi_solution[::-1], left_xi_solution[::-1]), axis=None)  
radial_PT_v025_p1 = np.concatenate((inside_P_solution[::-1], left_P_solution[::-1]), axis=None)  

normalised_radial_displacement_v025_p1 = np.concatenate((normalised_inside_xi_solution[::-1], normalised_left_xi_solution[::-1]), axis=None)  
normalised_radial_PT_v025_p1 = np.concatenate((normalised_inside_P_solution[::-1], normalised_left_P_solution[::-1]), axis=None)  


######  v_r   #######

outside_v_r = -frequency*left_xi_solution[::-1]
inside_v_r = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]

radial_vr_v025_p1 = np.concatenate((inside_v_r, outside_v_r), axis=None)    

normalised_outside_v_r = -w*normalised_left_xi_solution[::-1]
normalised_inside_v_r = -shift_freq_np(ix[::-1])*normalised_inside_xi_solution[::-1]

normalised_radial_vr_v025_p1 = np.concatenate((normalised_inside_v_r, normalised_outside_v_r), axis=None)    


#####

inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-(k**2*cT_e**2))*(c_e**2+vA_e**2))
radial_xi_z_v025_p1 = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 



inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - (k**2*vA_e**2)))
radial_xi_phi_v025_p1 = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    

normalised_inside_xi_phi = inside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_outside_xi_phi = outside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_radial_xi_phi_v025_p1 = np.concatenate((normalised_inside_xi_phi, normalised_outside_xi_phi), axis=None)    

######

######  v_phi   #######

def dv_phi(r):
  return sym.diff(v_iphi(r)/r)

dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")

outside_v_phi = -w*outside_xi_phi
inside_v_phi = -(shift_freq_np(ix[::-1])*inside_xi_phi) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1])

radial_v_phi_v025_p1 = np.concatenate((inside_v_phi, outside_v_phi), axis=None)    

normalised_inside_v_phi = inside_v_phi/np.amax(abs(outside_v_phi))
normalised_outside_v_phi = outside_v_phi/np.amax(abs(outside_v_phi))
normalised_radial_v_phi_v025_p1 = np.concatenate((normalised_inside_v_phi, normalised_outside_v_phi), axis=None)    


######  v_z   #######
U_e = 0.
U_i0 = 0.
dr=1e5
def v_iz(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2))) 

def dv_z(r):
  return sym.diff(v_iz(r)/r)

dv_z_np=sym.lambdify(rr,dv_z(rr),"numpy")

outside_v_z = -w*outside_xi_z
inside_v_z = -(shift_freq_np(ix[::-1])*inside_xi_z) - (dv_z_np(ix[::-1])*inside_xi_solution[::-1])

radial_v_z_v025_p1 = np.concatenate((inside_v_z, outside_v_z), axis=None)    

######################

#####  background

v_twist_01 = 0.1

def v_iphi_01(r):  
    return v_twist_01*(r**1.)   #0.

v_iphi_01_np=sym.lambdify(rr,v_iphi_01(rr),"numpy")   #In order to evaluate we need to switch to numpy

outisde_bkg_v_phi_01 = np.zeros(len(lx))

bkg_v_iphi_01 = np.concatenate((v_iphi_01_np(ix[::-1]), outisde_bkg_v_phi_01), axis=None)    




fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False)
ax.set_title('kink') 
ax.axvline(x=B, color='r', linestyle='--')
ax.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0.001, 2.)

ax.plot(spatial_v025_p1, radial_PT_v025_p1, 'k')

ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
#ax2.set_ylabel(r"$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0.001, 2.)

ax2.plot(spatial_v025_p1, radial_displacement_v025_p1, 'k')
#ax2.plot(spatial, normalised_radial_displacement_005, 'k')

ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_xlabel("$r$", fontsize=18)
#ax3.set_ylabel(r"$\hat{\u03be}_{\varphi}$", fontsize=18, rotation=0, labelpad=15)
ax3.set_xlim(0.001, 2.)

ax3.plot(spatial_v025_p1, radial_xi_phi_v025_p1, 'k')


#plt.show()
#exit()




spatial=spatial_v025_p1
########################################
time = np.linspace(0.01, 2.*np.pi, 80.)  # test

step = (max(time)-min(time))/len(time)

wavenum = k

z = np.linspace(0.01, 5., 6.)   #21  #31
THETA = np.linspace(0., 2.*np.pi, 50) 

radii, thetas, Z = np.meshgrid(spatial,THETA,z,sparse=False, indexing='ij')

print(Z.shape)

###########

xi_r = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_phi = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_z = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
PT = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_r = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_phi =  np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_z = np.zeros(((len(time), len(spatial), len(THETA), len(z))))


bkg_vel_x_01 = np.zeros(((len(spatial), len(THETA), len(z))))
bkg_vel_y_01 = np.zeros(((len(spatial), len(THETA), len(z))))


xi_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
xi_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
v_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
P_x = np.zeros(((len(time), len(spatial), len(THETA), len(z))))
P_y = np.zeros(((len(time), len(spatial), len(THETA), len(z))))


boundary_point_r = np.zeros(((len(spatial), len(THETA), len(z))))

bound_index = np.where(radii[:,0,0] == 1.)


for k in range(len(z)):
  for j in range(len(THETA)):
    for i in range(len(spatial)):
      for t in range(len(time)):
        xi_r[t,i,j,k] = radial_displacement_v025_p1[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t])  
        xi_phi[t,i,j,k] = radial_xi_phi_v025_p1[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t])        
        #xi_z[t,i,j,k] = radial_xi_z[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k])*np.sin(w*time[t])
        PT[t,i,j,k] = radial_PT_v025_p1[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]) 
        v_r[t,i,j,k] = 3e5*(radial_vr_v025_p1[i]*np.cos(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]))
        v_phi[t,i,j,k] = 3e5*(radial_v_phi_v025_p1[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]))
        v_z[t,i,j,k] = 3e5*(radial_v_z_v025_p1[i]*-np.sin(m*thetas[i,j,k])*np.cos(wavenum*Z[i,j,k] - w*time[t]))

        bkg_vel_x_01[i,j,k] = 5.*bkg_v_iphi_01[i]*-np.sin(thetas[i,j,k])  #v_iphi_005_np(ix)[i]*-np.sin(thetas[i,j,k])
        bkg_vel_y_01[i,j,k] = 5.*bkg_v_iphi_01[i]*np.cos(thetas[i,j,k])  #v_iphi_005_np(ix)[i]*np.cos(thetas[i,j,k]) 

        xi_x[t,i,j,k] = (xi_r[t,i,j,k]*np.cos(thetas[i,j,k]) - xi_phi[t,i,j,k]*np.sin(thetas[i,j,k]))
        xi_y[t,i,j,k] = (xi_r[t,i,j,k]*np.sin(thetas[i,j,k]) + xi_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        v_x[t,i,j,k] = (v_r[t,i,j,k]*np.cos(thetas[i,j,k]) - v_phi[t,i,j,k]*np.sin(thetas[i,j,k])) 
        v_y[t,i,j,k] = (v_r[t,i,j,k]*np.sin(thetas[i,j,k]) + v_phi[t,i,j,k]*np.cos(thetas[i,j,k]))
        P_x[t,i,j,k] = PT[t,i,j,k]*np.cos(thetas[i,j,k])
        P_y[t,i,j,k] = PT[t,i,j,k]*np.sin(thetas[i,j,k])
               
           
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

#################################################
print(x.shape)
print(y.shape)
print(z.shape)

el = 3
snapshot = 4  #14


print('v_x  =', v_x[snapshot,:,:,el])


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
#new_x = np.linspace(-np.pi/wavenum, np.pi/wavenum, 700)
#new_y = np.linspace(-np.pi/wavenum, np.pi/wavenum, 700)  
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
##print('vx_interp  =', vx_interp[::5,::5])
#
##ax.plot(new_xx, new_yy, 'k.', markersize=5)
#
#contour1 = ax.contourf(x[:,:,el], y[:,:,el], PT[snapshot,:,:,el], extend='both', cmap='bwr', alpha=0.3)  
#ax.scatter(boundary_point_x[:,el], boundary_point_y[:,el], s=4., c='blue')
#ax.plot(boundary_point_x[:,el], boundary_point_y[:,el])     
#ax.quiver(new_xx[::5,::5], new_yy[::5,::5], vx_interp[::5,::5], vy_interp[::5,::5], pivot='tail', color='black', scale_units='inches', scale=15., width=0.003)
##ax.quiver(new_xx, new_yy, vx_interp, vy_interp, pivot='tail', color='black', scale_units='inches', scale=15., width=0.003)
#
#fig.colorbar(contour1, ax=ax)
#
###########################
#
#ax2.plot(a,b, 'r--')
#
#ax2.set_title('Velocity+bkg')
#ax2.set_xlim(-1.5, 1.5)
#ax2.set_ylim(-1.5, 1.5)
#ax2.set_xlabel('$x$', fontsize=18)
#ax2.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)
#
#new_x = np.linspace(-np.pi/wavenum, np.pi/wavenum, 700)
#new_y = np.linspace(-np.pi/wavenum, np.pi/wavenum, 700)  
#
#new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')
#
#flat_x = x[:,:,el].flatten('C')
#flat_y = y[:,:,el].flatten('C')
#    
#points = np.transpose(np.vstack((flat_x, flat_y)))
#
#boundary_point_x_full = boundary_point_r*np.cos(thetas[bound_element,:,:]) + (v_x[snapshot,bound_element,:,:] + bkg_vel_x_025[bound_element,:,:])*step  
#boundary_point_y_full = boundary_point_r*np.sin(thetas[bound_element,:,:]) + (v_y[snapshot,bound_element,:,:] + bkg_vel_y_025[bound_element,:,:])*step   
#
#boundary_point_x_full = boundary_point_x_full[0,:,:]
#boundary_point_y_full = boundary_point_y_full[0,:,:]
#
#flat_vx_full = (v_x[snapshot,:,:,el] + bkg_vel_x_025[:,:,el]).flatten('C')
#flat_vy_full = (v_y[snapshot,:,:,el] + bkg_vel_y_025[:,:,el]).flatten('C')
#
#vx_interp_full = griddata(points, flat_vx_full, (new_xx, new_yy), method='cubic')
#vy_interp_full = griddata(points, flat_vy_full, (new_xx, new_yy), method='cubic')
#
##print('vx_interp  =', vx_interp[::5,::5])
#
##ax.plot(new_xx, new_yy, 'k.', markersize=5)
#
#contour2 = ax.contourf(x[:,:,el], y[:,:,el], PT[snapshot,:,:,el], extend='both', cmap='bwr', alpha=0.3)  
#ax2.scatter(boundary_point_x_full[:,el], boundary_point_y_full[:,el], s=4., c='blue')
#ax2.plot(boundary_point_x_full[:,el], boundary_point_y_full[:,el])     
#ax2.quiver(new_xx[::5,::5], new_yy[::5,::5], vx_interp_full[::5,::5], vy_interp_full[::5,::5], pivot='tail', color='black', scale_units='inches', scale=15., width=0.003)
#
#fig.colorbar(contour2, ax=ax2)
#
#
#
#
#
#plt.show()
#exit()



#########################

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, bitrate=4000)   #9fps if skipping 5 in time step   15 good


######   interpolate

fig = plt.figure(figsize=(20,8))  # try 22,8     #16,8 too narrow     
ax = plt.subplot(121)
ax2 = plt.subplot(122)

div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes('right', '5%', '5%')

circle_rad =1.
a = circle_rad*np.cos(THETA) #circle x
b = circle_rad*np.sin(THETA) #circle y


#############################################

def animate(i):     
   ax.clear()
   ax2.clear()
   
   ax.set_title(' $A = 0.1  -  p = 1  - kink - k = 1.65$ - t = %d' %i)
   ax.plot(a,b, 'r--')
   ax2.plot(a,b, 'r--')
   ax2.set_title(' Pert + Bkg Velocity - t = %d' %i)
   
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

######   
   new_x = np.linspace(-np.pi/wavenum, np.pi/wavenum, 800)
   new_y = np.linspace(-np.pi/wavenum, np.pi/wavenum, 800) # test
  
   new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')

   el = 3
      
   flat_x = x[:,:,el].flatten('C')
   flat_y = y[:,:,el].flatten('C')
       
   points = np.transpose(np.vstack((flat_x, flat_y)))
    
#####
    
   flat_vx = v_x[i,:,:,el].flatten('C')
   flat_vy = v_y[i,:,:,el].flatten('C')
   
   vx_interp = griddata(points, flat_vx, (new_xx, new_yy), method='cubic')
   vy_interp = griddata(points, flat_vy, (new_xx, new_yy), method='cubic')
   
   contour1 = ax.contourf(x[:,:,el], y[:,:,el], PT[i,:,:,el], extend='both', cmap='bwr', alpha=0.2)  
   ax.scatter(boundary_point_x[:,el], boundary_point_y[:,el], s=4., c='blue')
   ax.plot(boundary_point_x[:,el], boundary_point_y[:,el])     
   ax.quiver(new_xx[::18,::18], new_yy[::18,::18], vx_interp[::18,::18], vy_interp[::18,::18], pivot='tail', color='black', scale_units='inches', scale=5., width=0.003)   # 7 scale 

   cax.cla()
   fig.colorbar(contour1, cax=cax)


####################################################

   boundary_point_x_full = boundary_point_r*np.cos(thetas[bound_element,:,:]) + (v_x[i,bound_element,:,:] + bkg_vel_x_01[bound_element,:,:])*step    
   boundary_point_y_full = boundary_point_r*np.sin(thetas[bound_element,:,:]) + (v_y[i,bound_element,:,:] + bkg_vel_y_01[bound_element,:,:])*step    
 
   boundary_point_x_full = boundary_point_x_full[0,:,:]
   boundary_point_y_full = boundary_point_y_full[0,:,:]

   new_x = np.linspace(-np.pi/wavenum, np.pi/wavenum, 800)
   new_y = np.linspace(-np.pi/wavenum, np.pi/wavenum, 800) # test
  
   new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')

   el = 3
      
   flat_x = x[:,:,el].flatten('C')
   flat_y = y[:,:,el].flatten('C')
       
   points = np.transpose(np.vstack((flat_x, flat_y)))

    
   flat_vx_full = (v_x[i,:,:,el] + bkg_vel_x_01[:,:,el]).flatten('C')
   flat_vy_full = (v_y[i,:,:,el] + bkg_vel_y_01[:,:,el]).flatten('C')
   
   vx_interp_full = griddata(points, flat_vx_full, (new_xx, new_yy), method='cubic')
   vy_interp_full = griddata(points, flat_vy_full, (new_xx, new_yy), method='cubic')
   
   contour2 = ax2.contourf(x[:,:,el], y[:,:,el], PT[i,:,:,el], extend='both', cmap='bwr', alpha=0.2)  
   ax2.scatter(boundary_point_x_full[:,el], boundary_point_y_full[:,el], s=4., c='blue')
   ax2.plot(boundary_point_x_full[:,el], boundary_point_y_full[:,el])     
   ax2.quiver(new_xx[::18,::18], new_yy[::18,::18], vx_interp_full[::18,::18], vy_interp_full[::18,::18], pivot='tail', color='black', scale_units='inches', scale=5., width=0.003)

   cax2.cla()
   fig.colorbar(contour2, cax=cax2)

####################################################

anim = FuncAnimation(fig, animate, interval=100, frames=len(time)-1).save('twisted_flow_v01_p1_slow_surf_kink_vel_full_field.mp4', writer=writer)

exit()

