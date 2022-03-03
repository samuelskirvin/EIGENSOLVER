# Import the required modules
import numpy as np
import scipy as sc
import sympy as sym
#import matplotlib; matplotlib.use('agg') ##comment out to show figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pickle
import itertools
import numpy.polynomial.polynomial as poly
from scipy.integrate import odeint
from scipy.optimize import fsolve
import cmath
from matplotlib import animation
import time

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


###############



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
     
##################

####  Mihai case symmetric superalfvenic
vA_i = 1.
c_i = 1.3*vA_i    #coronal   #2.*vA_i/3.  photospheric 
vA_e = 0.*vA_i   #coronal   #0.  photospheric 

gamma=5./3.
rho_i = 9.
rho_e = 5.

c_e = np.sqrt((rho_i/rho_e)*c_i**2+gamma*0.5*vA_i**2)

#rho_e = rho_i*(c_i**2+gamma*0.5*vA_i**2)/(c_e**2+gamma*0.5*vA_e**2)
#c_e = 0.2*vA_i    #coronal    #3.*vA_i/4  photospheric 

U_i0 = 1.4*vA_i     #0.35*vA_i  coronal    
U_e = 0.          #-0.15*vA_i   photospheric      0 - coronal



########################################

#vA_i = 1.
#c_i = 1.3*vA_i    #coronal   #2.*vA_i/3.  photospheric 
#vA_e = 2.5*vA_i   #coronal   #0.  photospheric 
#c_e = 0.2*vA_i    #coronal    #3.*vA_i/4  photospheric 
#
#U_i0 = 1.1*vA_i     #0.35*vA_i  coronal    
#U_e = 0.          #-0.15*vA_i   photospheric      0 - coronal
#
#gamma=5./3.
#
#rho_i = 1.
#rho_e = rho_i*(c_i**2+gamma*0.5*vA_i**2)/(c_e**2+gamma*0.5*vA_e**2)
#
##rho_e = 2.27*rho_i          #2.27*rho_i  photospheric
##print('external density    =', rho_e)
#
R1 = rho_e/rho_i
#R1 = 0.8

print('Density ration external/internal    =', R1)


def cT_i():
    return np.sqrt(c_i**2 * vA_i**2 / (c_i**2 + vA_i**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))


a_alfven_i = 1.
a_alfven_e = vA_e/vA_i

a_sound_i = c_i/vA_i
a_sound_e = c_e/vA_i

mach_i0 = U_i0  #/vA_i
mach_e = U_e  #/vA_i

ix = np.linspace(-1, 1, 500)  # inside slab x values

x0=0.  #mean
dx=1e5  #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols


def U_i(x):                  # Define the internal alfven speed
    #return vA_i0*profile(xx)
    return (U_e + ((U_i0 - U_e)*sym.exp(-(x-x0)**2/dx**2)))   #make 0.5*

U_i_np=sym.lambdify(xx,U_i(xx),"numpy")   #In order to evaluate we need to switch to numpy

mach_i = U_i_np(ix)  #/vA_i
mach_e = U_e  #/vA_i

flow_bound = U_i_np(1.)


def dU_i(x):   #First derivative of profile in symbols    
    return sym.diff(U_i(x), x)  

dU_i_np=sym.lambdify(xx,dU_i(xx),"numpy")

def ddU_i(x):   #First derivative of profile in symbols    
    return sym.diff(dU_i(x), x)  

ddU_i_np=sym.lambdify(xx,ddU_i(xx),"numpy")


plt.figure()
plt.xlabel("x")
plt.ylabel("$U_{i}$")
ax = plt.subplot(111)
ax.plot(ix,U_i_np(ix));
ax.annotate( '$U_{e}$', xy=(1, U_e))
ax.annotate( '$U_{i}$', xy=(1, U_i0))
ax.axhline(y=U_i0, color='k', label='$U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label='$U_{e}$', linestyle='dashdot')

#plt.show()
#exit()

Kmax = 4.5

########   READ IN VARIABLES    #########

#with open('flow_width35_coronal.pickle', 'rb') as f:
#    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)


#with open('flow_complex_coronal_stablemodes.pickle', 'rb') as f:
#    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)

#with open('flow_complex_coronal_REALSOLS_005.pickle', 'rb') as f:
#    sol_omegas1_005, sol_ks1_005, sol_omegas_kink1_005, sol_ks_kink1_005 = pickle.load(f)   
#
#with open('flow_complex_coronal_REALSOLS_001.pickle', 'rb') as f:
#    sol_omegas1_001, sol_ks1_001, sol_omegas_kink1_001, sol_ks_kink1_001 = pickle.load(f)
#
#with open('flow_complex_coronal_REALSOLS_01.pickle', 'rb') as f:
#    sol_omegas1_01, sol_ks1_01, sol_omegas_kink1_01, sol_ks_kink1_01 = pickle.load(f)
#
with open('flow_complex_coronal_stablemodes.pickle', 'rb') as f:
    sol_omegas1_real, sol_ks1_real, sol_omegas_kink1_real, sol_ks_kink1_real = pickle.load(f)
#



with open('mihai_imaginary_pure_imag_iic.pickle', 'rb') as f:
    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1, sol_omegas1_imag, sol_ks1_imag, sol_omegas_kink1_imag, sol_ks_kink1_imag = pickle.load(f)


#with open('mihai_imaginary_small_imag_no_iic.pickle', 'rb') as f:
#    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1, sol_omegas1_imag, sol_ks1_imag, sol_omegas_kink1_imag, sol_ks_kink1_imag = pickle.load(f)


print(len(sol_omegas1))
print(len(sol_ks1))
print(len(sol_omegas_kink1))
print(len(sol_ks_kink1))

print(len(sol_omegas1_imag))
print(len(sol_ks1_imag))
print(len(sol_omegas_kink1_imag))
print(len(sol_ks_kink1_imag))

### SORT ARRAYS IN ORDER OF click_k[i] ###
#sol_omegas1_005 = [x for _,x in sorted(zip(sol_ks1_005,sol_omegas1_005))]
#sol_ks1_005 = np.sort(sol_ks1_005)
#
#sol_omegas1_005 = np.array(sol_omegas1_005)
#sol_ks1_005 = np.array(sol_ks1_005)
#
#sol_omegas_kink1_005 = [x for _,x in sorted(zip(sol_ks_kink1_005,sol_omegas_kink1_005))]
#sol_ks_kink1_005 = np.sort(sol_ks_kink1_005)
#
#sol_omegas_kink1_005 = np.array(sol_omegas_kink1_005)
#sol_ks_kink1_005 = np.array(sol_ks_kink1_005)

#sol_omegas1_imag_005 = [x for _,x in sorted(zip(sol_ks1_imag_005,sol_omegas1_imag_005))]
#sol_ks1_imag_005 = np.sort(sol_ks1_imag_005)
#
#sol_omegas1_imag_005 = np.array(sol_omegas1_imag_005)
#sol_ks1_imag_005 = np.array(sol_ks1_imag_005)
#
#sol_omegas_kink1_imag_005 = [x for _,x in sorted(zip(sol_ks_kink1_imag_005,sol_omegas_kink1_imag_005))]
#sol_ks_kink1_imag_005 = np.sort(sol_ks_kink1_imag_005)
#
#sol_omegas_kink1_imag_005 = np.array(sol_omegas_kink1_imag_005)
#sol_ks_kink1_imag_005 = np.array(sol_ks_kink1_imag_005)



### SORT ARRAYS IN ORDER OF click_k[i] ###

#sol_omegas1_01 = [x for _,x in sorted(zip(sol_ks1_01,sol_omegas1_01))]
#sol_ks1_01 = np.sort(sol_ks1_01)
#
#sol_omegas1_01 = np.array(sol_omegas1_01)
#sol_ks1_01 = np.array(sol_ks1_01)
#
#sol_omegas_kink1_01 = [x for _,x in sorted(zip(sol_ks_kink1_01,sol_omegas_kink1_01))]
#sol_ks_kink1_01 = np.sort(sol_ks_kink1_01)
#
#sol_omegas_kink1_01 = np.array(sol_omegas_kink1_01)
#sol_ks_kink1_01 = np.array(sol_ks_kink1_01)


#sol_omegas1_02 = [x for _,x in sorted(zip(sol_ks1_02,sol_omegas1_02))]
#sol_ks1_02 = np.sort(sol_ks1_02)
#
#sol_omegas1_02 = np.array(sol_omegas1_02)
#sol_ks1_02 = np.array(sol_ks1_02)
#
#sol_omegas_kink1_02 = [x for _,x in sorted(zip(sol_ks_kink1_02,sol_omegas_kink1_02))]
#sol_ks_kink1_02 = np.sort(sol_ks_kink1_02)
#
#sol_omegas_kink1_02 = np.array(sol_omegas_kink1_02)
#sol_ks_kink1_02 = np.array(sol_ks_kink1_02)

#sol_omegas1_imag_02 = [x for _,x in sorted(zip(sol_ks1_imag_02,sol_omegas1_imag_02))]
#sol_ks1_imag_02 = np.sort(sol_ks1_imag_02)
#
#sol_omegas1_imag_02 = np.array(sol_omegas1_imag_02)
#sol_ks1_imag_02 = np.array(sol_ks1_imag_02)
#
#sol_omegas_kink1_imag_02 = [x for _,x in sorted(zip(sol_ks_kink1_imag_02,sol_omegas_kink1_imag_02))]
#sol_ks_kink1_imag_02 = np.sort(sol_ks_kink1_imag_02)
#
#sol_omegas_kink1_imag_02 = np.array(sol_omegas_kink1_imag_02)
#sol_ks_kink1_imag_02 = np.array(sol_ks_kink1_imag_02)


### SORT ARRAYS IN ORDER OF click_k[i] ###
#sol_omegas1_001 = [x for _,x in sorted(zip(sol_ks1_001,sol_omegas1_001))]
#sol_ks1_001 = np.sort(sol_ks1_001)
#
#sol_omegas1_001 = np.array(sol_omegas1_001)
#sol_ks1_001 = np.array(sol_ks1_001)
#
#sol_omegas_kink1_001 = [x for _,x in sorted(zip(sol_ks_kink1_001,sol_omegas_kink1_001))]
#sol_ks_kink1_001 = np.sort(sol_ks_kink1_001)
#
#sol_omegas_kink1_001 = np.array(sol_omegas_kink1_001)
#sol_ks_kink1_001 = np.array(sol_ks_kink1_001)

#sol_omegas1_imag_001 = [x for _,x in sorted(zip(sol_ks1_imag_001,sol_omegas1_imag_001))]
#sol_ks1_imag_001 = np.sort(sol_ks1_imag_001)
#
#sol_omegas1_imag_001 = np.array(sol_omegas1_imag_001)
#sol_ks1_imag_001 = np.array(sol_ks1_imag_001)
#
#sol_omegas_kink1_imag_001 = [x for _,x in sorted(zip(sol_ks_kink1_imag_001,sol_omegas_kink1_imag_001))]
#sol_ks_kink1_imag_001 = np.sort(sol_ks_kink1_imag_001)
#
#sol_omegas_kink1_imag_001 = np.array(sol_omegas_kink1_imag_001)
#sol_ks_kink1_imag_001 = np.array(sol_ks_kink1_imag_001)


### SORT ARRAYS IN ORDER OF click_k[i] ###
#sol_omegas1_00 = [x for _,x in sorted(zip(sol_ks1_00,sol_omegas1_00))]
#sol_ks1_00 = np.sort(sol_ks1_00)
#
#sol_omegas1_00 = np.array(sol_omegas1_00)
#sol_ks1_00 = np.array(sol_ks1_00)
#
#sol_omegas_kink1_00 = [x for _,x in sorted(zip(sol_ks_kink1_00,sol_omegas_kink1_00))]
#sol_ks_kink1_00 = np.sort(sol_ks_kink1_00)
#
#sol_omegas_kink1_00 = np.array(sol_omegas_kink1_00)
#sol_ks_kink1_00 = np.array(sol_ks_kink1_00)

#sol_omegas1_imag_001 = [x for _,x in sorted(zip(sol_ks1_imag_001,sol_omegas1_imag_001))]
#sol_ks1_imag_001 = np.sort(sol_ks1_imag_001)
#
#sol_omegas1_imag_001 = np.array(sol_omegas1_imag_001)
#sol_ks1_imag_001 = np.array(sol_ks1_imag_001)
#
#sol_omegas_kink1_imag_001 = [x for _,x in sorted(zip(sol_ks_kink1_imag_001,sol_omegas_kink1_imag_001))]
#sol_ks_kink1_imag_001 = np.sort(sol_ks_kink1_imag_001)
#
#sol_omegas_kink1_imag_001 = np.array(sol_omegas_kink1_imag_001)
#sol_ks_kink1_imag_001 = np.array(sol_ks_kink1_imag_001)




#########################      UNCOMMENT THIS   !!!!!     ##############################
sol_omegas1 = [x for _,x in sorted(zip(sol_ks1,sol_omegas1))]
sol_ks1 = np.sort(sol_ks1)

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)

sol_omegas_kink1 = [x for _,x in sorted(zip(sol_ks_kink1,sol_omegas_kink1))]
sol_ks_kink1 = np.sort(sol_ks_kink1)

sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)

sol_omegas1_imag = [x for _,x in sorted(zip(sol_ks1_imag,sol_omegas1_imag))]
sol_ks1_imag = np.sort(sol_ks1_imag)

sol_omegas1_imag = np.array(sol_omegas1_imag)
sol_ks1_imag = np.array(sol_ks1_imag)

sol_omegas_kink1_imag = [x for _,x in sorted(zip(sol_ks_kink1_imag,sol_omegas_kink1_imag))]
sol_ks_kink1_imag = np.sort(sol_ks_kink1_imag)

sol_omegas_kink1_imag = np.array(sol_omegas_kink1_imag)
sol_ks_kink1_imag = np.array(sol_ks_kink1_imag)

############################################################################################
print(len(sol_omegas1))
print(len(sol_ks1))
print(len(sol_omegas_kink1))
print(len(sol_ks_kink1))

print(len(sol_omegas1_imag))
print(len(sol_ks1_imag))
print(len(sol_omegas_kink1_imag))
print(len(sol_ks_kink1_imag))
#####   CORONAL    ######
#
#### FAST BODY    ( vA_i  <  w/k  <  vA_e )
#fast_body_sausage_omega = []
#fast_body_sausage_k = []
#fast_body_kink_omega = []
#fast_body_kink_k = []
#
#### SLOW BODY    ( cT_i  <  w/k  <  c_i )
#slow_body_sausage_omega = []
#slow_body_sausage_k = []
#slow_body_kink_omega = []
#slow_body_kink_k = []
#
#
#### SLOW BACKWARD BODY   
#slow_backward_body_sausage_omega = []
#slow_backward_body_sausage_k = []
#slow_backward_body_kink_omega = []
#slow_backward_body_kink_k = []
#
#new_modes_sausage_k = []  
#new_modes_sausage_omega = []
#new_modes_kink_k = []
#new_modes_kink_omega = []
#
#for i in range(len(sol_ks1)):   #sausage mode
#  v_phase = sol_omegas1[i]/sol_ks1[i]
#      
#  if v_phase > vA_i and v_phase < vA_e:  
#      fast_body_sausage_omega.append(sol_omegas1[i])
#      fast_body_sausage_k.append(sol_ks1[i])
#      
#  # uncomment for W > 20
#  #if v_phase > cT_i() and v_phase < vA_i:  #vA_i0:   change after uniform
#  #    slow_body_sausage_omega.append(sol_omegas1[i])
#  #    slow_body_sausage_k.append(sol_ks1[i])
#
#  #if v_phase < cT_e(): 
#  #    slow_backward_body_sausage_omega.append(sol_omegas1[i])
#  #    slow_backward_body_sausage_k.append(sol_ks1[i])
#
#  if v_phase < flow_bound:
#      new_modes_sausage_k.append(sol_ks1[i])
#      new_modes_sausage_omega.append(sol_omegas1[i])
#      
#
#for i in range(len(sol_ks_kink1)):   #kink mode
#  v_phase = sol_omegas_kink1[i]/sol_ks_kink1[i]
#      
#  if v_phase > vA_i and v_phase < vA_e:  
#      fast_body_kink_omega.append(sol_omegas_kink1[i])
#      fast_body_kink_k.append(sol_ks_kink1[i])
#            
##  if v_phase > (c_e+U_i0) and v_phase < vA_i:
##      slow_body_kink_omega.append(sol_omegas_kink1[i])
##      slow_body_kink_k.append(sol_ks_kink1[i])
##
##  if v_phase < cT_e(): 
##      slow_backward_body_kink_omega.append(sol_omegas_kink1[i])
##      slow_backward_body_kink_k.append(sol_ks_kink1[i])
#
#  if v_phase < flow_bound:
#      new_modes_kink_k.append(sol_ks_kink1[i])
#      new_modes_kink_omega.append(sol_omegas_kink1[i])
# 
#
#fast_body_sausage_omega = np.array(fast_body_sausage_omega)
#fast_body_sausage_k = np.array(fast_body_sausage_k)
#fast_body_kink_omega = np.array(fast_body_kink_omega)
#fast_body_kink_k = np.array(fast_body_kink_k)
#
#slow_body_sausage_omega = np.array(slow_body_sausage_omega)
#slow_body_sausage_k = np.array(slow_body_sausage_k)
#slow_body_kink_omega = np.array(slow_body_kink_omega)
#slow_body_kink_k = np.array(slow_body_kink_k)
#
#slow_backward_body_sausage_omega = np.array(slow_backward_body_sausage_omega)
#slow_backward_body_sausage_k = np.array(slow_backward_body_sausage_k)
#slow_backward_body_kink_omega = np.array(slow_backward_body_kink_omega)
#slow_backward_body_kink_k = np.array(slow_backward_body_kink_k)
#           
#
#cutoff = 7.  
#cutoff2 = 8.5  
#cutoff3 = 2.5  #3.    #4.2 for W < 0.85
#cutoff4 = 6.
#fast_sausage_branch1_omega = []
#fast_sausage_branch1_k = []
#fast_sausage_branch2_omega = []
#fast_sausage_branch2_k = []
#fast_sausage_branch3_omega = []
#fast_sausage_branch3_k = []
#
#
#
#
##################################################
#for i in range(len(fast_body_sausage_omega)):   #sausage mode
#      
#  if fast_body_sausage_omega[i] > cutoff and fast_body_sausage_omega[i] < cutoff2:  
#      fast_sausage_branch1_omega.append(fast_body_sausage_omega[i])
#      fast_sausage_branch1_k.append(fast_body_sausage_k[i])
#      
#  elif fast_body_sausage_omega[i] < cutoff4 and fast_body_sausage_omega[i] > cutoff3:  
#      fast_sausage_branch2_omega.append(fast_body_sausage_omega[i])
#      fast_sausage_branch2_k.append(fast_body_sausage_k[i])
#
#     
############################################################################## 
#      
#fast_sausage_branch1_omega = np.array(fast_sausage_branch1_omega)
#fast_sausage_branch1_k = np.array(fast_sausage_branch1_k)
#fast_sausage_branch2_omega = np.array(fast_sausage_branch2_omega)
#fast_sausage_branch2_k = np.array(fast_sausage_branch2_k)
#fast_sausage_branch3_omega = np.array(fast_sausage_branch3_omega)
#fast_sausage_branch3_k = np.array(fast_sausage_branch3_k)
#
#
#cutoff_kink = 6.5 
#cutoff2_kink = 5.   # 6 for W < 1.5     #5.8 otherwise     7 for W < 0.85
#
#
###################################################   kink mode
#fast_kink_branch1_omega = []
#fast_kink_branch1_k = []
#fast_kink_branch2_omega = []
#fast_kink_branch2_k = []
#fast_kink_branch3_omega = []
#fast_kink_branch3_k = []
###################################################
#for i in range(len(fast_body_kink_omega)):   #sausage mode
#      
#  if fast_body_kink_omega[i] < cutoff_kink and fast_body_kink_omega[i] > cutoff2_kink:  
#      fast_kink_branch1_omega.append(fast_body_kink_omega[i])
#      fast_kink_branch1_k.append(fast_body_kink_k[i])
#         
#  elif  fast_body_kink_omega[i] < cutoff2_kink:  
#      fast_kink_branch2_omega.append(fast_body_kink_omega[i])
#      fast_kink_branch2_k.append(fast_body_kink_k[i])
#
#
###############################################
#index_to_remove = []
#for i in range(len(fast_kink_branch2_omega)-1):
#    ph_diff = abs((fast_kink_branch2_omega[i+1]/fast_kink_branch2_k[i+1]) - (fast_kink_branch2_omega[i]/fast_kink_branch2_k[i]))
#   
#    if ph_diff > 0.2:
#      index_to_remove.append(i+1)
#
#fast_kink_branch2_omega = np.delete(fast_kink_branch2_omega, index_to_remove)    
#fast_kink_branch2_k = np.delete(fast_kink_branch2_k, index_to_remove) 
#############################################  
#
#index_to_remove = []
#index_to_appendk = []
#index_to_appendomega = []
#for i in range(len(fast_kink_branch2_omega)):
#   if fast_kink_branch2_k[i] < 2.5 and fast_kink_branch2_omega[i] > 4.:      # if k < 2.5 and w > 4.5
#      index_to_remove.append(i)
#      
#      if fast_kink_branch2_omega[i] > 4.6:
#          index_to_appendk.append(fast_kink_branch2_k[i])
#          index_to_appendomega.append(fast_kink_branch2_omega[i])
#          #fast_kink_branch1_omega.append(fast_kink_branch2_omega[i])
#          #fast_kink_branch1_k.append(fast_kink_branch2_k[i])
#
#
#fast_kink_branch1_omega = np.append(fast_kink_branch1_omega, index_to_appendomega)    
#fast_kink_branch1_k = np.append(fast_kink_branch1_k, index_to_appendk) 
#    
#fast_kink_branch2_omega = np.delete(fast_kink_branch2_omega, index_to_remove)    
#fast_kink_branch2_k = np.delete(fast_kink_branch2_k, index_to_remove) 
#           
#fast_kink_branch1_omega = np.array(fast_kink_branch1_omega)
#fast_kink_branch1_k = np.array(fast_kink_branch1_k)
#
#fast_kink_branch1_omega = [x for _,x in sorted(zip(fast_kink_branch1_k,fast_kink_branch1_omega))]
#fast_kink_branch1_k = np.sort(fast_kink_branch1_k)
#
#
#fast_kink_branch2_omega = np.array(fast_kink_branch2_omega)
#fast_kink_branch2_k = np.array(fast_kink_branch2_k)
#fast_kink_branch3_omega = np.array(fast_kink_branch3_omega)
#fast_kink_branch3_k = np.array(fast_kink_branch3_k)
#
###################################################
#
#########################################################################   sausage polyfit
#
#if len(fast_sausage_branch1_omega) > 1:
#  FSB1_phase = fast_sausage_branch1_omega/fast_sausage_branch1_k
#  FSB1_k = fast_sausage_branch1_k
#  k_new = np.linspace(FSB1_k[0], FSB1_k[-1], num=len(FSB1_k)*10)
#  
#  coefs = poly.polyfit(FSB1_k, FSB1_phase, 6)   #  # 6 is good
#  ffit = poly.polyval(k_new, coefs)
#
#if len(fast_sausage_branch2_omega) > 1:
#  FSB2_phase = fast_sausage_branch2_omega/fast_sausage_branch2_k
#  FSB2_k = fast_sausage_branch2_k
#  k_new_2 = np.linspace(FSB2_k[0], FSB2_k[-1], num=len(FSB2_k)*10)  #FSB2_k[-1]
#  
#  coefs_2 = poly.polyfit(FSB2_k, FSB2_phase, 6)    # 6 order     # 1st order for W < 1.5
#  ffit_2 = poly.polyval(k_new_2, coefs_2)
#
#########################################################################   kink polyfit
#if len(fast_kink_branch1_omega) > 1:
#  FSB1_kink_phase = fast_kink_branch1_omega/fast_kink_branch1_k
#  FSB1_kink_k = fast_kink_branch1_k
#  k_kink_new = np.linspace(FSB1_kink_k[0], FSB1_kink_k[-1], num=len(FSB1_kink_k)*10)   #FSB1_kink_k[-1]
#  
#  coefs_kink = poly.polyfit(FSB1_kink_k, FSB1_kink_phase, 6) # 3 order for messing
#  ffit_kink = poly.polyval(k_kink_new, coefs_kink)
#
#if len(fast_kink_branch2_omega) > 1:
#  FSB2_kink_phase = fast_kink_branch2_omega/fast_kink_branch2_k
#  FSB2_kink_k = fast_kink_branch2_k
#  k_kink_new_2 = np.linspace(FSB2_kink_k[0], FSB2_kink_k[-1], num=len(FSB2_kink_k)*10)   #FSB2_kink_k[-1]
#  
#  coefs_2_kink = poly.polyfit(FSB2_kink_k, FSB2_kink_phase, 8)
#  ffit_2_kink = poly.polyval(k_kink_new_2, coefs_2_kink)
#
#
#
###########################################################################################
#
#if len(slow_backward_body_sausage_omega) > 1:
#  SBb_phase = slow_backward_body_sausage_omega/slow_backward_body_sausage_k
#  SBb_k = slow_backward_body_sausage_k
#  sbb_k_new = np.linspace(0, Kmax, num=len(SBb_k)*10)  #FSB2_k[-1]
#  
#  sbb_coefs = poly.polyfit(SBb_k, SBb_phase, 1)    # 6 order     # 1st order for W < 1.5
#  sbb_ffit = poly.polyval(sbb_k_new, sbb_coefs)
#
#
#if len(slow_backward_body_kink_omega) > 1:
#  SBbk_phase = slow_backward_body_kink_omega/slow_backward_body_kink_k
#  SBbk_k = slow_backward_body_kink_k
#  sbbk_k_new = np.linspace(0, Kmax, num=len(SBbk_k)*10)  #SBk_k[-1]
#  
#  sbbk_coefs = poly.polyfit(SBbk_k, SBbk_phase, 1)    # 6 order     # 1st order for W < 1.5
#  sbbk_ffit = poly.polyval(sbbk_k_new, sbbk_coefs)
#
#
###########################################################################################
#
#
###########################################################################################
#
#new_modes_sausage_k = np.array(new_modes_sausage_k)
#new_modes_sausage_omega = np.array(new_modes_sausage_omega)
#new_modes_kink_k = np.array(new_modes_kink_k)
#new_modes_kink_omega = np.array(new_modes_kink_omega)
#
#
#if len(new_modes_sausage_k) > 1:
#  newS_phase = new_modes_sausage_omega/new_modes_sausage_k
#  newS_k = new_modes_sausage_k
#  new_s_k_new = np.linspace(newS_k[0], newS_k[-1], num=len(newS_k)*10)  #FSB2_k[-1]
#  
#  new_s_coefs = poly.polyfit(newS_k, newS_phase, 4)    # 6 order     # 1st order for W < 1.5
#  new_s_ffit = poly.polyval(new_s_k_new, new_s_coefs)
#
#
#if len(new_modes_kink_k) > 1:
#  newK_phase = new_modes_kink_omega/new_modes_kink_k
#  newK_k = new_modes_kink_k
#  newK_k_new = np.linspace(newK_k[0], newK_k[-1], num=len(newK_k)*10)  #SBk_k[-1]
#  
#  newK_coefs = poly.polyfit(newK_k, newK_phase, 6)    # 6 order     # 1st order for W < 1.5
#  newK_ffit = poly.polyval(newK_k_new, newK_coefs)
#
#
###########################################################################################
#test_k_plot = np.linspace(0.01,Kmax,20)
#
#fig=plt.figure()
#ax = plt.subplot(111)
#plt.xlabel("$k$", fontsize=18)
#plt.ylabel('$\omega$', fontsize=22, rotation=0, labelpad=15)
#vA_e_plot = test_k_plot*vA_e
#vA_i_plot = test_k_plot*vA_i
#c_e_plot = test_k_plot*c_e
#c_i_plot = test_k_plot*c_i
#cT_e_plot = test_k_plot*cT_e()
#cT_i_plot = test_k_plot*cT_i()
##ax.plot(sol_ks1, sol_omegas1, 'b.', markersize=4.)
#ax.plot(fast_body_sausage_k, fast_body_sausage_omega, 'r.', markersize=4.)   #fast body sausage
#ax.plot(fast_body_kink_k, fast_body_kink_omega, 'b.', markersize=4.)   #fast body kink
##ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega, 'r.', markersize=4.)   #fast body branch 1
##ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega, 'r.', markersize=4.)   #fast body branch 2
##ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega, 'r.', markersize=4.)   #fast body branch 3
#ax.plot(test_k_plot, vA_e_plot, linestyle='dashdot', color='k')
#ax.plot(test_k_plot, vA_i_plot, linestyle='dashdot', color='k')
#ax.plot(test_k_plot, c_e_plot, linestyle='dashdot', color='k')
#ax.plot(test_k_plot, c_i_plot, linestyle='dashdot', color='k')
#ax.plot(test_k_plot, cT_e_plot, linestyle='dashdot', color='k')
#ax.plot(test_k_plot, cT_i_plot, linestyle='dashdot', color='k')
##ax.plot(slow_body_sausage_k, slow_body_sausage_omega, 'b.', markersize=4.)   #slow body sausage
##ax.plot(slow_body_kink_k, slow_body_kink_omega, 'b.', markersize=4.)   #slow body kink
#
#ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i_plot[-1]), fontsize=20)
##ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e_plot[-1]), fontsize=20)
#ax.annotate( '$c_{e},    c_{Te}$', xy=(Kmax, c_e_plot[-1]), fontsize=20)
#ax.annotate( '$c_{i}$', xy=(Kmax, c_i_plot[-1]), fontsize=20)
#ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e_plot[-1]), fontsize=20)
#ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i_plot[-1]), fontsize=20)
#ax.axhline(y=cutoff, color='r', linestyle='dashed', label='_nolegend_')
#ax.axhline(y=cutoff2, color='r', linestyle='dashed', label='_nolegend_')
#ax.annotate( '$\omega = {}$'.format(cutoff2), xy=(Kmax, cutoff2), fontsize=18, color='r')
#ax.annotate( '$\omega = {}$'.format(cutoff), xy=(Kmax, cutoff), fontsize=18, color='r')
#ax.axhline(y=cutoff3, color='r', linestyle='dashed', label='_nolegend_')
#ax.annotate( '$\omega = {}$'.format(cutoff3), xy=(Kmax, cutoff3), fontsize=18, color='r')
#ax.axhline(y=cutoff_kink, color='b', linestyle='dashed', label='_nolegend_')
#ax.axhline(y=cutoff2_kink, color='b', linestyle='dashed', label='_nolegend_')
#ax.annotate( '$\omega = {}$'.format(cutoff2_kink), xy=(Kmax, cutoff2_kink), fontsize=18, color='b')
#ax.annotate( '$\omega = {}$'.format(cutoff_kink), xy=(Kmax, cutoff_kink), fontsize=18, color='b')
#

########################################################################################################


########################################################################################################
Kmax = 1.5

plt.figure()
#plt.title("No iic $ 0 < \gamma < 0.001$")

#ax4= plt.subplot(221)
#ax = plt.subplot(222)
#ax2 = plt.subplot(223)
#ax3= plt.subplot(224)

ax = plt.subplot(111)

#ax.title.set_text('$-0.1 < \omega_i < 0.1$')
plt.title("iic  -----  $ 0 < \gamma < 0.001$")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)


ax.plot(sol_ks1_real, (sol_omegas1_real/sol_ks1_real), 'gx', markersize=6.)
ax.plot(sol_ks_kink1_real, (sol_omegas_kink1_real/sol_ks_kink1_real), 'gx', markersize=6.)

ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

ax.plot(sol_ks1_imag, (sol_omegas1_imag/sol_ks1_imag), 'rx', markersize=4.)
ax.plot(sol_ks_kink1_imag, (sol_omegas_kink1_imag/sol_ks_kink1_imag), 'bx', markersize=4.)


#ax.plot(sol_ks1_01, (sol_omegas1_01/sol_ks1_01), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1_01, (sol_omegas_kink1_01/sol_ks_kink1_01), 'b.', markersize=4.)

#ax.plot(sol_ks1_imag_01, (sol_omegas1_imag_01/sol_ks1_imag_01), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1_imag_01, (sol_omegas_kink1_imag_01/sol_ks_kink1_imag_01), 'b.', markersize=4.)


#ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega/fast_sausage_branch1_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega/fast_sausage_branch2_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega/fast_sausage_branch3_k, 'r.', markersize=4.)

#ax.plot(k_new, ffit, color='r')
#ax.plot(k_new_2, ffit_2, color='r')
#ax.plot(k_new_3, ffit_3, color='r')

#ax.plot(fast_kink_branch1_k, fast_kink_branch1_omega/fast_kink_branch1_k, 'b.', markersize=4.)
#ax.plot(fast_kink_branch2_k, fast_kink_branch2_omega/fast_kink_branch2_k, 'g.', markersize=4.)

#ax.plot(k_kink_new, ffit_kink, color='b')
#ax.plot(k_kink_new_2, ffit_2_kink, color='b')

#ax.plot(slow_body_sausage_k, (slow_body_sausage_omega/slow_body_sausage_k), 'r.', markersize=4.)
#ax.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)

#ax.plot(body_sausage_branch1_k, body_sausage_branch1_omega/body_sausage_branch1_k, 'r.', markersize=4.)   # body sausage
#ax.plot(body_kink_branch1_k, body_kink_branch1_omega/body_kink_branch1_k, 'b.', markersize=4.)   # body kink
#ax.plot(body_sausage_branch2_k, body_sausage_branch2_omega/body_sausage_branch2_k, 'g.', markersize=4.)   # body sausage
#ax.plot(body_kink_branch2_k, body_kink_branch2_omega/body_kink_branch2_k, 'b.', markersize=4.)   # body kink

#ax.plot(sb_k_new, sb_ffit, color='r')
#ax.plot(sbk_k_new, sbk_ffit, color='b')

#ax.plot(sbb_k_new, sbb_ffit, color='r')
#ax.plot(sbbk_k_new, sbbk_ffit, color='b')

#ax.plot(new_s_k_new, new_s_ffit, color='r')
#ax.plot(newK_k_new, newK_ffit, color='b')



ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_i(), color='k', linestyle='dashdot', label='_nolegend_')

#ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i()), fontsize=20)
ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)


ax.axhline(y=-cT_i()+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-vA_e+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.annotate( ' $-c_{Ti}+U_{i0}$', xy=(Kmax, -cT_i()+U_i0), fontsize=10)
ax.annotate( ' $-v_{Ai}+U_{i0}$', xy=(Kmax, -vA_i+U_i0), fontsize=10)
#ax.annotate( '$-v_{Ae}+U_{i0}$', xy=(Kmax, -vA_e+U_i0), fontsize=10)
#ax.axhline(y=c_e+U_i0, color='k', linestyle='solid', label='_nolegend_')
#ax.annotate( '$c_e + U_i$', xy=(Kmax, c_e+U_i0), fontsize=20)

#ax.axhline(y=flow_bound, color='k', linestyle='solid', label='_nolegend_')
#ax.annotate( '$U_0$', xy=(Kmax, flow_bound), fontsize=20)

#ax.axhline(y=U_i0, color='k', linestyle='solid', label='_nolegend_')
#ax.annotate( '$U_{max}$', xy=(Kmax, U_i0), fontsize=20)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

ax.set_xlim(0, Kmax)
#plt.savefig("flow_iic_no_gamma.png")


#plt.show()
#exit()



####################################################################

B=1.

click_k = sol_ks_kink1  #sol_ks_kink1
click_omega_real = sol_omegas_kink1   #sol_omegas_kink1
click_omega_imag = sol_omegas_kink1_imag   #sol_omegas_kink1_imag

image = []
coeffs = []

#fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for photospheric to remove blank space on plot  
fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False)   #  fig, (ax, ax2) = plt.subplots(2,1, sharex=False)  - for DD + one plot
fig2, (ax, ax4, ax5, ax6) = plt.subplots(4,1, sharex=False)
#ax.title.set_text("W = 2")
#plt.title(" ")
#plt.xlabel("$kx_{0}$", fontsize=18)
#plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False)

ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=18, rotation=0, labelpad=15)
ax.set_xlabel("$kx_{0}$", fontsize=12)

ax.plot(sol_ks1_real, (sol_omegas1_real/sol_ks1_real), 'gx', markersize=6.)
ax.plot(sol_ks_kink1_real, (sol_omegas_kink1_real/sol_ks_kink1_real), 'gx', markersize=6.)

ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

ax.plot(sol_ks1_imag, (sol_omegas1_imag/sol_ks1_imag), 'rx', markersize=4.)
ax.plot(sol_ks_kink1_imag, (sol_omegas_kink1_imag/sol_ks_kink1_imag), 'bx', markersize=4.)


ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')

ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i), fontsize=20)


ax.axhline(y=-cT_i()+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.annotate( ' $-c_{Ti}+U_{i0}$', xy=(Kmax, -cT_i()+U_i0), fontsize=10)
ax.annotate( ' $-v_{Ai}+U_{i0}$', xy=(Kmax, -vA_i+U_i0), fontsize=10)

#ax.set_ylim(0., vA_e+0.1)
ax.set_xlim(0, Kmax)
box = ax.get_position()
ax.set_position([box.x0, box.y0-0.1, box.width*0.85, box.height*1.75])  # [box.x0, box.y0-0.15, box.width*0.95, box.height*1.5]  USE THIS FOR ONE EXTRA PLOT

ax2.axvline(x=-B, color='r', linestyle='--')
ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$x$")
#ax2.set_ylabel("$\u03BE_{x_{REAL}}$")   #\xi
ax2.set_ylabel("$P_{x_{REAL}}$")      #\Pressure
ax2.set_ylim(-1.2,1.2)

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0-0.05, box2.width*1., box2.height*0.65])


ax3.axvline(x=-B, color='r', linestyle='--')
ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_xlabel("$x$")
#ax3.set_ylabel("$\u03BE_{x_{IMAG}}$")
ax3.set_ylabel("$P_{x_{IMAG}}$")
ax3.set_ylim(-1.2,1.2)

box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width*1., box3.height*0.65])


ax4.axvline(x=-B, color='r', linestyle='--')
ax4.axvline(x=B, color='r', linestyle='--')
ax4.set_xlabel("$x$")
ax4.set_ylabel("$F$")

ax4.set_yticks([4., 3.5, 3., 2.5, 2.])
box4 = ax4.get_position()
ax4.set_position([box4.x0, box4.y0-0.1, box4.width*1., box4.height*0.65])


ax5.axvline(x=-B, color='r', linestyle='--')
ax5.axvline(x=B, color='r', linestyle='--')
ax5.set_xlabel("$x$")
ax5.set_ylabel("$dF$")

ax5.set_yticks([2., 1., 0., -1., -2.])
box5 = ax5.get_position()
ax5.set_position([box5.x0, box5.y0-0.05, box5.width*1., box5.height*0.65])


ax6.axvline(x=-B, color='r', linestyle='--')
ax6.axvline(x=B, color='r', linestyle='--')
ax6.set_xlabel("$x$")
ax6.set_ylabel("$m_0^2$")

ax6.set_yticks([1., 0., -1., -1.5])
box6 = ax3.get_position()
ax6.set_position([box6.x0, box6.y0, box6.width*1., box6.height*0.65])


#plt.show()
#exit()

############################################################################################################
with stdout_redirected():
  for i in range(len(click_k)):
     lx = np.linspace(-3.*2.*np.pi/click_k[i], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
     ix = np.linspace(-1., 1., 500.)  # inside slab x values

     m_e = ((((click_k[i]**2*vA_e**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))**2)*((click_k[i]**2*c_e**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))**2))/((vA_e**2+c_e**2)*((click_k[i]**2*cT_e()**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))**2)))

     p_e_const = rho_e*(vA_e**2+c_e**2)*((click_k[i]**2*cT_e()**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))**2)/(((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))*((click_k[i]**2*c_e**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*mach_e))**2))


#######################################################################################################           
     def m0(x):    
       return ((((click_k[i]**2*c_i**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))**2)*((click_k[i]**2*vA_i**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))**2))/((c_i**2+vA_i**2)*((click_k[i]**2*cT_i()**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))**2)))  

     m0_np=sym.lambdify(xx,m0(xx),"numpy")           
##############################################

     def D(x):    
       return (2.*click_k[i]*dU_i(x)*((((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x))**2/((((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x))**2)-click_k[i]**2*c_i**2)) - ((click_k[i]**2*cT_i()**2)/((((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x))**2)-click_k[i]**2*cT_i()**2)))/((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x))) 
                   
     D_np=sym.lambdify(xx,D(xx),"numpy")

##############################################

     def coeff(x):
         return ((click_k[i]*ddU_i(x)/((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x)))+((click_k[i]*dU_i(x)*D(x))/((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x))) - m0(x))

     coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
##############################################  

     def P_Ti(x):    
       return (rho_i*(vA_i**2+c_i**2)*((click_k[i]**2*cT_i()**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))**2)/(((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))*((click_k[i]**2*c_i**2)-((click_omega_real[i] + 1j*click_omega_imag[i])-(click_k[i]*U_i(x)))**2))) 
      
     PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
     p_i_const = PT_i_np(ix)


     def add_P_Ti(x):
       return (-(click_k[i]*dU_i(x))/((click_omega_real[i] + 1j*click_omega_imag[i])-click_k[i]*U_i(x)))
     
     add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
     add_p_i_const = add_PT_i_np(ix)                         
########################################################################################################

     def dVx_dx_e(Vx_e, x_e):
            return [Vx_e[1], m_e*Vx_e[0]]
       
     V0 = [1e-8 + 1j*1e-8, 1e-15 + 1j*1e-15]              
     #V0 = [1e-8, 1e-15]              
     
     Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
     
     #left_solution = Ls[:,0].real*(click_omega_real[i]-click_k[i]*mach_i[0])/(click_omega_real[i]-click_k[i]*mach_e)  # Vx perturbation solution for left hand side
     #left_solution_imag = Ls[:,0].imag*(click_omega_imag[i]-click_k[i]*mach_i[0])/(click_omega_imag[i]-click_k[i]*mach_e)  # Vx perturbation solution for left hand side
     
     left_solution = Ls[:,0].real*(click_omega_real[i]*(((click_omega_real[i] - click_k[i]*mach_i[0])**2)+click_omega_imag[i]**2))/((click_omega_real[i]**2+click_omega_imag[i]**2)*(click_omega_real[i] - click_k[i]*mach_i[0]))   
     left_solution_imag = Ls[:,0].imag*(((click_omega_real[i] - click_k[i]*mach_i[0])**2)+click_omega_imag[i]**2)/(click_omega_real[i]**2+click_omega_imag[i]**2)

     left_P_solution = p_e_const.real*Ls[:,1].real  # Pressure perturbation solution for left hand side
     left_P_solution_imag = p_e_const.imag*Ls[:,1].imag  # Pressure perturbation solution for left hand side
     
     normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
     normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))

     normalised_left_P_solution_imag = left_P_solution_imag/np.amax(abs(left_P_solution_imag))
     normalised_left_vx_solution_imag = left_solution_imag/np.amax(abs(left_solution_imag))
     
     left_bound_vx = left_solution[-1] 
     left_bound_vx_imag = left_solution_imag[-1] 
                     
     def dVx_dx_i(Vx_i, x_i):
           return [Vx_i[1], (-D_np(x_i)*Vx_i[1] - coeff_np(x_i)*Vx_i[0])]
      
        
     def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
           U = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi[0] + 1j*dVxi[1]], ix, printmessg=0)
           u1 = U[:,0].real - left_bound_vx
           u2 = U[:,0].imag - left_bound_vx_imag  #+ for sausage   - for kink
           return [u1[-1], u2[-1]]
                               
     [dVxi, dVxi_imag] = fsolve(objective_dvxi, [1., 1.])    #zero is guess for roots of equation, maybe change to find more body modes??

     # now solve with optimal dvx
     
     Is = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi + 1j*dVxi_imag], ix)
     inside_solution = Is[:,0].real
     inside_solution_imag = Is[:,0].imag
     
     inside_P_solution = p_i_const.real*(Is[:,1].real - add_p_i_const.real*Is[:,0].real)
     inside_P_solution_imag = p_i_const.imag*(Is[:,1].imag - add_p_i_const.imag*Is[:,0].imag)
     
     normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
     normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
    
     normalised_inside_P_solution_imag = inside_P_solution_imag/np.amax(abs(left_P_solution_imag))
     normalised_inside_vx_solution_imag = inside_solution_imag/np.amax(abs(left_solution_imag))

     
     disp_dot, = ax.plot(click_k[i], click_omega_real[i]/click_k[i], 'k.',markersize=8.)
     disp_dot_imag, = ax.plot(click_k[i], click_omega_imag[i]/click_k[i], 'kx',markersize=8.)
    
     
     left_vx_plot, = ax2.plot(lx, normalised_left_vx_solution, 'b')
     left_vx_text = ax2.text(-5.5, 0.7, "External boundary value = {:.5g}".format(left_solution[-1]))
     inside_vx_plot, = ax2.plot(ix, normalised_inside_vx_solution, 'b')
     inside_vx_text = ax2.text(-5.5, -0.5, "Internal left boundary value = {:.5g}".format(inside_solution[0]))
     inside_right_vx_text = ax2.text(-5.5, -0.9, "Internal right boundary value = {:.5g}".format(inside_solution[-1]))

     left_vx_plot_imag, = ax3.plot(lx, normalised_left_vx_solution_imag, 'r')
     left_vx_text_imag = ax3.text(-5.5, 0.7, "External boundary value = {:.5g}".format(left_solution_imag[-1]))
     inside_vx_plot_imag, = ax3.plot(ix, normalised_inside_vx_solution_imag, 'r')
     inside_vx_text_imag = ax3.text(-5.5, -0.5, "Internal left boundary value = {:.5g}".format(inside_solution_imag[0]))
     inside_right_vx_text_imag = ax3.text(-5.5, -0.9, "Internal right boundary value = {:.5g}".format(inside_solution_imag[-1]))

      
#     left_pressure_plot, = ax2.plot(lx, normalised_left_P_solution, 'b') 
#     inside_pressure_plot, = ax2.plot(ix, normalised_inside_P_solution, 'b')
#     
#     left_P_text = ax2.text(-5.5, 0.7, "External boundary value = {:.5g}".format(left_P_solution[-1]))
#     inside_P_text = ax2.text(-5.5, -0.5, "Internal left boundary value = {:.5g}".format(inside_P_solution[0]))
#     inside_right_P_text = ax2.text(-5.5, -0.9, "Internal right boundary value = {:.5g}".format(inside_P_solution[-1]))
#     
#
#     left_pressure_plot_imag, = ax3.plot(lx, normalised_left_P_solution_imag, 'r') 
#     inside_pressure_plot_imag, = ax3.plot(ix, normalised_inside_P_solution_imag, 'r')
#     
#     left_P_text_imag = ax3.text(-5.5, 0.7, "External boundary value = {:.5g}".format(left_P_solution_imag[-1]))
#     inside_P_text_imag = ax3.text(-5.5, -0.5, "Internal left boundary value = {:.5g}".format(inside_P_solution_imag[0]))
#     inside_right_P_text_imag = ax3.text(-5.5, -0.9, "Internal right boundary value = {:.5g}".format(inside_P_solution_imag[-1]))


     #ax3.set_xlim(min(lx),1.2)
     ax2.set_xlim(-6.,1.2)
     ax3.set_xlim(-6.,1.2)
     
     #ax4.set_xlim(-1.2,1.2)
     #ax5.set_xlim(-1.2,1.2)
     #ax6.set_xlim(-1.2,1.2)
     
     image.append([disp_dot, disp_dot_imag, left_vx_plot, left_vx_text, inside_vx_plot, inside_vx_text, left_vx_plot_imag, left_vx_text_imag, inside_vx_plot_imag, inside_vx_text_imag, inside_right_vx_text, inside_right_vx_text_imag])

     #image.append([disp_dot, disp_dot_imag, left_pressure_plot, left_P_text, inside_pressure_plot, inside_P_text, left_pressure_plot_imag, left_P_text_imag, inside_pressure_plot_imag, inside_P_text_imag, inside_right_P_text, inside_right_P_text_imag])
     
     #image.append([disp_dot, left_pressure_plot, inside_pressure_plot, left_P_text, inside_P_text, inside_right_P_text])
     #image.append([disp_dot, left_vx_plot, inside_vx_plot, left_vx_text, inside_vx_text, inside_right_vx_text])

     #image.append([disp_dot, left_pressure_plot, inside_pressure_plot, left_P_text, inside_P_text, inside_right_P_text, left_vx_plot, inside_vx_plot, left_vx_text, inside_vx_text, inside_right_vx_text])
     
     #coeffs.append([disp_dot, F_plot, dF_plot, m0_plot])

Writer = animation.writers['ffmpeg']
writer = Writer(fps=3, bitrate=4000)   #9fps if skipping 5 in time step   15 good
           
anim = animation.ArtistAnimation(fig, image, interval=0.1, blit=True, repeat=False).save('imag_flow_xi_kink_test.mp4', writer=writer)
#coeffs_anim = animation.ArtistAnimation(fig2, coeffs, interval=0.1, blit=True, repeat=False).save('width2_coronal_coeffs_kinkfastbranch.mp4', writer=writer)
 
