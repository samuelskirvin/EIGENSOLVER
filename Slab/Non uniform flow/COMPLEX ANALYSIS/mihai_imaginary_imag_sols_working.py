# Import the required modules
import numpy as np
import scipy as sc
import matplotlib; matplotlib.use('agg') ##comment out to show figs
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sym
import math
from math import log10, floor
from scipy.optimize import fsolve
import cmath
from matplotlib import animation
from scipy import interpolate
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


from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
NUMPY_TRANSLATIONS["zoo"] = "nan"


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

#######################################################################

##############      BEGIN STEADY FLOW SLAB DISPERSION DIAGRAM       ###############


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


#rho_e = 2.27*rho_i          #2.27*rho_i  photospheric
#print('external density    =', rho_e)

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


# variables (reference only):
# W = omega/k
# K = k*x_0

Dmin = 0.
Dmax = 4.5
    
#Wmin = complex(0,0)    #Produces body modes??
#Wmax = complex(1.8,1.8)

Wmin = 0.
Wmax = 3.

ix = np.linspace(-1, 1, 500)  # inside slab x values

x0=0.  #mean
dx=1e5  #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols


def U_i(x):                  # Define the internal alfven speed
    #return vA_i0*profile(xx)
    return (U_e + ((U_i0 - U_e)*sym.exp(-(x-x0)**2/dx**2)))   #make 0.5*

#def U_i(x):     #constant              # Define the internal alfven speed
    #return vA_i0*profile(xx)
#    return mach_i0   #make 0.5*


U_i_np=sym.lambdify(xx,U_i(xx),"numpy")   #In order to evaluate we need to switch to numpy

mach_i = U_i_np(ix)  #/vA_i
mach_e = U_e  #/vA_i

plt.figure()
plt.xlabel("x")
plt.ylabel("$U_{i}$")
ax = plt.subplot(111)
#ax.plot(ix,U_i_np(ix));
ax.annotate( '$U_{e}$', xy=(1, U_e))
ax.annotate( '$U_{i}$', xy=(1, U_i0))
ax.axhline(y=U_i0, color='k', label='$U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label='$U_{e}$', linestyle='dashdot')


def dU_i(x):   #First derivative of profile in symbols    
    return sym.diff(U_i(x), x)  

dU_i_np=sym.lambdify(xx,dU_i(xx),"numpy")

def ddU_i(x):   #First derivative of profile in symbols    
    return sym.diff(dU_i(x), x)  

ddU_i_np=sym.lambdify(xx,ddU_i(xx),"numpy")

plt.figure()
plt.xlabel("x")
plt.ylabel("$dU_{i}$")
ax = plt.subplot(111)
#ax.plot(ix,dU_i_np(ix));
ax.annotate( '$U_{e}$', xy=(1, U_e))
ax.annotate( '$U_{i}$', xy=(1, U_i0))
ax.axhline(y=U_i0, color='k', label='$U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label='$U_{e}$', linestyle='dashdot')


plt.figure()
plt.xlabel("x")
plt.ylabel("$ddU_{i}$")
ax = plt.subplot(111)
#ax.plot(ix,ddU_i_np(ix));
ax.annotate( '$U_{e}$', xy=(1, U_e))
ax.annotate( '$U_{i}$', xy=(1, U_i0))
ax.axhline(y=U_i0, color='k', label='$U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label='$U_{e}$', linestyle='dashdot')



#####################################################
#speeds = [-vA_e, -c_e, 0., c_i, c_e, vA_i, vA_e, cT_i(), cT_e()]
speeds = [-0.5, 0.5, 0.]    #modified for extra res in imaginary region 
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


######################            BEGIN NEW METHOD FOR DISPERSION DIAGRAM          #############################

def round_to_1_sf(x):
   return round(x, -int(floor(log10(abs(x)))))


#wavenumber = np.linspace(0.5,3.5,55)
#freq = np.logspace(0.001,0.8,80)-1   #0.75 instead of 1?

#wavenumber = np.linspace(0.01,3.5,121)
#freq = np.logspace(0.001,0.75,5000)-1   #0.75 instead of 1?

#freq = np.logspace(0.001,0.8,200)-1   #0.75 instead of 1?


#plt.figure()
#plt.xlabel("k$x_{0}$")
#plt.ylabel("$\omega/k$")  
#for i in range(len(wavenumber)):
#  for j in range(len(freq)):
#      
#      plt.plot(wavenumber[i], (freq.real[j] + 1j*freq.imag[i])/wavenumber[i], 'b.')
#
#plt.plot(x_out_sausage,W_array_sausage, 'rx', markersize=2.)
#plt.plot(x_out_sausage_body,W_array_sausage_body, 'rx', markersize=2., label='_nolegend_')   
#plt.ylim(0, 2.)
#plt.xlim(0, 1.6)

#plt.figure()
#for i in range(len(wavenumber)):
#  for j in range(len(freq)):
#      
#      plt.plot(wavenumber[i], (freq.real[j] + 1j*freq.imag[i]), 'b.')
# 


### Redefine tube speeds so not normalised to alfven speed
def cT_i():
    return np.sqrt(c_i**2 * vA_i**2 / (c_i**2 + vA_i**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))



## INITIALLY HAVE ONE PAIR OF W AND K AND WORK THROUGH WITH JUST THAT PAIR - IF NO MATCH MOVE ONTO NEXT PAIR

d = 1.

#v_tol = 1e-6
#p_tol = 1e-3

p_tol = 1.5    #1e-6   # i.e 1% tolerance

#We need to create an array of dvx values to guess over
#dvx_guess = np.linspace(-0.015, 0.015, 30)

test_sols_inside = []
test_sols_outside = []

sol_omegas = []
sol_ks = []
sol_omegas1 = []
sol_ks1 = []
sol_omegas1_imag = []
sol_ks1_imag = []

test_w_k_sausage = []
test_p_diff_sausage = []
test_w_k_kink = []
test_p_diff_kink = []

P_diff_check = [0]
P_diff_loop_check = [0]
loop_sign_check = [0]
sign_check = [0]

P_diff_check_imag = [0]
P_diff_loop_check_imag = [0]
loop_sign_check_imag = [0]
sign_check_imag = [0]

all_ws = []
all_ks = []

all_ws_imag = []
all_ks_imag = []

loop_ws = []
loop_ws_imag = []

p_diff_sols = []
p_diff_sols_imag = []


image = []
image_kink = []

def sausage(wavenumber, sausage_ws, sausage_ks, sausage_ws_imag, sausage_ks_imag, freq):

#####    DEFINE FUNCTIONS TO LOCATE SOLUTIONS    #####################

  ################################################
#  def locate_sausage_imag(iomega, wavenum,itt_num_imag, omega_real):
#  
#      omega_real = omega_real
#      all_ws_imag[:] = []
#      itt_num_imag = itt_num_imag
#                
#      for s in range(len(iomega)):
#      
#         if itt_num_imag > 50:
#            break
#            
#         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
#         ix = np.linspace(-1., 1., 500.)  # inside slab x values
#  
#
#         m_e = ((((wavenum**2*vA_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)))
#      
#         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)/(((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))*((wavenum**2*c_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2))       
#
#
########################################################################################################           
#         def m0(x):    
#           return ((((wavenum**2*c_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)*((wavenum**2*vA_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenum**2*cT_i()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)))  
# 
#         m0_np=sym.lambdify(xx,m0(xx),"numpy")           
#    ##############################################
#               
#         def D(x):    
#           return (2.*wavenum*dU_i(x)*((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2/((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*c_i**2)) - ((wavenum**2*cT_i()**2)/((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*cT_i()**2)))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))  
#         
#         D_np=sym.lambdify(xx,D(xx),"numpy")
#    ##############################################
#    
#         def coeff(x):
#             return ((wavenum*ddU_i(x)/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))+((wavenum*dU_i(x)*D(x))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))) - m0(x))
#
#         coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
#    ##############################################
#    
#         def P_Ti(x):    
#           return (rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)/(((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))*((wavenum**2*c_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2))) 
#          
#         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
#         p_i_const = PT_i_np(ix)
#
#         def add_P_Ti(x):
#           return (-(wavenum*dU_i(x))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))
#            
#         add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
#         add_p_i_const = add_PT_i_np(ix)                            
#########################################################################################################
#                
#         if m_e < 0:
#             pass
#             
#         else:         
#             loop_ws_imag.append(iomega.imag[s])
#             
#             def dVx_dx_e(Vx_e, x_e):
#                    return [Vx_e[1], m_e.imag*Vx_e[0]]
#                                        
#                                 
#             V0 = [1e-8, 1e-15]  
#             Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
#             
#             left_solution_imag = Ls[:,0]*(iomega.imag[s]-wavenum*mach_i[0])/(iomega.imag[s]-wavenum*mach_e)  # Vx perturbation solution for left hand side
#            
#             left_P_solution_imag = p_e_const.imag*Ls[:,1]  # Pressure perturbation solution for left hand side
#             
#             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
#             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
#             
#             left_bound_vx_imag = left_solution_imag[-1] 
#                       
#             def dVx_dx_i(Vx_i, x_i):
#                    return [Vx_i[1], (-D_np(x_i).imag*Vx_i[1] - coeff_np(x_i).imag*Vx_i[0])]
#             
#                
#             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
#                   U = odeintz(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
#                   u1 = U[:,0] + left_bound_vx_imag
#                   return u1[-1] 
#
#                                      
#             dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
#                  
#             Is_imag = odeintz(dVx_dx_i, [left_bound_vx_imag, dVxi], ix)
#             
#             inside_solution_imag = Is_imag[:,0]
#             
#             inside_P_solution_imag = p_i_const.imag*(Is_imag[:,1] - add_p_i_const.imag*Is_imag[:,0])
#             
#             normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
#             normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
#             
#              
#             P_diff_loop_check.append((left_P_solution[-1]-inside_P_solution[0]))
#             loop_sign_check.append(P_diff_loop_check[-1]*P_diff_loop_check[-2])
#            
#             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
#                    sol_omegas1_imag.append(iomega.imag[s])
#                    sol_ks1_imag.append(wavenum)
#                    loop_ws_imag[:] = []
#                    loop_sign_check[:] = [0]
#                    break
#
#             elif loop_sign_check[-1] < 0 and len(loop_ws_imag)>1:   #If this is negative go back through and reduce gap
#                  
#                    iomega = np.linspace(loop_ws_imag[-2], loop_ws_imag[-1], 20) 
#                    wavenum = wavenum
#                    itt_num_imag =itt_num_imag +1
#                    loop_ws_imag[:] = []           
#                    #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real)

  
      
  ##############################################################################



  ################################################
  def locate_sausage(omega, wavenum,itt_num):
  
      all_ws[:] = []
      all_ws_imag[:] = []
      itt_num = itt_num
                
      for k in range(len(omega.real)):
        for s in range(len(omega.imag)):
        
         if itt_num > 6:   #was 100
            break
            
         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
         ix = np.linspace(-1., 1., 500.)  # inside slab x values
  
         m_e = ((((wavenum**2*vA_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)))
      
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)/(((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))*((wavenum**2*c_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2))       


#######################################################################################################           
         def m0(x):    
           return ((((wavenum**2*c_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)*((wavenum**2*vA_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenum**2*cT_i()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)))  
 
         m0_np=sym.lambdify(xx,m0(xx),"numpy")           
    ##############################################
               
         def D(x):    
           return (2.*wavenum*dU_i(x)*((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2/((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*c_i**2)) - ((wavenum**2*cT_i()**2)/((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*cT_i()**2)))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))  
         
         D_np=sym.lambdify(xx,D(xx),"numpy")
    ##############################################
    
         def coeff(x):
             return ((wavenum*ddU_i(x)/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))+((wavenum*dU_i(x)*D(x))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))) - m0(x))

         coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
    ##############################################
    
         def P_Ti(x):    
           return (rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)/(((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))*((wavenum**2*c_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2))) 
          
         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
         p_i_const = PT_i_np(ix)

         def add_P_Ti(x):
           return (-(wavenum*dU_i(x))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))
            
         add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
         add_p_i_const = add_PT_i_np(ix)                            
########################################################################################################
                
         if m_e.real < 0:
             pass

         if m_e.imag < 0:
             pass
                          
         else:
             loop_ws.append(omega.real[k])
             loop_ws_imag.append(omega.imag[s])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
                                        
                                 
             V0 = [1e-8 + 1j*1e-8, 1e-15 + 1j*1e-15]  
             
             Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
             
             left_solution = Ls[:,0].real*(omega.real[k] - wavenum*mach_i[0])/(omega.real[k] - wavenum*mach_e)
             left_solution_imag = Ls[:,0].imag*(omega.imag[s] - wavenum*mach_i[0])/(omega.imag[s] - wavenum*mach_e)
            
             left_P_solution = p_e_const.real*Ls[:,1].real  # Pressure perturbation solution for left hand side
             left_P_solution_imag = p_e_const.imag*Ls[:,1].imag  
             
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             
             left_bound_vx = left_solution[-1] 
             left_bound_vx_imag = left_solution_imag[-1] 
                       
             def dVx_dx_i(Vx_i, x_i):
                   return [Vx_i[1], (-D_np(x_i)*Vx_i[1] - coeff_np(x_i)*Vx_i[0])]
              
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi[0] + 1j*dVxi[1]], ix, printmessg=0)
                   u1 = U[:,0].real + left_bound_vx
                   u2 = U[:,0].imag + left_bound_vx_imag
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
             
              
             P_diff_loop_check.append((left_P_solution_imag[-1]-inside_P_solution_imag[0]))
             loop_sign_check.append(P_diff_loop_check[-1]*P_diff_loop_check[-2])

             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
                    sol_omegas1.append(omega.real[k])
                    sol_omegas1_imag.append(omega.imag[s])
                    sol_ks1.append(wavenum)
                    sol_ks1_imag.append(wavenum)
                    p_diff_sols.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    loop_ws[:] = []
                    loop_ws_imag[:] = []
                    loop_sign_check[:] = [0]
                    #omega_real = (omega.real[k] + 1j*omega.imag[s])
                    #iomega = np.linspace(-1., 1., 20)
                    #wavenum = wavenum
                    #itt_num_imag = 0
                    #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real)
                    break

#             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
#                    sol_omegas1_imag.append(omega.imag[s])
#                    sol_ks1_imag.append(wavenum)
#                    loop_ws_imag[:] = []
#                    #omega_real = (omega.real[k] + 1j*omega.imag[s])
#                    #iomega = np.linspace(-1., 1., 20)
#                    #wavenum = wavenum
#                    #itt_num_imag = 0
#                    #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real)
#                    break
                                                        
             elif loop_sign_check[-1] < 0 and len(loop_ws)>1 and len(loop_ws_imag)>1:   #If this is negative go back through and reduce gap
                  
                    omega = np.linspace(loop_ws[-2], loop_ws[-1], 3) +1j*np.linspace(loop_ws_imag[-2], loop_ws_imag[-1], 3) 
                    wavenum = wavenum
                    itt_num =itt_num +1
                    loop_ws[:] = []           
                    loop_ws_imag[:] = []  
                    locate_sausage(omega, wavenum, itt_num)
                     
   


#        if loop_sign_check[-1] < 0 and len(loop_ws)>1 and len(loop_ws_imag)>1:   #If this is negative go back through and reduce gap
#             
#               omega = np.linspace(loop_ws[-2], loop_ws[-1], 3) +1j*np.linspace(loop_ws_imag[-2], loop_ws_imag[-1], 3) 
#               wavenum = wavenum
#               itt_num =itt_num +1
#               loop_ws[:] = []           
#               loop_ws_imag[:] = []  
#               locate_sausage(omega, wavenum, itt_num)               
                 
  ##############################################################################
  
  
  with stdout_redirected():
        for j in range(len(freq.real)):
          for i in range(len(freq.imag)):
            loop_ws[:] = [] 
            loop_ws_imag[:] = []
                      
            lx = np.linspace(-3.*2.*np.pi/wavenumber, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
            ix = np.linspace(-1, 1, 500)  # inside slab x values
      
            m_e = ((((wavenumber**2*vA_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)*((wavenumber**2*c_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)))

            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)/(((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))*((wavenumber**2*c_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2))


#######################################################################################################           
            def m0(x):    
              return ((((wavenumber**2*c_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)*((wavenumber**2*vA_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenumber**2*cT_i()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)))  
    
            m0_np=sym.lambdify(xx,m0(xx),"numpy")           
       ##############################################
 
            def D(x):    
              return (2.*wavenumber*dU_i(x)*((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2/((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2)-wavenumber**2*c_i**2)) - ((wavenumber**2*cT_i()**2)/((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2)-wavenumber**2*cT_i()**2)))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))) 
                          
            D_np=sym.lambdify(xx,D(xx),"numpy")

       ##############################################
       
            def coeff(x):
                return ((wavenumber*ddU_i(x)/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x)))+((wavenumber*dU_i(x)*D(x))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))) - m0(x))
   
            coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
       ##############################################  
   
            def P_Ti(x):    
              return (rho_i*(vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)/(((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))*((wavenumber**2*c_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2))) 
             
            PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
            p_i_const = PT_i_np(ix)


            def add_P_Ti(x):
              return (-(wavenumber*dU_i(x))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x)))
            
            add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
            add_p_i_const = add_PT_i_np(ix)                         
########################################################################################################
                        
            if m_e.real < 0:
              pass               
            
            if m_e.imag < 0:
              pass 

            else:
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8 + 1j*1e-8, 1e-15 + 1j*1e-15]              
              
              Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
              
              left_solution = Ls[:,0].real*(freq.real[j]-wavenumber*mach_i[0])/(freq.real[j]-wavenumber*mach_e)  # Vx perturbation solution for left hand side
              left_solution_imag = Ls[:,0].imag*(freq.imag[i]-wavenumber*mach_i[0])/(freq.imag[i]-wavenumber*mach_e)  # Vx perturbation solution for left hand side
              
              left_P_solution = p_e_const.real*Ls[:,1].real  # Pressure perturbation solution for left hand side
              left_P_solution_imag = p_e_const.imag*Ls[:,1].imag  # Pressure perturbation solution for left hand side
              
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              
              left_bound_vx = left_solution[-1] 
              left_bound_vx_imag = left_solution_imag[-1] 
                              
              def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], (-D_np(x_i)*Vx_i[1] - coeff_np(x_i)*Vx_i[0])]
               
                 
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi[0] + 1j*dVxi[1]], ix, printmessg=0)
                    u1 = U[:,0].real + left_bound_vx
                    u2 = U[:,0].imag + left_bound_vx_imag
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
              
              P_diff_check.append((left_P_solution_imag[-1]-inside_P_solution_imag[0]))
              sign_check.append(P_diff_check[-1]*P_diff_check[-2])
                                      
              test_w_k_sausage.append((freq.real[j])/wavenumber)
              test_p_diff_sausage.append(abs(left_P_solution[-1]) - abs(inside_P_solution[0]))
              
              all_ks.append(wavenumber)
              all_ws.append(freq.real[j])
              all_ws_imag.append(freq.imag[i])
              
              if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:  
                    sol_omegas1.append(freq.real[j])
                    sol_omegas1_imag.append(freq.imag[i])
                    sol_ks1.append(wavenumber)
                    sol_ks1_imag.append(wavenumber)
                    p_diff_sols.append(abs(left_P_solution_imag[-1] - inside_P_solution_imag[0]))
                    all_ws[:] = []
                    #iomega = np.linspace(-1., 1., 20)
                    #omega_real = (freq.real[j] + 1j*freq.imag[i])
                    #itt_num_imag = 0
                    #wavenum = wavenumber
                    #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real) 

#              if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:  
#                    sol_omegas1_imag.append(freq.imag[i])
#                    sol_ks1_imag.append(wavenumber)
#                    all_ws_imag[:] = []
#                    #iomega = np.linspace(-1., 1., 20)
#                    #omega_real = (freq.real[j] + 1j*freq.imag[i])
#                    #itt_num_imag = 0
#                    #wavenum = wavenumber
#                    #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real) 
                                                                        
              if sign_check[-1] < 0:  
                  if len(all_ws)>1 and len(all_ws_imag)>1:
                  
                     if all_ks[-1] != all_ks[-2]:
                       sign_check[:] = []
                       
                      
                     else:
                          omega = np.linspace(all_ws[-2], all_ws[-1], 3) + 1j*np.linspace(all_ws_imag[-2], all_ws_imag[-1], 3)
                          wavenum = all_ks[-1]
                 #        #now repeat exact same process but in foccused omega range
                          itt_num = 0
                          all_ws[:] = []
                          all_ws_imag[:] = []
                          locate_sausage(omega, wavenum, itt_num) 


#          if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:  
#                sol_omegas1.append(freq.real[j])
#                #sol_omegas1_imag.append(freq.imag[i])
#                sol_ks1.append(wavenumber)
#                #sol_ks1_imag.append(wavenumber)
#                p_diff_sols.append(abs(left_P_solution[-1] - inside_P_solution[0]))
#                all_ws[:] = []
#                #iomega = np.linspace(-1., 1., 20)
#                #omega_real = (freq.real[j] + 1j*freq.imag[i])
#                #itt_num_imag = 0
#                #wavenum = wavenumber
#                #locate_sausage_imag(iomega, wavenum, itt_num_imag, omega_real) 
#                                         
#          if sign_check[-1] < 0:  
#              if len(all_ws)>1 and len(all_ws_imag)>1:
#              
#                 if all_ks[-1] != all_ks[-2]:
#                   sign_check[:] = []
#                   
#                  
#                 else:
#                      omega = np.linspace(all_ws[-2], all_ws[-1], 3) + 1j*np.linspace(all_ws_imag[-2], all_ws_imag[-1], 3)
#                      wavenum = all_ks[-1]
#             #        #now repeat exact same process but in foccused omega range
#                      itt_num = 0
#                      all_ws[:] = []
#                      all_ws_imag[:] = []
#                      locate_sausage(omega, wavenum, itt_num)
                          
                                                   
  sausage_ks.put(sol_ks1)
  sausage_ws.put(sol_omegas1)
  sausage_ks_imag.put(sol_ks1_imag)
  sausage_ws_imag.put(sol_omegas1_imag)
 
 ########   TEST KINK   ##########

sol_omegas_kink = []
sol_ks_kink = []
sol_omegas_kink1 = []
sol_ks_kink1 = []
sol_omegas_kink1_imag = []
sol_ks_kink1_imag = []

P_diff_check_kink = [0]
P_diff_loop_check_kink = [0]
loop_sign_check_kink = [0]
sign_check_kink = [0]
p_diff_sols_kink = []
loop_ws_kink = []

all_ws_kink = []
all_ks_kink = []

###
P_diff_check_kink_imag = [0]
P_diff_loop_check_kink_imag = [0]
loop_sign_check_kink_imag = [0]
sign_check_kink_imag = [0]
p_diff_sols_kink_imag = []
loop_ws_kink_imag = []

all_ws_kink_imag = []
all_ks_kink_imag = []




def kink(wavenumber, kink_ws, kink_ks, kink_ws_imag, kink_ks_imag, freq):
  
#  def locate_kink_imag(iomega, wavenum,itt_num_imag, omega_real):
#      omega_real = omega_real
#      all_ws_imag[:] = []
#      itt_num_imag = itt_num_imag
#                 
#      for s in range(len(iomega)):
#      
#         if itt_num_imag > 50:
#            break
#         
#            
#         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
#         ix = np.linspace(-1., 1., 500.)  # inside slab x values
#  
#         m_e = ((((wavenum**2*vA_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)))
#      
#         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2)/(((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))*((wavenum**2*c_e**2)-((omega_real+1j*iomega.imag[s])-(wavenum*mach_e))**2))       
#
#
########################################################################################################           
#         def m0(x):    
#           return ((((wavenum**2*c_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)*((wavenum**2*vA_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenum**2*cT_i()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)))  
# 
#         m0_np=sym.lambdify(xx,m0(xx),"numpy")           
#    ##############################################
#               
#         #def D(x):    
#         #  return (2.*wavenum*dU_i(x)*((((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2-(wavenum**2*cT_i()**2))+((wavenum**4*cT_i()**2*c_i**2)/((c_i**2+vA_i**2)*(((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2-(wavenum**2*cT_i()**2)))))/(((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))*(((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2-(wavenum**2*c_i**2))))           
#         
#         def D(x):    
#           return (2.*wavenum*dU_i(x)*((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2/((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*c_i**2)) - ((wavenum**2*cT_i()**2)/((((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*cT_i()**2)))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))  
#         
#         D_np=sym.lambdify(xx,D(xx),"numpy")
#    ##############################################
#    
#         def coeff(x):
#             return ((wavenum*ddU_i(x)/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))+((wavenum*dU_i(x)*D(x))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x))) - m0(x))
#
#         coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
#    ##############################################
#    
#         def P_Ti(x):    
#           return (rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2)/(((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))*((wavenum**2*c_i**2)-((omega_real+1j*iomega.imag[s])-(wavenum*U_i(x)))**2))) 
#          
#         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
#         p_i_const = PT_i_np(ix)
#
#         def add_P_Ti(x):
#           return (-(wavenum*dU_i(x))/((omega_real+1j*iomega.imag[s])-wavenum*U_i(x)))
#            
#         add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
#         add_p_i_const = add_PT_i_np(ix)                            
#########################################################################################################
#                
#         if m_e < 0:
#             pass
#             
#         else:
#             loop_ws_kink_imag.append(iomega.imag[s])
#             
#             def dVx_dx_e(Vx_e, x_e):
#                    return [Vx_e[1], m_e.imag*Vx_e[0]]
#             
#             V0 = [1e-8, 1e-15]  
#             Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
#             left_solution_imag = Ls[:,0]*(iomega.imag[s]-wavenum*mach_i[0])/(iomega.imag[s]-wavenum*mach_e)  # Vx perturbation solution for left hand side
#             
#             left_P_solution_imag = p_e_const.imag*Ls[:,1]  # Pressure perturbation solution for left hand side
#             
#             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
#             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
#             
#             left_bound_vx_imag = left_solution_imag[-1] 
#                       
#             def dVx_dx_i(Vx_i, x_i):
#                    return [Vx_i[1], (-D_np(x_i).imag*Vx_i[1] - coeff_np(x_i).imag*Vx_i[0])]
#             
#                
#             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
#                   U = odeintz(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
#                   u1 = U[:,0] - left_bound_vx_imag
#                   return u1[-1] 
#                   
#             dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
#                  
#             Is_imag = odeintz(dVx_dx_i, [left_bound_vx_imag, dVxi], ix)
#             
#             inside_solution_imag = Is_imag[:,0]
#             
#             inside_P_solution_imag = p_i_const.imag*(Is_imag[:,1] - add_p_i_const.imag*Is_imag[:,0])
#             
#             normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
#             normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
#             
#             P_diff_loop_check_kink.append((left_P_solution[-1]-inside_P_solution[0]))
#             loop_sign_check_kink.append(P_diff_loop_check_kink[-1]*P_diff_loop_check_kink[-2])
#       
#             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol: 
#                    sol_omegas_kink1_imag.append(iomega.imag[s])
#                    sol_ks_kink1_imag.append(wavenum)
#                    loop_ws_kink_imag[:] = []
#                    loop_sign_check_kink[:] = [0]
#                    break
#                                    
#             elif loop_sign_check_kink[-1] < 0 and len(loop_ws_kink_imag)>1:   #If this is negative go back through and reduce gap
#  
#                    iomega = np.linspace(loop_ws_kink_imag[-2], loop_ws_kink_imag[-1], 20)  
#                    wavenum = wavenum
#                    #now repeat exact same process but in foccused omega range
#                    itt_num_imag =itt_num_imag +1
#                    loop_ws_kink_imag[:] = []              
#                    #locate_kink_imag(iomega, wavenum, itt_num_imag, omega_real)

 #################################################################################################  
  
  def locate_kink(omega, wavenum,itt_num):
      all_ws[:] = []
      all_ws_imag[:] = []
      itt_num = itt_num
                 
      for k in range(len(omega.real)):
        for s in range(len(omega.imag)):
         
         if itt_num > 6:  #100
            break
         
            
         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
         ix = np.linspace(-1., 1., 500.)  # inside slab x values

         m_e = ((((wavenum**2*vA_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)))
      
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2)/(((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))*((wavenum**2*c_e**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*mach_e))**2))       


#######################################################################################################           
         def m0(x):    
           return ((((wavenum**2*c_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)*((wavenum**2*vA_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenum**2*cT_i()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)))  
 
         m0_np=sym.lambdify(xx,m0(xx),"numpy")           
    ##############################################
         
         def D(x):    
           return (2.*wavenum*dU_i(x)*((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2/((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*c_i**2)) - ((wavenum**2*cT_i()**2)/((((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))**2)-wavenum**2*cT_i()**2)))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))  
         
         D_np=sym.lambdify(xx,D(xx),"numpy")
    ##############################################
    
         def coeff(x):
             return ((wavenum*ddU_i(x)/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))+((wavenum*dU_i(x)*D(x))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x))) - m0(x))

         coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
    ##############################################
    
         def P_Ti(x):    
           return (rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2)/(((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))*((wavenum**2*c_i**2)-((omega.real[k] + 1j*omega.imag[s])-(wavenum*U_i(x)))**2))) 
          
         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
         p_i_const = PT_i_np(ix)

         def add_P_Ti(x):
           return (-(wavenum*dU_i(x))/((omega.real[k] + 1j*omega.imag[s])-wavenum*U_i(x)))
            
         add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
         add_p_i_const = add_PT_i_np(ix)                            
########################################################################################################
                
         if m_e.real < 0:
             pass

         if m_e.imag < 0:
             pass
                          
         else:
             loop_ws_kink.append(omega.real[k])
             loop_ws_kink_imag.append(omega.imag[s])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
             
             V0 = [1e-8 + 1j*1e-8, 1e-15 + 1j*1e-15]  
             Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
             
             left_solution = Ls[:,0].real*(omega.real[k] - wavenum*mach_i[0])/(omega.real[k] - wavenum*mach_e)   
             left_solution_imag = Ls[:,0].imag*(omega.imag[s] - wavenum*mach_i[0])/(omega.imag[s] - wavenum*mach_e) 
                          
             left_P_solution = p_e_const.real*Ls[:,1].real  # Pressure perturbation solution for left hand side
             left_P_solution_imag = p_e_const.imag*Ls[:,1].imag  # Pressure perturbation solution for left hand side
             
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             
             left_bound_vx = left_solution[-1] 
             left_bound_vx_imag = left_solution_imag[-1] 
                       
             def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], (-D_np(x_i)*Vx_i[1] - coeff_np(x_i)*Vx_i[0])]
             
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi[0] + 1j*dVxi[1]], ix, printmessg=0)
                   u1 = U[:,0].real - left_bound_vx
                   u2 = U[:,0].imag - left_bound_vx_imag
                   return [u1[-1], u2[-1]]
                                       
             [dVxi, dVxi_imag] = fsolve(objective_dvxi, [1., 1.])    #zero is guess for roots of equation, maybe change to find more body modes??
     
             # now solve with optimal dvx
             
             Is = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi + 1j*dVxi_imag], ix)
             inside_solution = Is[:,0].real
             inside_solution_imag = Is[:,0].imag
             
             inside_P_solution = p_i_const.real*(Is[:,1].real - add_p_i_const.real*Is[:,0].real)
             inside_P_solution_imag = p_i_const.imag*(Is[:,1].imag - add_p_i_const.imag*Is[:,0].imag)
             
             P_diff_loop_check_kink.append((left_P_solution_imag[-1]-inside_P_solution_imag[0]))
             loop_sign_check_kink.append(P_diff_loop_check_kink[-1]*P_diff_loop_check_kink[-2])
             
        
             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol: 
                    sol_omegas_kink1.append(omega.real[k])
                    sol_omegas_kink1_imag.append(omega.imag[s])
                    sol_ks_kink1.append(wavenum)
                    sol_ks_kink1_imag.append(wavenum)
                    p_diff_sols_kink.append(abs(left_P_solution_imag[-1] - inside_P_solution_imag[0]))
                    loop_ws_kink[:] = []
                    loop_ws_kink_imag[:] = []
                    loop_sign_check_kink[:] = [0]
                    #iomega = np.linspace(-1., 1., 20)
                    #omega_real = (omega.real[k] + 1j*omega.imag[s])
                    #itt_num_imag = 0
                    #wavenum = wavenum
                    #locate_kink_imag(iomega, wavenum, itt_num_imag, omega_real)
                    break

#             if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol: 
#                    sol_omegas_kink1_imag.append(omega.imag[s])
#                    sol_ks_kink1_imag.append(wavenum)
#                    loop_ws_kink_imag[:] = []
#                    #iomega = np.linspace(-1., 1., 20)
#                    #omega_real = (omega.real[k] + 1j*omega.imag[s])
#                    #itt_num_imag = 0
#                    #wavenum = wavenum
#                    #locate_kink_imag(iomega, wavenum, itt_num_imag, omega_real)
#                    break
                                                        
             elif loop_sign_check_kink[-1] < 0 and len(loop_ws_kink)>1 and len(loop_ws_kink_imag)>1:   #If this is negative go back through and reduce gap
  
                    omega = np.linspace(loop_ws_kink[-2], loop_ws_kink[-1], 3) + 1j*np.linspace(loop_ws_kink_imag[-2], loop_ws_kink_imag[-1], 3)
                    wavenum = wavenum
                    #now repeat exact same process but in foccused omega range
                    itt_num =itt_num +1
                    loop_ws_kink[:] = []
                    loop_ws_kink_imag[:] = []              
                    locate_kink(omega, wavenum, itt_num)



#        if loop_sign_check_kink[-1] < 0 and len(loop_ws_kink)>1 and len(loop_ws_kink_imag)>1:   #If this is negative go back through and reduce gap
# 
#               omega = np.linspace(loop_ws_kink[-2], loop_ws_kink[-1], 3) + 1j*np.linspace(loop_ws_kink_imag[-2], loop_ws_kink_imag[-1], 3)
#               wavenum = wavenum
#               #now repeat exact same process but in foccused omega range
#               itt_num =itt_num +1
#               loop_ws_kink[:] = []
#               loop_ws_kink_imag[:] = []              
#               locate_kink(omega, wavenum, itt_num)   
                            
 ##############################################################################
  
  
  with stdout_redirected():

        for j in range(len(freq.real)):
          for i in range(len(freq.imag)):
            lx = np.linspace(-3.*2.*np.pi/wavenumber, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
            ix = np.linspace(-1, 1, 500)  # inside slab x values
      
            m_e = ((((wavenumber**2*vA_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)*((wavenumber**2*c_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)))
      
            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2)/(((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))*((wavenumber**2*c_e**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*mach_e))**2))


#######################################################################################################           
            def m0(x):    
              return ((((wavenumber**2*c_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)*((wavenumber**2*vA_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2))/((c_i**2+vA_i**2)*((wavenumber**2*cT_i()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)))  
    
            m0_np=sym.lambdify(xx,m0(xx),"numpy")           
       ##############################################
                                      
            def D(x):    
              return (2.*wavenumber*dU_i(x)*((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2/((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2)-wavenumber**2*c_i**2)) - ((wavenumber**2*cT_i()**2)/((((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))**2)-wavenumber**2*cT_i()**2)))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))) 
                          
            D_np=sym.lambdify(xx,D(xx),"numpy")

       ##############################################
       
            def coeff(x):
                return ((wavenumber*ddU_i(x)/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x)))+((wavenumber*dU_i(x)*D(x))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x))) - m0(x))
   
            coeff_np=sym.lambdify(xx,coeff(xx),"numpy")
       ##############################################  
   
            def P_Ti(x):    
              return (rho_i*(vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2)/(((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))*((wavenumber**2*c_i**2)-((freq.real[j] + 1j*freq.imag[i])-(wavenumber*U_i(x)))**2))) 
             
            PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
            p_i_const = PT_i_np(ix)


            def add_P_Ti(x):
              return (-(wavenumber*dU_i(x))/((freq.real[j] + 1j*freq.imag[i])-wavenumber*U_i(x)))
            
            add_PT_i_np=sym.lambdify(xx,add_P_Ti(xx),"numpy") 
            add_p_i_const = add_PT_i_np(ix)                         
########################################################################################################
                        
            if m_e.real < 0:
              pass               

            if m_e.imag < 0:
              pass               
                        
            else:
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8 + 1j*1e-8, 1e-15 + 1j*1e-15]
             
              Ls = odeintz(dVx_dx_e, V0, lx, printmessg=0)
              
              left_solution = Ls[:,0].real*(freq.real[j] - wavenumber*mach_i[0])/(freq.real[j] - wavenumber*mach_e) 
              left_solution_imag = Ls[:,0].imag*(freq.imag[i] - wavenumber*mach_i[0])/(freq.imag[i] - wavenumber*mach_e) 
              
              left_P_solution = p_e_const.real*Ls[:,1].real  # Pressure perturbation solution for left hand side
              left_P_solution_imag = p_e_const.imag*Ls[:,1].imag  # Pressure perturbation solution for left hand side
              
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              
              left_bound_vx = left_solution[-1] 
              left_bound_vx_imag = left_solution_imag[-1] 
                                          
              def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], (-D_np(x_i)*Vx_i[1] - coeff_np(x_i)*Vx_i[0])]
               
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi[0] + 1j*dVxi[1]], ix, printmessg=0)
                    u1 = U[:,0].real - left_bound_vx
                    u2 = U[:,0].imag - left_bound_vx_imag
                    return [u1[-1], u2[-1]]
                                        
              [dVxi, dVxi_imag] = fsolve(objective_dvxi, [1., 1.])    #zero is guess for roots of equation, maybe change to find more body modes??
      
              # now solve with optimal dvx
              
              Is = odeintz(dVx_dx_i, [left_bound_vx + 1j*left_bound_vx_imag, dVxi + 1j*dVxi_imag], ix)
              inside_solution = Is[:,0].real
              inside_solution_imag = Is[:,0].imag
              
              inside_P_solution = p_i_const.real*(Is[:,1].real - add_p_i_const.real*Is[:,0].real)
              inside_P_solution_imag = p_i_const.imag*(Is[:,1].imag - add_p_i_const.imag*Is[:,0].imag)
               
              P_diff_check_kink.append((left_P_solution_imag[-1]-inside_P_solution_imag[0]))
              sign_check_kink.append(P_diff_check_kink[-1]*P_diff_check_kink[-2])

      
              test_p_diff_kink.append(abs(left_P_solution[-1]) - abs(inside_P_solution[0]))
  
              all_ks_kink.append(wavenumber)
              all_ws_kink.append(freq.real[j])
              all_ws_kink_imag.append(freq.imag[i])
              
              
              if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
                    sol_omegas_kink1.append(freq.real[j])
                    sol_omegas_kink1_imag.append(freq.imag[i])
                    sol_ks_kink1.append(wavenumber)
                    sol_ks_kink1_imag.append(wavenumber)
                    p_diff_sols_kink.append(abs(left_P_solution_imag[-1] - inside_P_solution_imag[0]))
                    all_ws_kink[:] = []
                    all_ws_kink_imag[:] = []
                    #omega_real = (freq.real[j] + 1j*freq.imag[i])
                    #iomega = np.linspace(-1., 1., 20)
                    #wavenum = wavenumber
                    #itt_num_imag = 0
                    #locate_kink_imag(iomega, wavenum, itt_num_imag, omega_real)
                    

#              if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
#                    sol_omegas_kink1_imag.append(freq.imag[i])
#                    sol_ks_kink1_imag.append(wavenumber)
#                    all_ws_kink_imag[:] = []
#                    all_ws_kink[:] = []
#                                                   
              if sign_check_kink[-1] < 0: #and (round(inside_P_solution[0]) == round(inside_P_solution[-1])): 
                  if len(all_ws_kink)>1 and len(all_ws_kink_imag)>1:
                  
                     if all_ks_kink[-1] != all_ks_kink[-2]:
                       sign_check_kink[:] = []
                                          
                     else:
                           omega = np.linspace(all_ws_kink[-2], all_ws_kink[-1], 3) + 1j*np.linspace(all_ws_kink_imag[-2], all_ws_kink_imag[-1], 3)
                           wavenum = all_ks_kink[-1]
                  #        #now repeat exact same process but in foccused omega range
                           itt_num = 0
                           all_ws_kink[:] = []
                           all_ws_kink_imag[:] = []
                           locate_kink(omega, wavenum, itt_num) 


#          if (abs(left_P_solution_imag[-1] - inside_P_solution_imag[0])*100/max(abs(left_P_solution_imag[-1]), abs(inside_P_solution_imag[0]))) < p_tol:
#                sol_omegas_kink1_imag.append(freq.imag[i])
#                sol_ks_kink1_imag.append(wavenumber)
#                all_ws_kink_imag[:] = []
#                all_ws_kink[:] = []
#                  
#          if sign_check_kink[-1] < 0: #and (round(inside_P_solution[0]) == round(inside_P_solution[-1])): 
#              if len(all_ws_kink)>1 and len(all_ws_kink_imag)>1:
#              
#                 if all_ks_kink[-1] != all_ks_kink[-2]:
#                   sign_check_kink[:] = []
#                                      
#                 else:
#                       omega = np.linspace(all_ws_kink[-2], all_ws_kink[-1], 3) + 1j*np.linspace(all_ws_kink_imag[-2], all_ws_kink_imag[-1], 3)
#                       wavenum = all_ks_kink[-1]
#              #        #now repeat exact same process but in foccused omega range
#                       itt_num = 0
#                       #all_ws_kink[:] = []
#                       all_ws_kink_imag[:] = []
#                       locate_kink(omega, wavenum, itt_num)
                           
                                                                              
  kink_ks.put(sol_ks_kink1)
  kink_ws.put(sol_omegas_kink1)                    
  kink_ks_imag.put(sol_ks_kink1_imag)
  kink_ws_imag.put(sol_omegas_kink1_imag) 

wavenumber = np.linspace(0.01,1.5,30)      #(1.5, 1.8), 5


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    
    sausage_ws = multiprocessing.Queue()
    sausage_ks = multiprocessing.Queue()
    sausage_ws_imag = multiprocessing.Queue()
    sausage_ks_imag = multiprocessing.Queue()
        
    processes_kink = []
    
    kink_ws = multiprocessing.Queue()
    kink_ks = multiprocessing.Queue()
    kink_ws_imag = multiprocessing.Queue()
    kink_ks_imag = multiprocessing.Queue()
    
    for k in wavenumber:
      for i in range(len(speeds)-1):
     
         test_freq = np.linspace(speeds[i]*k, speeds[i+1]*k, 6.) + 1j*np.linspace(-0.3, 0.3, 6.)  
         
         task = multiprocessing.Process(target=sausage, args=(k, sausage_ws, sausage_ks, sausage_ws_imag, sausage_ks_imag, test_freq))
         task_kink = multiprocessing.Process(target=kink, args=(k, kink_ws, kink_ks, kink_ws_imag, kink_ks_imag, test_freq))
         
         processes.append(task)
         processes_kink.append(task_kink)
         task.start()
         task_kink.start()
        

    for p in processes:
        p.join()
        
    for p in processes_kink:
        p.join()

    sol_ks1 = [sausage_ks.get() for p in processes]
    sol_omegas1 = [sausage_ws.get() for p in processes]
    
    sol_ks1 = list(itertools.chain(*sol_ks1))   #flatten out the list of lists into one single list
    sol_omegas1 = list(itertools.chain(*sol_omegas1))
    
    sol_ks_kink1 = [kink_ks.get() for p in processes_kink]
    sol_omegas_kink1 = [kink_ws.get() for p in processes_kink]
    
    sol_ks_kink1 = list(itertools.chain(*sol_ks_kink1))
    sol_omegas_kink1 = list(itertools.chain(*sol_omegas_kink1))

##########
    sol_ks1_imag = [sausage_ks_imag.get() for p in processes]
    sol_omegas1_imag = [sausage_ws_imag.get() for p in processes]
    
    sol_ks1_imag = list(itertools.chain(*sol_ks1_imag))   #flatten out the list of lists into one single list
    sol_omegas1_imag = list(itertools.chain(*sol_omegas1_imag))
    
    sol_ks_kink1_imag = [kink_ks_imag.get() for p in processes_kink]
    sol_omegas_kink1_imag = [kink_ws_imag.get() for p in processes_kink]
    
    sol_ks_kink1_imag = list(itertools.chain(*sol_ks_kink1_imag))
    sol_omegas_kink1_imag = list(itertools.chain(*sol_omegas_kink1_imag))
##########

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)
    
sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)

sol_omegas1_imag = np.array(sol_omegas1_imag)
sol_ks1_imag = np.array(sol_ks1_imag)
    
sol_omegas_kink1_imag = np.array(sol_omegas_kink1_imag)
sol_ks_kink1_imag = np.array(sol_ks_kink1_imag)


with open('mihai_imaginary_test_imag.pickle', 'wb') as f:
    pickle.dump([sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1, sol_omegas1_imag, sol_ks1_imag, sol_omegas_kink1_imag, sol_ks_kink1_imag], f)
 

  
fig=plt.figure()
ax = plt.subplot(111)
#plt.title('Sausage Mode flow test')
plt.xlabel("$kx_{0}$", fontsize=20)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=25, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'b.', markersize=6.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e+U_e, color='k', linestyle='solid', label='_nolegend_')

ax.annotate( '$c_{Ti}$', xy=(max(wavenumber), cT_i()), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(5, cT_e()), fontsize=20)
ax.annotate( '$c_{e}$', xy=(max(wavenumber), c_e), fontsize=20)
#ax.annotate( '$c_{i}, c_{Ti}$', xy=(max(wavenumber), c_i), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(max(wavenumber), vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(max(wavenumber), vA_i), fontsize=20)
ax.annotate( '$c_e - U_e$', xy=(max(wavenumber), c_e+U_e), fontsize=20)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.85, box.height])

#ax.legend(('Method Solutions','Known Solutions'), loc='center left', bbox_to_anchor=(1, 0.3))

#plt.savefig("smooth_flow_dispersion_diagram_sausage_coronal.png")



fig=plt.figure()
ax = plt.subplot(111)
#plt.title('Kink Mode flow test')
plt.xlabel("$kx_{0}$", fontsize=20)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=25, rotation=0, labelpad=15)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=6.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e+U_e, color='k', linestyle='solid', label='_nolegend_')

ax.annotate( '$c_{Ti}$', xy=(max(wavenumber), cT_i()), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(max(wavenumber), cT_e()), fontsize=20)
ax.annotate( '$c_{e}$', xy=(max(wavenumber), c_e), fontsize=20)
#ax.annotate( '$c_{i},  c_{Ti}$', xy=(max(wavenumber), c_i), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(max(wavenumber), vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(max(wavenumber), vA_i), fontsize=20)
ax.annotate( '$c_e - U_e$', xy=(max(wavenumber), c_e+U_e), fontsize=20)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.85, box.height])

ax.legend(('Method Solutions','Known Solutions'), loc='center left', bbox_to_anchor=(1, 0.3))

#plt.savefig("smooth_flow_dispersion_diagram_kink_coronal.png")

#plt.figure()
#plt.xlabel("$\omega/k$")
#plt.ylabel("Pressure difference at boundary")
#plt.plot(test_w_k_kink, test_p_diff_kink, 'b.')
#plt.axvline(x=vA_e, color='r', linestyle='--')
#plt.axvline(x=vA_i, color='b', linestyle='--')
#plt.axvline(x=c_e, color='r')
#plt.axvline(x=c_i, color='b')
#plt.axvline(x=cT_e(), color='g')
#plt.axvline(x=cT_i(), color='g', linestyle='--')
#plt.axhline(y=c_e+U_e, color='k')
#plt.ylim(0, 1.5)
#
#plt.savefig("Nakariakov_test_pressure_diff_kink.png")


#plt.show()

#########   FULL TEST DISPERSION DIAGRAM      ################
plt.figure()
ax = plt.subplot(111)
plt.title('Width = symmetric Mihai complex')
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=22, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

ax.plot(sol_ks1_imag, (sol_omegas1_imag/sol_ks1_imag), 'rx', markersize=4.)
ax.plot(sol_ks_kink1_imag, (sol_omegas_kink1_imag/sol_ks_kink1_imag), 'bx', markersize=4.)

ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i(), color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=c_e+U_e, color='k', linestyle='solid', label='_nolegend_')

#ax.plot(test_k,test_W, 'x') #test plot

#ax.annotate( '$c_{e}$', xy=(Dmax, c_e), fontsize=20)
ax.annotate( '$c_{i},   c_{Ti}$', xy=(Dmax, c_i), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Dmax, vA_e), fontsize=20)
ax.annotate( '$c_{e},   c_{Te}$', xy=(Dmax, c_e), fontsize=20)   #coronal
ax.annotate( '$v_{Ai}$', xy=(Dmax, vA_i), fontsize=20)
#ax.annotate( '$c_e - U_e$', xy=(Dmax, c_e+U_e), fontsize=20)   

#ax.set_ylim(-vA_e-0.1, vA_e+0.1)
ax.set_yticks([])


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

#ax.legend(('Sausage','Kink'), loc='center left', bbox_to_anchor=(1, 0.7))

plt.savefig("mihai_imaginary_test_imag.png")
print('Simulation took {} seconds'.format(time.time() - starttime))
