# Import the required modules
import numpy as np
import scipy as sc
import matplotlib; matplotlib.use('agg') ##comment out to show figs
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sym
#from sympy import besselj, jn
import math
from math import log10, floor
from scipy.optimize import fsolve
import cmath
from matplotlib import animation
from scipy import interpolate
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
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
 
    
    ############################################################
##############      BEGIN SYMMETRIC SLAB DISPERSION DIAGRAM       ###############

# INITIALISE VALUES FOR SPECIFIC REGIME (E.G. CORONA / PHOTOSPHERE) AND PROFILES

## variables when no profile
c_i0 = 1.
vA_i0 = 1.2*c_i0   #1.2*c_i #-coronal        #1.9*c_i  -photospheric

vA_e = 3.*c_i0   #3.*c_i #-coronal        #0.8*c_i -photospheric
c_e = 0.4*c_i0    #0.4*c_i #- coronal          #1.3*c_i  -photospheric

cT_i0 = np.sqrt((c_i0**2 * vA_i0**2)/(c_i0**2 + vA_i0**2))

gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)

print('rho_e    =', rho_e)


P_0 = c_i0**2*rho_i0/gamma
P_e = c_e**2*rho_e/gamma

T_0 = P_0/rho_i0
T_e = P_e/rho_e

B_0 = vA_i0*np.sqrt(rho_i0)
B_e = vA_e*np.sqrt(rho_e)


P_tot_0 = P_0 + B_0**2/2.
P_tot_e = P_e + B_e**2/2.

print('PT_i   =', P_tot_0)
print('PT_e   =', P_tot_e)


Kmax = 3.5

ix = np.linspace(-1., 1., 1e6)  # inside slab x values     #was 500

#ix2 = np.linspace(-1, 0, 500.)  # inside slab x values

x0=0.  #mean
dx=0.9  #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density

#def profile(xx):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
#    return (rho_i0 - 0.00001*sym.exp(-(xx-x0)**2/dx**2))   # this is subtract but add for test


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   # this is subtract but add for test


#def profile(x):       # Bessel(like) Function
#    return 1./10.*(rho_i0*(sym.sin(10.*x)/x))/4. + 3./4.   


#def profile(x):       # Define the internal profile as a function of variable x   (Periodic threaded waveguide)
#    return (3.*rho_e + (rho_i0 - 3.*rho_e)*sym.cos(2.*sym.pi*x + sym.pi))       #2.28 = W ~ 0.6     #3. corresponds to W ~ 1.5



def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)


def T_i(x):
    return (P_i(x)/rho_i(x))

def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.


def vA_i(x):                  # Define the internal alfven speed
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

#def c_i(x):                  # Define the internal sound speed
#    return c_i0/profile(x)

#####
def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)
#####


def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))



rho_i_np=sym.lambdify(xx,rho_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
cT_i_np=sym.lambdify(xx,cT_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
c_i_np=sym.lambdify(xx,c_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
vA_i_np=sym.lambdify(xx,vA_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
P_i_np=sym.lambdify(xx,P_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
T_i_np=sym.lambdify(xx,T_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
B_i_np=sym.lambdify(xx,B_i(xx),"numpy")   #In order to evaluate we need to switch to numpy
PT_i_np=sym.lambdify(xx,PT_i(xx),"numpy")   #In order to evaluate we need to switch to numpy

#####   SPEEDS AT BOUNDARY    ###

rho_bound = rho_i_np(-1)
c_bound = c_i_np(-1)
vA_bound = vA_i_np(-1)
cT_bound = np.sqrt(c_bound**2 * vA_bound**2 / (c_bound**2 + vA_bound**2))

P_bound = P_i_np(-1)
print(P_bound)
internal_pressure = P_i_np(ix)
print(internal_pressure)

print('c_i0 ratio   =',   c_e/c_i0)
print('c_i ratio   =',   c_e/c_bound)
print('c_i boundary   =', c_bound)
print('vA_i boundary   =', vA_bound)
print('rho_i boundary   =', rho_bound)


#####################################################
#speeds = [c_i0, c_e, vA_i0, vA_e, cT_i0, cT_e()]

speeds = [1.0, cT_bound, 1.2, 1.3, 0.9]

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

#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################

#
#plt.figure()
##plt.title("width = 1.5")
##plt.xlabel("x")
##plt.ylabel("$\u03C1_{i}$",fontsize=25)
#ax = plt.subplot(331)
#ax.title.set_text("Density")
#ax.plot(ix,rho_i_np(ix));
#ax.annotate( '$\u03C1_{e}$', xy=(1, rho_e),fontsize=25)
#ax.annotate( '$\u03C1_{i}$', xy=(1, rho_i0),fontsize=25)
#ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot')
#ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot')
#
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$c_{i}$")
#ax2 = plt.subplot(332)
#ax2.title.set_text("$c_{i}$")
#ax2.plot(ix,c_i_np(ix));
#ax2.annotate( '$c_e$', xy=(1, c_e),fontsize=25)
#ax2.annotate( '$c_{i0}$', xy=(1, c_i0),fontsize=25)
#ax2.annotate( '$c_{B}$', xy=(1, c_bound), fontsize=20)
#ax2.axhline(y=c_i0, color='k', label='$c_{i0}$', linestyle='dashdot')
#ax2.axhline(y=c_e, color='k', label='$c_{e}$', linestyle='dashdot')
#ax2.fill_between(ix, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
#
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$vA_{i}$")
#ax3 = plt.subplot(333)
#ax3.title.set_text("$v_{Ai}$")
#ax3.plot(ix,vA_i_np(ix));
#ax3.annotate( '$v_{Ae}$', xy=(1, vA_e),fontsize=25)
#ax3.annotate( '$v_{Ai}$', xy=(1, vA_i0),fontsize=25)
#ax3.annotate( '$v_{AB}$', xy=(1, vA_bound), fontsize=20)
#ax3.axhline(y=vA_i0, color='k', label='$v_{Ai}$', linestyle='dashdot')
#ax3.axhline(y=vA_e, color='k', label='$v_{Ae}$', linestyle='dashdot')
#ax3.fill_between(ix, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
#
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$cT_{i}$")
#ax4 = plt.subplot(334)
#ax4.title.set_text("$c_{Ti}$")
#ax4.plot(ix,cT_i_np(ix));
#ax4.annotate( '$c_{Te}$', xy=(1, cT_e()),fontsize=25)
#ax4.annotate( '$c_{Ti}$', xy=(1, cT_i0),fontsize=25)
#ax4.annotate( '$c_{TB}$', xy=(1, cT_bound), fontsize=20)
#ax4.axhline(y=cT_i0, color='k', label='$c_{Ti}$', linestyle='dashdot')
#ax4.axhline(y=cT_e(), color='k', label='$c_{Te}$', linestyle='dashdot')
#ax4.fill_between(ix, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
#
#
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$P$")
#ax5 = plt.subplot(335)
#ax5.title.set_text("Gas Pressure")
#ax5.annotate( '$P_{0}$', xy=(1, P_0),fontsize=25)
#ax5.annotate( '$P_{e}$', xy=(1, P_e),fontsize=25)
#ax5.axhline(y=P_0, color='k', label='$P_{i}$', linestyle='dashdot')
#ax5.axhline(y=P_e, color='k', label='$P_{e}$', linestyle='dashdot')
#ax5.axhline(y=P_i_np(ix), color='b', label='$P_{e}$', linestyle='solid')
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$T$")
#ax6 = plt.subplot(336)
#ax6.title.set_text("Temperature")
#ax6.plot(ix,T_i_np(ix));
#ax6.annotate( '$T_{0}$', xy=(1, T_0),fontsize=25)
#ax6.annotate( '$T_{e}$', xy=(1, T_e),fontsize=25)
#ax6.axhline(y=T_0, color='k', label='$T_{i}$', linestyle='dashdot')
#ax6.axhline(y=T_e, color='k', label='$T_{e}$', linestyle='dashdot')
##ax.axhline(y=P_i_np(ix), color='b', label='$P_{e}$', linestyle='solid')
#
#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$B$")
#ax7 = plt.subplot(337)
#ax7.title.set_text("Mag Field")
#ax7.annotate( '$B_{0}$', xy=(1, B_0),fontsize=25)
#ax7.annotate( '$B_{e}$', xy=(1, B_e),fontsize=25)
#ax7.axhline(y=B_0, color='k', label='$B_{i}$', linestyle='dashdot')
#ax7.axhline(y=B_e, color='k', label='$B_{e}$', linestyle='dashdot')
#ax7.axhline(y=B_i_np(ix), color='b', label='$B_{e}$', linestyle='solid')
#
#
##plt.xlabel("x")
##plt.ylabel("$P_T$")
#ax8 = plt.subplot(338)
#ax8.title.set_text("Tot Pressure (uniform)")
#ax8.annotate( '$P_{T0}$', xy=(1, P_tot_0),fontsize=25,color='b')
#ax8.annotate( '$P_{Te}$', xy=(1, P_tot_e),fontsize=25,color='r')
#ax8.axhline(y=P_tot_0, color='b', label='$P_{Ti0}$', linestyle='dashdot')
#ax8.axhline(y=P_tot_e, color='r', label='$P_{Te}$', linestyle='dashdot')
#
##ax8.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
##ax8.axhline(y=B_i_np(ix), color='b', label='$P_{Te}$', linestyle='solid')
#
#ax9 = plt.subplot(339)
#ax9.title.set_text("Tot Pressure  (Modified)")
#ax9.annotate( '$P_{T0},  P_{Te}$', xy=(1, P_tot_0),fontsize=25,color='b')
#
#ax9.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
#
#plt.suptitle("Bessel", fontsize=14)
#plt.tight_layout()
##plt.savefig("coronal_profiles_periodic16wavelths_W15.png")
###plt.savefig("coronal_profiles_225.png")
#
#plt.show()
#exit()

################     BEGIN NEW METHOD FOR DISPERSION       ###########################
def round_to_1_sf(x):
   return round(x, -int(floor(log10(abs(x)))))

#########   mid k
#wavenumber = np.linspace(1.5,2.5,6)      #(1.5, 1.8), 5
#freq = np.logspace(0.43,0.56,10)-1    #1000    (0.001, 0.8)
#########

#########   small k
#wavenumber = np.linspace(0.3,0.9,7)      #(1.5, 1.8), 5
#freq = np.logspace(0.12,0.35,30)-1    #1000    (0.001, 0.8)
#########

#########  large k
#wavenumber = np.linspace(3.,3.5,6)      #(1.5, 1.8), 5
#freq = np.logspace(0.62,0.8,30)-1    #1000    (0.001, 0.8)   10
#########


#wavenumber = np.linspace(0.2,0.9,8)      #(1.5, 1.8), 5
#freq = np.logspace(0.1,0.37,65)-1    #1000    (0.001, 0.8)

#wavenumber = np.linspace(0.01,3.5,25)      #(1.5, 1.8), 5
#freq = np.logspace(0.001,0.85,100)-1    #1000    (0.001, 0.8)



## INITIALLY HAVE ONE PAIR OF W AND K AND WORK THROUGH WITH JUST THAT PAIR - IF NO MATCH MOVE ONTO NEXT PAIR


d = 1.

p_tol = 1.   # i.e 1% tolerance

test_sols_inside = []
test_sols_outside = []

sol_omegas = []
sol_ks = []
sol_omegas1 = []
sol_ks1 = []

test_w_k_sausage = []
test_p_diff_sausage = []
test_w_k_kink = []
test_p_diff_kink = []

P_diff_check = [0]
P_diff_loop_check = [0]
loop_sign_check = [0]
sign_check = [0]

all_ws = []
all_ks = []

loop_ws = []

p_diff_sols = []

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

image = []

def sausage(wavenumber, sausage_ws, sausage_ks, freq):
  #freq = np.logspace(0.001,1.,350)-1   #0.75 instead of 1?
  
  ##########    DEFINE A FUNCTION WHICH LOCATES SOLUTION    #####################
  
  def locate_sausage(omega, wavenum,itt_num):
      all_ws[:] = []
      itt_num = itt_num
  
                
      for k in range(len(omega)):
                   
         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain   
            
         #m_i = ((((wavenumber[i]**2*vA_i**2)-freq[j]**2)*((wavenumber[i]**2*c_i**2)-freq[j]**2))/((vA_i**2+c_i**2)*((wavenumber[i]**2*cT_i()**2)-freq[j]**2)))
         m_e = ((((wavenum**2*vA_e**2)-omega[k]**2)*((wavenum**2*c_e**2)-omega[k]**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-omega[k]**2)))
   
         #p_i_const = rho_i0*(vA_i0**2+c_i0**2)*((wavenum**2*cT_i0**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_i0**2)-omega[k]**2))
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_e**2)-omega[k]**2))
   
         
         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
                     
         def F(x):  
           return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2))/((wavenum**2*c_i(x)**2)-omega[k]**2))  
         
         F_np=sym.lambdify(xx,F(xx),"numpy")  
                   
         def dF(x):   #First derivative of profile in symbols    
           return sym.diff(F(x), x)  
  
         dF_np=sym.lambdify(xx,dF(xx),"numpy")
                     
         def m0(x):    
           return ((((wavenum**2*c_i(x)**2)-omega[k]**2)*((wavenum**2*vA_i(x)**2)-omega[k]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2)))  
          
         m0_np=sym.lambdify(xx,m0(xx),"numpy")
         
         def P_Ti(x):    
           return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_i(x)**2)-omega[k]**2)))  
          
         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
         p_i_const = PT_i_np(ix)
         #p_i_const = p_i_const[0]
         
         ######################################################
         
         if itt_num > 100:
            break 
                 
         if m_e < 0:
             pass
         
         else:
             
             loop_ws.append(omega[k])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
               
             V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
             Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
             left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
             left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
           
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             left_bound_vx = left_solution[-1] 
                      
             def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]
             
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                   u1 = U[:,0] + left_bound_vx
                   return u1[-1] 
                   
             dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
                 
             Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
             inside_solution = Is[:,0]
             #inside_P_solution = p_i_const*Is[:,1]
             
             inside_P_solution = np.multiply(p_i_const, Is[:,1])
                           
             normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
             normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
             
             P_diff_loop_check.append((left_P_solution[-1]-inside_P_solution[0]))
             loop_sign_check.append(P_diff_loop_check[-1]*P_diff_loop_check[-2])
              
             if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:
                    sol_omegas1.append(omega[k])
                    sol_ks1.append(wavenum)
                    p_diff_sols.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    loop_ws[:] = []
                    loop_sign_check[:] = [0]
                    break
                
             elif loop_sign_check[-1] < 0 and len(loop_ws)>1:   #If this is negative go back through and reduce gap
                 
                    omega = np.linspace(loop_ws[-2], loop_ws[-1], 3)
                    wavenum = wavenum
                    #now repeat exact same process but in foccused omega range
                    itt_num =itt_num +1
                    loop_ws[:] = []
                    locate_sausage(omega, wavenum, itt_num)
                    
                    
  
      
  ##############################################################################
  
            
  with stdout_redirected():
        for j in range(len(freq)):
            loop_ws[:] = [] 
                      
            lx = np.linspace(-3.*2.*np.pi/wavenumber, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
  
            m_e = ((((wavenumber**2*vA_e**2)-freq[j]**2)*((wavenumber**2*c_e**2)-freq[j]**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-freq[j]**2)))
      
            #p_i_const = rho_i0*(vA_i0**2+c_i0**2)*((wavenumber**2*cT_i0**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_i0**2)-freq[j]**2))
            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_e**2)-freq[j]**2))
   
            
            ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
                        
            def F(x):  
              return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2))/((wavenumber**2*c_i(x)**2)-freq[j]**2))  
   
            F_np=sym.lambdify(xx,F(xx),"numpy")   
           
            def dF(x):   #First derivative of profile in symbols    
              return sym.diff(F(x), x)
            
            dF_np=sym.lambdify(xx,dF(xx),"numpy")    
              
            def m0(x):    
              return ((((wavenumber**2*c_i(x)**2)-freq[j]**2)*((wavenumber**2*vA_i(x)**2)-freq[j]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2))) 
              
            m0_np=sym.lambdify(xx,m0(xx),"numpy")  
            
            def P_Ti(x):    
              return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_i(x)**2)-freq[j]**2)))  
          
            PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
            p_i_const = PT_i_np(ix)
            #p_i_const = p_i_const[0] 
             
            ######################################################
  
            if m_e < 0:
              pass             
                  
            else:               
              all_ks.append(wavenumber)
              all_ws.append(freq[j])
              
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
              Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
              left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
              left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
            
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              left_bound_vx = left_solution[-1] 
                    
              def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           
                 
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                    u1 = U[:,0] + left_bound_vx
                    return u1[-1] 
                    
              dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
      
              # now solve with optimal dvx
              
              Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
              inside_solution = Is[:,0]
              #inside_P_solution = p_i_const*Is[:,1]
              
              inside_P_solution = np.multiply(p_i_const, Is[:,1])              
              
              normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
              normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
            
              P_diff_check.append((left_P_solution[-1]-inside_P_solution[0]))
              sign_check.append(P_diff_check[-1]*P_diff_check[-2])
            
              if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:
                    sol_omegas1.append(freq[j])
                    sol_ks1.append(wavenumber)
                    p_diff_sols.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    all_ws[:] = []
                    
             
              if sign_check[-1] < 0: #and (round(inside_P_solution[0]) == round(inside_P_solution[-1])): 
                  if len(all_ws)>1:
                  
                     if all_ks[-1] != all_ks[-2]:
                       sign_check[:] = []
                       
                     else:           
                       omega = np.linspace(all_ws[-2], all_ws[-1], 3)
                       wavenum = all_ks[-1]
              #        #now repeat exact same process but in foccused omega range
                       itt_num = 0
                       all_ws[:] = []
                       locate_sausage(omega, wavenum, itt_num) 

  sausage_ks.put(sol_ks1)
  sausage_ws.put(sol_omegas1)

##################################################################################################################################
##################################################################################################################################
########   TEST KINK   ##########

sol_omegas_kink = []
sol_ks_kink = []
sol_omegas_kink1 = []
sol_ks_kink1 = []
P_diff_check_kink = [0]
P_diff_loop_check_kink = []
loop_ws_kink = []

all_ws_kink = []
all_ks_kink = []

P_diff_check_kink = [0]
P_diff_loop_check_kink = [0]
loop_sign_check_kink = [0]
sign_check_kink = [0]
p_diff_sols_kink = []



def kink(wavenumber, kink_ws, kink_ks, freq):
  #freq = np.logspace(0.001,1.,350)-1   #0.75 instead of 1?

  def locate_kink(omega, wavenum,itt_num):
      all_ws[:] = []
      itt_num = itt_num
                    
      for k in range(len(omega)):
      
         if itt_num > 100:
            break
         
         lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
            
         #m_i = ((((wavenumber[i]**2*vA_i**2)-freq[j]**2)*((wavenumber[i]**2*c_i**2)-freq[j]**2))/((vA_i**2+c_i**2)*((wavenumber[i]**2*cT_i()**2)-freq[j]**2)))
         m_e = ((((wavenum**2*vA_e**2)-omega[k]**2)*((wavenum**2*c_e**2)-omega[k]**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-omega[k]**2)))
   
         #p_i_const = rho_i0*(vA_i0**2+c_i0**2)*((wavenum**2*cT_i0**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_i0**2)-omega[k]**2))
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_e**2)-omega[k]**2))
  
         
         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
                     
         def F(x):  
           return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2))/((wavenum**2*c_i(x)**2)-omega[k]**2))  
         
         F_np=sym.lambdify(xx,F(xx),"numpy")  
                   
         def dF(x):   #First derivative of profile in symbols    
           return sym.diff(F(x), x)  
  
         dF_np=sym.lambdify(xx,dF(xx),"numpy")
                     
         def m0(x):    
           return ((((wavenum**2*c_i(x)**2)-omega[k]**2)*((wavenum**2*vA_i(x)**2)-omega[k]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2)))  
          
         m0_np=sym.lambdify(xx,m0(xx),"numpy")
         
         def P_Ti(x):    
           return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum**2*cT_i(x)**2)-omega[k]**2)/(omega[k]*((wavenum**2*c_i(x)**2)-omega[k]**2)))  
          
         PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
         p_i_const = PT_i_np(ix)
         #p_i_const = p_i_const[0]
         
         ######################################################  
                     
         if m_e < 0:
             pass
         
         else:
             loop_ws_kink.append(omega[k])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
               
             V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
             Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
             left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
             left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
           
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             left_bound_vx = left_solution[-1] 
                    
             def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                   u1 = U[:,0] - left_bound_vx
                   return u1[-1] 
                   
             dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
                 
             Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
             inside_solution = Is[:,0]
             #inside_P_solution = p_i_const*Is[:,1]

             inside_P_solution = np.multiply(p_i_const, Is[:,1])

             normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
             normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
             
             P_diff_loop_check_kink.append((left_P_solution[-1]-inside_P_solution[0]))
             loop_sign_check_kink.append(P_diff_loop_check_kink[-1]*P_diff_loop_check_kink[-2])           
                  
             if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:
                    sol_omegas_kink1.append(omega[k])
                    sol_ks_kink1.append(wavenum)
                    p_diff_sols_kink.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    loop_ws_kink[:] = []
                    loop_sign_check_kink[:] = [0]
                    break
                
             elif loop_sign_check_kink[-1] < 0 and len(loop_ws_kink)>1:   #If this is negative go back through and reduce gap
  
                    omega = np.linspace(loop_ws_kink[-2], loop_ws_kink[-1], 3)
                    wavenum = wavenum
                    #now repeat exact same process but in foccused omega range
                    itt_num =itt_num +1
                    loop_ws_kink[:] = []              
                    locate_kink(omega, wavenum, itt_num)
     
      
  ##############################################################################
  
            
  with stdout_redirected():
        for j in range(len(freq)):
      
            lx = np.linspace(-3.*2.*np.pi/wavenumber, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
  
            m_e = ((((wavenumber**2*vA_e**2)-freq[j]**2)*((wavenumber**2*c_e**2)-freq[j]**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-freq[j]**2)))
      
            #p_i_const = rho_i0*(vA_i0**2+c_i0**2)*((wavenumber**2*cT_i0**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_i0**2)-freq[j]**2))
            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_e**2)-freq[j]**2))
   
            
            ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
                        
            def F(x):  
              return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2))/((wavenumber**2*c_i(x)**2)-freq[j]**2))  
   
            F_np=sym.lambdify(xx,F(xx),"numpy")   
           
            def dF(x):   #First derivative of profile in symbols    
              return sym.diff(F(x), x)
            
            dF_np=sym.lambdify(xx,dF(xx),"numpy")    
              
            def m0(x):    
              return ((((wavenumber**2*c_i(x)**2)-freq[j]**2)*((wavenumber**2*vA_i(x)**2)-freq[j]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2))) 
              
            m0_np=sym.lambdify(xx,m0(xx),"numpy")  
            
            def P_Ti(x):    
              return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenumber**2*cT_i(x)**2)-freq[j]**2)/(freq[j]*((wavenumber**2*c_i(x)**2)-freq[j]**2)))  
          
            PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
            p_i_const = PT_i_np(ix)
            #p_i_const = p_i_const[0] 
             
            ######################################################      
            
      
            if m_e < 0:
              pass              
                  
            else: 
              
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
              Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
              left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
              left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
            
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              left_bound_vx = left_solution[-1] 
                                 
              def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]            
                 
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                    u1 = U[:,0] - left_bound_vx
                    return u1[-1] 
                    
              dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??
      
              # now solve with optimal dvx
              
              Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
              inside_solution = Is[:,0]
              #inside_P_solution = p_i_const*Is[:,1]

              inside_P_solution = np.multiply(p_i_const, Is[:,1])
              
              normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
              normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
              
              P_diff_check_kink.append((left_P_solution[-1]-inside_P_solution[0]))
              sign_check_kink.append(P_diff_check_kink[-1]*P_diff_check_kink[-2])
             
              test_p_diff_kink.append(abs(left_P_solution[-1]) - abs(inside_P_solution[0]))
              
              #P_diff_check.append((inside_P_solution[0]-left_P_solution[-1])/max(abs(left_P_solution[-1]), abs(inside_P_solution[0])))
              P_diff_check_kink.append((left_P_solution[-1]-inside_P_solution[0]))
  
              all_ks_kink.append(wavenumber)
              all_ws_kink.append(freq[j])
                                   
              
              if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:
                    sol_omegas_kink1.append(freq[j])
                    sol_ks_kink1.append(wavenumber)
                    p_diff_sols_kink.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    all_ws_kink[:] = []
                    
             
              if sign_check_kink[-1] < 0: #and (round(inside_P_solution[0]) == round(inside_P_solution[-1])): 
                  if len(all_ws_kink)>1:
                  
                     if all_ks_kink[-1] != all_ks_kink[-2]:
                       sign_check_kink[:] = []
                                          
                     else:           
                       omega = np.linspace(all_ws_kink[-2], all_ws_kink[-1], 3)
                       wavenum = all_ks_kink[-1]
              #        #now repeat exact same process but in foccused omega range
                       itt_num = 0
                       all_ws_kink[:] = []
                       locate_kink(omega, wavenum, itt_num) 
               
  kink_ks.put(sol_ks_kink1)
  kink_ws.put(sol_omegas_kink1)                    
                                             
 
#wavenumber = np.linspace(0.01,3.5,110.)      # was 100   

wavenumber = np.linspace(0.001,0.75, 25.)      #(1.5, 1.8), 5     


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    
    sausage_ws = multiprocessing.Queue()
    sausage_ks = multiprocessing.Queue()
    
    processes_kink = []
    
    kink_ws = multiprocessing.Queue()
    kink_ks = multiprocessing.Queue()

    
    for k in wavenumber:
      for i in range(len(speeds)-1):
     
         test_freq = np.linspace(speeds[i]*k, speeds[i+1]*k, 25.)   #was 60
         
         task = multiprocessing.Process(target=sausage, args=(k, sausage_ws, sausage_ks, test_freq))
         task_kink = multiprocessing.Process(target=kink, args=(k, kink_ws, kink_ks, test_freq))
         
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

 

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)

sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)


#with open('Bessellike.pickle', 'wb') as f:
#    pickle.dump([sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1], f)


with open('width09_zoom_coronal.pickle', 'wb') as f:
    pickle.dump([sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1], f)
 
 
#################    BEGIN FULL PLOTS   ################################
plt.figure()
#ax = plt.subplot(111)
fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)
plt.title("Sausage Mode  $A = 0.0$")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'b.', markersize=4.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=max(c_i_np(ix)), color='r', linestyle='solid', label='_nolegend_')
ax.axhline(y=max(cT_i_np(ix)), color='b', linestyle='solid', label='_nolegend_')
#ax.plot(test_k,test_W, 'x') #test plot

ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
#ax.annotate( '$c_{e},   c_{Te}$', xy=(Kmax, c_e), fontsize=20)  #Coronal
ax.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax.annotate( '$c_{i max}$', xy=(max(wavenumber), max(c_i_np(ix))), fontsize=20)
ax.annotate( '$c_{Ti max}$', xy=(max(wavenumber), max(cT_i_np(ix))), fontsize=20)


ax2.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=max(c_i_np(ix)), color='r', linestyle='solid', label='_nolegend_')
ax2.axhline(y=max(cT_i_np(ix)), color='b', linestyle='solid', label='_nolegend_')
#ax.plot(test_k,test_W, 'x') #test plot

ax2.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax2.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
ax2.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax2.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax2.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax2.annotate( '$c_{i max}$', xy=(max(wavenumber), max(c_i_np(ix))), fontsize=20)
ax2.annotate( '$c_{Ti max}$', xy=(max(wavenumber), max(cT_i_np(ix))), fontsize=20)

ax.set_ylim(0.6, 1.4)  
ax2.set_ylim(1.8, 2.)  # remove blank space

ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()
ax2.set_yticks([])
ax.set_yticks([])

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
fig.tight_layout()

box = ax.get_position()
box2 = ax2.get_position()
ax.set_position([box.x0, box.y0, box.width*0.85, box.height*1.8])
ax2.set_position([box2.x0, box2.y0+0.3, box2.width*0.85, box2.height*0.2])

plt.savefig("test_dispersion_diagram_sausage_coronal.png")

plt.figure()
#ax = plt.subplot(111)
fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)
plt.title("Kink Mode  $A = 0.0$")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
#ax.plot(sol_ks_kink, (sol_omegas_kink/sol_ks_kink), 'b.', markersize=5.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=max(c_i_np(ix)), color='r', linestyle='solid', label='_nolegend_')
ax.axhline(y=max(cT_i_np(ix)), color='b', linestyle='solid', label='_nolegend_')
#ax.plot(test_k,test_W, 'x') #test plot

ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
#ax.annotate( '$c_{e},   c_{Te}$', xy=(Kmax, c_e), fontsize=20)  #Coronal
ax.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax.annotate( '$c_{i max}$', xy=(max(wavenumber), max(c_i_np(ix))), fontsize=20)
ax.annotate( '$c_{Ti max}$', xy=(max(wavenumber), max(cT_i_np(ix))), fontsize=20)


#ax2.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax2.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=max(c_i_np(ix)), color='r', linestyle='solid', label='_nolegend_')
ax2.axhline(y=max(cT_i_np(ix)), color='b', linestyle='solid', label='_nolegend_')
#ax.plot(test_k,test_W, 'x') #test plot

ax2.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax2.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
ax2.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax2.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax2.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax2.annotate( '$c_{i max}$', xy=(max(wavenumber), max(c_i_np(ix))), fontsize=20)
ax2.annotate( '$c_{Ti max}$', xy=(max(wavenumber), max(cT_i_np(ix))), fontsize=20)

ax.set_ylim(0.6, 1.4)  
ax2.set_ylim(1.8, 2.)  # remove blank space

ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()
ax2.set_yticks([])
ax.set_yticks([])

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
fig.tight_layout()

box = ax.get_position()
box2 = ax2.get_position()
ax.set_position([box.x0, box.y0, box.width*0.85, box.height*1.8])
ax2.set_position([box2.x0, box2.y0+0.3, box2.width*0.85, box2.height*0.2])

plt.savefig("test_dispersion_diagram_kink_coronal.png")


#########   FULL TEST DISPERSION DIAGRAM      ################
plt.figure()
#ax = plt.subplot(111)
#fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)
ax = plt.subplot(111)
#plt.title("Bessel-like")
plt.title("W = 0.9")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( '$c_{e},   c_{Te}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)

ax.set_ylim(0., vA_e+0.1)
ax.set_yticks([])


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

#plt.savefig("Bessellike_dispersion_diagram_FULL.png")
plt.savefig("smoth_width_coronal_09_dispersion_diagram_ZOOM.png")


###################
#plt.figure()
#ax = plt.subplot(111)
#plt.xlabel("$kx_{0}$", fontsize=18)
#plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
#ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
#ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=max(c_i_np(ix)), color='r', linestyle='solid', label='_nolegend_')
#ax.axhline(y=max(cT_i_np(ix)), color='b', linestyle='solid', label='_nolegend_')
#
#ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
##ax.annotate( '$c_{e},   c_{Te}$', xy=(Kmax, c_e), fontsize=20)  #Coronal
#ax.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
#ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
#ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
#ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
#ax.annotate( '$c_{i max}$', xy=(max(wavenumber), max(c_i_np(ix))), fontsize=20)
#ax.annotate( '$c_{Ti max}$', xy=(max(wavenumber), max(cT_i_np(ix))), fontsize=20)
#
#
#plt.savefig("test_dispersion_diagram_all_sols.png")
print('Simulation took {} seconds'.format(time.time() - starttime))
