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




########################################
c_i0 = 1.
vA_i0 = 2.*c_i0  #-photospheric

vA_e = 0.5*c_i0  #-photospheric
c_e = 1.5*c_i0    #-photospheric

cT_i0 = np.sqrt((c_i0**2 * vA_i0**2)/(c_i0**2 + vA_i0**2))
cT_e = np.sqrt((c_e**2 * vA_e**2)/(c_e**2 + vA_e**2))


gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)


c_kink = np.sqrt(((rho_i0*vA_i0**2)+(rho_e*vA_e**2))/(rho_i0+rho_e))


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

Kmax = 4.

ix = np.linspace(-1, 1, 500)  # inside slab x values
ix2 = np.linspace(-1, 0, 500)  # inside slab x values

x0=0.  #mean
dx=0.9  #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   



#def profile(x):       # Define the internal profile as a function of variable x   (Besslike)
#    return (-1./10.*(rho_i0*(sym.sin(10.*x)/x)) + rho_i0)/4. + rho_i0  



#a = 1.   #inhomogeneity width for epstein profile
#def profile(x):       # Define the internal profile as a function of variable x   (Epstein Profile)
#    return ((rho_i0 - rho_e)/(sym.cosh(x/a)**4)**2 + rho_e)


def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)


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


def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)


def T_i(x):
    return (P_i(x)/rho_i(x))

def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.




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
c_kink_bound = np.sqrt(((rho_bound*vA_bound**2)+(rho_e*vA_e**2))/(rho_bound+rho_e))

P_bound = P_i_np(-1)
print(P_bound)
internal_pressure = P_i_np(ix)
print(internal_pressure)

print('c_i0 ratio   =',   c_e/c_i0)
print('c_i ratio   =',   c_e/c_bound)
print('c_i boundary   =', c_bound)
print('vA_i boundary   =', vA_bound)
print('rho_i boundary   =', rho_bound)


#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################
#
#
#plt.figure()
#plt.title("width = 1.5")
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
##ax.plot(ix,P_i_np(ix));
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
##ax.plot(ix,B_i_np(ix));
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
##ax.plot(ix,B_i_np(ix));
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
##ax.plot(ix,B_i_np(ix));
#ax9.annotate( '$P_{T0},  P_{Te}$', xy=(1, P_tot_0),fontsize=25,color='b')
#
#ax9.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
##ax8.axhline(y=B_i_np(ix), color='b', label='$P_{Te}$', linestyle='solid')
#
#plt.suptitle("W = 0.9", fontsize=14)
#plt.tight_layout()
##plt.savefig("photospheric_profiles_09.png")
#
##plt.show()
##exit()
wavenumber = np.linspace(0.01,4.,20)      #(1.5, 1.8), 5     


########   READ IN VARIABLES    #########

with open('Cylindrical_photospheric_width_09.pickle', 'rb') as f:
    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)


with open('Cylindrical_photospheric_width_09_slowmodes.pickle', 'rb') as f:
    sol_omegas1_zoom, sol_ks1_zoom, sol_omegas_kink1_zoom, sol_ks_kink1_zoom = pickle.load(f)




sol_omegas1 = np.concatenate((sol_omegas1, sol_omegas1_zoom), axis=None)
sol_ks1 = np.concatenate((sol_ks1, sol_ks1_zoom), axis=None)
sol_omegas_kink1 = np.concatenate((sol_omegas_kink1, sol_omegas_kink1_zoom), axis=None)
sol_ks_kink1 = np.concatenate((sol_ks_kink1, sol_ks_kink1_zoom), axis=None)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas1 = [x for _,x in sorted(zip(sol_ks1,sol_omegas1))]
sol_ks1 = np.sort(sol_ks1)

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)

sol_omegas_kink1 = [x for _,x in sorted(zip(sol_ks_kink1,sol_omegas_kink1))]
sol_ks_kink1 = np.sort(sol_ks_kink1)

sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)



### FAST SURFACE    ( c_i  <  w/k  <  c_e )
fast_surf_sausage_omega = []
fast_surf_sausage_k = []
fast_surf_kink_omega = []
fast_surf_kink_k = []

### SLOW SURFACE    ( vA_e  <  w/k  <  cT_i )
slow_surf_sausage_omega = []
slow_surf_sausage_k = []
slow_surf_kink_omega = []
slow_surf_kink_k = []

### BODY    ( vA_e  <  w/k  <  cT_i )
body_sausage_omega = []
body_sausage_k = []
body_kink_omega = []
body_kink_k = []

sausage_extra_branch_k = []
sausage_extra_branch_omega = []

for i in range(len(sol_ks1)):   #sausage mode
  v_phase = sol_omegas1[i]/sol_ks1[i]
      
  if v_phase > c_kink_bound and v_phase < c_e:  
      fast_surf_sausage_omega.append(sol_omegas1[i])
      fast_surf_sausage_k.append(sol_ks1[i])
      
      
  elif v_phase > 0.8 and v_phase < cT_bound:
      slow_surf_sausage_omega.append(sol_omegas1[i])
      slow_surf_sausage_k.append(sol_ks1[i])

  elif v_phase > cT_i0 and v_phase < c_i0:
      body_sausage_omega.append(sol_omegas1[i])
      body_sausage_k.append(sol_ks1[i])
            

for i in range(len(sol_ks_kink1)):   #kink mode
  v_phase = sol_omegas_kink1[i]/sol_ks_kink1[i]
      
  if v_phase > c_kink_bound and v_phase < 1.4:  
      fast_surf_kink_omega.append(sol_omegas_kink1[i])
      fast_surf_kink_k.append(sol_ks_kink1[i])
      
      
  elif v_phase > 0.8 and v_phase < cT_bound:
      slow_surf_kink_omega.append(sol_omegas_kink1[i])
      slow_surf_kink_k.append(sol_ks_kink1[i])

  elif v_phase > cT_i0 and v_phase < c_i0:
      body_kink_omega.append(sol_omegas_kink1[i])
      body_kink_k.append(sol_ks_kink1[i])
 
 
 
index_to_remove = []
for i in range(len(fast_surf_sausage_omega)-1):
  ph_diff = abs((fast_surf_sausage_omega[i+1]/fast_surf_sausage_k[i+1]) - (fast_surf_sausage_omega[i]/fast_surf_sausage_k[i]))
  
  if ph_diff > 0.03:  
      index_to_remove.append(i)
      
fast_surf_sausage_omega = np.delete(fast_surf_sausage_omega, index_to_remove)
fast_surf_sausage_k = np.delete(fast_surf_sausage_k, index_to_remove)

     
      
fast_surf_sausage_omega = np.array(fast_surf_sausage_omega)
fast_surf_sausage_k = np.array(fast_surf_sausage_k)
fast_surf_kink_omega = np.array(fast_surf_kink_omega)
fast_surf_kink_k = np.array(fast_surf_kink_k)
slow_surf_sausage_omega = np.array(slow_surf_sausage_omega)
slow_surf_sausage_k = np.array(slow_surf_sausage_k)
slow_surf_kink_omega = np.array(slow_surf_kink_omega)
slow_surf_kink_k = np.array(slow_surf_kink_k)

body_kink_omega = np.array(body_kink_omega)
body_kink_k = np.array(body_kink_k)
body_sausage_omega = np.array(body_sausage_omega)
body_sausage_k = np.array(body_sausage_k)
           
           
           
########################### sausage body
body_sausage_branch1_omega = [0.01*cT_i0, 0.05*cT_i0, 0.1*cT_i0, 0.2*cT_i0, 0.3*cT_i0]
body_sausage_branch1_k = [0.01, 0.05, 0.1, 0.2, 0.3]
body_sausage_branch2_omega = [0.01*cT_i0, 0.1*cT_i0, 0.2*cT_i0, 0.3*cT_i0]
body_sausage_branch2_k = [0.01, 0.1, 0.2, 0.3] 


for i in range(len(body_sausage_omega)):   #sausage mode
  v_phase = body_sausage_omega[i]/body_sausage_k[i]
  
  if v_phase > cT_i0 and v_phase < c_i0 and body_sausage_k[i] < 1.9:
      body_sausage_branch1_omega.append(body_sausage_omega[i])
      body_sausage_branch1_k.append(body_sausage_k[i])

  if v_phase > 0.92 and v_phase < c_i0 and body_sausage_k[i] > 2.:
      body_sausage_branch1_omega.append(body_sausage_omega[i])
      body_sausage_branch1_k.append(body_sausage_k[i])

  if v_phase < 0.92 and v_phase > 0.906 and body_sausage_k[i] > 2.3:
      body_sausage_branch2_omega.append(body_sausage_omega[i])
      body_sausage_branch2_k.append(body_sausage_k[i])

      
#body_sausage_branch1_omega = body_sausage_branch1_omega[::-1]
#body_sausage_branch1_k = body_sausage_branch1_k[::-1]


#index_to_remove = []
#for i in range(len(body_sausage_branch1_omega)-1):
#    ph_diff = abs((body_sausage_branch1_omega[i+1]/body_sausage_branch1_k[i+1]) - (body_sausage_branch1_omega[i]/body_sausage_branch1_k[i]))
#   
#    if ph_diff > 0.015:
#      index_to_remove.append(i+1)
#      body_sausage_branch2_omega.append(body_sausage_omega[i+1])
#      body_sausage_branch2_k.append(body_sausage_k[i+1])              
#

 
body_sausage_branch1_omega = np.array(body_sausage_branch1_omega)
body_sausage_branch1_k = np.array(body_sausage_branch1_k)
body_sausage_branch2_omega = np.array(body_sausage_branch2_omega)
body_sausage_branch2_k = np.array(body_sausage_branch2_k) 

#body_sausage_branch1_omega = body_sausage_branch1_omega[::-1]
#body_sausage_branch1_k = body_sausage_branch1_k[::-1]



################################################


#   sausage body polyfit
SB_phase = body_sausage_branch1_omega/body_sausage_branch1_k  
SB_k = body_sausage_branch1_k 
SB_k_new = np.linspace(SB_k[0], SB_k[-1], num=len(SB_k)*10)      #SB_k[-1]

SB_coefs = poly.polyfit(SB_k, SB_phase, 4)
SB_ffit = poly.polyval(SB_k_new, SB_coefs)

if len(body_sausage_branch2_omega) > 1:
  SB2_phase = body_sausage_branch2_omega/body_sausage_branch2_k  
  SB2_k = body_sausage_branch2_k 
  SB2_k_new = np.linspace(SB2_k[0], Kmax, num=len(SB2_k)*10)    #did finish at Kmax other than W=2.5     SB2_k[-1]
  
  SB2_coefs = poly.polyfit(SB2_k, SB2_phase, 1)   #was 6    # 1 good for W > 3
  SB2_ffit = poly.polyval(SB2_k_new, SB2_coefs)

    
################   kink body  ################
body_kink_branch1_omega = [0.01*cT_i0, 0.1*cT_i0, 0.2*cT_i0, 0.3*cT_i0]
body_kink_branch1_k = [0.01, 0.1, 0.2, 0.3]
body_kink_branch2_omega = [0.01*cT_i0, 0.1*cT_i0, 0.25*cT_i0, 0.45*cT_i0]
body_kink_branch2_k = [0.01, 0.1, 0.25, 0.45] 


for i in range(len(body_kink_omega)):   #kink mode
  v_phase = body_kink_omega[i]/body_kink_k[i]
  
  if v_phase > 0.912 and v_phase < c_i0 and body_kink_k[i] > 2.2:
      body_kink_branch1_omega.append(body_kink_omega[i])
      body_kink_branch1_k.append(body_kink_k[i])
      
  if v_phase < 0.912 and v_phase > cT_i0 and body_kink_k[i] < 2.2:
      body_kink_branch1_omega.append(body_kink_omega[i])
      body_kink_branch1_k.append(body_kink_k[i])

  if v_phase < 0.915 and v_phase > cT_i0 and body_kink_k[i] > 2.7:
      body_kink_branch2_omega.append(body_kink_omega[i])
      body_kink_branch2_k.append(body_kink_k[i])


#body_kink_branch1_omega = body_kink_branch1_omega[::-1]
#body_kink_branch1_k = body_kink_branch1_k[::-1]


FSK_phase = fast_surf_kink_omega/fast_surf_kink_k  #combined fast surface and body kink
FSK_k = fast_surf_kink_k
FSK_k_new = np.linspace(0, Kmax, num=len(FSK_k)*10)

FSK_coefs = poly.polyfit(FSK_k, FSK_phase, 8)    #was 6
FSK_ffit = poly.polyval(FSK_k_new, FSK_coefs)

##############################################################


      
body_kink_branch1_omega = np.array(body_kink_branch1_omega)
body_kink_branch1_k = np.array(body_kink_branch1_k)
body_kink_branch2_omega = np.array(body_kink_branch2_omega)
body_kink_branch2_k = np.array(body_kink_branch2_k) 


#   kink body polyfit
KB_phase = body_kink_branch1_omega/body_kink_branch1_k  
KB_k = body_kink_branch1_k 
KB_k_new = np.linspace(KB_k[0], Kmax, num=len(KB_k)*1)    #10 works    # 1 for W=2.5     #Kmax for w =< 1    KB_k[-1] otherwise   

KB_coefs = poly.polyfit(KB_k, KB_phase, 3)     # use 6    # use 1 for W <=1.75
KB_ffit = poly.polyval(KB_k_new, KB_coefs)

if len(body_kink_branch2_omega) > 1:
   KB2_phase = body_kink_branch2_omega/body_kink_branch2_k  
   KB2_k = body_kink_branch2_k 
   KB2_k_new = np.linspace(KB2_k[0], KB2_k[-1], num=len(KB2_k)*10)
   
   KB2_coefs = poly.polyfit(KB2_k, KB2_phase, 1)   # was 6   # 1 good
   KB2_ffit = poly.polyval(KB2_k_new, KB2_coefs)


##############################################################################################

test_k_plot = np.linspace(0.01,Kmax,20)

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("$k$", fontsize=18)
plt.ylabel('$\omega$', fontsize=22, rotation=0, labelpad=15)
vA_e_plot = test_k_plot*vA_e
vA_i_plot = test_k_plot*vA_i0
c_e_plot = test_k_plot*c_e
c_i_plot = test_k_plot*c_i0
cT_e_plot = test_k_plot*cT_e
cT_i_plot = test_k_plot*cT_i0
#ax.plot(sol_ks1, sol_omegas1, 'b.', markersize=4.)


#ax.plot(body_sausage_k, body_sausage_omega, 'r.', markersize=4.)   # body sausage
ax.plot(body_kink_k, body_kink_omega, 'b.', markersize=4.)   # body kink
#ax.plot(fast_surf_sausage_k, fast_surf_sausage_omega, 'r.', markersize=4.)   #fast surface sausage
#ax.plot(fast_surf_kink_k, fast_surf_kink_omega, 'b.', markersize=4.)   #fast surface kink
#ax.plot(slow_surf_sausage_k, slow_surf_sausage_omega, 'r.', markersize=4.)   # slow surface sausage
#ax.plot(slow_surf_kink_k, slow_surf_kink_omega, 'b.', markersize=4.)   # slow surface kink

ax.plot(test_k_plot, vA_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, vA_i_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, c_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, c_i_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, cT_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, cT_i_plot, linestyle='dashdot', color='k')
ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i_plot[-1]), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e_plot[-1]), fontsize=20)
ax.annotate( '$c_{e}$', xy=(Kmax, c_e_plot[-1]), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i_plot[-1]), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e_plot[-1]), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i_plot[-1]), fontsize=20)
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


########################################################################   sausage polyfit
FSS_phase = fast_surf_sausage_omega/fast_surf_sausage_k   #fast surface
FSS_k = fast_surf_sausage_k
FSS_k_new = np.linspace(1.18, Kmax, num=len(FSS_k)*10)

FSS_coefs = poly.polyfit(FSS_k, FSS_phase,6)    #was 6
FSS_ffit = poly.polyval(FSS_k_new, FSS_coefs)

if len(slow_surf_sausage_omega) > 1:
  SSS_phase = slow_surf_sausage_omega/slow_surf_sausage_k   #slow surface
  SSS_k = slow_surf_sausage_k
  SSS_k_new = np.linspace(1.12, Kmax, num=len(SSS_k)*10)
  
  SSS_coefs = poly.polyfit(SSS_k, SSS_phase, 1)
  SSS_ffit = poly.polyval(SSS_k_new, SSS_coefs)


########################################################################   kink polyfit
#FSK_phase = fast_surf_kink_omega/fast_surf_kink_k   #fast surface
#FSK_k = fast_surf_kink_k
##FSK_k_new = np.linspace(FSK_k[0], Kmax, num=len(FSK_k)*10)
#FSK_k_new = np.linspace(0, Kmax, num=len(FSK_k)*10)
#
#FSK_coefs = poly.polyfit(FSK_k, FSK_phase, 4)    #was 6
#FSK_ffit = poly.polyval(FSK_k_new, FSK_coefs)
#
#
#
index = []
for i in range(len(FSK_ffit)):
    if FSK_ffit[i] < cT_i0:
        index.append(i)

FSK_ffit2 = np.delete(FSK_ffit, [index], None)
FSK_k_new2 = np.delete(FSK_k_new, [index], None)


if len(slow_surf_kink_omega) > 1:
  SSK_phase = slow_surf_kink_omega/slow_surf_kink_k   #slow surface
  SSK_k = slow_surf_kink_k
  SSK_k_new = np.linspace(1.12, SSK_k[-1], num=len(SSK_k)*10)
  
  SSK_coefs = poly.polyfit(SSK_k, SSK_phase, 1)   #was 6     #use 2 for W=2.5
  SSK_ffit = poly.polyval(SSK_k_new, SSK_coefs)



#################    BEGIN FULL PLOTS   ################################     !!!!!!!!!!!!!!!!     PHOTOSPHERIC       !!!!!!!!!!!!!!!!!!!!!

plt.figure()

fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for photospheric to remove blank space on plot

ax2.set_title("$ W = 0.9$")

#ax = plt.subplot(111)
plt.xlabel("$ka$", fontsize=18)
#plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax2.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
#ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

#ax2.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
#ax2.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

#ax.plot(sausage_extra_branch_k, (sausage_extra_branch_omega/sausage_extra_branch_k), 'k.', markersize=8.)
#ax.plot(sausage_extra_branch2_k, (sausage_extra_branch2_omega/sausage_extra_branch2_k), 'g.', markersize=4.)

#ax.plot(body_sausage_k, (body_sausage_omega/body_sausage_k), 'r.', markersize=4.)
#ax.plot(body_kink_k, (body_kink_omega/body_kink_k), 'b.', markersize=4.)

#ax.plot(combine_kink_k, combine_kink_omega/combine_kink_k, 'b.', markersize=4.)   # body kink
#ax.plot(fast_surf_kink_k, fast_surf_kink_omega/fast_surf_kink_k, 'b.', markersize=4.)   # body kink
#ax.plot(fast_surf_sausage_k, fast_surf_sausage_omega/fast_surf_sausage_k, 'r.', markersize=4.)   # body kink
#ax.plot(body_sausage_branch1_k, body_sausage_branch1_omega/body_sausage_branch1_k, 'r.', markersize=4.)   # body sausage
#ax.plot(body_kink_branch1_k, body_kink_branch1_omega/body_kink_branch1_k, 'b.', markersize=4.)   # body kink
#ax.plot(body_sausage_branch2_k, body_sausage_branch2_omega/body_sausage_branch2_k, 'r.', markersize=4.)   # body sausage
#ax.plot(body_kink_branch2_k, body_kink_branch2_omega/body_kink_branch2_k, 'b.', markersize=4.)   # body kink

ax.plot(SB_k_new, SB_ffit, color='r')
#ax.plot(SB2_k_new, SB2_ffit, color='r')
ax.plot(KB_k_new, KB_ffit, color='b')    
#ax.plot(KB2_k_new, KB2_ffit, color='b')


ax.plot(FSS_k_new, FSS_ffit, color='r')
#ax.plot(SSS_k_new, SSS_ffit, color='r')
ax.plot(FSK_k_new, FSK_ffit, color='b')    
#ax.plot(SSK_k_new, SSK_ffit, color='b')


ax2.plot(SB_k_new, SB_ffit, color='r')
ax2.plot(SB2_k_new, SB2_ffit, color='r')
ax2.plot(KB_k_new, KB_ffit, color='b')    
ax2.plot(KB2_k_new, KB2_ffit, color='b')


ax2.plot(FSS_k_new, FSS_ffit, color='r')
#ax2.plot(SSS_k_new, SSS_ffit, color='r')
ax2.plot(FSK_k_new, FSK_ffit, color='b')    
#ax2.plot(SSK_k_new, SSK_ffit, color='b')



ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

ax.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink_bound, color='k', linestyle='dashdot', label='_nolegend_')

ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

ax2.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_kink_bound, color='k', linestyle='dashdot', label='_nolegend_')
#ax2.axhline(y=c_kink_bound2, color='k', linestyle='dashed', label='_nolegend_')
#ax2.axhline(y=c_kink_bound3, color='k', linestyle='dotted', label='_nolegend_')


ax.annotate( ' $c_{TB}$', xy=(Kmax, cT_bound), fontsize=20)
ax.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{B}$', xy=(Kmax, c_bound), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax.annotate( ' $c_{kB}$', xy=(Kmax, c_kink_bound), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)


ax2.annotate( ' $c_{TB}$', xy=(Kmax, cT_bound), fontsize=20)
ax2.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax2.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( ' $c_{B}$', xy=(Kmax, c_bound), fontsize=20)
ax2.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax2.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=20)
ax2.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax2.annotate( ' $c_{kB}$', xy=(Kmax, c_kink_bound-0.005), fontsize=20)
ax2.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)


ax.fill_between(wavenumber, cT_i0, cT_bound, color='blue', alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
#ax.fill_between(wavenumber, -vA_i0, -vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, c_kink, c_kink_bound, color='gold', alpha=0.2)    # fill between uniform case and boundary values
#ax.fill_between(wavenumber, -c_kink, -c_kink_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values


ax2.fill_between(wavenumber, cT_i0, cT_bound, color='blue', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
#ax.fill_between(wavenumber, -vA_i0, -vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, c_kink, c_kink_bound, color='gold', alpha=0.2)    # fill between uniform case and boundary values
#ax.fill_between(wavenumber, -c_kink, -c_kink_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values


ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
fig.tight_layout()


ax2.set_ylim(1.15, c_e)   # whole plot
ax.set_ylim(0.7, 1.05)   # whole plot

ax.set_xlim(0., 4.)   # whole plot
ax2.set_xlim(0., 4.)   # whole plot


box = ax.get_position()
ax.set_position([box.x0+0.05, box.y0, box.width*0.9, box.height*0.85])

box2 = ax2.get_position()
ax2.set_position([box2.x0+0.05, box2.y0-0.05, box2.width*0.9, box2.height*1.1])


plt.savefig("photospheric_gaussian_density_curves_09.png")
#plt.savefig("cylinder_uniform_photospheric.png")


plt.show()
exit()

######################################################################  end polyfit

##d_omega_sausage_branch1 = np.gradient(fast_sausage_branch1_omega)
##d_k_sausage_branch1 = np.gradient(fast_sausage_branch1_k)
#
#d_omega_sausage_branch1 = np.diff(fast_sausage_branch1_omega)
#d_k_sausage_branch1 = np.diff(fast_sausage_branch1_k)
#
##d_omega_kink = np.gradient(sol_omegas_kink1)
##d_k_kink = np.gradient(sol_ks_kink1)
#
#plt.figure()
#ax = plt.subplot(111)
#plt.xlabel("$kx_{0}$", fontsize=18)
#plt.ylabel(r'$\frac{d\omega}{dk}$', fontsize=22, rotation=0, labelpad=15)
#ax.plot((d_omega_sausage_branch1/d_k_sausage_branch1),fast_sausage_branch1_k[:-1], 'b.', markersize=4.)
#
#



plt.figure()
ax = plt.subplot(111)
plt.title("")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
#ax.plot(sol_ks_kink, (sol_omegas_kink/sol_ks_kink), 'b.', markersize=5.)
#ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
#ax.plot(fast_body_kink_k, (fast_body_kink_omega/fast_body_kink_k), 'b.', markersize=4.)   #fast body
#ax.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)   #slow body

ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)

ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

#ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
#ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
#ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)


ax.annotate( '$c_{B}$', xy=(Kmax, c_bound), fontsize=20, color='k')
ax.annotate( '$v_{AB}$', xy=(Kmax, vA_bound), fontsize=20, color='k')
ax.annotate( '$c_{TB}$', xy=(Kmax, cT_bound), fontsize=20, color='k')
ax.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')


ax.fill_between(wavenumber, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values


#ax.set_ylim(0., vA_e+0.1)
#ax.set_yticks([])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

#plt.savefig("test_dispersion_diagram_kink_coronal.png")
plt.show()
exit()
