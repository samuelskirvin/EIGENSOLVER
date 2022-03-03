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
vA_i0 = 1.9*c_i0  #-photospheric

vA_e = 0.8*c_i0  #-photospheric
c_e = 1.3*c_i0    #-photospheric

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

ix = np.linspace(-1., 1., 500.)  # inside slab x values

x0=0.  #mean
dx=1.5 #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


#def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
#    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   



def profile(x):       # Define the internal profile as a function of variable x   (Besslike)
    return (-1./10.*(rho_i0*(sym.sin(10.*x)/x)) + rho_i0)/4. + rho_i0  



#def profile(x):       # Define the internal profile as a function of variable x   (Inverted Besslike)
#    return (1./10.*(rho_i0*(sym.sin(10.*x)/x)) + rho_i0)/4. + rho_i0/2. 

#################
dx1e5=1e5 #standard dev
def profile_1e5(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx1e5**2)))   
profile1e5_np=sym.lambdify(xx,profile_1e5(xx),"numpy")   #In order to evaluate we need to switch to numpy

dx3=3. #standard dev
def profile_3(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx3**2)))   
profile3_np=sym.lambdify(xx,profile_3(xx),"numpy")   #In order to evaluate we need to switch to numpy

dx15=1.5 #standard dev
def profile_15(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx15**2)))   
profile15_np=sym.lambdify(xx,profile_15(xx),"numpy")   #In order to evaluate we need to switch to numpy

dx09=0.9 #standard dev
def profile_09(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx09**2)))   
profile09_np=sym.lambdify(xx,profile_09(xx),"numpy")   #In order to evaluate we need to switch to numpy


def profile_besslike(x):       # Bessel(like) Function
    return (-1./10.*(rho_i0*(sym.sin(10.*x)/x)) + rho_i0)/4. + rho_i0
profile_besslike_np=sym.lambdify(xx,profile_besslike(xx),"numpy")   #In order to evaluate we need to switch to numpy


###################
#a = 1.   #inhomogeneity width for epstein profile
#def profile(x):       # Define the internal profile as a function of variable x   (Epstein Profile)
#    return ((rho_i0 - rho_e)/(sym.cosh(x/a)**4)**2 + rho_e)
####################



#def profile(x):       # Define the internal profile as a function of variable x   (Periodic threaded waveguide)
#    return (3.*rho_e + (rho_i0 - 3.*rho_e)*sym.cos(2.*sym.pi*x + sym.pi))    #2.28 = W ~ 0.6     #3. corresponds to W ~ 1.5



def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#def P_i(x):   #constant temp
#    return ((c_i0)**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

#def T_i(x):    # constant temp
#    return (P_i(x)*rho_i0/(P_0*rho_i(x)))


##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))


#def vA_i(x):                  # consatnt temp
#    return B_0/(sym.sqrt(rho_i(x)))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

#def B_i(x):  #constant temp
#    return (sym.sqrt(2*(P_tot_0 - P_i(x))))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

#def c_i(x):                  # keeps c_i constant
#    return c_i0*sym.sqrt(P_i(x))*sym.sqrt(rho_i0)/(sym.sqrt(P_0)*sym.sqrt(rho_i(x)))

#def c_i(x):                  # keeps c_i constant
#    return sym.sqrt(P_i(x))/sym.sqrt(gamma*rho_i(x))

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#####################################################
speeds = [c_i0, c_e, vA_i0, vA_e, cT_i0, cT_e()]
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
print('internal gas pressure  =', internal_pressure)

print('c_i0 ratio   =',   c_e/c_i0)
print('c_i ratio   =',   c_e/c_bound)
print('c_i boundary   =', c_bound)
print('vA_i boundary   =', vA_bound)
print('rho_i boundary   =', rho_bound)

print('temp = ', T_i_np(ix))

#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################


plt.figure()
#plt.title("width = 1.5")
plt.xlabel("$x$",fontsize=25)
plt.ylabel("$\u03C1$",fontsize=25, rotation=0)
ax = plt.subplot(111)   #331
#ax.title.set_text("Density")
#ax.plot(ix,rho_i_np(ix), 'k')#, linestyle='dashdot');
ax.plot(ix,profile09_np(ix), 'r')#, linestyle='dotted');
ax.plot(ix,profile15_np(ix), 'g')#, linestyle='dashed');
ax.plot(ix,profile3_np(ix), color='goldenrod')#, linestyle='dotted');
ax.plot(ix,profile1e5_np(ix), 'k')#, linestyle='solid');
ax.plot(ix,profile_besslike_np(ix), 'b')#, linestyle='solid');
ax.annotate( '$\u03C1_{e}$', xy=(1.2, rho_e),fontsize=25)
ax.annotate( '$\u03C1_{i}$', xy=(1.2, rho_i0),fontsize=25)
ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot', alpha=0.25)
ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot', alpha=0.25)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(0.98, 1.85)
ax.axvline(x=-1, color='r', linestyle='--')
ax.axvline(x=1, color='r', linestyle='--')

#plt.savefig("photospheric_slab_density_profiles.png")

#plt.show()
#exit()

plt.figure()

ax = plt.subplot(331)   #331
#ax.title.set_text("Density")
ax.plot(ix,rho_i_np(ix), 'k')#, linestyle='dashdot');
ax.annotate( '$\u03C1_{e}$', xy=(1.2, rho_e),fontsize=25)
ax.annotate( '$\u03C1_{i}$', xy=(1.2, rho_i0),fontsize=25)
ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot', alpha=0.25)
ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot', alpha=0.25)

ax.set_xlim(-1.2, 1.2)
#ax.set_ylim(0.98, 1.85)
ax.axvline(x=-1, color='r', linestyle='--')
ax.axvline(x=1, color='r', linestyle='--')



#plt.xlabel("x")
#plt.ylabel("$c_{i}$")
ax2 = plt.subplot(332)
ax2.title.set_text("$c_{i}$")
ax2.plot(ix,c_i_np(ix));
ax2.annotate( '$c_e$', xy=(1, c_e),fontsize=25)
ax2.annotate( '$c_{i0}$', xy=(1, c_i0),fontsize=25)
ax2.annotate( '$c_{B}$', xy=(1, c_bound), fontsize=20)
ax2.axhline(y=c_i0, color='k', label='$c_{i0}$', linestyle='dashdot')
ax2.axhline(y=c_e, color='k', label='$c_{e}$', linestyle='dashdot')
ax2.fill_between(ix, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values


#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$vA_{i}$")
ax3 = plt.subplot(333)
ax3.title.set_text("$v_{Ai}$")
ax3.plot(ix,vA_i_np(ix));
ax3.annotate( '$v_{Ae}$', xy=(1, vA_e),fontsize=25)
ax3.annotate( '$v_{Ai}$', xy=(1, vA_i0),fontsize=25)
ax3.annotate( '$v_{AB}$', xy=(1, vA_bound), fontsize=20)
ax3.axhline(y=vA_i0, color='k', label='$v_{Ai}$', linestyle='dashdot')
ax3.axhline(y=vA_e, color='k', label='$v_{Ae}$', linestyle='dashdot')
ax3.fill_between(ix, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values


#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$cT_{i}$")
ax4 = plt.subplot(334)
ax4.title.set_text("$c_{Ti}$")
ax4.plot(ix,cT_i_np(ix));
ax4.annotate( '$c_{Te}$', xy=(1, cT_e()),fontsize=25)
ax4.annotate( '$c_{Ti}$', xy=(1, cT_i0),fontsize=25)
ax4.annotate( '$c_{TB}$', xy=(1, cT_bound), fontsize=20)
ax4.axhline(y=cT_i0, color='k', label='$c_{Ti}$', linestyle='dashdot')
ax4.axhline(y=cT_e(), color='k', label='$c_{Te}$', linestyle='dashdot')
ax4.fill_between(ix, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values



#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$P$")
ax5 = plt.subplot(335)
ax5.title.set_text("Gas Pressure")
#ax5.plot(ix,P_i_np(ix));
ax5.annotate( '$P_{0}$', xy=(1, P_0),fontsize=25)
ax5.annotate( '$P_{e}$', xy=(1, P_e),fontsize=25)
ax5.axhline(y=P_0, color='k', label='$P_{i}$', linestyle='dashdot')
ax5.axhline(y=P_e, color='k', label='$P_{e}$', linestyle='dashdot')
ax5.axhline(y=P_i_np(ix), color='b', label='$P_{e}$', linestyle='solid')

#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$T$")
ax6 = plt.subplot(336)
ax6.title.set_text("Temperature")
ax6.plot(ix,T_i_np(ix));
ax6.annotate( '$T_{0}$', xy=(1, T_0),fontsize=25)
ax6.annotate( '$T_{e}$', xy=(1, T_e),fontsize=25)
ax6.axhline(y=T_0, color='k', label='$T_{i}$', linestyle='dashdot')
ax6.axhline(y=T_e, color='k', label='$T_{e}$', linestyle='dashdot')
#ax6.axhline(y=T_i_np(ix), color='b', linestyle='solid')
#ax6.set_ylim(0., 1.1)

#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$B$")
ax7 = plt.subplot(337)
ax7.title.set_text("Mag Field")
#ax7.plot(ix,B_i_np(ix));
ax7.annotate( '$B_{0}$', xy=(1, B_0),fontsize=25)
ax7.annotate( '$B_{e}$', xy=(1, B_e),fontsize=25)
ax7.axhline(y=B_0, color='k', label='$B_{i}$', linestyle='dashdot')
ax7.axhline(y=B_e, color='k', label='$B_{e}$', linestyle='dashdot')
ax7.axhline(y=B_i_np(ix), color='b', label='$B_{e}$', linestyle='solid')


#plt.xlabel("x")
#plt.ylabel("$P_T$")
ax8 = plt.subplot(338)
ax8.title.set_text("Tot Pressure (uniform)")
ax8.annotate( '$P_{T0}$', xy=(1, P_tot_0),fontsize=25,color='b')
ax8.annotate( '$P_{Te}$', xy=(1, P_tot_e),fontsize=25,color='r')
ax8.axhline(y=P_tot_0, color='b', label='$P_{Ti0}$', linestyle='dashdot')
ax8.axhline(y=P_tot_e, color='r', label='$P_{Te}$', linestyle='dashdot')

#ax8.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
#ax8.axhline(y=B_i_np(ix), color='b', label='$P_{Te}$', linestyle='solid')

ax9 = plt.subplot(339)
ax9.title.set_text("Tot Pressure  (Modified)")
ax9.annotate( '$P_{T0},  P_{Te}$', xy=(1, P_tot_0),fontsize=25,color='b')
#ax9.plot(ix,PT_i_np(ix));

ax9.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')

plt.suptitle("W = 3", fontsize=14)
plt.tight_layout()
#plt.savefig("photospheric_profiles_09.png")

#plt.show()
#exit()

wavenumber = np.linspace(0.01,3.5,20)      #(1.5, 1.8), 5     


########   READ IN VARIABLES    #########

with open('width1e5.pickle', 'rb') as f:
    sol_omegas1e5, sol_ks1e5, sol_omegas_kink1e5, sol_ks_kink1e5 = pickle.load(f)

with open('width3.pickle', 'rb') as f:
    sol_omegas3, sol_ks3, sol_omegas_kink3, sol_ks_kink3 = pickle.load(f)

with open('width15.pickle', 'rb') as f:
    sol_omegas15, sol_ks15, sol_omegas_kink15, sol_ks_kink15 = pickle.load(f)

with open('width09.pickle', 'rb') as f:
    sol_omegas09, sol_ks09, sol_omegas_kink09, sol_ks_kink09 = pickle.load(f)

with open('Besslike.pickle', 'rb') as f:
    sol_omegas_bess, sol_ks_bess, sol_omegas_kink_bess, sol_ks_kink_bess = pickle.load(f)


with open('Besslike.pickle', 'rb') as f:
    sol_omegas_bess, sol_ks_bess, sol_omegas_kink_bess, sol_ks_kink_bess = pickle.load(f)

with open('Besslike_zoom.pickle', 'rb') as f:
    sol_omegas_bess_zoom, sol_ks_bess_zoom, sol_omegas_kink_bess_zoom, sol_ks_kink_bess_zoom = pickle.load(f)


sol_omegas_bess = np.concatenate((sol_omegas_bess, sol_omegas_bess_zoom), axis=None)
sol_ks_bess = np.concatenate((sol_ks_bess, sol_ks_bess_zoom), axis=None)
sol_omegas_kink_bess = np.concatenate((sol_omegas_kink_bess, sol_omegas_kink_bess_zoom), axis=None)
sol_ks_kink_bess = np.concatenate((sol_ks_kink_bess, sol_ks_kink_bess_zoom), axis=None)

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


### SORT ARRAYS IN ORDER OF wavenumber ###
sol_omegas_bess = [x for _,x in sorted(zip(sol_ks_bess,sol_omegas_bess))]
sol_ks_bess = np.sort(sol_ks_bess)

sol_omegas_bess = np.array(sol_omegas_bess)
sol_ks_bess = np.array(sol_ks_bess)

sol_omegas_kink_bess = [x for _,x in sorted(zip(sol_ks_kink_bess,sol_omegas_kink_bess))]
sol_ks_kink_bess = np.sort(sol_ks_kink_bess)

sol_omegas_kink_bess = np.array(sol_omegas_kink_bess)
sol_ks_kink_bess = np.array(sol_ks_kink_bess)

##########################################################################

fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for photospheric to remove blank space on plot
plt.title(" ")
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)


#ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
#ax.plot(sol_ks3, (sol_omegas3/sol_ks3), color='goldenrod', marker='.', linestyle='', markersize=4.)
#ax.plot(sol_ks15, (sol_omegas15/sol_ks15), 'g.', markersize=4.)
#ax.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)
#ax.plot(sol_ks_bess, (sol_omegas_bess/sol_ks_bess), 'b.', markersize=4.)


ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)
ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), color='goldenrod', marker='.', linestyle='', markersize=4.)
ax.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'g.', markersize=4.)
ax.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'r.', markersize=4.)
ax.plot(sol_ks_kink_bess, (sol_omegas_kink_bess/sol_ks_kink_bess), 'b.', markersize=4.)

#ax2.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
#ax2.plot(sol_ks3, (sol_omegas3/sol_ks3), color='goldenrod', marker='.', linestyle='', markersize=4.)
#ax2.plot(sol_ks15, (sol_omegas15/sol_ks15), 'g.', markersize=4.)
#ax2.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)
#ax2.plot(sol_ks_bess, (sol_omegas_bess/sol_ks_bess), 'b.', markersize=4.)


ax2.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)
ax2.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), color='goldenrod', marker='.', linestyle='', markersize=4.)
ax2.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'g.', markersize=4.)
ax2.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'r.', markersize=4.)
ax2.plot(sol_ks_kink_bess, (sol_omegas_kink_bess/sol_ks_kink_bess), 'b.', markersize=4.)


ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
#ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)


ax.annotate( '$c_{B}$', xy=(Kmax, c_bound), fontsize=20, color='k')
ax.annotate( '$v_{AB}$', xy=(Kmax, vA_bound), fontsize=20, color='k')
ax.annotate( '$c_{TB}$', xy=(Kmax, cT_bound), fontsize=20, color='k')
ax.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')


ax2.annotate( '$c_{B}$', xy=(Kmax, c_bound), fontsize=20, color='k')
ax2.annotate( '$v_{AB}$', xy=(Kmax, vA_bound), fontsize=20, color='k')
ax2.annotate( '$c_{TB}$', xy=(Kmax, cT_bound), fontsize=20, color='k')
ax2.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')


ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
#ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

#ax2.annotate( '$c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax2.annotate( '$c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( '$c_{Te}$', xy=(Kmax, cT_e()), fontsize=20)
#ax2.annotate( '$c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax2.annotate( '$v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
#ax2.annotate( '$v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)


ax.fill_between(wavenumber, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values

ax2.fill_between(wavenumber, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values


ax.set_ylim(0.6, 1.4)  
ax2.set_ylim(1.8, 2.)  # remove blank space

ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()


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
ax2.set_position([box2.x0, box2.y0+0.3, box2.width*0.85, box2.height*0.2])    #+0.3

#plt.savefig("uniform_coronal_curves_dots_shaded09.png")
#plt.show()
#exit()





# fast cutoff  == 1.1

# slow body > 0.9    < 1.0

#######
sol_ks1e5_singleplot_fast = []
sol_omegas1e5_singleplot_fast = []
sol_ks1e5_singleplot_slow = []
sol_omegas1e5_singleplot_slow = []

sol_ks1e5_singleplot_fast_kink = []
sol_omegas1e5_singleplot_fast_kink = []
sol_ks1e5_singleplot_slow_kink = []
sol_omegas1e5_singleplot_slow_kink = []


for i in range(len(sol_ks1e5)):
    if sol_ks1e5[i] > 2.025 and sol_ks1e5[i] < 2.075:
      if sol_omegas1e5[i]/sol_ks1e5[i] > 1.1:
        sol_ks1e5_singleplot_fast.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_fast.append(sol_omegas1e5[i])
        
    if sol_ks1e5[i] > 2.625 and sol_ks1e5[i] < 2.675:  
      if sol_omegas1e5[i]/sol_ks1e5[i] < 0.951 and sol_omegas1e5[i]/sol_ks1e5[i] > 0.93: 
        sol_ks1e5_singleplot_slow.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_slow.append(sol_omegas1e5[i])  

for i in range(len(sol_ks_kink1e5)):
    if sol_ks_kink1e5[i] > 2.025 and sol_ks_kink1e5[i] < 2.075:        
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > 1.:
         sol_ks1e5_singleplot_fast_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_fast_kink.append(sol_omegas_kink1e5[i])        
    
    if sol_ks_kink1e5[i] > 2.825 and sol_ks_kink1e5[i] < 2.85:  
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < 1. and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > 0.9: 
         sol_ks1e5_singleplot_slow_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_slow_kink.append(sol_omegas_kink1e5[i])
        
        
sol_ks1e5_singleplot_fast = np.array(sol_ks1e5_singleplot_fast)
sol_omegas1e5_singleplot_fast = np.array(sol_omegas1e5_singleplot_fast)    
sol_ks1e5_singleplot_slow = np.array(sol_ks1e5_singleplot_slow)
sol_omegas1e5_singleplot_slow = np.array(sol_omegas1e5_singleplot_slow)    
     
sol_ks1e5_singleplot_fast_kink = np.array(sol_ks1e5_singleplot_fast_kink)
sol_omegas1e5_singleplot_fast_kink = np.array(sol_omegas1e5_singleplot_fast_kink)
sol_ks1e5_singleplot_slow_kink = np.array(sol_ks1e5_singleplot_slow_kink)
sol_omegas1e5_singleplot_slow_kink = np.array(sol_omegas1e5_singleplot_slow_kink)

########
sol_ks3_singleplot_fast = []
sol_omegas3_singleplot_fast = []
sol_ks3_singleplot_slow = []
sol_omegas3_singleplot_slow = []

sol_ks3_singleplot_fast_kink = []
sol_omegas3_singleplot_fast_kink = []
sol_ks3_singleplot_slow_kink = []
sol_omegas3_singleplot_slow_kink = []


for i in range(len(sol_ks3)):
    if sol_ks3[i] > 2.025 and sol_ks3[i] < 2.075:
      if sol_omegas3[i]/sol_ks3[i] > 1.1:
        sol_ks3_singleplot_fast.append(sol_ks3[i])
        sol_omegas3_singleplot_fast.append(sol_omegas3[i])
        
    if sol_ks3[i] > 2.625 and sol_ks3[i] < 2.675:  
      if sol_omegas3[i]/sol_ks3[i] < 0.95 and sol_omegas3[i]/sol_ks3[i] > 0.9:
        sol_ks3_singleplot_slow.append(sol_ks3[i])
        sol_omegas3_singleplot_slow.append(sol_omegas3[i])    

for i in range(len(sol_ks_kink3)):
    if sol_ks_kink3[i] > 2.025 and sol_ks_kink3[i] < 2.075:        
       if sol_omegas_kink3[i]/sol_ks_kink3[i] > 1.:
         sol_ks3_singleplot_fast_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_fast_kink.append(sol_omegas_kink3[i])        

    if sol_ks_kink3[i] > 2.825 and sol_ks_kink3[i] < 2.85:    
       if sol_omegas_kink3[i]/sol_ks_kink3[i] < 1. and sol_omegas_kink3[i]/sol_ks_kink3[i] > 0.89: 
         sol_ks3_singleplot_slow_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_slow_kink.append(sol_omegas_kink3[i])
         
         
sol_ks3_singleplot_fast = np.array(sol_ks3_singleplot_fast)
sol_omegas3_singleplot_fast = np.array(sol_omegas3_singleplot_fast)    
sol_ks3_singleplot_slow = np.array(sol_ks3_singleplot_slow)
sol_omegas3_singleplot_slow = np.array(sol_omegas3_singleplot_slow)    

sol_ks3_singleplot_fast_kink = np.array(sol_ks3_singleplot_fast_kink)
sol_omegas3_singleplot_fast_kink = np.array(sol_omegas3_singleplot_fast_kink)
sol_ks3_singleplot_slow_kink = np.array(sol_ks3_singleplot_slow_kink)
sol_omegas3_singleplot_slow_kink = np.array(sol_omegas3_singleplot_slow_kink)
     
########    
sol_ks15_singleplot_fast = []
sol_omegas15_singleplot_fast = []
sol_ks15_singleplot_slow = []
sol_omegas15_singleplot_slow = []

sol_ks15_singleplot_fast_kink = []
sol_omegas15_singleplot_fast_kink = []
sol_ks15_singleplot_slow_kink = []
sol_omegas15_singleplot_slow_kink = []


for i in range(len(sol_ks15)):
    if sol_ks15[i] > 2.025 and sol_ks15[i] < 2.075:
      if sol_omegas15[i]/sol_ks15[i] > 1.1:
        sol_ks15_singleplot_fast.append(sol_ks15[i])
        sol_omegas15_singleplot_fast.append(sol_omegas15[i])
        
    if sol_ks15[i] > 2.625 and sol_ks15[i] < 2.675:  
      if sol_omegas15[i]/sol_ks15[i] < 0.95 and sol_omegas15[i]/sol_ks15[i] > 0.9: 
        sol_ks15_singleplot_slow.append(sol_ks15[i])
        sol_omegas15_singleplot_slow.append(sol_omegas15[i])    

for i in range(len(sol_ks_kink15)):
    if sol_ks_kink15[i] > 2.025 and sol_ks_kink15[i] < 2.075:        
       if sol_omegas_kink15[i]/sol_ks_kink15[i] > 1.:
         sol_ks15_singleplot_fast_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_fast_kink.append(sol_omegas_kink15[i])        
 
    if sol_ks_kink15[i] > 2.825 and sol_ks_kink15[i] < 2.85:  
      if sol_omegas_kink15[i]/sol_ks_kink15[i] < 0.95 and sol_omegas_kink15[i]/sol_ks_kink15[i] > 0.88:
         sol_ks15_singleplot_slow_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_slow_kink.append(sol_omegas_kink15[i])
         
         
sol_ks15_singleplot_fast = np.array(sol_ks15_singleplot_fast)
sol_omegas15_singleplot_fast = np.array(sol_omegas15_singleplot_fast)    
sol_ks15_singleplot_slow = np.array(sol_ks15_singleplot_slow)
sol_omegas15_singleplot_slow = np.array(sol_omegas15_singleplot_slow)    

sol_ks15_singleplot_fast_kink = np.array(sol_ks15_singleplot_fast_kink)
sol_omegas15_singleplot_fast_kink = np.array(sol_omegas15_singleplot_fast_kink)
sol_ks15_singleplot_slow_kink = np.array(sol_ks15_singleplot_slow_kink)
sol_omegas15_singleplot_slow_kink = np.array(sol_omegas15_singleplot_slow_kink)


########    
sol_ks09_singleplot_fast = []
sol_omegas09_singleplot_fast = []
sol_ks09_singleplot_slow = []
sol_omegas09_singleplot_slow = []

sol_ks09_singleplot_fast_kink = []
sol_omegas09_singleplot_fast_kink = []
sol_ks09_singleplot_slow_kink = []
sol_omegas09_singleplot_slow_kink = []

for i in range(len(sol_ks09)):
    if sol_ks09[i] > 2.025 and sol_ks09[i] < 2.075:
      if sol_omegas09[i]/sol_ks09[i] > 1.1:
        sol_ks09_singleplot_fast.append(sol_ks09[i])
        sol_omegas09_singleplot_fast.append(sol_omegas09[i])
        
      #else:
      
    if sol_ks09[i] > 2.625 and sol_ks09[i] < 2.675:  
      if sol_omegas09[i]/sol_ks09[i] < 0.95 and sol_omegas09[i]/sol_ks09[i] > 0.9: 
        sol_ks09_singleplot_slow.append(sol_ks09[i])
        sol_omegas09_singleplot_slow.append(sol_omegas09[i])    

for i in range(len(sol_ks_kink09)):
    if sol_ks_kink09[i] > 2.025 and sol_ks_kink09[i] < 2.075:        
       if sol_omegas_kink09[i]/sol_ks_kink09[i] > 1.:
         sol_ks09_singleplot_fast_kink.append(sol_ks_kink09[i])
         sol_omegas09_singleplot_fast_kink.append(sol_omegas_kink09[i])        
 
       elif sol_omegas_kink09[i]/sol_ks_kink09[i] > 0.9: 
         sol_ks09_singleplot_slow_kink.append(sol_ks_kink09[i])
         sol_omegas09_singleplot_slow_kink.append(sol_omegas_kink09[i])



sol_ks09_singleplot_fast = np.array(sol_ks09_singleplot_fast)
sol_omegas09_singleplot_fast = np.array(sol_omegas09_singleplot_fast)    
sol_ks09_singleplot_slow = np.array(sol_ks09_singleplot_slow)
sol_omegas09_singleplot_slow = np.array(sol_omegas09_singleplot_slow)    
  
sol_ks09_singleplot_fast_kink = np.array(sol_ks09_singleplot_fast_kink)
sol_omegas09_singleplot_fast_kink = np.array(sol_omegas09_singleplot_fast_kink)
sol_ks09_singleplot_slow_kink = np.array(sol_ks09_singleplot_slow_kink)
sol_omegas09_singleplot_slow_kink = np.array(sol_omegas09_singleplot_slow_kink)
 


########    
sol_ks_bess_singleplot_fast = []
sol_omegas_bess_singleplot_fast = []
sol_ks_bess_singleplot_slow = []
sol_omegas_bess_singleplot_slow = []

sol_ks_bess_singleplot_fast_kink = []
sol_omegas_bess_singleplot_fast_kink = []
sol_ks_bess_singleplot_slow_kink = []
sol_omegas_bess_singleplot_slow_kink = []


for i in range(len(sol_ks_bess)):
    if sol_ks_bess[i] > 2.025 and sol_ks_bess[i] < 2.075:
      if sol_omegas_bess[i]/sol_ks_bess[i] > 1.1:
        sol_ks_bess_singleplot_fast.append(sol_ks_bess[i])
        sol_omegas_bess_singleplot_fast.append(sol_omegas_bess[i])
        
    if sol_ks_bess[i] > 2.625 and sol_ks_bess[i] < 2.675:  
      if sol_omegas_bess[i]/sol_ks_bess[i] < 0.95 and sol_omegas_bess[i]/sol_ks_bess[i] > 0.9: 
        sol_ks_bess_singleplot_slow.append(sol_ks_bess[i])
        sol_omegas_bess_singleplot_slow.append(sol_omegas_bess[i])    


for i in range(len(sol_ks_kink_bess)):
    if sol_ks_kink_bess[i] > 2.025 and sol_ks_kink_bess[i] < 2.075:        
       if sol_omegas_kink_bess[i]/sol_ks_kink_bess[i] > 1.:
         sol_ks_bess_singleplot_fast_kink.append(sol_ks_kink_bess[i])
         sol_omegas_bess_singleplot_fast_kink.append(sol_omegas_kink_bess[i])        
 
       elif sol_omegas_kink_bess[i]/sol_ks_kink_bess[i] < 1.2 and sol_omegas_kink_bess[i]/sol_ks_kink_bess[i] > 0.9: 
         sol_ks_bess_singleplot_slow_kink.append(sol_ks_kink_bess[i])
         sol_omegas_bess_singleplot_slow_kink.append(sol_omegas_kink_bess[i])


sol_ks_bess_singleplot_fast = np.array(sol_ks_bess_singleplot_fast)
sol_omegas_bess_singleplot_fast = np.array(sol_omegas_bess_singleplot_fast)    
sol_ks_bess_singleplot_slow = np.array(sol_ks_bess_singleplot_slow)
sol_omegas_bess_singleplot_slow = np.array(sol_omegas_bess_singleplot_slow)    
  
sol_ks_bess_singleplot_fast_kink = np.array(sol_ks_bess_singleplot_fast_kink)
sol_omegas_bess_singleplot_fast_kink = np.array(sol_omegas_bess_singleplot_fast_kink)
sol_ks_bess_singleplot_slow_kink = np.array(sol_ks_bess_singleplot_slow_kink)
sol_omegas_bess_singleplot_slow_kink = np.array(sol_omegas_bess_singleplot_slow_kink)
 
##########################################

#ax.plot(sol_ks1e5_singleplot_fast, (sol_omegas1e5_singleplot_fast/sol_ks1e5_singleplot_fast), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_fast, (sol_omegas3_singleplot_fast/sol_ks3_singleplot_fast), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_fast, (sol_omegas15_singleplot_fast/sol_ks15_singleplot_fast), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_fast, (sol_omegas09_singleplot_fast/sol_ks09_singleplot_fast), 'r.', markersize=10.)
#ax.plot(sol_ks_bess_singleplot_fast, (sol_omegas_bess_singleplot_fast/sol_ks_bess_singleplot_fast), 'b.', markersize=10.)


ax.plot(sol_ks1e5_singleplot_fast_kink, (sol_omegas1e5_singleplot_fast_kink/sol_ks1e5_singleplot_fast_kink), 'k.', markersize=10.)
ax.plot(sol_ks3_singleplot_fast_kink, (sol_omegas3_singleplot_fast_kink/sol_ks3_singleplot_fast_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
ax.plot(sol_ks15_singleplot_fast_kink, (sol_omegas15_singleplot_fast_kink/sol_ks15_singleplot_fast_kink), 'g.', markersize=10.)
ax.plot(sol_ks09_singleplot_fast_kink, (sol_omegas09_singleplot_fast_kink/sol_ks09_singleplot_fast_kink), 'r.', markersize=10.)
ax.plot(sol_ks_bess_singleplot_fast_kink, (sol_omegas_bess_singleplot_fast_kink/sol_ks_bess_singleplot_fast_kink), 'b.', markersize=10.)


#ax.plot(sol_ks1e5_singleplot_slow, (sol_omegas1e5_singleplot_slow/sol_ks1e5_singleplot_slow), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_slow, (sol_omegas3_singleplot_slow/sol_ks3_singleplot_slow), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_slow, (sol_omegas15_singleplot_slow/sol_ks15_singleplot_slow), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_slow, (sol_omegas09_singleplot_slow/sol_ks09_singleplot_slow), 'r.', markersize=10.)
#ax.plot(sol_ks_bess_singleplot_slow, (sol_omegas_bess_singleplot_slow/sol_ks_bess_singleplot_slow), 'b.', markersize=10.)


#ax.plot(sol_ks1e5_singleplot_slow_kink, (sol_omegas1e5_singleplot_slow_kink/sol_ks1e5_singleplot_slow_kink), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_slow_kink, (sol_omegas3_singleplot_slow_kink/sol_ks3_singleplot_slow_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_slow_kink, (sol_omegas15_singleplot_slow_kink/sol_ks15_singleplot_slow_kink), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_slow_kink, (sol_omegas09_singleplot_slow_kink/sol_ks09_singleplot_slow_kink), 'r.', markersize=10.)
#ax.plot(sol_ks_bess_singleplot_slow_kink, (sol_omegas_bess_singleplot_slow_kink/sol_ks_bess_singleplot_slow_kink), 'b.', markersize=10.)

   
#plt.savefig("photospheric_slab_fast_kink_disp_diagram.png")

#print(len(sol_ks1e5_singleplot_fast))

print('k   =', sol_ks1e5_singleplot_fast)
print('w   =', sol_omegas1e5_singleplot_fast)


print('k val   =', sol_ks1e5_singleplot_fast[0])
print('w val   =', sol_omegas1e5_singleplot_fast[0])

#plt.show()
#exit()   

wavenum = sol_ks1e5_singleplot_fast_kink
frequency = sol_omegas1e5_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
x0=0.  #mean
dx=1e5 #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   

def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#############


lx = np.linspace(-7.*2.*np.pi/wavenum[0], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
       
m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)))

p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_e**2)-frequency[0]**2))

      
######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
      
def F(x):  
    return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2))/((wavenum[0]**2*c_i(x)**2)-frequency[0]**2))  

F_np=sym.lambdify(xx,F(xx),"numpy")   

      
def dF(x):   #First derivative of profile in symbols    
    return sym.diff(F(x), x)

dF_np=sym.lambdify(xx,dF(xx),"numpy")    
#dF_plot, = ax5.plot(ix, dF_np(ix), 'b')
       
def m0(x):    
    return ((((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)*((wavenum[0]**2*vA_i(x)**2)-frequency[0]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)))

m0_np=sym.lambdify(xx,m0(xx),"numpy")  
#m0_plot, = ax6.plot(ix, m0_np(ix), 'b')

def P_Ti(x):    
    return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)))  

PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
p_i_const = PT_i_np(ix)

######################################################


def dVx_dx_e(Vx_e, x_e):
    return [Vx_e[1], m_e*Vx_e[0]]

V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side

normalised_left_P_solution_1e5 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_vx_solution_1e5 = left_solution/np.amax(abs(left_solution))
left_bound_vx = left_solution[-1] 
  
def dVx_dx_i(Vx_i, x_i):
    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           

def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
    u1 = U[:,0] - left_bound_vx    # + for sausage,   - for kink
    return u1[-1] 
  
dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??

# now solve with optimal dvx

Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
inside_solution = Is[:,0]
#inside_P_solution = p_i_const*Is[:,1]

inside_P_solution = np.multiply(p_i_const, Is[:,1])              

normalised_inside_P_solution_1e5 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_vx_solution_1e5 = inside_solution/np.amax(abs(left_solution))



normalisation_factor_pressure = np.amax(abs(left_P_solution))
normalisation_factor_vx = np.amax(abs(left_solution))




wavenum = sol_ks3_singleplot_fast_kink
frequency = sol_omegas3_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
x0=0.  #mean
dx=3. #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   

def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#############


lx = np.linspace(-7.*2.*np.pi/wavenum[0], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
       
m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)))

p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_e**2)-frequency[0]**2))

      
######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
      
def F(x):  
    return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2))/((wavenum[0]**2*c_i(x)**2)-frequency[0]**2))  

F_np=sym.lambdify(xx,F(xx),"numpy")   

      
def dF(x):   #First derivative of profile in symbols    
    return sym.diff(F(x), x)

dF_np=sym.lambdify(xx,dF(xx),"numpy")    
#dF_plot, = ax5.plot(ix, dF_np(ix), 'b')
       
def m0(x):    
    return ((((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)*((wavenum[0]**2*vA_i(x)**2)-frequency[0]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)))

m0_np=sym.lambdify(xx,m0(xx),"numpy")  
#m0_plot, = ax6.plot(ix, m0_np(ix), 'b')

def P_Ti(x):    
    return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)))  

PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
p_i_const = PT_i_np(ix)

######################################################


def dVx_dx_e(Vx_e, x_e):
    return [Vx_e[1], m_e*Vx_e[0]]

V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side

normalised_left_P_solution_3 = left_P_solution/np.amax(abs(left_P_solution))    #normalisation_factor_pressure   #
normalised_left_vx_solution_3 = left_solution/np.amax(abs(left_solution))     #normalisation_factor_vx   #
left_bound_vx = left_solution[-1] 
  
def dVx_dx_i(Vx_i, x_i):
    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           

def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
    u1 = U[:,0] - left_bound_vx    # + for sausage,   - for kink
    return u1[-1] 
  
dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??

# now solve with optimal dvx

Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
inside_solution = Is[:,0]
#inside_P_solution = p_i_const*Is[:,1]

inside_P_solution = np.multiply(p_i_const, Is[:,1])              

normalised_inside_P_solution_3 = inside_P_solution/np.amax(abs(left_P_solution))   #normalisation_factor_pressure   #
normalised_inside_vx_solution_3 = inside_solution/np.amax(abs(left_solution))    #normalisation_factor_vx   #

################################


wavenum = sol_ks15_singleplot_fast_kink
frequency = sol_omegas15_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
x0=0.  #mean
dx=1.5 #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   

def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#############


lx = np.linspace(-7.*2.*np.pi/wavenum[0], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
       
m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)))

p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_e**2)-frequency[0]**2))

      
######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
      
def F(x):  
    return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2))/((wavenum[0]**2*c_i(x)**2)-frequency[0]**2))  

F_np=sym.lambdify(xx,F(xx),"numpy")   

      
def dF(x):   #First derivative of profile in symbols    
    return sym.diff(F(x), x)

dF_np=sym.lambdify(xx,dF(xx),"numpy")    
#dF_plot, = ax5.plot(ix, dF_np(ix), 'b')
       
def m0(x):    
    return ((((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)*((wavenum[0]**2*vA_i(x)**2)-frequency[0]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)))

m0_np=sym.lambdify(xx,m0(xx),"numpy")  
#m0_plot, = ax6.plot(ix, m0_np(ix), 'b')

def P_Ti(x):    
    return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)))  

PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
p_i_const = PT_i_np(ix)

######################################################


def dVx_dx_e(Vx_e, x_e):
    return [Vx_e[1], m_e*Vx_e[0]]

V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side

normalised_left_P_solution_15 = left_P_solution/np.amax(abs(left_P_solution))       #normalisation_factor_pressure   #
normalised_left_vx_solution_15 = left_solution/np.amax(abs(left_solution))         #normalisation_factor_vx   #
left_bound_vx = left_solution[-1] 
  
def dVx_dx_i(Vx_i, x_i):
    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           

def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
    u1 = U[:,0] - left_bound_vx    # + for sausage,   - for kink
    return u1[-1] 
  
dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??

# now solve with optimal dvx

Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
inside_solution = Is[:,0]
#inside_P_solution = p_i_const*Is[:,1]

inside_P_solution = np.multiply(p_i_const, Is[:,1])              

normalised_inside_P_solution_15 = inside_P_solution/np.amax(abs(left_P_solution))      #normalisation_factor_pressure    #
normalised_inside_vx_solution_15 = inside_solution/np.amax(abs(left_solution))       #normalisation_factor_vx     #



################################


wavenum = sol_ks09_singleplot_fast_kink
frequency = sol_omegas09_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
x0=0.  #mean
dx=0.9 #standard dev

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(x-x0)**2/dx**2)))   

def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#############


lx = np.linspace(-7.*2.*np.pi/wavenum[0], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
       
m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)))

p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_e**2)-frequency[0]**2))

      
######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
      
def F(x):  
    return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2))/((wavenum[0]**2*c_i(x)**2)-frequency[0]**2))  

F_np=sym.lambdify(xx,F(xx),"numpy")   

      
def dF(x):   #First derivative of profile in symbols    
    return sym.diff(F(x), x)

dF_np=sym.lambdify(xx,dF(xx),"numpy")    
#dF_plot, = ax5.plot(ix, dF_np(ix), 'b')
       
def m0(x):    
    return ((((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)*((wavenum[0]**2*vA_i(x)**2)-frequency[0]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)))

m0_np=sym.lambdify(xx,m0(xx),"numpy")  
#m0_plot, = ax6.plot(ix, m0_np(ix), 'b')

def P_Ti(x):    
    return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)))  

PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
p_i_const = PT_i_np(ix)

######################################################


def dVx_dx_e(Vx_e, x_e):
    return [Vx_e[1], m_e*Vx_e[0]]

V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side

normalised_left_P_solution_09 = left_P_solution/np.amax(abs(left_P_solution))     #normalisation_factor_pressure    #
normalised_left_vx_solution_09 = left_solution/np.amax(abs(left_solution))      # normalisation_factor_vx   #
left_bound_vx = left_solution[-1] 
  
def dVx_dx_i(Vx_i, x_i):
    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           

def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
    u1 = U[:,0] - left_bound_vx    # + for sausage,   - for kink
    return u1[-1] 
  
dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??

# now solve with optimal dvx

Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
inside_solution = Is[:,0]
#inside_P_solution = p_i_const*Is[:,1]

inside_P_solution = np.multiply(p_i_const, Is[:,1])              

normalised_inside_P_solution_09 = inside_P_solution/np.amax(abs(left_P_solution))      #normalisation_factor_pressure    #
normalised_inside_vx_solution_09 = inside_solution/np.amax(abs(left_solution))      #normalisation_factor_vx    #


##################################

wavenum = sol_ks_bess_singleplot_fast_kink
frequency = sol_omegas_bess_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############

xx=sym.symbols('x')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(x):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (-1./10.*(rho_i0*(sym.sin(10.*x)/x)) + rho_i0)/4. + rho_i0  

def rho_i(x):                  # Define the internal density profile
    return rho_A*profile(x)

#############################################

def P_i(x):
    return ((c_i(x))**2*rho_i(x)/gamma)

#############################################

def T_i(x):     #use this for constant P
    return (P_i(x)/rho_i(x))

##########################################
def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))

##########################################
def B_i(x):
    return (vA_i(x)*sym.sqrt(rho_i(x)))

###################################

def PT_i(x):
    return P_i(x) + B_i(x)**2/2.

###################################

def c_i(x):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(x)) - 0.5*gamma*vA_i(x)**2)

###################################################
def cT_i(x):                 # Define the internal tube speed
    return sym.sqrt(((c_i(x))**2 * (vA_i(x))**2) / ((c_i(x))**2 + (vA_i(x))**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

#############


lx = np.linspace(-7.*2.*np.pi/wavenum[0], -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
       
m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)))

p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum[0]**2*cT_e()**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_e**2)-frequency[0]**2))

      
######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
      
def F(x):  
    return ((rho_i(x)*(c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2))/((wavenum[0]**2*c_i(x)**2)-frequency[0]**2))  

F_np=sym.lambdify(xx,F(xx),"numpy")   

      
def dF(x):   #First derivative of profile in symbols    
    return sym.diff(F(x), x)

dF_np=sym.lambdify(xx,dF(xx),"numpy")    
#dF_plot, = ax5.plot(ix, dF_np(ix), 'b')
       
def m0(x):    
    return ((((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)*((wavenum[0]**2*vA_i(x)**2)-frequency[0]**2))/((c_i(x)**2+vA_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)))

m0_np=sym.lambdify(xx,m0(xx),"numpy")  
#m0_plot, = ax6.plot(ix, m0_np(ix), 'b')

def P_Ti(x):    
    return (rho_i(x)*(vA_i(x)**2+c_i(x)**2)*((wavenum[0]**2*cT_i(x)**2)-frequency[0]**2)/(frequency[0]*((wavenum[0]**2*c_i(x)**2)-frequency[0]**2)))  

PT_i_np=sym.lambdify(xx,P_Ti(xx),"numpy")
p_i_const = PT_i_np(ix)

######################################################


def dVx_dx_e(Vx_e, x_e):
    return [Vx_e[1], m_e*Vx_e[0]]

V0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
left_solution = Ls[:,0]      # Vx perturbation solution for left hand side
left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side

normalised_left_P_solution_bess = left_P_solution/np.amax(abs(left_P_solution))     #normalisation_factor_pressure    #
normalised_left_vx_solution_bess = left_solution/np.amax(abs(left_solution))      #normalisation_factor_vx   #
left_bound_vx = left_solution[-1] 
  
def dVx_dx_i(Vx_i, x_i):
    return [Vx_i[1], ((-dF_np(x_i)/F_np(x_i))*Vx_i[1] + m0_np(x_i)*Vx_i[0])]           

def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
    u1 = U[:,0] - left_bound_vx    # + for sausage,   - for kink
    return u1[-1] 
  
dVxi, = fsolve(objective_dvxi, 1.)    #zero is guess for roots of equation, maybe change to find more body modes??

# now solve with optimal dvx

Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
inside_solution = Is[:,0]
#inside_P_solution = p_i_const*Is[:,1]

inside_P_solution = np.multiply(p_i_const, Is[:,1])              

normalised_inside_P_solution_bess = inside_P_solution/np.amax(abs(left_P_solution))      #normalisation_factor_pressure    #
normalised_inside_vx_solution_bess = inside_solution/np.amax(abs(left_solution))      #normalisation_factor_vx    #








###################################################################
fig, (ax, ax2) = plt.subplots(2,1, sharex=False) 
ax.axvline(x=-B, color='r', linestyle='--')
ax.axvline(x=B, color='r', linestyle='--')
#ax.set_xlabel("$x$", fontsize=18)
ax.set_ylabel("$P_T$", fontsize=18, rotation=0, labelpad=15)
#ax.set_ylim(-1.2,1.2)
ax.set_xlim(-3.,1.2)
ax.plot(lx, normalised_left_P_solution_1e5, 'k')
ax.plot(ix, normalised_inside_P_solution_1e5, 'k')

ax.plot(lx, normalised_left_P_solution_3, color='goldenrod')
ax.plot(ix, normalised_inside_P_solution_3, color='goldenrod')

ax.plot(lx, normalised_left_P_solution_15, 'g')
ax.plot(ix, normalised_inside_P_solution_15, 'g')

ax.plot(lx, normalised_left_P_solution_09, 'r')
ax.plot(ix, normalised_inside_P_solution_09, 'r')

ax.plot(lx, normalised_left_P_solution_bess, 'b')
ax.plot(ix, normalised_inside_P_solution_bess, 'b')



ax2.axvline(x=-B, color='r', linestyle='--')
ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$x$", fontsize=18)
ax2.set_ylabel("$v_x$", fontsize=18, rotation=0, labelpad=15)
#ax2.set_ylim(-1.2,1.2)
ax2.set_xlim(-3.,1.2)
ax2.plot(lx, normalised_left_vx_solution_1e5, 'k')
ax2.plot(ix, normalised_inside_vx_solution_1e5, 'k')

ax2.plot(lx, normalised_left_vx_solution_3, color='goldenrod')
ax2.plot(ix, normalised_inside_vx_solution_3, color='goldenrod')

ax2.plot(lx, normalised_left_vx_solution_15, 'g')
ax2.plot(ix, normalised_inside_vx_solution_15, 'g')

ax2.plot(lx, normalised_left_vx_solution_09, 'r')
ax2.plot(ix, normalised_inside_vx_solution_09, 'r')

ax2.plot(lx, normalised_left_vx_solution_bess, 'b')
ax2.plot(ix, normalised_inside_vx_solution_bess, 'b')


plt.savefig("photospheric_slab_fastsurf_kink_eigenfunctions_stdnorm.png")


###############################################################################


sol_ks1 = sol_ks_bess
sol_omegas1 = sol_omegas_bess

sol_omegas_kink1 = sol_omegas_kink_bess
sol_ks_kink1 = sol_ks_kink_bess


#sol_ks1 = sol_ks15
#sol_omegas1 = sol_omegas15

#sol_omegas_kink1 = sol_omegas_kink15
#sol_ks_kink1 = sol_ks_kink15



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
      
  if v_phase > c_i0 and v_phase < c_e:  
      fast_surf_sausage_omega.append(sol_omegas1[i])
      fast_surf_sausage_k.append(sol_ks1[i])
      
      
  elif v_phase > vA_e and v_phase < cT_i0:
      slow_surf_sausage_omega.append(sol_omegas1[i])
      slow_surf_sausage_k.append(sol_ks1[i])

  elif v_phase > cT_i0 and v_phase < c_i0:
      body_sausage_omega.append(sol_omegas1[i])
      body_sausage_k.append(sol_ks1[i])
            

for i in range(len(sol_ks_kink1)):   #kink mode
  v_phase = sol_omegas_kink1[i]/sol_ks_kink1[i]
      
  if v_phase > c_i0 and v_phase < c_e:  
      fast_surf_kink_omega.append(sol_omegas_kink1[i])
      fast_surf_kink_k.append(sol_ks_kink1[i])
      
      
  elif v_phase > vA_e and v_phase < cT_i0:
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
body_sausage_branch1_omega = []
body_sausage_branch1_k = []
body_sausage_branch2_omega = []
body_sausage_branch2_k = [] 

body_sausage_omega = body_sausage_omega[::-1]
body_sausage_k = body_sausage_k[::-1]

body_sausage_branch1_omega.append(body_sausage_omega[0])
body_sausage_branch1_k.append(body_sausage_k[0])  

for i in range(len(body_sausage_omega)-1):
    
    k_diff = abs((body_sausage_k[i+1] - body_sausage_branch1_k[-1]))
    #omega_diff = slow_body_sausage_omega[i+1] - slow_body_sausage_omega[i]
    ph_diff = abs((body_sausage_omega[i+1]/body_sausage_k[i+1]) - (body_sausage_branch1_omega[-1]/body_sausage_branch1_k[-1]))
    #ph_diff = abs((body_sausage_omega[i+1]/body_sausage_k[i+1]) - (body_sausage_branch1_omega[i]/body_sausage_branch1_k[i]))
    
    if ph_diff < 0.01 and k_diff < 0.5:    #remove k_diff for W=2.5
      body_sausage_branch1_omega.append(body_sausage_omega[i+1])
      body_sausage_branch1_k.append(body_sausage_k[i+1])         

    else:
      body_sausage_branch2_omega.append(body_sausage_omega[i+1])
      body_sausage_branch2_k.append(body_sausage_k[i+1])
       
 
#### test delete this after   #### 
index_to_remove = []

for i in range(len(body_sausage_branch1_omega)-1):
  
  ph_diff = abs((body_sausage_branch1_omega[i+1]/body_sausage_branch1_k[i+1]) - (body_sausage_branch1_omega[i]/body_sausage_branch1_k[i]))
  
  if ph_diff > 0.01:  #0.01   for small W
      index_to_remove.append(i)


body_sausage_branch1_omega = np.delete(body_sausage_branch1_omega, index_to_remove)
body_sausage_branch1_k = np.delete(body_sausage_branch1_k, index_to_remove)

 
body_sausage_branch1_omega = np.array(body_sausage_branch1_omega)
body_sausage_branch1_k = np.array(body_sausage_branch1_k)
body_sausage_branch2_omega = np.array(body_sausage_branch2_omega)
body_sausage_branch2_k = np.array(body_sausage_branch2_k) 

body_sausage_branch1_omega = body_sausage_branch1_omega[::-1]
body_sausage_branch1_k = body_sausage_branch1_k[::-1]


index_to_remove = []

for i in range(len(body_sausage_branch2_omega)-1):
  
  ph_diff = abs((body_sausage_branch2_omega[i+1]/body_sausage_branch2_k[i+1]) - (body_sausage_branch2_omega[i]/body_sausage_branch2_k[i]))
  
  if ph_diff > 0.05:  #0.01
      index_to_remove.append(i)

## emergency deleting !!   ###
for i in range(len(body_sausage_branch2_omega)):
    ph_speed = body_sausage_branch2_omega[i]/body_sausage_branch2_k[i]
    if ph_speed > c_bound:
        index_to_remove.append(i)
        sausage_extra_branch_k.append(body_sausage_branch2_k[i])
        sausage_extra_branch_omega.append(body_sausage_branch2_omega[i])
 
      
body_sausage_branch2_omega = np.delete(body_sausage_branch2_omega, index_to_remove)
body_sausage_branch2_k = np.delete(body_sausage_branch2_k, index_to_remove)



######     MODIFY EXTRA BRANCH      ###########
sausage_extra_branch_k = np.array(sausage_extra_branch_k)
sausage_extra_branch_omega = np.array(sausage_extra_branch_omega)

sausage_extra_branch2_k = []
sausage_extra_branch2_omega = []

index_to_remove = []

for i in range(len(sausage_extra_branch_omega)):
  ph_speed = abs(sausage_extra_branch_omega[i]/sausage_extra_branch_k[i])  
  #ph_diff = abs((sausage_extra_branch_omega[i+1]/sausage_extra_branch_k[i+1]) - (sausage_extra_branch_omega[i]/sausage_extra_branch_k[i]))
  
  if ph_speed < 0.92 and sausage_extra_branch_k[i] < 2:   #0.9
      index_to_remove.append(i)
      sausage_extra_branch2_k.append(sausage_extra_branch_k[i])
      sausage_extra_branch2_omega.append(sausage_extra_branch_omega[i])
  

  if ph_speed < 0.93 and sausage_extra_branch_k[i] > 2:
      index_to_remove.append(i)
      sausage_extra_branch2_k.append(sausage_extra_branch_k[i])
      sausage_extra_branch2_omega.append(sausage_extra_branch_omega[i])
           

sausage_extra_branch_omega = np.delete(sausage_extra_branch_omega, index_to_remove)
sausage_extra_branch_k = np.delete(sausage_extra_branch_k, index_to_remove)

index_to_remove = []

for i in range(len(sausage_extra_branch_omega)-1):  
  ph_diff = abs((sausage_extra_branch_omega[i+1]/sausage_extra_branch_k[i+1]) - (sausage_extra_branch_omega[i]/sausage_extra_branch_k[i]))
  
  if ph_diff > 0.01:  #0.01
      index_to_remove.append(i)
sausage_extra_branch_omega = np.delete(sausage_extra_branch_omega, index_to_remove)
sausage_extra_branch_k = np.delete(sausage_extra_branch_k, index_to_remove)
  
  
index_to_remove = []

for i in range(len(sausage_extra_branch2_omega)-1):  
  ph_diff = abs((sausage_extra_branch2_omega[i+1]/sausage_extra_branch2_k[i+1]) - (sausage_extra_branch2_omega[i]/sausage_extra_branch2_k[i]))
  
  if ph_diff > 0.01:  #0.01
      index_to_remove.append(i)
sausage_extra_branch2_omega = np.delete(sausage_extra_branch2_omega, index_to_remove)
sausage_extra_branch2_k = np.delete(sausage_extra_branch2_k, index_to_remove)
      
sausage_extra_branch2_k = np.array(sausage_extra_branch2_k)
sausage_extra_branch2_omega = np.array(sausage_extra_branch2_omega)

if len(sausage_extra_branch_omega) > 1:
  ESB2_phase = sausage_extra_branch_omega/sausage_extra_branch_k  
  ESB2_k = sausage_extra_branch_k 
  ESB2_k_new = np.linspace(ESB2_k[0], ESB2_k[-1], num=len(ESB2_k)*10)    #did finish at Kmax other than W=2.5
  
  ESB2_coefs = poly.polyfit(ESB2_k, ESB2_phase, 2)   #was 6    # 1 good for W > 3
  ESB2_ffit = poly.polyval(ESB2_k_new, ESB2_coefs)

if len(sausage_extra_branch2_omega) > 1:
  ESB3_phase = sausage_extra_branch2_omega/sausage_extra_branch2_k  
  ESB3_k = sausage_extra_branch2_k 
  ESB3_k_new = np.linspace(ESB3_k[-1], Kmax, num=len(ESB3_k)*10)    #did finish at Kmax other than W=2.5
  
  ESB3_coefs = poly.polyfit(ESB3_k, ESB3_phase, 1)   #was 6    # 1 good for W > 3
  ESB3_ffit = poly.polyval(ESB3_k_new, ESB3_coefs)

################################################

body_sausage_branch2_omega = body_sausage_branch2_omega[::-1]
body_sausage_branch2_k = body_sausage_branch2_k[::-1]

#   sausage body polyfit
SB_phase = body_sausage_branch1_omega/body_sausage_branch1_k  
SB_k = body_sausage_branch1_k 
SB_k_new = np.linspace(SB_k[0], SB_k[-1], num=len(SB_k)*10)      #SB_k[-1]

SB_coefs = poly.polyfit(SB_k, SB_phase, 6)
SB_ffit = poly.polyval(SB_k_new, SB_coefs)

if len(body_sausage_branch2_omega) > 1:
  SB2_phase = body_sausage_branch2_omega/body_sausage_branch2_k  
  SB2_k = body_sausage_branch2_k 
  SB2_k_new = np.linspace(SB2_k[0], SB2_k[-1], num=len(SB2_k)*10)    #did finish at Kmax other than W=2.5     SB2_k[-1]
  
  SB2_coefs = poly.polyfit(SB2_k, SB2_phase, 1)   #was 6    # 1 good for W > 3
  SB2_ffit = poly.polyval(SB2_k_new, SB2_coefs)

    
################   kink body  ################
body_kink_branch1_omega = []
body_kink_branch1_k = []
body_kink_branch2_omega = []
body_kink_branch2_k = [] 

body_kink_omega = body_kink_omega[::-1]
body_kink_k = body_kink_k[::-1]

body_kink_branch1_omega.append(body_kink_omega[0])
body_kink_branch1_k.append(body_kink_k[0])  

for i in range(len(body_kink_omega)-1):
  if body_kink_k[i+1] > 0.8:  
    ph_diff = abs((body_kink_omega[i+1]/body_kink_k[i+1]) - (body_kink_branch1_omega[-1]/body_kink_branch1_k[-1]))
    
    if ph_diff < 0.01:
      body_kink_branch1_omega.append(body_kink_omega[i+1])
      body_kink_branch1_k.append(body_kink_k[i+1])         

    else:
      body_kink_branch2_omega.append(body_kink_omega[i+1])
      body_kink_branch2_k.append(body_kink_k[i+1])

combine_kink_k = []
combine_kink_omega = []
for i in range(len(body_kink_omega)):
    if body_kink_k[i] < 0.8:
        combine_kink_k.append(body_kink_k[i])
        combine_kink_omega.append(body_kink_omega[i])
      
combine_kink_k = np.array(combine_kink_k)
combine_kink_omega = np.array(combine_kink_omega)

#######    COMBINE FAST SURFACE KINK WITH BODY BRANCH   ######

combine_kink_k = np.concatenate((combine_kink_k, fast_surf_kink_k), axis=0)
combine_kink_omega = np.concatenate((combine_kink_omega, fast_surf_kink_omega), axis=0)

combine_kink_omega = [x for _,x in sorted(zip(combine_kink_k,combine_kink_omega))]
combine_kink_k = np.sort(combine_kink_k)

index_to_remove = []

for i in range(len(combine_kink_omega)-1):
  
  ph_diff = abs((combine_kink_omega[i+1]/combine_kink_k[i+1]) - (combine_kink_omega[i]/combine_kink_k[i]))
  
  if ph_diff > 0.02:  
      index_to_remove.append(i)
      
combine_kink_omega = np.delete(combine_kink_omega, index_to_remove)
combine_kink_k = np.delete(combine_kink_k, index_to_remove)


#combine_kink_k = np.array(combine_kink_k)
#combine_kink_omega = np.array(combine_kink_omega)

FSK_phase = combine_kink_omega/combine_kink_k  #combined fast surface and body kink
FSK_k = combine_kink_k
FSK_k_new = np.linspace(FSK_k[0], Kmax, num=len(FSK_k)*10)

FSK_coefs = poly.polyfit(FSK_k, FSK_phase, 8)    #was 6
FSK_ffit = poly.polyval(FSK_k_new, FSK_coefs)

##############################################################


      
body_kink_branch1_omega = np.array(body_kink_branch1_omega)
body_kink_branch1_k = np.array(body_kink_branch1_k)
body_kink_branch2_omega = np.array(body_kink_branch2_omega)
body_kink_branch2_k = np.array(body_kink_branch2_k) 

body_kink_branch1_omega = body_kink_branch1_omega[::-1]
body_kink_branch1_k = body_kink_branch1_k[::-1]
body_kink_branch2_omega = body_kink_branch2_omega[::-1]
body_kink_branch2_k = body_kink_branch2_k[::-1]

index_to_remove = []

for i in range(len(body_kink_branch2_omega)-1):
  
  ph_diff = abs((body_kink_branch2_omega[i+1]/body_kink_branch2_k[i+1]) - (body_kink_branch2_omega[i]/body_kink_branch2_k[i]))
  
  if ph_diff > 0.01:  
      index_to_remove.append(i)

### emergency deleting !!   ###
for i in range(len(body_kink_branch2_omega)):
    ph_speed = body_kink_branch2_omega[i]/body_kink_branch2_k[i]
    if ph_speed > c_bound:
        index_to_remove.append(i)
        
             
body_kink_branch2_omega = np.delete(body_kink_branch2_omega, index_to_remove)
body_kink_branch2_k = np.delete(body_kink_branch2_k, index_to_remove)

#   kink body polyfit
KB_phase = body_kink_branch1_omega/body_kink_branch1_k  
KB_k = body_kink_branch1_k 
KB_k_new = np.linspace(KB_k[0], Kmax, num=len(KB_k)*1)    #10 works    # 1 for W=2.5     #Kmax for w =< 1    KB_k[-1] otherwise   

KB_coefs = poly.polyfit(KB_k, KB_phase, 1)     # use 6    # use 1 for W <=1.75
KB_ffit = poly.polyval(KB_k_new, KB_coefs)

if len(body_kink_branch2_omega) > 1:
   KB2_phase = body_kink_branch2_omega/body_kink_branch2_k  
   KB2_k = body_kink_branch2_k 
   KB2_k_new = np.linspace(KB2_k[0], KB2_k[-1], num=len(KB2_k)*10)
   
   KB2_coefs = poly.polyfit(KB2_k, KB2_phase, 6)   # was 6   # 1 good
   KB2_ffit = poly.polyval(KB2_k_new, KB2_coefs)



FSS_phase = fast_surf_sausage_omega/fast_surf_sausage_k   #fast surface
FSS_k = fast_surf_sausage_k
FSS_k_new = np.linspace(FSS_k[0], Kmax, num=len(FSS_k)*10)

FSS_coefs = poly.polyfit(FSS_k, FSS_phase,6)    #was 6
FSS_ffit = poly.polyval(FSS_k_new, FSS_coefs)

if len(slow_surf_sausage_omega) > 1:
  SSS_phase = slow_surf_sausage_omega/slow_surf_sausage_k   #slow surface
  SSS_k = slow_surf_sausage_k
  SSS_k_new = np.linspace(SSS_k[0], Kmax, num=len(SSS_k)*10)    #SSS_k[0]
  
  SSS_coefs = poly.polyfit(SSS_k, SSS_phase, 6)
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
  SSK_k_new = np.linspace(SSK_k[0], SSK_k[-1], num=len(SSK_k)*10)   #SSK_k[-1]
  
  SSK_coefs = poly.polyfit(SSK_k, SSK_phase, 15)   #was 6     #use 2 for W=2.5
  SSK_ffit = poly.polyval(SSK_k_new, SSK_coefs)






###############################################################################

###  PLOT FULL CASE


fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot2grid((6, 5), (0, 0), rowspan=6)
#ax1.set_title("Profile", fontsize=18)
ax1.annotate( ' $\u03C1_{e}$', xy=(1.2, rho_e),fontsize=18)
ax1.annotate( ' $\u03C1_{i}$', xy=(1.2, rho_i0),fontsize=18)
ax1.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot', alpha=0.25)
ax1.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot', alpha=0.25)
ax1.set_xlabel("$x$",fontsize=18)
ax1.set_ylabel("$\u03C1$",fontsize=18, rotation=0)

ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(0.95, 1.85)
ax1.axvline(x=-1, color='r', linestyle='--')
ax1.axvline(x=1, color='r', linestyle='--')

ax1.plot(ix,profile_besslike_np(ix), 'b')
#ax1.plot(ix,profile15_np(ix), 'b')

box1 = ax1.get_position()
ax1.set_position([box1.x0-0.075, box1.y0, box1.width, box1.height])


#ax2 = plt.subplot2grid((2, 5), (0, 1), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((6, 5), (3, 1), rowspan=4, colspan=2)
#ax2.set_title("Dispersion Diagram", fontsize=18)
ax2.set_xlabel("$kx_{0}$", fontsize=18)
ax2.set_ylabel(r'$\frac{\omega}{k}$', fontsize=20, rotation=0, labelpad=15)

#ax2.plot(sol_ks_kink_bess, (sol_omegas_kink_bess/sol_ks_kink_bess), 'b.', markersize=4.)
ax2.plot(sol_ks_bess_singleplot_fast, (sol_omegas_bess_singleplot_fast/sol_ks_bess_singleplot_fast), 'b.', markersize=10.)

#ax2.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'b.', markersize=4.)


#ax2.plot(sol_ks15, (sol_omegas15/sol_ks15), 'b.', markersize=4.)
#ax2.plot(sol_ks1e5_singleplot_slow, (sol_omegas1e5_singleplot_slow/sol_ks1e5_singleplot_slow), 'b.', markersize=12.)
#ax2.plot(sol_ks15_singleplot_slow_kink, (sol_omegas15_singleplot_slow_kink/sol_ks15_singleplot_slow_kink), 'b.', markersize=12.)



#ax2.plot(combine_kink_k, combine_kink_omega/combine_kink_k, 'b.', markersize=4.)   # body kink
#ax2.plot(fast_surf_kink_k, fast_surf_kink_omega/fast_surf_kink_k, 'b.', markersize=4.)   # body kink
ax2.plot(fast_surf_sausage_k, fast_surf_sausage_omega/fast_surf_sausage_k, 'b.', markersize=4.)   # body kink
#ax2.plot(body_sausage_branch1_k, body_sausage_branch1_omega/body_sausage_branch1_k, 'b.', markersize=4.)   # body sausage
#ax2.plot(body_kink_branch1_k, body_kink_branch1_omega/body_kink_branch1_k, 'b.', markersize=4.)   # body kink
#ax2.plot(body_sausage_branch2_k, body_sausage_branch2_omega/body_sausage_branch2_k, 'b.', markersize=4.)   # body sausage
#ax2.plot(body_kink_branch2_k, body_kink_branch2_omega/body_kink_branch2_k, 'b.', markersize=4.)   # body kink
#ax2.plot(slow_surf_kink_k, slow_surf_kink_omega/slow_surf_kink_k, 'b.', markersize=4.)   # body kink
#ax2.plot(slow_surf_sausage_k, slow_surf_sausage_omega/slow_surf_sausage_k, 'b.', markersize=4.)   # body kink

#ax2.plot(SB_k_new, SB_ffit, color='b')
#ax2.plot(SB2_k_new, SB2_ffit, color='b')
#ax2.plot(KB_k_new, KB_ffit, color='b')    
#ax2.plot(KB2_k_new, KB2_ffit, color='b')


#if len(ESB2_k_new) > 1:
#    ax2.plot(ESB2_k_new, ESB2_ffit, color='b')
#    ax2.plot(sausage_extra_branch_k, sausage_extra_branch_omega/sausage_extra_branch_k, 'b.', markersize=4.)   # body kink

#if len(ESB3_k_new) > 1:
#    ax2.plot(ESB3_k_new, ESB3_ffit, color='b')
#    ax2.plot(sausage_extra_branch2_k, sausage_extra_branch2_omega/sausage_extra_branch2_k, 'b.', markersize=4.)   # body kink


ax2.plot(FSS_k_new, FSS_ffit, color='b')
#ax2.plot(SSS_k_new, SSS_ffit, color='b')
#ax2.plot(FSK_k_new2, FSK_ffit2, color='b')    
#ax2.plot(SSK_k_new, SSK_ffit, color='b')


ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax2.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=18)
ax2.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=18)
ax2.annotate( ' $c_{Te}$', xy=(Kmax, cT_e()), fontsize=18)

ax2.annotate( ' $c_{B}$', xy=(Kmax, c_bound), fontsize=18, color='k')
ax2.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=18, color='k')
ax2.annotate( ' $c_{TB}$', xy=(Kmax, cT_bound), fontsize=18, color='k')
ax2.axhline(y=c_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_bound, color='k', linestyle='dashdot', label='_nolegend_')

ax2.fill_between(wavenumber, cT_i0, cT_bound, alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, c_i0, c_bound, color='green', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values

ax2.set_ylim(0.6, 1.4)  

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width, box2.height*1.7])


ax2b = plt.subplot2grid((6, 5), (0, 1), rowspan=1, colspan=2)

box2b = ax2b.get_position()
ax2b.set_position([box2b.x0, box2b.y0, box2b.width, box2b.height])


ax2b.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2b.axhline(y=vA_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2b.fill_between(wavenumber, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
ax2b.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=18, color='k')


ax2.spines['top'].set_visible(False)
ax2b.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2b.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .015  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2b.transAxes, color='k', clip_on=False)
ax2b.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2b.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
##fig.tight_layout()

ax2b.set_ylim(1.8, 2.)  # remove blank space
ax2b.set_xticks([])



ax3 = plt.subplot2grid((6, 5), (0, 3), colspan=2, rowspan=3)
#ax3.set_title("Eigenfunctions", fontsize=18)
ax3.axvline(x=-B, color='r', linestyle='--')
ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_ylabel("$P_T$", fontsize=18, rotation=0, labelpad=15)
ax3.set_xlim(-3.,1.2)
ax3.plot(lx, normalised_left_P_solution_bess, 'b')
ax3.plot(ix, normalised_inside_P_solution_bess, 'b')

#ax3.plot(lx, normalised_left_P_solution_15, 'b')
#ax3.plot(ix, normalised_inside_P_solution_15, 'b')

box3 = ax3.get_position()
ax3.set_position([box3.x0+0.075, box3.y0+0.03, box3.width, box3.height*0.91])


ax4 = plt.subplot2grid((6, 5), (3, 3), colspan=2, rowspan=3)
ax4.axvline(x=-B, color='r', linestyle='--')
ax4.axvline(x=B, color='r', linestyle='--')
ax4.set_xlabel("$x$", fontsize=18)
ax4.set_ylabel("$v_x$", fontsize=18, rotation=0, labelpad=15)
ax4.set_xlim(-3.,1.2)
ax4.plot(lx, normalised_left_vx_solution_bess, 'b')
ax4.plot(ix, normalised_inside_vx_solution_bess, 'b')

#ax4.plot(lx, normalised_left_vx_solution_15, 'b')
#ax4.plot(ix, normalised_inside_vx_solution_15, 'b')

box4 = ax4.get_position()
ax4.set_position([box4.x0+0.075, box4.y0, box4.width, box4.height*0.91])

#plt.savefig("photospheric_slab_fastsurf_kink_bess_full.png")

plt.show()
exit()
