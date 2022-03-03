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
vA_i0 = 2.*c_i0  #-photospheric

vA_e = 0.5*c_i0  #-photospheric
c_e = 1.5*c_i0    #-photospheric

cT_i0 = np.sqrt((c_i0**2 * vA_i0**2)/(c_i0**2 + vA_i0**2))
cT_e = np.sqrt(c_e**2 * vA_e**2 / (c_e**2 + vA_e**2))

gamma=5./3.

rho_i0 = 1.
rho_e = rho_i0*(c_i0**2+gamma*0.5*vA_i0**2)/(c_e**2+gamma*0.5*vA_e**2)

print('rho_e    =', rho_e)


c_kink = np.sqrt(((rho_i0*vA_i0**2)+(rho_e*vA_e**2))/(rho_i0+rho_e))

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

Kmax = 4.5

ix = np.linspace(1., 0.001, 1e3)  # inside slab x values


r0=0.  #mean
dr=1e5 #standard dev

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

#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################


plt.figure()
#plt.title("width = 1.5")
plt.xlabel("$r$",fontsize=25)
plt.ylabel("$\u03C1$",fontsize=25, rotation=0)
ax = plt.subplot(111)   #331
#ax.title.set_text("Density")
#ax.plot(ix,rho_i_np(ix), 'k')#, linestyle='dashdot');
ax.plot(ix,profile09_np(ix), 'r')#, linestyle='dotted');
ax.plot(ix,profile15_np(ix), 'g')#, linestyle='dashed');
ax.plot(ix,profile3_np(ix), color='goldenrod')#, linestyle='dotted');
ax.plot(ix,profile1e5_np(ix), 'k')#, linestyle='solid');
ax.annotate( ' $\u03C1_{e}$', xy=(1.2, rho_e),fontsize=25)
ax.annotate( ' $\u03C1_{0i}$', xy=(1.2, rho_i0),fontsize=25)
ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot', alpha=0.25)
ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot', alpha=0.25)

ax.set_xlim(0., 1.2)
ax.set_ylim(0.95, 1.8)
ax.axvline(x=-1, color='r', linestyle='--')
ax.axvline(x=1, color='r', linestyle='--')

plt.savefig("photospheric_cylinder_density_profiles.png")

#plt.show()
#exit()

#plt.figure()
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
ax4.annotate( '$c_{Te}$', xy=(1, cT_e),fontsize=25)
ax4.annotate( '$c_{Ti}$', xy=(1, cT_i0),fontsize=25)
ax4.annotate( '$c_{TB}$', xy=(1, cT_bound), fontsize=20)
ax4.axhline(y=cT_i0, color='k', label='$c_{Ti}$', linestyle='dashdot')
ax4.axhline(y=cT_e, color='k', label='$c_{Te}$', linestyle='dashdot')
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

with open('Cylindrical_photospheric_width_1e5.pickle', 'rb') as f:
    sol_omegas1e5, sol_ks1e5, sol_omegas_kink1e5, sol_ks_kink1e5 = pickle.load(f)


with open('Cylindrical_photospheric_width_1e5_slowmodes.pickle', 'rb') as f:
    sol_omegas1e5_zoom, sol_ks1e5_zoom, sol_omegas_kink1e5_zoom, sol_ks_kink1e5_zoom = pickle.load(f)


sol_omegas1e5 = np.concatenate((sol_omegas1e5, sol_omegas1e5_zoom), axis=None)
sol_ks1e5 = np.concatenate((sol_ks1e5, sol_ks1e5_zoom), axis=None)
sol_omegas_kink1e5 = np.concatenate((sol_omegas_kink1e5, sol_omegas_kink1e5_zoom), axis=None)
sol_ks_kink1e5 = np.concatenate((sol_ks_kink1e5, sol_ks_kink1e5_zoom), axis=None)



with open('Cylindrical_photospheric_width_3.pickle', 'rb') as f:
    sol_omegas3, sol_ks3, sol_omegas_kink3, sol_ks_kink3 = pickle.load(f)


with open('Cylindrical_photospheric_width_3_slowmodes.pickle', 'rb') as f:
    sol_omegas3_zoom, sol_ks3_zoom, sol_omegas_kink3_zoom, sol_ks_kink3_zoom = pickle.load(f)


sol_omegas3 = np.concatenate((sol_omegas3, sol_omegas3_zoom), axis=None)
sol_ks3 = np.concatenate((sol_ks3, sol_ks3_zoom), axis=None)
sol_omegas_kink3 = np.concatenate((sol_omegas_kink3, sol_omegas_kink3_zoom), axis=None)
sol_ks_kink3 = np.concatenate((sol_ks_kink3, sol_ks_kink3_zoom), axis=None)


with open('Cylindrical_photospheric_width_15.pickle', 'rb') as f:
    sol_omegas15, sol_ks15, sol_omegas_kink15, sol_ks_kink15 = pickle.load(f)


with open('Cylindrical_photospheric_width_15_slowmodes.pickle', 'rb') as f:
    sol_omegas15_zoom, sol_ks15_zoom, sol_omegas_kink15_zoom, sol_ks_kink15_zoom = pickle.load(f)


sol_omegas15 = np.concatenate((sol_omegas15, sol_omegas15_zoom), axis=None)
sol_ks15 = np.concatenate((sol_ks15, sol_ks15_zoom), axis=None)
sol_omegas_kink15 = np.concatenate((sol_omegas_kink15, sol_omegas_kink15_zoom), axis=None)
sol_ks_kink15 = np.concatenate((sol_ks_kink15, sol_ks_kink15_zoom), axis=None)



with open('Cylindrical_photospheric_width_09.pickle', 'rb') as f:
    sol_omegas09, sol_ks09, sol_omegas_kink09, sol_ks_kink09 = pickle.load(f)


with open('Cylindrical_photospheric_width_09_slowmodes.pickle', 'rb') as f:
    sol_omegas09_zoom, sol_ks09_zoom, sol_omegas_kink09_zoom, sol_ks_kink09_zoom = pickle.load(f)


sol_omegas09 = np.concatenate((sol_omegas09, sol_omegas09_zoom), axis=None)
sol_ks09 = np.concatenate((sol_ks09, sol_ks09_zoom), axis=None)
sol_omegas_kink09 = np.concatenate((sol_omegas_kink09, sol_omegas_kink09_zoom), axis=None)
sol_ks_kink09 = np.concatenate((sol_ks_kink09, sol_ks_kink09_zoom), axis=None)


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

plt.figure()

fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for photospheric to remove blank space on plot

#ax2.set_title("$ W = 3$")

plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)


#ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
#ax.plot(sol_ks3, (sol_omegas3/sol_ks3), color='goldenrod', marker='.', linestyle='', markersize=4.)
#ax.plot(sol_ks15, (sol_omegas15/sol_ks15), 'g.', markersize=4.)
#ax.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)

#ax2.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
#ax2.plot(sol_ks3, (sol_omegas3/sol_ks3), color='goldenrod', marker='.', linestyle='', markersize=4.)
#ax2.plot(sol_ks15, (sol_omegas15/sol_ks15), 'g.', markersize=4.)
#ax2.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)

ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)
ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), color='goldenrod', marker='.', linestyle='', markersize=4.)
ax.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'g.', markersize=4.)
ax.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'r.', markersize=4.)

ax2.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)
ax2.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), color='goldenrod', marker='.', linestyle='', markersize=4.)
ax2.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'g.', markersize=4.)
ax2.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'r.', markersize=4.)


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

ax.set_xlim(0., 4.5)   # whole plot
ax2.set_xlim(0., 4.5)   # whole plot


box = ax.get_position()
ax.set_position([box.x0+0.05, box.y0, box.width*0.9, box.height*0.85])

box2 = ax2.get_position()
ax2.set_position([box2.x0+0.05, box2.y0-0.05, box2.width*0.9, box2.height*1.1])


#plt.show()
#exit()





# fast cutoff  == 2.

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
    if sol_ks1e5[i] > 2.7 and sol_ks1e5[i] < 2.8:
      if sol_omegas1e5[i]/sol_ks1e5[i] > c_kink:
        sol_ks1e5_singleplot_fast.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_fast.append(sol_omegas1e5[i])
        
      elif sol_omegas1e5[i]/sol_ks1e5[i] > cT_i0 and sol_omegas1e5[i]/sol_ks1e5[i] < c_i0:
        sol_ks1e5_singleplot_slow.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_slow.append(sol_omegas1e5[i])    


for i in range(len(sol_ks_kink1e5)):
    if sol_ks_kink1e5[i] > 1.25 and sol_ks_kink1e5[i] < 1.35:        
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > 1.15 and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < 1.5:
         sol_ks1e5_singleplot_fast_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_fast_kink.append(sol_omegas_kink1e5[i])        

    if sol_ks_kink1e5[i] > 3.15 and sol_ks_kink1e5[i] < 3.2:        
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < 1. and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > 0.92: 
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
    if sol_ks3[i] > 2.7 and sol_ks3[i] < 2.8:
      if sol_omegas3[i]/sol_ks3[i] > c_kink:
        sol_ks3_singleplot_fast.append(sol_ks3[i])
        sol_omegas3_singleplot_fast.append(sol_omegas3[i])
        
      elif sol_omegas3[i]/sol_ks3[i] > cT_i0 and sol_omegas3[i]/sol_ks3[i] < c_i0:
        sol_ks3_singleplot_slow.append(sol_ks3[i])
        sol_omegas3_singleplot_slow.append(sol_omegas3[i])    

for i in range(len(sol_ks_kink3)):
    if sol_ks_kink3[i] > 1.25 and sol_ks_kink3[i] < 1.35:        
       if sol_omegas_kink3[i]/sol_ks_kink3[i] > 1.15 and sol_omegas_kink3[i]/sol_ks_kink3[i] < 1.5:
         sol_ks3_singleplot_fast_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_fast_kink.append(sol_omegas_kink3[i])        
 
    if sol_ks_kink3[i] > 3.15 and sol_ks_kink3[i] < 3.2:        
       if sol_omegas_kink3[i]/sol_ks_kink3[i] < 1. and sol_omegas_kink3[i]/sol_ks_kink3[i] > 0.91: 
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
    if sol_ks15[i] > 2.7 and sol_ks15[i] < 2.8:
      if sol_omegas15[i]/sol_ks15[i] > c_kink:
        sol_ks15_singleplot_fast.append(sol_ks15[i])
        sol_omegas15_singleplot_fast.append(sol_omegas15[i])
        
      elif sol_omegas15[i]/sol_ks15[i] > 0.91 and sol_omegas15[i]/sol_ks15[i] < c_i0:
        sol_ks15_singleplot_slow.append(sol_ks15[i])
        sol_omegas15_singleplot_slow.append(sol_omegas15[i])    

for i in range(len(sol_ks_kink15)):
    if sol_ks_kink15[i] > 1.25 and sol_ks_kink15[i] < 1.35:        
       if sol_omegas_kink15[i]/sol_ks_kink15[i] > 1.15 and sol_omegas_kink15[i]/sol_ks_kink15[i] < 1.5:
         sol_ks15_singleplot_fast_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_fast_kink.append(sol_omegas_kink15[i])        
 
    if sol_ks_kink15[i] > 3.13 and sol_ks_kink15[i] < 3.2:        
       if sol_omegas_kink15[i]/sol_ks_kink15[i] < 0.91 and sol_omegas_kink15[i]/sol_ks_kink15[i] > 0.895: 
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
    if sol_ks09[i] > 2.7 and sol_ks09[i] < 2.75:
      if sol_omegas09[i]/sol_ks09[i] > c_kink:
        sol_ks09_singleplot_fast.append(sol_ks09[i])
        sol_omegas09_singleplot_fast.append(sol_omegas09[i])
        
      elif sol_omegas09[i]/sol_ks09[i] > cT_i0 and sol_omegas09[i]/sol_ks09[i] < c_i0:
        sol_ks09_singleplot_slow.append(sol_ks09[i])
        sol_omegas09_singleplot_slow.append(sol_omegas09[i])    

for i in range(len(sol_ks_kink09)):
    if sol_ks_kink09[i] > 1.25 and sol_ks_kink09[i] < 1.35:        
       if sol_omegas_kink09[i]/sol_ks_kink09[i] > 1.15 and sol_omegas_kink09[i]/sol_ks_kink09[i] < 1.5:
         sol_ks09_singleplot_fast_kink.append(sol_ks_kink09[i])
         sol_omegas09_singleplot_fast_kink.append(sol_omegas_kink09[i])        
 
       elif sol_omegas_kink09[i]/sol_ks_kink09[i] < 1.2: 
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
 
   


#ax.plot(sol_ks1e5_singleplot_fast, (sol_omegas1e5_singleplot_fast/sol_ks1e5_singleplot_fast), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_fast, (sol_omegas3_singleplot_fast/sol_ks3_singleplot_fast), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_fast, (sol_omegas15_singleplot_fast/sol_ks15_singleplot_fast), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_fast, (sol_omegas09_singleplot_fast/sol_ks09_singleplot_fast), 'r.', markersize=10.)

#ax2.plot(sol_ks1e5_singleplot_fast, (sol_omegas1e5_singleplot_fast/sol_ks1e5_singleplot_fast), 'k.', markersize=10.)
#ax2.plot(sol_ks3_singleplot_fast, (sol_omegas3_singleplot_fast/sol_ks3_singleplot_fast), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax2.plot(sol_ks15_singleplot_fast, (sol_omegas15_singleplot_fast/sol_ks15_singleplot_fast), 'g.', markersize=10.)
#ax2.plot(sol_ks09_singleplot_fast, (sol_omegas09_singleplot_fast/sol_ks09_singleplot_fast), 'r.', markersize=10.)

#ax.plot(sol_ks1e5_singleplot_fast_kink, (sol_omegas1e5_singleplot_fast_kink/sol_ks1e5_singleplot_fast_kink), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_fast_kink, (sol_omegas3_singleplot_fast_kink/sol_ks3_singleplot_fast_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_fast_kink, (sol_omegas15_singleplot_fast_kink/sol_ks15_singleplot_fast_kink), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_fast_kink, (sol_omegas09_singleplot_fast_kink/sol_ks09_singleplot_fast_kink), 'r.', markersize=10.)

#ax2.plot(sol_ks1e5_singleplot_fast_kink, (sol_omegas1e5_singleplot_fast_kink/sol_ks1e5_singleplot_fast_kink), 'k.', markersize=10.)
#ax2.plot(sol_ks3_singleplot_fast_kink, (sol_omegas3_singleplot_fast_kink/sol_ks3_singleplot_fast_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax2.plot(sol_ks15_singleplot_fast_kink, (sol_omegas15_singleplot_fast_kink/sol_ks15_singleplot_fast_kink), 'g.', markersize=10.)
#ax2.plot(sol_ks09_singleplot_fast_kink, (sol_omegas09_singleplot_fast_kink/sol_ks09_singleplot_fast_kink), 'r.', markersize=10.)


#ax.plot(sol_ks1e5_singleplot_slow, (sol_omegas1e5_singleplot_slow/sol_ks1e5_singleplot_slow), 'k.', markersize=10.)
#ax.plot(sol_ks3_singleplot_slow, (sol_omegas3_singleplot_slow/sol_ks3_singleplot_slow), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax.plot(sol_ks15_singleplot_slow, (sol_omegas15_singleplot_slow/sol_ks15_singleplot_slow), 'g.', markersize=10.)
#ax.plot(sol_ks09_singleplot_slow, (sol_omegas09_singleplot_slow/sol_ks09_singleplot_slow), 'r.', markersize=10.)

#ax2.plot(sol_ks1e5_singleplot_slow, (sol_omegas1e5_singleplot_slow/sol_ks1e5_singleplot_slow), 'k.', markersize=10.)
#ax2.plot(sol_ks3_singleplot_slow, (sol_omegas3_singleplot_slow/sol_ks3_singleplot_slow), color='goldenrod', marker='.', linestyle='', markersize=10.)
#ax2.plot(sol_ks15_singleplot_slow, (sol_omegas15_singleplot_slow/sol_ks15_singleplot_slow), 'g.', markersize=10.)
#ax2.plot(sol_ks09_singleplot_slow, (sol_omegas09_singleplot_slow/sol_ks09_singleplot_slow), 'r.', markersize=10.)


ax.plot(sol_ks1e5_singleplot_slow_kink, (sol_omegas1e5_singleplot_slow_kink/sol_ks1e5_singleplot_slow_kink), 'k.', markersize=10.)
ax.plot(sol_ks3_singleplot_slow_kink, (sol_omegas3_singleplot_slow_kink/sol_ks3_singleplot_slow_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
ax.plot(sol_ks15_singleplot_slow_kink, (sol_omegas15_singleplot_slow_kink/sol_ks15_singleplot_slow_kink), 'g.', markersize=10.)
ax.plot(sol_ks09_singleplot_slow_kink, (sol_omegas09_singleplot_slow_kink/sol_ks09_singleplot_slow_kink), 'r.', markersize=10.)

ax2.plot(sol_ks1e5_singleplot_slow_kink, (sol_omegas1e5_singleplot_slow_kink/sol_ks1e5_singleplot_slow_kink), 'k.', markersize=10.)
ax2.plot(sol_ks3_singleplot_slow_kink, (sol_omegas3_singleplot_slow_kink/sol_ks3_singleplot_slow_kink), color='goldenrod', marker='.', linestyle='', markersize=10.)
ax2.plot(sol_ks15_singleplot_slow_kink, (sol_omegas15_singleplot_slow_kink/sol_ks15_singleplot_slow_kink), 'g.', markersize=10.)
ax2.plot(sol_ks09_singleplot_slow_kink, (sol_omegas09_singleplot_slow_kink/sol_ks09_singleplot_slow_kink), 'r.', markersize=10.)

  
#plt.savefig("coronal_cylinder_fast_sausage_disp_diagram.png")

#print(len(sol_ks1e5_singleplot_fast))

print('k   =', sol_ks1e5_singleplot_fast)
print('w   =', sol_omegas1e5_singleplot_fast)


#plt.show()
#exit()   


############################

#wavenum = sol_ks1e5_singleplot_fast
#frequency = sol_omegas1e5_singleplot_fast

#wavenum = sol_ks1e5_singleplot_slow
#frequency = sol_omegas1e5_singleplot_slow

#wavenum = sol_ks1e5_singleplot_fast_kink
#frequency = sol_omegas1e5_singleplot_fast_kink

wavenum = sol_ks1e5_singleplot_slow_kink
frequency = sol_omegas1e5_singleplot_slow_kink

################################

B = 1.
#m = 0.   #SAUSAGE
m = 1.   #KINK

######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
r0=0.  #mean
dr=1e5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   

def rho_i(r):                  # Define the internal density profile
    return rho_A*profile(r)


B_twist = 0.
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


v_twist = 0.
def v_iphi(r):  
    return 0.  #v_twist*r   #0.

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy 
 
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

#############

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))


shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum[0]*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum[0]*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################

lx = np.linspace(3.*2.*np.pi/wavenum[0], 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain      

m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e**2)-frequency[0]**2)))

xi_e_const = -1/(rho_e*((wavenum[0]**2*vA_e**2)-frequency[0]**2))

         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

def D(r):  
  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")

def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_phi**2/r) + (2*shift_freq(r)**2*B_phi**2/r)+(2*shift_freq(r)*B_phi*v_phi*((m*B_phi/r)+(wavenum[0]*B_i(r)))/r))

def T(r):
  return ((((m*B_phi/r)+(wavenum[0]*B_i(r)))*B_phi) + rho_i(r)*v_phi*shift_freq(r))

def C1(r):
  return ((Q(r)*shift_freq(r)) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
   
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[0]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
   
def C3_diff(r):
  return ((B_phi/r)**2 - (rho_i(r)*(v_phi/r)**2))


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
       return [P_e[1], (-P_e[1]/r_e + (m_e+m/(r_e**2))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)
left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
       return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]

    
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]  #U[:,1]  for sausage   U[:,0]  for kink    #was plus (assuming pressure is symmetric for sausage)
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))

inside_xi_phi_1e5 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_1e5 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency[0]**2 - wavenum[0]**2*vA_e**2)))
radial_xi_phi_1e5 = np.concatenate((inside_xi_phi_1e5, outside_xi_phi_1e5), axis=None)    

normalised_inside_xi_phi_1e5 = inside_xi_phi_1e5/np.amax(abs(outside_xi_phi_1e5))
normalised_outside_xi_phi_1e5 = outside_xi_phi_1e5/np.amax(abs(outside_xi_phi_1e5))
normalised_radial_xi_phi_1e5 = np.concatenate((normalised_inside_xi_phi_1e5, normalised_outside_xi_phi_1e5), axis=None)    



normalisation_factor_pressure = np.amax(abs(left_P_solution))
normalisation_factor_xi = np.amax(abs(left_xi_solution))


#################################

#wavenum = sol_ks3_singleplot_fast
#frequency = sol_omegas3_singleplot_fast

#wavenum = sol_ks3_singleplot_slow
#frequency = sol_omegas3_singleplot_slow

#wavenum = sol_ks3_singleplot_fast_kink
#frequency = sol_omegas3_singleplot_fast_kink

wavenum = sol_ks3_singleplot_slow_kink
frequency = sol_omegas3_singleplot_slow_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
r0=0.  #mean
dr=3. #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   

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

#############

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))


shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

########
def f_B(r): 
  return (m*B_iphi(r)/r + wavenum[0]*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum[0]*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################

lx = np.linspace(3.*2.*np.pi/wavenum[0], 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain      

m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e**2)-frequency[0]**2)))

xi_e_const = -1/(rho_e*((wavenum[0]**2*vA_e**2)-frequency[0]**2))

         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

def D(r):  
  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")

def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_phi**2/r) + (2*shift_freq(r)**2*B_phi**2/r)+(2*shift_freq(r)*B_phi*v_phi*((m*B_phi/r)+(wavenum[0]*B_i(r)))/r))

def T(r):
  return ((((m*B_phi/r)+(wavenum[0]*B_i(r)))*B_phi) + rho_i(r)*v_phi*shift_freq(r))

def C1(r):
  return ((Q(r)*shift_freq(r)) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
   
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[0]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
   
def C3_diff(r):
  return ((B_phi/r)**2 - (rho_i(r)*(v_phi/r)**2))


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
       return [P_e[1], (-P_e[1]/r_e + (m_e+m/(r_e**2))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)
left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution_3 = left_P_solution/np.amax(abs(left_P_solution))  #normalisation_factor_pressure
normalised_left_xi_solution_3 = left_xi_solution/np.amax(abs(left_xi_solution))  #normalisation_factor_xi
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
       return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]

    
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]  #U[:,1]  for sausage   U[:,0]  for kink    #was plus (assuming pressure is symmetric for sausage)
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side


normalised_inside_P_solution_3 = inside_P_solution/np.amax(abs(left_P_solution))  #normalisation_factor_pressure
normalised_inside_xi_solution_3 = inside_xi_solution/np.amax(abs(left_xi_solution))   #normalisation_factor_xi

inside_xi_phi_3 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_3 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency[0]**2 - wavenum[0]**2*vA_e**2)))
radial_xi_phi_3 = np.concatenate((inside_xi_phi_3, outside_xi_phi_3), axis=None)    

normalised_inside_xi_phi_3 = inside_xi_phi_3/np.amax(abs(outside_xi_phi_3))
normalised_outside_xi_phi_3 = outside_xi_phi_3/np.amax(abs(outside_xi_phi_3))
normalised_radial_xi_phi_3 = np.concatenate((normalised_inside_xi_phi_3, normalised_outside_xi_phi_3), axis=None)    


################################

#wavenum = sol_ks15_singleplot_fast
#frequency = sol_omegas15_singleplot_fast

#wavenum = sol_ks15_singleplot_slow
#frequency = sol_omegas15_singleplot_slow

#wavenum = sol_ks15_singleplot_fast_kink
#frequency = sol_omegas15_singleplot_fast_kink

wavenum = sol_ks15_singleplot_slow_kink
frequency = sol_omegas15_singleplot_slow_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
r0=0.  #mean
dr=1.5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   

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

#############

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))


shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")


#########

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum[0]*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum[0]*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################
lx = np.linspace(3.*2.*np.pi/wavenum[0], 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain      

m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e**2)-frequency[0]**2)))

xi_e_const = -1/(rho_e*((wavenum[0]**2*vA_e**2)-frequency[0]**2))

         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

def D(r):  
  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")

def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_phi**2/r) + (2*shift_freq(r)**2*B_phi**2/r)+(2*shift_freq(r)*B_phi*v_phi*((m*B_phi/r)+(wavenum[0]*B_i(r)))/r))

def T(r):
  return ((((m*B_phi/r)+(wavenum[0]*B_i(r)))*B_phi) + rho_i(r)*v_phi*shift_freq(r))

def C1(r):
  return ((Q(r)*shift_freq(r)) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
   
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[0]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
   
def C3_diff(r):
  return ((B_phi/r)**2 - (rho_i(r)*(v_phi/r)**2))


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
       return [P_e[1], (-P_e[1]/r_e + (m_e+m/(r_e**2))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)
left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution_15 = left_P_solution/np.amax(abs(left_P_solution))   #normalisation_factor_pressure
normalised_left_xi_solution_15 = left_xi_solution/np.amax(abs(left_xi_solution))   #normalisation_factor_xi
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
       return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]

    
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]  #U[:,1]  for sausage   U[:,0]  for kink    #was plus (assuming pressure is symmetric for sausage)
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side


normalised_inside_P_solution_15 = inside_P_solution/np.amax(abs(left_P_solution))  #normalisation_factor_pressure
normalised_inside_xi_solution_15 = inside_xi_solution/np.amax(abs(left_xi_solution))  #normalisation_factor_xi

inside_xi_phi_15 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_15 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency[0]**2 - wavenum[0]**2*vA_e**2)))
radial_xi_phi_15 = np.concatenate((inside_xi_phi_15, outside_xi_phi_15), axis=None)    

normalised_inside_xi_phi_15 = inside_xi_phi_15/np.amax(abs(outside_xi_phi_15))
normalised_outside_xi_phi_15 = outside_xi_phi_15/np.amax(abs(outside_xi_phi_15))
normalised_radial_xi_phi_15 = np.concatenate((normalised_inside_xi_phi_15, normalised_outside_xi_phi_15), axis=None)    




################################

#wavenum = sol_ks09_singleplot_fast
#frequency = sol_omegas09_singleplot_fast

wavenum = sol_ks09_singleplot_slow
frequency = sol_omegas09_singleplot_slow

#wavenum = sol_ks09_singleplot_fast_kink
#frequency = sol_omegas09_singleplot_fast_kink


B = 1.
######        BEGIN SHOOTING METHOD WITH KNOWN EIGENVALUES        ##########


#############
r0=0.  #mean
dr=0.9 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density


def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   

def rho_i(r):                  # Define the internal density profile
    return rho_A*profile(r)


B_twist = 0.
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


v_twist = 0.
def v_iphi(r):  
    return 0.  #v_twist*r   #0.

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy 
 
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

#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum[0]*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum[0]*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################

lx = np.linspace(3.*2.*np.pi/wavenum[0], 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain      

m_e = ((((wavenum[0]**2*vA_e**2)-frequency[0]**2)*((wavenum[0]**2*c_e**2)-frequency[0]**2))/((vA_e**2+c_e**2)*((wavenum[0]**2*cT_e**2)-frequency[0]**2)))

xi_e_const = -1/(rho_e*((wavenum[0]**2*vA_e**2)-frequency[0]**2))

         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency[0] - (m*v_phi/r) + wavenum[0]*v_z)
  
def alfven_freq(r):
  return ((m*B_phi/r)+(wavenum[0]*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))


shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")


def D(r):  
  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

D_np=sym.lambdify(rr,D(rr),"numpy")

def Q(r):
  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_phi**2/r) + (2*shift_freq(r)**2*B_phi**2/r)+(2*shift_freq(r)*B_phi*v_phi*((m*B_phi/r)+(wavenum[0]*B_i(r)))/r))

def T(r):
  return ((((m*B_phi/r)+(wavenum[0]*B_i(r)))*B_phi) + rho_i(r)*v_phi*shift_freq(r))

def C1(r):
  return ((Q(r)*shift_freq(r)) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))

C1_np=sym.lambdify(rr,C1(rr),"numpy")
   
def C2(r):   
  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[0]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
   
def C3_diff(r):
  return ((B_phi/r)**2 - (rho_i(r)*(v_phi/r)**2))


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
       return [P_e[1], (-P_e[1]/r_e + (m_e+m/(r_e**2))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)
left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution_09 = left_P_solution/np.amax(abs(left_P_solution))  #normalisation_factor_pressure
normalised_left_xi_solution_09 = left_xi_solution/np.amax(abs(left_xi_solution))   #normalisation_factor_xi
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
       return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]

    
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,1]  #U[:,1]  for sausage   U[:,0]  for kink    #was plus (assuming pressure is symmetric for sausage)
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]

inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side


normalised_inside_P_solution_09 = inside_P_solution/np.amax(abs(left_P_solution))  #normalisation_factor_pressure
normalised_inside_xi_solution_09 = inside_xi_solution/np.amax(abs(left_xi_solution))   #normalisation_factor_xi

inside_xi_phi_09 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_09 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency[0]**2 - wavenum[0]**2*vA_e**2)))
radial_xi_phi_09 = np.concatenate((inside_xi_phi_09, outside_xi_phi_09), axis=None)    

normalised_inside_xi_phi_09 = inside_xi_phi_09/np.amax(abs(outside_xi_phi_09))
normalised_outside_xi_phi_09 = outside_xi_phi_09/np.amax(abs(outside_xi_phi_09))
normalised_radial_xi_phi_09 = np.concatenate((normalised_inside_xi_phi_09, normalised_outside_xi_phi_09), axis=None)    


plt.figure()
plt.plot(lx[::-1], outside_xi_phi_09)
#plt.show() 

###################################################################

##plt.rcParams["text.usetex"] =True

#fig, (ax, ax2) = plt.subplots(2,1, sharex=False)    # use for sausage because phi component is zero 
fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False) 

ax.axvline(x=-B, color='r', linestyle='--')
ax.axvline(x=B, color='r', linestyle='--')
ax.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax.set_ylim(-9.,1.2)
ax.set_xlim(0.,2.5)
ax.plot(lx, normalised_left_P_solution, 'k')
ax.plot(ix, normalised_inside_P_solution, 'k')

ax.plot(lx, normalised_left_P_solution_3, color='goldenrod')
ax.plot(ix, normalised_inside_P_solution_3, color='goldenrod')

ax.plot(lx, normalised_left_P_solution_15, 'g')
ax.plot(ix, normalised_inside_P_solution_15, 'g')

#ax.plot(lx, normalised_left_P_solution_09, 'r')
#ax.plot(ix, normalised_inside_P_solution_09, 'r')



ax2.axvline(x=-B, color='r', linestyle='--')
ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)

ax2.set_ylim(-2.,6.)
ax2.set_xlim(0.,2.5)
ax2.plot(lx, normalised_left_xi_solution, 'k')
ax2.plot(ix, normalised_inside_xi_solution, 'k')

ax2.plot(lx, normalised_left_xi_solution_3, color='goldenrod')
ax2.plot(ix, normalised_inside_xi_solution_3, color='goldenrod')

ax2.plot(lx, normalised_left_xi_solution_15, 'g')
ax2.plot(ix, normalised_inside_xi_solution_15, 'g')

#ax2.plot(lx, normalised_left_xi_solution_09, 'r')
#ax2.plot(ix, normalised_inside_xi_solution_09, 'r')



ax3.axvline(x=-B, color='r', linestyle='--')
ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_xlabel("$r$", fontsize=18)
ax3.set_ylabel("$\hat{\u03be}_{\u03C6}$", fontsize=18, rotation=0, labelpad=15)

ax3.set_ylim(-1.2,10.)
ax3.set_xlim(0.,2.5)

ax3.plot(lx[::-1], normalised_outside_xi_phi_1e5, 'k')
ax3.plot(ix[::-1], normalised_inside_xi_phi_1e5, 'k')

ax3.plot(lx[::-1], normalised_outside_xi_phi_3, color='goldenrod')
ax3.plot(ix[::-1], normalised_inside_xi_phi_3, color='goldenrod')

ax3.plot(lx[::-1], normalised_outside_xi_phi_15, 'g')
ax3.plot(ix[::-1], normalised_inside_xi_phi_15, 'g')

#ax3.plot(lx[::-1], normalised_outside_xi_phi_09, 'r')
#ax3.plot(ix[::-1], normalised_inside_xi_phi_09, 'r')


plt.savefig("photospheric_density_slow_kink_eigenfunctions_k32.png")

plt.show()
exit()
