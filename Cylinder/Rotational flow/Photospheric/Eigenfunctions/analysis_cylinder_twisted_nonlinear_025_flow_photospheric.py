# Import the required modules
import numpy as np
import scipy as sc
import sympy as sym
#import matplotlib; matplotlib.use('agg') ##comment out to show figures
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pickle
import itertools
import numpy.polynomial.polynomial as poly
from matplotlib import animation

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


B_twist = 0.1
def B_iphi(r):  #constant rho
    return 0.  #B_twist*r   #0.


v_twist = 0.25

power = 1.5

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


#####################################################


###################################################

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

#####   SPEEDS AT BOUNDARY    ###

rho_bound = rho_i_np(1)
c_bound = c_i_np(1)
vA_bound = vA_i_np(1)
cT_bound = np.sqrt(c_bound**2 * vA_bound**2 / (c_bound**2 + vA_bound**2))

P_bound = P_i_np(1)
internal_pressure = P_i_np(ix)


print('cT  =', cT_bound,  cT_i_np(1), cT_i_np(0))
print('c_I  =', c_bound,  c_i_np(1), c_i_np(0))
print('v_A  =', vA_bound,  vA_i_np(1), vA_i_np(0))


P_TOT_i_bound = P_i_np(1) + (B_0**2)/2. - rho_i_np(1)*v_iphi_np(1)**2/2.
print(' mod P_tot inside at bound   =', P_TOT_i_bound)


#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################


plt.figure()
plt.title("width = 1.5")
plt.xlabel("x")
plt.ylabel("$\u03C1_{i}$",fontsize=25)
ax = plt.subplot(331)
ax.title.set_text("Density")
#ax.plot(ix,rho_i_np(ix));
ax.annotate( '$\u03C1_{e}$', xy=(1, rho_e),fontsize=25)
ax.annotate( '$\u03C1_{i}$', xy=(1, rho_i0),fontsize=25)
ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot')
ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot')
ax.axhline(y=rho_i_np(ix), color='b', label='$\u03C1_{i}$', linestyle='solid')
ax.set_ylim(0., 1.1)

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
#ax2.axhline(y=c_i_np(ix), color='b', linestyle='solid')
#ax2.set_ylim(-10., 1.)


#plt.figure()
#plt.xlabel("x")
#plt.ylabel("$vA_{i}$")
ax3 = plt.subplot(333)
ax3.title.set_text("$v_{Ai}$")
#ax3.plot(ix,vA_i_np(ix));
ax3.annotate( '$v_{Ae}$', xy=(1, vA_e),fontsize=25)
ax3.annotate( '$v_{Ai}$', xy=(1, vA_i0),fontsize=25)
ax3.annotate( '$v_{AB}$', xy=(1, vA_bound), fontsize=20)
ax3.axhline(y=vA_i0, color='k', label='$v_{Ai}$', linestyle='dashdot')
ax3.axhline(y=vA_e, color='k', label='$v_{Ae}$', linestyle='dashdot')
ax3.fill_between(ix, vA_i0, vA_bound, color='orange', alpha=0.2)    # fill between uniform case and boundary values
ax3.axhline(y=vA_i_np(ix), color='b', linestyle='solid')


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
ax5.plot(ix,P_i_np(ix));
ax5.annotate( '$P_{0}$', xy=(1, P_0),fontsize=25)
ax5.annotate( '$P_{e}$', xy=(1, P_e),fontsize=25)
ax5.axhline(y=P_0, color='k', label='$P_{i}$', linestyle='dashdot')
ax5.axhline(y=P_e, color='k', label='$P_{e}$', linestyle='dashdot')
#ax5.axhline(y=P_i_np(ix), color='b', label='$P_{e}$', linestyle='solid')

#plt.figure()
plt.xlabel("x")
plt.ylabel("$T$")
ax6 = plt.subplot(336)
ax6.title.set_text("Temperature")
ax6.plot(ix,T_i_np(ix));
ax6.annotate( '$T_{0}$', xy=(1, T_0),fontsize=25)
ax6.annotate( '$T_{e}$', xy=(1, T_e),fontsize=25)
ax6.axhline(y=T_0, color='k', label='$T_{i}$', linestyle='dashdot')
ax6.axhline(y=T_e, color='k', label='$T_{e}$', linestyle='dashdot')
#ax6.axhline(y=T_i_np(ix), color='b', linestyle='solid')
ax6.set_ylim(0., 1.1)


#ax7 = plt.subplot(337)
#ax7.title.set_text("Mag Field z")
##ax7.plot(ix,B_i_np(ix));
#ax7.annotate( '$B_{0}$', xy=(1, B_0),fontsize=25)
#ax7.annotate( '$B_{e}$', xy=(1, B_e),fontsize=25)
#ax7.axhline(y=B_0, color='k', label='$T_{i}$', linestyle='dashdot')
#ax7.axhline(y=B_e, color='k', label='$T_{e}$', linestyle='dashdot')
#ax7.axhline(y=B_i_np(ix), color='b', linestyle='solid')


ax7 = plt.subplot(337)
ax7.title.set_text("v_phi")
ax7.plot(ix,v_iphi_np(ix));


#
##plt.figure()
##plt.xlabel("x")
##plt.ylabel("$B$")
#ax7 = plt.subplot(337)
#ax7.title.set_text("Mag Field phi")
##ax7.plot(ix,B_i_np(ix));
#ax7.plot(ix,B_iphi_np(ix));
#ax7.annotate( '$B_{0}$', xy=(1, B_0),fontsize=25)
#ax7.annotate( '$B_{e}$', xy=(1, B_e),fontsize=25)
#ax7.axhline(y=B_0, color='k', label='$B_{i}$', linestyle='dashdot')
#ax7.axhline(y=B_e, color='k', label='$B_{e}$', linestyle='dashdot')
##ax7.axhline(y=B_iz_np(ix), color='b', label='$B_{e}$', linestyle='solid')


plt.xlabel("x")
plt.ylabel("$P_T$")
ax8 = plt.subplot(338)
ax8.title.set_text("Tot Pressure (uniform)")
ax8.annotate( '$P_{T0}$', xy=(1, P_tot_0),fontsize=25,color='b')
ax8.annotate( '$P_{Te}$', xy=(1, P_tot_e),fontsize=25,color='r')
ax8.axhline(y=P_tot_0, color='b', label='$P_{Ti0}$', linestyle='dashdot')
ax8.axhline(y=P_tot_e, color='r', label='$P_{Te}$', linestyle='dashdot')
#ax8.plot(ix,PT_i_np(ix));

ax8.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')

#ax8.axhline(y=B_iz_np(ix), color='b', label='$P_{Te}$', linestyle='solid')

ax9 = plt.subplot(339)
ax9.title.set_text("Tot Pressure  (Modified)")
ax9.annotate( '$P_{T0}$', xy=(1, P_TOT_i_bound),fontsize=25,color='b')
#ax9.plot(ix,PT_i_np(ix));
ax9.axhline(y=P_tot_e, color='r', label='$P_{Te}$', linestyle='dashdot')
ax9.annotate( '$P_{Te}$', xy=(1, P_tot_e),fontsize=25,color='r')

ax9.axhline(y=PT_i_np(ix), color='k', label='$P_{Ti}$', linestyle='solid')
#ax9.set_ylim(-1., 1.)



###############    Non-linear velocity twist profiles   ########

v_twist = 0.05

v_twist_neg = 0.01



power_07 = 0.7
def v_iphi_p07(r):  
    return v_twist*(r**0.7) 
v_iphi_p07_np=sym.lambdify(rr,v_iphi_p07(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_08 = 0.8
def v_iphi_p08(r):  
    return v_twist*(r**0.8) 
v_iphi_p08_np=sym.lambdify(rr,v_iphi_p08(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_085 = 0.85
def v_iphi_p085(r):  
    return v_twist*(r**0.85) 
v_iphi_p085_np=sym.lambdify(rr,v_iphi_p085(rr),"numpy")   #In order to evaluate we need to switch to numpy



power_09 = 0.9
def v_iphi_p09(r):  
    return v_twist*(r**0.9) 
v_iphi_p09_np=sym.lambdify(rr,v_iphi_p09(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_095 = 0.95
def v_iphi_p095(r):  
    return v_twist*(r**0.95) 
v_iphi_p095_np=sym.lambdify(rr,v_iphi_p095(rr),"numpy")   #In order to evaluate we need to switch to numpy




power_1 = 1.
def v_iphi_p1(r):  
    return v_twist*(r**1.) 
v_iphi_p1_np=sym.lambdify(rr,v_iphi_p1(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_15 = 1.5
def v_iphi_p15(r):  
    return v_twist*(r**1.5) 
v_iphi_p15_np=sym.lambdify(rr,v_iphi_p15(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_2 = 2.
def v_iphi_p2(r):  
    return v_twist*(r**2.) 
v_iphi_p2_np=sym.lambdify(rr,v_iphi_p2(rr),"numpy")   #In order to evaluate we need to switch to numpy


power_3 = 3.
def v_iphi_p3(r):  
    return v_twist*(r**3.) 
v_iphi_p3_np=sym.lambdify(rr,v_iphi_p3(rr),"numpy")   #In order to evaluate we need to switch to numpy
 



plt.figure()
#plt.title("$v_{0\varphi}$ profiles")
plt.xlabel("r")
plt.ylabel(r"$v_{0\varphi}$",fontsize=25)
ax = plt.subplot(111)
#ax.set_ylim(0., 1.)  # use for negative powers
ax.set_ylim(0., 0.05)
#ax.set_ylim(0., 0.1)

#ax.plot(ix, v_iphi_p07_np(ix), color='red', label=r'$v_{\varphi} \propto r^{0.7}$')
ax.plot(ix, v_iphi_p08_np(ix), color='deepskyblue', label=r'$v_{\varphi} \propto r^{0.8}$')
#ax.plot(ix, v_iphi_p085_np(ix), color='darkorange')
#ax.plot(ix, v_iphi_p09_np(ix), color='goldenrod', label=r'$v_{\varphi} \propto r^{0.9}$')
#ax.plot(ix, v_iphi_p095_np(ix), 'g')
ax.plot(ix, v_iphi_p1_np(ix), 'k', label=r'$v_{\varphi} \propto r^{1.0}$')
ax.plot(ix, v_iphi_p15_np(ix), color='red', linestyle='dotted', label=r'$v_{\varphi} \propto r^{1.5}$')
#ax.plot(ix, v_iphi_p2_np(ix), color='green', linestyle='dotted', label=r'$v_{\varphi} \propto r^{2.0}$')

leg = ax.legend(loc='lower right')



####   normalised v_phi  ########

plt.figure()
#plt.title("$v_{0\varphi}$ profiles")
plt.xlabel("r")
#plt.ylabel(r"$v_{0\varphi}$",fontsize=25)
ax = plt.subplot(111)
#ax.set_ylim(0., 1.)  # use for negative powers
ax.set_ylim(0., 1.)
#ax.set_ylim(0., 0.1)

#ax.plot(ix, v_iphi_p05_np(ix)/v_twist, color='magenta')
#ax.plot(ix, v_iphi_p07_np(ix)/v_twist, color='red', label=r'$v_{\varphi} \propto r^{0.7}$')
ax.plot(ix, v_iphi_p08_np(ix)/v_twist, color='deepskyblue', label=r'$v_{\varphi} \propto r^{0.8}$')
#ax.plot(ix, v_iphi_p085_np(ix), color='darkorange')
#ax.plot(ix, v_iphi_p09_np(ix)/v_twist, color='goldenrod', label=r'$v_{\varphi} \propto r^{0.9}$')
#ax.plot(ix, v_iphi_p095_np(ix), 'g')
ax.plot(ix, v_iphi_p1_np(ix)/v_twist, 'k', label=r'$v_{\varphi} \propto r^{1.0}$')
ax.plot(ix, v_iphi_p15_np(ix)/v_twist, color='red', linestyle='dotted', label=r'$v_{\varphi} \propto r^{1.5}$')
#ax.plot(ix, v_iphi_p2_np(ix)/v_twist, color='green', linestyle='dotted', label=r'$v_{\varphi} \propto r^{2.0}$')

leg = ax.legend(loc='lower right')

#plt.show()
#exit()





########   READ IN VARIABLES    power = 0.8   #########

#################################################


with open('Cylindrical_photospheric_vtwist025_power08_slow_kink.pickle', 'rb') as f:
    sol_omegas_kinkp08_pos_slow, sol_ks_kinkp08_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power08_fund_kink.pickle', 'rb') as f:
    sol_omegas_kinkp08_pos_fast, sol_ks_kinkp08_pos_fast = pickle.load(f)


sol_omegas_kinkp08 = np.concatenate((sol_omegas_kinkp08_pos_slow, sol_omegas_kinkp08_pos_fast), axis=None)  
sol_ks_kinkp08 = np.concatenate((sol_ks_kinkp08_pos_slow, sol_ks_kinkp08_pos_fast), axis=None)



## SORT ARRAYS IN ORDER OF WAVENUMBER ###

sol_omegas_kinkp08 = [x for _,x in sorted(zip(sol_ks_kinkp08,sol_omegas_kinkp08))]
sol_ks_kinkp08 = np.sort(sol_ks_kinkp08)

sol_omegas_kinkp08 = np.array(sol_omegas_kinkp08)
sol_ks_kinkp08 = np.array(sol_ks_kinkp08)



####################################################################




########   READ IN VARIABLES    power = 1   #########

#################################################


#with open('Cylindrical_photospheric_vtwist005_pos_fast_sausage.pickle', 'rb') as f:
#    sol_omegasp1_pos_fast, sol_ksp1_pos_fast = pickle.load(f)
#
#with open('Cylindrical_photospheric_vtwist005_pos_slow_sausage.pickle', 'rb') as f:
#    sol_omegasp1_pos_slow, sol_ksp1_pos_slow = pickle.load(f)



with open('Cylindrical_photospheric_vtwist025_power1_slow_kink.pickle', 'rb') as f:
    sol_omegas_kinkp1_pos_slow, sol_ks_kinkp1_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power1_fund_kink.pickle', 'rb') as f:
    sol_omegas_kinkp1_pos_fast, sol_ks_kinkp1_pos_fast = pickle.load(f)


#sol_omegasp1 = np.concatenate((sol_omegasp1_pos_slow, sol_omegasp1_pos_fast), axis=None)  
#sol_ksp1 = np.concatenate((sol_ksp1_pos_slow, sol_ksp1_pos_fast), axis=None)  

sol_omegas_kinkp1 = np.concatenate((sol_omegas_kinkp1_pos_slow, sol_omegas_kinkp1_pos_fast), axis=None)  
sol_ks_kinkp1 = np.concatenate((sol_ks_kinkp1_pos_slow, sol_ks_kinkp1_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
#sol_omegasp1 = [x for _,x in sorted(zip(sol_ksp1,sol_omegasp1))]
#sol_ksp1 = np.sort(sol_ksp1)

#sol_omegasp1 = np.array(sol_omegasp1)
#sol_ksp1 = np.array(sol_ksp1)

sol_omegas_kinkp1 = [x for _,x in sorted(zip(sol_ks_kinkp1,sol_omegas_kinkp1))]
sol_ks_kinkp1 = np.sort(sol_ks_kinkp1)

sol_omegas_kinkp1 = np.array(sol_omegas_kinkp1)
sol_ks_kinkp1 = np.array(sol_ks_kinkp1)





########   READ IN VARIABLES    power = 1.25   #########

#################################################

#with open('Cylindrical_photospheric_vtwist005_power125_sausage_fast.pickle', 'rb') as f:
#    sol_omegasp125_pos_fast, sol_ksp125_pos_fast = pickle.load(f)    # need to rename variables
#
#with open('Cylindrical_photospheric_vtwist005_power125_sausage_slow.pickle', 'rb') as f:
#    sol_omegasp125_pos_slow, sol_ksp125_pos_slow = pickle.load(f)



with open('Cylindrical_photospheric_vtwist025_power125_slow_kink.pickle', 'rb') as f:
    sol_omegas_kinkp125_pos_slow, sol_ks_kinkp125_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power125_fund_kink.pickle', 'rb') as f:
    sol_omegas_kinkp125_pos_fast, sol_ks_kinkp125_pos_fast = pickle.load(f)



#sol_omegasp125 = np.concatenate((sol_omegasp125_pos_fast, sol_omegasp125_pos_slow), axis=None)  
#sol_ksp125 = np.concatenate((sol_ksp125_pos_fast, sol_ksp125_pos_slow), axis=None)  

sol_omegas_kinkp125 = np.concatenate((sol_omegas_kinkp125_pos_fast, sol_omegas_kinkp125_pos_slow), axis=None)  
sol_ks_kinkp125 = np.concatenate((sol_ks_kinkp125_pos_fast, sol_ks_kinkp125_pos_slow), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
#sol_omegasp125 = [x for _,x in sorted(zip(sol_ksp125,sol_omegasp125))]
#sol_ksp125 = np.sort(sol_ksp125)

#sol_omegasp125 = np.array(sol_omegasp125)
#sol_ksp125 = np.array(sol_ksp125)

sol_omegas_kinkp125 = [x for _,x in sorted(zip(sol_ks_kinkp125,sol_omegas_kinkp125))]
sol_ks_kinkp125 = np.sort(sol_ks_kinkp125)

sol_omegas_kinkp125 = np.array(sol_omegas_kinkp125)
sol_ks_kinkp125 = np.array(sol_ks_kinkp125)






########   READ IN VARIABLES    power = 1.5   #########

#################################################

#with open('Cylindrical_photospheric_vtwist005_power15_sausage_fast.pickle', 'rb') as f:
#    sol_omegasp15_pos_fast, sol_ksp15_pos_fast = pickle.load(f)    # need to rename variables
#
#with open('Cylindrical_photospheric_vtwist005_power15_sausage_slow.pickle', 'rb') as f:
#    sol_omegasp15_pos_slow, sol_ksp15_pos_slow = pickle.load(f)



with open('Cylindrical_photospheric_vtwist025_power15_slow_kink.pickle', 'rb') as f:
    sol_omegas_kinkp15_pos_slow, sol_ks_kinkp15_pos_slow = pickle.load(f)

with open('Cylindrical_photospheric_vtwist025_power15_fund_kink.pickle', 'rb') as f:
    sol_omegas_kinkp15_pos_fast, sol_ks_kinkp15_pos_fast = pickle.load(f)



#sol_omegasp15 = np.concatenate((sol_omegasp15_pos_fast, sol_omegasp15_pos_slow), axis=None)  
#sol_ksp15 = np.concatenate((sol_ksp15_pos_fast, sol_ksp15_pos_slow), axis=None)  

sol_omegas_kinkp15 = np.concatenate((sol_omegas_kinkp15_pos_slow, sol_omegas_kinkp15_pos_fast), axis=None)  
sol_ks_kinkp15 = np.concatenate((sol_ks_kinkp15_pos_slow, sol_ks_kinkp15_pos_fast), axis=None)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
#sol_omegasp15 = [x for _,x in sorted(zip(sol_ksp15,sol_omegasp15))]
#sol_ksp15 = np.sort(sol_ksp15)

#sol_omegasp15 = np.array(sol_omegasp15)
#sol_ksp15 = np.array(sol_ksp15)

sol_omegas_kinkp15 = [x for _,x in sorted(zip(sol_ks_kinkp15,sol_omegas_kinkp15))]
sol_ks_kinkp15 = np.sort(sol_ks_kinkp15)

sol_omegas_kinkp15 = np.array(sol_omegas_kinkp15)
sol_ks_kinkp15 = np.array(sol_ks_kinkp15)



###############################################################


########################################################################################################

plt.figure()
#plt.title("$ v_{\varphi} = 0.01$")
ax = plt.subplot(111)
plt.xlabel("$kr$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks_kinkp08, (sol_omegas_kinkp08/sol_ks_kinkp08), marker='.', linestyle='', color='deepskyblue', markersize=4.)

#ax.plot(sol_ks_kinkp095, (sol_omegas_kinkp095/sol_ks_kinkp095), marker='.', linestyle='', color='green', markersize=4.)

ax.plot(sol_ks_kinkp1, (sol_omegas_kinkp1/sol_ks_kinkp1), marker='.', linestyle='', color='black', markersize=4.)

#ax.plot(sol_ksp1, (sol_omegasp1/sol_ksp1), 'k.', markersize=4.)
#ax.plot(sol_ks_kinkp1, (sol_omegas_kinkp1/sol_ks_kinkp1), 'k.', markersize=4.)


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


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.92, box.height])

ax.set_ylim(0.8, c_e+0.05)   # whole

#ax.set_ylim(0.85, 1.05)  #slow body

#ax.set_ylim(1.9, 5)
#ax.set_ylim(-1.05, -0.8)  #neg slow body


#plt.grid(color='r', linestyle='-', linewidth=2)


#plt.grid(True)
#plt.savefig("cylinder_coronal_vphi_001_neg_slowbody.png")

#plt.show()
#exit()



#######################################################################
#######################################################################


plt.figure(figsize=(14,18))
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax.set_title(r'$\propto r^{0.8}$')
ax2.set_title(r'$\propto r^{1.}$')
ax3.set_title(r'$\propto r^{1.25}$')
ax4.set_title(r'$\propto r^{1.5}$')

#ax.set_xlabel("$kr$", fontsize=16)
ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
#ax2.set_xlabel("$kr$", fontsize=16)
ax2.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax3.set_xlabel("$kr$", fontsize=16)
ax3.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax4.set_xlabel("$kr$", fontsize=16)
ax4.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.set_xlim(0., 4.)  
ax.set_ylim(0.85, c_e+0.01)  
ax2.set_ylim(0.85, c_e+0.01)  
ax3.set_ylim(0.85, c_e+0.01)  
ax4.set_ylim(0.85, c_e+0.01)  



#ax.plot(sol_ksp08, (sol_omegasp08/sol_ksp08), 'r.', markersize=4.)
ax.plot(sol_ks_kinkp08, (sol_omegas_kinkp08/sol_ks_kinkp08), 'b.', markersize=4.)

#ax2.plot(sol_ksp1, (sol_omegasp1/sol_ksp1), 'r.', markersize=4.)
ax2.plot(sol_ks_kinkp1, (sol_omegas_kinkp1/sol_ks_kinkp1), 'b.', markersize=4.)

#ax3.plot(sol_ksp125, (sol_omegasp125/sol_ksp125), 'r.', markersize=4.)
ax3.plot(sol_ks_kinkp125, (sol_omegas_kinkp125/sol_ks_kinkp125), 'b.', markersize=4.)

#ax4.plot(sol_ksp15, (sol_omegasp15/sol_ksp15), 'r.', markersize=4.)
ax4.plot(sol_ks_kinkp15, (sol_omegas_kinkp15/sol_ks_kinkp15), 'b.', markersize=4.)


ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax2.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax2.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax2.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax3.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax3.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax3.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax4.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax4.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax4.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)

ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')


#plt.show()
#exit()

########################################

sol_ksp08_singleplot_fast_kink = []
sol_omegasp08_singleplot_fast_kink = []
sol_ksp08_singleplot_slow_surf_kink = []
sol_omegasp08_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kinkp08)):
    if sol_ks_kinkp08[i] > 0.5 and sol_ks_kinkp08[i] < 0.55:        
       if sol_omegas_kinkp08[i]/sol_ks_kinkp08[i] > 1.35 and sol_omegas_kinkp08[i]/sol_ks_kinkp08[i] < c_e:
         sol_ksp08_singleplot_fast_kink.append(sol_ks_kinkp08[i])
         sol_omegasp08_singleplot_fast_kink.append(sol_omegas_kinkp08[i])        

    if sol_ks_kinkp08[i] > 0.45 and sol_ks_kinkp08[i] < 0.55:         
       if sol_omegas_kinkp08[i]/sol_ks_kinkp08[i] > c_kink and sol_omegas_kinkp08[i]/sol_ks_kinkp08[i] < 1.31: 
         sol_ksp08_singleplot_slow_surf_kink.append(sol_ks_kinkp08[i])
         sol_omegasp08_singleplot_slow_surf_kink.append(sol_omegas_kinkp08[i])

    
sol_ksp08_singleplot_fast_kink = np.array(sol_ksp08_singleplot_fast_kink)
sol_omegasp08_singleplot_fast_kink = np.array(sol_omegasp08_singleplot_fast_kink)
sol_ksp08_singleplot_slow_surf_kink = np.array(sol_ksp08_singleplot_slow_surf_kink)
sol_omegasp08_singleplot_slow_surf_kink = np.array(sol_omegasp08_singleplot_slow_surf_kink)


######################################


sol_ksp1_singleplot_fast_kink = []
sol_omegasp1_singleplot_fast_kink = []
sol_ksp1_singleplot_slow_kink = []
sol_omegasp1_singleplot_slow_kink = []
sol_ksp1_singleplot_slow_surf_kink = []
sol_omegasp1_singleplot_slow_surf_kink = []


for i in range(len(sol_ks_kinkp1)):
    if sol_ks_kinkp1[i] > 0.5 and sol_ks_kinkp1[i] < 0.55:        
       if sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] > 1.35 and sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] < c_e:
         sol_ksp1_singleplot_fast_kink.append(sol_ks_kinkp1[i])
         sol_omegasp1_singleplot_fast_kink.append(sol_omegas_kinkp1[i])        

    if sol_ks_kinkp1[i] > 0.5 and sol_ks_kinkp1[i] < 0.55:         
       if sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] > 1.2 and sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] < 1.35: 
         sol_ksp1_singleplot_slow_surf_kink.append(sol_ks_kinkp1[i])
         sol_omegasp1_singleplot_slow_surf_kink.append(sol_omegas_kinkp1[i])

    if sol_ks_kinkp1[i] > 2.3 and sol_ks_kinkp1[i] < 2.4:         
       if sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] > c_i0 and sol_omegas_kinkp1[i]/sol_ks_kinkp1[i] < 1.1: 
         sol_ksp1_singleplot_slow_kink.append(sol_ks_kinkp1[i])
         sol_omegasp1_singleplot_slow_kink.append(sol_omegas_kinkp1[i])

    
sol_ksp1_singleplot_fast_kink = np.array(sol_ksp1_singleplot_fast_kink)
sol_omegasp1_singleplot_fast_kink = np.array(sol_omegasp1_singleplot_fast_kink)
sol_ksp1_singleplot_slow_kink = np.array(sol_ksp1_singleplot_slow_kink)
sol_omegasp1_singleplot_slow_kink = np.array(sol_omegasp1_singleplot_slow_kink)
sol_ksp1_singleplot_slow_surf_kink = np.array(sol_ksp1_singleplot_slow_surf_kink)
sol_omegasp1_singleplot_slow_surf_kink = np.array(sol_omegasp1_singleplot_slow_surf_kink)



##############################

sol_ksp125_singleplot_fast_kink = []
sol_omegasp125_singleplot_fast_kink = []
sol_ksp125_singleplot_slow_kink = []
sol_omegasp125_singleplot_slow_kink = []


for i in range(len(sol_ks_kinkp125)):
    if sol_ks_kinkp125[i] > 0.5 and sol_ks_kinkp125[i] < 0.55:        
       if sol_omegas_kinkp125[i]/sol_ks_kinkp125[i] > 1.35 and sol_omegas_kinkp125[i]/sol_ks_kinkp125[i] < c_e:
         sol_ksp125_singleplot_fast_kink.append(sol_ks_kinkp125[i])
         sol_omegasp125_singleplot_fast_kink.append(sol_omegas_kinkp125[i])        

    if sol_ks_kinkp125[i] > 1.75 and sol_ks_kinkp125[i] < 1.79:         
       if sol_omegas_kinkp125[i]/sol_ks_kinkp125[i] > 0.94 and sol_omegas_kinkp125[i]/sol_ks_kinkp125[i] < 0.95: 
         sol_ksp125_singleplot_slow_kink.append(sol_ks_kinkp125[i])
         sol_omegasp125_singleplot_slow_kink.append(sol_omegas_kinkp125[i])

  
     
sol_ksp125_singleplot_fast_kink = np.array(sol_ksp125_singleplot_fast_kink)
sol_omegasp125_singleplot_fast_kink = np.array(sol_omegasp125_singleplot_fast_kink)
sol_ksp125_singleplot_slow_kink = np.array(sol_ksp125_singleplot_slow_kink)
sol_omegasp125_singleplot_slow_kink = np.array(sol_omegasp125_singleplot_slow_kink)



##############################

sol_ksp15_singleplot_fast_kink = []
sol_omegasp15_singleplot_fast_kink = []
sol_ksp15_singleplot_slow_kink = []
sol_omegasp15_singleplot_slow_kink = []


for i in range(len(sol_ks_kinkp15)):
    if sol_ks_kinkp15[i] > 0.5 and sol_ks_kinkp15[i] < 0.55:        
       if sol_omegas_kinkp15[i]/sol_ks_kinkp15[i] > 1.35 and sol_omegas_kinkp15[i]/sol_ks_kinkp15[i] < c_e:
         sol_ksp15_singleplot_fast_kink.append(sol_ks_kinkp15[i])
         sol_omegasp15_singleplot_fast_kink.append(sol_omegas_kinkp15[i])        

    if sol_ks_kinkp15[i] > 1.75 and sol_ks_kinkp15[i] < 1.79:         
       if sol_omegas_kinkp15[i]/sol_ks_kinkp15[i] > 0.94 and sol_omegas_kinkp15[i]/sol_ks_kinkp15[i] < 0.95: 
         sol_ksp15_singleplot_slow_kink.append(sol_ks_kinkp15[i])
         sol_omegasp15_singleplot_slow_kink.append(sol_omegas_kinkp15[i])

     
sol_ksp15_singleplot_fast_kink = np.array(sol_ksp15_singleplot_fast_kink)
sol_omegasp15_singleplot_fast_kink = np.array(sol_omegasp15_singleplot_fast_kink)
sol_ksp15_singleplot_slow_kink = np.array(sol_ksp15_singleplot_slow_kink)
sol_omegasp15_singleplot_slow_kink = np.array(sol_omegasp15_singleplot_slow_kink)



###################################################################

plt.figure(figsize=(14,18))
ax = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)


ax.set_title(r'$\propto r^{0.8}$')
ax2.set_title(r'$\propto r^{1.}$')
ax3.set_title(r'$\propto r^{1.25}$')
ax4.set_title(r'$\propto r^{1.5}$')

#ax.set_xlabel("$kr$", fontsize=16)
ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
#ax2.set_xlabel("$kr$", fontsize=16)
ax2.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax3.set_xlabel("$kr$", fontsize=16)
ax3.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)
ax4.set_xlabel("$kr$", fontsize=16)
ax4.set_ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.set_xlim(0., 4.)  
ax.set_ylim(0.85, c_e+0.01)  
ax2.set_ylim(0.85, c_e+0.01)  
ax3.set_ylim(0.85, c_e+0.01)  
ax4.set_ylim(0.85, c_e+0.01)  



#ax.plot(sol_ksp08, (sol_omegasp08/sol_ksp08), 'r.', markersize=4.)
ax.plot(sol_ks_kinkp08, (sol_omegas_kinkp08/sol_ks_kinkp08), 'b.', markersize=4.)

#ax2.plot(sol_ksp1, (sol_omegasp1/sol_ksp1), 'r.', markersize=4.)
ax2.plot(sol_ks_kinkp1, (sol_omegas_kinkp1/sol_ks_kinkp1), 'b.', markersize=4.)

#ax3.plot(sol_ksp125, (sol_omegasp125/sol_ksp125), 'r.', markersize=4.)
ax3.plot(sol_ks_kinkp125, (sol_omegas_kinkp125/sol_ks_kinkp125), 'b.', markersize=4.)

#ax4.plot(sol_ksp15, (sol_omegasp15/sol_ksp15), 'r.', markersize=4.)
ax4.plot(sol_ks_kinkp15, (sol_omegas_kinkp15/sol_ks_kinkp15), 'b.', markersize=4.)


ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax2.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax2.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax2.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax3.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax3.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax3.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax4.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax4.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax4.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)

ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax3.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax4.axhline(y=c_kink, color='k', linestyle='dashdot', label='_nolegend_')



ax.plot(sol_ksp08_singleplot_fast_kink, (sol_omegasp08_singleplot_fast_kink/sol_ksp08_singleplot_fast_kink), 'b.', markersize=15.)
ax.plot(sol_ksp08_singleplot_slow_surf_kink, (sol_omegasp08_singleplot_slow_surf_kink/sol_ksp08_singleplot_slow_surf_kink), 'b.', markersize=15.)


ax2.plot(sol_ksp1_singleplot_fast_kink, (sol_omegasp1_singleplot_fast_kink/sol_ksp1_singleplot_fast_kink), 'b.', markersize=15.)
ax2.plot(sol_ksp1_singleplot_slow_surf_kink, (sol_omegasp1_singleplot_slow_surf_kink/sol_ksp1_singleplot_slow_surf_kink), 'b.', markersize=15.)
ax2.plot(sol_ksp1_singleplot_slow_kink, (sol_omegasp1_singleplot_slow_kink/sol_ksp1_singleplot_slow_kink), 'b.', markersize=15.)


ax3.plot(sol_ksp125_singleplot_fast_kink, (sol_omegasp125_singleplot_fast_kink/sol_ksp125_singleplot_fast_kink), 'b.', markersize=15.)
ax3.plot(sol_ksp125_singleplot_slow_kink, (sol_omegasp125_singleplot_slow_kink/sol_ksp125_singleplot_slow_kink), 'b.', markersize=15.)


ax4.plot(sol_ksp15_singleplot_fast_kink, (sol_omegasp15_singleplot_fast_kink/sol_ksp15_singleplot_fast_kink), 'b.', markersize=15.)
#ax4.plot(sol_ksp15_singleplot_slow_kink, (sol_omegasp15_singleplot_slow_kink/sol_ksp15_singleplot_slow_kink), 'b.', markersize=15.)


#plt.show()
#exit()



##################     KINK     ###############################





####################################################
#wavenum = sol_ksp08_singleplot_fast_kink[0]
#frequency = sol_omegasp08_singleplot_fast_kink[0]
 
wavenum = sol_ksp08_singleplot_slow_surf_kink[0]
frequency = sol_omegasp08_singleplot_slow_surf_kink[0]

k = wavenum
w = frequency
#####################################################

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

m = 1.

B_twist = 0.01
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


v_twist = 0.25
def v_iphi(r):  
    return v_twist*(r**0.8) 

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain   #1.75 before
  
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

spatial_p08 = np.concatenate((ix[::-1], lx[::-1]), axis=None)     

radial_displacement_p08 = np.concatenate((inside_xi_solution[::-1], left_xi_solution[::-1]), axis=None)  
radial_PT_p08 = np.concatenate((inside_P_solution[::-1], left_P_solution[::-1]), axis=None)  

normalised_radial_displacement_p08 = np.concatenate((normalised_inside_xi_solution[::-1], normalised_left_xi_solution[::-1]), axis=None)  
normalised_radial_PT_p08 = np.concatenate((normalised_inside_P_solution[::-1], normalised_left_P_solution[::-1]), axis=None)  


######  v_r   #######

outside_v_r = -frequency*left_xi_solution[::-1]
inside_v_r = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]

radial_vr_p08 = np.concatenate((inside_v_r, outside_v_r), axis=None)    

normalised_outside_v_r = -w*normalised_left_xi_solution[::-1]
normalised_inside_v_r = -shift_freq_np(ix[::-1])*normalised_inside_xi_solution[::-1]

normalised_radial_vr_p08 = np.concatenate((normalised_inside_v_r, normalised_outside_v_r), axis=None)    


#####

inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-(k**2*cT_e**2))*(c_e**2+vA_e**2))
radial_xi_z_p08 = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 



inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - (k**2*vA_e**2)))
radial_xi_phi_p08 = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    

normalised_inside_xi_phi = inside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_outside_xi_phi = outside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_radial_xi_phi_p08 = np.concatenate((normalised_inside_xi_phi, normalised_outside_xi_phi), axis=None)    

######

######  v_phi   #######

def dv_phi(r):
  return sym.diff(v_iphi(r)/r)

dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")

outside_v_phi = -w*outside_xi_phi
inside_v_phi = -(shift_freq_np(ix[::-1])*inside_xi_phi) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1])

radial_v_phi_p08 = np.concatenate((inside_v_phi, outside_v_phi), axis=None)    

normalised_inside_v_phi = inside_v_phi/np.amax(abs(outside_v_phi))
normalised_outside_v_phi = outside_v_phi/np.amax(abs(outside_v_phi))
normalised_radial_v_phi_p08 = np.concatenate((normalised_inside_v_phi, normalised_outside_v_phi), axis=None)    


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

radial_v_z_p08 = np.concatenate((inside_v_z, outside_v_z), axis=None)    

######################





####################################################
#wavenum = sol_ksp1_singleplot_fast_kink[0]
#frequency = sol_omegasp1_singleplot_fast_kink[0]
 
#wavenum = sol_ksp1_singleplot_slow_kink[0]
#frequency = sol_omegasp1_singleplot_slow_kink[0]

wavenum = sol_ksp1_singleplot_slow_surf_kink[0]
frequency = sol_omegasp1_singleplot_slow_surf_kink[0]

k = wavenum
w = frequency
#####################################################

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

m = 1.

B_twist = 0.01
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


v_twist = 0.25
def v_iphi(r):  
    return v_twist*(r**1.) 

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain   #1.75 before
  
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

spatial_p1 = np.concatenate((ix[::-1], lx[::-1]), axis=None)     

radial_displacement_p1 = np.concatenate((inside_xi_solution[::-1], left_xi_solution[::-1]), axis=None)  
radial_PT_p1 = np.concatenate((inside_P_solution[::-1], left_P_solution[::-1]), axis=None)  

normalised_radial_displacement_p1 = np.concatenate((normalised_inside_xi_solution[::-1], normalised_left_xi_solution[::-1]), axis=None)  
normalised_radial_PT_p1 = np.concatenate((normalised_inside_P_solution[::-1], normalised_left_P_solution[::-1]), axis=None)  


######  v_r   #######

outside_v_r = -frequency*left_xi_solution[::-1]
inside_v_r = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]

radial_vr_p1 = np.concatenate((inside_v_r, outside_v_r), axis=None)    

normalised_outside_v_r = -w*normalised_left_xi_solution[::-1]
normalised_inside_v_r = -shift_freq_np(ix[::-1])*normalised_inside_xi_solution[::-1]

normalised_radial_vr_p1 = np.concatenate((normalised_inside_v_r, normalised_outside_v_r), axis=None)    


#####

inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-(k**2*cT_e**2))*(c_e**2+vA_e**2))
radial_xi_z_p1 = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 



inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - (k**2*vA_e**2)))
radial_xi_phi_p1 = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    

normalised_inside_xi_phi = inside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_outside_xi_phi = outside_xi_phi/np.amax(abs(outside_xi_phi))
normalised_radial_xi_phi_p1 = np.concatenate((normalised_inside_xi_phi, normalised_outside_xi_phi), axis=None)    

######

######  v_phi   #######

def dv_phi(r):
  return sym.diff(v_iphi(r)/r)

dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")

outside_v_phi = -w*outside_xi_phi
inside_v_phi = -(shift_freq_np(ix[::-1])*inside_xi_phi) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1])

radial_v_phi_p1 = np.concatenate((inside_v_phi, outside_v_phi), axis=None)    

normalised_inside_v_phi = inside_v_phi/np.amax(abs(outside_v_phi))
normalised_outside_v_phi = outside_v_phi/np.amax(abs(outside_v_phi))
normalised_radial_v_phi_p1 = np.concatenate((normalised_inside_v_phi, normalised_outside_v_phi), axis=None)    


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

radial_v_z_p1 = np.concatenate((inside_v_z, outside_v_z), axis=None)    

######################


#
#####################################################
#wavenum = sol_ksp15_singleplot_fast_kink[0]
#frequency = sol_omegasp15_singleplot_fast_kink[0]
# 
##wavenum = sol_ksp15_singleplot_slow_kink[0]
##frequency = sol_omegasp15_singleplot_slow_kink[0]
#
#k = wavenum
#w = frequency
######################################################
#
#rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols
#
#m = 1.
#
#B_twist = 0.01
#def B_iphi(r):  
#    return 0.   #B_twist*r   #0.
#
#B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy
#
#def B_i(r):  
#    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))
#
#B_i_np=sym.lambdify(rr,B_i(rr),"numpy")
#
#
#def f_B(r): 
#  return (m*B_iphi(r)/r + wavenum*B_i(r))
#
#f_B_np=sym.lambdify(rr,f_B(rr),"numpy")
#
#def g_B(r): 
#  return (m*B_i(r)/r + wavenum*B_iphi(r))
#
#g_B_np=sym.lambdify(rr,g_B(rr),"numpy")
#
#
#v_twist = 0.05
#def v_iphi(r):  
#    return v_twist*(r**1.5) 
#
#v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy
#
#
#lx = np.linspace(6.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain   #1.75 before
#  
#m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
#   
#xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))
#
#         ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
#
#def shift_freq(r):
#  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z)
#  
#def alfven_freq(r):
#  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))
#
#def cusp_freq(r):
#  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))
#
#shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
#alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
#cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")
#
#def D(r):  
#  return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))
#
#D_np=sym.lambdify(rr,D(rr),"numpy")
#
#def Q(r):
#  return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_iphi(r)**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*v_iphi(r)*((m*B_iphi(r)/r)+(wavenum*B_i(r)))/r))
#
#Q_np=sym.lambdify(rr,Q(rr),"numpy")
#
#def T(r):
#  return ((((m*B_iphi(r)/r)+(wavenum*B_i(r)))*B_iphi(r)) + rho_i(r)*v_iphi(r)*shift_freq(r))
#
#T_np=sym.lambdify(rr,T(rr),"numpy")
#
#def C1(r):
#  return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))
#
#C1_np=sym.lambdify(rr,C1(rr),"numpy")
#   
#def C2(r):   
#  return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
#
#C2_np=sym.lambdify(rr,C2(rr),"numpy")  
#   
#def C3_diff(r):
#  return ((B_iphi(r)/r)**2 - (rho_i(r)*(v_iphi(r)/r)**2))           
#
#def C3(r):
#  return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
#  
#C3_np=sym.lambdify(rr,C3(rr),"numpy")
#
#
###########################################
#                             
#def F(r):  
#  return ((r*D(r))/C3(r))
#
#F_np=sym.lambdify(rr,F(rr),"numpy")  
#          
#def dF(r):   #First derivative of profile in symbols    
#  return sym.diff(F(r), r)  
#
#dF_np=sym.lambdify(rr,dF(rr),"numpy")
#
#
#def g(r):            
#  return ( -(sym.diff((r*C1(r)/C3(r)), r)) - (r*(C2(r)-(C1(r)**2/C3(r)))/D(r)))
#  
#g_np=sym.lambdify(rr,g(rr),"numpy")
#  
#######################################################       
#
#def dP_dr_e(P_e, r_e):
#       return [P_e[1], (-P_e[1]/r_e + (m_e+((1.**2)/(r_e**2)))*P_e[0])]
#  
#P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
#Ls = odeintz(dP_dr_e, P0, lx)
#left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side
#    
#left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side
#
#normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
#normalised_left_xi_solution = left_xi_solution/np.amax(abs(left_xi_solution))
#left_bound_P = left_P_solution[-1] 
#                    
#
#def dP_dr_i(P_i, r_i):
#      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
#      
#          
#def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
#      U = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
#      u1 = U[:,0] + ((B_iphi_np(1)**2) - (rho_i_np(1)*v_iphi_np(1)**2))*left_xi_solution[-1]
#      return u1[-1] 
#      
#dPi, = fsolve(objective_dPi, 0.001) 
#
## now solve with optimal dvx
#
#Is = odeintz(dP_dr_i, [left_bound_P, dPi], ix)
#inside_P_solution = Is[:,0]
#
##inside_xi_solution = xi_i_const*Is[:,1] #+ Is[:,0]/ix)   # Pressure perturbation solution for left hand side
#inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side
#
#normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
#normalised_inside_xi_solution = inside_xi_solution/np.amax(abs(left_xi_solution))
#
#
##########
#
#spatial_p15 = np.concatenate((ix[::-1], lx[::-1]), axis=None)     
#
#radial_displacement_p15 = np.concatenate((inside_xi_solution[::-1], left_xi_solution[::-1]), axis=None)  
#radial_PT_p15 = np.concatenate((inside_P_solution[::-1], left_P_solution[::-1]), axis=None)  
#
#normalised_radial_displacement_p15 = np.concatenate((normalised_inside_xi_solution[::-1], normalised_left_xi_solution[::-1]), axis=None)  
#normalised_radial_PT_p15 = np.concatenate((normalised_inside_P_solution[::-1], normalised_left_P_solution[::-1]), axis=None)  
#
#
#######  v_r   #######
#
#outside_v_r = -frequency*left_xi_solution[::-1]
#inside_v_r = -shift_freq_np(ix[::-1])*inside_xi_solution[::-1]
#
#radial_vr_p15 = np.concatenate((inside_v_r, outside_v_r), axis=None)    
#
#normalised_outside_v_r = -w*normalised_left_xi_solution[::-1]
#normalised_inside_v_r = -shift_freq_np(ix[::-1])*normalised_inside_xi_solution[::-1]
#
#normalised_radial_vr_p15 = np.concatenate((normalised_inside_v_r, normalised_outside_v_r), axis=None)    
#
#
######
#
#inside_xi_z = ((f_B_np(ix[::-1])*(c_i0**2/(c_i0**2 + vA_i0**2))*(shift_freq_np(ix[::-1])**2*inside_P_solution[::-1] - Q_np(ix[::-1])*inside_xi_solution[::-1])/(shift_freq_np(ix[::-1])**2*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - cusp_freq_np(ix[::-1])**2))) - (((2.*shift_freq_np(ix[::-1])*v_iphi_np(ix[::-1])*B_iphi_np(ix[::-1]) + f_B_np(ix[::-1])*v_iphi_np(ix[::-1])**2))*(inside_xi_solution[::-1]/ix[::-1])) - (B_iphi_np(ix[::-1])*(g_B_np(ix[::-1])*inside_P_solution[::-1] - 2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(B_i_np(ix[::-1])*rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/(B_iphi_np(ix[::-1])**2/B_i_np(ix[::-1]) + B_i_np(ix[::-1]))
#outside_xi_z = k*c_e**2*w**2*left_P_solution[::-1]/(rho_e*(w**2-(k**2*cT_e**2))*(c_e**2+vA_e**2))
#radial_xi_z_p15 = np.concatenate((inside_xi_z, outside_xi_z), axis=None) 
#
#
#
#inside_xi_phi = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*T_np(ix[::-1])*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i_np(ix[::-1])*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))) + (B_iphi_np(ix[::-1])*inside_xi_z))/B_i_np(ix[::-1])
#outside_xi_phi = (m*left_P_solution[::-1]/lx[::-1])/(rho_e*(w**2 - (k**2*vA_e**2)))
#radial_xi_phi_p15 = np.concatenate((inside_xi_phi, outside_xi_phi), axis=None)    
#
#normalised_inside_xi_phi = inside_xi_phi/np.amax(abs(outside_xi_phi))
#normalised_outside_xi_phi = outside_xi_phi/np.amax(abs(outside_xi_phi))
#normalised_radial_xi_phi_p15 = np.concatenate((normalised_inside_xi_phi, normalised_outside_xi_phi), axis=None)    
#
#######
#
#######  v_phi   #######
#
#def dv_phi(r):
#  return sym.diff(v_iphi(r)/r)
#
#dv_phi_np=sym.lambdify(rr,dv_phi(rr),"numpy")
#
#outside_v_phi = -w*outside_xi_phi
#inside_v_phi = -(shift_freq_np(ix[::-1])*inside_xi_phi) - (dv_phi_np(ix[::-1])*ix[::-1]*inside_xi_solution[::-1])
#
#radial_v_phi_p15 = np.concatenate((inside_v_phi, outside_v_phi), axis=None)    
#
#normalised_inside_v_phi = inside_v_phi/np.amax(abs(outside_v_phi))
#normalised_outside_v_phi = outside_v_phi/np.amax(abs(outside_v_phi))
#normalised_radial_v_phi_p15 = np.concatenate((normalised_inside_v_phi, normalised_outside_v_phi), axis=None)    
#
#
#######  v_z   #######
#U_e = 0.
#U_i0 = 0.
#dr=1e5
#def v_iz(r):
#    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2))) 
#
#def dv_z(r):
#  return sym.diff(v_iz(r)/r)
#
#dv_z_np=sym.lambdify(rr,dv_z(rr),"numpy")
#
#outside_v_z = -w*outside_xi_z
#inside_v_z = -(shift_freq_np(ix[::-1])*inside_xi_z) - (dv_z_np(ix[::-1])*inside_xi_solution[::-1])
#
#radial_v_z_p15 = np.concatenate((inside_v_z, outside_v_z), axis=None)    
#
#######################






###################################################################
B = 1.

###### KINK   #######

fig, (ax3, ax, ax2) = plt.subplots(3,1, sharex=False)

ax.axvline(x=B, color='r', linestyle='--')

ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax3.set_ylim(-1.5,0.)
ax3.set_xlim(0.001, 2.)

#ax3.plot(spatial_p075, normalised_radial_PT_p075, color='goldenrod')
ax3.plot(spatial_p08, normalised_radial_PT_p08, color='deepskyblue')
#ax3.plot(spatial_p085, normalised_radial_PT_p085, color='green')
#ax3.plot(spatial_p09, normalised_radial_PT_p09, color='magenta')
ax3.plot(spatial_p1, normalised_radial_PT_p1, 'k')
#ax3.plot(spatial_p15, normalised_radial_PT_p15, 'r')
#ax3.plot(spatial_p2, normalised_radial_PT_p2, 'g')



ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$\hat{v}_r$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0.001, 2.)
ax.set_ylim(-2.,3.5)

#ax.plot(spatial_p075, normalised_radial_vr_p075, color='goldenrod')
ax.plot(spatial_p08, normalised_radial_vr_p08, color='deepskyblue')
#ax.plot(spatial_p085, normalised_radial_vr_p085, color='green')
#ax.plot(spatial_p09, normalised_radial_vr_p09, color='magenta')
ax.plot(spatial_p1, normalised_radial_vr_p1, 'k')
#ax.plot(spatial_p15, normalised_radial_vr_p15, 'r')
#ax.plot(spatial_p2, normalised_radial_vr_p2, 'g')


ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel(r"$\hat{v}_{\varphi}$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0.001, 2.)


#ax2.plot(spatial_p075, normalised_radial_v_phi_p075, color='goldenrod')
ax2.plot(spatial_p08, normalised_radial_v_phi_p08, color='deepskyblue')
#ax2.plot(spatial_p085, normalised_radial_v_phi_p085, color='green')
#ax2.plot(spatial_p09, normalised_radial_v_phi_p09, color='magenta')
ax2.plot(spatial_p1, normalised_radial_v_phi_p1, 'k')
#ax2.plot(spatial_p15, normalised_radial_v_phi_p15, 'r')
#ax2.plot(spatial_p2, normalised_radial_v_phi_p2, 'g')



#########################


fig, (ax, ax2) = plt.subplots(2,1, sharex=False)


ax.set_xlabel("$r$", fontsize=18)
ax.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)
ax.set_xlim(0.001, 2.)
ax.set_ylim(-2.2, 1.2)

#ax.plot(spatial_p075, normalised_radial_displacement_p075, color='goldenrod')
ax.plot(spatial_p08, normalised_radial_displacement_p08, color='deepskyblue')
#ax.plot(spatial_p085, normalised_radial_displacement_p085, color='green')
#ax.plot(spatial_p09, normalised_radial_displacement_p09, color='magenta')
ax.plot(spatial_p1, normalised_radial_displacement_p1, 'k')
#ax.plot(spatial_p15, normalised_radial_displacement_p15, 'r')
#ax.plot(spatial_p2, normalised_radial_displacement_p2, 'g')


ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
#ax2.set_ylabel(r"$\hat{\U03BE}_{\varphi}$", fontsize=18, rotation=0, labelpad=15)
ax2.set_ylabel(r"$\hat{\xi}_{\varphi}$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0.001, 2.)


#ax2.plot(spatial_p075, normalised_radial_xi_phi_p075, color='goldenrod')
ax2.plot(spatial_p08, normalised_radial_xi_phi_p08, color='deepskyblue')
#ax2.plot(spatial_p085, normalised_radial_xi_phi_p085, color='green')
#ax2.plot(spatial_p09, normalised_radial_xi_phi_p09, color='magenta')
ax2.plot(spatial_p1, normalised_radial_xi_phi_p1, 'k')
#ax2.plot(spatial_p15, normalised_radial_xi_phi_p15, 'r')
#ax2.plot(spatial_p2, normalised_radial_xi_phi_p2, 'g')




plt.show()
exit()









### movie   ####

image = []

B = 1.

fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False) 

ax.set_ylabel(r'$\frac{\omega}{k}$', fontsize=18, rotation=0, labelpad=15)
ax.set_xlabel("$kr$", fontsize=12)

ax.plot(sol_ksv001, (sol_omegasv001/sol_ksv001), 'r.', markersize=4.)
ax.plot(sol_ks_kinkv001, (sol_omegas_kinkv001/sol_ks_kinkv001), 'b.', markersize=4.)

ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_kink, color='k', label='$c_{k}$', linestyle='dashdot')

ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{e}, c_{Te}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)

#ax.set_ylim(0., 5.)
#ax.set_ylim(0.87, 1.05)
ax.set_ylim(1.9, 3.1)


ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax2.set_xlim(0.001, 2.)
ax2.set_ylim(-1.5, 1.5)


ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_xlabel("$r$", fontsize=18)
ax3.set_ylabel("$\hat{\u03be}_x$", fontsize=18, rotation=0, labelpad=15)
ax3.set_xlim(0.001, 2.)
ax3.set_ylim(-3.5, 3.5)




rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

B_twist = 0.01
def B_iphi(r):  
    return 0.   #B_twist*r   #0.

B_iphi_np=sym.lambdify(rr,B_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy

v_twist = 0.01
def v_iphi(r):  
    return v_twist*r   #0.

v_iphi_np=sym.lambdify(rr,v_iphi(rr),"numpy")   #In order to evaluate we need to switch to numpy


#wavenum = fast_sausage_branch2_k
#frequency = fast_sausage_branch2_omega

#wavenum = slow_body_sausage_branch1_k
#frequency = slow_body_sausage_branch1_omega



#######################     USE FOR SAUSAGE      ###############################

#for i in range(len(wavenum)):
#    m = 0.
#    
#    lx = np.linspace(3.*2.*np.pi/(wavenum[i]), 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain      
#                       
#    m_e = ((((wavenum[i]**2*vA_e**2)-frequency[i]**2)*((wavenum[i]**2*c_e**2)-frequency[i]**2))/((vA_e**2+c_e**2)*((wavenum[i]**2*cT_e**2)-frequency[i]**2)))
#    
#    xi_e_const = -1/(rho_e*((wavenum[i]**2*vA_e**2)-frequency[i]**2))       
#    
#                   ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
#    
#    def shift_freq(r):
#      return (frequency[i] - (m*(v_iphi(r))/r) - wavenum[i]*v_z)
#      
#    def alfven_freq(r):
#      return ((m*B_iphi(r)/r)+(wavenum[i]*B_i(r))/(sym.sqrt(rho_i(r))))
#    
#    def cusp_freq(r):
#      return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2 + vA_i(r)**2)))
#    
#    def D(r):  
#      return (rho_i(r)*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))
#    
#    D_np=sym.lambdify(rr,D(rr),"numpy")
#    
#    def Q(r):
#      return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*(v_iphi(r))**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*(v_iphi(r))*((m*B_iphi(r)/r)+(wavenum[i]*B_i(r)))/r))
#    
#    def T(r):
#      return ((((m*B_iphi(r)/r)+(wavenum[i]*B_i(r)))*B_iphi(r)) + rho_i(r)*(v_iphi(r))*shift_freq(r))
#    
#    def C1(r):
#      return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))
#    
#    C1_np=sym.lambdify(rr,C1(rr),"numpy")
#        
#    def C2(r):   
#      return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[i]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
#       
#    def C3_diff(r):
#      return ((B_iphi(r)/r)**2 - (rho_i(r)*((v_iphi(r))/r)**2))
#    
#    def C3(r):
#      return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
#      
#    C3_np=sym.lambdify(rr,C3(rr),"numpy")
#                                  
#    def F(r):  
#      return ((r*D(r))/C3(r))
#    
#    F_np=sym.lambdify(rr,F(rr),"numpy")  
#              
#    def dF(r):   #First derivative of profile in symbols    
#      return sym.diff(F(r), r)  
#    
#    dF_np=sym.lambdify(rr,dF(rr),"numpy")
#    
#    
#    def g(r):            
#      return ( -(sym.diff((r*C1(r)/C3(r)), r)) - (r*(C2(r)-(C1(r)**2/C3(r)))/D(r)))
#      
#    g_np=sym.lambdify(rr,g(rr),"numpy")
#      
#    ###################################################### 
#    
#    disp_dot, = ax.plot(wavenum[i], frequency[i]/wavenum[i], 'k.',markersize=8.)
#               
#    def dP_dr_e(P_e, r_e):
#           return [P_e[1], (-P_e[1]/r_e + (m_e+0/(r_e**2))*P_e[0])]
#      
#    P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
#    Ls = odeint(dP_dr_e, P0, lx)
#    left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side
#    
#    left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side
#    
#    normalised_left_P_solution_v001 = left_P_solution/np.amax(abs(left_P_solution))
#    normalised_left_xi_solution_v001 = left_xi_solution/np.amax(abs(left_xi_solution))
#    left_bound_P = left_P_solution[-1] 
#
#    left_P_plot, = ax2.plot(lx, normalised_left_P_solution_v001, 'k')                       
#    left_xi_plot, = ax3.plot(lx, normalised_left_xi_solution_v001, 'k')
#    
#    def dP_dr_i(P_i, r_i):
#           return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
#    
#               
#    def objective_dPi(dPi):   
#          U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
#          u1 = U[:,1] 
#          return u1[-1] 
#          
#    dPi, = fsolve(objective_dPi, 0.001) 
#    
#    Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
#    inside_P_solution = Is[:,0]
#    
#    inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side
#    
#    normalised_inside_P_solution_v001 = inside_P_solution/np.amax(abs(left_P_solution))
#    normalised_inside_xi_solution_v001 = inside_xi_solution/np.amax(abs(left_xi_solution))
#
#    inside_P_plot, = ax2.plot(ix, normalised_inside_P_solution_v001, 'k')                       
#    inside_xi_plot, = ax3.plot(ix, normalised_inside_xi_solution_v001, 'k')
#    
#    image.append([disp_dot, left_P_plot, inside_P_plot, left_xi_plot, inside_xi_plot])
 
####################################################################################

####################################################################################


##############      USE FOR KINK     ############################################### 
 

#wavenum = slow_body_kink_branch1_k
#frequency = slow_body_kink_branch1_omega

wavenum = fast_kink_branch2_k
frequency = fast_kink_branch2_omega

m = 1.


for i in range(len(wavenum)):
    lx = np.linspace(2.5*2.*np.pi/wavenum[i], 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain
    
    m_e = ((((wavenum[i]**2*vA_e**2)-frequency[i]**2)*((wavenum[i]**2*c_e**2)-frequency[i]**2))/((vA_e**2+c_e**2)*((wavenum[i]**2*cT_e**2)-frequency[i]**2)))
      
            ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
    
    def shift_freq(r):
      return (frequency[i] - (m*v_iphi(r)/r) - wavenum[i]*v_z)
     
    def alfven_freq(r):
      return ((m*B_iphi(r)/r)+(wavenum[i]*B_i(r))/(sym.sqrt(rho_i(r))))
    
    def cusp_freq(r):
      return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))
    
    def D(r):  
      return (rho_i(r)*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))
    
    D_np=sym.lambdify(rr,D(rr),"numpy")
    
    ###         
    D_e_np = (rho_e*(c_e**2+vA_e**2)*(frequency[i]**2 - wavenum[i]**2*vA_e**2)*(frequency[i]**2 - wavenum[i]**2*cT_e**2))
    ###
    
    
    def Q(r):
      return ((-(shift_freq(r)**2 - alfven_freq(r)**2)*rho_i(r)*v_iphi(r)**2/r) + (2*shift_freq(r)**2*B_iphi(r)**2/r)+(2*shift_freq(r)*B_iphi(r)*v_iphi(r)*((m*B_iphi(r)/r)+(wavenum[i]*B_i(r)))/r))
    
    def T(r):
      return ((((m*B_iphi(r)/r)+(wavenum[i]*B_i(r)))*B_iphi(r)) + rho_i(r)*v_iphi(r)*shift_freq(r))
    
    def C1(r):
      return ((Q(r)*shift_freq(r)**2) - (2*m*(c_i(r)**2+vA_i(r)**2)*(shift_freq(r)**2-cusp_freq(r)**2)*T(r)/r**2))
    
    C1_np=sym.lambdify(rr,C1(rr),"numpy")
      
    def C2(r):   
      return ( shift_freq(r)**4 - ((c_i(r)**2 + vA_i(r)**2)*(m**2/r**2 + wavenum[i]**2)*(shift_freq(r)**2 - cusp_freq(r)**2)))
    
    C2_np=sym.lambdify(rr,C2(rr),"numpy")
    
    ###
    def C2_e(r):   
      return (frequency[i]**4 - ((c_e**2 + vA_e**2)*(m**2/r**2 + wavenum[i]**2)*(frequency[i]**2 - wavenum[i]**2*cT_e**2)))
    
    C2_e_np=sym.lambdify(rr,C2_e(rr),"numpy")
    ###
    
      
    def C3_diff(r):
      return ((B_iphi(r)/r)**2 - (rho_i(r)*(v_iphi(r)/r)**2))
    
    
    def C3(r):
      return ( (D(r)*(rho_i(r)*(shift_freq(r)**2 - alfven_freq(r)**2) + (r*sym.diff(C3_diff(r), r)))) + (Q(r)**2 - (4*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 -  cusp_freq(r)**2)*T(r)**2/r**2)) )
     
    C3_np=sym.lambdify(rr,C3(rr),"numpy")
    
    ###         
    C3_e_np = D_e_np*rho_e*(frequency[i]**2 - wavenum[i]**2*vA_e**2)
    ###
    
    ##########################################
                                 
    def F(r):  
      return D(r)/(r*C2(r))
    
    F_np=sym.lambdify(rr,F(rr),"numpy")  
    
    def F_e(r):  
      return (D_e_np/(r*C2_e(r)))
    
    F_e_np=sym.lambdify(rr,F_e(rr),"numpy")
    
             
    def dF(r):   #First derivative of profile in symbols    
      return sym.diff(F(r), r)  
    
    dF_np=sym.lambdify(rr,dF(rr),"numpy")
    
    
    def dF_e(r):   #First derivative of profile in symbols    
      return sym.diff(F_e(r), r)  
    
    dF_e_np=sym.lambdify(rr,dF_e(rr),"numpy")
    
    
    def g(r):            
      return  (sym.diff(C1(r)/(r*C2(r)), r)) - ((C3(r)-(C1(r)**2/C2(r)))/(r*D(r)))
     
    g_np=sym.lambdify(rr,g(rr),"numpy")
    
    def g_e(r):            
      return (-rho_e*(frequency[i]**2 - wavenum[i]**2*vA_e**2))/r
     
    g_e_np=sym.lambdify(rr,g_e(rr),"numpy")  
     
    ######################################################       
    
    disp_dot, = ax.plot(wavenum[i], frequency[i]/wavenum[i], 'k.',markersize=8.)
    
    def dxi_dr_e(xi_e, r_e):
           return [xi_e[1], ((-(2./r_e + dF_e_np(r_e)/F_e_np(r_e)))*xi_e[1] - ((dF_e_np(r_e)/(F_e_np(r_e)*r_e) - g_e_np(r_e)/F_e_np(r_e)))*xi_e[0])]
      
    
    xi0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
    Ls = odeint(dxi_dr_e, xi0, lx)
    left_xi_solution = Ls[:,0]      # Vx perturbation solution for left hand side
    
    left_P_solution = -D_e_np*(Ls[:,1] + Ls[:,0]/lx)/(C2_e_np(lx))    # Pressure perturbation solution for left hand side
    
    normalised_left_P_solution_v001_kink = left_P_solution/np.amax(abs(left_P_solution))
    normalised_left_xi_solution_v001_kink = left_xi_solution/np.amax(abs(left_xi_solution))
    left_bound_xi = left_xi_solution[-1] 
         
    left_P_plot_kink, = ax2.plot(lx, normalised_left_P_solution_v001_kink, 'k')                       
    left_xi_plot_kink, = ax3.plot(lx, normalised_left_xi_solution_v001_kink, 'k')
     
    def dxi_dr_i(xi_i, r_i):
           return [xi_i[1], ((-(2./r_i + dF_np(r_i)/F_np(r_i)))*xi_i[1] - ((dF_np(r_i)/(F_np(r_i)*r_i) - g_np(r_i)/F_np(r_i)))*xi_i[0])]
    
        
    def objective_dxii(dxii):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
          U = odeint(dxi_dr_i, [left_bound_xi, dxii], ix)
          u1 = U[:,1]       #was plus (assuming pressure is symmetric for sausage)
          return u1[-1] 
          
    dxii, = fsolve(objective_dxii, 0.001) 
    
    Is = odeint(dxi_dr_i, [left_bound_xi, dxii], ix)
    inside_xi_solution = Is[:,0]
    
    inside_P_solution = (C1_np(ix)*Is[:,0] - D_np(ix)*(Is[:,1] + Is[:,0]/ix))/(C2_np(ix))     # Pressure perturbation solution for left hand side
    
    normalised_inside_P_solution_v001_kink = inside_P_solution/np.amax(abs(left_P_solution))
    normalised_inside_xi_solution_v001_kink = inside_xi_solution/np.amax(abs(left_xi_solution))

    inside_P_plot_kink, = ax2.plot(ix, normalised_inside_P_solution_v001_kink, 'k')                       
    inside_xi_plot_kink, = ax3.plot(ix, normalised_inside_xi_solution_v001_kink, 'k')
    
    image.append([disp_dot, left_P_plot_kink, inside_P_plot_kink, left_xi_plot_kink, inside_xi_plot_kink])
 

Writer = animation.writers['ffmpeg']
writer = Writer(fps=3, bitrate=4000)   #9fps if skipping 5 in time step   15 good
   
anim = animation.ArtistAnimation(fig, image, interval=0.1, blit=True, repeat=False).save('cylinder_v001_coronal_kinkbranch_kink.mp4', writer=writer)



plt.show()
exit()



