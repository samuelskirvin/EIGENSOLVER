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

Kmax = 4.5
wavenumber = np.linspace(0.01,Kmax,20)      #(1.5, 1.8), 5     

ix = np.linspace(-1., -0.001, 500.)  # inside slab x values

r0=0.  #mean
dr=0.9 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

rho_A = 1.   #Amplitude of internal density
A = 1.

#def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
#    return (B_e + ((B_0 - B_e)*sym.exp(-(r-r0)**2/dr**2)))   

def profile(r):       # Define the internal profile as a function of variable x   (Inverted Gaussian)
    return (rho_e + ((rho_i0 - rho_e)*sym.exp(-(r-r0)**2/dr**2)))   

###################
#a = 1.   #inhomogeneity width for epstein profile
#def profile(x):       # Define the internal profile as a function of variable x   (Epstein Profile)
#    return ((rho_i0 - rho_e)/(sym.cosh(x/a)**4)**2 + rho_e)
####################



#def profile(x):       # Define the internal profile as a function of variable x   (Periodic threaded waveguide)
#    return (3.*rho_e + (rho_i0 - 3.*rho_e)*sym.cos(2.*sym.pi*x + sym.pi))    #2.28 = W ~ 0.6     #3. corresponds to W ~ 1.5




################################################
def rho_i(r):                  # Define the internal density profile
    return rho_A*profile(r)

#def rho_i(x):                  # nope
#    return sym.sqrt(B_i(x)*sym.sqrt(rho_i0)*vA_i0/(B_0*vA_i(x)))

#def rho_i(r):                  
#    return rho_i0

#############################################

def P_i(r):
    return ((c_i(r))**2*rho_i(r)/gamma)

#def P_i(x):   #constant temp
#    return ((c_i0)**2*rho_i(x)/gamma)

#############################################

def T_i(r):     #use this for constant P
    return (P_i(r)/rho_i(r))

#def T_i(x):    # constant temp
#    return (P_i(x)*rho_i0/(P_0*rho_i(x)))


##########################################
#def vA_i(x):                  # Define the internal alfven speed     #This works!!!!
#    return vA_i0*sym.sqrt(rho_i0)/sym.sqrt(profile(x))


#def vA_i(x):                  # keep rho const
#    return vA_i0*sym.sqrt(rho_i0)*B_i(x)/sym.sqrt(rho_i(x))


def vA_i(r):                  # consatnt temp
    return (B_i(r)+B_phi)/(sym.sqrt(rho_i(r)))

##########################################
#def B_i(x):
#    return (vA_i(x)*sym.sqrt(rho_i(x)))

#def B_i(r):  #constant rho
#    return (A*profile(r))


def B_i(r):  #constant rho
    return B_0


###################################

def PT_i(r):
    return P_i(r) + B_i(r)**2/2.

###################################

def c_i(r):                  # Define the internal sound speed
    return sym.sqrt((rho_e*(c_e**2 + 0.5*gamma*vA_e**2)/rho_i(r)) - 0.5*gamma*vA_i(r)**2)

#def c_i(x):                  # keeps c_i constant
#    return c_i0*sym.sqrt(P_i(x))*sym.sqrt(rho_i0)/(sym.sqrt(P_0)*sym.sqrt(rho_i(x)))

#def c_i(x):                  # keeps c_i constant
#    return sym.sqrt(P_i(x))/sym.sqrt(gamma*rho_i(x))

###################################################
def cT_i(r):                 # Define the internal tube speed
    return sym.sqrt(((c_i(r))**2 * (vA_i(r))**2) / ((c_i(r))**2 + (vA_i(r))**2))


#####################################################
speeds = [c_i0, c_e, vA_i0, vA_e, cT_i0, cT_e -c_e, -c_i0, -vA_i0, -vA_e, -cT_i0, -cT_e]  #added -c_e and cT_i0
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

c_kink_bound = np.sqrt(((rho_bound*vA_bound**2)+(rho_e*vA_e**2))/(rho_bound+rho_e))
c_kink_bound2 = np.sqrt(((rho_bound*vA_i0**2)+(rho_e*vA_e**2))/(rho_bound+rho_e))

c_kink_test = np.sqrt(B_0**2+B_e**2/(rho_e+rho_i0))
c_kink_test2 = np.sqrt(B_0**2+B_e**2/(rho_e+rho_bound))

print('c_kink_bound    =', c_kink_bound)
print('c_kink_bound2    =', c_kink_bound2)
print('c_kink_test    =', c_kink_test)
print('c_kink_test2    =', c_kink_test2)


def c_k(r): 
  return sym.sqrt((B_i(r)**2 + B_e**2)/(rho_e + rho_i(r)))

c_kink_np=sym.lambdify(rr,c_k(rr),"numpy")   #In order to evaluate we need to switch to numpy


def c_k2(r): 
  return sym.sqrt(((rho_i(r)*vA_i(r)**2) + rho_e*vA_e**2)/(rho_e + rho_i(r)))

c_kink2_np=sym.lambdify(rr,c_k2(rr),"numpy")   #In order to evaluate we need to switch to numpy

def c_quasi(r):
  return sym.sqrt(((rho_i(r)*vA_i(r)**2) + rho_bound*vA_bound**2)/(rho_e + rho_i(r)))

c_quasi_np=sym.lambdify(rr,c_quasi(rr),"numpy")   #In order to evaluate we need to switch to numpy

def c_quasi2(r):
  return sym.sqrt(((rho_e*vA_e**2) + rho_bound*vA_bound**2)/(rho_e + rho_bound))
c_quasi2_np=sym.lambdify(rr,c_quasi2(rr),"numpy")   #In order to evaluate we need to switch to numpy


plt.figure()
ax=plt.subplot(111)
ax.plot(ix, c_kink_np(ix))
ax.plot(ix, c_kink2_np(ix), 'g--')
plt.xlabel('$r$')

ax.plot(ix, c_quasi_np(ix), 'r--')
ax.axhline(y=c_quasi2_np(ix))

plt.ylabel('$c_k(r)$')
ax.axhline(y=3.356, color='r')
#################       NEED TO NOTE        ######################################
###########     It is seen that the density profile created is        ############
###########     actually the profile for sqrt(rho_i) as c_i0 &        ############
###########     vA_i0 are divided by this and not sqrt(profile)       ############
##################################################################################


plt.figure()
plt.title("width = 1.5")
#plt.xlabel("x")
#plt.ylabel("$\u03C1_{i}$",fontsize=25)
ax = plt.subplot(331)
ax.title.set_text("Density")
ax.plot(ix,rho_i_np(ix));
ax.annotate( '$\u03C1_{e}$', xy=(1, rho_e),fontsize=25)
ax.annotate( '$\u03C1_{i}$', xy=(1, rho_i0),fontsize=25)
ax.axhline(y=rho_i0, color='k', label='$\u03C1_{i}$', linestyle='dashdot')
ax.axhline(y=rho_e, color='k', label='$\u03C1_{e}$', linestyle='dashdot')
#ax.axhline(y=rho_i_np(ix), color='b', label='$\u03C1_{i}$', linestyle='solid')
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

plt.suptitle("W = 1e5", fontsize=14)
plt.tight_layout()






########   READ IN VARIABLES    #########



with open('Cylindrical_coronal_width09.pickle', 'rb') as f:
    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)


#with open('Cylindrical_coronal_flow_n015.pickle', 'rb') as f:
#    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)


### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas09 = [x for _,x in sorted(zip(sol_ks1,sol_omegas1))]
sol_ks09 = np.sort(sol_ks1)

sol_omegas09 = np.array(sol_omegas09)
sol_ks09 = np.array(sol_ks09)

sol_omegas_kink09 = [x for _,x in sorted(zip(sol_ks_kink1,sol_omegas_kink1))]
sol_ks_kink09 = np.sort(sol_ks_kink1)

sol_omegas_kink09 = np.array(sol_omegas_kink09)
sol_ks_kink09 = np.array(sol_ks_kink09)


####   CORONAL    ######

### FAST BODY    ( vA_i  <  w/k  <  vA_e )
fast_body_sausage_omega = []
fast_body_sausage_k = []
fast_body_kink_omega = []
fast_body_kink_k = []

### SLOW BODY    ( cT_i  <  w/k  <  c_i )
slow_body_sausage_omega = []
slow_body_sausage_k = []
slow_body_kink_omega = []
slow_body_kink_k = []


### SLOW FAST BODY   
slow_fast_body_sausage_omega = []
slow_fast_body_sausage_k = []
slow_fast_body_kink_omega = []
slow_fast_body_kink_k = []



### SLOW BACKWARD BODY   
slow_backward_body_sausage_omega = []
slow_backward_body_sausage_k = []
slow_backward_body_kink_omega = []
slow_backward_body_kink_k = []

new_modes_sausage_k = []
new_modes_sausage_omega = []
new_modes_kink_k = []
new_modes_kink_omega = []



for i in range(len(sol_ks09)):   #sausage mode
  v_phase = sol_omegas09[i]/sol_ks09[i]
      
  if v_phase > vA_i0 and v_phase < vA_e:  
      fast_body_sausage_omega.append(sol_omegas09[i])
      fast_body_sausage_k.append(sol_ks09[i])
      
  if v_phase > cT_i0 and v_phase < c_i0:  
      slow_body_sausage_omega.append(sol_omegas09[i])
      slow_body_sausage_k.append(sol_ks09[i])

  if v_phase < -cT_i0 and v_phase > -c_i0: 
      slow_backward_body_sausage_omega.append(sol_omegas09[i])
      slow_backward_body_sausage_k.append(sol_ks09[i])

  if v_phase > -vA_e and v_phase < -vA_i0:
      slow_fast_body_sausage_omega.append(sol_omegas09[i])
      slow_fast_body_sausage_k.append(sol_ks09[i])
            
  else:
      new_modes_sausage_k.append(sol_ks09[i])
      new_modes_sausage_omega.append(sol_omegas09[i])
      



for i in range(len(sol_ks_kink09)):   #kink mode
  v_phase = sol_omegas_kink09[i]/sol_ks_kink09[i]
      
  if v_phase > vA_i0 and v_phase < vA_e: 
      fast_body_kink_omega.append(sol_omegas_kink09[i])
      fast_body_kink_k.append(sol_ks_kink09[i])
            
  if v_phase > cT_i0 and v_phase < c_i0: 
      slow_body_kink_omega.append(sol_omegas_kink09[i])
      slow_body_kink_k.append(sol_ks_kink09[i])

  if v_phase < -cT_i0 and v_phase > -c_i0:
      slow_backward_body_kink_omega.append(sol_omegas_kink09[i])
      slow_backward_body_kink_k.append(sol_ks_kink09[i])

  if v_phase > -vA_e and v_phase < -vA_i0:
      slow_fast_body_kink_omega.append(sol_omegas_kink09[i])
      slow_fast_body_kink_k.append(sol_ks_kink09[i])
            
  else:
      new_modes_kink_k.append(sol_ks_kink09[i])
      new_modes_kink_omega.append(sol_omegas_kink09[i])



fast_body_sausage_omega = np.array(fast_body_sausage_omega)
fast_body_sausage_k = np.array(fast_body_sausage_k)
fast_body_kink_omega = np.array(fast_body_kink_omega)
fast_body_kink_k = np.array(fast_body_kink_k)

slow_body_sausage_omega = np.array(slow_body_sausage_omega)
slow_body_sausage_k = np.array(slow_body_sausage_k)
slow_body_kink_omega = np.array(slow_body_kink_omega)
slow_body_kink_k = np.array(slow_body_kink_k)

 
slow_fast_body_sausage_omega = np.array(slow_fast_body_sausage_omega)
slow_fast_body_sausage_k = np.array(slow_fast_body_sausage_k)
slow_fast_body_kink_omega = np.array(slow_fast_body_kink_omega)
slow_fast_body_kink_k = np.array(slow_fast_body_kink_k)

new_modes_sausage_k = np.array(new_modes_sausage_k)
new_modes_sausage_omega = np.array(new_modes_sausage_omega)
new_modes_kink_k = np.array(new_modes_kink_k)
new_modes_kink_omega = np.array(new_modes_kink_omega)


cutoff = 15.  
cutoff2 = 5.  


fast_sausage_branch1_omega = []
fast_sausage_branch1_k = []
fast_sausage_branch2_omega = []
fast_sausage_branch2_k = []


fast_sausage_branch3_omega = []
fast_sausage_branch3_k = []


#################################################
for i in range(len(fast_body_sausage_omega)):   #sausage mode
      
  if fast_body_sausage_omega[i] > cutoff:  
      fast_sausage_branch1_omega.append(fast_body_sausage_omega[i])
      fast_sausage_branch1_k.append(fast_body_sausage_k[i])
      
  elif fast_body_sausage_omega[i] < cutoff and fast_body_sausage_omega[i] > cutoff2:  
      fast_sausage_branch2_omega.append(fast_body_sausage_omega[i])
      fast_sausage_branch2_k.append(fast_body_sausage_k[i])


fast_sausage_branch1_omega = fast_sausage_branch1_omega[::-1]
fast_sausage_branch1_k = fast_sausage_branch1_k[::-1]

##############################################
index_to_remove = []
for i in range(len(fast_sausage_branch1_omega)-1):
    ph_diff = abs((fast_sausage_branch1_omega[i+1]/fast_sausage_branch1_k[i+1]) - (fast_sausage_branch1_omega[i]/fast_sausage_branch1_k[i]))
   
    if abs(fast_sausage_branch1_k[i] - fast_sausage_branch1_k[i+1]) > 0.5:
      index_to_remove.append(i+1)

fast_sausage_branch1_omega = np.delete(fast_sausage_branch1_omega, index_to_remove)    
fast_sausage_branch1_k = np.delete(fast_sausage_branch1_k, index_to_remove) 
############################################  

fast_sausage_branch1_omega = fast_sausage_branch1_omega[::-1]
fast_sausage_branch1_k = fast_sausage_branch1_k[::-1]

     
############################################################################# 
      
fast_sausage_branch1_omega = np.array(fast_sausage_branch1_omega)
fast_sausage_branch1_k = np.array(fast_sausage_branch1_k)
fast_sausage_branch2_omega = np.array(fast_sausage_branch2_omega)
fast_sausage_branch2_k = np.array(fast_sausage_branch2_k)
fast_sausage_branch3_omega = np.array(fast_sausage_branch3_omega)
fast_sausage_branch3_k = np.array(fast_sausage_branch3_k)



########################################

slow_fast_sausage_branch1_omega = []
slow_fast_sausage_branch1_k = []
slow_fast_sausage_branch2_omega = []
slow_fast_sausage_branch2_k = []
slow_fast_sausage_branch3_omega = []
slow_fast_sausage_branch3_k = []


#################################################
for i in range(len(slow_fast_body_sausage_omega)):   #sausage mode
  v_phase = slow_fast_body_sausage_omega[i]/slow_fast_body_sausage_k[i]
  
  if slow_fast_body_sausage_k[i] < 2.6 and slow_fast_body_sausage_k[i] > 1.:
      slow_fast_sausage_branch1_omega.append(slow_fast_body_sausage_omega[i])
      slow_fast_sausage_branch1_k.append(slow_fast_body_sausage_k[i])
      
  if slow_fast_body_sausage_k[i] > 2.6 and v_phase > -3.6:
      slow_fast_sausage_branch1_omega.append(slow_fast_body_sausage_omega[i])
      slow_fast_sausage_branch1_k.append(slow_fast_body_sausage_k[i])
      
  elif slow_fast_body_sausage_k[i] > 2.6 and v_phase < -3.6:        
      slow_fast_sausage_branch2_omega.append(slow_fast_body_sausage_omega[i])
      slow_fast_sausage_branch2_k.append(slow_fast_body_sausage_k[i])              


     
############################################################################# 
      
slow_fast_sausage_branch1_omega = np.array(slow_fast_sausage_branch1_omega)
slow_fast_sausage_branch1_k = np.array(slow_fast_sausage_branch1_k)
slow_fast_sausage_branch2_omega = np.array(slow_fast_sausage_branch2_omega)
slow_fast_sausage_branch2_k = np.array(slow_fast_sausage_branch2_k)
slow_fast_sausage_branch3_omega = np.array(slow_fast_sausage_branch3_omega)
slow_fast_sausage_branch3_k = np.array(slow_fast_sausage_branch3_k)




cutoff_kink = 11. 
cutoff2_kink = 18. 
cutoff3_kink = 6.

##################################################   kink mode
fast_kink_branch1_omega = []
fast_kink_branch1_k = []
fast_kink_branch2_omega = []
fast_kink_branch2_k = []
fast_kink_branch3_omega = []
fast_kink_branch3_k = []
##################################################
for i in range(len(fast_body_kink_omega)):   
      
  if fast_body_kink_omega[i] > cutoff_kink and fast_body_kink_omega[i] < cutoff2_kink:  
      fast_kink_branch1_omega.append(fast_body_kink_omega[i])
      fast_kink_branch1_k.append(fast_body_kink_k[i])
         
#  elif fast_body_kink_omega[i] < cutoff3_kink and fast_body_kink_omega[i]/fast_body_kink_k[i] > vA_i0:  
#      fast_kink_branch2_omega.append(fast_body_kink_omega[i])
#      fast_kink_branch2_k.append(fast_body_kink_k[i])

  elif fast_body_kink_omega[i]/fast_body_kink_k[i] > vA_bound and fast_body_kink_omega[i]/fast_body_kink_k[i] < c_kink_bound:  
      fast_kink_branch2_omega.append(fast_body_kink_omega[i])
      fast_kink_branch2_k.append(fast_body_kink_k[i])

  elif fast_body_kink_omega[i] > cutoff2_kink:
      fast_kink_branch3_omega.append(fast_body_kink_omega[i])
      fast_kink_branch3_k.append(fast_body_kink_k[i])
      
      
fast_kink_branch1_omega = np.array(fast_kink_branch1_omega)
fast_kink_branch1_k = np.array(fast_kink_branch1_k)
fast_kink_branch2_omega = np.array(fast_kink_branch2_omega)
fast_kink_branch2_k = np.array(fast_kink_branch2_k)
fast_kink_branch3_omega = np.array(fast_kink_branch3_omega)
fast_kink_branch3_k = np.array(fast_kink_branch3_k)

###################################################

#########################################################################   sausage polyfit

if len(fast_sausage_branch1_omega) > 1:
  FSB1_phase = fast_sausage_branch1_omega/fast_sausage_branch1_k
  FSB1_k = fast_sausage_branch1_k
  k_new = np.linspace(3.345, Kmax, num=len(FSB1_k)*10)
  
  coefs = poly.polyfit(FSB1_k, FSB1_phase, 3)   #  # 6 is good
  ffit = poly.polyval(k_new, coefs)

if len(fast_sausage_branch2_omega) > 1:
  FSB2_phase = fast_sausage_branch2_omega/fast_sausage_branch2_k
  FSB2_k = fast_sausage_branch2_k
  k_new_2 = np.linspace(1.591, FSB2_k[-1], num=len(FSB2_k)*10)  #FSB2_k[-1]
  
  coefs_2 = poly.polyfit(FSB2_k, FSB2_phase, 6)    # 6 order     # 1st order for W < 1.5
  ffit_2 = poly.polyval(k_new_2, coefs_2)

#########################################################################   kink polyfit

if len(fast_kink_branch1_omega) > 1:
  FSB1_kink_phase = fast_kink_branch1_omega/fast_kink_branch1_k
  FSB1_kink_k = fast_kink_branch1_k
  k_kink_new = np.linspace(2.385, Kmax, num=len(FSB1_kink_k)*10)   #FSB1_kink_k[-1]
  
  coefs_kink = poly.polyfit(FSB1_kink_k, FSB1_kink_phase, 6) # 3 order for messing
  ffit_kink = poly.polyval(k_kink_new, coefs_kink)

if len(fast_kink_branch2_omega) > 1:
  FSB2_kink_phase = fast_kink_branch2_omega/fast_kink_branch2_k
  FSB2_kink_k = fast_kink_branch2_k
  k_kink_new_2 = np.linspace(0., 1.9, num=len(FSB2_kink_k)*10)   #FSB2_kink_k[-1]
  
  coefs_2_kink = poly.polyfit(FSB2_kink_k, FSB2_kink_phase, 4)
  ffit_2_kink = poly.polyval(k_kink_new_2, coefs_2_kink)

if len(fast_kink_branch3_omega) > 1:
  FSB3_kink_phase = fast_kink_branch3_omega/fast_kink_branch3_k
  FSB3_kink_k = fast_kink_branch3_k
  k_kink_new_3 = np.linspace(3.43, FSB3_kink_k[-1], num=len(FSB3_kink_k)*10)   #FSB2_kink_k[-1]
  
  coefs_3_kink = poly.polyfit(FSB3_kink_k, FSB3_kink_phase, 2)
  ffit_3_kink = poly.polyval(k_kink_new_3, coefs_3_kink)


#################################################



###################################################

slow_fast_kink_branch1_omega = []
slow_fast_kink_branch1_k = []
slow_fast_kink_branch2_omega = []
slow_fast_kink_branch2_k = []
slow_fast_kink_branch3_omega = []
slow_fast_kink_branch3_k = []
##################################################
for i in range(len(slow_fast_body_kink_omega)):   #sausage mode
  v_phase = slow_fast_body_kink_omega[i]/slow_fast_body_kink_k[i]  
  
    
  if slow_fast_body_kink_omega[i]/slow_fast_body_kink_k[i] > -3.:  
      slow_fast_kink_branch1_omega.append(slow_fast_body_kink_omega[i])
      slow_fast_kink_branch1_k.append(slow_fast_body_kink_k[i])
         
  
  if slow_fast_body_kink_k[i] < 3.5 and slow_fast_body_kink_omega[i]/slow_fast_body_kink_k[i] < -3.:  
      slow_fast_kink_branch2_omega.append(slow_fast_body_kink_omega[i])
      slow_fast_kink_branch2_k.append(slow_fast_body_kink_k[i])

  elif slow_fast_body_kink_k[i] > 3.5 and v_phase > -4. and v_phase < -3.: 
      slow_fast_kink_branch2_omega.append(slow_fast_body_kink_omega[i])
      slow_fast_kink_branch2_k.append(slow_fast_body_kink_k[i])  
  
  elif slow_fast_body_kink_k[i] > 3.5 and v_phase < -4.: 
      slow_fast_kink_branch3_omega.append(slow_fast_body_kink_omega[i])
      slow_fast_kink_branch3_k.append(slow_fast_body_kink_k[i])  
  
##############################################

slow_fast_kink_branch1_omega = slow_fast_kink_branch1_omega[::-1]
slow_fast_kink_branch1_k = slow_fast_kink_branch1_k[::-1]

index_to_remove = []
for i in range(len(slow_fast_kink_branch1_omega)-1):
    ph_diff = abs((slow_fast_kink_branch1_omega[i+1]/slow_fast_kink_branch1_k[i+1]) - (slow_fast_kink_branch1_omega[i]/slow_fast_kink_branch1_k[i]))
   
    if ph_diff > 0.2:
      index_to_remove.append(i+1)

slow_fast_kink_branch1_omega = np.delete(slow_fast_kink_branch1_omega, index_to_remove)    
slow_fast_kink_branch1_k = np.delete(slow_fast_kink_branch1_k, index_to_remove) 

slow_fast_kink_branch1_omega = [x for _,x in sorted(zip(slow_fast_kink_branch1_k,slow_fast_kink_branch1_omega))]
slow_fast_kink_branch1_k = np.sort(slow_fast_kink_branch1_k)



slow_fast_kink_branch2_omega = slow_fast_kink_branch2_omega[::-1]
slow_fast_kink_branch2_k = slow_fast_kink_branch2_k[::-1]

index_to_remove = []
for i in range(len(slow_fast_kink_branch2_omega)-1):
    ph_diff = abs((slow_fast_kink_branch2_omega[i+1]/slow_fast_kink_branch2_k[i+1]) - (slow_fast_kink_branch2_omega[i]/slow_fast_kink_branch2_k[i]))
   
    if ph_diff > 0.2:
      index_to_remove.append(i+1)

slow_fast_kink_branch2_omega = np.delete(slow_fast_kink_branch2_omega, index_to_remove)    
slow_fast_kink_branch2_k = np.delete(slow_fast_kink_branch2_k, index_to_remove) 

slow_fast_kink_branch2_omega = [x for _,x in sorted(zip(slow_fast_kink_branch2_k,slow_fast_kink_branch2_omega))]
slow_fast_kink_branch2_k = np.sort(slow_fast_kink_branch2_k)

############################################  
         
slow_fast_kink_branch1_omega = np.array(slow_fast_kink_branch1_omega)
slow_fast_kink_branch1_k = np.array(slow_fast_kink_branch1_k)
slow_fast_kink_branch2_omega = np.array(slow_fast_kink_branch2_omega)
slow_fast_kink_branch2_k = np.array(slow_fast_kink_branch2_k)



############################################


slow_body_sausage_branch1_omega = []
slow_body_sausage_branch1_k = []
slow_body_sausage_branch2_omega = []
slow_body_sausage_branch2_k = []


#################################################
for i in range(len(slow_body_sausage_omega)):   #sausage mode
  v_phase = slow_body_sausage_omega[i]/slow_body_sausage_k[i]
  
  if v_phase > 0.94 and v_phase < c_i0 and slow_body_sausage_k[i] > 1.4:
      slow_body_sausage_branch1_omega.append(slow_body_sausage_omega[i])
      slow_body_sausage_branch1_k.append(slow_body_sausage_k[i])

  elif v_phase > cT_i0 and v_phase < 0.94 and slow_body_sausage_k[i] < 1.4:
      slow_body_sausage_branch1_omega.append(slow_body_sausage_omega[i])
      slow_body_sausage_branch1_k.append(slow_body_sausage_k[i])
      
  elif v_phase < 0.94 and v_phase > cT_i0 and slow_body_sausage_k[i] > 1.4:
      slow_body_sausage_branch2_omega.append(slow_body_sausage_omega[i])
      slow_body_sausage_branch2_k.append(slow_body_sausage_k[i])

slow_body_sausage_branch1_omega = slow_body_sausage_branch1_omega[::-1]
slow_body_sausage_branch1_k = slow_body_sausage_branch1_k[::-1]


index_to_remove = []
for i in range(len(slow_body_sausage_branch1_omega)-1):
    ph_diff = abs((slow_body_sausage_branch1_omega[i+1]/slow_body_sausage_branch1_k[i+1]) - (slow_body_sausage_branch1_omega[i]/slow_body_sausage_branch1_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i+1)
      slow_body_sausage_branch2_omega.append(slow_body_sausage_omega[i+1])
      slow_body_sausage_branch2_k.append(slow_body_sausage_k[i+1])                     

slow_body_sausage_branch1_omega = np.delete(slow_body_sausage_branch1_omega, index_to_remove)    
slow_body_sausage_branch1_k = np.delete(slow_body_sausage_branch1_k, index_to_remove) 


index_to_remove = []
for i in range(len(slow_body_sausage_branch2_omega)-1):
    ph_diff = abs((slow_body_sausage_branch2_omega[i+1]/slow_body_sausage_branch2_k[i+1]) - (slow_body_sausage_branch2_omega[i]/slow_body_sausage_branch2_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i+1)
            
slow_body_sausage_branch2_omega = np.delete(slow_body_sausage_branch2_omega, index_to_remove)    
slow_body_sausage_branch2_k = np.delete(slow_body_sausage_branch2_k, index_to_remove)   
############################################################################# 
      
slow_body_sausage_branch1_omega = np.array(slow_body_sausage_branch1_omega)
slow_body_sausage_branch1_k = np.array(slow_body_sausage_branch1_k)
slow_body_sausage_branch2_omega = np.array(slow_body_sausage_branch2_omega)
slow_body_sausage_branch2_k = np.array(slow_body_sausage_branch2_k)

###################################################

#################################################

slow_body_kink_branch1_omega = []
slow_body_kink_branch1_k = []
slow_body_kink_branch2_omega = []
slow_body_kink_branch2_k = []


#################################################
for i in range(len(slow_body_kink_omega)):   #kink mode
  v_phase = slow_body_kink_omega[i]/slow_body_kink_k[i]
  
  if v_phase > cT_i0 and v_phase < c_i0:
      slow_body_kink_branch1_omega.append(slow_body_kink_omega[i])
      slow_body_kink_branch1_k.append(slow_body_kink_k[i])
      

index_to_remove = []
for i in range(len(slow_body_kink_branch1_omega)-1):
    ph_diff = abs((slow_body_kink_branch1_omega[i+1]/slow_body_kink_branch1_k[i+1]) - (slow_body_kink_branch1_omega[i]/slow_body_kink_branch1_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i+1)
      slow_body_kink_branch2_omega.append(slow_body_kink_omega[i+1])
      slow_body_kink_branch2_k.append(slow_body_kink_k[i+1])              

       

slow_body_kink_branch1_omega = np.delete(slow_body_kink_branch1_omega, index_to_remove)    
slow_body_kink_branch1_k = np.delete(slow_body_kink_branch1_k, index_to_remove) 

slow_body_kink_branch1_omega = slow_body_kink_branch1_omega[::-1]
slow_body_kink_branch1_k = slow_body_kink_branch1_k[::-1]


index_to_remove = []
for i in range(len(slow_body_kink_branch1_omega)-1):
    ph_diff = abs((slow_body_kink_branch1_omega[i+1]/slow_body_kink_branch1_k[i+1]) - (slow_body_kink_branch1_omega[i]/slow_body_kink_branch1_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i)

slow_body_kink_branch1_omega = np.delete(slow_body_kink_branch1_omega, index_to_remove)    
slow_body_kink_branch1_k = np.delete(slow_body_kink_branch1_k, index_to_remove) 

slow_body_kink_branch1_omega = [x for _,x in sorted(zip(slow_body_kink_branch1_k,slow_body_kink_branch1_omega))]
slow_body_kink_branch1_k = np.sort(slow_body_kink_branch1_k)
   
############################################################################# 
      
slow_body_kink_branch1_omega = np.array(slow_body_kink_branch1_omega)
slow_body_kink_branch1_k = np.array(slow_body_kink_branch1_k)
slow_body_kink_branch2_omega = np.array(slow_body_kink_branch2_omega)
slow_body_kink_branch2_k = np.array(slow_body_kink_branch2_k)

###################################################



##############################################
#index_to_remove = []
#for i in range(len(slow_backward_body_sausage_omega)-1):
#    ph_diff = abs((slow_backward_body_sausage_omega[i+1]/slow_backward_body_sausage_k[i+1])-(slow_backward_body_sausage_omega[i]/slow_backward_body_sausage_k[i]))
#   
#    if ph_diff > 0.01:
#      index_to_remove.append(i+1)
#
#slow_backward_body_sausage_omega = np.delete(slow_backward_body_sausage_omega, index_to_remove)    
#slow_backward_body_sausage_k = np.delete(slow_backward_body_sausage_k, index_to_remove) 
#############################################  
#
###############################################
#index_to_remove = []
#for i in range(len(slow_backward_body_kink_omega)-1):
#    ph_diff = abs((slow_backward_body_kink_omega[i+1]/slow_backward_body_kink_k[i+1])-(slow_backward_body_kink_omega[i]/slow_backward_body_kink_k[i]))
#   
#    if ph_diff > 0.01:
#      index_to_remove.append(i+1)
#
#slow_backward_body_kink_omega = np.delete(slow_backward_body_kink_omega, index_to_remove)    
#slow_backward_body_kink_k = np.delete(slow_backward_body_kink_k, index_to_remove) 
############################################# 
#
#slow_backward_body_sausage_omega = np.array(slow_backward_body_sausage_omega)
#slow_backward_body_sausage_k = np.array(slow_backward_body_sausage_k)
#slow_backward_body_kink_omega = np.array(slow_backward_body_kink_omega)
#slow_backward_body_kink_k = np.array(slow_backward_body_kink_k)
#
#
############################################################################################
#
#if len(slow_backward_body_sausage_omega) > 1:
#  SBb_phase = slow_backward_body_sausage_omega/slow_backward_body_sausage_k
#  SBb_k = slow_backward_body_sausage_k
#  sbb_k_new = np.linspace(1.6, SBb_k[-1], num=len(SBb_k)*10)  #FSB2_k[-1]
#  
#  sbb_coefs = poly.polyfit(SBb_k, SBb_phase, 6)    # 6 order     # 1st order for W < 1.5
#  sbb_ffit = poly.polyval(sbb_k_new, sbb_coefs)
#
#
#if len(slow_backward_body_kink_omega) > 1:
#  SBbk_phase = slow_backward_body_kink_omega/slow_backward_body_kink_k
#  SBbk_k = slow_backward_body_kink_k
#  sbbk_k_new = np.linspace(3.34, SBbk_k[-1], num=len(SBbk_k)*10)  #SBk_k[-1]
#  
#  sbbk_coefs = poly.polyfit(SBbk_k, SBbk_phase, 6)    # 6 order     # 1st order for W < 1.5
#  sbbk_ffit = poly.polyval(sbbk_k_new, sbbk_coefs)
#

###########################################################################################

###########################################################################################

if len(slow_body_sausage_branch1_omega) > 1:
  sb_phase = slow_body_sausage_branch1_omega/slow_body_sausage_branch1_k
  sb_k = slow_body_sausage_branch1_k
  sb_k_new = np.linspace(sb_k[-1], Kmax, num=len(sb_k)*10)  #FSB2_k[-1]
  
  sb_coefs = poly.polyfit(sb_k, sb_phase, 6)    # 6 order     # 1st order for W < 1.5
  sb_ffit = poly.polyval(sb_k_new, sb_coefs)


if len(slow_body_sausage_branch2_omega) > 1:
  sb2_phase = slow_body_sausage_branch2_omega/slow_body_sausage_branch2_k
  sb2_k = slow_body_sausage_branch2_k
  sb2_k_new = np.linspace(sb2_k[0], Kmax, num=len(sb2_k)*10)  #FSB2_k[-1]
  
  sb2_coefs = poly.polyfit(sb2_k, sb2_phase, 1)    # 6 order     # 1st order for W < 1.5
  sb2_ffit = poly.polyval(sb2_k_new, sb2_coefs)




if len(slow_body_kink_branch1_omega) > 1:
  sbk_phase = slow_body_kink_branch1_omega/slow_body_kink_branch1_k
  sbk_k = slow_body_kink_branch1_k
  sbk_k_new = np.linspace(0., sbk_k[-1], num=len(sbk_k)*10)  #SBk_k[-1]
  
  sbk_coefs = poly.polyfit(sbk_k, sbk_phase, 6)    # 6 order     # 1st order for W < 1.5
  sbk_ffit = poly.polyval(sbk_k_new, sbk_coefs)


if len(slow_body_kink_branch2_omega) > 1:
  sbk2_phase = slow_body_kink_branch2_omega/slow_body_kink_branch2_k
  sbk2_k = slow_body_kink_branch2_k
  sbk2_k_new = np.linspace(0., sbk2_k[-1], num=len(sbk2_k)*10)  #SBk_k[-1]
  
  sbk2_coefs = poly.polyfit(sbk2_k, sbk2_phase, 6)    # 6 order     # 1st order for W < 1.5
  sbk2_ffit = poly.polyval(sbk2_k_new, sbk2_coefs)

###########################################################################################
#
#
##########################################################################   sausage polyfit
#SFSB1_phase = slow_fast_sausage_branch1_omega/slow_fast_sausage_branch1_k
#SFSB1_k = slow_fast_sausage_branch1_k
#SFk_new = np.linspace(1.2, SFSB1_k[-1], num=len(SFSB1_k)*10)
#
#SFcoefs = poly.polyfit(SFSB1_k, SFSB1_phase, 6)   #  # 6 is good
#SFffit = poly.polyval(SFk_new, SFcoefs)
#
#
#if len(slow_fast_sausage_branch2_omega) > 1:
#  SFSB2_phase = slow_fast_sausage_branch2_omega/slow_fast_sausage_branch2_k
#  SFSB2_k = slow_fast_sausage_branch2_k
#  SFk_new_2 = np.linspace(2.7, SFSB2_k[-1], num=len(SFSB2_k)*10)  #FSB2_k[-1]
#  
#  SFcoefs_2 = poly.polyfit(SFSB2_k, SFSB2_phase, 6)    # 6 order     # 1st order for W < 1.5
#  SFffit_2 = poly.polyval(SFk_new_2, SFcoefs_2)
#
#
##########################################################################   kink polyfit
#SFSB1_kink_phase = slow_fast_kink_branch1_omega/slow_fast_kink_branch1_k
#SFSB1_kink_k = slow_fast_kink_branch1_k
#SFk_kink_new = np.linspace(SFSB1_kink_k[0], SFSB1_kink_k[-1], num=len(SFSB1_kink_k)*10)   #FSB1_kink_k[-1]
#
#SFcoefs_kink = poly.polyfit(SFSB1_kink_k, SFSB1_kink_phase, 6) # 3 order for messing
#SFffit_kink = poly.polyval(SFk_kink_new, SFcoefs_kink)
#
#if len(slow_fast_kink_branch2_omega) > 1:
#  SFSB2_kink_phase = slow_fast_kink_branch2_omega/slow_fast_kink_branch2_k
#  SFSB2_kink_k = slow_fast_kink_branch2_k
#  SFk_kink_new_2 = np.linspace(1.8, SFSB2_kink_k[-1], num=len(SFSB2_kink_k)*10)   #FSB2_kink_k[-1]
#  
#  SFcoefs_2_kink = poly.polyfit(SFSB2_kink_k, SFSB2_kink_phase, 4)
#  SFffit_2_kink = poly.polyval(SFk_kink_new_2, SFcoefs_2_kink)
#


#################################################


###########################################################################################
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
ax.plot(fast_body_sausage_k, fast_body_sausage_omega, 'r.', markersize=4.)   #fast body sausage
ax.plot(fast_body_kink_k, fast_body_kink_omega, 'b.', markersize=4.)   #fast body kink
#ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega, 'r.', markersize=4.)   #fast body branch 1
#ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega, 'r.', markersize=4.)   #fast body branch 2
#ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega, 'r.', markersize=4.)   #fast body branch 3
ax.plot(test_k_plot, vA_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, vA_i_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, c_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, c_i_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, cT_e_plot, linestyle='dashdot', color='k')
ax.plot(test_k_plot, cT_i_plot, linestyle='dashdot', color='k')
#ax.plot(slow_body_sausage_k, slow_body_sausage_omega, 'b.', markersize=4.)   #slow body sausage
#ax.plot(slow_body_kink_k, slow_body_kink_omega, 'b.', markersize=4.)   #slow body kink

ax.annotate( '$c_{Ti}$', xy=(Kmax, cT_i_plot[-1]), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e_plot[-1]), fontsize=20)
ax.annotate( '$c_{e},    c_{Te}$', xy=(Kmax, c_e_plot[-1]), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Kmax, c_i_plot[-1]), fontsize=20)
ax.annotate( '$v_{Ae}$', xy=(Kmax, vA_e_plot[-1]), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(Kmax, vA_i_plot[-1]), fontsize=20)
ax.axhline(y=cutoff, color='r', linestyle='dashed', label='_nolegend_')
ax.axhline(y=cutoff2, color='r', linestyle='dashed', label='_nolegend_')
ax.annotate( '$\omega = {}$'.format(cutoff2), xy=(Kmax, cutoff2), fontsize=18, color='r')
ax.annotate( '$\omega = {}$'.format(cutoff), xy=(Kmax, cutoff), fontsize=18, color='r')
#ax.axhline(y=cutoff3, color='r', linestyle='dashed', label='_nolegend_')
#ax.annotate( '$\omega = {}$'.format(cutoff3), xy=(Kmax, cutoff3), fontsize=18, color='r')
ax.axhline(y=cutoff_kink, color='b', linestyle='dashed', label='_nolegend_')
ax.axhline(y=cutoff2_kink, color='b', linestyle='dashed', label='_nolegend_')
ax.annotate( '$\omega = {}$'.format(cutoff2_kink), xy=(Kmax, cutoff2_kink), fontsize=18, color='b')
ax.annotate( '$\omega = {}$'.format(cutoff_kink), xy=(Kmax, cutoff_kink), fontsize=18, color='b')


########################################################################################################


########################################################################################################

fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for zoom to remove blank space on plot

ax2.set_title("$ W = 0.9$")

plt.xlabel("$ka$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)
#ax.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'b.', markersize=4.)

#ax2.plot(sol_ks09, (sol_omegas09/sol_ks09), 'r.', markersize=4.)
#ax2.plot(sol_ks_kink09, (sol_omegas_kink09/sol_ks_kink09), 'b.', markersize=4.)

#ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega/fast_sausage_branch1_k, 'y.', markersize=4.)
#ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega/fast_sausage_branch2_k, 'k.', markersize=4.)
#ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega/fast_sausage_branch3_k, 'r.', markersize=4.)

ax.plot(k_new, ffit, color='r')
ax.plot(k_new_2, ffit_2, color='r')
#ax.plot(k_new_3, ffit_3, color='r')

ax2.plot(k_new, ffit, color='r')
ax2.plot(k_new_2, ffit_2, color='r')


#ax.plot(fast_kink_branch1_k, fast_kink_branch1_omega/fast_kink_branch1_k, 'b.', markersize=4.)
#ax.plot(fast_kink_branch2_k, fast_kink_branch2_omega/fast_kink_branch2_k, 'g.', markersize=4.)

ax.plot(k_kink_new, ffit_kink, color='b')
ax.plot(k_kink_new_2, ffit_2_kink, color='b')
#ax.plot(k_kink_new_3, ffit_3_kink, color='b')

ax2.plot(k_kink_new, ffit_kink, color='b')
ax2.plot(k_kink_new_2, ffit_2_kink, color='b')
#ax2.plot(k_kink_new_3, ffit_3_kink, color='b')


#ax.plot(slow_body_sausage_k, (slow_body_sausage_omega/slow_body_sausage_k), 'r.', markersize=4.)
#ax.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)

#ax.plot(slow_body_sausage_branch1_k, slow_body_sausage_branch1_omega/slow_body_sausage_branch1_k, 'y.', markersize=4.)   # body sausage
#ax.plot(slow_body_kink_branch1_k, slow_body_kink_branch1_omega/slow_body_kink_branch1_k, 'y.', markersize=4.)   # body kink
#ax.plot(slow_body_sausage_branch2_k, slow_body_sausage_branch2_omega/slow_body_sausage_branch2_k, 'g.', markersize=4.)   # body sausage
#ax.plot(slow_body_kink_branch2_k, slow_body_kink_branch2_omega/slow_body_kink_branch2_k, 'g.', markersize=4.)   # body kink

#ax.plot(sb_k_new, sb_ffit, color='r')
#ax.plot(sb2_k_new, sb2_ffit, color='r')

#ax.plot(sbk_k_new, sbk_ffit, color='b')
#ax.plot(sbk2_k_new, sbk2_ffit, color='b')

#ax.plot(sbb_k_new, sbb_ffit, color='r')
#ax.plot(sbbk_k_new, sbbk_ffit, color='b')

#ax.plot(new_s_k_new, new_s_ffit, color='r')
#ax.plot(newK_k_new, newK_ffit, color='b')

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


ax.annotate( ' $c_{TB}$', xy=(Kmax, cT_bound), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax.annotate( ' $c_{e}, c_{Te}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{B}$', xy=(Kmax, c_bound), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=20)
ax.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax.annotate( ' $c_{kB}$', xy=(Kmax, c_kink_bound), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)


ax2.annotate( ' $c_{TB}$', xy=(Kmax, cT_bound), fontsize=20)
#ax2.annotate( '$c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax2.annotate( ' $c_{e}, c_{Te}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( ' $c_{B}$', xy=(Kmax, c_bound), fontsize=20)
ax2.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax2.annotate( ' $v_{AB}$', xy=(Kmax, vA_bound), fontsize=20)
ax2.annotate( ' $c_{k}$', xy=(Kmax, c_kink), fontsize=20)
ax2.annotate( ' $c_{kB}$', xy=(Kmax, c_kink_bound), fontsize=20)
ax2.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)


#ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-c_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-c_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-cT_e, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
#ax.axhline(y=-c_kink, color='k', label='$c_{k}$', linestyle='dashdot')
#
#ax.annotate( '$-c_{Ti}$', xy=(Kmax, -cT_i0), fontsize=20)
##ax.annotate( '$c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
#ax.annotate( '$-c_{e}, -c_{Te}$', xy=(Kmax, -c_e), fontsize=20)
#ax.annotate( '$-c_{i}$', xy=(Kmax, -c_i0), fontsize=20)
#ax.annotate( '$-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
#ax.annotate( '$-v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)
#ax.annotate( '$-c_{k}$', xy=(Kmax, -c_kink), fontsize=20)


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



ax.set_ylim(0.875, 1.1)
ax2.set_ylim(1.95, 5.05)

ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()
#ax2.set_yticks([])
#ax.set_yticks([])

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
ax.set_position([box.x0, box.y0, box.width*0.85, box.height*0.5])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0-0.2, box2.width*0.85, box2.height*1.55])


plt.savefig("Cylinder_width_09_coronal_curves.png")

plt.show()
exit()

