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
from scipy.optimize import fsolve
from scipy.integrate import odeint

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

Kmax = 4.

ix = np.linspace(-1., -0.001, 1e3)  # inside slab x values


rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

r0=0.  #mean
dr=0.6 #standard dev



U_i0 = 0.05*c_i0     #0.35*vA_i  coronal    
U_e = 0.          #-0.15*vA_i   photospheric      0 - coronal


def v_z(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2))) 



################################################

def rho_i(r):                  
    return rho_i0

#############################################

def P_i(r):
    return ((c_i(r))**2*rho_i(r)/gamma)

#############################################

def T_i(r):     #use this for constant P
    return (P_i(r)/rho_i(r))

##########################################

def vA_i(r):                  # consatnt temp
    return (B_i(r)+B_iphi(r))/(sym.sqrt(rho_i(r)))

##########################################

def B_i(r):  #constant rho
    return (B_0*sym.sqrt(1. - 2*(B_iphi(r)**2/B_0**2)))


B_twist = 0.01

def B_iphi(r):  #constant rho
    return 0.    #B_twist*r   #0.   #B_twist*r


v_twist = 0.1
def v_iphi(r):  #constant rho
    return 0.     #v_twist*r    #0.   #v_twist*r   #0.

###################################

def PT_i(r):   # USE THIS ONE
    return (sym.diff((P_i(r) + (B_i(r)**2 + B_iphi(r)**2)/2.), r) + B_iphi(r)**2/r)

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
v_z_np=sym.lambdify(rr,v_z(rr),"numpy")   #In order to evaluate we need to switch to numpy



flow_bound = v_z_np(-1)

plt.figure()
plt.title("width = 1.5")
#plt.xlabel("x")
#plt.ylabel("$\u03C1_{i}$",fontsize=25)
ax = plt.subplot(111)
ax.title.set_text("v_z")
ax.plot(ix,v_z_np(ix));
ax.plot(-ix,v_z_np(-ix));
ax.annotate( '$U_{e}$', xy=(1, U_e),fontsize=25)
ax.annotate( '$U_{i}$', xy=(1, U_i0),fontsize=25)
ax.axhline(y=U_i0, color='k', label='$U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label='$U_{e}$', linestyle='dashdot')
#ax.set_ylim(0., 1.1)

#plt.show()
#exit()


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_06.pickle', 'rb') as f:
    sol_omegas06, sol_ks06, sol_omegas_kink06, sol_ks_kink06 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas06 = [x for _,x in sorted(zip(sol_ks06,sol_omegas06))]
sol_ks06 = np.sort(sol_ks06)

sol_omegas06 = np.array(sol_omegas06)
sol_ks06 = np.array(sol_ks06)

sol_omegas_kink06 = [x for _,x in sorted(zip(sol_ks_kink06,sol_omegas_kink06))]
sol_ks_kink06 = np.sort(sol_ks_kink06)

sol_omegas_kink06 = np.array(sol_omegas_kink06)
sol_ks_kink06 = np.array(sol_ks_kink06)



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



for i in range(len(sol_ks06)):   #sausage mode
  v_phase = sol_omegas06[i]/sol_ks06[i]
      
  if v_phase > vA_i0 and v_phase < vA_e:  
      fast_body_sausage_omega.append(sol_omegas06[i])
      fast_body_sausage_k.append(sol_ks06[i])
      
  if v_phase > cT_i0+U_i0 and v_phase < c_i0+U_i0:  
      slow_body_sausage_omega.append(sol_omegas06[i])
      slow_body_sausage_k.append(sol_ks06[i])

  if v_phase < -cT_i0 and v_phase > -c_i0 + U_i0: 
      slow_backward_body_sausage_omega.append(sol_omegas06[i])
      slow_backward_body_sausage_k.append(sol_ks06[i])

  if v_phase > -vA_e and v_phase < -vA_i0:
      slow_fast_body_sausage_omega.append(sol_omegas06[i])
      slow_fast_body_sausage_k.append(sol_ks06[i])
            
  else:
      new_modes_sausage_k.append(sol_ks06[i])
      new_modes_sausage_omega.append(sol_omegas06[i])
      



for i in range(len(sol_ks_kink06)):   #kink mode
  v_phase = sol_omegas_kink06[i]/sol_ks_kink06[i]
      
  if v_phase > vA_i0 and v_phase < vA_e: 
      fast_body_kink_omega.append(sol_omegas_kink06[i])
      fast_body_kink_k.append(sol_ks_kink06[i])
            
  if v_phase > cT_i0+U_i0 and v_phase < c_i0: 
      slow_body_kink_omega.append(sol_omegas_kink06[i])
      slow_body_kink_k.append(sol_ks_kink06[i])

  if v_phase < -cT_i0 and v_phase > -c_i0+U_i0:
      slow_backward_body_kink_omega.append(sol_omegas_kink06[i])
      slow_backward_body_kink_k.append(sol_ks_kink06[i])

  if v_phase > -vA_e and v_phase < -vA_i0:
      slow_fast_body_kink_omega.append(sol_omegas_kink06[i])
      slow_fast_body_kink_k.append(sol_ks_kink06[i])
            
  else:
      new_modes_kink_k.append(sol_ks_kink06[i])
      new_modes_kink_omega.append(sol_omegas_kink06[i])



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


cutoff = 13.  
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




cutoff_kink = 9. 
cutoff_kink2 = 15. 


##################################################   kink mode
fast_kink_branch1_omega = []
fast_kink_branch1_k = []
fast_kink_branch2_omega = []
fast_kink_branch2_k = []
fast_kink_branch3_omega = []
fast_kink_branch3_k = []
##################################################
for i in range(len(fast_body_kink_omega)):   #sausage mode
      
  if fast_body_kink_omega[i] > cutoff_kink and fast_body_kink_omega[i] < cutoff_kink2:  
      fast_kink_branch1_omega.append(fast_body_kink_omega[i])
      fast_kink_branch1_k.append(fast_body_kink_k[i])
         
  elif fast_body_kink_omega[i] < cutoff_kink and fast_body_kink_omega[i]/fast_body_kink_k[i] > vA_i0:  
      fast_kink_branch2_omega.append(fast_body_kink_omega[i])
      fast_kink_branch2_k.append(fast_body_kink_k[i])

  elif fast_body_kink_omega[i] > cutoff_kink2:  
      fast_kink_branch3_omega.append(fast_body_kink_omega[i])
      fast_kink_branch3_k.append(fast_body_kink_k[i])


index_to_remove = []
for i in range(len(fast_kink_branch2_omega)-1):
    ph_diff = abs((fast_kink_branch2_omega[i+1]/fast_kink_branch2_k[i+1]) - (fast_kink_branch2_omega[i]/fast_kink_branch2_k[i]))
   
    if ph_diff > 0.05:
      index_to_remove.append(i+1)

fast_kink_branch2_omega = np.delete(fast_kink_branch2_omega, index_to_remove)    
fast_kink_branch2_k = np.delete(fast_kink_branch2_k, index_to_remove) 



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
  k_new = np.linspace(2.8, Kmax, num=len(FSB1_k)*4)
  
  coefs = poly.polyfit(FSB1_k, FSB1_phase, 3)   #  # 6 is good
  ffit = poly.polyval(k_new, coefs)

if len(fast_sausage_branch2_omega) > 1:
  FSB2_phase = fast_sausage_branch2_omega/fast_sausage_branch2_k
  FSB2_k = fast_sausage_branch2_k
  k_new_2 = np.linspace(1.2, FSB2_k[-1], num=len(FSB2_k)*10)  #FSB2_k[-1]
  
  coefs_2 = poly.polyfit(FSB2_k, FSB2_phase, 6)    # 6 order     # 1st order for W < 1.5
  ffit_2 = poly.polyval(k_new_2, coefs_2)

#########################################################################   kink polyfit

if len(fast_kink_branch1_omega) > 1:
  FSB1_kink_phase = fast_kink_branch1_omega/fast_kink_branch1_k
  FSB1_kink_k = fast_kink_branch1_k
  k_kink_new = np.linspace(1.8, FSB1_kink_k[-1], num=len(FSB1_kink_k)*10)   #FSB1_kink_k[-1]
  
  coefs_kink = poly.polyfit(FSB1_kink_k, FSB1_kink_phase, 4) # 3 order for messing
  ffit_kink = poly.polyval(k_kink_new, coefs_kink)

if len(fast_kink_branch2_omega) > 1:
  FSB2_kink_phase = fast_kink_branch2_omega/fast_kink_branch2_k
  FSB2_kink_k = fast_kink_branch2_k
  k_kink_new_2 = np.linspace(FSB2_kink_k[0], Kmax, num=len(FSB2_kink_k)*10)   #FSB2_kink_k[-1]
  
  coefs_2_kink = poly.polyfit(FSB2_kink_k, FSB2_kink_phase, 8)
  ffit_2_kink = poly.polyval(k_kink_new_2, coefs_2_kink)


if len(fast_kink_branch3_omega) > 1:
  FSB3_kink_phase = fast_kink_branch3_omega/fast_kink_branch3_k
  FSB3_kink_k = fast_kink_branch3_k
  k_kink_new_3 = np.linspace(3.5, FSB3_kink_k[-1], num=len(FSB3_kink_k)*10)   #FSB3_kink_k[-1]
  
  coefs_3_kink = poly.polyfit(FSB3_kink_k, FSB3_kink_phase, 4)
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


slow_body_sausage_branch1_omega = [0.1*(cT_i0+U_i0)]
slow_body_sausage_branch1_k = [0.1]
slow_body_sausage_branch2_omega = []
slow_body_sausage_branch2_k = []


#################################################
for i in range(len(slow_body_sausage_omega)):   #sausage mode
  v_phase = slow_body_sausage_omega[i]/slow_body_sausage_k[i]

  if v_phase > 0.975 and v_phase < c_i0+0.01 and slow_body_sausage_k[i] < 4.:
      slow_body_sausage_branch1_omega.append(slow_body_sausage_omega[i])
      slow_body_sausage_branch1_k.append(slow_body_sausage_k[i])
   
  if v_phase > cT_i0+U_i0 and v_phase < c_i0+0.01 and slow_body_sausage_k[i] < 1.5:
      slow_body_sausage_branch1_omega.append(slow_body_sausage_omega[i])
      slow_body_sausage_branch1_k.append(slow_body_sausage_k[i])
      
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
  
#  if v_phase > cT_i0+U_i0 and v_phase < c_i0+0.01:
  if v_phase > 0.97 and slow_body_kink_k[i] > 3.2:
      slow_body_kink_branch1_omega.append(slow_body_kink_omega[i])
      slow_body_kink_branch1_k.append(slow_body_kink_k[i])
      
  elif v_phase < 0.97 and slow_body_kink_k[i] < 3.2:
      slow_body_kink_branch1_omega.append(slow_body_kink_omega[i])
      slow_body_kink_branch1_k.append(slow_body_kink_k[i])


index_to_remove = []
for i in range(len(slow_body_kink_branch1_omega)-1):
    ph_diff = abs((slow_body_kink_branch1_omega[i+1]/slow_body_kink_branch1_k[i+1]) - (slow_body_kink_branch1_omega[i]/slow_body_kink_branch1_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i+1)
      slow_body_kink_branch2_omega.append(slow_body_kink_branch1_omega[i+1])
      slow_body_kink_branch2_k.append(slow_body_kink_branch1_k[i+1])              

       

slow_body_kink_branch1_omega = np.delete(slow_body_kink_branch1_omega, index_to_remove)    
slow_body_kink_branch1_k = np.delete(slow_body_kink_branch1_k, index_to_remove) 

slow_body_kink_branch1_omega = slow_body_kink_branch1_omega[::-1]
slow_body_kink_branch1_k = slow_body_kink_branch1_k[::-1]


index_to_remove = []
for i in range(len(slow_body_kink_branch1_omega)-1):
    ph_diff = abs((slow_body_kink_branch1_omega[i+1]/slow_body_kink_branch1_k[i+1]) - (slow_body_kink_branch1_omega[i]/slow_body_kink_branch1_k[i]))
   
    if ph_diff > 0.02:
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
index_to_remove = []
for i in range(len(slow_backward_body_sausage_omega)):
    ph_speed = (slow_backward_body_sausage_omega[i]/slow_backward_body_sausage_k[i])
    if ph_speed > -0.92 and slow_backward_body_sausage_k[i] > 2.5:
      index_to_remove.append(i)

    
slow_backward_body_sausage_omega = np.delete(slow_backward_body_sausage_omega, index_to_remove)    
slow_backward_body_sausage_k = np.delete(slow_backward_body_sausage_k, index_to_remove) 


index_to_remove = []
for i in range(len(slow_backward_body_sausage_omega)-1):
    ph_diff = abs((slow_backward_body_sausage_omega[i+1]/slow_backward_body_sausage_k[i+1])-(slow_backward_body_sausage_omega[i]/slow_backward_body_sausage_k[i]))
    if ph_diff > 0.01:
      index_to_remove.append(i+1)

    
slow_backward_body_sausage_omega = np.delete(slow_backward_body_sausage_omega, index_to_remove)    
slow_backward_body_sausage_k = np.delete(slow_backward_body_sausage_k, index_to_remove) 
############################################  


##############################################
index_to_remove = []
for i in range(len(slow_backward_body_kink_omega)-1):
    ph_diff = abs((slow_backward_body_kink_omega[i+1]/slow_backward_body_kink_k[i+1])-(slow_backward_body_kink_omega[i]/slow_backward_body_kink_k[i]))
   
    if ph_diff > 0.01:
      index_to_remove.append(i+1)

slow_backward_body_kink_omega = np.delete(slow_backward_body_kink_omega, index_to_remove)    
slow_backward_body_kink_k = np.delete(slow_backward_body_kink_k, index_to_remove) 
############################################ 

slow_backward_body_sausage_omega = np.array(slow_backward_body_sausage_omega)
slow_backward_body_sausage_k = np.array(slow_backward_body_sausage_k)
slow_backward_body_kink_omega = np.array(slow_backward_body_kink_omega)
slow_backward_body_kink_k = np.array(slow_backward_body_kink_k)


###########################################################################################

if len(slow_backward_body_sausage_omega) > 1:
  SBb_phase = slow_backward_body_sausage_omega/slow_backward_body_sausage_k
  SBb_k = slow_backward_body_sausage_k
  sbb_k_new = np.linspace(SBb_k[0], 3.245, num=len(SBb_k)*10)  #FSB2_k[-1]
  
  sbb_coefs = poly.polyfit(SBb_k, SBb_phase, 6)    # 6 order     # 1st order for W < 1.5
  sbb_ffit = poly.polyval(sbb_k_new, sbb_coefs)


if len(slow_backward_body_kink_omega) > 1:
  SBbk_phase = slow_backward_body_kink_omega/slow_backward_body_kink_k
  SBbk_k = slow_backward_body_kink_k
  sbbk_k_new = np.linspace(SBbk_k[0], SBbk_k[-1], num=len(SBbk_k)*10)  #SBk_k[-1]
  
  sbbk_coefs = poly.polyfit(SBbk_k, SBbk_phase, 6)    # 6 order     # 1st order for W < 1.5
  sbbk_ffit = poly.polyval(sbbk_k_new, sbbk_coefs)


###########################################################################################

###########################################################################################

if len(slow_body_sausage_branch1_omega) > 1:
  sb_phase = slow_body_sausage_branch1_omega/slow_body_sausage_branch1_k
  sb_k = slow_body_sausage_branch1_k
  sb_k_new = np.linspace(0.4, 3.55, num=len(sb_k)*10)  #FSB2_k[-1]
  
  sb_coefs = poly.polyfit(sb_k, sb_phase, 4)    # 6 order     # 1st order for W < 1.5
  sb_ffit = poly.polyval(sb_k_new, sb_coefs)



if len(slow_body_kink_branch1_omega) > 1:
  sbk_phase = slow_body_kink_branch1_omega/slow_body_kink_branch1_k
  sbk_k = slow_body_kink_branch1_k
  sbk_k_new = np.linspace(1.83, Kmax, num=len(sbk_k)*10)  #SBk_k[-1]
  
  sbk_coefs = poly.polyfit(sbk_k, sbk_phase, 1)    # 6 order     # 1st order for W < 1.5
  sbk_ffit = poly.polyval(sbk_k_new, sbk_coefs)


###########################################################################################


#########################################################################   sausage polyfit
SFSB1_phase = slow_fast_sausage_branch1_omega/slow_fast_sausage_branch1_k
SFSB1_k = slow_fast_sausage_branch1_k
SFk_new = np.linspace(1.1, SFSB1_k[-1], num=len(SFSB1_k)*10)

SFcoefs = poly.polyfit(SFSB1_k, SFSB1_phase, 6)   #  # 6 is good
SFffit = poly.polyval(SFk_new, SFcoefs)


if len(slow_fast_sausage_branch2_omega) > 1:
  SFSB2_phase = slow_fast_sausage_branch2_omega/slow_fast_sausage_branch2_k
  SFSB2_k = slow_fast_sausage_branch2_k
  SFk_new_2 = np.linspace(2.6, Kmax, num=len(SFSB2_k)*10)  #FSB2_k[-1]
  
  SFcoefs_2 = poly.polyfit(SFSB2_k, SFSB2_phase, 6)    # 6 order     # 1st order for W < 1.5
  SFffit_2 = poly.polyval(SFk_new_2, SFcoefs_2)


#########################################################################   kink polyfit
SFSB1_kink_phase = slow_fast_kink_branch1_omega/slow_fast_kink_branch1_k
SFSB1_kink_k = slow_fast_kink_branch1_k
SFk_kink_new = np.linspace(SFSB1_kink_k[0], SFSB1_kink_k[-1], num=len(SFSB1_kink_k)*10)   #FSB1_kink_k[-1]

SFcoefs_kink = poly.polyfit(SFSB1_kink_k, SFSB1_kink_phase, 6) # 3 order for messing
SFffit_kink = poly.polyval(SFk_kink_new, SFcoefs_kink)

if len(slow_fast_kink_branch2_omega) > 1:
  SFSB2_kink_phase = slow_fast_kink_branch2_omega/slow_fast_kink_branch2_k
  SFSB2_kink_k = slow_fast_kink_branch2_k
  SFk_kink_new_2 = np.linspace(1.7, Kmax, num=len(SFSB2_kink_k)*10)   #FSB2_kink_k[-1]
  
  SFcoefs_2_kink = poly.polyfit(SFSB2_kink_k, SFSB2_kink_phase, 4)
  SFffit_2_kink = poly.polyval(SFk_kink_new_2, SFcoefs_2_kink)



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

ax.plot(sol_ks06, (sol_omegas06), 'r.', markersize=4.)
ax.plot(sol_ks_kink06, (sol_omegas_kink06), 'b.', markersize=4.)


########################################################################################################


########################################################################################################

fig, (ax2, ax) = plt.subplots(2, 1, sharex=True)   #split figure for zoom to remove blank space on plot

ax2.set_title("$ W = 0.6$")

#plt.figure()
#plt.title("$ W = 1e5$")
#ax = plt.subplot(111)
plt.xlabel("$ka$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
#ax.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'b.', markersize=4.)
#ax.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega/fast_sausage_branch1_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega/fast_sausage_branch2_k, 'r.', markersize=4.)
#ax.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega/fast_sausage_branch3_k, 'r.', markersize=4.)
ax.plot(k_new, ffit, color='r')
ax.plot(k_new_2, ffit_2, color='r')
#ax.plot(k_new_3, ffit_3, color='r')
#ax.plot(fast_kink_branch1_k, fast_kink_branch1_omega/fast_kink_branch1_k, 'b.', markersize=4.)
#ax.plot(fast_kink_branch2_k, fast_kink_branch2_omega/fast_kink_branch2_k, 'g.', markersize=4.)
ax.plot(k_kink_new, ffit_kink, color='b')
ax.plot(k_kink_new_2, ffit_2_kink, color='b')
#ax.plot(slow_body_sausage_k, (slow_body_sausage_omega/slow_body_sausage_k), 'r.', markersize=4.)
#ax.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)
#ax.plot(body_sausage_branch1_k, body_sausage_branch1_omega/body_sausage_branch1_k, 'r.', markersize=4.)   # body sausage
#ax.plot(slow_body_kink_branch1_k, slow_body_kink_branch1_omega/slow_body_kink_branch1_k, 'g.', markersize=8.)   # body kink
#ax.plot(body_sausage_branch2_k, body_sausage_branch2_omega/body_sausage_branch2_k, 'g.', markersize=4.)   # body sausage
#ax.plot(body_kink_branch2_k, body_kink_branch2_omega/body_kink_branch2_k, 'b.', markersize=4.)   # body kink
ax.plot(sb_k_new, sb_ffit, color='r')
ax.plot(sbk_k_new, sbk_ffit, color='b')
ax.plot(sbb_k_new, sbb_ffit, color='r')
ax.plot(sbbk_k_new, sbbk_ffit, color='b')
ax.plot(SFk_new, SFffit, color='r')
ax.plot(SFk_new_2, SFffit_2, color='r')
ax.plot(SFk_kink_new, SFffit_kink, color='b')
ax.plot(SFk_kink_new_2, SFffit_2_kink, color='b')

#ax2.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
#ax2.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'b.', markersize=4.)
#a2x.plot(fast_sausage_branch1_k, fast_sausage_branch1_omega/fast_sausage_branch1_k, 'r.', markersize=4.)
#a2x.plot(fast_sausage_branch2_k, fast_sausage_branch2_omega/fast_sausage_branch2_k, 'r.', markersize=4.)
#a2x.plot(fast_sausage_branch3_k, fast_sausage_branch3_omega/fast_sausage_branch3_k, 'r.', markersize=4.)
ax2.plot(k_new, ffit, color='r')
ax2.plot(k_new_2, ffit_2, color='r')
#a2x.plot(k_new_3, ffit_3, color='r')
#a2x.plot(fast_kink_branch1_k, fast_kink_branch1_omega/fast_kink_branch1_k, 'b.', markersize=4.)
#a2x.plot(fast_kink_branch2_k, fast_kink_branch2_omega/fast_kink_branch2_k, 'g.', markersize=4.)
ax2.plot(k_kink_new, ffit_kink, color='b')
ax2.plot(k_kink_new_2, ffit_2_kink, color='b')
#a2x.plot(slow_body_sausage_k, (slow_body_sausage_omega/slow_body_sausage_k), 'r.', markersize=4.)
#a2x.plot(slow_body_kink_k, (slow_body_kink_omega/slow_body_kink_k), 'b.', markersize=4.)
#a2x.plot(body_sausage_branch1_k, body_sausage_branch1_omega/body_sausage_branch1_k, 'r.', markersize=4.)   # body sausage
#a2x.plot(slow_body_kink_branch1_k, slow_body_kink_branch1_omega/slow_body_kink_branch1_k, 'g.', markersize=8.)   # body kink
#a2x.plot(body_sausage_branch2_k, body_sausage_branch2_omega/body_sausage_branch2_k, 'g.', markersize=4.)   # body sausage
#a2x.plot(body_kink_branch2_k, body_kink_branch2_omega/body_kink_branch2_k, 'b.', markersize=4.)   # body kink
ax2.plot(sb_k_new, sb_ffit, color='r')
ax2.plot(sbk_k_new, sbk_ffit, color='b')
ax2.plot(sbb_k_new, sbb_ffit, color='r')
ax2.plot(sbbk_k_new, sbbk_ffit, color='b')
ax2.plot(SFk_new, SFffit, color='r')
ax2.plot(SFk_new_2, SFffit_2, color='r')
ax2.plot(SFk_kink_new, SFffit_kink, color='b')
ax2.plot(SFk_kink_new_2, SFffit_2_kink, color='b')


ax2.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0-0.01), fontsize=20)
ax2.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax2.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax2.annotate( ' $c_{i}$', xy=(Kmax, c_i0-0.01), fontsize=20)
ax2.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax2.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)
ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.annotate( ' $-c_{Ti}$', xy=(Kmax, -cT_i0-0.01), fontsize=20)
ax.annotate( ' $-c_{Te}$', xy=(Kmax, -cT_e), fontsize=20)
ax.annotate( ' $-c_{e}$', xy=(Kmax, -c_e), fontsize=20)
ax.annotate( ' $-c_{i}$', xy=(Kmax, -c_i0-0.01), fontsize=20)
ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)
ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=c_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax2.axhline(y=cT_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}+U_{i0}$', xy=(Kmax, -vA_i0+U_i0), fontsize=10)
ax2.annotate( ' $v_{Ai}+U_{i0}$', xy=(Kmax, vA_i0+U_i0), fontsize=12)
ax2.annotate( ' $c_{i}+U_{i0}$', xy=(Kmax, c_i0+U_i0), fontsize=12)
ax2.annotate( ' $c_{Ti}+U_{i0}$', xy=(Kmax, cT_i0+U_i0), fontsize=12)   #0.02 added to separate annotations
ax2.annotate( ' $c_{i}+U_{B}$', xy=(Kmax, c_i0+flow_bound+0.002), fontsize=12)
ax2.annotate( ' $c_{Ti}+U_{B}$', xy=(Kmax, cT_i0+flow_bound+0.002), fontsize=12)
ax.annotate( ' $-c_{i}+U_{i0}$', xy=(Kmax, -c_i0+U_i0), fontsize=12)      #fontsize 8 for non zoom
ax.annotate( ' $-c_{Ti}+U_{i0}$', xy=(Kmax, -cT_i0+U_i0), fontsize=12)
ax.annotate( ' $-c_{i}+U_{B}$', xy=(Kmax, -c_i0+flow_bound), fontsize=12)
ax.annotate( ' $-c_{Ti}+U_{B}$', xy=(Kmax, -cT_i0+flow_bound+0.002), fontsize=12)
ax.axhline(y=-cT_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')



ax.set_ylim(-1.05, -0.8)
ax2.set_ylim(0.85, 1.1)



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
ax.set_position([box.x0, box.y0, box.width*0.85, box.height])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width*0.85, box2.height])

wavenumber = np.linspace(0.01,4.,20)         
ax.fill_between(wavenumber, -cT_i0+U_i0, -cT_i0+flow_bound, color='red', alpha=0.2)    # fill between uniform case and boundary values
ax2.fill_between(wavenumber, cT_i0+U_i0, cT_i0+flow_bound, color='red', alpha=0.2)    # fill between uniform case and boundary values


plt.savefig("cylinder_flow_width06_coronal_curves_ZOOM_continuum.png")

plt.show()
exit()
##########


sol_ks06_singleplot_fast = []
sol_omegas06_singleplot_fast = []
sol_ks06_singleplot_slow = []
sol_omegas06_singleplot_slow = []

sol_ks06_singleplot_fast_kink = []
sol_omegas06_singleplot_fast_kink = []
sol_ks06_singleplot_slow_kink = []
sol_omegas06_singleplot_slow_kink = []

sol_ks06_singleplot_backwards_sausage = []
sol_omegas06_singleplot_backwards_sausage = []
sol_ks06_singleplot_backwards_kink = []
sol_omegas06_singleplot_backwards_kink = []


for i in range(len(sol_ks06)):
    if sol_ks06[i] > 3.02 and sol_ks06[i] < 3.1:
      if sol_omegas06[i]/sol_ks06[i] > vA_i0 and sol_omegas06[i]/sol_ks06[i] < 4.:
        sol_ks06_singleplot_fast.append(sol_ks06[i])
        sol_omegas06_singleplot_fast.append(sol_omegas06[i])

    if sol_ks06[i] > 1.02 and sol_ks06[i] < 1.1:        
      if sol_omegas06[i]/sol_ks06[i] > 0.95 and sol_omegas06[i]/sol_ks06[i] < 1.:
        sol_ks06_singleplot_slow.append(sol_ks06[i])
        sol_omegas06_singleplot_slow.append(sol_omegas06[i])    

    if sol_ks06[i] > 0.9 and sol_ks06[i] < 0.95:
       if sol_omegas06[i]/sol_ks06[i] > -0.95 and sol_omegas06[i]/sol_ks06[i] < -0.9: 
         sol_ks06_singleplot_backwards_sausage.append(sol_ks06[i])
         sol_omegas06_singleplot_backwards_sausage.append(sol_omegas06[i])



for i in range(len(sol_ks_kink06)):
    if sol_ks_kink06[i] > 3.0 and sol_ks_kink06[i] < 3.05:        
       if sol_omegas_kink06[i]/sol_ks_kink06[i] > vA_i0 and sol_omegas_kink06[i]/sol_ks_kink06[i] < 2.8:
         sol_ks06_singleplot_fast_kink.append(sol_ks_kink06[i])
         sol_omegas06_singleplot_fast_kink.append(sol_omegas_kink06[i])        

    if sol_ks_kink06[i] > 1.95 and sol_ks_kink06[i] < 2.35:         
       if sol_omegas_kink06[i]/sol_ks_kink06[i] > 0.945 and sol_omegas_kink06[i]/sol_ks_kink06[i] < 0.99 : 
         sol_ks06_singleplot_slow_kink.append(sol_ks_kink06[i])
         sol_omegas06_singleplot_slow_kink.append(sol_omegas_kink06[i])

    if sol_ks_kink06[i] > 3. and sol_ks_kink06[i] < 3.05:
       if sol_omegas_kink06[i]/sol_ks_kink06[i] > -c_i0+U_i0 and sol_omegas_kink06[i]/sol_ks_kink06[i] < -cT_i0+flow_bound: 
         sol_ks06_singleplot_backwards_kink.append(sol_ks_kink06[i])
         sol_omegas06_singleplot_backwards_kink.append(sol_omegas_kink06[i])



sol_ks06_singleplot_fast = np.array(sol_ks06_singleplot_fast)
sol_omegas06_singleplot_fast = np.array(sol_omegas06_singleplot_fast)    
sol_ks06_singleplot_slow = np.array(sol_ks06_singleplot_slow)
sol_omegas06_singleplot_slow = np.array(sol_omegas06_singleplot_slow)    
     
sol_ks06_singleplot_fast_kink = np.array(sol_ks06_singleplot_fast_kink)
sol_omegas06_singleplot_fast_kink = np.array(sol_omegas06_singleplot_fast_kink)
sol_ks06_singleplot_slow_kink = np.array(sol_ks06_singleplot_slow_kink)
sol_omegas06_singleplot_slow_kink = np.array(sol_omegas06_singleplot_slow_kink)

sol_ks06_singleplot_backwards_sausage = np.array(sol_ks06_singleplot_backwards_sausage)
sol_omegas06_singleplot_backwards_sausage = np.array(sol_omegas06_singleplot_backwards_sausage)    
sol_ks06_singleplot_backwards_kink = np.array(sol_ks06_singleplot_backwards_kink)
sol_omegas06_singleplot_backwards_kink = np.array(sol_omegas06_singleplot_backwards_kink)    




#####################################################################

plt.figure()
#plt.title("$ W = 1e5$")
ax = plt.subplot(111)
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
ax.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'b.', markersize=4.)

ax.plot(sol_ks06_singleplot_fast_kink[0], (sol_omegas06_singleplot_fast_kink[0]/sol_ks06_singleplot_fast_kink[0]), 'k.', markersize=10.)
ax.plot(sol_ks06_singleplot_slow[0], (sol_omegas06_singleplot_slow[0]/sol_ks06_singleplot_slow[0]), 'k.', markersize=10.)
ax.plot(sol_ks06_singleplot_slow_kink[0], (sol_omegas06_singleplot_slow_kink[0]/sol_ks06_singleplot_slow_kink[0]), 'k.', markersize=10.)
ax.plot(sol_ks06_singleplot_fast[0], (sol_omegas06_singleplot_fast[0]/sol_ks06_singleplot_fast[0]), 'k.', markersize=10.)



ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax.annotate( ' $c_{Ti}$', xy=(Kmax, cT_i0), fontsize=20)
ax.annotate( ' $c_{Te}$', xy=(Kmax, cT_e), fontsize=20)
ax.annotate( ' $c_{e}$', xy=(Kmax, c_e), fontsize=20)
ax.annotate( ' $c_{i}$', xy=(Kmax, c_i0), fontsize=20)
ax.annotate( ' $v_{Ae}$', xy=(Kmax, vA_e), fontsize=20)
ax.annotate( ' $v_{Ai}$', xy=(Kmax, vA_i0), fontsize=20)

ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax.annotate( ' $-c_{Ti}$', xy=(Kmax, -cT_i0), fontsize=20)
ax.annotate( ' $-c_{Te}$', xy=(Kmax, -cT_e), fontsize=20)
ax.annotate( ' $-c_{e}$', xy=(Kmax, -c_e), fontsize=20)
ax.annotate( ' $-c_{i}$', xy=(Kmax, -c_i0), fontsize=20)
ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}$', xy=(Kmax, -vA_i0), fontsize=20)

ax.axhline(y=-vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-vA_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')

ax.axhline(y=c_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')

ax.annotate( ' $-v_{Ae}$', xy=(Kmax, -vA_e), fontsize=20)
ax.annotate( ' $-v_{Ai}+U_{i0}$', xy=(Kmax, -vA_i0+U_i0), fontsize=10)

ax.annotate( ' $v_{Ai}+U_{i0}$', xy=(Kmax, vA_i0+U_i0), fontsize=12)
ax.annotate( ' $c_{i}+U_{i0}$', xy=(Kmax, c_i0+U_i0), fontsize=12)
ax.annotate( ' $c_{Ti}+U_{i0}$', xy=(Kmax, cT_i0+U_i0), fontsize=12)   #0.02 added to separate annotations

ax.annotate( ' $c_{i}+U_{B}$', xy=(Kmax, c_i0+flow_bound), fontsize=12)
ax.annotate( ' $c_{Ti}+U_{B}$', xy=(Kmax, cT_i0+flow_bound), fontsize=12)

ax.annotate( ' $-c_{i}+U_{i0}$', xy=(Kmax, -c_i0+U_i0), fontsize=12)      #fontsize 8 for non zoom
ax.annotate( ' $-c_{Ti}+U_{i0}$', xy=(Kmax, -cT_i0+U_i0), fontsize=12)
ax.annotate( ' $-c_{i}+U_{B}$', xy=(Kmax, -c_i0+flow_bound), fontsize=12)
ax.annotate( ' $-c_{Ti}+U_{B}$', xy=(Kmax, -cT_i0+flow_bound), fontsize=12)

ax.axhline(y=-cT_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-cT_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')

ax.axhline(y=-c_i0+U_i0, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=-c_i0+flow_bound, color='k', linestyle='dashdot', label='_nolegend_')

ax.set_ylim(-5., vA_e)
#ax.set_yticks([])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])


plt.show()
exit()

####################################################

#wavenum = sol_ks125_singleplot_fast[0]
#frequency = sol_omegas125_singleplot_fast[0]

wavenum = sol_ks125_singleplot_slow[0]
frequency = sol_omegas125_singleplot_slow[0]

#wavenum = sol_ks1125_singleplot_backwards_sausage[0]
#frequency = sol_omegas1125_singleplot_backwards_sausage[0]

#wavenum = sol_ks125_singleplot_fast_kink[0]
#frequency = sol_omegas125_singleplot_fast_kink[0]

#wavenum = sol_ks125_singleplot_slow_kink[0]
#frequency = sol_omegas125_singleplot_slow_kink[0]

#wavenum = sol_ks1125_singleplot_backwards_kink[0]
#frequency = sol_omegas1125_singleplot_backwards_kink[0]

#wavenum = sol_ks_kink125[105]
#frequency = sol_omegas_kink125[105]


#####################################################
r0=0.  #mean
dr=1.25 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 0.    # 1 = kink , 0 = sausage

lx = np.linspace(-3.*2.*np.pi/wavenum, -1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

def D(r):  
  return (rho_i(r)*(c_i(r)**2 + vA_i(r)**2)*(shift_freq(r)**2 - alfven_freq(r)**2)*(shift_freq(r)**2 - cusp_freq(r)**2))

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
   
def C3_diff(r):
  return ((B_iphi(r)/r)**2 - (rho_i(r)*(v_iphi(r)/r)**2))

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
       return [P_e[1], (-P_e[1]/r_e + (m_e+(m**2/(r_e**2)))*P_e[0])]
  
P0 = [1e-8, 1e-8]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
Ls = odeint(dP_dr_e, P0, lx)
left_P_solution = Ls[:,0]      # Vx perturbation solution for left hand side

left_xi_solution = xi_e_const*Ls[:,1]    # Pressure perturbation solution for left hand side

normalised_left_P_solution_1e5 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution_1e5 = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
      
          
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,1]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, -0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_1e5 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_1e5 = inside_xi_solution/np.amax(abs(left_xi_solution))




###################################################################
B = 1.

fig, (ax, ax2) = plt.subplots(2,1, sharex=False) 
ax.axvline(x=-B, color='r', linestyle='--')
ax.axvline(x=B, color='r', linestyle='--')
#ax.set_xlabel("$x$")
ax.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
#ax.set_ylim(-1.2,1.2)
ax.set_xlim(-3.,1.2)
ax.plot(lx, normalised_left_P_solution_1e5, 'k')
ax.plot(ix, normalised_inside_P_solution_1e5, 'k')
ax.plot(-lx, normalised_left_P_solution_1e5, 'k--')
ax.plot(-ix, normalised_inside_P_solution_1e5, 'k--')



ax2.axvline(x=-B, color='r', linestyle='--')
ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_x$", fontsize=18, rotation=0, labelpad=15)

#ax2.set_ylim(-1.2,1.2)
ax2.set_xlim(-3.,1.2)
ax2.plot(lx, normalised_left_xi_solution_1e5, 'k')
ax2.plot(ix, normalised_inside_xi_solution_1e5, 'k')
ax2.plot(-lx, -normalised_left_xi_solution_1e5, 'k--')
ax2.plot(-ix, -normalised_inside_xi_solution_1e5, 'k--')

plt.savefig("flow_width125_slow_sausage_eigenfunctions.png")

plt.show()
exit()
