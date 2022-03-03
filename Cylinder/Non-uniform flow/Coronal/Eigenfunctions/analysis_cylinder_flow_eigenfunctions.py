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
from scipy.interpolate import griddata
from scipy import interpolate

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

ix = np.linspace(1., 0.001, 100.)  #1e3


rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols

r0=0.  #mean

dr1e5=1e5 #standard dev
dr3=3. #standard dev
dr15=1.5 #standard dev
dr1=1. #standard dev
dr06=0.6 #standard dev



U_i0 = 0.05*c_i0     #0.35*vA_i  coronal    
U_e = 0.          #-0.15*vA_i   photospheric      0 - coronal

def v_z(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr1e5**2))) 

def v_z_1e5(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr1e5**2))) 

def v_z_3(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr3**2))) 

def v_z_15(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr15**2))) 

def v_z_1(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr1**2))) 

def v_z_06(r):
    return  (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr06**2))) 


v_z_1e5_np=sym.lambdify(rr,v_z_1e5(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_z_3_np=sym.lambdify(rr,v_z_3(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_z_15_np=sym.lambdify(rr,v_z_15(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_z_1_np=sym.lambdify(rr,v_z_1(rr),"numpy")   #In order to evaluate we need to switch to numpy
v_z_06_np=sym.lambdify(rr,v_z_06(rr),"numpy")   #In order to evaluate we need to switch to numpy

#####################
####  define differentiated vz profiles   #####

def dv_z_1e5(r):
  return sym.diff(v_z_1e5(r), r)

def dv_z_3(r):
  return sym.diff(v_z_3(r), r)
  
def dv_z_15(r):
  return sym.diff(v_z_15(r), r)
  
def dv_z_1(r):
  return sym.diff(v_z_1(r), r)
  
def dv_z_06(r):
  return sym.diff(v_z_06(r), r)
  
dv_z_1e5_np=sym.lambdify(rr,dv_z_1e5(rr),"numpy")   #In order to evaluate we need to switch to numpy
dv_z_3_np=sym.lambdify(rr,dv_z_3(rr),"numpy")   #In order to evaluate we need to switch to numpy
dv_z_15_np=sym.lambdify(rr,dv_z_15(rr),"numpy")   #In order to evaluate we need to switch to numpy
dv_z_1_np=sym.lambdify(rr,dv_z_1(rr),"numpy")   #In order to evaluate we need to switch to numpy
dv_z_06_np=sym.lambdify(rr,dv_z_06(rr),"numpy")   #In order to evaluate we need to switch to numpy

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
#plt.title("width = 1.5")
plt.xlabel("$r$", fontsize=22)
#plt.ylabel("$\u03C1_{i}$",fontsize=25)
plt.ylabel("$v_z$",fontsize=22, rotation=0, labelpad=15)
ax = plt.subplot(111)
ax.title.set_text("$v_z$ profiles")
ax.plot(ix,v_z_1e5_np(ix), 'k');
#ax.plot(-ix,v_z_1e5_np(-ix), 'k');

ax.plot(ix,v_z_3_np(ix), color='goldenrod', linestyle='solid');
#ax.plot(-ix,v_z_3_np(-ix), color='goldenrod', linestyle='solid');

ax.plot(ix,v_z_15_np(ix), 'b');
#ax.plot(-ix,v_z_15_np(-ix), 'b');

ax.plot(ix,v_z_1_np(ix), color='magenta', linestyle='solid');
#ax.plot(-ix,v_z_1_np(-ix), color='magenta', linestyle='solid');

ax.plot(ix,v_z_06_np(ix), 'r');
#ax.plot(-ix,v_z_06_np(-ix), 'r');

ax.annotate( ' $U_{e}$', xy=(1.1, U_e),fontsize=22)
ax.annotate( ' $U_{0i}$', xy=(1.1, U_i0),fontsize=22)
ax.axhline(y=U_i0, color='k', label=' $U_{i}$', linestyle='dashdot')
ax.axhline(y=U_e, color='k', label=' $U_{e}$', linestyle='dashdot')
ax.set_ylim(0., 0.06)
ax.set_xlim(0, 1.1)

ax.axvline(x=-1, color='r', linestyle='--')
ax.axvline(x=1, color='r', linestyle='--')


plt.figure()
#plt.title("width = 1.5")
plt.xlabel("$r$", fontsize=22)
#plt.ylabel("$\u03C1_{i}$",fontsize=25)
plt.ylabel("$v_z$",fontsize=22, rotation=0, labelpad=15)
ax = plt.subplot(111)
ax.title.set_text("$dv_z$ profiles")
ax.plot(ix,dv_z_1e5_np(ix), 'k');
#ax.plot(-ix,v_z_1e5_np(-ix), 'k');

ax.plot(ix,dv_z_3_np(ix), color='goldenrod', linestyle='solid');
#ax.plot(-ix,v_z_3_np(-ix), color='goldenrod', linestyle='solid');

ax.plot(ix,dv_z_15_np(ix), 'b');
#ax.plot(-ix,v_z_15_np(-ix), 'b');

ax.plot(ix,dv_z_1_np(ix), color='magenta', linestyle='solid');
#ax.plot(-ix,v_z_1_np(-ix), color='magenta', linestyle='solid');

ax.plot(ix,dv_z_06_np(ix), 'r');
#ax.plot(-ix,v_z_06_np(-ix), 'r');

#ax.annotate( ' $U_{e}$', xy=(1.1, U_e),fontsize=22)
#ax.annotate( ' $U_{0i}$', xy=(1.1, U_i0),fontsize=22)
#ax.axhline(y=U_i0, color='k', label=' $U_{i}$', linestyle='dashdot')
#ax.axhline(y=U_e, color='k', label=' $U_{e}$', linestyle='dashdot')
#ax.set_ylim(0., 0.06)
ax.set_xlim(0, 1.1)

ax.axvline(x=-1, color='r', linestyle='--')
ax.axvline(x=1, color='r', linestyle='--')


#plt.savefig("flow005_vz_profiles.png")
#plt.show()
#exit()


####    background vorticity    #######
z = np.linspace(0.01, 10., 11.)   #21  #31
THETA = np.linspace(0, 2.*np.pi, 50) #50

radii, thetas, Z = np.meshgrid(ix,THETA,z,sparse=False, indexing='ij')



vort_x_1e5 = np.zeros(((len(ix), len(THETA), len(z))))
vort_y_1e5 = np.zeros(((len(ix), len(THETA), len(z))))

vort_x_3 = np.zeros(((len(ix), len(THETA), len(z))))
vort_y_3 = np.zeros(((len(ix), len(THETA), len(z))))

vort_x_15 = np.zeros(((len(ix), len(THETA), len(z))))
vort_y_15 = np.zeros(((len(ix), len(THETA), len(z))))

vort_x_1 = np.zeros(((len(ix), len(THETA), len(z))))
vort_y_1 = np.zeros(((len(ix), len(THETA), len(z))))

vort_x_06 = np.zeros(((len(ix), len(THETA), len(z))))
vort_y_06 = np.zeros(((len(ix), len(THETA), len(z))))



for k in range(len(z)):
  for j in range(len(THETA)):
    for i in range(len(ix)):
        vort_x_1e5[i,j,k] = dv_z_1e5_np(ix)[i]*np.sin(thetas[i,j,k])
        vort_y_1e5[i,j,k] = -dv_z_1e5_np(ix)[i]*np.cos(thetas[i,j,k])

        vort_x_3[i,j,k] = dv_z_3_np(ix)[i]*np.sin(thetas[i,j,k])
        vort_y_3[i,j,k] = -dv_z_3_np(ix)[i]*np.cos(thetas[i,j,k])

        vort_x_15[i,j,k] = dv_z_15_np(ix)[i]*np.sin(thetas[i,j,k])
        vort_y_15[i,j,k] = -dv_z_15_np(ix)[i]*np.cos(thetas[i,j,k])

        vort_x_1[i,j,k] = dv_z_1_np(ix)[i]*np.sin(thetas[i,j,k])
        vort_y_1[i,j,k] = -dv_z_1_np(ix)[i]*np.cos(thetas[i,j,k])

        vort_x_06[i,j,k] = dv_z_06_np(ix)[i]*np.sin(thetas[i,j,k])
        vort_y_06[i,j,k] = -dv_z_06_np(ix)[i]*np.cos(thetas[i,j,k])



         
#boundary_point_r = radii[bound_index] 
#boundary_point_z = Z[bound_index]        
#
#bound_element = bound_index[0][0] 
#
#boundary_point_x = boundary_point_r*np.cos(thetas[bound_element,:,:]) + v_x[0,bound_element,:,:]*step
#boundary_point_y = boundary_point_r*np.sin(thetas[bound_element,:,:]) + v_y[0,bound_element,:,:]*step  # for small k
##boundary_point_z = boundary_point_z + v_z[bound_index]
#
#boundary_point_x = boundary_point_x[0,:,:]
#boundary_point_y = boundary_point_y[0,:,:]
#boundary_point_z = boundary_point_z[0,:,:]
#

x = radii*np.cos(thetas)
y = radii*np.sin(thetas)
z = Z

#################################################
print(x.shape)
print(y.shape)
print(z.shape)

el = 6


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)


circle_rad =1.
a = circle_rad*np.cos(THETA) #circle x
b = circle_rad*np.sin(THETA) #circle y

ax.plot(a,b, 'r--')

ax.set_title('Vorticity  -  W = 1e5')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18, rotation=0, labelpad=15)


new_x = np.linspace(-1., 1., 400)
new_y = np.linspace(-1., 1., 400)  # 100

new_xx, new_yy = np.meshgrid(new_x,new_y, sparse=False, indexing='ij')

flat_x = x[:,:,el].flatten('C')
flat_y = y[:,:,el].flatten('C')
    
points = np.transpose(np.vstack((flat_x, flat_y)))


flat_vx_1e5 = vort_x_1e5[:,:,el].flatten('C')
flat_vy_1e5 = vort_y_1e5[:,:,el].flatten('C')

vx_interp_1e5 = griddata(points, flat_vx_1e5, (new_xx, new_yy), method='cubic')
vy_interp_1e5 = griddata(points, flat_vy_1e5, (new_xx, new_yy), method='cubic')

###
flat_vx_3 = vort_x_3[:,:,el].flatten('C')
flat_vy_3 = vort_y_3[:,:,el].flatten('C')

vx_interp_3 = griddata(points, flat_vx_3, (new_xx, new_yy), method='cubic')
vy_interp_3 = griddata(points, flat_vy_3, (new_xx, new_yy), method='cubic')
###

flat_vx_15 = vort_x_15[:,:,el].flatten('C')
flat_vy_15 = vort_y_15[:,:,el].flatten('C')

vx_interp_15 = griddata(points, flat_vx_15, (new_xx, new_yy), method='cubic')
vy_interp_15 = griddata(points, flat_vy_15, (new_xx, new_yy), method='cubic')

###

flat_vx_1 = vort_x_1[:,:,el].flatten('C')
flat_vy_1 = vort_y_1[:,:,el].flatten('C')

vx_interp_1 = griddata(points, flat_vx_1, (new_xx, new_yy), method='cubic')
vy_interp_1 = griddata(points, flat_vy_1, (new_xx, new_yy), method='cubic')

###

flat_vx_06 = vort_x_06[:,:,el].flatten('C')
flat_vy_06 = vort_y_06[:,:,el].flatten('C')

vx_interp_06 = griddata(points, flat_vx_06, (new_xx, new_yy), method='cubic')
vy_interp_06 = griddata(points, flat_vy_06, (new_xx, new_yy), method='cubic')

###

ax.quiver(new_xx[::25,::25], new_yy[::25,::25], vx_interp_1e5[::25,::25], vy_interp_1e5[::25,::25], pivot='tail', color='black', scale_units='inches', scale=0.1, width=0.003)


plt.show()
exit()

#### NOTE   -  Vorticity should be same at all heights for background flow, check this is correct




########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_1e5.pickle', 'rb') as f:
    sol_omegas1e5, sol_ks1e5, sol_omegas_kink1e5, sol_ks_kink1e5 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas1e5 = [x for _,x in sorted(zip(sol_ks1e5,sol_omegas1e5))]
sol_ks1e5 = np.sort(sol_ks1e5)

sol_omegas1e5 = np.array(sol_omegas1e5)
sol_ks1e5 = np.array(sol_ks1e5)

sol_omegas_kink1e5 = [x for _,x in sorted(zip(sol_ks_kink1e5,sol_omegas_kink1e5))]
sol_ks_kink1e5 = np.sort(sol_ks_kink1e5)

sol_omegas_kink1e5 = np.array(sol_omegas_kink1e5)
sol_ks_kink1e5 = np.array(sol_ks_kink1e5)


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_3.pickle', 'rb') as f:
    sol_omegas3, sol_ks3, sol_omegas_kink3, sol_ks_kink3 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas3 = [x for _,x in sorted(zip(sol_ks3,sol_omegas3))]
sol_ks3 = np.sort(sol_ks3)

sol_omegas3 = np.array(sol_omegas3)
sol_ks3 = np.array(sol_ks3)

sol_omegas_kink3 = [x for _,x in sorted(zip(sol_ks_kink3,sol_omegas_kink3))]
sol_ks_kink3 = np.sort(sol_ks_kink3)

sol_omegas_kink3 = np.array(sol_omegas_kink3)
sol_ks_kink3 = np.array(sol_ks_kink3)



########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_15.pickle', 'rb') as f:
    sol_omegas15, sol_ks15, sol_omegas_kink15, sol_ks_kink15 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas15 = [x for _,x in sorted(zip(sol_ks15,sol_omegas15))]
sol_ks15 = np.sort(sol_ks15)

sol_omegas15 = np.array(sol_omegas15)
sol_ks15 = np.array(sol_ks15)

sol_omegas_kink15 = [x for _,x in sorted(zip(sol_ks_kink15,sol_omegas_kink15))]
sol_ks_kink15 = np.sort(sol_ks_kink15)

sol_omegas_kink15 = np.array(sol_omegas_kink15)
sol_ks_kink15 = np.array(sol_ks_kink15)


########   READ IN VARIABLES    #########

with open('Cylindrical_coronal_flow_1.pickle', 'rb') as f:
    sol_omegas1, sol_ks1, sol_omegas_kink1, sol_ks_kink1 = pickle.load(f)



### SORT ARRAYS IN ORDER OF WAVENUMBER ###
sol_omegas1 = [x for _,x in sorted(zip(sol_ks1,sol_omegas1))]
sol_ks1 = np.sort(sol_ks1)

sol_omegas1 = np.array(sol_omegas1)
sol_ks1 = np.array(sol_ks1)

sol_omegas_kink1 = [x for _,x in sorted(zip(sol_ks_kink1,sol_omegas_kink1))]
sol_ks_kink1 = np.sort(sol_ks_kink1)

sol_omegas_kink1 = np.array(sol_omegas_kink1)
sol_ks_kink1 = np.array(sol_ks_kink1)


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



########################################################################################################

plt.figure()
#plt.title("$ W = 1e5$")
ax = plt.subplot(111)
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
#ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)

ax.plot(sol_ks3, (sol_omegas3/sol_ks3), marker='.', markersize=4., color='goldenrod', linestyle ='')
#ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), marker='.', markersize=4., color='goldenrod', linestyle ='')

ax.plot(sol_ks15, (sol_omegas15/sol_ks15), 'b.', markersize=4.)
#ax.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'b.', markersize=4.)

ax.plot(sol_ks1, (sol_omegas1/sol_ks1), marker='.', markersize=4., color='magenta', linestyle ='')
#ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), marker='.', markersize=4., color='magenta', linestyle ='')

ax.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
#ax.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'r.', markersize=4.)



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

#plt.savefig("flow_width2_coronal_curves_dots.png")
#plt.savefig("Mihai_symmetric_flow.png")
#plt.show()
#exit()

##########


sol_ks1e5_singleplot_fast = []
sol_omegas1e5_singleplot_fast = []
sol_ks1e5_singleplot_slow = []
sol_omegas1e5_singleplot_slow = []

sol_ks1e5_singleplot_fast_kink = []
sol_omegas1e5_singleplot_fast_kink = []
sol_ks1e5_singleplot_slow_kink = []
sol_omegas1e5_singleplot_slow_kink = []

sol_ks1e5_singleplot_backwards_sausage = []
sol_omegas1e5_singleplot_backwards_sausage = []
sol_ks1e5_singleplot_backwards_kink = []
sol_omegas1e5_singleplot_backwards_kink = []


for i in range(len(sol_ks1e5)):
    if sol_ks1e5[i] > 3.02 and sol_ks1e5[i] < 3.1:
      if sol_omegas1e5[i]/sol_ks1e5[i] > vA_i0 and sol_omegas1e5[i]/sol_ks1e5[i] < 4.:
        sol_ks1e5_singleplot_fast.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_fast.append(sol_omegas1e5[i])

    if sol_ks1e5[i] > 1.02 and sol_ks1e5[i] < 1.1:        
      if sol_omegas1e5[i]/sol_ks1e5[i] > 0.95 and sol_omegas1e5[i]/sol_ks1e5[i] < 1.:
        sol_ks1e5_singleplot_slow.append(sol_ks1e5[i])
        sol_omegas1e5_singleplot_slow.append(sol_omegas1e5[i])    

    if sol_ks1e5[i] > 0.9 and sol_ks1e5[i] < 0.95:
       if sol_omegas1e5[i]/sol_ks1e5[i] > -0.95 and sol_omegas1e5[i]/sol_ks1e5[i] < -0.9: 
         sol_ks1e5_singleplot_backwards_sausage.append(sol_ks1e5[i])
         sol_omegas1e5_singleplot_backwards_sausage.append(sol_omegas1e5[i])



for i in range(len(sol_ks_kink1e5)):
    if sol_ks_kink1e5[i] > 3.0 and sol_ks_kink1e5[i] < 3.05:
    #if sol_ks_kink1e5[i] > 0.96 and sol_ks_kink1e5[i] < 1.04:        
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > vA_i0 and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < 2.8:
         sol_ks1e5_singleplot_fast_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_fast_kink.append(sol_omegas_kink1e5[i])        

    if sol_ks_kink1e5[i] > 1.9 and sol_ks_kink1e5[i] < 2.:         
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < 1. and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > 0.95 : 
         sol_ks1e5_singleplot_slow_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_slow_kink.append(sol_omegas_kink1e5[i])

    if sol_ks_kink1e5[i] > 3. and sol_ks_kink1e5[i] < 3.05:
       if sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] > -c_i0+U_i0 and sol_omegas_kink1e5[i]/sol_ks_kink1e5[i] < -cT_i0+flow_bound: 
         sol_ks1e5_singleplot_backwards_kink.append(sol_ks_kink1e5[i])
         sol_omegas1e5_singleplot_backwards_kink.append(sol_omegas_kink1e5[i])


sol_ks1e5_singleplot_fast = np.array(sol_ks1e5_singleplot_fast)
sol_omegas1e5_singleplot_fast = np.array(sol_omegas1e5_singleplot_fast)    
sol_ks1e5_singleplot_slow = np.array(sol_ks1e5_singleplot_slow)
sol_omegas1e5_singleplot_slow = np.array(sol_omegas1e5_singleplot_slow)    
     
sol_ks1e5_singleplot_fast_kink = np.array(sol_ks1e5_singleplot_fast_kink)
sol_omegas1e5_singleplot_fast_kink = np.array(sol_omegas1e5_singleplot_fast_kink)
sol_ks1e5_singleplot_slow_kink = np.array(sol_ks1e5_singleplot_slow_kink)
sol_omegas1e5_singleplot_slow_kink = np.array(sol_omegas1e5_singleplot_slow_kink)

sol_ks1e5_singleplot_backwards_sausage = np.array(sol_ks1e5_singleplot_backwards_sausage)
sol_omegas1e5_singleplot_backwards_sausage = np.array(sol_omegas1e5_singleplot_backwards_sausage)    
sol_ks1e5_singleplot_backwards_kink = np.array(sol_ks1e5_singleplot_backwards_kink)
sol_omegas1e5_singleplot_backwards_kink = np.array(sol_omegas1e5_singleplot_backwards_kink)    




##########


sol_ks3_singleplot_fast = []
sol_omegas3_singleplot_fast = []
sol_ks3_singleplot_slow = []
sol_omegas3_singleplot_slow = []

sol_ks3_singleplot_fast_kink = []
sol_omegas3_singleplot_fast_kink = []
sol_ks3_singleplot_slow_kink = []
sol_omegas3_singleplot_slow_kink = []

sol_ks3_singleplot_backwards_sausage = []
sol_omegas3_singleplot_backwards_sausage = []
sol_ks3_singleplot_backwards_kink = []
sol_omegas3_singleplot_backwards_kink = []


for i in range(len(sol_ks3)):
    if sol_ks3[i] > 3.02 and sol_ks3[i] < 3.2:
      if sol_omegas3[i]/sol_ks3[i] > vA_i0 and sol_omegas3[i]/sol_ks3[i] < 4.:
        sol_ks3_singleplot_fast.append(sol_ks3[i])
        sol_omegas3_singleplot_fast.append(sol_omegas3[i])

    if sol_ks3[i] > 1.02 and sol_ks3[i] < 1.1:        
      if sol_omegas3[i]/sol_ks3[i] > 0.95 and sol_omegas3[i]/sol_ks3[i] < 1.:
        sol_ks3_singleplot_slow.append(sol_ks3[i])
        sol_omegas3_singleplot_slow.append(sol_omegas3[i])    

    if sol_ks3[i] > 0.9 and sol_ks3[i] < 0.95:
       if sol_omegas3[i]/sol_ks3[i] > -0.95 and sol_omegas3[i]/sol_ks3[i] < -0.9: 
         sol_ks3_singleplot_backwards_sausage.append(sol_ks3[i])
         sol_omegas3_singleplot_backwards_sausage.append(sol_omegas3[i])



for i in range(len(sol_ks_kink3)):
    if sol_ks_kink3[i] > 3.0 and sol_ks_kink3[i] < 3.05:
    #if sol_ks_kink3[i] > 0.96 and sol_ks_kink3[i] < 1.04:        
       if sol_omegas_kink3[i]/sol_ks_kink3[i] > vA_i0 and sol_omegas_kink3[i]/sol_ks_kink3[i] < 2.8:
         sol_ks3_singleplot_fast_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_fast_kink.append(sol_omegas_kink3[i])        

    if sol_ks_kink3[i] > 1.9 and sol_ks_kink3[i] < 2.:         
       if sol_omegas_kink3[i]/sol_ks_kink3[i] > 0.96 and sol_omegas_kink3[i]/sol_ks_kink3[i] < 0.98 : 
         sol_ks3_singleplot_slow_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_slow_kink.append(sol_omegas_kink3[i])

    if sol_ks_kink3[i] > 3. and sol_ks_kink3[i] < 3.05:
       if sol_omegas_kink3[i]/sol_ks_kink3[i] > -c_i0+U_i0 and sol_omegas_kink3[i]/sol_ks_kink3[i] < -cT_i0+flow_bound: 
         sol_ks3_singleplot_backwards_kink.append(sol_ks_kink3[i])
         sol_omegas3_singleplot_backwards_kink.append(sol_omegas_kink3[i])



sol_ks3_singleplot_fast = np.array(sol_ks3_singleplot_fast)
sol_omegas3_singleplot_fast = np.array(sol_omegas3_singleplot_fast)    
sol_ks3_singleplot_slow = np.array(sol_ks3_singleplot_slow)
sol_omegas3_singleplot_slow = np.array(sol_omegas3_singleplot_slow)    
     
sol_ks3_singleplot_fast_kink = np.array(sol_ks3_singleplot_fast_kink)
sol_omegas3_singleplot_fast_kink = np.array(sol_omegas3_singleplot_fast_kink)
sol_ks3_singleplot_slow_kink = np.array(sol_ks3_singleplot_slow_kink)
sol_omegas3_singleplot_slow_kink = np.array(sol_omegas3_singleplot_slow_kink)

sol_ks3_singleplot_backwards_sausage = np.array(sol_ks3_singleplot_backwards_sausage)
sol_omegas3_singleplot_backwards_sausage = np.array(sol_omegas3_singleplot_backwards_sausage)    
sol_ks3_singleplot_backwards_kink = np.array(sol_ks3_singleplot_backwards_kink)
sol_omegas3_singleplot_backwards_kink = np.array(sol_omegas3_singleplot_backwards_kink)    

##########


sol_ks15_singleplot_fast = []
sol_omegas15_singleplot_fast = []
sol_ks15_singleplot_slow = []
sol_omegas15_singleplot_slow = []

sol_ks15_singleplot_fast_kink = []
sol_omegas15_singleplot_fast_kink = []
sol_ks15_singleplot_slow_kink = []
sol_omegas15_singleplot_slow_kink = []

sol_ks15_singleplot_backwards_sausage = []
sol_omegas15_singleplot_backwards_sausage = []
sol_ks15_singleplot_backwards_kink = []
sol_omegas15_singleplot_backwards_kink = []


for i in range(len(sol_ks15)):
    if sol_ks15[i] > 3.02 and sol_ks15[i] < 3.1:
      if sol_omegas15[i]/sol_ks15[i] > vA_i0 and sol_omegas15[i]/sol_ks15[i] < 4.:
        sol_ks15_singleplot_fast.append(sol_ks15[i])
        sol_omegas15_singleplot_fast.append(sol_omegas15[i])

    if sol_ks15[i] > 1.02 and sol_ks15[i] < 1.1:        
      if sol_omegas15[i]/sol_ks15[i] > 0.95 and sol_omegas15[i]/sol_ks15[i] < 1.:
        sol_ks15_singleplot_slow.append(sol_ks15[i])
        sol_omegas15_singleplot_slow.append(sol_omegas15[i])    

    if sol_ks15[i] > 0.9 and sol_ks15[i] < 0.95:
       if sol_omegas15[i]/sol_ks15[i] > -0.95 and sol_omegas15[i]/sol_ks15[i] < -0.9: 
         sol_ks15_singleplot_backwards_sausage.append(sol_ks15[i])
         sol_omegas15_singleplot_backwards_sausage.append(sol_omegas15[i])



for i in range(len(sol_ks_kink15)):
    if sol_ks_kink15[i] > 3.0 and sol_ks_kink15[i] < 3.05:  
    #if sol_ks_kink15[i] > 0.96 and sol_ks_kink15[i] < 1.04:      
       if sol_omegas_kink15[i]/sol_ks_kink15[i] > vA_i0 and sol_omegas_kink15[i]/sol_ks_kink15[i] < 2.8:
         sol_ks15_singleplot_fast_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_fast_kink.append(sol_omegas_kink15[i])        

    if sol_ks_kink15[i] > 1.9 and sol_ks_kink15[i] < 2.:         
       if sol_omegas_kink15[i]/sol_ks_kink15[i] > 0.96 and sol_omegas_kink15[i]/sol_ks_kink15[i] < 0.98 : 
         sol_ks15_singleplot_slow_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_slow_kink.append(sol_omegas_kink15[i])

    if sol_ks_kink15[i] > 3. and sol_ks_kink15[i] < 3.05:
       if sol_omegas_kink15[i]/sol_ks_kink15[i] > -c_i0+U_i0 and sol_omegas_kink15[i]/sol_ks_kink15[i] < -cT_i0+flow_bound: 
         sol_ks15_singleplot_backwards_kink.append(sol_ks_kink15[i])
         sol_omegas15_singleplot_backwards_kink.append(sol_omegas_kink15[i])



sol_ks15_singleplot_fast = np.array(sol_ks15_singleplot_fast)
sol_omegas15_singleplot_fast = np.array(sol_omegas15_singleplot_fast)    
sol_ks15_singleplot_slow = np.array(sol_ks15_singleplot_slow)
sol_omegas15_singleplot_slow = np.array(sol_omegas15_singleplot_slow)    
     
sol_ks15_singleplot_fast_kink = np.array(sol_ks15_singleplot_fast_kink)
sol_omegas15_singleplot_fast_kink = np.array(sol_omegas15_singleplot_fast_kink)
sol_ks15_singleplot_slow_kink = np.array(sol_ks15_singleplot_slow_kink)
sol_omegas15_singleplot_slow_kink = np.array(sol_omegas15_singleplot_slow_kink)

sol_ks15_singleplot_backwards_sausage = np.array(sol_ks15_singleplot_backwards_sausage)
sol_omegas15_singleplot_backwards_sausage = np.array(sol_omegas15_singleplot_backwards_sausage)    
sol_ks15_singleplot_backwards_kink = np.array(sol_ks15_singleplot_backwards_kink)
sol_omegas15_singleplot_backwards_kink = np.array(sol_omegas15_singleplot_backwards_kink)    



##########


sol_ks1_singleplot_fast = []
sol_omegas1_singleplot_fast = []
sol_ks1_singleplot_slow = []
sol_omegas1_singleplot_slow = []

sol_ks1_singleplot_fast_kink = []
sol_omegas1_singleplot_fast_kink = []
sol_ks1_singleplot_slow_kink = []
sol_omegas1_singleplot_slow_kink = []

sol_ks1_singleplot_backwards_sausage = []
sol_omegas1_singleplot_backwards_sausage = []
sol_ks1_singleplot_backwards_kink = []
sol_omegas1_singleplot_backwards_kink = []


for i in range(len(sol_ks1)):
    if sol_ks1[i] > 3.02 and sol_ks1[i] < 3.1:
      if sol_omegas1[i]/sol_ks1[i] > vA_i0 and sol_omegas1[i]/sol_ks1[i] < 4.:
        sol_ks1_singleplot_fast.append(sol_ks1[i])
        sol_omegas1_singleplot_fast.append(sol_omegas1[i])

    if sol_ks1[i] > 1.02 and sol_ks1[i] < 1.1:        
      if sol_omegas1[i]/sol_ks1[i] > 0.95 and sol_omegas1[i]/sol_ks1[i] < 1.:
        sol_ks1_singleplot_slow.append(sol_ks1[i])
        sol_omegas1_singleplot_slow.append(sol_omegas1[i])    

    if sol_ks1[i] > 0.9 and sol_ks1[i] < 0.95:
       if sol_omegas1[i]/sol_ks1[i] > -0.95 and sol_omegas1[i]/sol_ks1[i] < -0.9: 
         sol_ks1_singleplot_backwards_sausage.append(sol_ks1[i])
         sol_omegas1_singleplot_backwards_sausage.append(sol_omegas1[i])



for i in range(len(sol_ks_kink1)):
    if sol_ks_kink1[i] > 3.0 and sol_ks_kink1[i] < 3.05: 
    #if sol_ks_kink1[i] > 0.96 and sol_ks_kink1[i] < 1.04:       
       if sol_omegas_kink1[i]/sol_ks_kink1[i] > vA_i0 and sol_omegas_kink1[i]/sol_ks_kink1[i] < 2.8:
         sol_ks1_singleplot_fast_kink.append(sol_ks_kink1[i])
         sol_omegas1_singleplot_fast_kink.append(sol_omegas_kink1[i])        

    if sol_ks_kink1[i] > 1.9 and sol_ks_kink1[i] < 2.:         
       if sol_omegas_kink1[i]/sol_ks_kink1[i] > 0.95 and sol_omegas_kink1[i]/sol_ks_kink1[i] < 0.99 : 
         sol_ks1_singleplot_slow_kink.append(sol_ks_kink1[i])
         sol_omegas1_singleplot_slow_kink.append(sol_omegas_kink1[i])

    if sol_ks_kink1[i] > 3. and sol_ks_kink1[i] < 3.05:
       if sol_omegas_kink1[i]/sol_ks_kink1[i] > -c_i0+U_i0 and sol_omegas_kink1[i]/sol_ks_kink1[i] < -cT_i0+flow_bound: 
         sol_ks1_singleplot_backwards_kink.append(sol_ks_kink1[i])
         sol_omegas1_singleplot_backwards_kink.append(sol_omegas_kink1[i])



sol_ks1_singleplot_fast = np.array(sol_ks1_singleplot_fast)
sol_omegas1_singleplot_fast = np.array(sol_omegas1_singleplot_fast)    
sol_ks1_singleplot_slow = np.array(sol_ks1_singleplot_slow)
sol_omegas1_singleplot_slow = np.array(sol_omegas1_singleplot_slow)    
     
sol_ks1_singleplot_fast_kink = np.array(sol_ks1_singleplot_fast_kink)
sol_omegas1_singleplot_fast_kink = np.array(sol_omegas1_singleplot_fast_kink)
sol_ks1_singleplot_slow_kink = np.array(sol_ks1_singleplot_slow_kink)
sol_omegas1_singleplot_slow_kink = np.array(sol_omegas1_singleplot_slow_kink)

sol_ks1_singleplot_backwards_sausage = np.array(sol_ks1_singleplot_backwards_sausage)
sol_omegas1_singleplot_backwards_sausage = np.array(sol_omegas1_singleplot_backwards_sausage)    
sol_ks1_singleplot_backwards_kink = np.array(sol_ks1_singleplot_backwards_kink)
sol_omegas1_singleplot_backwards_kink = np.array(sol_omegas1_singleplot_backwards_kink)    



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
    #if sol_ks_kink06[i] > 0.96 and sol_ks_kink06[i] < 1.04:
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


##############################################################################################################


plt.figure()
#plt.title("$ W = 1e5$")
ax = plt.subplot(111)
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$\frac{\omega}{k}$', fontsize=22, rotation=0, labelpad=15)

#ax.plot(sol_ks1e5, (sol_omegas1e5/sol_ks1e5), 'k.', markersize=4.)
ax.plot(sol_ks_kink1e5, (sol_omegas_kink1e5/sol_ks_kink1e5), 'k.', markersize=4.)

#ax.plot(sol_ks3, (sol_omegas3/sol_ks3), marker='.', markersize=4., color='goldenrod', linestyle ='')
ax.plot(sol_ks_kink3, (sol_omegas_kink3/sol_ks_kink3), marker='.', markersize=4., color='goldenrod', linestyle ='')

#ax.plot(sol_ks15, (sol_omegas15/sol_ks15), 'b.', markersize=4.)
ax.plot(sol_ks_kink15, (sol_omegas_kink15/sol_ks_kink15), 'b.', markersize=4.)

#ax.plot(sol_ks1, (sol_omegas1/sol_ks1), marker='.', markersize=4., color='magenta', linestyle ='')
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), marker='.', markersize=4., color='magenta', linestyle ='')

#ax.plot(sol_ks06, (sol_omegas06/sol_ks06), 'r.', markersize=4.)
ax.plot(sol_ks_kink06, (sol_omegas_kink06/sol_ks_kink06), 'r.', markersize=4.)


#ax.plot(sol_ks1e5_singleplot_fast_kink[0], (sol_omegas1e5_singleplot_fast_kink[0]/sol_ks1e5_singleplot_fast_kink[0]), 'k.', markersize=10.)
#ax.plot(sol_ks1e5_singleplot_slow[0], (sol_omegas1e5_singleplot_slow[0]/sol_ks1e5_singleplot_slow[0]), 'k.', markersize=15.)
#ax.plot(sol_ks1e5_singleplot_slow_kink[0], (sol_omegas1e5_singleplot_slow_kink[0]/sol_ks1e5_singleplot_slow_kink[0]), 'k.', markersize=10.)
#ax.plot(sol_ks1e5_singleplot_fast[0], (sol_omegas1e5_singleplot_fast[0]/sol_ks1e5_singleplot_fast[0]), 'k.', markersize=15.)


ax.plot(sol_ks3_singleplot_fast_kink[0], (sol_omegas3_singleplot_fast_kink[0]/sol_ks3_singleplot_fast_kink[0]), marker='.', markersize=10., color='goldenrod', linestyle='')
#ax.plot(sol_ks3_singleplot_slow[0], (sol_omegas3_singleplot_slow[0]/sol_ks3_singleplot_slow[0]),marker='.', markersize=15., color='goldenrod', linestyle='')
#ax.plot(sol_ks3_singleplot_slow_kink[0], (sol_omegas3_singleplot_slow_kink[0]/sol_ks3_singleplot_slow_kink[0]), marker='.', markersize=10., color='goldenrod', linestyle='')
#ax.plot(sol_ks3_singleplot_fast[0], (sol_omegas3_singleplot_fast[0]/sol_ks3_singleplot_fast[0]), marker='.', markersize=15., color='goldenrod', linestyle='')


#ax.plot(sol_ks15_singleplot_fast_kink[0], (sol_omegas15_singleplot_fast_kink[0]/sol_ks15_singleplot_fast_kink[0]), 'b.', markersize=10.)
#ax.plot(sol_ks15_singleplot_slow[0], (sol_omegas15_singleplot_slow[0]/sol_ks15_singleplot_slow[0]), 'b.', markersize=15.)
#ax.plot(sol_ks15_singleplot_slow_kink[0], (sol_omegas15_singleplot_slow_kink[0]/sol_ks15_singleplot_slow_kink[0]), 'b.', markersize=10.)
#ax.plot(sol_ks15_singleplot_fast[0], (sol_omegas15_singleplot_fast[0]/sol_ks15_singleplot_fast[0]), 'b.', markersize=15.)


#ax.plot(sol_ks1_singleplot_fast_kink[0], (sol_omegas1_singleplot_fast_kink[0]/sol_ks1_singleplot_fast_kink[0]), marker='.', markersize=10., color='magenta', linestyle='')
#ax.plot(sol_ks1_singleplot_slow[0], (sol_omegas1_singleplot_slow[0]/sol_ks1_singleplot_slow[0]), marker='.', markersize=15., color='magenta', linestyle='')
#ax.plot(sol_ks1_singleplot_slow_kink[0], (sol_omegas1_singleplot_slow_kink[0]/sol_ks1_singleplot_slow_kink[0]), marker='.', markersize=10., color='magenta', linestyle='')
#ax.plot(sol_ks1_singleplot_fast[0], (sol_omegas1_singleplot_fast[0]/sol_ks1_singleplot_fast[0]), marker='.', markersize=15., color='magenta', linestyle='')


#ax.plot(sol_ks06_singleplot_fast_kink[0], (sol_omegas06_singleplot_fast_kink[0]/sol_ks06_singleplot_fast_kink[0]), 'r.', markersize=15.)
#ax.plot(sol_ks06_singleplot_slow[0], (sol_omegas06_singleplot_slow[0]/sol_ks06_singleplot_slow[0]), 'r.', markersize=15.)
#ax.plot(sol_ks06_singleplot_slow_kink[0], (sol_omegas06_singleplot_slow_kink[0]/sol_ks06_singleplot_slow_kink[0]), 'r.', markersize=15.)
#ax.plot(sol_ks06_singleplot_fast[0], (sol_omegas06_singleplot_fast[0]/sol_ks06_singleplot_fast[0]), 'r.', markersize=15.)



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

#ax.set_ylim(-5., vA_e)
ax.set_ylim(0., vA_e)
#ax.set_yticks([])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])


#plt.show()
#exit()
#####################################################################


####################################################

#wavenum = sol_ks1e5_singleplot_fast[0]
#frequency = sol_omegas1e5_singleplot_fast[0]

#wavenum = sol_ks1e5_singleplot_slow[0]
#frequency = sol_omegas1e5_singleplot_slow[0]

wavenum = sol_ks1e5_singleplot_fast_kink[0]
frequency = sol_omegas1e5_singleplot_fast_kink[0]

#wavenum = sol_ks1e5_singleplot_slow_kink[0]
#frequency = sol_omegas1e5_singleplot_slow_kink[0]


#####################################################
#####################################################
r0=0.  #mean
dr=1e5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 1.    # 1 = kink , 0 = sausage

lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################
def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

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
      u1 = U[:,0]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_1e5 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_1e5 = inside_xi_solution/np.amax(abs(left_xi_solution))

inside_xi_phi_1e5 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_1e5 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency**2 - wavenum**2*vA_e**2)))
radial_xi_phi_1e5 = np.concatenate((inside_xi_phi_1e5, outside_xi_phi_1e5), axis=None)    

normalised_inside_xi_phi_1e5 = inside_xi_phi_1e5/np.amax(abs(outside_xi_phi_1e5))
normalised_outside_xi_phi_1e5 = outside_xi_phi_1e5/np.amax(abs(outside_xi_phi_1e5))
normalised_radial_xi_phi_1e5 = np.concatenate((normalised_inside_xi_phi_1e5, normalised_outside_xi_phi_1e5), axis=None)    



####################################################

#wavenum = sol_ks3_singleplot_fast[0]
#frequency = sol_omegas3_singleplot_fast[0]

#wavenum = sol_ks3_singleplot_slow[0]
#frequency = sol_omegas3_singleplot_slow[0]

wavenum = sol_ks3_singleplot_fast_kink[0]
frequency = sol_omegas3_singleplot_fast_kink[0]

#wavenum = sol_ks3_singleplot_slow_kink[0]
#frequency = sol_omegas3_singleplot_slow_kink[0]


#####################################################
#####################################################
r0=0.  #mean
dr=3. #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 1.    # 1 = kink , 0 = sausage

lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################
def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

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

normalised_left_P_solution_3 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution_3 = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
      
          
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_3 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_3 = inside_xi_solution/np.amax(abs(left_xi_solution))

inside_xi_phi_3 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_3 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency**2 - wavenum**2*vA_e**2)))
radial_xi_phi_3 = np.concatenate((inside_xi_phi_3, outside_xi_phi_3), axis=None)    

normalised_inside_xi_phi_3 = inside_xi_phi_3/np.amax(abs(outside_xi_phi_3))
normalised_outside_xi_phi_3 = outside_xi_phi_3/np.amax(abs(outside_xi_phi_3))
normalised_radial_xi_phi_3 = np.concatenate((normalised_inside_xi_phi_3, normalised_outside_xi_phi_3), axis=None)    



####################################################

#wavenum = sol_ks15_singleplot_fast[0]
#frequency = sol_omegas15_singleplot_fast[0]

#wavenum = sol_ks15_singleplot_slow[0]
#frequency = sol_omegas15_singleplot_slow[0]

wavenum = sol_ks15_singleplot_fast_kink[0]
frequency = sol_omegas15_singleplot_fast_kink[0]

#wavenum = sol_ks15_singleplot_slow_kink[0]
#frequency = sol_omegas15_singleplot_slow_kink[0]


#####################################################
#####################################################
r0=0.  #mean
dr=1.5 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 1.    # 1 = kink , 0 = sausage

lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########
#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################

def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

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

normalised_left_P_solution_15 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution_15 = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
      
          
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_15 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_15 = inside_xi_solution/np.amax(abs(left_xi_solution))

inside_xi_phi_15 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_15 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency**2 - wavenum**2*vA_e**2)))
radial_xi_phi_15 = np.concatenate((inside_xi_phi_15, outside_xi_phi_15), axis=None)    

normalised_inside_xi_phi_15 = inside_xi_phi_15/np.amax(abs(outside_xi_phi_15))
normalised_outside_xi_phi_15 = outside_xi_phi_15/np.amax(abs(outside_xi_phi_15))
normalised_radial_xi_phi_15 = np.concatenate((normalised_inside_xi_phi_15, normalised_outside_xi_phi_15), axis=None)    



####################################################

#wavenum = sol_ks1_singleplot_fast[0]
#frequency = sol_omegas1_singleplot_fast[0]

#wavenum = sol_ks1_singleplot_slow[0]
#frequency = sol_omegas1_singleplot_slow[0]

wavenum = sol_ks1_singleplot_fast_kink[0]
frequency = sol_omegas1_singleplot_fast_kink[0]

#wavenum = sol_ks1_singleplot_slow_kink[0]
#frequency = sol_omegas1_singleplot_slow_kink[0]


#####################################################
#####################################################
r0=0.  #mean
dr=1. #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 1.    # 1 = kink , 0 = sausage

lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################
def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

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

normalised_left_P_solution_1 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution_1 = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 


def dP_dr_i(P_i, r_i):
      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
      
          
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_1 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_1 = inside_xi_solution/np.amax(abs(left_xi_solution))

inside_xi_phi_1 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_1 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency**2 - wavenum**2*vA_e**2)))
radial_xi_phi_1 = np.concatenate((inside_xi_phi_1, outside_xi_phi_1), axis=None)    

normalised_inside_xi_phi_1 = inside_xi_phi_1/np.amax(abs(outside_xi_phi_1))
normalised_outside_xi_phi_1 = outside_xi_phi_1/np.amax(abs(outside_xi_phi_1))
normalised_radial_xi_phi_1 = np.concatenate((normalised_inside_xi_phi_1, normalised_outside_xi_phi_1), axis=None)    



####################################################

#wavenum = sol_ks06_singleplot_fast[0]
#frequency = sol_omegas06_singleplot_fast[0]

#wavenum = sol_ks06_singleplot_slow[0]
#frequency = sol_omegas06_singleplot_slow[0]

wavenum = sol_ks06_singleplot_fast_kink[0]
frequency = sol_omegas06_singleplot_fast_kink[0]

#wavenum = sol_ks06_singleplot_slow_kink[0]
#frequency = sol_omegas06_singleplot_slow_kink[0]


#####################################################
#####################################################
r0=0.  #mean
dr=0.6 #standard dev

rr=sym.symbols('r')   #In order to differentiate profile we need to use sympy notation using symbols


def v_z(r):                  # Define the internal alfven speed
    return (U_e + ((U_i0 - U_e)*sym.exp(-(r-r0)**2/dr**2)))  

m = 1.    # 1 = kink , 0 = sausage

lx = np.linspace(3.*2.*np.pi/wavenum, 1., 500.)  # Number of wavelengths/2*pi accomodated in the domain

m_e = ((((wavenum**2*vA_e**2)-frequency**2)*((wavenum**2*c_e**2)-frequency**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e**2)-frequency**2)))
    
xi_e_const = -1/(rho_e*((wavenum**2*vA_e**2)-frequency**2))

      ######   BEGIN DIFFERENTIABLE FUNCTIONS   ##########

#############

def f_B(r): 
  return (m*B_iphi(r)/r + wavenum*B_i(r))

f_B_np=sym.lambdify(rr,f_B(rr),"numpy")

def g_B(r): 
  return (m*B_i(r)/r + wavenum*B_iphi(r))

g_B_np=sym.lambdify(rr,g_B(rr),"numpy")

#################

def shift_freq(r):
  return (frequency - (m*v_iphi(r)/r) - wavenum*v_z(r))
  
def alfven_freq(r):
  return ((m*B_iphi(r)/r)+(wavenum*B_i(r))/(sym.sqrt(rho_i(r))))

def cusp_freq(r):
  return ((alfven_freq(r)*c_i(r))/(sym.sqrt(c_i(r)**2+vA_i(r)**2)))

shift_freq_np=sym.lambdify(rr,shift_freq(rr),"numpy")
alfven_freq_np=sym.lambdify(rr,alfven_freq(rr),"numpy")
cusp_freq_np=sym.lambdify(rr,cusp_freq(rr),"numpy")

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

normalised_left_P_solution_06 = left_P_solution/np.amax(abs(left_P_solution))
normalised_left_xi_solution_06 = left_xi_solution/np.amax(abs(left_xi_solution))
left_bound_P = left_P_solution[-1] 

def dP_dr_i(P_i, r_i):
      return [P_i[1], ((-dF_np(r_i)/F_np(r_i))*P_i[1] + (g_np(r_i)/F_np(r_i))*P_i[0])]
      
          
def objective_dPi(dPi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
      U = odeint(dP_dr_i, [left_bound_P, dPi], ix)
      u1 = U[:,0]    #for sausage mode we solve P' = 0 at centre of cylinder    1 for sausage 0 for kink
      return u1[-1] 
      
dPi, = fsolve(objective_dPi, 0.001) 

# now solve with optimal dvx

Is = odeint(dP_dr_i, [left_bound_P, dPi], ix)
inside_P_solution = Is[:,0]
inside_xi_solution = (C1_np(ix)*Is[:,0] + D_np(ix)*Is[:,1])/C3_np(ix)     # Pressure perturbation solution for left hand side

normalised_inside_P_solution_06 = inside_P_solution/np.amax(abs(left_P_solution))
normalised_inside_xi_solution_06 = inside_xi_solution/np.amax(abs(left_xi_solution))


inside_xi_phi_06 = (((g_B_np(ix[::-1])*inside_P_solution[::-1]-2.*B_i_np(ix[::-1])*0.*(inside_xi_solution[::-1]/ix[::-1]))/(rho_i0*(shift_freq_np(ix[::-1])**2 - alfven_freq_np(ix[::-1])**2))))/B_i_np(ix[::-1])
outside_xi_phi_06 = (m*left_P_solution[::-1])/(lx[::-1]*(rho_e*(frequency**2 - wavenum**2*vA_e**2)))
radial_xi_phi_06 = np.concatenate((inside_xi_phi_06, outside_xi_phi_06), axis=None)    

normalised_inside_xi_phi_06 = inside_xi_phi_06/np.amax(abs(outside_xi_phi_06))
normalised_outside_xi_phi_06 = outside_xi_phi_06/np.amax(abs(outside_xi_phi_06))
normalised_radial_xi_phi_06 = np.concatenate((normalised_inside_xi_phi_06, normalised_outside_xi_phi_06), axis=None)    



##################################################################

###################################################################
B = 1.

#fig, (ax, ax2) = plt.subplots(2,1, sharex=False) 
fig, (ax, ax2, ax3) = plt.subplots(3,1, sharex=False) 

ax.axvline(x=-B, color='r', linestyle='--')
ax.axvline(x=B, color='r', linestyle='--')
#ax.set_xlabel("$x$")
ax.set_ylabel("$\hat{P}_T$", fontsize=18, rotation=0, labelpad=15)
ax.set_ylim(0.,1.)
ax.set_xlim(0.,1.5)
ax.plot(lx, normalised_left_P_solution_1e5, 'k')
ax.plot(ix, normalised_inside_P_solution_1e5, 'k')
#ax.plot(-lx, -normalised_left_P_solution_1e5, 'k--')
#ax.plot(-ix, -normalised_inside_P_solution_1e5, 'k--')


ax.plot(lx, normalised_left_P_solution_3, linestyle='solid', color='goldenrod')
ax.plot(ix, normalised_inside_P_solution_3, linestyle='solid', color='goldenrod')
#ax.plot(-lx, -normalised_left_P_solution_3, linestyle='dashed', color='goldenrod')
#ax.plot(-ix, -normalised_inside_P_solution_3, linestyle='dashed', color='goldenrod')


ax.plot(lx, normalised_left_P_solution_15, 'b')
ax.plot(ix, normalised_inside_P_solution_15, 'b')
#ax.plot(-lx, -normalised_left_P_solution_15, 'b--')
#ax.plot(-ix, -normalised_inside_P_solution_15, 'b--')


ax.plot(lx, normalised_left_P_solution_1, linestyle='solid', color='magenta')
ax.plot(ix, normalised_inside_P_solution_1, linestyle='solid', color='magenta')
#ax.plot(-lx, -normalised_left_P_solution_1, linestyle='dashed', color='magenta')
#ax.plot(-ix, -normalised_inside_P_solution_1, linestyle='dashed', color='magenta')


ax.plot(lx, normalised_left_P_solution_06, 'r')
ax.plot(ix, normalised_inside_P_solution_06, 'r')
#ax.plot(-lx, -normalised_left_P_solution_06, 'r--')
#ax.plot(-ix, -normalised_inside_P_solution_06, 'r--')



ax2.axvline(x=-B, color='r', linestyle='--')
ax2.axvline(x=B, color='r', linestyle='--')
ax2.set_xlabel("$r$", fontsize=18)
ax2.set_ylabel("$\hat{\u03be}_r$", fontsize=18, rotation=0, labelpad=15)

ax2.set_ylim(0.,2.5)
ax2.set_xlim(0.,1.5)

ax2.plot(lx, normalised_left_xi_solution_1e5, 'k')
ax2.plot(ix, normalised_inside_xi_solution_1e5, 'k')
#ax2.plot(-lx, normalised_left_xi_solution_1e5, 'k--')
#ax2.plot(-ix, normalised_inside_xi_solution_1e5, 'k--')


ax2.plot(lx, normalised_left_xi_solution_3, linestyle='solid', color='goldenrod')
ax2.plot(ix, normalised_inside_xi_solution_3, linestyle='solid', color='goldenrod')
#ax2.plot(-lx, normalised_left_xi_solution_3, linestyle='dashed', color='goldenrod')
#ax2.plot(-ix, normalised_inside_xi_solution_3, linestyle='dashed', color='goldenrod')


ax2.plot(lx, normalised_left_xi_solution_15, 'b')
ax2.plot(ix, normalised_inside_xi_solution_15, 'b')
#ax2.plot(-lx, normalised_left_xi_solution_15, 'b--')
#ax2.plot(-ix, normalised_inside_xi_solution_15, 'b--')


ax2.plot(lx, normalised_left_xi_solution_1, linestyle='solid', color='magenta')
ax2.plot(ix, normalised_inside_xi_solution_1, linestyle='solid', color='magenta')
#ax2.plot(-lx, normalised_left_xi_solution_1, linestyle='dashed', color='magenta')
#ax2.plot(-ix, normalised_inside_xi_solution_1, linestyle='dashed', color='magenta')


ax2.plot(lx, normalised_left_xi_solution_06, 'r')
ax2.plot(ix, normalised_inside_xi_solution_06, 'r')
#ax2.plot(-lx, normalised_left_xi_solution_06, 'r--')
#ax2.plot(-ix, normalised_inside_xi_solution_06, 'r--')


ax3.axvline(x=-B, color='r', linestyle='--')
ax3.axvline(x=B, color='r', linestyle='--')
ax3.set_xlabel("$r$", fontsize=18)
ax3.set_ylabel("$\hat{\u03be}_{\u03C6}$", fontsize=18, rotation=0, labelpad=15)

ax3.set_ylim(-1.,7.)
ax3.set_xlim(0.,1.5)

ax3.plot(lx[::-1], normalised_outside_xi_phi_1e5, 'k')
ax3.plot(ix[::-1], normalised_inside_xi_phi_1e5, 'k')

ax3.plot(lx[::-1], normalised_outside_xi_phi_3, color='goldenrod')
ax3.plot(ix[::-1], normalised_inside_xi_phi_3, color='goldenrod')

ax3.plot(lx[::-1], normalised_outside_xi_phi_15, 'b')
ax3.plot(ix[::-1], normalised_inside_xi_phi_15, 'b')

ax3.plot(lx[::-1], normalised_outside_xi_phi_1, linestyle='solid', color='magenta')
ax3.plot(ix[::-1], normalised_inside_xi_phi_1, linestyle='solid', color='magenta')

ax3.plot(lx[::-1], normalised_outside_xi_phi_06, 'r')
ax3.plot(ix[::-1], normalised_inside_xi_phi_06, 'r')


plt.savefig("flow005_fast_kink_eigenfunctions_k3.png")

plt.show()
exit()

