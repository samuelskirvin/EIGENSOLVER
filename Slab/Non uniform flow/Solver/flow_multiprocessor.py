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

#######################################################################

##############      BEGIN STEADY FLOW SLAB DISPERSION DIAGRAM       ###############


vA_i = 1.
c_i = 2.*vA_i/3.     #0.3*vA_i  coronal   #2.*vA_i/3.  photospheric 
vA_e = 0. #2.5*vA_i     #2.5*vA_i coronal   #0.  photospheric 
c_e = 3.*vA_i/4    #0.2*vA_i  coronal    #3.*vA_i/4  photospheric 

U_i = 0. #0.35*vA_i     #0.35*vA_i  coronal    
U_e = -0.15*vA_i           #-0.15*vA_i   photospheric

gamma=5./3.

rho_i = 1.
rho_e = rho_i*(c_i**2+gamma*0.5*vA_i**2)/(c_e**2+gamma*0.5*vA_e**2)

#rho_e = 2.27*rho_i          #2.27*rho_i  photospheric
#print('external density    =', rho_e)

R1 = rho_e/rho_i
#R1 = 0.8

print('Density ration external/internal    =', R1)


def cT_i():
    return np.sqrt(c_i**2 / (c_i**2 + vA_i**2))

def cT_e():
    return np.sqrt(c_e**2 * vA_e**2 / vA_i**2*(c_e**2 + vA_e**2))


a_alfven_i = 1.
a_alfven_e = vA_e/vA_i

a_sound_i = c_i/vA_i
a_sound_e = c_e/vA_i

mach_i = U_i  #/vA_i
mach_e = U_e  #/vA_i


# variables (reference only):
# W = omega/k
# K = k*x_0


def m0(W):
    return np.sqrt((c_i**2 - (W-mach_i)**2)*(vA_i**2 - (W-mach_i)**2) / ((c_i**2 + vA_i**2)*(cT_i()**2 - (W-mach_i)**2)))

def me(W):
    return np.sqrt((c_e**2 - (W-mach_e)**2)*(vA_e**2 - (W-mach_e)**2) / ((c_e**2 + vA_e**2)*(cT_e()**2 - (W-mach_e)**2)))

def n0(W):
    return np.sqrt(abs((c_i**2 - (W-mach_i)**2)*(vA_i**2 - (W-mach_i)**2) / ((c_i**2 + vA_i**2)*(cT_i()**2 - (W-mach_i)**2))))


def disp_rel_sausage(W,K):
    return R1*(vA_e**2-(W-mach_e)**2)*m0(W)*np.tanh(K*m0(W))/(me(W)*(vA_i**2 - (W-mach_i)**2)) +1

def disp_rel_kink(W,K):
    return R1*(vA_e**2-(W-mach_e)**2)*m0(W)/(np.tanh(K*m0(W))*me(W)*(vA_i**2 - (W-mach_i)**2)) +1
  
def disp_rel_sausage_body(W,K):
    return R1*(vA_e**2-(W-mach_e)**2)*n0(W)*np.tan(K*n0(W))/(me(W)*(vA_i**2 - (W-mach_i)**2)) -1

def disp_rel_kink_body(W,K):
    return R1*(vA_e**2-(W-mach_e)**2)*n0(W)/(np.tan(K*n0(W))*me(W)*(vA_i**2 - (W-mach_i)**2)) +1
  


Dmin = 0.
Dmax = 3.5
    
#Wmin = complex(0,0)    #Produces body modes??
#Wmax = complex(1.8,1.8)

Wmin = 0.
Wmax = 3.
 

step=0.001  
body_step = 0.0001 

D_range = np.linspace(Dmin, Dmax, 71)
#W_range = np.linspace(Wmin, Wmax, 51)
W_range = np.arange(Wmin, Wmax, step)


W_body_range = np.arange(cT_i(), c_i, body_step)

#W_body_range = np.arange(c_e, vA_i, body_step)   #Use for coronal  --   np.arange(cT_i(), c_i, body_step)
W_body_range2 = np.arange(vA_i, vA_e, body_step)   #Use for coronal

W_array_sausage = []
x_out_sausage = []

W_array_kink = []
x_out_kink = []

W_array_sausage_body = []
x_out_sausage_body = []

W_array_kink_body = []
x_out_kink_body = []

for x in D_range:         #solve dispersion relation using root finding bisection method 
  for V in W_range:
        
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_sausage = disp_rel_sausage(V_1, x) * disp_rel_sausage(V_2, x)
        
    if bisect_sausage < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_sausage = (V_1 + V_2) / 2
      
      x_out_sausage.append(x)
      W_array_sausage.append(midpoint_V_sausage)
    
      #W_array = W_array.append(midpoint_V)    #append root (solution) to solution array
      
    else:
      pass
      #print('no solution, boo!')

for x in D_range:         #solve dispersion relation using root finding bisection method 
  for V in W_range:
        
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_kink = disp_rel_kink(V_1, x) * disp_rel_kink(V_2, x)
    
        
    if bisect_kink < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_kink = (V_1 + V_2) / 2
      
      x_out_kink.append(x)
      W_array_kink.append(midpoint_V_kink)
         
    else:
      pass
      #print('no solution, boo!')

##### Begin body  ##############
for x in D_range:         #solve dispersion relation using root finding bisection method 
  for V in W_body_range:
         
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_sausage_body = disp_rel_sausage_body(V_1, x) * disp_rel_sausage_body(V_2, x)
        
    if bisect_sausage_body < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_sausage_body = (V_1 + V_2) / 2
      
      x_out_sausage_body.append(x)
      W_array_sausage_body.append(midpoint_V_sausage_body)
      
  for V in W_body_range2:
         
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_sausage_body = disp_rel_sausage_body(V_1, x) * disp_rel_sausage_body(V_2, x)
        
    if bisect_sausage_body < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_sausage_body = (V_1 + V_2) / 2
      
      x_out_sausage_body.append(x)
      W_array_sausage_body.append(midpoint_V_sausage_body)
 


for x in D_range:         #solve dispersion relation using root finding bisection method 
  for V in W_body_range:
          
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_kink_body = disp_rel_kink_body(V_1, x) * disp_rel_kink_body(V_2, x)
    
    #print('bisect =', bisect)
        
    if bisect_kink_body < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_kink_body = (V_1 + V_2) / 2
      
      x_out_kink_body.append(x)
      W_array_kink_body.append(midpoint_V_kink_body)
    
      #W_array = W_array.append(midpoint_V)    #append root (solution) to solution array
      
  for V in W_body_range2:
          
    V_1 = V          # V here is phase speed      
    V_2 = V_1 + step    #define new w/k value to be current value + small step size
    
    bisect_kink_body = disp_rel_kink_body(V_1, x) * disp_rel_kink_body(V_2, x)
    
    #print('bisect =', bisect)
        
    if bisect_kink_body < 0 :   #if product of two values is less than zero then there is a sign change and root in between
      #print('yay a solution!')
      midpoint_V_kink_body = (V_1 + V_2) / 2
      
      x_out_kink_body.append(x)
      W_array_kink_body.append(midpoint_V_kink_body)
    
      #W_array = W_array.append(midpoint_V)    #append root (solution) to solution array

      

#test_k = x_out_sausage[71]    #pick random point on dispersion curve for test
#test_W = W_array_sausage[71]   #remember this is w/k not frequency!!!!

#test_k = x_out_kink[15]    #pick random point on dispersion curve for test
#test_W = W_array_kink[15]   #remember this is w/k not frequency!!!!

W_array_kink_body1 = []
x_out_kink_body1 = []

for i in range(len(W_array_kink_body)):
    test_body_kink_sol = disp_rel_kink_body(W_array_kink_body[i],x_out_kink_body[i])
    
    if test_body_kink_sol < 0.0001:
        W_array_kink_body1.append(W_array_kink_body[i])
        x_out_kink_body1.append(x_out_kink_body[i])


W_array_sausage_body1 = []
x_out_sausage_body1 = []

for i in range(len(W_array_sausage_body)):
    test_body_sausage_sol = disp_rel_sausage_body(W_array_sausage_body[i],x_out_sausage_body[i])
    
    if test_body_sausage_sol < 0.0001:
        W_array_sausage_body1.append(W_array_sausage_body[i])
        x_out_sausage_body1.append(x_out_sausage_body[i])



###########
# Dispersion relation only valid for m_e > 0 therefore we check!

#print('m_e check    =', me(test_W))
#print('m_0 check    =', m0(test_W))

x_out_sausage=np.array(x_out_sausage)
W_array_sausage=np.array(W_array_sausage)
x_out_kink=np.array(x_out_kink)
W_array_kink=np.array(W_array_kink)
x_out_sausage_body1=np.array(x_out_sausage_body1)
W_array_sausage_body1=np.array(W_array_sausage_body1)
x_out_kink_body1=np.array(x_out_kink_body1)
W_array_kink_body1=np.array(W_array_kink_body1)

    
fig=plt.figure()
ax = plt.subplot(111)
#plt.title('Steady Flow Known Dispersion')
plt.xlabel("$kx_{0}$", fontsize=20)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=25, rotation=0, labelpad=15)
ax.plot(x_out_sausage,W_array_sausage, 'r.', markersize=4.)  
ax.plot(x_out_kink,W_array_kink, 'b.', markersize=4.) 
ax.plot(x_out_sausage_body1,W_array_sausage_body1, 'r.', markersize=4., label='_nolegend_')  # Solve only in region where Ct < w/k < Va
ax.plot(x_out_kink_body1,W_array_kink_body1, 'b.', markersize=4., label='_nolegend_') 
ax.axhline(y=vA_e, color='k', label='$v_{Ae}$', linestyle='dashdot')
ax.axhline(y=vA_i, color='k', label='$v_{Ai}$', linestyle='dashdot')
ax.axhline(y=c_e, color='k', label='$c_{e}$', linestyle='dashdot')
ax.axhline(y=c_i, color='k', label='$c_{i}$', linestyle='dashdot')
ax.axhline(y=cT_e(), color='k', label='$c_{Te}$', linestyle='dashdot')
ax.axhline(y=cT_i(), color='k', label='$c_{Ti}$', linestyle='dashdot')
ax.axhline(y=c_e+U_e, color='k', linestyle='solid')

ax.annotate( '$c_{Ti}$', xy=(Dmax, cT_i()), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(Dmax, cT_e()), fontsize=20)
ax.annotate( '$c_{e}$', xy=(Dmax, c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(Dmax, c_i), fontsize=20)
#ax.annotate( '$v_{Ae}$', xy=(Dmax, vA_e), fontsize=20)
ax.annotate( '$v_{Ae},   c_{Te}$', xy=(Dmax, vA_e), fontsize=20)   #photospheric
ax.annotate( '$v_{Ai}$', xy=(Dmax, vA_i), fontsize=20)
ax.annotate( '$c_e - U_e$', xy=(Dmax, c_e+U_e), fontsize=20)   

#ax.plot(test_k,test_W, 'x') #test plot

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.85, box.height])

#ax.legend(('Sausage','Kink'), loc='center left', bbox_to_anchor=(1, 0.3))

plt.savefig("Nakariakov_known_dispersion.png")



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
#      plt.plot(wavenumber[i], freq[j]/wavenumber[i], 'b.')
#
#plt.plot(x_out_sausage,W_array_sausage, 'rx', markersize=2.)
#plt.plot(x_out_sausage_body,W_array_sausage_body, 'rx', markersize=2., label='_nolegend_')   
#plt.ylim(0, 2.)
#plt.xlim(0, 1.6)

#plt.figure()
#for i in range(len(wavenumber)):
#  for j in range(len(freq)):
#      
#      plt.plot(wavenumber[i], freq[j], 'b.')
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

p_tol = 1e-6   # i.e 1% tolerance

#We need to create an array of dvx values to guess over
#dvx_guess = np.linspace(-0.015, 0.015, 30)

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


image = []
image_kink = []

def sausage(wavenumber, sausage_ws, sausage_ks, freq): #, freq):
  #freq = np.logspace(0.001,0.55,500)-1   #0.3 as frequency range normalised between 0 and 1   #was 250

#####    DEFINE FUNCTIONS TO LOCATE SOLUTIONS    #####################

  ################################################
  def locate_sausage(omega, wavenum,itt_num):
  
      all_ws[:] = []
      itt_num = itt_num
                
      for k in range(len(omega)):
      
         if itt_num > 200:  #500
            break
            
         lx = np.linspace(-7*2*np.pi/wavenum, -1, 500)  # Number of wavelengths/2*pi accomodated in the domain
         ix = np.linspace(-1, 1, 500)  # inside slab x values
  
         m_i = ((((wavenum**2*vA_i**2)-(omega[k]-(wavenum*mach_i))**2)*((wavenum**2*c_i**2)-(omega[k]-(wavenum*mach_i))**2))/((vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-(omega[k]-(wavenum*mach_i))**2)))
         m_e = ((((wavenum**2*vA_e**2)-(omega[k]-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-(omega[k]-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-(omega[k]-(wavenum*mach_e))**2)))
      
         p_i_const = rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-(omega[k]-(wavenum*mach_i))**2)/((omega[k]-(wavenum*mach_i))*((wavenum**2*c_i**2)-(omega[k]-(wavenum*mach_i))**2))
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-(omega[k]-(wavenum*mach_e))**2)/((omega[k]-(wavenum*mach_e))*((wavenum**2*c_e**2)-(omega[k]-(wavenum*mach_e))**2))       
                
         if m_e < 0:
             print('non physical solution')
             #pass
                  
         else:
             loop_ws.append(omega[k])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
               
             V0 = [1e-8, 1e-15]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
             Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
             left_solution = Ls[:,0]*(omega[k]-wavenum*mach_i)/(omega[k]-wavenum*mach_e)       # Vx perturbation solution for left hand side
             left_P_solution = p_e_const*Ls[:,1]#*(omega[k]-wavenum*mach_i)/(omega[k]-wavenum*mach_e)   # Pressure perturbation solution for left hand side
           
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             left_bound_vx = left_solution[-1] 
                       
             def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], m_i*Vx_i[0]]
             
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                   u1 = U[:,0] + left_bound_vx
                   return u1[-1] 
                   
             dVxi, = fsolve(objective_dvxi, 0.5)    #zero is guess for roots of equation, maybe change to find more body modes??
                 
             Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
             inside_solution = Is[:,0]
             inside_P_solution = p_i_const*Is[:,1]
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
             #elif (abs(P_diff_loop_check[-2]) < abs(P_diff_loop_check[-1])) and (abs(P_diff_loop_check[-2]) < abs(P_diff_loop_check[-3])):
                  
                    omega = np.linspace(loop_ws[-2], loop_ws[-1], 3)
                    wavenum = wavenum
                    #now repeat exact same process but in foccused omega range
                    itt_num =itt_num +1
                    loop_ws[:] = []           
                    locate_sausage(omega, wavenum, itt_num)
                    
  
      
  ##############################################################################
  
  
  with stdout_redirected():
      #for i in range(len(wavenumber)):
        for j in range(len(freq)):
            loop_ws[:] = [] 
                      
            lx = np.linspace(-7*2*np.pi/wavenumber, -1, 500)  # Number of wavelengths/2*pi accomodated in the domain
            ix = np.linspace(-1, 1, 500)  # inside slab x values
      
            m_i = ((((wavenumber**2*vA_i**2)-(freq[j]-(wavenumber*mach_i))**2)*((wavenumber**2*c_i**2)-(freq[j]-(wavenumber*mach_i))**2))/((vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-(freq[j]-(wavenumber*mach_i))**2)))
            m_e = ((((wavenumber**2*vA_e**2)-(freq[j]-(wavenumber*mach_e))**2)*((wavenumber**2*c_e**2)-(freq[j]-(wavenumber*mach_e))**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-(freq[j]-(wavenumber*mach_e))**2)))
      
            p_i_const = rho_i*(vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-(freq[j]-(wavenumber*mach_i))**2)/((freq[j]-(wavenumber*mach_i))*((wavenumber**2*c_i**2)-(freq[j]-(wavenumber*mach_i))**2))
            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-(freq[j]-(wavenumber*mach_e))**2)/((freq[j]-(wavenumber*mach_e))*((wavenumber**2*c_e**2)-(freq[j]-(wavenumber*mach_e))**2))
              
            if m_e < 0:
              print('non physical solution')
              #pass                   
                  
            else: 
              
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8, 1e-15]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
              Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
              left_solution = Ls[:,0]*(freq[j]-wavenumber*mach_i)/(freq[j]-wavenumber*mach_e)   # Vx (displacement) perturbation solution for left boundary
              left_P_solution = p_e_const*Ls[:,1]#*(freq[j]-wavenumber[i]*mach_i)/(freq[j]-wavenumber[i]*mach_e) #Pressure perturbation solution for left hand side
                   #print('length of left', len(left_solution), left_solution[-1])
  
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              left_bound_vx = left_solution[-1]

              def dVx_dx_i(Vx_i, x_i):
                     return [Vx_i[1], m_i*Vx_i[0]]
              
                 
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                    u1 = U[:,0] + left_bound_vx
                    return u1[-1] 
                    
              dVxi, = fsolve(objective_dvxi, 0.5)    #zero is guess for roots of equation, maybe change to find more body modes??
      
              # now solve with optimal dvx
              
              Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
              inside_solution = Is[:,0]
              inside_P_solution = p_i_const*Is[:,1]
              normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
              normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
              
              P_diff_check.append((left_P_solution[-1]-inside_P_solution[0]))
              sign_check.append(P_diff_check[-1]*P_diff_check[-2])
                                      
              test_w_k_sausage.append(freq[j]/wavenumber)
              test_p_diff_sausage.append(abs(left_P_solution[-1]) - abs(inside_P_solution[0]))
              
              all_ks.append(wavenumber)
              all_ws.append(freq[j])
              
              if (abs(left_P_solution[-1] - inside_P_solution[0])*100/max(abs(left_P_solution[-1]), abs(inside_P_solution[0]))) < p_tol:
                    sol_omegas1.append(freq[j])
                    sol_ks1.append(wavenumber)
                    #sausage_ks.put(wavenumber)
                    #sausage_ws.put(freq[j])
                    p_diff_sols.append(abs(left_P_solution[-1] - inside_P_solution[0]))
                    all_ws[:] = []
                    
             
              if sign_check[-1] < 0: #and (round(inside_P_solution[0]) == round(inside_P_solution[-1])): 
                  if len(all_ws)>1:
                  
                     if all_ks[-1] != all_ks[-2]:
                       sign_check[:] = []
                       
                      
                     else:         
                       loop_omega = np.linspace(all_ws[-2], all_ws[-1], 3)
                       wavenum = all_ks[-1]
              #        #now repeat exact same process but in foccused omega range
                       itt_num = 0
                       all_ws[:] = []
                       #p_diff_start = P_diff_check[-2]
                       locate_sausage(loop_omega, wavenum, itt_num) 

  
  sausage_ks.put(sol_ks1)
  sausage_ws.put(sol_omegas1)

 
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



def kink(wavenumber, kink_ws, kink_ks, freq): #, freq):
  #freq = np.logspace(0.001,0.55,500)-1   #0.75 instead of 1?  
  
  def locate_kink(omega, wavenum,itt_num):
      all_ws[:] = []
      itt_num = itt_num
                 
      for k in range(len(omega)):
      
         if itt_num > 200: #500
            break
            
         lx = np.linspace(-7*2*np.pi/wavenum, -1, 500)  # Number of wavelengths/2*pi accomodated in the domain
         ix = np.linspace(-1, 1, 500)  # inside slab x values
  
         m_i = ((((wavenum**2*vA_i**2)-(omega[k]-(wavenum*mach_i))**2)*((wavenum**2*c_i**2)-(omega[k]-(wavenum*mach_i))**2))/((vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-(omega[k]-(wavenum*mach_i))**2)))
         m_e = ((((wavenum**2*vA_e**2)-(omega[k]-(wavenum*mach_e))**2)*((wavenum**2*c_e**2)-(omega[k]-(wavenum*mach_e))**2))/((vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-(omega[k]-(wavenum*mach_e))**2)))
      
         p_i_const = rho_i*(vA_i**2+c_i**2)*((wavenum**2*cT_i()**2)-(omega[k]-(wavenum*mach_i))**2)/((omega[k]-(wavenum*mach_i))*((wavenum**2*c_i**2)-(omega[k]-(wavenum*mach_i))**2))
         p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenum**2*cT_e()**2)-(omega[k]-(wavenum*mach_e))**2)/((omega[k]-(wavenum*mach_e))*((wavenum**2*c_e**2)-(omega[k]-(wavenum*mach_e))**2))       
                
         if m_e < 0:
             print('non physical solution')
             #pass
             
         else:
             loop_ws_kink.append(omega[k])
             
             def dVx_dx_e(Vx_e, x_e):
                    return [Vx_e[1], m_e*Vx_e[0]]
               
             V0 = [1e-8, 1e-15]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
             Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
             left_solution = Ls[:,0]*(omega[k]-wavenum*mach_i)/(omega[k]-wavenum*mach_e)       # Vx perturbation solution for left hand side
             #left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
             left_P_solution = p_e_const*Ls[:,1]  # Pressure perturbation solution for left hand side
           
             normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
             normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
             left_bound_vx = left_solution[-1] 
                       
             def dVx_dx_i(Vx_i, x_i):
                    return [Vx_i[1], m_i*Vx_i[0]]
             
                
             def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                   U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                   u1 = U[:,0] - left_bound_vx
                   return u1[-1] 
                   
             dVxi, = fsolve(objective_dvxi, 0.5)    #zero is guess for roots of equation, maybe change to find more body modes??
                 
             Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
             inside_solution = Is[:,0]
             inside_P_solution = p_i_const*Is[:,1]
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
      #for i in range(len(wavenumber)):
        for j in range(len(freq)):
        
            lx = np.linspace(-7*2*np.pi/wavenumber, -1, 500)  # Number of wavelengths/2*pi accomodated in the domain
            ix = np.linspace(-1, 1, 500)  # inside slab x values
  
            m_i = ((((wavenumber**2*vA_i**2)-(freq[j]-(wavenumber*mach_i))**2)*((wavenumber**2*c_i**2)-(freq[j]-(wavenumber*mach_i))**2))/((vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-(freq[j]-(wavenumber*mach_i))**2)))
            m_e = ((((wavenumber**2*vA_e**2)-(freq[j]-(wavenumber*mach_e))**2)*((wavenumber**2*c_e**2)-(freq[j]-(wavenumber*mach_e))**2))/((vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-(freq[j]-(wavenumber*mach_e))**2)))
      
            p_i_const = rho_i*(vA_i**2+c_i**2)*((wavenumber**2*cT_i()**2)-(freq[j]-(wavenumber*mach_i))**2)/((freq[j]-(wavenumber*mach_i))*((wavenumber**2*c_i**2)-(freq[j]-(wavenumber*mach_i))**2))
            p_e_const = rho_e*(vA_e**2+c_e**2)*((wavenumber**2*cT_e()**2)-(freq[j]-(wavenumber*mach_e))**2)/((freq[j]-(wavenumber*mach_e))*((wavenumber**2*c_e**2)-(freq[j]-(wavenumber*mach_e))**2))
                        
            if m_e < 0:
              print('non physical solution')
              #pass               
            
            else:
              def dVx_dx_e(Vx_e, x_e):
                     return [Vx_e[1], m_e*Vx_e[0]]
                
              V0 = [1e-8, 1e-15]   #Initial conditions of vx(0), vx'(0)  assume much much less than one at infinity
              Ls = odeint(dVx_dx_e, V0, lx, printmessg=0)
              left_solution = Ls[:,0]*(freq[j]-wavenumber*mach_i)/(freq[j]-wavenumber*mach_e)       # Vx perturbation solution for left hand side
              left_P_solution = p_e_const*Ls[:,1]   # Pressure perturbation solution for left hand side
                   #print('length of left', len(left_solution), left_solution[-1])
              
              normalised_left_P_solution = left_P_solution/np.amax(abs(left_P_solution))
              normalised_left_vx_solution = left_solution/np.amax(abs(left_solution))
              left_bound_vx = left_solution[-1] 
                                  
              def dVx_dx_i(Vx_i, x_i):
                     return [Vx_i[1], m_i*Vx_i[0]]
              
                 
              def objective_dvxi(dVxi):    # We know that Vx(0) = 0 however do not know value of dVx at boundary inside slab so use fsolve to calculate
                    U = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix, printmessg=0)
                    u1 = U[:,0] - left_bound_vx
                    return u1[-1] 
                    
              dVxi, = fsolve(objective_dvxi, 0.5)    #zero is guess for roots of equation, maybe change to find more body modes??
      
              # now solve with optimal dvx
              
              Is = odeint(dVx_dx_i, [left_bound_vx, dVxi], ix)
              inside_solution = Is[:,0]
              inside_P_solution = p_i_const*Is[:,1]
              normalised_inside_P_solution = inside_P_solution/np.amax(abs(left_P_solution))
              normalised_inside_vx_solution = inside_solution/np.amax(abs(left_solution))
              
              P_diff_check_kink.append((left_P_solution[-1]-inside_P_solution[0]))
              sign_check_kink.append(P_diff_check_kink[-1]*P_diff_check_kink[-2])
      
              test_p_diff_kink.append(abs(left_P_solution[-1]) - abs(inside_P_solution[0]))
  
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
                       loop_omega = np.linspace(all_ws_kink[-2], all_ws_kink[-1], 3)
                       wavenum = all_ks_kink[-1]
              #        #now repeat exact same process but in foccused omega range
                       itt_num = 0
                       all_ws_kink[:] = []
                       locate_kink(loop_omega, wavenum, itt_num) 
          
  kink_ks.put(sol_ks_kink1)
  kink_ws.put(sol_omegas_kink1)                    




wavenumber = np.linspace(0.01,3.5,350)      #(1.5, 1.8), 5
freq = np.logspace(0.001,0.55,80)-1   #0.3 as frequency range normalised between 0 and 1   #was 250


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    
    sausage_ws = multiprocessing.Queue()
    sausage_ks = multiprocessing.Queue()
    
    processes_kink = []
    
    kink_ws = multiprocessing.Queue()
    kink_ks = multiprocessing.Queue()

    
    for i in wavenumber:
        task = multiprocessing.Process(target=sausage, args=(i, sausage_ws, sausage_ks, freq))
        task_kink = multiprocessing.Process(target=kink, args=(i, kink_ws, kink_ks, freq))
        processes.append(task)
        processes_kink.append(task_kink)
        task.start()
        task_kink.start()
        
        #################   attempt to retrieve better body mode resolution
        body_freq = np.linspace(cT_i()*i, (c_e+U_e)*i, 100)
        
        body_task = multiprocessing.Process(target=sausage, args=(i, sausage_ws, sausage_ks, body_freq))
        body_task_kink = multiprocessing.Process(target=kink, args=(i, kink_ws, kink_ks, body_freq))
        processes.append(body_task)
        processes_kink.append(body_task_kink)
        body_task.start()
        body_task_kink.start()

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


  
fig=plt.figure()
ax = plt.subplot(111)
#plt.title('Sausage Mode flow test')
plt.xlabel("$kx_{0}$", fontsize=20)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=25, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'b.', markersize=6.)
ax.plot(x_out_sausage,W_array_sausage, 'rx', markersize=2.)  
ax.plot(x_out_sausage_body1,W_array_sausage_body1, 'rx', markersize=2., label='_nolegend_') 
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

plt.savefig("Nakariakov_test_dispersion_diagram_sausage_mp.png")



fig=plt.figure()
ax = plt.subplot(111)
#plt.title('Kink Mode flow test')
plt.xlabel("$kx_{0}$", fontsize=20)
plt.ylabel(r'$\frac{\omega}{k v_{Ai}}$', fontsize=25, rotation=0, labelpad=15)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=6.)
ax.plot(x_out_kink,W_array_kink, 'rx', markersize=2.)  
ax.plot(x_out_kink_body1,W_array_kink_body1, 'rx', markersize=2., label='_nolegend_') 
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

plt.savefig("Nakariakov_test_dispersion_diagram_kink_mp.png")

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
#plt.title('Dispersion Diagram')
plt.xlabel("$kx_{0}$", fontsize=18)
plt.ylabel(r'$ \frac{\omega}{k v_{Ai} }  $', fontsize=25, rotation=0, labelpad=15)
ax.plot(sol_ks1, (sol_omegas1/sol_ks1), 'r.', markersize=4.)
ax.plot(sol_ks_kink1, (sol_omegas_kink1/sol_ks_kink1), 'b.', markersize=4.)
ax.axhline(y=vA_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=vA_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_i, color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_e(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=cT_i(), color='k', linestyle='dashdot', label='_nolegend_')
ax.axhline(y=c_e+U_e, color='k', linestyle='solid', label='_nolegend_')

#ax.plot(test_k,test_W, 'x') #test plot

ax.annotate( '$c_{Ti}$', xy=(max(wavenumber), cT_i()), fontsize=20)
#ax.annotate( '$c_{Te}$', xy=(max(wavenumber), cT_e()), fontsize=20)
#ax.annotate( '$c_{e},   c_{Te}$', xy=(max(wavenumber), c_e), fontsize=20)
#ax.annotate( '$c_{i},   c_{Ti}$', xy=(max(wavenumber), c_i), fontsize=20)
ax.annotate( '$c_{e}$', xy=(max(wavenumber), c_e), fontsize=20)
ax.annotate( '$c_{i}$', xy=(max(wavenumber), c_i), fontsize=20)
ax.annotate( '$v_{Ae},   c_{Te}$', xy=(max(wavenumber), vA_e), fontsize=20)
ax.annotate( '$v_{Ai}$', xy=(max(wavenumber), vA_i), fontsize=20)  #photopshere
ax.annotate( '$c_e - U_e$', xy=(max(wavenumber), c_e+U_e), fontsize=20)

ax.set_ylim(0., vA_i+0.1)
ax.set_yticks([])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

#ax.legend(('Sausage','Kink'), loc='center left', bbox_to_anchor=(1, 0.7))

plt.savefig("flow_test_dispersion_diagram_FULL_mp.png")
print('Simulation took {} seconds'.format(time.time() - starttime))
