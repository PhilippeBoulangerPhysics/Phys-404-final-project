import numpy as np
import matplotlib.pyplot as plt
sigma = 5.67e-8  # MB constant, W m^-2 K^-4
from scipy.optimize import fsolve
from scipy.special import gammaincc

# Let's start by making Figure 1

def equation30(tau_0,tau_rc,betan4):
    """Equation 30 written as lhs-rhs = 0 where we return (lhs-rhs). Used to solve it using fsolve.

    Args:
        tau_0 (float): Value of tau_0
        tau_rc (float): Value of tau_rc
        betan4 (float): Value of 4*beta/n

    Returns:
        float : Value of lhs - rhs of equation 30.
    """
    # Define lhs of Eq. 30.
    lhs = (tau_0/tau_rc)**(betan4)*np.exp(-D*(tau_0-tau_rc))*(1 + (np.exp(D*tau_0)/(D*tau_0)**(betan4))*(gammaincc(1+betan4,D*tau_rc)-gammaincc(1+betan4,D*tau_0)))
    #Define the rhs of Eq. 30
    rhs = (2+D*tau_rc)/(1+D*tau_rc)
    # Calculate lhs - rhs
    f = lhs-rhs
    return f

def solve_for_tau(tau_rc,betan4,guess):
    """Solves Equation 30 to find tau_0.

    Args:
        tau_rc (float): Value of tau_rc
        betan4 (float): Value = 4* beta /n
        guess (float): Initial guess for Newton's method solver
    """
    # Solve Eq. 30 to find the corresponding tau_0
    tau0 = fsolve(equation30,guess,(tau_rc,betan4))
    return tau0

D = 1.66 # Set value of D
# Define our ranging values of
# 4 * beta / n
betan = np.linspace(0.2,1,50)
# D * taur_rc
Dtaurc = np.array([0.01,0.2,0.5,1,2])
# initial guesses for tau_0 (from the plot)
guesses = np.array([0.06,0.5,2/3,1.3,1.6])/D
# Values of tau_rc
taurc = Dtaurc/D
# loop for all constant D * tau_rc contours
for i in range(len(Dtaurc)):
    # Define array to store our values of tau_0
    tau0_arr = np.zeros_like(betan)
    for j in range(len(betan)):
        # Solve for tau_0 
        tau0_tmp = solve_for_tau(taurc[i],betan[j],guesses[i])
        tau0_arr[j] = tau0_tmp
    # Plot the contour
    plt.plot(betan,D*tau0_arr,label = r"$D \cdot \tau_{rc}$" + f"{Dtaurc[i]}")
    
# Set up the plot ptoperly.
plt.ylim(10,0.01)
plt.xlim(0.2,1)
plt.yscale("log")
plt.legend()
plt.xlabel(r"$4 \beta /n$")
plt.ylabel(r"$D \cdot \tau_0$")
plt.savefig("figure1.png",dpi = 300)


# Figure 2

def equation31(tau_rc,betan4):
    """Set up equation 31 as lhs-rhs = 0.

    Args:
        tau_rc (float): Value of tau_rc
        betan4 (float): Value = 4* beta /n

    Returns:
        float: lhs - rhs
    """
    rhs = (2+D*tau_rc)/(1+D*tau_rc)
    lhs = gammaincc(1+betan4,D*tau_rc)/(np.exp(-D*tau_rc)*(D*tau_rc)**betan4)
    return lhs - rhs

def solve_for_taurc(betan4,guess):
    """Solves Equation 31 to find tau_rc.

    Args:
        tau_rc (float): Value of tau_rc
        betan4 (float): Value = 4* beta /n
        guess (float): Initial guess for Newton's method solver
    """
    rc = fsolve(equation31,guess,betan4)
    return rc

# Make array to save tau_rc values
taurc_arr = np.zeros_like(betan)
# Set up the linear guesses
guess_linear = 10**(4*betan-2.8)
for j in range(len(betan)):
    # Solve for tau_rc
    taurc_tmp = solve_for_taurc(betan[j],guess_linear[j])
    taurc_arr[j] = taurc_tmp
    
plt.clf()
plt.plot(betan,D*taurc_arr)
plt.ylim(10,0.01)
plt.xlim(0.2,1)
plt.xlabel(r"$4 \beta /n$")
plt.ylabel(r"$D \cdot \tau_{rc}$")
plt.yscale("log")
plt.savefig("figure2.png",dpi = 300)
plt.close()
plt.clf()

# let's see why it doesnt converge
# make range au possible taurc
tau_test = np.linspace(0.01,10,100)/D
for b in [0.2,0.3,0.5,0.6,0.7,0.8,0.9]:
    result = equation31(tau_test,b)
    plt.plot(D*tau_test,result,label = r"$4\cdot \beta /n$" + f" = {b}")
plt.xlabel(r"$D\cdot \tau_{rc}$")
plt.ylabel("result of equation 31")
plt.ylim(-1,1)
plt.xlim(0,10)
plt.axhline(0,0,1000,color = "black", linestyle = "dashed")
plt.legend()
plt.savefig("figureextra.png",dpi = 300)
plt.close()
plt.clf()


# Figure 3
# We set from the paper
tau0 = 2
n=2
def get_temp(tau,F1,D,k1,Fi=0):
    """Calculates the radiative equilibirum temperature using a modified Eq. 18 and Eq. 24 when k = 0
        Returns sigma * T^4
    Args:
        tau (np array): Array of the optical depth values.
        F1 (float): Value of stellar flux F_1
        D (float): Values of D
        k1 (float): Values of k_1
        Fi (int, optional): Values of F_i. Defaults to 0.

    Returns:
        radiative equilibirum temperature : W/m^2
    """
    if k1 != 0:
        # Implement equation 18
        term2 = F1/2 * ( 1 + D/k1 + (k1/D - D/k1)*np.exp(-k1*tau))
        term4 = Fi/2 * (1 + D*tau)
        return term2+term4
    else: 
        # Implement equation 24
        return (F+Fi)/2 * (1+D*tau)
# Set up tau space
tau = np.linspace(0,tau0,100)
plt.clf()
# For each value of k, calculate the temperature profile
for kD in [0.5,10,0.1,0]:
    k = kD*D
    F = 1370
    # Get temperature
    T = get_temp(tau,F,D,k)
    temp = T/F
    # Conver to p/p0
    p = (tau/tau0)**(1/n)
    # Plot
    plt.plot(temp,p,label = f"k/D = {k/D}")
plt.yscale("log")
plt.xscale("log")
plt.xlim(right = 10)
plt.xlabel(r"$\sigma T(p)^4 /F$")
plt.ylabel(r"$p/p_0$")
plt.ylim(1,0.1)
plt.legend()
plt.tight_layout()
plt.savefig("figure3.png",dpi = 300)

# Venus
# Set values
gamma = 1.3
alpha = 0.8

T0 = 730
p0 = 92
Fi=0
n = np.array([1,2])
beta = alpha*(gamma-1)/gamma
betan = np.array([0.7,0.4])
tau0= np.array([400,2e5])
taurc= np.array([1,0.1])
p_rc = p0*(taurc/tau0)**(1/n)
F = 160

def get_temp_conv(p,p0,T0,beta):
    """Calculates the temperature in the convection regime

    Args:
        p (np array): pressure
        p0 (float): surface pressure
        T0 (float): surface temperature
        beta (float): value of beta

    Returns:
        np.array : temperature profile
    """
    return T0*(p/p0)**(beta)

def get_temp_rad(p,tau0,F,Fi,D,n_):
    """Calculates the temperature in the radiation regime

    Args:
        p (np array ): pressure
        tau0 (_type_): value fo tau0
        F (float): Radiative flux
        Fi (float): value of F_i
        D (float): value of D
        n_ (int): value of n

    Returns:
        np array : temperature
    """
    # Calculate sigma* T^4
    tmp =(F+Fi)/2 * (1+D*(tau0*(p/p0)**n_))
    # Return T
    return (tmp/sigma)**(1/4)
plt.clf()
# Loop for both values of tau0 and n
for i in range(2):
    # Set values of taurc, tau0 and n
    tau0_cur = tau0[i]
    taurc_cur = taurc[i]
    n_cur = n[i]
    # Define radiative and convective regions
    p_rad = np.linspace(0,p_rc[i],100)
    p_conv = np.linspace(p_rc[i],p0,1000)[1:]
    # Calculate the temperature profile for both regions
    temp_conv = get_temp_conv(p_conv,p0,T0,beta)
    temp_rad = get_temp_rad(p_rad,tau0_cur,F,0,D,n_cur)
    # Bring everything back
    p = np.append(p_rad,p_conv)
    temp = np.append(temp_rad,temp_conv)
    # Plot
    plt.plot(temp,p,label = f"n = {n_cur}")
plt.yscale("log")
plt.ylim(100,0.01)
plt.xlabel("T [K]")
plt.ylabel("p [bar]")
plt.legend()
plt.xlim(150,800)
plt.savefig("figure8.png",dpi = 300)