import numpy as np
import matplotlib.pyplot as plt
sigma = 5.67e-8  # MB constant, W m^-2 K^-4
from scipy.optimize import fsolve
from scipy.special import gammaincc

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

def get_temp_rad(p,p0,tau0,F1,F2,D,k1,k2,Fi=0):
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
    term2 = F1/2 * ( 1 + D/k1 + (k1/D - D/k1)*np.exp(-k1*tau0* (p/p0)))
    term3 = F2/2 * ( 1 + D/k2 + (k2/D - D/k2)*np.exp(-k2*tau0* (p/p0)))
    term4 = Fi/2 * (1 + D*tau0* (p/p0))
    t = term2+term3+term4
    return (t/sigma)**(1/4)

def equation21(taus):
    tau_rc, tau0 = taus
    lhs = sigma*(T0**4) *(tau_rc/tau0)**(4*beta/n)
    term2 = F1/2 * ( 1 + D/k1 + (k1/D - D/k1)*np.exp(-k1*tau_rc))
    term3 = F2/2 * ( 1 + D/k2 + (k2/D - D/k2)*np.exp(-k2*tau_rc))
    term4 = Fi/2 * (1 + D*tau_rc)
    rhs = term2+term3+term4
    return lhs - rhs

def equation19(taurc):
    term1 = (F1/2)*(1+D/k1 + (1 - D / k1) * np.exp(-k1 * taurc))
    term2 = (F2/2)*(1+D/k2 + (1 - D / k2) * np.exp(-k2 * taurc))
    term3 = (Fi/2)*(2+D*taurc)
    return term1 + term2 + term3

def equation13(taurc,tau0):
    coef = sigma*T0**4*np.exp(-D*taurc)
    term1 = np.exp(-D*tau0)
    term2 = (1/(D*tau0)**(4*beta)) * gammaincc(1 + 4*beta,D*taurc)
    term3 = (1/(D*tau0)**(4*beta)) * gammaincc(1 + (4*beta),D*tau0)
    return coef * (term1 + term2 - term3)

def condition(taus):
    taurc,tau0 = taus
    return equation19(taurc) - equation13(taurc,tau0)

def system_tau(vars):
    taurc,tau0 = vars

    return [equation21([taurc,tau0]),condition([taurc,tau0])]

Fi = 0  # Earth has no internal energy source
n = 1  # Pressure dependence of opacity, given
alpha = 6/9  # Ratio of moist lapse rate (6) to dry adiabatic lapse rate
gamma = 1.4 # Ratio of cp to cv for nitrogen and oxygen, approximate
beta = 0.171  # Function of alpha and gamma, approximate
p0 = 101.3  # Given, kPa
T0 = 288  # Given, K
D = 1.66  # Trial use, in paper
F1 = 1500-1370
k1 = 100
F2 = 1370
k2 = 1e-2


taurc_test = np.linspace(0,1,500)
tau0_test = np.linspace(0,2,500)
# make a grid
X, Y = np.meshgrid(taurc_test, tau0_test)
# eval fct on grid
eq21 = equation21([X, Y])
cond = condition([X, Y])
# Plot 
fig,axs = plt.subplots(1,1)
contour2 = axs.contourf(taurc_test, tau0_test, cond, cmap = "coolwarm", levels=50)
contour1 = axs.contourf(taurc_test, tau0_test, eq21,cmap = "gist_ncar", levels=np.linspace(-500,500,101))
fig.suptitle(" Condition 1 and 2's region where  they cross 0")
axs.set_xlabel(r"$\tau_{rc}$")
axs.set_ylabel(r"$\tau_{0}$")
axs.set_title(f"k1 = {k1}, k2 = {k2}, D= {D}")
plt.colorbar(contour2)
plt.colorbar(contour1)
plt.savefig(f"param_space k1 = {k1}, k2 = {k2}, D= {D}.png",dpi = 300)
plt.show()

# Solve
# Initial guess
initial_guess = [2/3, 1.1]
taurc,tau0 = fsolve(system_tau, initial_guess)
print(" ")
print(taurc,tau0)
print(" ")


p_rc = p0*(taurc/tau0)**(1/n)
p = np.linspace(0,p0, 100)
# Define radiative and convective regions
p_rad = np.linspace(0,p_rc,1000)
p_conv = np.linspace(p_rc,p0,100)[1:]
# Calculate the temperature profile for both regions
temp_conv = get_temp_conv(p_conv,p0,T0,beta)
temp_rad = get_temp_rad(p_rad,p0,tau0,F1,F2,D,k1,k2,Fi=0)
p = np.append(p_rad,p_conv)
temp = np.append(temp_rad,temp_conv)
# Plot

plt.plot(get_temp_conv(p,p0,T0,beta),p,label = "conv")
plt.plot(get_temp_rad(p,p0,tau0,F1,F2,D,k1,k2,Fi=0),p,label = "rad")
plt.plot(temp,p,label = "full profile")
plt.yscale("log")
plt.legend()#
plt.xlim(100,800)
plt.xlabel("T [K]")
plt.ylabel("P [kPa]")
plt.ylim(p0,0.01)
plt.savefig(f"figure8Earth k1 = {k1}, k2 = {k2}, D= {D}.png.png")
plt.clf()
plt.close()

