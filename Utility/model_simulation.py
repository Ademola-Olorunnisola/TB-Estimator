# -*- coding: utf-8 -*-
"""sveirt_simulation.py - SVEIRT TB Model with Observable Measurement Options"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

############################################################################################
# Global parameters
############################################################################################
N  = 220e6
Lambda = 37.9 / 1000 / 365    # birth/recruitment rate
mu     = 11.5 / 1000 / 365    # natural death rate
delta  = 268 / N          # TB disease-induced death rate
eps    = 1 / 365
phi    = 0.03
rho    = 0.21 * (1 / 180)






############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec,
          Lambda=Lambda, mu=mu, delta=delta, eps=eps, phi=phi, rho=rho,
          return_state_names=False):
        """
        Continuous time dynamics for SVEIRT TB model.

        Parameters
        ----------
        x_vec : array-like, shape (12,)
            State vector [S, V, E, I, T, R, beta, eps, gamma, sigma, phi, rho]
        u_vec : array-like, shape (2,)
            Control vector [alpha, tau]
            alpha : vaccination rate
            tau   : treatment rate
        Returns
        -------
        x_dot : numpy array, shape (12,)
        """
        if return_state_names:
            return ['S', 'V', 'E', 'I', 'T', 'R',
                    'beta', 'eps', 'gamma', 'sigma', 'phi', 'rho']

        # Extract states
        S     = x_vec[0]
        V     = x_vec[1]
        E     = x_vec[2]
        I     = x_vec[3]
        T     = x_vec[4]
        R     = x_vec[5]
        beta  = x_vec[6]
        eps   = x_vec[7]
        gamma = x_vec[8]
        sigma = x_vec[9]
        phi   = x_vec[10]
        rho   = x_vec[11]

        # Extract controls
        alpha = u_vec[0]   # vaccination rate
        tau   = u_vec[1]   # treatment rate

        lam = beta * I

        # f0 — drift dynamics (uncontrolled)
        f0_contribution = np.array([
            Lambda - lam*S - mu*S,
            -sigma*lam*V - mu*V,
            lam*S + sigma*lam*V + (phi*R*I) - (eps + mu)*E,
            eps*E + rho*T - (mu + delta)*I,
            -(gamma + rho + mu)*T,
            gamma*T - mu*R - (phi*R*I),
            0, 0, 0, 0, 0, 0
        ])

        # f1 — multiplied by control alpha (vaccination)
        f1_contribution = alpha * np.array([
            -S, S, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ])

        # f2 — multiplied by control tau (treatment)
        f2_contribution = tau * np.array([
            0, 0, 0, -I, I, 0,
            0, 0, 0, 0, 0, 0
        ])

        return f0_contribution + f1_contribution + f2_contribution

############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def observe_IVR(self, x_vec, u_vec, return_measurement_names=False):
        """Observe Infectious, Vaccinated, Recovered"""
        if return_measurement_names:
            return ['I', 'V', 'R']
        return np.array([x_vec[3], x_vec[1], x_vec[5]])

    def observe_IVRЕТ(self, x_vec, u_vec, return_measurement_names=False):
        """Observe Infectious, Vaccinated, Recovered, Exposed, Treatment"""
        if return_measurement_names:
            return ['I', 'V', 'R', 'E', 'T']
        return np.array([x_vec[3], x_vec[1], x_vec[5], x_vec[2], x_vec[4]])

    def observe_I(self, x_vec, u_vec, return_measurement_names=False):
        """Observe Infectious only"""
        if return_measurement_names:
            return ['I']
        return np.array([x_vec[3]])

############################################################################################
# Numeric parameters & initial conditions
############################################################################################

beta =  0.15   # Transmission rate (every Infectious person infects 2 others on average)
sigma =  0.40  # Vaccination Inefficiency (50% effective)
gamma = 1 / 180 # Treatment completion (6 months treatment)
eps = 1 / 365   # Latency progression rate
phi = 0.03      # Exogenous reinfection rate (i.e R faces 43% infection risk)
rho = 0.21 * (1 / 180) # treatment failure (21% Failure)



I0      = 361000 / N
V0      = 0.65                 # BCG coverage ~65% (WHO immunization data)
R0   = 0.02                 # Estimated recovered proportion
E0      = 3 * I0               # Exposed typically 3× infectious for TB
T0      = 0.6 * I0             # Those already on treatment
S0      = 1.0 - V0 - E0 - I0 - T0 - R0

# State vector includes parameters as states
x0 = np.array([S0, V0, E0, I0, T0, R0,
               beta, eps, gamma, sigma, phi, rho])

# Control inputs
alpha  = 0.01    
tau    = 0.005 
u_vec = np.array([alpha, tau])

print("State names :", F().f(x0, u_vec, return_state_names=True))
print("Measurement names (IVR)    :", H('observe_IVR').h(x0, u_vec, return_measurement_names=True))
print("Measurement names (IVRET)  :", H('observe_IVRЕТ').h(x0, u_vec, return_measurement_names=True))

R0_val = (beta * eps) / ((eps + mu) * (tau + mu + delta))
print(f"\nR0 = {R0_val:.4f}")

############################################################################################
# Solve ODE
############################################################################################
t = np.linspace(0, 365, 1000)

def ode_wrapper(x, t):
    return F().f(x, u_vec)

sol = odeint(ode_wrapper, x0, t)

S_n = sol[:, 0];  V_n = sol[:, 1];  E_n = sol[:, 2]
I_n = sol[:, 3];  T_n = sol[:, 4];  R_n = sol[:, 5]

############################################################################################
# Plots
############################################################################################
compartment_plots = [
    (S_n, 'Susceptible (S)',  '#2196F3'),
    (V_n, 'Vaccinated (V)',   '#9C27B0'),
    (E_n, 'Exposed (E)',      '#FF9800'),
    (I_n, 'Infectious (I)',   '#F44336'),
    (T_n, 'On Treatment (T)','#009688'),
    (R_n, 'Recovered (R)',    '#4CAF50'),
]

for y, title, color in compartment_plots:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(t, y, color=color, linewidth=2)
    ax.set_title(title, fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel("Days", fontsize=10, color='gray')
    ax.set_ylabel("People", fontsize=10, color='gray')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9, colors='gray')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    plt.show()
