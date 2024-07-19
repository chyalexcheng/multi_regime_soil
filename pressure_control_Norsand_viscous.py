import numpy as np
import matplotlib.pyplot as plt

# Norsand critical state parameters
gamma = 0.75  # [0.9 - 1.4] 'altitude of CSL @ 1kPa'
lambda_val = 0.015  # [0.01 - 0.07] 'slope CSL defined on natural log'
M = 1.26  # [1.2 - 1.5] 'critical state friction ratio triaxial compression'
N = 0.4  # [0.2 - 0.5] 'volumetric coupling coefficient'
H = 200.0  # [25 - 500] 'plastic hardening modulus for loading'
XiM = 8  # [2 - 5] 'relates maximum dilatancy to state variable (psi)'
Ir = 200.0  # [100 - 600] 'shear rigidity'
nu = 0.20  # [0.1 0.3] 'Poissons ratio'

# Initial conditions
p0 = 100.0  # [kPa] 'inital pressure'
e0 = 0.68  # [-] 'init void ratio'
Phi_0 = 1.0 / (1.0 + e0)

# Maximum shear strain and number of load steps for the quasi-static stage
eqp_tot = 100.0  # [%] 'total plastic shear strain'
time = 100.0  # [s] 'second'
load_length = int(1e5)  # [-] 'loadsteps'
dt = time / load_length

# Collisional contribution parameters
Delta_Phi = 1.0
M_c = 1.8

# Define a loading history
eqp_inc = eqp_tot / (1e2 * load_length)  # [-] 'increment applied plastic Plastic shear strain'
eqp_inc_history = np.concatenate([
    np.ones(round(1.0 / eqp_inc)) * eqp_inc,
    np.linspace(eqp_inc, 100 * eqp_inc, round(1.0 / eqp_inc)),
    np.ones(round(1.0 / eqp_inc)) * 100 * eqp_inc,
    np.linspace(100 * eqp_inc, eqp_inc, round(0.5 / eqp_inc)),
    np.ones(round(0.5 / eqp_inc)) * eqp_inc,
])
load_length = eqp_inc_history.shape[0]

plt.figure('Load history')
plt.plot(np.arange(load_length) * dt, eqp_inc_history / dt, '-b')
plt.xlabel('Time [s]')
plt.ylabel('Shear strain rate [1/s]')

# Declarations
p = np.zeros(load_length)
q = np.zeros(load_length)
pim = np.zeros(load_length)
ev = np.zeros(load_length)
eq = np.zeros(load_length)
e = np.zeros(load_length)
psi = np.zeros(load_length)
d_p_c = np.zeros(load_length)
p_c = np.zeros(load_length)
q_c = np.zeros(load_length)
p_total = np.zeros(load_length)
q_total = np.zeros(load_length)

# Derived parameters
ratio = 1 / (1 - N) ** ((N - 1) / N)  # [-] 'ratio mean effective stress (p) and image stress (pim)'
Xim = XiM / (1 - XiM * lambda_val / M)  # [-] 'relationship between soil property Xi_tc to Norsand property Xi'

# Initialize state variables
p[0] = p0
p_total[0] = p0
pim[0] = ratio * p0
e[0] = e0
psi[0] = e0 - (gamma - lambda_val * np.log(pim[0])) + lambda_val * np.log(pim[0] / p[0])

# Loadstep cycle pressure control shear
for i, eqp_inc in enumerate(eqp_inc_history[:-1]):
    # Update image state
    e[i + 1] = e0 - (1 + e0) * ev[i]
    psi[i + 1] = e[i + 1] - (gamma - lambda_val * np.log(pim[i])) + lambda_val * np.log(pim[i] / p[i])

    # Update image friction ratio
    Mim = M + Xim * N * np.abs(psi[i + 1])

    # Apply hardening increment
    pim_max = p[i] * np.exp(-Xim * psi[i + 1] / M)
    pim_inc = H * (pim_max - pim[i]) * eqp_inc
    pim[i + 1] = pim[i] + pim_inc

    # Calculate plastic volumetric strain increment
    Dp = Mim - q[i] / p[i]
    evp_inc = Dp * eqp_inc

    # Calculate bulk and shear modulus
    mu = Ir * p[i]
    K = mu * (2 * (1 + nu)) / (3 * (1 - 2 * nu))

    # Apply consistency condition to calculate stress ratio increment (drained)
    eta_inc = (pim_inc / pim[i]) / (
            1 / M + 1 / (1e5 - q[i] / p[i]))  # Note: 3.0 is used instead of 3 in the denominator
    # Calculate mean effective stress
    p_total_inc = p_total[i] * eta_inc / (1e5 - q_total[i] / p_total[i])
    p_total[i + 1] = p_total[i] + p_total_inc
    p_inc = p_total_inc - d_p_c[i]
    p[i + 1] = p[i] + p_inc

    # Calculate new stress ratio and shear stress
    eta = M * (1 + np.log(pim[i + 1] / p[i + 1]))
    q[i + 1] = eta * p[i + 1]

    # Calculate collisional stress
    d_eqp_inc = eqp_inc - eqp_inc_history[i - 1] if i > 0 else 0
    if d_eqp_inc != 0:
        Phi = 1.0 / (1.0 + e[i + 1])
        sign = np.sign(d_eqp_inc * eq[i])
        evp_inc += - Phi_0 / Phi ** 2 * Delta_Phi * d_eqp_inc
        d_p_c[i + 1] = p_total[i] / (lambda_val * Phi ** 2) * Delta_Phi * d_eqp_inc
        p_c[i + 1] = p_c[i] + d_p_c[i + 1]
        q_c[i + 1] = q_c[i] + d_p_c[i + 1] * M_c
    else:
        p_c[i + 1] = p_c[i]
        q_c[i + 1] = q_c[i]

    p_total[i + 1] = p[i + 1] + p_c[i + 1]
    q_total[i + 1] = q[i + 1] + q_c[i + 1]

    # Update volumetric and Plastic shear strain increments
    eve_inc = p_inc / K
    eqe_inc = (q[i + 1] - q[i]) / (3 * mu)
    ev[i + 1] = ev[i] + eve_inc + evp_inc
    eq[i + 1] = eq[i] + eqe_inc + eqp_inc

# Display results
plt.figure('Pressure controlled simple shear')
plt.subplot(2, 4, 1)
plt.plot(np.arange(load_length) * dt, p, '-b', label='Quasi-static')
plt.plot(np.arange(load_length) * dt, p_total, '-g', label='Total')
plt.xlabel('Time [s]')
plt.ylabel('Volumetric stress [kPa]')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(np.arange(load_length) * dt, q, '-b', label='Quasi-static')
plt.plot(np.arange(load_length) * dt, q_total, '-g', label='Total')
plt.xlabel('Time [s]')
plt.ylabel('Deviatoric stress [kPa]')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(np.arange(load_length) * dt, ev, '-b')
plt.xlabel('Time [s]')
plt.ylabel('Volumetric strain [-]')

plt.subplot(2, 4, 4)
plt.plot(np.arange(load_length) * dt, psi, '-b')
plt.xlabel('Time [s]')
plt.ylabel('State variable (psi) [-]')

plt.subplot(2, 4, 5)
plt.plot(np.arange(load_length) * dt, p_c, '-b')
plt.xlabel('Time [s]')
plt.ylabel('Collisional volumetric stress [kPa]')

plt.subplot(2, 4, 6)
plt.plot(p, q, '-*b', markevery=100, label='Quasi-static')
plt.plot(p_total, q_total, '-g', label='Total')
plt.plot([0, np.max(p)], [0, M * np.max(p)], '-+r', label='CSL')
plt.xlabel('Volumetric stress [kPa]')
plt.ylabel('Deviatoric stress [kPa]')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(p, e, '-b', label='Quasi-static')
plt.plot(p_total, e, '-g', label='Total')
plt.xlabel('Volumetric stress [kPa]')
plt.ylabel('void ratio [-]')
plt.legend()

plt.subplot(2, 4, 8)
plt.plot(np.arange(load_length) * dt, q_total / p_total, 'b')
plt.xlabel('Time [s]')
plt.ylabel('Stress ratio q/p [-]')

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.97, right=0.97, left=0.06, bottom=0.06)
plt.show()
