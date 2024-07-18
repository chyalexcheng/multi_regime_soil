import numpy as np
import matplotlib.pyplot as plt

# Norsand input
gamma = 0.75  # [0.9 - 1.4] 'altitude of CSL @ 1kPa'
lambda_val = 0.015  # [0.01 - 0.07] 'slope CSL defined on natural log'
M_tc = 1.26  # [1.2 - 1.5] 'critical state friction ratio triaxial compression'
N = 0.4  # [0.2 - 0.5] 'volumetric coupling coefficient'
H = 200.0  # [25 - 500] 'plastic hardening modulus for loading'
Xim_tc = 8  # [2 - 5] 'relates maximum dilatancy to state variable (psi)'
Ir = 100.0  # [100 - 600] 'shear rigidity'
nu = 0.20  # [0.1 0.3] 'Poissons ratio'

# CD input
p0 = 100.0  # [kPa] 'inital pressure'
e0 = 0.7  # [-] 'init void ratio'
Phi_0 = 1.0 / (1.0 + e0)
eqp_tot = 100.0  # [%] 'total applied plastic deviatoric strain'
lst = int(1e5)  # [-] 'loadsteps'

# Collisional contribution parameters
Delta_Phi = -1.0
M_tc_c = 1.8

# Derived parameters
eqp_inc = eqp_tot / (1e2 * lst)  # [-] 'increment applied plastic Plastic shear strain'
# Keep eqp_inc constant until 30% and linearly increase Plastic shear strain increment to 10 * eqp_inc 
eqp_inc_history = np.concatenate([
    np.ones(round(1.0 / eqp_inc)) * eqp_inc,
    np.linspace(eqp_inc, 100 * eqp_inc, round(1.0 / eqp_inc)),
    np.ones(round(1.0 / eqp_inc)) * 100 * eqp_inc,
    np.linspace(100 * eqp_inc, eqp_inc, round(0.5 / eqp_inc)),
    np.ones(round(0.5 / eqp_inc)) * eqp_inc,
    ])
ratio = 1 / (1 - N) ** ((N - 1) / N)  # [-] 'ratio mean effective stress (p) and image stress (pim)'
Xim = Xim_tc / (1 - Xim_tc * lambda_val / M_tc)  # [-] 'relationship between soil property Xi_tc to Norsand property Xi'

# Declarations
lst = eqp_inc_history.shape[0]
p = np.zeros(lst + 1)
q = np.zeros(lst + 1)
pim = np.zeros(lst + 1)
ev = np.zeros(lst + 1)
eq = np.zeros(lst + 1)
e = np.zeros(lst + 1)
psi = np.zeros(lst + 1)
d_p_c = np.zeros(lst + 1)
p_c = np.zeros(lst + 1)
q_c = np.zeros(lst + 1)
p_total = np.zeros(lst + 1)
q_total = np.zeros(lst + 1)

# Initial conditions
p[0] = p0
p_total[0] = p0
pim[0] = ratio * p0
e[0] = e0
psi[0] = e0 - (gamma - lambda_val * np.log(pim[0])) + lambda_val * np.log(pim[0] / p[0])

# Loadstep cycle CU test
for i, eqp_inc in enumerate(eqp_inc_history):
    # Update image state
    e[i + 1] = e0 - (1 + e0) * ev[i]
    psi[i + 1] = e[i + 1] - (gamma - lambda_val * np.log(pim[i])) + lambda_val * np.log(pim[i] / p[i])

    # Update image friction ratio
    Mim = M_tc - Xim * N * np.abs(psi[i + 1])

    # Apply hardening increment
    pim_inc = pim[i] * H * Mim / M_tc * (p[i] / pim[i]) ** 2 * (np.exp(-Xim * psi[i + 1] / M_tc) - pim[i] / p[i]) * eqp_inc
    pim[i + 1] = pim[i] + pim_inc

    # Calculate plastic volumetric strain increment
    Dp = Mim - q[i] / p[i]
    evp_inc = Dp * eqp_inc

    # Calculate bulk and shear modulus
    mu = Ir * p[i]
    K = mu * (2 * (1 + nu)) / (3 * (1 - 2 * nu))

    # Calculate collisional stress
    d_eqp_inc = eqp_inc - eqp_inc_history[i-1]
    if d_eqp_inc != 0:
        Phi = 1.0/ (1.0 + e[i + 1])
        sign = np.sign(d_eqp_inc * eq[i])
        evp_inc += Phi_0 / Phi**2 * Delta_Phi * d_eqp_inc
        d_p_c[i + 1] = - p_total[i] / (lambda_val*Phi**2) * Delta_Phi * d_eqp_inc
        p_c[i + 1] = p_c[i] + d_p_c[i + 1]
        q_c[i + 1] = q_c[i] + d_p_c[i + 1] * M_tc_c
    else:
        p_c[i + 1] = p_c[i]
        q_c[i + 1] = q_c[i]

    # Calculate elastic strain increment % mean effective stress increment (undrained)
    eve_inc = -evp_inc
    p_inc = K * eve_inc

    # Apply consistency condition to calculate mean effective stress on the yield surface
    Cc = (pim[i] / p[i]) * (1 + pim_inc / pim[i] - p_inc / p[i])
    p_total[i + 1] = pim[i + 1] / Cc
    p[i + 1] = p[i] + p_inc - d_p_c[i]

    # Calculate the stress ratio & shear stress
    eta = Mim * (1 - np.log(p[i + 1] / pim[i + 1]))
    q[i + 1] = eta * p[i + 1]

    p_total[i + 1] = p[i + 1] + p_c[i + 1]
    q_total[i + 1] = q[i + 1] + q_c[i + 1]

    # Update volumetric and deviatoric strain increments
    eqe_inc = (q[i + 1] - q[i]) / (3 * mu)
    ev[i + 1] = ev[i] + eve_inc + evp_inc
    eq[i + 1] = eq[i] + eqe_inc + eqp_inc


# Display results
plt.figure('Volume controlled simple shear')
plt.subplot(2, 4, 1)
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, p, '-b', label='Quasi-static')
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, p_total, '-g', label='Total')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('Volumetric stress [kPa]')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, q, '-b', label='Quasi-static')
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, q_total, '-g', label='Total')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('Deviatoric stress [kPa]')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, 1e2 * ev, '-b')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('Volumetric strain [%]')

plt.subplot(2, 4, 4)
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, psi, '-b')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('State variable (psi) [-]')

plt.subplot(2, 4, 5)
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, p_c, '-b')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('Collisional volumetric stress [kPa]')

plt.subplot(2, 4, 6)
plt.plot(p, q, '-*b', markevery=100, label='Quasi-static')
plt.plot(p_total, q_total, '-g', label='Total')
plt.plot([0, np.max(p)], [0, M_tc * np.max(p)], '-+r', label='CSL')
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
plt.plot(np.arange(lst + 1) / lst * np.sum(eqp_inc_history) * 100, q_total/p_total, 'b')
plt.xlabel('Plastic shear strain [%]')
plt.ylabel('Stress ratio q/p [-]')

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.97, right=0.97, left=0.06, bottom=0.06)
plt.show()

