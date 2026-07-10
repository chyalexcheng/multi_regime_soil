import numpy as np
import matplotlib.pyplot as plt

# Norsand input
gamma = 0.75  # [0.9 - 1.4] 'altitude of CSL @ 1kPa'
lambda_val = 0.015  # [0.01 - 0.07] 'slope CSL defined on natural log'
M_tc = 1.26  # [1.2 - 1.5] 'critical state friction ratio triaxial compression'
N = 0.4  # [0.2 - 0.5] 'volumetric coupling coefficient'
H = 200.0  # [25 - 500] 'plastic hardening modulus for loading'
Xim_tc = 8  # [2 - 5] 'relates maximum dilatancy to state variable (psi)'
Ir = 200.0  # [100 - 600] 'shear rigidity'
nu = 0.20  # [0.1 0.3] 'Poissons ratio'

# CD input
p0 = 100.0  # [kPa] 'inital pressure'
e0 = 0.68  # [-] 'init void ratio'
Phi_0 = 1.0 / (1.0 + e0)
eqp_tot = 100.0  # [%] 'total applied plastic deviatoric strain'
lst = int(1e5)  # [-] 'loadsteps'

# Derived parameters
eqp_inc = eqp_tot / (1e2 * lst)  # [-] 'increment applied plastic deviatoric strain'
ratio = 1 / (1 - N) ** ((N - 1) / N)  # [-] 'ratio mean effective stress (p) and image stress (pim)'
Xim = Xim_tc / (1 - Xim_tc * lambda_val / M_tc)  # [-] 'relationship between soil property Xi_tc to Norsand property Xi'

# Declarations
p = np.zeros(lst + 1)
q = np.zeros(lst + 1)
pim = np.zeros(lst + 1)
ev = np.zeros(lst + 1)
eq = np.zeros(lst + 1)
e = np.zeros(lst + 1)
psi = np.zeros(lst + 1)

# Initial conditions
p[0] = p0
pim[0] = ratio * p0
e[0] = e0
psi[0] = e0 - (gamma - lambda_val * np.log(pim[0])) + lambda_val * np.log(pim[0] / p[0])

# Loadstep cycle CD test
for i in range(lst):
    # Update image state
    e[i + 1] = e0 - (1 + e0) * ev[i]
    psi[i + 1] = e[i + 1] - (gamma - lambda_val * np.log(pim[i])) + lambda_val * np.log(pim[i] / p[i])

    # Update image friction ratio
    Mim = M_tc + Xim * N * np.abs(psi[i + 1])

    # Apply hardening increment
    pim_max = p[i] * np.exp(-Xim * psi[i + 1] / M_tc)
    pim_inc = H * (pim_max - pim[i]) * eqp_inc
    pim[i + 1] = pim[i] + pim_inc

    # Calculate plastic volumetric strain increment
    Dp = Mim - q[i] / p[i]
    evp_inc = Dp * eqp_inc

    # Calculate bulk and shear modulus
    mu = Ir * p[i]
    K = mu * (2 * (1 + nu)) / (3 * (1 - 2 * nu))

    # Apply consistency condition to calculate stress ratio increment (drained)
    eta_inc = (pim_inc / pim[i]) / (1 / M_tc + 1 / (1e5 - q[i] / p[i]))  # Note: 3.0 is used instead of 3 in the denominator
    # Calculate mean effective stress
    p_inc = p[i] * eta_inc / (1e5 - q[i] / p[i])
    p[i + 1] = p[i] + p_inc

    # Calculate new stress ratio and shear stress
    eta = M_tc * (1 + np.log(pim[i + 1] / p[i + 1]))
    q[i + 1] = eta * p[i + 1]

    # Update volumetric and deviatoric strain increments
    eve_inc = p_inc / K
    eqe_inc = (q[i + 1] - q[i]) / (3 * mu)
    ev[i + 1] = ev[i] + eve_inc + evp_inc
    eq[i + 1] = eq[i] + eqe_inc + eqp_inc

# Display results
plt.figure('CU test results')
plt.subplot(2, 4, 1)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, p)
plt.xlabel('axial strain [%]')
plt.ylabel('volumetric stress [kPa]')

plt.subplot(2, 4, 2)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, q)
plt.xlabel('axial strain [%]')
plt.ylabel('deviatoric stress [kPa]')

plt.subplot(2, 4, 3)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, 1e2 * ev)
plt.xlabel('axial strain [%]')
plt.ylabel('volumetric strain [%]')

plt.subplot(2, 4, 4)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, 1e2 * eq)
plt.xlabel('axial strain [%]')
plt.ylabel('deviatoric strain [%]')

plt.subplot(2, 4, 5)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, pim)
plt.xlabel('axial strain [%]')
plt.ylabel('image stress [kPa]')

plt.subplot(2, 4, 6)
plt.plot(p, q, '-*b')
plt.plot([0, np.max(p)], [0, M_tc * np.max(p)], '-+r')
plt.xlabel('volumetric stress [kPa]')
plt.ylabel('deviatoric stress [kPa]')

plt.subplot(2, 4, 7)
plt.plot(p, e)
plt.xlabel('volumetric stress [kPa]')
plt.ylabel('void ratio [-]')

plt.subplot(2, 4, 8)
plt.plot(np.arange(lst + 1) / lst * eqp_tot, psi)
plt.xlabel('axial strain [%]')
plt.ylabel('image state variable (psi) [-]')

plt.show()

