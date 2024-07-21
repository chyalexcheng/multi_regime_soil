# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# Cam-clay critical state parameters
pc_0 = 150.0  # 'Initial consolidation pressure [kPa]'
M = 0.95  # 'Critical friction angle'
lambda_val = 0.2  # 'Lambda'
kappa = 0.04  # 'kappa'
N = 2.5  # 'Intercept of the normal consolidation line'
nu = 0.15  # 'Poisson ratio'

# Initial conditions
p0 = 150.0  # 'Initial confining pressure [kPa]'
V = N - (lambda_val * np.log(pc_0)) + (kappa * np.log(pc_0 / p0))  # Specific Volume
e0 = V - 1  # Initial void ratio
Phi_0 = 1.0 / (1.0 + e0)  # Initial solid volume fraction
deformation_mode = 'drained'

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
    # np.linspace(eqp_inc, 100 * eqp_inc, round(1.0 / eqp_inc)),
    # np.ones(round(1.0 / eqp_inc)) * 100 * eqp_inc,
    # np.linspace(100 * eqp_inc, eqp_inc, round(0.5 / eqp_inc)),
    # np.ones(round(0.5 / eqp_inc)) * eqp_inc,
])
load_length = eqp_inc_history.shape[0]

plt.figure('Load history')
plt.plot(np.arange(load_length) * dt, eqp_inc_history / dt, '-b')
plt.xlabel('Time [s]')
plt.ylabel('Shear strain rate [1/s]')

# Declarations
p = np.zeros(load_length)
q = np.zeros(load_length)
u = np.zeros(load_length)
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
OCR = pc_0 / p0  # Over Consolidation Ratio

# Initialize state variables
pc = pc_0
p[0] = p0
p_total[0] = p0
e[0] = e0
sigma = np.array([p0, p0, p0, 0.0, 0.0, 0.0])
epsilon = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Initialize the yield surf surface
yield_surf = (q[0] ** 2 / M ** 2 + p[0] ** 2) - p[0] * pc

# Loadstep cycle pressure control shear
for i, eqp_inc in enumerate(eqp_inc_history[:-1]):
    # Stiffness matrix and strain increment vector and derivatives of the yield function
    De = np.zeros([6, 6])
    df_ds = np.zeros([6, 1])
    df_dep = np.zeros([6, 1])

    # Calculate the bulk and shear modulus
    K = V * p[i] / kappa  # Bulk Modulus
    G = (3 * K * (1 - 2 * nu)) / (2 * (1 + nu))  # Shear Modulus
    if yield_surf == 0:
        pc = (q[i] ** 2 / M ** 2 + p[i] ** 2) / p[i]
    else:
        pc = pc_0

    # Elastic Stiffness and other Matrix
    for m in range(6):
        for n in range(6):
            if m <= 2:
                if yield_surf < 0:
                    df_ds[m, 0] = 0
                    df_dep[m, 0] = 0
                else:
                    df_ds[m, 0] = (2 * p[i] - pc) / 3 + 3 * (sigma[m] - p[i]) / M ** 2
                    df_dep[m, 0] = (-p[i]) * pc * (1 + e[i]) / (lambda_val - kappa) * 1
                if m == n:
                    De[m, n] = K + 4 / 3 * G
                elif n <= 2:
                    De[m, n] = K - 2 / 3 * G
            if m > 2:
                df_ds[m, 0] = 0
                df_dep[m, 0] = 0
                if m == n:
                    De[m, n] = G
                else:
                    De[m, n] = 0

        # If the yield surface is negative, the stiffness matrix is elastic
        if yield_surf < 0:
            D = De
        # If the yield surface is positive, the stiffness matrix is elastic-plastic
        else:
            D = De - (De.dot(df_ds).dot(df_ds.T).dot(De)) / (
                    -(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds))

    # Fill the strain increment vector
    if deformation_mode == 'drained':
        # For drained (or pressure control) conditions
        d_epsilon = np.array(
            [eqp_inc, - D[1, 0] / (D[1, 1] + D[1, 2]) * eqp_inc, - D[2, 0] / (D[2, 1] + D[2, 2]) * eqp_inc, 0., 0., 0.])
    elif deformation_mode == 'undrained':
        d_epsilon = np.array([eqp_inc, -eqp_inc / 2., -eqp_inc / 2., 0., 0., 0.])

    # Update stress
    d_sigma = D.dot(d_epsilon)
    sigma += d_sigma
    epsilon += d_epsilon

    # Compute stress and strain invariants
    ev[i + 1] = np.sum(epsilon[:3])
    epsilon_s = epsilon - np.array([1, 1, 1, 0, 0, 0]) * ev[i + 1]
    # eq[i + 1] = np.sqrt(2. * epsilon_s.dot(epsilon_s))
    eq[i + 1] = 2./3. * (epsilon[0] - epsilon[2])

    p[i + 1] = np.sum(sigma[:3]) / 3.0
    p_s = sigma - np.array([1., 1., 1., 0, 0, 0]) * p[i + 1]
    # q[i + 1] = np.sqrt(1. / 2. * p_s.dot(p_s))
    q[i + 1] = sigma[0] - sigma[2]
    u[i + 1] = p0 + q[i + 1] / 3. - p[i + 1]

    # Update specific volume
    V = N - (lambda_val * np.log(pc)) + (kappa * np.log(pc / p[i + 1]))
    e[i + 1] = V - 1.0

    if yield_surf < 0:
        yield_surf = q[i + 1] ** 2 + M ** 2 * p[i + 1] ** 2 - M ** 2 * p[i + 1] * pc
    else:
        yield_surf = 0

    # # Normal Consolidation Line (NCL)
    # pNCL = np.linspace(1, pc, nIter)
    # qNCL = M * pNCL
    # eNCL = (N - lambda_val * np.log(pNCL)) - 1
    #
    # # Critical State Line (CSL)
    # pCSL = pNCL
    # Gamma = 1 + e[i] + lambda_val * np.log(p[i])
    # eCSL = (Gamma - lambda_val * np.log(pCSL)) - 1
    #
    # # Final Yield Surface
    # # CSL in p-q space
    # p_fyield = np.linspace(1, pc, nIter)
    # q_fyield = M * p_fyield
    # # Plot the final yield locus
    # qyf = (M ** 2 * (pc * p_fyield - p_fyield ** 2)) ** 0.5

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
