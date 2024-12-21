# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'

# Cam-clay critical state parameters
pc_0 = 150.0  # 'Initial consolidation pressure [kPa]'
M = 1.0  # 'Critical friction angle'
lambda_val = 0.2  # 'Lambda'
kappa = 0.04  # 'kappa'
N = 2.5  # 'Intercept of the normal consolidation line'
nu = 0.15  # 'Poisson ratio'

# Initial conditions
p0 = 150.0  # 'Initial confining pressure [kPa]'
V = N - (lambda_val * np.log(pc_0)) + (kappa * np.log(pc_0 / p0))  # Specific Volume
void_ratio_0 = V - 1  # Initial void ratio
Phi_0 = 1.0 / (1.0 + void_ratio_0)  # Initial solid volume fraction
deformation_mode = input('Enter the deformation mode: [1] drained and [2] undrained\n')

# Maximum shear strain and number of load steps for the quasi-static stage
eqp_tot = 100.0  # [%] 'total plastic shear strain'
time = 100.0  # [s] 'second'
load_length = int(1e5)  # [-] 'loadsteps'
dt = time / load_length

# Collisional contribution parameters
Delta_Phi = 50.0
M_c = 2.0
# TODO: for some reason, there is a factor of three missing...
M_c /= 3.0

# Define a loading history
eqp_inc = eqp_tot / (1e2 * load_length)  # [-] 'incrementl applied plastic shear strain'
eqp_inc_history = np.concatenate([
    np.ones(round(1.0 / eqp_inc)) * eqp_inc,
    np.linspace(eqp_inc, 100 * eqp_inc, round(0.5 / eqp_inc)),
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
void_ratio_total = np.zeros(load_length)
void_ratio_q = np.zeros(load_length)
p = np.zeros(load_length)
q = np.zeros(load_length)
u = np.zeros(load_length)
ev = np.zeros(load_length)
eq = np.zeros(load_length)
d_p_c = np.zeros(load_length)
p_c = np.zeros(load_length)
q_c = np.zeros(load_length)
p_total = np.zeros(load_length)
q_total = np.zeros(load_length)
pc_history = np.zeros(load_length)

# Derived parameters
OCR = pc_0 / p0  # Over Consolidation Ratio

# Initialize state variables
K_c = 0
G_c = 0
pc = pc_0
p[0] = p0
p_total[0] = p0
sigma = np.array([p0, p0, p0, 0.0, 0.0, 0.0])
sigma_q = np.array([p0, p0, p0, 0.0, 0.0, 0.0])
sigma_c = np.array([0, 0, 0, 0.0, 0.0, 0.0])
epsilon = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

de_v_c = 0
de_q_direction = np.zeros(6)
void_ratio_total[0] = void_ratio_0
void_ratio_q[0] = void_ratio_0

# Initialize the yield surf surface
yield_surf = (q[0] ** 2 / M ** 2 + p[0] ** 2) - p[0] * pc

# Loadstep cycle pressure control shear
for i, eqp_inc in enumerate(eqp_inc_history[:-1]):
    # Acceleration increment
    d_eqp_inc = eqp_inc - eqp_inc_history[i - 1] if i > 0 else 0

    # Stiffness matrix and strain increment vector and derivatives of the yield function
    De = np.zeros([6, 6])
    # TODO: HC: check if the stiffness matrix for collisional stress increments can be formulated like elasticity
    D_c = np.zeros([6, 6])
    df_ds = np.zeros([6, 1])
    df_dep = np.zeros([6, 1])

    # Calculate the bulk and shear modulus
    K = V * p[i] / kappa  # Bulk Modulus
    G = (3 * K * (1 - 2 * nu)) / (2 * (1 + nu))  # Shear Modulus

    # Update pre-consolidation pressure
    if yield_surf == 0:
        pc = (q[i] ** 2 / M ** 2 + p[i] ** 2) / p[i]
    else:
        pc = pc_0
    pc_history[i + 1] = pc

    # Elastic Stiffness and other Matrix
    for m in range(6):
        for n in range(6):
            if m <= 2:
                if yield_surf < 0:
                    df_ds[m, 0] = 0
                    df_dep[m, 0] = 0
                else:
                    df_ds[m, 0] = (2 * p[i] - pc) / 3 + 3 * (sigma_q[m] - p[i]) / M ** 2
                    df_dep[m, 0] = (-p[i]) * pc * (1 + void_ratio_q[i]) / (lambda_val - kappa) * 1
                if m == n:
                    De[m, n] = K + K_c + 4 / 3 * G
                    D_c[m, n] = 4 / 3 * G_c
                elif n <= 2:
                    De[m, n] = K + K_c - 2 / 3 * G
                    D_c[m, n] = - 2 / 3 * G_c
            if m > 2:
                df_ds[m, 0] = 0
                df_dep[m, 0] = 0
                if m == n:
                    De[m, n] = G
                    D_c[m, n] = G_c
                else:
                    De[m, n] = 0
                    D_c[m, n] = 0

        # If the yield surface is negative, the stiffness matrix is elastic
        if yield_surf < 0:
            D_q = De

        # If the yield surface is positive, the stiffness matrix is elastic-plastic
        else:
            D_q = De - (De.dot(df_ds).dot(df_ds.T).dot(De)) / (
                    -(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds))

    def compute_stress_increment(args):
        global K_c, G_c, de_v_c, de_q_direction
        ratio = args[0]
        # Fill the strain increment vector
        if deformation_mode == 'drained':
            # For drained (or pressure control) conditions
            d_epsilon = np.array(
                [eqp_inc, ratio * eqp_inc, ratio * eqp_inc, 0., 0., 0.])
            if d_eqp_inc != 0:
                d_d_epsilon = np.array(
                    [d_eqp_inc, ratio * d_eqp_inc, ratio * d_eqp_inc, 0., 0., 0.])
            else:
                d_d_epsilon = np.array([0., 0., 0., 0., 0., 0.])
        elif deformation_mode == 'undrained':
            d_epsilon = np.array(
                [eqp_inc, -eqp_inc / 2., -eqp_inc / 2., 0., 0., 0.])
            d_d_epsilon = np.array(
                [d_eqp_inc, -d_eqp_inc / 2., -d_eqp_inc / 2., 0., 0., 0.])

        # Get the direction of the deviatoric strain
        de_v = np.sum(d_epsilon[:3])
        de_q = d_epsilon - np.array([1., 1., 1., 0, 0, 0]) * de_v
        de_q_norm = np.sqrt(2. / 3. * de_q.dot(de_q))
        de_q_direction = de_q / de_q_norm

        # Calculate collisional stress
        if d_eqp_inc != 0:
            # Get sign of the acceleration rate, positive: accelerate; negative decelerate.
            # Now simplified as sign(d_eqp_inc). Should have been the sign between de_q_direction and e_q_direction tensors
            #sign = np.sign(d_eqp_inc * eq[i])

            Phi = 1.0 / (1.0 + (void_ratio_total[i]-void_ratio_q[i]))
            de_v_c = - Phi_0 / Phi ** 2 * Delta_Phi * d_eqp_inc
            d_p_c[i + 1] = p[i] / (lambda_val * Phi ** 2) * Delta_Phi * d_eqp_inc

            K_c = d_p_c[i + 1] / de_v_c
            G_c = d_p_c[i + 1] * M_c / d_eqp_inc
        else:
            p_c[i + 1] = p_c[i]
            q_c[i + 1] = q_c[i]
            K_c = 0
            G_c = 0

        d_sigma_q = D_q.dot(d_epsilon) - d_p_c[i + 1] * np.array([1., 1., 1., 0, 0, 0])
        d_sigma_c = D_c.dot(d_d_epsilon) + d_p_c[i + 1] * np.array([1., 1., 1., 0, 0, 0])

        return d_sigma_q, d_sigma_c, d_epsilon

    def servo_control(ratio):
        d_sigma_q, d_sigma_c, d_epsilon = compute_stress_increment(ratio)
        d_sigma = d_sigma_q + d_sigma_c
        return abs(np.sum(d_sigma[:3]))

    # Update stress
    ratio = - D_q[1, 0] / (D_q[1, 1] + D_q[1, 2])
    if deformation_mode == 'undrained':
        d_sigma_q, d_sigma_c, d_epsilon = compute_stress_increment([ratio])
    elif deformation_mode == 'drained':
        solution = root(servo_control, ratio)
        ratio = solution.x
        d_sigma_q, d_sigma_c, d_epsilon = compute_stress_increment(ratio)
    
    sigma_q += d_sigma_q
    sigma_c += d_sigma_c
    sigma = sigma_q + sigma_c # + d_p_c[i + 1] * np.array([1., 1., 1., 0, 0, 0]) + d_p_c[i + 1] * M_c * de_q_direction
    epsilon += d_epsilon  # + 1. / 3 * de_v_c * np.array([1., 1., 1., 0, 0, 0])

    # Update stress invariants
    p_total[i + 1] = np.sum(sigma[:3]) / 3.0
    p_s = sigma - np.array([1., 1., 1., 0, 0, 0]) * p_total[i + 1]
    q_total[i + 1] = np.sqrt(3. / 2. * p_s.dot(p_s))

    p_c[i + 1] = np.sum(sigma_c[:3]) / 3.0
    p_s_c = sigma_c - np.array([1., 1., 1., 0, 0, 0]) * p_c[i + 1]
    q_c[i + 1] = np.sqrt(3. / 2. * p_s_c.dot(p_s_c))

    # Compute stress and strain invariants
    ev[i + 1] = np.sum(epsilon[:3])
    epsilon_s = epsilon - np.array([1, 1, 1, 0, 0, 0]) * ev[i + 1]
    eq[i + 1] = np.sqrt(2. / 3. * epsilon_s.dot(epsilon_s))

    p[i + 1] = np.sum(sigma_q[:3]) / 3.0
    p_q_s = sigma_q - np.array([1., 1., 1., 0, 0, 0]) * p[i + 1]
    q[i + 1] = np.sqrt(3. / 2. * p_q_s.dot(p_q_s))
    u[i + 1] = p0 + q_total[i + 1] / 3. - p_total[i + 1]

    # Update specific volume
    V = N - (lambda_val * np.log(pc)) + (kappa * np.log(pc / p[i + 1]))
    void_ratio_q[i + 1] = V - 1
    void_ratio_total[i + 1] = void_ratio_0 - (1 + void_ratio_0) * ev[i + 1]

    if yield_surf < 0:
        yield_surf = q[i + 1] ** 2 + M ** 2 * p[i + 1] ** 2 - M ** 2 * p[i + 1] * pc
    else:
        yield_surf = 0

# Display results
plt.figure('Pressure controlled simple shear')

# Full text long labels
time_label = r'Time ($t$) [s]'
mu_label = r'Ratio of deviatoric stress to pressure $q/p$ [-]'
pressure_label = r'Pressure $p$ [kPa]'
dev_stress_label = r'Deviatoric stress $q$ [kPa]'
ev_label = r'Volumetric strain $\varepsilon_v$ [-]'
preconsolidation_p_label = 'Pre-consolidation stress [kPa]'
p_ratio_label = r'Ratio of collisional and quasi-static stresses [-]'
e_label = r'Void ratio $e$ [-]'

# Symbolic short labels
ctime = 't [s]'
cmu = 'q/p [-]'
cp = 'p [kPa]'
cq = 'q [kPa]'
cev = 'eps [-]'
cppc = 'pc(t) [kPa]'
cpcoll = 'p_c/p [-]'
cvr = 'e [-]'

# pressure vs. time
plt.subplot(2, 4, 1)
plt.plot(np.arange(load_length) * dt, p, '-b', label='qstat')
plt.plot(np.arange(load_length) * dt, p_total, '-g', label='tot')
# begin and endpoints
plt.plot(0 * dt, p[0], 'bx', label='')
plt.plot(0 * dt, p_total[0], 'g+', label='')
plt.plot(load_length * dt, p[-1], 'bo', label='')
plt.plot(load_length * dt, p_total[-1], 'g.', label='')
# labels
plt.xlabel(ctime)
plt.ylabel(cp)
plt.legend()

# deviatoric stress (q) vs. time
plt.subplot(2, 4, 2)
plt.plot(np.arange(load_length) * dt, q, '-b', label='qstat')
plt.plot(np.arange(load_length) * dt, q_total, '-g', label='tot')
# begin and endpoints
plt.plot(0 * dt, q[0], 'bx', label='')
plt.plot(0 * dt, q_total[0], 'g+', label='')
plt.plot(load_length * dt, q[-1], 'bo', label='')
plt.plot(load_length * dt, q_total[-1], 'g.', label='')
# labels
plt.xlabel(ctime)
plt.ylabel(cq)
plt.legend()

# volumetric and deviatoric strain vs. time
ax1 = plt.subplot(2, 4, 3)
ax1.plot(np.arange(load_length) * dt, eq, '-b', label='dev')
ax1.plot(0 * dt, eq[0], 'bx', label='')
ax1.plot(load_length * dt,eq[-1], 'bo', label='')
ax1.set_xlabel(ctime)
ax1.set_ylabel('Deviatoric Strain', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create the secondary y-axis
ax2 = ax1.twinx()

# Plot the second curve on the right y-axis
ax2.plot(np.arange(load_length) * dt, ev, '--g', label='v')
ax2.plot(0 * dt, ev[0], 'gx', label='')
ax2.plot(load_length * dt, ev[-1], 'g.', label='')
ax2.set_ylabel('Volumetric Strain', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Add legends for clarity
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# pre-consolidation pressure vs. time
plt.subplot(2, 4, 4)
plt.plot(np.arange(load_length) * dt, pc_history, '-b')
# begin and endpoints
plt.plot(0 * dt, pc_history[0], 'bx', label='')
plt.plot(load_length * dt, pc_history[-1], 'b.', label='')
# labels
plt.xlabel(ctime)
plt.ylabel(cppc)

# collisional stress (iso) vs. time
plt.subplot(2, 4, 5)
plt.plot(np.arange(load_length) * dt, p_c / p, '-b')
# begin and endpoints
plt.plot(0 * dt, p_c[0] / p[0], 'bx', label='')
plt.plot(load_length * dt, p_c[-1] / p[-1], 'b.', label='')
# labels
plt.xlabel(ctime)
plt.ylabel(cpcoll)

# deviatoric stress vs. p
plt.subplot(2, 4, 6)
imark = 500
plt.plot(p, q, '-+b', markevery=imark, label='qstat')
plt.plot(p_total, q_total, '-g', label='tot')
plt.plot([0, np.max(p)], [0, M * np.max(p)], '-+r', label='CSL')
# begin and endpoints
plt.plot(p[0], q[0], 'bx', label='')
plt.plot(p[-1], q[-1], 'b.', label='')
plt.plot(p_total[0], q_total[0], 'g+', label='')
plt.plot(p_total[-1], q_total[-1], 'g.', label='')
# plt.plot(p_total[-1], M * np.max(p), 'rs', label='')
# labels
plt.xlabel(cp)
plt.ylabel(cq)
plt.legend()

# void ratios vs. p
plt.subplot(2, 4, 7)
plt.plot(p, void_ratio_q, '-b', label='qstat')
plt.plot(p_total, void_ratio_total, '-g', label='tot')
# begin and endpoints
plt.plot(p[0], void_ratio_q[0], 'bx', label='')
plt.plot(p_total[0], void_ratio_total[0], 'g+', label='')
plt.plot(p[-1], void_ratio_q[-1], 'b.', label='')
plt.plot(p_total[-1], void_ratio_total[-1], 'g.', label='')
# labels
plt.xlabel(cp)
plt.ylabel(cvr)
# plt.legend()

# bulk friction vs. time
plt.subplot(2, 4, 8)
plt.plot(np.arange(load_length) * dt, q_total / p_total, 'b')
# begin and endpoints
plt.plot(0 * dt, q_total[0] / p_total[0], 'b+', label='')
plt.plot(load_length * dt, q_total[-1] / p_total[-1], 'b.', label='')
# labels
plt.xlabel(ctime)
plt.ylabel(cmu)


# Individual production plots

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)

# Pressure vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, p, '-b', label=r"Quasi-static pressure ($p^{\mathrm{q}}$)")
plt.plot(np.arange(load_length) * dt, p_total, '-g', label=r"Total pressure ($p$)")
plt.plot(0 * dt, p[0], 'b.', label='')
plt.plot(0 * dt, p_total[0], 'g.', label='')
plt.plot(load_length * dt, p[-1], 'bx', label='')
plt.plot(load_length * dt, p_total[-1], 'gx', label='')
plt.xlabel(time_label)
plt.ylabel(pressure_label)
plt.legend()
plt.savefig(f"{deformation_mode}_p_and_p_c_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Deviatoric Stress (q) vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, q, '-b', label=r"Quasi-static deviatoric stress ($q^{\mathrm{q}}$)")
plt.plot(np.arange(load_length) * dt, q_total, '-g', label=r"Total deviatoric stress ($p$)")
plt.plot(0 * dt, q[0], 'b.', label='')
plt.plot(0 * dt, q_total[0], 'g.', label='')
plt.plot(load_length * dt, q[-1], 'bx', label='')
plt.plot(load_length * dt, q_total[-1], 'gx', label='')
plt.xlabel(time_label)
plt.ylabel(dev_stress_label)
plt.legend()
plt.savefig(f"{deformation_mode}_q_and_q_c_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Volumetric and Deviatoric Strain vs. Time
plt.figure(figsize=figsize)
ax1 = plt.gca()
ax1.plot(np.arange(load_length) * dt, eq, '-b', label='')
ax1.plot(0 * dt, eq[0], 'b.', label='')
ax1.plot(load_length * dt, eq[-1], 'bx', label='')
ax1.set_xlabel(time_label)
ax1.set_ylabel(r'Deviatoric strain ($\gamma$)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(np.arange(load_length) * dt, ev, '-g')
ax2.plot(0 * dt, ev[0], 'g.', label='')
ax2.plot(load_length * dt, ev[-1], 'gx', label='')
ax2.set_ylabel(r"Volumetric strain ($\varepsilon_v$)", color='g')
ax2.tick_params(axis='y', labelcolor='g')
plt.savefig(f"{deformation_mode}_e_v_and_gamma_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Pre-consolidation Pressure vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, pc_history, '-b')
plt.plot(0 * dt, pc_history[0], 'b.', label='')
plt.plot(load_length * dt, pc_history[-1], 'bx', label='')
plt.xlabel(time_label)
plt.ylabel(preconsolidation_p_label)
plt.savefig(f"{deformation_mode}_preconsolidate_p_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Ratio between collisional and quasistatic stress vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length-1) * dt, p_c[1:] / p[1:], '-b', label=r'Pressures ($p^{\mathrm{c}}/p^{\mathrm{q}}$)')
plt.plot(0 * dt, p_c[1] / p[1], 'b.', label='')
plt.plot(load_length * dt, p_c[-1] / p[-1], 'bx', label='')
plt.plot(np.arange(load_length-1) * dt, q_c[1:] / q[1:], '-g', label=r'Deviatoric stresses ($q^{\mathrm{c}}/q^{\mathrm{q}}$)')
plt.plot(0 * dt, q_c[1] / q[1], 'g.', label='')
plt.plot(load_length * dt, q_c[-1] / q[-1], 'gx', label='')
plt.xlabel(time_label)
plt.ylabel(p_ratio_label)
plt.legend()
plt.savefig(f"{deformation_mode}_p_c_over_p_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Deviatoric Stress vs. Pressure
plt.figure(figsize=figsize)
plt.plot(p, q, '-+b', markevery=500, label=r"Quasi-static stress")
plt.plot(p_total, q_total, '-g', label=r"Total stress")
plt.plot([0, np.max(p)], [0, M * np.max(p)], '-+r', label='Critical state line')
plt.plot(p[0], q[0], 'b.', label='')
plt.plot(p[-1], q[-1], 'bx', label='')
plt.plot(p_total[0], q_total[0], 'g.', label='')
plt.plot(p_total[-1], q_total[-1], 'gx', label='')
plt.xlabel(pressure_label)
plt.ylabel(dev_stress_label)
plt.legend()
plt.savefig(f"{deformation_mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Void Ratios vs. Pressure
plt.figure(figsize=figsize)
plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
plt.plot(p[0], void_ratio_q[0], 'b.', label='')
plt.plot(p_total[0], void_ratio_total[0], 'g.', label='')
plt.plot(p[-1], void_ratio_q[-1], 'bx', label='')
plt.plot(p_total[-1], void_ratio_total[-1], 'gx', label='')
plt.xlabel(pressure_label)
plt.ylabel(e_label)
plt.legend()
plt.savefig(f"{deformation_mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Bulk Friction vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, q / p, '-b', label=r"Quasi-static friction ($\mu^{\mathrm{q}}$)")
plt.plot(np.arange(load_length) * dt, q_total / p_total, '-g', label=r"Total friction ($\mu$)")
plt.plot(0 * dt, q[0] / p[0], 'b.', label='')
plt.plot(0 * dt, q_total[0] / p_total[0], 'g.', label='')
plt.plot(load_length * dt, q[-1] / p[-1], 'bx', label='')
plt.plot(load_length * dt, q_total[-1] / p_total[-1], 'gx', label='')
plt.xlabel(time_label)
plt.ylabel(mu_label)
plt.legend()
plt.savefig(f"{deformation_mode}_mu_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

plt.show()
