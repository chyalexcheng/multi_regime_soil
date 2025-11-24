# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 
from matplotlib import rcParams
from matplotlib import cm

rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'

# Cam-clay critical state parameters
pc_0_list = np.array([150.0, 300, 450])  # 'Initial consolidation pressure [kPa]'
M = 1.0  # 'Critical friction angle'
lambda_val = 0.2  # 'Lambda'
kappa = 0.04  # 'kappa'
N = 2.5  # 'Intercept of the normal consolidation line'
nu = 0.15  # 'Poisson ratio'

# Initial conditions
p0_list = np.array([150.0, 300, 450])  # 'Initial confining pressure [kPa]'
V_list = N - (lambda_val * np.log(p0_list)) + (kappa * np.log(pc_0_list / p0_list))  # Specific Volume
void_ratio_0_list = V_list - 1  # Initial void ratio
Phi_0_list = 1.0 / (1.0 + void_ratio_0_list)  # Initial solid volume fraction
deformation_mode = input('Enter the deformation mode: [1] drained and [2] undrained\n')

# Maximum shear strain and number of load steps for the quasi-static stage
eqp_tot = 100.0  # [%] 'total plastic shear strain'
time = 100.0  # [s] 'second'
load_length = int(1e5)  # [-] 'loadsteps'
dt = time / load_length

# Collisional contribution parameters
Delta_Phi = 25.0
M_c = 1.5

# Define a loading history
eqp_inc = eqp_tot / (1e2 * load_length)  # [-] 'incrementâlly applied plastic shear strain'
load_segment_0 = np.ones(round(1.0 / eqp_inc)) * eqp_inc
num_of_loading_steps = 99
load_interval = np.log10(999 * eqp_inc/num_of_loading_steps )
load_segment_s = [load_segment_0]
for i in range(num_of_loading_steps ):
    load_segment_s.append(np.linspace(eqp_inc + i * 10**load_interval, eqp_inc + (i+1) * 10**load_interval, round(0.5 / eqp_inc)))
    load_segment_s.append(np.ones(round(0.5 / eqp_inc)) * eqp_inc + (i+1) * 10**load_interval)
eqp_inc_history = np.concatenate(load_segment_s)
point1 = round(1.0 / eqp_inc)
points = [point1 + i * round(1.0 / eqp_inc) - 1 for i in range(100)]
load_length = eqp_inc_history.shape[0]

plt.figure('Load history')
plt.plot(np.arange(load_length) * dt, eqp_inc_history / dt, '-b')
plt.plot(0 * dt, eqp_inc_history[0] / dt, 'b.', ms=10, label='')
plt.plot(load_length * dt, eqp_inc_history[-1] / dt, 'bx', ms=10, label='')
plt.xlabel(r'Time $t$ [s]')
plt.ylabel(r'Shear rate $\dot{\gamma}$ [1/s]')
plt.savefig(f"load_history_mu_I_compare.png", dpi=300, bbox_inches="tight")

# Full text long labels
time_label = r'Time ($t$) [s]'
mu_label = r'Ratio of deviatoric stress to pressure $q/p$ [-]'
pressure_label = r'Pressure $p$ [kPa]'
dev_stress_label = r'Deviatoric stress $q$ [kPa]'
ev_label = r'Volumetric strain $\varepsilon_v$ [-]'
preconsolidation_p_label = 'Pre-consolidation stress [kPa]'
p_ratio_label = r'Ratio of collisional and quasi-static stresses [-]'
e_label = r'Void ratio $e$ [-]'

# collect output per initial condition
p_total_list = []
q_total_list = []
void_ratio_total_list = []
p_list = []
q_list = []
void_ratio_q_list = []
p_c_list = []
q_c_list = []

for pc_0, p0, V, void_ratio_0, Phi_0 in zip(pc_0_list, p0_list, V_list, void_ratio_0_list, Phi_0_list):
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
    
                Phi = 1.0 / (1.0 + void_ratio_q[i])
                de_v_c = - Phi / Phi ** 2 * Delta_Phi * d_eqp_inc
                d_p_c[i + 1] = p[i] / ((lambda_val - kappa) * Phi ** 2) * Delta_Phi * d_eqp_inc
    
                K_c = d_p_c[i + 1] / de_v_c
                G_c = d_p_c[i + 1] * M_c / d_eqp_inc / 3
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
    
    #%% collect results
    p_total_list.append(p_total)
    q_total_list.append(q_total)
    void_ratio_total_list.append(void_ratio_total)
    p_list.append(p)
    q_list.append(q)
    void_ratio_q_list.append(void_ratio_q)
    p_c_list.append(p_c)
    q_c_list.append(q_c)

    #%% Individual production plots
    
    # Set figure size for half-width of A4
    a4_half_width = 6  # in inches
    fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
    fig_height = a4_half_width / fig_aspect_ratio
    figsize = (a4_half_width, fig_height)
    
    # Pressure vs. Time
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length) * dt, p, '-b', label=r"Quasi-static pressure ($p^{\mathrm{q}}$)")
    plt.plot(np.arange(load_length) * dt, p_total, '-g', label=r"Total pressure ($p$)")
    plt.plot(0 * dt, p[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, p_total[0], 'g.', ms=10, label='')
    plt.plot(load_length * dt, p[-1], 'bx', ms=10, label='')
    plt.plot(load_length * dt, p_total[-1], 'gx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(pressure_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_p_and_p_c_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Deviatoric Stress (q) vs. Time
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length) * dt, q, '-b', label=r"Quasi-static deviatoric stress ($q^{\mathrm{q}}$)")
    plt.plot(np.arange(load_length) * dt, q_total, '-g', label=r"Total deviatoric stress ($p$)")
    plt.plot(0 * dt, q[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, q_total[0], 'g.', ms=10, label='')
    plt.plot(load_length * dt, q[-1], 'bx', ms=10, label='')
    plt.plot(load_length * dt, q_total[-1], 'gx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(dev_stress_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_q_and_q_c_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Volumetric and Deviatoric Strain vs. Time
    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    ax1.plot(np.arange(load_length) * dt, eq, '-b', ms=10, label='')
    ax1.plot(0 * dt, eq[0], 'b.', ms=10, label='')
    ax1.plot(load_length * dt, eq[-1], 'bx', ms=10, label='')
    ax1.set_xlabel(time_label)
    ax1.set_ylabel(r'Deviatoric strain ($\gamma$)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(np.arange(load_length) * dt, ev, '-g')
    ax2.plot(0 * dt, ev[0], 'g.', ms=10, label='')
    ax2.plot(load_length * dt, ev[-1], 'gx', ms=10, label='')
    ax2.set_ylabel(r"Volumetric strain ($\varepsilon_v$)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.savefig(f"{deformation_mode}_e_v_and_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Pre-consolidation Pressure vs. Time
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length) * dt, pc_history, '-b')
    plt.plot(0 * dt, pc_history[0], 'b.', ms=10, label='')
    plt.plot(load_length * dt, pc_history[-1], 'bx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(preconsolidation_p_label)
    plt.savefig(f"{deformation_mode}_preconsolidate_p_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Ratio between collisional and quasistatic stress vs. Time
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length-1) * dt, p_c[1:] / p[1:], '-b', label=r'Pressures ($p^{\mathrm{c}}/p^{\mathrm{q}}$)')
    plt.plot(0 * dt, p_c[1] / p[1], 'b.', ms=10, label='')
    plt.plot(load_length * dt, p_c[-1] / p[-1], 'bx', ms=10, label='')
    plt.plot(np.arange(load_length-1) * dt, q_c[1:] / q[1:], '-g', label=r'Deviatoric stresses ($q^{\mathrm{c}}/q^{\mathrm{q}}$)')
    plt.plot(0 * dt, q_c[1] / q[1], 'g.', ms=10, label='')
    plt.plot(load_length * dt, q_c[-1] / q[-1], 'gx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(p_ratio_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_p_c_over_p_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Deviatoric Stress vs. Pressure
    plt.figure(figsize=figsize)
    plt.plot([0, np.max(p) * 1.5], [0, M * np.max(p) * 1.5], '-r', label='Critical state line')
    plt.plot(p, q, '-b', markevery=500, label=r"Quasi-static stress")
    plt.plot(p_total, q_total, '-g', label=r"Total stress")
    plt.plot(p[0], q[0], 'b.', ms=10, label='')
    plt.plot(p[-1], q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[0], q_total[0], 'g.', ms=10, label='')
    plt.plot(p_total[-1], q_total[-1], 'gx', ms=10, label='')
    plt.xlabel(pressure_label)
    plt.ylabel(dev_stress_label)
    plt.xlim(0,260)
    plt.ylim(0,300)
    plt.legend()
    plt.savefig(f"{deformation_mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Void Ratios vs. Pressure
    plt.figure(figsize=figsize)
    plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
    plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
    plt.plot(p[0], void_ratio_q[0], 'b.', ms=10, label='')
    plt.plot(p_total[0], void_ratio_total[0], 'g.', ms=10, label='')
    plt.plot(p[-1], void_ratio_q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[-1], void_ratio_total[-1], 'gx', ms=10, label='')
    plt.xlabel(pressure_label)
    plt.ylabel(e_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Bulk Friction vs. Time
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length) * dt, q / p, '-b', label=r"Quasi-static friction ($\mu^{\mathrm{q}}$)")
    plt.plot(np.arange(load_length) * dt, q_total / p_total, '-g', label=r"Total friction ($\mu$)")
    plt.plot(0 * dt, q[0] / p[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, q_total[0] / p_total[0], 'g.', ms=10, label='')
    plt.plot(load_length * dt, q[-1] / p[-1], 'bx', ms=10, label='')
    plt.plot(load_length * dt, q_total[-1] / p_total[-1], 'gx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(mu_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_mu_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")
    
    # Void ratios vs. Time
    void_ratio_e = void_ratio_total - void_ratio_q
    plt.figure(figsize=figsize)
    plt.plot(np.arange(load_length) * dt, void_ratio_q-void_ratio_q[0], '-b', label=r"$e^{\mathrm{p}}-e^{\mathrm{p}}_0$")
    plt.plot(np.arange(load_length) * dt, void_ratio_e-void_ratio_e[0], '-g', label=r"$e^{\mathrm{e}}$")
    plt.plot(0 * dt, 0, 'b.', ms=10, label='')
    plt.plot(0 * dt, 0, 'g.', ms=10, label='')
    plt.plot(load_length * dt, void_ratio_q[-1]-void_ratio_q[0], 'bx', ms=10, label='')
    plt.plot(load_length * dt, void_ratio_e[-1]-void_ratio_e[0], 'gx', ms=10, label='')
    plt.xlabel(time_label)
    plt.ylabel(e_label)
    plt.legend()
    plt.savefig(f"{deformation_mode}_e_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")

# save the stead-state data into npy file.
np.save(f'{deformation_mode}_steady_state.npy',
        (np.array(p_total_list)[:, points],
         np.array(q_total_list)[:, points],
         np.array(void_ratio_total_list)[:, points],
         np.array(p_list)[:, points],
         np.array(q_list)[:, points],
         np.array(void_ratio_q_list)[:, points],
         np.array(p_c_list)[:, points],
         np.array(q_c_list)[:, points]))
np.save(f'{deformation_mode}_d_gamma_history.npy', eqp_inc_history[points] / dt)

# mu(I) and phi(I) fit to the data

from scipy.optimize import least_squares, curve_fit

# Define mu(I, p) and phi(I, p) for Bo = 0
def mu_fit_func(I, muinf_fit, Imu_fit, pmu_fit, mu0_const, p_const):
    p_const /= 1e3
    fI = 1 + (muinf_fit / mu0_const - 1) / (1 + Imu_fit / I)
    fp = 1 - (p_const / pmu_fit)**0.5
    return mu0_const * fI * fp

# phi(I) with Iphi, pphi to fit (Bo=0, p=constant like 0)
def phi_fit_func(I, Iphi_fit, pphi_fit, phi0_const, p_const):
    p_const /= 1e3
    gI = 1 - I / Iphi_fit
    gp = 1 + p_const / pphi_fit
    return phi0_const * gI * gp

def mu_residuals(params, I, mu_obs, p_const):
    muinf_fit, Imu_fit, pmu_fit = params
    return mu_fit_func(I, muinf_fit, Imu_fit, pmu_fit, mu_obs[0], p_const) - mu_obs

def phi_residuals(params, I, phi_obs, p_const):
    Iphi_fit, pphi_fit = params
    return phi_fit_func(I, Iphi_fit, pphi_fit, phi_obs[0], p_const) - phi_obs

def mu_I_classic(I, M_c_fit, I0):
    global M
    return M + (M_c_fit - M) / (I0 / I + 1)

# Prepare Inferno colormap
n = len(p0_list)
colors = [cm.inferno(0.2 + 0.6 * i / (n - 1)) for i in range(n)]

# Gamma dot vs. mu
particle_diameter = 1e-3
particle_density = 2500
mu_phi_I_fit = True

plt.figure(figsize=figsize)
for idx, (p0, q_total, p_total) in enumerate(zip(p0_list, q_total_list, p_total_list)):
    color = colors[idx]
    inertia_number = (
        eqp_inc_history[points] / dt * particle_diameter /
        np.sqrt(p0 * 1e3 / particle_density)
    )

    plt.semilogx(inertia_number, (q_total / p_total)[points],
                 label=f'$p_0 = {p0:.0f}$ kPa', color=color, zorder=3, linewidth=1.5)

    if mu_phi_I_fit:
        mu_val = (q_total / p_total)[points]
        popt, pcov = curve_fit(
            mu_I_classic,
            inertia_number, mu_val,
            p0=[1.5, 1e-3],
            bounds=([1.5, 1e-6], [2.0, 1e-3])
        )
        I_fit = np.logspace(np.log10(min(inertia_number)), np.log10(2e-3), 200)
        plt.semilogx(I_fit, mu_I_classic(I_fit, *popt),
                     '--', color=color, label='', zorder=1, linewidth=1.0)
        print(f'I_0 = {popt[-1]}, mu_c = {popt[0]}')
    if int(p0) == 150:
        shear_rate = eqp_inc_history[points] / dt
        idx = (np.abs(shear_rate - 1.0)).argmin()  # closest to shear rate = 1.0
        plt.plot(inertia_number[idx], (q_total / p_total)[points][idx],
                 marker='*', color=color, markersize=12, zorder=4, label='', mfc='none')

plt.xlabel(r'Inertial number $I$ [-]')
plt.ylabel(mu_label)
plt.xlim(1e-6)
plt.ylim(min((q_total_list/p_total_list).flatten()) * 0.98, max((q_total_list/p_total_list).flatten()) * 1.02)
plt.axhline(1.0, color='red', linestyle='--', label=r'$\mu^{\mathrm{cs}}$')
plt.axhline(1.5, color='red', linestyle=':', label=r'$\mu^{\mathrm{c}}$')
plt.legend()
plt.savefig(f"{deformation_mode}_mu_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png",
            dpi=300, bbox_inches="tight")

# Gamma dot vs. phi
plt.figure(figsize=figsize)
for idx, (p0, void_ratio_total) in enumerate(zip(p0_list, void_ratio_total_list)):
    color = colors[idx]
    inertia_number = (
        eqp_inc_history[points] / dt * particle_diameter /
        np.sqrt(p0 * 1e3 / particle_density)
    )
    phi_val = 1.0 / (1.0 + void_ratio_total[points])
    plt.semilogx(inertia_number, phi_val,
                 label=f'$p_0 = {p0:.0f}$ kPa', color=color, zorder=3, linewidth=1.5)

    if mu_phi_I_fit:
        result_phi = least_squares(phi_residuals, x0=[5.4, 0.3],
                                   args=(inertia_number, phi_val, p0))
        fitted_phi = phi_fit_func(I_fit, *np.append(result_phi.x, phi_val[0]), p0)
        plt.semilogx(I_fit, fitted_phi,
                     '--', color=color, label='', zorder=1, linewidth=1.0)
    if int(p0) == 150:
        shear_rate = eqp_inc_history[points] / dt
        idx = (np.abs(shear_rate - 1.0)).argmin()  # closest to shear rate = 1.0
        plt.plot(inertia_number[idx], (phi_val)[points][idx],
                 marker='*', color=color, markersize=12, zorder=4, label='', mfc='none')

plt.xlabel(r'Inertial number $I$ [-]')
plt.ylabel(r'Solid volume fraction $\phi$ [-]')
plt.xlim(1e-6)
phi_q = 1.0 / (1.0 + void_ratio_total_list)
plt.ylim(min(phi_q.flatten()) * 0.98, max(phi_q.flatten()) * 1.02)
plt.legend()
plt.savefig(f"{deformation_mode}_phi_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png",
            dpi=300, bbox_inches="tight")

plt.show()

# Gamma dot vs. phi_q
plt.figure(figsize=figsize)
for idx, (p0, void_ratio_q) in enumerate(zip(p0_list, void_ratio_q_list)):
    color = colors[idx]
    plt.plot(eqp_inc_history[points] / dt, 1.0 / (1.0 + void_ratio_q[points]), label=f'$p_0 = {p0:.0f}$ kPa', color=color)
#    ax = plt.gca()
#    color = ax.get_lines()[-1].get_color()
#    plt.plot(eqp_inc_history[points[0]] / dt, 1.0 / (1.0 + void_ratio_q[points[0]]), '.', color=color, ms=10, label='')
#    plt.plot(eqp_inc_history[points[-1]] / dt, 1.0 / (1.0 + void_ratio_q[points[-1]]), 'x', color=color, ms=10, label='')
    if int(p0) == 150:
        shear_rate = eqp_inc_history[points] / dt
        idx = (np.abs(shear_rate - 1.0)).argmin()  # closest to shear rate = 1.0
        plt.plot(eqp_inc_history[points][idx] /dt, (1.0 / (1.0 + void_ratio_q[points]))[idx],
                 marker='*', color=color, markersize=12, zorder=4, label='', mfc='none')
plt.xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
plt.ylabel(r'Critical-state solid volume fraction $\phi^{\mathrm{cs}}$ [-]')
plt.xlim(0,10)
plt.legend()
plt.savefig(f"{deformation_mode}_phi_q_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")

# Gamma dot vs. (\lambda-\kappa)\phi_q**2/Delta_Phi for unified and I_0/d*\sqrt(p_0/rho) + \dot{\gamma} for mu(I)
plt.figure(figsize=figsize)
for idx, (p0, p, void_ratio_q) in enumerate(zip(p0_list, p_list, void_ratio_q_list)):
    color = colors[idx]
    phi_q = 1.0 / (1.0 + void_ratio_q[points])
    p_c_rate = (lambda_val-kappa) * phi_q**2 / Delta_Phi
    plt.semilogy(eqp_inc_history[points] / dt, p_c_rate, label=f'$p_0 = {p0:.0f}$ kPa (CMCC)', color=color)
    ax = plt.gca()
    color = ax.get_lines()[-1].get_color()
    p_c_rate_mu_I = p_c_rate[0] + eqp_inc_history[points] / dt
    plt.semilogy(eqp_inc_history[points] / dt, p_c_rate_mu_I, '--', color=color, label=f'$p_0 = {p0:.0f}$ kPa ($\mu(I)$)')
    #plt.semilogy(eqp_inc_history[points[-1]] / dt, p_c_rate_mu_I[-1], 'x', color=color, ms=10, label='')
    if int(p0) == 150:
        shear_rate = eqp_inc_history[points] / dt
        idx = (np.abs(shear_rate - 1.0)).argmin()  # closest to shear rate = 1.0
        print(idx)
        plt.plot(eqp_inc_history[points][idx] /dt, p_c_rate[idx],
                 marker='*', color=color, markersize=12, zorder=4, label='', mfc='none')

plt.xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
plt.ylabel(r'Collisional stress decay rate $f(\dot\gamma})$ [1/s]')
plt.xlim(0,10)
plt.legend()
plt.savefig(f"{deformation_mode}_f_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")

# Gamma dot vs. p_q / ((\lambda-\kappa)\phi_q**2/Delta_Phi) for unified and p_q / (I_0/d*\sqrt(p_0/rho) + \dot{\gamma}) for mu(I)
plt.figure(figsize=figsize)
for idx, (p0, p, void_ratio_q) in enumerate(zip(p0_list, p_list, void_ratio_q_list)):
    color = colors[idx]
    phi_q = 1.0 / (1.0 + void_ratio_q[points])
    p_c_rate = p[points] / ((lambda_val-kappa) * phi_q**2 / Delta_Phi)
    plt.semilogy(eqp_inc_history[points] / dt, M_c*p_c_rate, label=f'$p_0 = {p0:.0f}$ kPa (CMCC)', color=color)
    ax = plt.gca()
    color = ax.get_lines()[-1].get_color()
    #plt.semilogy(eqp_inc_history[points[0]] / dt, p_c_rate[0], '.', color=color, ms=10, label='')
    #plt.semilogy(eqp_inc_history[points[-1]] / dt, p_c_rate[-1], 'x', color=color, ms=10, label='')
    p_c_rate_mu_I = p[points] / (p[0]/p_c_rate[0] + eqp_inc_history[points] / dt)
    print(f'scaled I_0 = {p[0]*1e3/p_c_rate[0] * particle_diameter * np.sqrt(p0 *1e3/particle_density)}')
    plt.semilogy(eqp_inc_history[points] / dt, M_c*p_c_rate_mu_I, '--', color=color, label=f'$p_0 = {p0:.0f}$ kPa ($\mu(I)$)')
    #plt.semilogy(eqp_inc_history[points[-1]] / dt, p_c_rate_mu_I[-1], 'x', color=color, ms=10, label='')
    if int(p0) == 150:
        shear_rate = eqp_inc_history[points] / dt
        idx = (np.abs(shear_rate - 1.0)).argmin()  # closest to shear rate = 1.0
        print(idx)
        plt.plot(eqp_inc_history[points][idx] /dt, M_c*p_c_rate[idx],
                 marker='*', color=color, markersize=12, zorder=4, label='', mfc='none')

plt.xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
plt.ylabel(r'Collisional dynamic viscosity ' + 
           r'$\frac{\dot{q}^\mathrm{c}}{\ddot{\gamma}}$ [kPa$\cdot$s]')
plt.xlim(0,10)
plt.legend()
plt.savefig(f"{deformation_mode}_d_p_c_gamma_{void_ratio_0:.3f}_{OCR:.3f}_{p0}.png", dpi=300, bbox_inches="tight")

plt.show()