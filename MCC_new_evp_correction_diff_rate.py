# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 
from matplotlib import rcParams
import matplotlib.colors as mcolors

rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'

def lighten(color, amount):
    """
    Lighten a color by mixing it with white.
    amount = 0 -> original
    amount = 1 -> white
    """
    try:
        c = np.array(mcolors.to_rgb(color))
    except ValueError:
        c = np.array(mcolors.to_rgb(mcolors.CSS4_COLORS[color]))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * c + amount * white)

def segmented_plot(ax, x, y, points, **kwargs):
    idx = [0] + points + [len(x)]
    for i in range(len(idx)-1):
        i0, i1 = idx[i], idx[i+1]
        if i1 > i0:
            ax.plot(x[i0:i1], y[i0:i1], **kwargs)

# Cam-clay critical state parameters
pc_0 = 3 * 150.0  # 'Initial consolidation pressure [kPa]'
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
Delta_Phi = 25.0
M_c = 1.5

# Time needed to accelerate or decelerate
accel_times = [0.1, 0.5, 1.0]
total_time = 2.5
n_runs = len(accel_times)

# collect output per acceleration rate
p_total_list = []
q_total_list = []
void_ratio_total_list = []
p_list = []
q_list = []
void_ratio_q_list = []
p_c_list = []
q_c_list = []

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)

for k, accel_time in enumerate(accel_times):

    alpha = (k / max(n_runs-1, 1)) * 0.7  
    # 0.7 so they never become too washed-out (tune if needed)

    color_qs    = lighten('b', alpha)   # quasi-static curves
    color_total = lighten('g', alpha)   # total curves
    color_csl   = 'r'                   # critical state line
    color_mark  = 'k'                   # point markers (keep black)
    color_load  = lighten('k', alpha)   # load history curve (light dark-grey)
    
    # Define a loading history
    eqp_inc = eqp_tot / (1e2 * load_length)  # [-] 'incrementâlly applied plastic shear strain'
    eqp_inc_history = np.concatenate([
        np.ones(round(1.0 / eqp_inc)) * eqp_inc,
        np.linspace(eqp_inc, 100 * eqp_inc, round(accel_time / eqp_inc)),
        np.ones(round((total_time -2*accel_time)/ eqp_inc)) * 100 * eqp_inc,
        np.linspace(100 * eqp_inc, eqp_inc, round(accel_time / eqp_inc)),
        np.ones(round(0.5 / eqp_inc)) * eqp_inc,
    ])
    point1 = round(1.0 / eqp_inc)
    point2 = point1 + round(accel_time / eqp_inc)
    point3 = point2 + round((total_time -2*accel_time) / eqp_inc)
    point4 = point3 + round(accel_time / eqp_inc)
    points = [point1, point2, point3, point4]
    markers = ['^', '>', 'v', '<']
    total_load_length = eqp_inc_history.shape[0]
    
    fig_load_history = plt.figure('Load history')
    plt.plot([-10, 360], [0, 0], '--k')
    plt.plot(np.arange(total_load_length ) * dt, eqp_inc_history / dt, color=color_load)
    x_offset = total_load_length  * dt * 0.03
    y_offset = max(eqp_inc_history) / dt * 0.03
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        plt.plot(p_i * dt, eqp_inc_history[p_i] / dt, f'{m_i}', color=color_load, ms=10, mfc='none');
    plt.plot(0 * dt, eqp_inc_history[0] / dt, '.', color=color_load, ms=10, label='')
    plt.plot(total_load_length  * dt, eqp_inc_history[-1] / dt, 'x', color=color_load, ms=10, label='')
    plt.xlabel(r'Time $t$ [s]')
    plt.ylabel(r'Axial strain rate $\dot{\varepsilon_{zz}}$ [1/s]')
    ymin, ymax = plt.ylim()
    plt.xlim(-10, 410)
    
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
        De_c = np.zeros([6, 6])
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
                        De[m, n] = K + 4 / 3 * G
                        De_c[m, n] = K_c
                        D_c[m, n] = 4 / 3 * G_c
                    elif n <= 2:
                        De[m, n] = K + 2 / 3 * G
                        De_c[m, n] = K_c
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
    
                # Get the deviatoric part of the acceleration rate tensor ?
                Phi = 1.0 / (1.0 + void_ratio_q[i])
                de_v_c = - Phi / Phi ** 2 * Delta_Phi * d_eqp_inc
                #d_p_c[i + 1] = p[i] / ((lambda_val - kappa) * Phi ** 2) * Delta_Phi * d_eqp_inc
                d_p_c[i + 1] = p[i] / ((lambda_val - 0) * Phi ** 2) * Delta_Phi * d_eqp_inc
    
                K_c = d_p_c[i + 1] / de_v_c
                G_c = d_p_c[i + 1] * M_c / d_eqp_inc / 3
            else:
                p_c[i + 1] = p_c[i]
                q_c[i + 1] = q_c[i]
                de_v_c = 0
                K_c = 0
                G_c = 0
    
            # Get collision-induced plastic strain increment
            d_epsilon_v_c = 1./3. * de_v_c * np.array([1., 1., 1., 0, 0, 0])
            
            d_sigma_q = D_q.dot(d_epsilon - d_epsilon_v_c)
            d_sigma_c = De_c.dot(d_epsilon_v_c) + D_c.dot(d_d_epsilon)
    
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

    # Display results
    plt.figure('Pressure controlled simple shear')
    
    # Full text long labels
    time_label = r'Time ($t$) [s]'
    mu_label = r'Ratio of deviatoric stress to pressure $q/p$ [-]'
    pressure_label = r'Pressure $p$ [kPa]'
    dev_stress_label = r'Deviatoric stress $q$ [kPa]'
    ev_label = r'Volumetric strain $\varepsilon_v$ [-]'
    preconsolidation_p_label = 'Pre-consolidation pressure [kPa]'
    p_ratio_label = r'Ratio of collisional and quasi-static stresses [-]'
    e_label = r'Void ratio $e$ [-]'
    ev_label = r'Volumetric strain $\varepsilon_v$ [-]'
    
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
    plt.plot(0 * dt, p[0], 'bx', ms=10, label='')
    plt.plot(0 * dt, p_total[0], 'g+', ms=10, label='')
    plt.plot(load_length * dt, p[-1], 'bo', ms=10, label='')
    plt.plot(load_length * dt, p_total[-1], 'g.', ms=10, label='')
    # labels
    plt.xlabel(ctime)
    plt.ylabel(cp)
    plt.legend()
    
    # deviatoric stress (q) vs. time
    plt.subplot(2, 4, 2)
    plt.plot(np.arange(load_length) * dt, q, '-b', label='qstat')
    plt.plot(np.arange(load_length) * dt, q_total, '-g', label='tot')
    # begin and endpoints
    plt.plot(0 * dt, q[0], 'bx', ms=10, label='')
    plt.plot(0 * dt, q_total[0], 'g+', ms=10, label='')
    plt.plot(load_length * dt, q[-1], 'bo', ms=10, label='')
    plt.plot(load_length * dt, q_total[-1], 'g.', ms=10, label='')
    # labels
    plt.xlabel(ctime)
    plt.ylabel(cq)
    plt.legend()
    
    # volumetric and deviatoric strain vs. time
    ax1 = plt.subplot(2, 4, 3)
    ax1.plot(np.arange(load_length) * dt, eq, '-b', label='dev')
    ax1.plot(0 * dt, eq[0], 'bx', ms=10, label='')
    ax1.plot(load_length * dt,eq[-1], 'bo', ms=10, label='')
    ax1.set_xlabel(ctime)
    ax1.set_ylabel('Deviatoric Strain', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create the secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot the second curve on the right y-axis
    ax2.plot(np.arange(load_length) * dt, ev, '--g', label='v')
    ax2.plot(0 * dt, ev[0], 'gx', ms=10, label='')
    ax2.plot(load_length * dt, ev[-1], 'g.', ms=10, label='')
    ax2.set_ylabel('Volumetric Strain', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add legends for clarity
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # pre-consolidation pressure vs. time
    plt.subplot(2, 4, 4)
    plt.plot(np.arange(load_length) * dt, pc_history, '-b')
    # begin and endpoints
    plt.plot(0 * dt, pc_history[0], 'bx', ms=10, label='')
    plt.plot(load_length * dt, pc_history[-1], 'b.', ms=10, label='')
    # labels
    plt.xlabel(ctime)
    plt.ylabel(cppc)
    
    # collisional stress (iso) vs. time
    plt.subplot(2, 4, 5)
    plt.plot(np.arange(load_length) * dt, p_c / p, '-b')
    # begin and endpoints
    plt.plot(0 * dt, p_c[0] / p[0], 'bx', ms=10, label='')
    plt.plot(load_length * dt, p_c[-1] / p[-1], 'b.', ms=10, label='')
    # labels
    plt.xlabel(ctime)
    plt.ylabel(cpcoll)
    
    # deviatoric stress vs. p
    plt.subplot(2, 4, 6)
    imark = 500
    plt.plot(p, q, '-b', markevery=imark, label='qstat')
    plt.plot(p_total, q_total, '-g', label='tot')
    plt.plot([0, np.max(p)], [0, M * np.max(p)], '-r', label='CSL')
    # begin and endpoints
    plt.plot(p[0], q[0], 'bx', ms=10, label='')
    plt.plot(p[-1], q[-1], 'b.', ms=10, label='')
    plt.plot(p_total[0], q_total[0], 'g+', ms=10, label='')
    plt.plot(p_total[-1], q_total[-1], 'g.', ms=10, label='')
    # plt.plot(p_total[-1], M * np.max(p), 'rs', ms=10, label='')
    # labels
    plt.xlabel(cp)
    plt.ylabel(cq)
    plt.legend()
    
    # void ratios vs. p
    plt.subplot(2, 4, 7)
    plt.plot(p, void_ratio_q, '-b', label='qstat')
    plt.plot(p_total, void_ratio_total, '-g', label='tot')
    # begin and endpoints
    plt.plot(p[0], void_ratio_q[0], 'bx', ms=10, label='')
    plt.plot(p_total[0], void_ratio_total[0], 'g+', ms=10, label='')
    plt.plot(p[-1], void_ratio_q[-1], 'b.', ms=10, label='')
    plt.plot(p_total[-1], void_ratio_total[-1], 'g.', ms=10, label='')
    # labels
    plt.xlabel(cp)
    plt.ylabel(cvr)
    # plt.legend()
    
    # bulk friction vs. time
    plt.subplot(2, 4, 8)
    plt.plot(np.arange(load_length) * dt, q_total / p_total, 'b')
    # begin and endpoints
    plt.plot(0 * dt, q_total[0] / p_total[0], 'b+', ms=10, label='')
    plt.plot(load_length * dt, q_total[-1] / p_total[-1], 'b.', ms=10, label='')
    # labels
    plt.xlabel(ctime)
    plt.ylabel(cmu)
    
    plt.show()   

    #%% Individual production plots
    
    # Deviatoric Stress vs. Pressure
    fig_p_q = plt.figure(figsize=figsize)
    plt.plot([0, np.max(p) * 1.5], [0, M * np.max(p) * 1.5], '-r', label='Critical state line')
    plt.plot(p, q, '-b', markevery=500, label=r"Quasi-static stress")
    plt.plot(p_total, q_total, '-g', label=r"Total stress")
    plt.plot(p[0], q[0], 'b.', ms=10, label='')
    plt.plot(p[-1], q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[0], q_total[0], 'g.', ms=10, label='')
    plt.plot(p_total[-1], q_total[-1], 'gx', ms=10, label='')
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        plt.plot(p_total[p_i], q_total[p_i], f'g{m_i}',ms=10, mfc='none')
        plt.plot(p[p_i], q[p_i], f'b{m_i}',ms=10, mfc='none')
    plt.xlabel(pressure_label)
    plt.ylabel(dev_stress_label)
    #plt.xlim(0,350)
    #plt.ylim(0,393.75)
    plt.legend(loc='upper left')
    
    # Void Ratios vs. Pressure
    fig_e_p = plt.figure(figsize=figsize)
    plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
    plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
    plt.plot(p[0], void_ratio_q[0], 'b.', ms=10, label='')
    plt.plot(p_total[0], void_ratio_total[0], 'g.', ms=10, label='')
    plt.plot(p[-1], void_ratio_q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[-1], void_ratio_total[-1], 'gx', ms=10, label='')
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        plt.plot(p[p_i], void_ratio_q[p_i], f'b{m_i}',ms=10, mfc='none')
        plt.plot(p_total[p_i], void_ratio_total[p_i], f'g{m_i}',ms=10, mfc='none')
    xlim = plt.xlim()
    ylim = plt.ylim()
    pCSL = np.linspace(10,400,1000)
    Gamma = void_ratio_q[-1] + lambda_val*np.log(p[-1])
    eCSL = Gamma - lambda_val*np.log(pCSL);
    plt.clf() # clear everything in this figure
    plt.plot(pCSL, eCSL, '-r', label='Critical state line')
    plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
    plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
    plt.plot(p[0], void_ratio_q[0], 'b.', ms=10, label='')
    plt.plot(p_total[0], void_ratio_total[0], 'g.', ms=10, label='')
    plt.plot(p[-1], void_ratio_q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[-1], void_ratio_total[-1], 'gx', ms=10, label='')
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        plt.plot(p[p_i], void_ratio_q[p_i], f'b{m_i}',ms=10, mfc='none')
        plt.plot(p_total[p_i], void_ratio_total[p_i], f'g{m_i}',ms=10, mfc='none')
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.xlim(145, 330)
    #plt.ylim(0.26, 0.33)
    plt.xlabel(pressure_label)
    plt.ylabel(e_label)
    
     # Gamma dot vs. mu
    fig_gamma_dot_mu = plt.figure(figsize=figsize)
    plt.plot(eqp_inc_history / dt, (q_total/p_total))
    plt.xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    plt.ylabel(mu_label)
    plt.ylim(1)
    plt.savefig(f"{deformation_mode}_mu_gamma_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=figsize)
    plt.plot(eqp_inc_history[point1:point2] / dt, 1/(1+void_ratio_total[point1:point2]))
    plt.plot(eqp_inc_history[point2:point3] / dt, 1/(1+void_ratio_total[point2:point3]))
    plt.plot(eqp_inc_history[point3:point4] / dt, 1/(1+void_ratio_total[point3:point4]))
    plt.plot(eqp_inc_history[point4:] / dt, 1/(1+void_ratio_total[point4:]))
    x_offset = 0.03
    y_offset = 0.00
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        plt.plot(eqp_inc_history[p_i] / dt, 1/(1+void_ratio_total[p_i]), f'k{m_i}',ms=10, mfc='none');
        plt.text(eqp_inc_history[p_i] / dt + x_offset, 1/(1+void_ratio_total[p_i])  + y_offset, f"{i+1}", fontsize=12)
    plt.xlabel('Shear rate')
    plt.ylabel('Solid volume fraction (total)')
    

# Save load history figure    
fig_load_history.savefig(f"load_history.png", dpi=300, bbox_inches="tight")
fig_p_q.savefig(f"{deformation_mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")
fig_e_p.savefig(f"{deformation_mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")
fig_gamma_dot_mu.savefig("gamma_phi_transient.png", dpi=300, bbox_inches="tight")