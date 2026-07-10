# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

def get_deviatoric_strain(tensor):
    eps_v = np.sum(tensor[:3])
    eps_dev = tensor - np.array([eps_v/3., eps_v/3., eps_v/3., 0., 0., 0.])
    dev_contract = (
        eps_dev[0]**2 + eps_dev[1]**2 + eps_dev[2]**2
        + 0.5 * (eps_dev[3]**2 + eps_dev[4]**2 + eps_dev[5]**2)
    )
    eps_q = np.sqrt(2.0/3.0 * dev_contract)
    direction = eps_dev / eps_q if eps_q > 0.0 else np.zeros_like(tensor)
    return eps_q, direction

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
    idx = points
    for i in range(len(idx)-1):
        i0, i1 = idx[i], idx[i+1]
        if i1 > i0:
            ax.plot(x[i0:i1], y[i0:i1], **kwargs)

def expand_ylim_for_annotations(ax, top_frac=0.10, bottom_frac=0.03):
    y_min_cur, y_max_cur = ax.get_ylim()
    y_diff = np.abs(y_max_cur - y_min_cur)
    if y_diff == 0:
        y_diff = 1.0
    ax.set_ylim(y_min_cur - bottom_frac * y_diff, y_max_cur + top_frac * y_diff)

# Load history
def plot_load(load_length, dt, eqp_inc_history, label, color_load, points, markers):
    ax_load.plot(np.arange(load_length ) * dt, eqp_inc_history / dt, label=label, color=color_load)
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if m_i == '^':
            mfc = 'red'
        else:
            mfc = 'none'
        ax_load.plot(p_i * dt, eqp_inc_history[p_i] / dt, f'{m_i}', color=color_load, ms=10, mfc=mfc)
    ax_load.plot(0 * dt, eqp_inc_history[0] / dt, '.', color=color_load, ms=10, label='')
    ax_load.plot(load_length  * dt, eqp_inc_history[-1] / dt, 'x', color=color_load, ms=10, label='')
    ax_load.set_xlabel(r'Time $t$ [s]')
    ax_load.set_ylabel(r'Axial strain rate $\dot{\varepsilon}_{zz}$ [1/s]')
    ax_load.set_xlim(-10, 410)
    ax_load.legend()
    
# Deviatoric Stress vs. Pressure
def plot_p_q(p, q, label, color_qs, points):
    point1 = points[0]
    segmented_plot(ax_p_q, p, q, [point1, len(p)-1], label=label, color=color_qs)
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if m_i == '^':
            mfc = 'red'
        else:
            mfc = 'none'
        ax_p_q.plot(p[p_i], q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)
    ax_p_q.set_xlabel(pressure_label)
    ax_p_q.set_ylabel(dev_stress_label)
    ax_p_q.legend()    
    
# Void Ratios vs. Pressure
def plot_e_p(p, void_ratio_q, label, color_qs, points):
    point1 = points[0]
    segmented_plot(ax_e_p, p, void_ratio_q, [point1, len(p)-1], label=label, color=color_qs)
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if m_i == '^':
            mfc = 'red'
        else:
            mfc = 'none'
        ax_e_p.plot(p[p_i], void_ratio_q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)
    ax_e_p.set_xlabel(pressure_label)
    ax_e_p.set_ylabel(e_label)
    ax_e_p.legend()    

# Gamma dot vs. mu
def plot_gamma_mu(xg, yg, label, color_total, points):
    point1 = points[0]
    segmented_plot(ax_mu, xg, yg, [point1, len(p)-1], label=label, color=color_total)
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if m_i == '^':
            mfc = 'red'
        else:
            mfc = 'none'
        ax_mu.plot(xg[p_i], yg[p_i], marker=m_i, color=color_total, ms=10, mfc=mfc)
    ax_mu.set_xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    ax_mu.set_ylabel(mu_label)
    ax_mu.legend()

# Solid volume fraction vs. shear rate
def plot_gamma_e(void_ratio, xg, label, color_total, points):
    point1 = points[0]
    segmented_plot(ax_void, xg, void_ratio, [point1, len(p)-1], label=label, color=color_total)
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if m_i == '^':
            mfc = 'red'
        else:
            mfc = 'none'
        ax_void.plot(xg[p_i], void_ratio[p_i], marker=m_i, color=color_total, ms=10, mfc=mfc)
    ax_void.set_xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    ax_void.set_ylabel(r'Void ratio $e$')
    ax_void.legend()

# Cam-clay critical state parameters
pc_0 = 1 * 150.0  # 'Initial consolidation pressure [kPa]'
M = 1.0  # 'Critical friction angle'
lambda_val = 0.2  # 'Lambda'
kappa = 0.04  # 'kappa'
Gamma = 1.39  # 'Intercept of the critical state line'
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
Delta_Phi = 0.025
M_c = 1.5

# Time needed to accelerate or decelerate
accel_times = [0.02, 0.1, 0.5]
total_time = 2.5
n_runs = len(accel_times)

# collect output per acceleration rate
p_total_list = []
q_total_list = []
void_ratio_total_list = []
p_list = []
q_list = []
void_ratio_q_list = []
pc_history_list = []
p_c_list = []
q_c_list = []
eq_list = []
ev_list = []
ev_cp_list = []
ev_qp_list = []

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)

for k, accel_time in enumerate(accel_times):

    alpha = (k / max(n_runs-1, 1)) * 0.7  
    # 0.7 so they never become too washed-out (tune if needed)

    color_qs    = lighten('b', alpha)   # quasi-static curves
    color_total = lighten('k', alpha)   # total curves
    color_csl   = 'r'                   # critical state line
    color_load  = lighten('k', alpha)   # load history curve (light dark-grey)
    
    # Define a loading history
    eqp_inc = eqp_tot / (1e2 * time / dt)  # [-] 'incrementâlly applied plastic shear strain'
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
    load_length = eqp_inc_history.shape[0]
    
    # Declarations
    void_ratio_total = np.zeros(load_length)
    void_ratio_q = np.zeros(load_length)
    void_ratio_cp = np.zeros(load_length)
    p = np.zeros(load_length)
    q = np.zeros(load_length)
    u = np.zeros(load_length)
    ev = np.zeros(load_length)
    ev_cp = np.zeros(load_length)
    ev_qp = np.zeros(load_length)
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
    void_ratio_cp[0] = void_ratio_0

    # Initialize the yield surf surface
    yield_surf = (q[0] ** 2 / M ** 2 + p[0] ** 2) - p[0] * pc
    pc_history[0] = pc
    
    # Loadstep cycle pressure control shear
    for i, eqp_inc in enumerate(eqp_inc_history[:-1]):
        # Acceleration increment
        d_eqp_inc = eqp_inc_history[i + 1] - eqp_inc if i > 0 else 0
    
        # Stiffness matrix and strain increment vector and derivatives of the yield function
        De = np.zeros([6, 6])
        df_ds = np.zeros([6, 1])
        df_dep = np.zeros([6, 1])

        # Calculate the bulk and shear modulus
        K = V * p[i] / kappa  # Bulk Modulus
        G = (3 * K * (1 - 2 * nu)) / (2 * (1 + nu))  # Shear Modulus

        # Update pre-consolidation pressure
        if yield_surf == 0:
            # pc = pc_history[i] * (1.0 + V * de_v_p / (lambda_val - kappa))
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
                    elif n <= 2:
                        De[m, n] = K - 2 / 3 * G
                if m > 2:
                    df_ds[m, 0] = 0
                    df_dep[m, 0] = 0
                    if m == n:
                        De[m, n] = G
                    else:
                        De[m, n] = 0

        def get_dynamic_stiffness(K_c, G_c):
            D_c = np.zeros([6, 6])
            De_c = np.zeros([6, 6])
            for m in range(6):
                for n in range(6):
                    if m <= 2:
                        if m == n:
                            D_c[m, n] = K_c + 4 / 3 * G_c
                            De_c[m, n] = K_c
                        elif n <= 2:
                            D_c[m, n] = K_c - 2 / 3 * G_c
                            De_c[m, n] = K_c
                    if m > 2:
                        if m == n:
                            D_c[m, n] = G_c
                        else:
                            D_c[m, n] = 0
            return D_c, De_c

        # If the yield surface is negative, the stiffness matrix is elastic
        if yield_surf < 0:
            D_q = De

        # If the yield surface is positive, the stiffness matrix is elastic-plastic
        else:
            D_q = De - (De.dot(df_ds).dot(df_ds.T).dot(De)) / (
                    -(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds))

        def compute_stress_increment(args):
            global K_c, G_c, de_v_c, de_q_direction, d_eqp_inc
            ratio_1, ratio_2 = args
            # Fill the strain increment vector
            if deformation_mode == 'drained':
                # For drained (or pressure control) conditions
                d_epsilon = np.array(
                    [eqp_inc, ratio_1 * eqp_inc, ratio_1 * eqp_inc, 0., 0., 0.])
                if d_eqp_inc != 0:
                    d_d_epsilon = np.array(
                        [d_eqp_inc, ratio_2 * d_eqp_inc, ratio_2 * d_eqp_inc, 0., 0., 0.])
                else:
                    d_d_epsilon = np.array([0., 0., 0., 0., 0., 0.])
            elif deformation_mode == 'undrained':
                d_epsilon = np.array(
                    [eqp_inc, -eqp_inc / 2., -eqp_inc / 2., 0., 0., 0.])
                d_d_epsilon = np.array(
                    [d_eqp_inc, -d_eqp_inc / 2., -d_eqp_inc / 2., 0., 0., 0.])

            # Magnitude of the increment in deviatoric shear strain rate (from tensor),
            d_eqp_inc_mag, _ = get_deviatoric_strain(d_d_epsilon)
            # Keep acceleration/deceleration sign convention from loading-history increment.
            d_eqp_inc_eff = np.sign(d_eqp_inc) * d_eqp_inc_mag

            # Calculate dynamic stress
            if d_eqp_inc_mag != 0:
                # Get the deviatoric part of the acceleration rate tensor ?
                Phi = 1.0 / (1.0 + void_ratio_total[i])
                de_v_c = - Phi / Phi ** 2 * Delta_Phi * d_eqp_inc_eff / dt
                # d_p_c[i + 1] = p[i] / ((lambda_val - kappa) * Phi ** 2) * Delta_Phi * d_eqp_inc_eff / dt
                d_p_c[i + 1] = p[i] / ((lambda_val - 0) * Phi ** 2) * Delta_Phi * d_eqp_inc_eff / dt

                K_c = d_p_c[i + 1] / de_v_c
                G_c = d_p_c[i + 1] * M_c / d_eqp_inc_eff / 3
            else:
                p_c[i + 1] = p_c[i]
                q_c[i + 1] = q_c[i]
                de_v_c = 0
                K_c = 0
                G_c = 0

            # Get dynamic stiffness matrices
            D_c, De_c = get_dynamic_stiffness(K_c, G_c)
            # Get dynamic stress-induced plastic strain increment
            d_epsilon_v_c = 1./3. * de_v_c * np.array([1., 1., 1., 0, 0, 0])
            
            # Get quasi-static stress induced plastic volumetric strain increment
            if yield_surf < 0:
                de_v_p = 0
            else:
                # Compute the plastic multiplier increment
                num = (df_ds.T).dot(De).dot(d_epsilon - d_epsilon_v_c)
                den = -(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds)
                # if size of num and den are not 1, through an error
                if num.shape != (1,) or den.shape != (1, 1):
                    raise ValueError(f"num and den should be scalars, but got num: {num}, den: {den}")
                # cast num and den to float to avoid the case where num and den are both zero, which causes dLambda to be an array instead of a scalar
                dLambda = float(num[0] / den[0, 0])
                # Get the plastic volumetric strain increment due to quasi-static stress
                d_epsilon_p = dLambda * df_ds
                de_v_p = float(d_epsilon_p[0, 0] + d_epsilon_p[1, 0] + d_epsilon_p[2, 0])

            # Update stress increments
            d_sigma_q = D_q.dot(d_epsilon - d_epsilon_v_c)
            d_sigma_c = De_c.dot(d_epsilon_v_c) + D_c.dot(d_d_epsilon - 1./3. * d_d_epsilon[:3].sum() * np.array([1., 1., 1., 0, 0, 0]))

            return d_sigma_q, d_sigma_c, d_epsilon, de_v_p
    
        def servo_control(ratios):
            ratio_1, ratio_2 = ratios
            d_sigma_q, d_sigma_c, _, _ = compute_stress_increment([ratio_1, ratio_2])
            d_sigma = d_sigma_q + d_sigma_c
            return np.sum(d_sigma[:3])

        def advance_constant_mean_pressure(eqp_inc, p_target,
                                            tolerance=1e-3,
                                            max_iterations=100,
                                            small_number=1e-14):
            # No acceleration: ratio_2 has no effect on the result, so keep it equal to ratio_1.
            denom = D_q[1, 1] + D_q[1, 2]
            delta_eps_r = - D_q[1, 0] / denom * eqp_inc if np.abs(denom) > small_number else 0.0

            for _ in range(max_iterations):
                ratio_loc = delta_eps_r / eqp_inc if np.abs(eqp_inc) > small_number else 0.0
                d_sigma_q_loc, d_sigma_c_loc, d_epsilon_loc, de_v_p_loc = compute_stress_increment([ratio_loc, ratio_loc])

                sigma_trial = (sigma_q + d_sigma_q_loc) + (sigma_c + d_sigma_c_loc)
                p_trial = np.sum(sigma_trial[:3]) / 3.0
                residual = p_trial - p_target

                if np.abs(residual) <= tolerance:
                    return ratio_loc, d_sigma_q_loc, d_sigma_c_loc, d_epsilon_loc, de_v_p_loc

                fd_step = max(1e-8 * max(1.0, np.abs(delta_eps_r), np.abs(eqp_inc)), small_number)
                ratio_fd = (delta_eps_r + fd_step) / eqp_inc if np.abs(eqp_inc) > small_number else 0.0
                d_sigma_q_fd, d_sigma_c_fd, _, _ = compute_stress_increment([ratio_fd, ratio_fd])
                sigma_trial_fd = (sigma_q + d_sigma_q_fd) + (sigma_c + d_sigma_c_fd)
                p_trial_fd = np.sum(sigma_trial_fd[:3]) / 3.0
                dp_deps_r = (p_trial_fd - p_trial) / fd_step

                if np.abs(dp_deps_r) < small_number:
                    dp_deps_r = (
                        D_q[0, 1] + D_q[0, 2]
                        + D_q[1, 1] + D_q[1, 2]
                        + D_q[2, 1] + D_q[2, 2]
                    ) / 3.0

                if np.abs(dp_deps_r) < small_number:
                    raise RuntimeError('Pressure-control Jacobian is singular')

                delta_eps_r = delta_eps_r - residual / dp_deps_r

            raise RuntimeError('Constant-pressure iteration did not converge')

        def solve_ratios_levenberg_marquardt(ratio_1_guess, ratio_2_guess,
                                              tolerance=1e-6,
                                              max_iterations=50,
                                              lm_lambda_init=1e-2,
                                              small_number=1e-14):
            # With acceleration (d_eqp_inc != 0): both ratio_1 and ratio_2 affect the
            # residual, but there is still only 1 equation (mean-stress residual) for
            # 2 unknowns. Use a damped Gauss-Newton (Levenberg-Marquardt) minimum-norm
            # step, which is stable even when d_eqp_inc is small (unlike dividing by it).
            x = np.array([ratio_1_guess, ratio_2_guess], dtype=float)
            lm_lambda = lm_lambda_init
            residual = servo_control(x)

            for _ in range(max_iterations):
                if np.abs(residual) <= tolerance:
                    break

                fd_step_1 = max(1e-6 * max(1.0, np.abs(x[0])), small_number)
                fd_step_2 = max(1e-6 * max(1.0, np.abs(x[1])), small_number)
                residual_fd1 = servo_control([x[0] + fd_step_1, x[1]])
                residual_fd2 = servo_control([x[0], x[1] + fd_step_2])
                J = np.array([
                    (residual_fd1 - residual) / fd_step_1,
                    (residual_fd2 - residual) / fd_step_2,
                ])

                J_norm_sq = J.dot(J)
                if J_norm_sq < small_number:
                    break

                for _ in range(10):
                    step = -(residual * J) / (J_norm_sq + lm_lambda)
                    x_trial = x + step
                    residual_trial = servo_control(x_trial)

                    if np.abs(residual_trial) < np.abs(residual):
                        x = x_trial
                        residual = residual_trial
                        lm_lambda = max(lm_lambda * 0.5, 1e-8)
                        break
                    else:
                        lm_lambda *= 4.0
                else:
                    break

            return x[0], x[1]

        # Update stress
        if deformation_mode == 'undrained':
            d_sigma_q, d_sigma_c, d_epsilon, de_v_p = compute_stress_increment([-0.5, -0.5])
        elif deformation_mode == 'drained':
            if d_eqp_inc == 0:
                # No acceleration: use the pressure-control Newton line-search
                ratio_1, d_sigma_q, d_sigma_c, d_epsilon, de_v_p = advance_constant_mean_pressure(
                    eqp_inc=eqp_inc,
                    p_target=p0,
                )
                ratio_2 = ratio_1
            else:
                ratio = - D_q[1, 0] / (D_q[1, 1] + D_q[1, 2])
                ratio_1, ratio_2 = solve_ratios_levenberg_marquardt(ratio, ratio)
                d_sigma_q, d_sigma_c, d_epsilon, de_v_p = compute_stress_increment([ratio_1, ratio_2])
        
        sigma_q += d_sigma_q
        sigma_c += d_sigma_c
        sigma = sigma_q + sigma_c
        epsilon += d_epsilon
    
        # Update stress invariants
        p_total[i + 1] = np.sum(sigma[:3]) / 3.0
        p_s = sigma - np.array([1., 1., 1., 0, 0, 0]) * p_total[i + 1]
        q_total[i + 1] = np.sqrt(3. / 2. * p_s.dot(p_s))
        pc_history[i + 1] = pc

        p_c[i + 1] = np.sum(sigma_c[:3]) / 3.0
        p_s_c = sigma_c - np.array([1., 1., 1., 0, 0, 0]) * p_c[i + 1]
        q_c[i + 1] = np.sqrt(3. / 2. * p_s_c.dot(p_s_c))
    
        # Compute stress and strain invariants
        ev[i + 1] = np.sum(epsilon[:3])
        epsilon_s = epsilon - np.array([1, 1, 1, 0, 0, 0]) * ev[i + 1]
        eq[i + 1] = np.sqrt(2. / 3. * epsilon_s.dot(epsilon_s))
        ev_cp[i + 1] = ev_cp[i] + de_v_c
        ev_qp[i + 1] = ev_qp[i] + de_v_p
    
        p[i + 1] = np.sum(sigma_q[:3]) / 3.0
        p_q_s = sigma_q - np.array([1., 1., 1., 0, 0, 0]) * p[i + 1]
        q[i + 1] = np.sqrt(3. / 2. * p_q_s.dot(p_q_s))
        u[i + 1] = p0 + q_total[i + 1] / 3. - p_total[i + 1]
    
        # Update specific volume
        V = N - (lambda_val * np.log(pc)) + (kappa * np.log(pc / p[i + 1]))
        void_ratio_q[i + 1] = V - 1
        void_ratio_total[i + 1] = void_ratio_total[i] - (1 + void_ratio_total[i]) * d_epsilon[:3].sum()
    
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
    pc_history_list.append(pc_history)
    p_c_list.append(p_c)
    q_c_list.append(q_c)
    eq_list.append(eq)
    ev_list.append(ev)
    ev_cp_list.append(ev_cp)
    ev_qp_list.append(ev_qp)

    # Display results
    plt.figure('Pressure controlled simple shear')
    
    # Full text long labels
    time_label = r'Time ($t$) [s]'
    mu_label = r'Stress ratio $q/p$ [-]'
    pressure_label = r'Pressure $p$ [kPa]'
    dev_stress_label = r'Deviatoric stress $q$ [kPa]'
    ev_label = r'Volumetric strain $\varepsilon_v$ [-]'
    preconsolidation_p_label = 'Pre-consolidation pressure [kPa]'
    p_ratio_label = r'Ratio of dynamic and quasi-static stresses [-]'
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
    
    # dynamic stress (iso) vs. time
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
    # Save figure
    accel = max(eqp_inc_history[1:]-eqp_inc_history[0:-1])/dt/dt
    plt.savefig(f'{round(accel,3):.2f}_summary' + '.png', dpi=300, bbox_inches='tight')   

# Replot the figures
fig_load_history, ax_load = plt.subplots()
fig_p_q, ax_p_q = plt.subplots(figsize=figsize)
fig_e_p, ax_e_p = plt.subplots(figsize=figsize)
fig_mu, ax_mu = plt.subplots(figsize=figsize)
fig_void, ax_void = plt.subplots(figsize=figsize)
for k, (p_total, q_total, void_ratio_total,
        p, q, void_ratio_q,
        p_c, q_c,
        eq, ev, ev_cp, ev_qp,
        accel_time) in enumerate(reversed(list(zip(p_total_list, q_total_list, void_ratio_total_list,
                                               p_list, q_list, void_ratio_q_list,
                                               p_c_list, q_c_list,
                                               eq_list, ev_list, ev_cp_list, ev_qp_list,
                                               accel_times)))):
    alpha = (1 - k / max(n_runs - 1, 1)) * 0.7
    color_qs    = lighten('b', alpha)   # quasi-static curves
    color_csl   = 'r'                   # critical state line
    color_load  = lighten('k', alpha)   # load history curve (light dark-grey)

    # Define a loading history
    eqp_inc = eqp_tot / (1e2 * time / dt)  # [-] 'incrementâlly applied plastic shear strain'
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
    markers = ['^', '>', 'v', '<', 'x']
    load_length = eqp_inc_history.shape[0]

    # Individual production plots
    accel = max(eqp_inc_history[1:]-eqp_inc_history[0:-1])/dt/dt
    points = [point1, point2, point3, point4, len(p)-1]
    label = rf'$|\ddot{{\varepsilon}}_{{zz}}| = {np.ceil(accel*1e2)/1e2:.2f}$ s' + r'$^{-2}$'
    plot_load(load_length, dt, eqp_inc_history, label, color_load, points, markers)
    plot_p_q(p, q, label, color_qs, points)
    # plot_p_q(p_total, q_total, label, color_load, points)
    plot_e_p(p, void_ratio_q, label, color_qs, points)
    # plot_e_p(p_total, void_ratio_total, label, color_load, points)
    gamma_dot = (eq[1:] - eq[:-1]) / dt
    gamma_dot = np.insert(gamma_dot, 0, 0)
    plot_gamma_mu(gamma_dot, q/p, label, color_qs, points)
    # plot_gamma_mu(gamma_dot, q_total/p_total, label, color_load, points)
    plot_gamma_e(void_ratio_q, gamma_dot, label, color_qs, points)
    # plot_gamma_e(void_ratio_total, gamma_dot, label, color_load, points)

x_min, x_max = ax_p_q.get_xlim()
y_min, y_max = ax_p_q.get_ylim()
p_line = np.linspace(x_min, x_max, 100)
# Critical state line in p-q
ax_p_q.plot(p_line, M * p_line, '-', color=color_csl, label='Critical state line', zorder=0)
ax_p_q.set_xlim(x_min, x_max)
ax_p_q.set_ylim(y_min, y_max)
ax_p_q.legend()
# Critical state line in e-p
Gamma = void_ratio_q[point1] + lambda_val * np.log(p[point1])
eCSL = Gamma - lambda_val * np.log(p_line)
ax_e_p.plot(p_line, eCSL, '-', color=color_csl, label='Critical state line', zorder=0)
ax_e_p.set_xlim(x_min, x_max)
ax_e_p.legend()
# Dilatancy relation in gamma phi
x_min, x_max = ax_void.get_xlim()
gamma_line = np.linspace(x_min, x_max, 100)
dilatancy_line = 1.0/(1.0 + void_ratio_q_list[0][point1]) - Delta_Phi * gamma_line
ax_void.plot(gamma_line, (1.0 - dilatancy_line) / dilatancy_line, color=color_csl, label='Rate-induced dilatancy', zorder=0)
ax_void.set_xlim(x_min, x_max)
ax_void.legend()
# Bulk friction in gamma mu
ax_mu.plot(gamma_line, np.ones(100) * M, color=color_csl, label=r'$\mu^{\mathrm{cs}}$', zorder=0)
ax_mu.set_xlim(x_min, x_max)
ax_mu.legend()

# Expand y-limits once at the end so numeric annotations stay inside the axes
expand_ylim_for_annotations(ax_load, top_frac=0.05)
expand_ylim_for_annotations(ax_p_q, top_frac=0.3)
expand_ylim_for_annotations(ax_e_p)
expand_ylim_for_annotations(ax_mu, top_frac=0.05, bottom_frac=0.05)
expand_ylim_for_annotations(ax_void, top_frac=0.3)

fig_load_history.savefig(f"{deformation_mode}_load_history.png", dpi=300, bbox_inches="tight")
fig_p_q.savefig(f"{deformation_mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
fig_e_p.savefig(f"{deformation_mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
fig_mu.savefig(f"{deformation_mode}_gamma_vs_mu_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
fig_void.savefig(f"{deformation_mode}_gamma_vs_phi_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
plt.show()