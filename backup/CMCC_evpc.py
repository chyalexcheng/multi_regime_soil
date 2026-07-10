# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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

# Cam-clay critical state parameters
pc_0 = 3 * 150.0  # 'Initial consolidation pressure [kPa]'
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
Delta_Phi = 25.0
M_c = 1.5

# Define a loading history
eqp_inc = eqp_tot / (1e2 * load_length)  # [-] 'incrementâlly applied plastic shear strain'
eqp_inc_history = np.concatenate([
    np.ones(round(1.0 / eqp_inc)) * eqp_inc,
    np.linspace(eqp_inc, 100 * eqp_inc, round(0.5 / eqp_inc)),
    np.ones(round(1.0 / eqp_inc)) * 100 * eqp_inc,
    np.linspace(100 * eqp_inc, eqp_inc, round(0.5 / eqp_inc)),
    np.ones(round(0.5 / eqp_inc)) * eqp_inc,
])
point1 = round(1.0 / eqp_inc)
point2 = point1 + round(0.5 / eqp_inc)
point3 = point2 + round(1.0 / eqp_inc)
point4 = point3 + round(0.5 / eqp_inc)
points = [point1, point2, point3, point4]
markers = ['^', '>', 'v', '<']
load_length = eqp_inc_history.shape[0]

plt.figure('Load history')
plt.plot(np.arange(load_length) * dt, eqp_inc_history / dt, '-k', label=rf'$|\ddot{{\varepsilon}}_{{zz}}| = 0.02$ s' + r'$^{-2}$')
x_offset = load_length * dt * 0.03
y_offset = max(eqp_inc_history) / dt * 0.03
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, eqp_inc_history[p_i] / dt, f'k{m_i}',ms=10, mfc='none');
    plt.text(p_i * dt + x_offset, eqp_inc_history[p_i] / dt  + y_offset, f"{i+1}", fontsize=14)
plt.plot(0 * dt, eqp_inc_history[0] / dt, 'k.', ms=10, label='')
plt.plot(load_length * dt, eqp_inc_history[-1] / dt, 'kx', ms=10, label='')
plt.xlabel(r'Time $t$ [s]')
plt.ylabel(r'Axial strain rate $\dot{\varepsilon_{zz}}$ [1/s]')
ymin, ymax = plt.ylim()
plt.xlim(-10, 360)
plt.ylim(ymin, ymax + np.ceil(y_offset * 10) / 10)
plt.legend()
plt.savefig(f"load_history.png", dpi=300, bbox_inches="tight")

# Declarations
void_ratio_total = np.zeros(load_length)
void_ratio_q = np.zeros(load_length)
void_ratio_cp = np.zeros(load_length)
p = np.zeros(load_length)
q = np.zeros(load_length)
u = np.zeros(load_length)
ev = np.zeros(load_length)
ev_e = np.zeros(load_length)
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

de_q_direction = np.zeros(6)
void_ratio_total[0] = void_ratio_0
void_ratio_q[0] = void_ratio_0
void_ratio_cp[0] = void_ratio_0

# Initialize the yield surf surface
yield_surf = (q[0] ** 2 / M ** 2 + p[0] ** 2) - p[0] * pc

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
        global K_c, G_c, de_v_c, d_epsilon_prev
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
            de_v_c = - Phi / Phi ** 2 * Delta_Phi * d_eqp_inc_eff
            # d_p_c[i + 1] = p[i] / ((lambda_val - kappa) * Phi ** 2) * Delta_Phi * d_eqp_inc_eff
            d_p_c[i + 1] = p[i] / ((lambda_val - 0) * Phi ** 2) * Delta_Phi * d_eqp_inc_eff

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
                                          tolerance=1e-6,        # convergence threshold on |residual| = |sum(d_sigma[:3])| [kPa]; stop once mean-stress imbalance is this small
                                          max_iterations=50,     # cap on outer LM iterations (accept/reject steps) before giving up and returning current best estimate
                                          lm_lambda_init=1e-2,   # initial damping factor lambda: larger -> more gradient-descent-like (safer, smaller steps); smaller -> more Gauss-Newton-like (faster, riskier steps)
                                          small_number=1e-14):   # generic near-zero threshold used to guard against division by zero / degenerate Jacobians
        """
        Solve for the two ratios (ratio_1, ratio_2) that drive the mean stress increment to zero
        under acceleration/deceleration (d_eqp_inc != 0) conditions. The two ratios are
        coupled through the stress increment equations, so we use a damped Gauss-Newton
        (Levenberg-Marquardt) approach to find a solution that minimizes the residual.
        
        Parameters
        ----------
        ratio_1_guess : float
            Initial guess for the first ratio (d_epsilon[1] / d_epsilon[0]).
        ratio_2_guess : float
            Initial guess for the second ratio (d_d_epsilon[1] / d_d_epsilon[0]).
        tolerance : float, optional
            Convergence threshold on |residual| = |sum(d_sigma[:3])| [kPa]; stop once mean-stress imbalance is this small. Defaults to 1e-6.
        max_iterations : int, optional
            Cap on outer LM iterations (accept/reject steps) before giving up and returning current best estimate. Defaults to 50.
        lm_lambda_init : float, optional
            Initial damping factor lambda: larger -> more gradient-descent-like (safer, smaller steps); smaller -> more Gauss-Newton-like (faster, riskier steps). Defaults to 1e-2.
        small_number : float, optional
            Generic near-zero threshold used to guard against division by zero / degenerate Jacobians. Defaults to 1e-14.
        
        Returns
        -------
        tuple[float, float]
            The converged values of (ratio_1, ratio_2).
        """
        x = np.array([ratio_1_guess, ratio_2_guess], dtype=float)
        lm_lambda = lm_lambda_init
        residual = servo_control(x)

        for _ in range(max_iterations):
            if np.abs(residual) <= tolerance:
                break

            # Forward-difference step sizes for the Jacobian: relative step of 1e-6 (times
            # the current parameter magnitude, floored at 1.0) balances truncation error
            # (step too large) against floating-point cancellation error (step too small).
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

            # Damped minimum-norm step: (J^T J + lm_lambda * I) dx = -J^T residual,
            # solved in closed form for the rank-1 (1 equation, 2 unknowns) case.
            # Inner loop: up to 10 trust-region-style trials to find a lambda that
            # actually reduces |residual| before giving up on this outer iteration.
            for _ in range(10):
                step = -(residual * J) / (J_norm_sq + lm_lambda)
                x_trial = x + step
                residual_trial = servo_control(x_trial)
                if np.abs(residual_trial) < np.abs(residual):
                    x = x_trial
                    residual = residual_trial
                    # Step accepted: relax damping toward Gauss-Newton (halve lambda),
                    # but never let it fall below 1e-8 (keeps a minimum regularization).
                    lm_lambda = max(lm_lambda * 0.5, 1e-8)
                    break
                # Step rejected: increase damping 4x (more gradient-descent-like) and retry.
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
            # With acceleration: solve the coupled ratios via damped Gauss-Newton
            # (Levenberg-Marquardt), which is stable even when d_eqp_inc is small.
            ratio_1, ratio_2 = solve_ratios_levenberg_marquardt(ratio_1, ratio_2)
            d_sigma_q, d_sigma_c, d_epsilon, de_v_p = compute_stress_increment([ratio_1, ratio_2])

    sigma_q += d_sigma_q
    sigma_c += d_sigma_c
    sigma = sigma_q + sigma_c
    epsilon += d_epsilon

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
    ev_cp[i + 1] = ev_cp[i] + de_v_c
    ev_qp[i + 1] = ev_qp[i] + de_v_p

    p[i + 1] = np.sum(sigma_q[:3]) / 3.0
    p_q_s = sigma_q - np.array([1., 1., 1., 0, 0, 0]) * p[i + 1]
    q[i + 1] = np.sqrt(3. / 2. * p_q_s.dot(p_q_s))
    u[i + 1] = p0 + q_total[i + 1] / 3. - p_total[i + 1]

    # Update specific volume
    V = N - (lambda_val * np.log(pc)) + (kappa * np.log(pc / p[i + 1]))
    void_ratio_q[i + 1] = V - 1
    void_ratio_cp[i + 1] = void_ratio_cp[i] - (1 + void_ratio_total[i]) * de_v_c
    void_ratio_total[i + 1] = void_ratio_total[i] - (1 + void_ratio_total[i]) * d_epsilon[:3].sum()

    if yield_surf < 0:
        yield_surf = q[i + 1] ** 2 + M ** 2 * p[i + 1] ** 2 - M ** 2 * p[i + 1] * pc
    else:
        yield_surf = 0

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


#%% Individual production plots

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)
text_box_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='gray')

# Pressure vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, p_total, '-g', label=r"Total pressure ($p$)")
plt.plot(np.arange(load_length) * dt, p, '-b', label=r"Quasi-static pressure ($p^{\mathrm{q}}$)")
plt.plot(0 * dt, p[0], 'b.', ms=10, label='')
plt.plot(0 * dt, p_total[0], 'g.', ms=10, label='')
plt.plot(load_length * dt, p[-1], 'bx', ms=10, label='')
plt.plot(load_length * dt, p_total[-1], 'gx', ms=10, label='')
x_offset = load_length * dt * 0.035
y_offset_p = max(p) * 0.01
y_offset_p_total = max(p_total) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, p[p_i], f'b{m_i}',ms=10, mfc='none');
    plt.plot(p_i * dt, p_total[p_i], f'g{m_i}',ms=10, mfc='none');
    plt.text(p_i * dt + x_offset, p[p_i] - y_offset_p, f"{i+1}", fontsize=14, color='blue', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, p_total[p_i] - y_offset_p_total, f"{i+1}", fontsize=14, color='green', bbox=text_box_props)
plt.xlabel(time_label)
plt.ylabel(pressure_label)
if deformation_mode == 'undrained' and OCR == 3.0:
    plt.ylim(-10, 360)
elif deformation_mode == 'undrained' and OCR == 1.0:
    plt.ylim(-5, 165)
plt.legend()
plt.savefig(f"{deformation_mode}_p_and_p_c_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Deviatoric Stress (q) vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, q_total, '-g', label=r"Total deviatoric stress ($q$)")
plt.plot(np.arange(load_length) * dt, q, '-b', label=r"Quasi-static deviatoric stress ($q^{\mathrm{q}}$)")
plt.plot(0 * dt, q[0], 'b.', ms=10, label='')
plt.plot(0 * dt, q_total[0], 'g.', ms=10, label='')
plt.plot(load_length * dt, q[-1], 'bx', ms=10, label='')
plt.plot(load_length * dt, q_total[-1], 'gx', ms=10, label='')
y_offset_q = max(q_total) * 0.01
y_offset_p_total = max(p_total) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, q[p_i], f'b{m_i}',ms=10, mfc='none')
    plt.plot(p_i * dt, q_total[p_i], f'g{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, q[p_i] - y_offset_q, f"{i+1}", fontsize=12, color='blue', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, q_total[p_i] - y_offset_p_total, f"{i+1}", fontsize=12, color='green', bbox=text_box_props)
plt.xlabel(time_label)
plt.ylabel(dev_stress_label)
if deformation_mode == 'drained':
    plt.ylim(-7, 220)
elif deformation_mode == 'undrained' and OCR == 3.0:
    plt.ylim(-10, 360)
elif deformation_mode == 'undrained' and OCR == 1.0:
    plt.ylim(-5, 165)
plt.legend(loc='lower center')
plt.savefig(f"{deformation_mode}_q_and_q_c_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Volumetric and Deviatoric Strain vs. Time
plt.figure(figsize=figsize)
ax1 = plt.gca()
ax1.plot(np.arange(load_length) * dt, eq, '-k', ms=10, label='')
ax1.plot(0 * dt, eq[0], 'k.', ms=10, label='')
ax1.plot(load_length * dt, eq[-1], 'kx', ms=10, label='')
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, eq[p_i], f'k{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, eq[p_i], f"{i+1}", fontsize=14, color='gray', bbox=text_box_props)
ax1.set_xlabel(time_label)
ax1.set_ylim(0,160)
ax1.set_yticks(np.linspace(0, 160, 6))
ax1.set_ylabel(r'Deviatoric strain ($\gamma$)', color='black')
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
ax2.plot(np.arange(load_length) * dt, ev, '-', color='gray')
ax2.plot(0 * dt, ev[0], '.', ms=10, label='', color='gray')
ax2.plot(load_length * dt, ev[-1], 'x', ms=10, label='', color='gray')
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, ev[p_i], f'{m_i}',ms=10, mfc='none', color='gray')
    plt.text(p_i * dt + x_offset, ev[p_i] - 0.02, f"{i+1}", fontsize=14, color='gray', bbox=text_box_props)
ax2.set_ylabel(r"Volumetric strain ($\varepsilon_v$)", color='gray')
ax2.set_ylim(bottom=-0.15, top=0.1)
ax2.tick_params(axis='y', labelcolor='gray')
plt.savefig(f"{deformation_mode}_e_v_and_gamma_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Pre-consolidation Pressure vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, pc_history, '-b')
plt.plot(0 * dt, pc_history[0], 'b.', ms=10, label='')
plt.plot(load_length * dt, pc_history[-1], 'bx', ms=10, label='')
y_offset = max(pc_history) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, pc_history[p_i], f'b{m_i}',ms=10, mfc='none');
    plt.text(p_i * dt + x_offset, pc_history[p_i] + y_offset, f"{i+1}", fontsize=14, color='blue', bbox=text_box_props)
plt.xlabel(time_label)
plt.ylabel(preconsolidation_p_label)
plt.savefig(f"{deformation_mode}_preconsolidate_p_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Ratio between dynamic and quasistatic stress vs. Time
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length-1) * dt, p_c[1:] / p[1:], '-c', label=r'Pressures ($p^{\mathrm{d}}/p^{\mathrm{q}}$)')
plt.plot(0 * dt, p_c[1] / p[1], 'c.', ms=10, label='')
plt.plot(load_length * dt, p_c[-1] / p[-1], 'cx', ms=10, label='')
plt.plot(np.arange(load_length-1) * dt, q_c[1:] / q[1:], '-.c', label=r'Deviatoric stresses ($q^{\mathrm{d}}/q^{\mathrm{q}}$)')
plt.plot(0 * dt, q_c[1] / q[1], 'c.', ms=10, label='')
plt.plot(load_length * dt, q_c[-1] / q[-1], 'cx', ms=10, label='')
y_offset_p = max(p_c[1:] / p[1:]) * 0.01
y_offset_q = max(q_c[1:] / q[1:]) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, p_c[p_i]/p[p_i], f'c{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, p_c[p_i]/p[p_i] + y_offset_p, f"{i+1}", fontsize=14, color='cyan', bbox=text_box_props)
    plt.plot(p_i * dt, q_c[p_i]/q[p_i], f'c{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, q_c[p_i]/q[p_i] + y_offset_q, f"{i+1}", fontsize=14, color='cyan', bbox=text_box_props)
plt.xlabel(time_label)
plt.ylabel(p_ratio_label)
plt.legend()
plt.savefig(f"{deformation_mode}_p_c_over_p_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Deviatoric Stress vs. Pressure
plt.figure(figsize=figsize)
plt.plot([0, np.max(p) * 1.5], [0, M * np.max(p) * 1.5], '-r', label='Critical state line')
plt.plot(p_total, q_total, '-g', label=r"Total path")
plt.plot(p, q, '-b', markevery=500, label=r"Quasi-static path")
plt.plot(p[0], q[0], 'b.', ms=10, label='')
plt.plot(p[-1], q[-1], 'bx', ms=10, label='')
plt.plot(p_total[0], q_total[0], 'g.', ms=10, label='')
plt.plot(p_total[-1], q_total[-1], 'gx', ms=10, label='')
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_total[p_i], q_total[p_i], f'g{m_i}',ms=10, mfc='none')
    plt.plot(p[p_i], q[p_i], f'b{m_i}',ms=10, mfc='none')
plt.xlabel(pressure_label)
plt.ylabel(dev_stress_label)
if deformation_mode == 'drained':
    plt.xlim(0,160)
    plt.ylim(0,225)
elif deformation_mode == 'undrained':
    if OCR > 1.0:
        plt.xlim(0,350)
        plt.ylim(0,400)
    else:
        plt.xlim(0,160)
        plt.ylim(0,180)
plt.legend(loc='upper left')
plt.savefig(f"{deformation_mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Void Ratios vs. Pressure
plt.figure(figsize=figsize)
plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
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
plt.plot(p_total, void_ratio_total, '-g', label='Total void ratio')
plt.plot(p, void_ratio_q, '-b', label=r"Quasi-static void ratio")
plt.plot(p[0], void_ratio_q[0], 'b.', ms=10, label='')
plt.plot(p_total[0], void_ratio_total[0], 'g.', ms=10, label='')
plt.plot(p[-1], void_ratio_q[-1], 'bx', ms=10, label='')
plt.plot(p_total[-1], void_ratio_total[-1], 'gx', ms=10, label='')
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p[p_i], void_ratio_q[p_i], f'b{m_i}',ms=10, mfc='none')
    plt.plot(p_total[p_i], void_ratio_total[p_i], f'g{m_i}',ms=10, mfc='none')
plt.xlim(xlim)
plt.ylim(ylim)
if deformation_mode == 'drained':
    plt.ylim(0.32, 0.50)
elif deformation_mode == 'undrained':
    if OCR > 1.0:
        plt.ylim(0.26, 0.33)
    else:
        plt.ylim(0.44, 0.52)
plt.xlabel(pressure_label)
plt.ylabel(e_label)
#plt.legend()
plt.savefig(f"{deformation_mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Bulk Friction vs. Time
plt.figure(figsize=figsize)
plt.axhline(1.0, color='red', linestyle='--', label=r'$\mu^{\mathrm{cs}}$')
plt.axhline(1.5, color='red', linestyle=':', label=r'$\mu^{\mathrm{d}}$')
plt.plot(np.arange(load_length) * dt, q_total / p_total, '-g', label=r"Total friction ($\mu$)")
plt.plot(np.arange(load_length) * dt, q / p, '-b', label=r"Quasi-static friction ($\mu^{\mathrm{q}}$)")
plt.plot(0 * dt, q[0] / p[0], 'b.', ms=10, label='')
plt.plot(0 * dt, q_total[0] / p_total[0], 'g.', ms=10, label='')
plt.plot(load_length * dt, q[-1] / p[-1], 'bx', ms=10, label='')
plt.plot(load_length * dt, q_total[-1] / p_total[-1], 'gx', ms=10, label='')
y_offset_quasi_ratio = max(q / p) * 0.01
y_offset_total_ratio = max(q_total / p_total) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, q[p_i] / p[p_i], f'b{m_i}',ms=10, mfc='none')
    plt.plot(p_i * dt, q_total[p_i] / p_total[p_i], f'g{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, q[p_i] / p[p_i] - y_offset_quasi_ratio, f"{i+1}", fontsize=14, color='blue', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, q_total[p_i] / p_total[p_i] - y_offset_total_ratio, f"{i+1}", fontsize=14, color='green', bbox=text_box_props)
plt.xlabel(time_label)
plt.ylabel(mu_label)
plt.legend()
plt.savefig(f"{deformation_mode}_mu_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Void ratios (volumetric strains) vs. Time
ev_e = ev - ev_qp - ev_cp
plt.figure(figsize=figsize)
plt.plot(np.arange(load_length) * dt, ev, '-', color='gray', label=r"$\varepsilon_v$")
plt.plot(np.arange(load_length) * dt, ev_e, '-g', label=r"$\varepsilon_v^{\mathrm{e}}$")
plt.plot(np.arange(load_length) * dt, ev_qp, '-b', label=r"$\varepsilon_v^{\mathrm{p,q}}$")
plt.plot(np.arange(load_length) * dt, ev_cp, '-m', label=r"$\varepsilon_v^{\mathrm{p,c}}$")
x_offset = load_length * dt * 0.04
y_offset_qp = max(ev_qp) * 0.01
y_offset_e = max(ev_e) * 0.01
y_offset_cp = max(ev_cp) * 0.01
for i, (p_i, m_i) in enumerate(zip(points, markers)):
    plt.plot(p_i * dt, ev[p_i], f'{m_i}',color='gray',ms=10, mfc='none')
    plt.plot(p_i * dt, ev_qp[p_i], f'b{m_i}',ms=10, mfc='none')
    plt.plot(p_i * dt, ev_e[p_i], f'g{m_i}',ms=10, mfc='none')
    plt.plot(p_i * dt, ev_cp[p_i], f'm{m_i}',ms=10, mfc='none')
    plt.text(p_i * dt + x_offset, ev[p_i] - y_offset_e, f"{i+1}", fontsize=12, color='gray', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, ev_qp[p_i] + y_offset_qp, f"{i+1}", fontsize=12, color='blue', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, ev_e[p_i] + y_offset_e, f"{i+1}", fontsize=12, color='green', bbox=text_box_props)
    plt.text(p_i * dt + x_offset, ev_cp[p_i] + y_offset_cp, f"{i+1}", fontsize=12, color='magenta', bbox=text_box_props)
plt.plot(0 * dt, 0, 'b.', ms=10, label='')
plt.plot(0 * dt, 0, 'g.', ms=10, label='')
plt.plot(0 * dt, 0, 'm.', ms=10, label='')
plt.plot(load_length * dt, ev_qp[-1], 'bx', ms=10, label='')
plt.plot(load_length * dt, ev_e[-1], 'gx', ms=10, label='')
plt.plot(load_length * dt, ev_cp[-1], 'mx', ms=10, label='')
if deformation_mode == 'drained':
    plt.ylim(-0.15, 0.1)
elif deformation_mode == 'undrained':
    plt.ylim(-0.05, 0.05)
    plt.yticks(np.linspace(-0.05, 0.05, 5))
plt.xlabel(time_label)
plt.ylabel(ev_label)
plt.legend()
plt.savefig(f"{deformation_mode}_e_{void_ratio_0:.3f}_{OCR:.3f}.png", dpi=300, bbox_inches="tight")

# Gamma dot vs. mu
plt.figure(figsize=figsize)
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
    plt.text(eqp_inc_history[p_i] / dt + x_offset, 1/(1+void_ratio_total[p_i])  + y_offset, f"{i+1}", fontsize=14)
plt.xlabel('Shear rate')
plt.ylabel('Solid volume fraction (total)')
plt.savefig("gamma_phi_transient.png", dpi=300, bbox_inches="tight")

#%% Save all data to a .npz file
input_data = {
    'deformation_mode': np.array(deformation_mode),
    'p0': p0,
    'pc_0': pc_0,
    'OCR': OCR,
    'M': M,
    'lambda_val': lambda_val,
    'kappa': kappa,
    'Gamma': Gamma,
    'N': N,
    'nu': nu,
    'Delta_Phi': Delta_Phi,
    'M_c': M_c,
    'time': time,
    'dt': dt,
    'load_length': load_length,
    'eqp_tot': eqp_tot,
    'eqp_inc_history': eqp_inc_history,
    'void_ratio_0': void_ratio_0,
    'Phi_0': Phi_0,
}

output_data = {
    'p': p,
    'q': q,
    'p_total': p_total,
    'q_total': q_total,
    'p_c': p_c,
    'q_c': q_c,
    'u': u,
    'ev': ev,
    'eq': eq,
    'ev_e': ev_e,
    'ev_cp': ev_cp,
    'ev_qp': ev_qp,
    'void_ratio_q': void_ratio_q,
    'void_ratio_cp': void_ratio_cp,
    'void_ratio_total': void_ratio_total,
    'pc_history': pc_history,
    'd_p_c': d_p_c,
    'sigma': sigma,
    'sigma_q': sigma_q,
    'sigma_c': sigma_c,
    'epsilon': epsilon,
}

np.savez_compressed(
    f"{deformation_mode}_data_{void_ratio_0:.3f}_{OCR:.3f}.npz",
    **{f'input__{k}': v for k, v in input_data.items()},
    **{f'output__{k}': v for k, v in output_data.items()},
)