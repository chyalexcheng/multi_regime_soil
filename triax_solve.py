from typing import Dict

import numpy as np

from cmcc_core import (
    MaterialParams,
    build_quasistatic_stiffness,
    check_deformation_mode,
    compute_increment,
)


def run_triaxial_path_with_servo(
    deformation_mode: str,
    params: MaterialParams,
    history: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run triaxial load path with servo control, updating `data` in-place."""
    mode = check_deformation_mode(deformation_mode)

    for i, eqp_inc in enumerate(history["eqp_inc_history"][:-1]):
        d_eqp_inc = eqp_inc - history["eqp_inc_history"][i - 1] if i > 0 else 0.0

        stiffness = build_quasistatic_stiffness(i, data, params)
        data["pc"] = stiffness["pc"]
        data["pc_history"][i + 1] = data["pc"]

        def servo_control(ratios):
            ratio_1, ratio_2 = ratios
            inc = compute_increment(
                i, float(ratio_1), float(ratio_2), mode,
                float(eqp_inc), float(d_eqp_inc), data, params, stiffness,
                float(history["dt"]),
            )
            d_sigma = inc["d_sigma_q"] + inc["d_sigma_c"]
            return np.sum(d_sigma[:3])

        def advance_constant_mean_pressure(eqp_inc, p_target,
                                            tolerance=1e-3,
                                            max_iterations=100,
                                            small_number=1e-14):
            # No acceleration: ratio_2 has no effect on the result, so keep it equal to ratio_1.
            D_q = stiffness["D_q"]
            denom = D_q[1, 1] + D_q[1, 2]
            delta_eps_r = - D_q[1, 0] / denom * eqp_inc if np.abs(denom) > small_number else 0.0

            for _ in range(max_iterations):
                ratio_loc = delta_eps_r / eqp_inc if np.abs(eqp_inc) > small_number else 0.0
                inc_loc = compute_increment(
                    i, ratio_loc, ratio_loc, mode,
                    float(eqp_inc), float(d_eqp_inc), data, params, stiffness,
                    float(history["dt"]),
                )

                sigma_trial = (data["sigma_q"] + inc_loc["d_sigma_q"]) + (data["sigma_c"] + inc_loc["d_sigma_c"])
                p_trial = np.sum(sigma_trial[:3]) / 3.0
                residual = p_trial - p_target

                if np.abs(residual) <= tolerance:
                    return ratio_loc, inc_loc

                fd_step = max(1e-8 * max(1.0, np.abs(delta_eps_r), np.abs(eqp_inc)), small_number)
                ratio_fd = (delta_eps_r + fd_step) / eqp_inc if np.abs(eqp_inc) > small_number else 0.0
                inc_fd = compute_increment(
                    i, ratio_fd, ratio_fd, mode,
                    float(eqp_inc), float(d_eqp_inc), data, params, stiffness,
                    float(history["dt"]),
                )
                sigma_trial_fd = (data["sigma_q"] + inc_fd["d_sigma_q"]) + (data["sigma_c"] + inc_fd["d_sigma_c"])
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
        if mode == 'undrained':
            inc = compute_increment(
                i, -0.5, -0.5, mode, float(eqp_inc), float(d_eqp_inc), data, params, stiffness,
                float(history["dt"]),
            )
        else:
            if d_eqp_inc == 0:
                # No acceleration: use the pressure-control Newton line-search
                ratio_1, inc = advance_constant_mean_pressure(
                    eqp_inc=float(eqp_inc),
                    p_target=params.p0,
                )
                ratio_2 = ratio_1
            else:
                D_q = stiffness["D_q"]
                ratio0 = -D_q[1, 0] / (D_q[1, 1] + D_q[1, 2])
                ratio_1, ratio_2 = solve_ratios_levenberg_marquardt(ratio0, ratio0)
                inc = compute_increment(
                    i, ratio_1, ratio_2, mode,
                    float(eqp_inc), float(d_eqp_inc), data, params, stiffness,
                    float(history["dt"]),
                )

        data["d_p_c"][i + 1] = inc["d_p_c_next"]
        data["K_c"] = inc["K_c_new"]
        data["G_c"] = inc["G_c_new"]

        data["sigma_q"] += inc["d_sigma_q"]
        data["sigma_c"] += inc["d_sigma_c"]
        data["sigma"] = data["sigma_q"] + data["sigma_c"]
        data["epsilon"] += inc["d_epsilon"]

        data["p_total"][i + 1] = np.sum(data["sigma"][:3]) / 3.0
        p_s = data["sigma"] - np.array([1., 1., 1., 0, 0, 0]) * data["p_total"][i + 1]
        data["q_total"][i + 1] = np.sqrt(3. / 2. * p_s.dot(p_s))

        data["p_c"][i + 1] = np.sum(data["sigma_c"][:3]) / 3.0
        p_s_c = data["sigma_c"] - np.array([1., 1., 1., 0, 0, 0]) * data["p_c"][i + 1]
        data["q_c"][i + 1] = np.sqrt(3. / 2. * p_s_c.dot(p_s_c))

        data["ev"][i + 1] = np.sum(data["epsilon"][:3])
        epsilon_s = data["epsilon"] - np.array([1, 1, 1, 0, 0, 0]) * data["ev"][i + 1]
        data["eq"][i + 1] = np.sqrt(2. / 3. * epsilon_s.dot(epsilon_s))
        data["ev_cp"][i + 1] = data["ev_cp"][i] + inc["de_v_c"]
        data["ev_qp"][i + 1] = data["ev_qp"][i] + inc["de_v_p"]

        data["p"][i + 1] = np.sum(data["sigma_q"][:3]) / 3.0
        p_q_s = data["sigma_q"] - np.array([1., 1., 1., 0, 0, 0]) * data["p"][i + 1]
        data["q"][i + 1] = np.sqrt(3. / 2. * p_q_s.dot(p_q_s))
        data["u"][i + 1] = params.p0 + data["q_total"][i + 1] / 3. - data["p_total"][i + 1]

        data["V"] = params.N - (params.lambda_val * np.log(data["pc"])) + (params.kappa * np.log(data["pc"] / data["p"][i + 1]))
        data["void_ratio_q"][i + 1] = data["V"] - 1
        data["void_ratio_cp"][i + 1] = data["void_ratio_cp"][i] - (1 + data["void_ratio_total"][i]) * inc["de_v_c"]
        data["void_ratio_total"][i + 1] = data["void_ratio_total"][i] - (1 + data["void_ratio_total"][i]) * inc["d_epsilon"][:3].sum()

        if data["yield_surf"] < 0:
            data["yield_surf"] = data["q"][i + 1] ** 2 + params.M ** 2 * data["p"][i + 1] ** 2 - params.M ** 2 * data["p"][i + 1] * data["pc"]
        else:
            data["yield_surf"] = 0

    data["ev_e"] = data["ev"] - data["ev_qp"] - data["ev_cp"]
    data["deformation_mode"] = mode
    data["M"] = params.M

    return data
