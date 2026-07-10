from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def save_case_npz(path: Path, input_data: Dict, output_data: Dict) -> None:
    """Save input/output dicts to a compressed .npz file with input__/output__ prefixes.
    Shared between cmcc_driver.py and cmcc_diff_rates_driver.py to avoid duplication."""
    np.savez_compressed(
        path,
        **{f"input__{k}": v for k, v in input_data.items()},
        **{f"output__{k}": v for k, v in output_data.items()},
    )


@dataclass
class MaterialParams:
    pc_0: float
    M: float = 1.0
    lambda_val: float = 0.2
    kappa: float = 0.04
    Gamma: float = 1.39
    N: float = 2.5
    nu: float = 0.15
    p0: float = 150.0
    Delta_Phi: float = 0.025
    M_c: float = 1.5


def check_deformation_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"1", "drained"}:
        return "drained"
    if m in {"2", "undrained"}:
        return "undrained"
    raise ValueError("mode must be drained/undrained or 1/2")


def get_deviatoric_strain(tensor: np.ndarray) -> Tuple[float, np.ndarray]:
    eps_v = np.sum(tensor[:3])
    eps_dev = tensor - np.array([eps_v / 3.0, eps_v / 3.0, eps_v / 3.0, 0.0, 0.0, 0.0])
    dev_contract = (
        eps_dev[0] ** 2
        + eps_dev[1] ** 2
        + eps_dev[2] ** 2
        + 2.0 * (eps_dev[3] ** 2 + eps_dev[4] ** 2 + eps_dev[5] ** 2)
    )
    eps_q = np.sqrt(2.0 / 3.0 * dev_contract)
    direction = eps_dev / eps_q if eps_q > 0.0 else np.zeros_like(tensor)
    return float(eps_q), direction


def initialize(load_length: int, params: MaterialParams) -> Dict[str, np.ndarray]:
    V = params.N - (params.lambda_val * np.log(params.pc_0)) + (params.kappa * np.log(params.pc_0 / params.p0))
    void_ratio_0 = V - 1

    data = {
        "void_ratio_total": np.zeros(load_length),
        "void_ratio_q": np.zeros(load_length),
        "void_ratio_cp": np.zeros(load_length),
        "p": np.zeros(load_length),
        "q": np.zeros(load_length),
        "u": np.zeros(load_length),
        "ev": np.zeros(load_length),
        "ev_e": np.zeros(load_length),
        "ev_cp": np.zeros(load_length),
        "ev_qp": np.zeros(load_length),
        "eq": np.zeros(load_length),
        "d_p_c": np.zeros(load_length),
        "p_c": np.zeros(load_length),
        "q_c": np.zeros(load_length),
        "p_total": np.zeros(load_length),
        "q_total": np.zeros(load_length),
        "pc_history": np.zeros(load_length),
        "sigma": np.array([params.p0, params.p0, params.p0, 0.0, 0.0, 0.0]),
        "sigma_q": np.array([params.p0, params.p0, params.p0, 0.0, 0.0, 0.0]),
        "sigma_c": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "epsilon": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "V": V,
        "void_ratio_0": void_ratio_0,
        "Phi_0": 1.0 / (1.0 + void_ratio_0),
        "OCR": params.pc_0 / params.p0,
        "K_c": 0.0,
        "G_c": 0.0,
        "pc": params.pc_0,
        "yield_surf": 0.0,
    }
    data["p"][0] = params.p0
    data["p_total"][0] = params.p0
    data["void_ratio_total"][0] = void_ratio_0
    data["void_ratio_q"][0] = void_ratio_0
    data["void_ratio_cp"][0] = void_ratio_0
    data["yield_surf"] = (data["q"][0] ** 2 / params.M ** 2 + data["p"][0] ** 2) - data["p"][0] * data["pc"]
    return data


def build_quasistatic_stiffness(i: int, data: Dict[str, np.ndarray], params: MaterialParams) -> Dict[str, np.ndarray]:
    """Quasi-static core block from original lines ~141-188."""
    # Stiffness matrix and strain increment vector and derivatives of the yield function
    De = np.zeros([6, 6])
    # TODO: HC: check if the stiffness matrix for dynamic stress increments can be formulated like elasticity
    D_c = np.zeros([6, 6])
    De_c = np.zeros([6, 6])
    df_ds = np.zeros([6, 1])
    df_dep = np.zeros([6, 1])

    # Calculate the bulk and shear modulus
    K = data["V"] * data["p"][i] / params.kappa
    G = (3 * K * (1 - 2 * params.nu)) / (2 * (1 + params.nu))

    # Update pre-consolidation pressure
    if data["yield_surf"] == 0:
        # pc = pc_history[i] * (1.0 + V * de_v_p / (lambda_val - kappa))
        pc = (data["q"][i] ** 2 / params.M ** 2 + data["p"][i] ** 2) / data["p"][i]
    else:
        pc = params.pc_0

    # Elastic Stiffness and other Matrix
    for m in range(6):
        for n in range(6):
            if m <= 2:
                if data["yield_surf"] < 0:
                    df_ds[m, 0] = 0
                    df_dep[m, 0] = 0
                else:
                    df_ds[m, 0] = (2 * data["p"][i] - pc) / 3 + 3 * (data["sigma_q"][m] - data["p"][i]) / params.M ** 2
                    df_dep[m, 0] = (-data["p"][i]) * pc * (1 + data["void_ratio_q"][i]) / (params.lambda_val - params.kappa)
                if m == n:
                    De[m, n] = K + 4 / 3 * G
                    De_c[m, n] = data["K_c"]
                    D_c[m, n] = 4 / 3 * data["G_c"]
                elif n <= 2:
                    De[m, n] = K - 2 / 3 * G
                    De_c[m, n] = data["K_c"]
                    D_c[m, n] = -2 / 3 * data["G_c"]
            if m > 2:
                df_ds[m, 0] = 0
                df_dep[m, 0] = 0
                if m == n:
                    De[m, n] = G
                    D_c[m, n] = data["G_c"]
                else:
                    De[m, n] = 0
                    D_c[m, n] = 0

        # If the yield surface is negative, the stiffness matrix is elastic
        if data["yield_surf"] < 0:
            D_q = De
        else:
            # If the yield surface is positive, the stiffness matrix is elastic-plastic
            D_q = De - (De.dot(df_ds).dot(df_ds.T).dot(De)) / (
                -(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds)
            )

    return {
        "De": De,
        "D_c": D_c,
        "De_c": De_c,
        "df_ds": df_ds,
        "df_dep": df_dep,
        "D_q": D_q,
        "pc": pc,
    }


def compute_increment(
    i: int,
    ratio_1: float,
    ratio_2: float,
    deformation_mode: str,
    eqp_inc: float,
    d_eqp_inc: float,
    data: Dict[str, np.ndarray],
    params: MaterialParams,
    stiffness: Dict[str, np.ndarray],
    dt: float,
) -> Dict[str, np.ndarray]:
    """Collisional + quasi-static core block from original lines ~209-249."""
    mode = check_deformation_mode(deformation_mode)

    # Fill the strain increment vector
    if mode == 'drained':
        # For drained (or pressure control) conditions
        d_epsilon = np.array([eqp_inc, ratio_1 * eqp_inc, ratio_1 * eqp_inc, 0., 0., 0.])
        if d_eqp_inc != 0:
            d_d_epsilon = np.array([d_eqp_inc, ratio_2 * d_eqp_inc, ratio_2 * d_eqp_inc, 0., 0., 0.])
        else:
            d_d_epsilon = np.array([0., 0., 0., 0., 0., 0.])
    else:
        d_epsilon = np.array([eqp_inc, -eqp_inc / 2., -eqp_inc / 2., 0., 0., 0.])
        d_d_epsilon = np.array([d_eqp_inc, -d_eqp_inc / 2., -d_eqp_inc / 2., 0., 0., 0.])

    # Magnitude of the increment in deviatoric shear strain rate (from tensor),
    d_eqp_inc_mag, _ = get_deviatoric_strain(d_d_epsilon)
    # Keep acceleration/deceleration sign convention from loading-history increment.
    d_eqp_inc_eff = np.sign(d_eqp_inc) * d_eqp_inc_mag

    # Calculate dynamic stress
    if d_eqp_inc_mag != 0:
        # Get the deviatoric part of the acceleration rate tensor ?
        Phi = 1.0 / (1.0 + data["void_ratio_total"][i])
        Phi_c = 1.0 / (1.0 + data["void_ratio_cp"][i])
        de_v_c = - Phi / Phi_c ** 2 * params.Delta_Phi * d_eqp_inc_eff / dt
        # d_p_c[i + 1] = p[i] / ((lambda_val - kappa) * Phi ** 2) * Delta_Phi * d_eqp_inc_eff / dt
        d_p_c_next = data["p"][i] / ((params.lambda_val - 0) * Phi ** 2) * params.Delta_Phi * d_eqp_inc_eff / dt
        K_c_new = d_p_c_next / de_v_c
        G_c_new = d_p_c_next * params.M_c / d_eqp_inc_eff / 3
    else:
        de_v_c = 0
        d_p_c_next = 0.0
        K_c_new = 0
        G_c_new = 0

    # Get dynamic stress-induced plastic strain increment
    d_epsilon_v_c = 1. / 3. * de_v_c * np.array([1., 1., 1., 0, 0, 0])

    # Get quasi-static stress induced plastic volumetric strain increment
    if data["yield_surf"] < 0:
        de_v_p = 0
    else:
        # Compute the plastic multiplier increment
        num = (stiffness["df_ds"].T).dot(stiffness["De"]).dot(d_epsilon - d_epsilon_v_c)
        den = -(stiffness["df_dep"].T).dot(stiffness["df_ds"]) + (stiffness["df_ds"].T).dot(stiffness["De"]).dot(stiffness["df_ds"])
        # if size of num and den are not 1, through an error
        if num.shape != (1,) or den.shape != (1, 1):
            raise ValueError(f"num and den should be scalars, but got num: {num}, den: {den}")
        # cast num and den to float to avoid the case where num and den are both zero, which causes dLambda to be an array instead of a scalar
        dLambda = float(num[0] / den[0, 0])
        # Get the plastic volumetric strain increment due to quasi-static stress
        d_epsilon_p = dLambda * stiffness["df_ds"]
        de_v_p = float(d_epsilon_p[0, 0] + d_epsilon_p[1, 0] + d_epsilon_p[2, 0])

    # Update stress increments
    d_sigma_q = stiffness["D_q"].dot(d_epsilon - d_epsilon_v_c)
    d_sigma_c = stiffness["De_c"].dot(d_epsilon_v_c) + stiffness["D_c"].dot(
        d_d_epsilon - 1. / 3. * d_d_epsilon[:3].sum() * np.array([1., 1., 1., 0, 0, 0])
    )

    return {
        "d_sigma_q": d_sigma_q,
        "d_sigma_c": d_sigma_c,
        "d_epsilon": d_epsilon,
        "de_v_p": de_v_p,
        "de_v_c": de_v_c,
        "d_p_c_next": d_p_c_next,
        "K_c_new": K_c_new,
        "G_c_new": G_c_new,
    }


def split_input_output(data: Dict[str, np.ndarray], params: MaterialParams, mode: str, history: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    input_data = {
        "deformation_mode": np.array(check_deformation_mode(mode)),
        "p0": params.p0,
        "pc_0": params.pc_0,
        "OCR": data["OCR"],
        "M": params.M,
        "lambda_val": params.lambda_val,
        "kappa": params.kappa,
        "Gamma": params.Gamma,
        "N": params.N,
        "nu": params.nu,
        "Delta_Phi": params.Delta_Phi,
        "M_c": params.M_c,
        "time": history["time"],
        "dt": history["dt"],
        "load_length": history["load_length"],
        "eqp_tot": history["eqp_tot"],
        "eqp_inc_history": history["eqp_inc_history"],
        "void_ratio_0": data["void_ratio_0"],
        "Phi_0": data["Phi_0"],
    }
    output_data = {
        "p": data["p"],
        "q": data["q"],
        "p_total": data["p_total"],
        "q_total": data["q_total"],
        "p_c": data["p_c"],
        "q_c": data["q_c"],
        "u": data["u"],
        "ev": data["ev"],
        "eq": data["eq"],
        "ev_e": data["ev_e"],
        "ev_cp": data["ev_cp"],
        "ev_qp": data["ev_qp"],
        "void_ratio_q": data["void_ratio_q"],
        "void_ratio_cp": data["void_ratio_cp"],
        "void_ratio_total": data["void_ratio_total"],
        "pc_history": data["pc_history"],
        "d_p_c": data["d_p_c"],
        "sigma": data["sigma"],
        "sigma_q": data["sigma_q"],
        "sigma_c": data["sigma_c"],
        "epsilon": data["epsilon"],
    }
    return input_data, output_data
