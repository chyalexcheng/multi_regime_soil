import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from cmcc_core import (
    MaterialParams,
    check_deformation_mode,
    initialize,
    save_case_npz,
    split_input_output,
)
from triax_solve import run_triaxial_path_with_servo
from cmcc_plotting import make_plots


def make_loading_history(eqp_tot: float = 100.0, time: float = 100.0, base_load_length: int = int(1e5)) -> Dict[str, np.ndarray]:
    # Define a loading history
    dt = time / base_load_length
    eqp_inc = eqp_tot / (1e2 * base_load_length)
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
    return {
        "eqp_tot": eqp_tot,
        "time": time,
        "dt": dt,
        "eqp_inc_history": eqp_inc_history,
        "load_length": eqp_inc_history.shape[0],
        "points": points,
        "markers": markers,
    }


def run_case(
    deformation_mode: str,
    params: MaterialParams,
    history: Dict[str, np.ndarray],
    save_npz: bool = True,
    make_all_plots: bool = True,
    output_dir: str = ".",
) -> Dict[str, Dict[str, np.ndarray]]:
    mode = check_deformation_mode(deformation_mode)

    data = initialize(history["load_length"], params)
    data = run_triaxial_path_with_servo(mode, params, history, data)

    history_out = dict(history)
    input_data, output_data = split_input_output(data, params, mode, history_out)

    if save_npz:
        save_case_npz(
            Path(output_dir) / f"{mode}_data_{data['void_ratio_0']:.3f}_{data['OCR']:.3f}.npz",
            input_data,
            output_data,
        )

    if make_all_plots:
        plot_payload = dict(output_data)
        plot_payload.update(
            {
                "deformation_mode": mode,
                "void_ratio_0": float(data["void_ratio_0"]),
                "OCR": float(data["OCR"]),
                "M": float(params.M),
                "lambda_val": float(params.lambda_val),
            }
        )
        make_plots(plot_payload, history_out, output_dir=output_dir)

    return {"input": input_data, "output": output_data, "history": history_out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="drained", help="deformation mode: drained/undrained or 1/2")
    parser.add_argument("--ocr", type=float, default=3.0, help="OCR value, e.g., 1 or 3")
    args = parser.parse_args()

    mode = check_deformation_mode(args.mode)

    eqp_tot: float = 100.0
    time: float = 100.0
    base_load_length: int = int(1e5)
    history = make_loading_history(eqp_tot=eqp_tot, time=time, base_load_length=base_load_length)

    ocr = args.ocr

    params = MaterialParams(pc_0=ocr * 150.0)
    run_case(mode, params, history, save_npz=True, make_all_plots=True, output_dir=".")
