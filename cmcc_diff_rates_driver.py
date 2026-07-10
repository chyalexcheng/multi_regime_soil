import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from cmcc_core import (
    MaterialParams,
    check_deformation_mode,
    initialize,
    save_case_npz,
    split_input_output,
)
from cmcc_plotting import make_diff_rates_comparison_plots, make_diff_rates_summary_plot
from triax_solve import run_triaxial_path_with_servo


def make_diff_rates_history(
    accel_time: float,
    eqp_tot: float = 100.0,
    time: float = 100.0,
    base_load_length: int = int(1e5),
    total_time: float = 2.5,
) -> Dict[str, np.ndarray]:
    """Build a loading history that ramps the shear-strain rate up to 100x and back
    down over `accel_time` seconds each way, holding `total_time` seconds in between.
    Mirrors the per-accel_time loading history from backup/CMCC_evpc_diff_rates.py.
    """
    dt = time / base_load_length
    eqp_inc = eqp_tot / (1e2 * time / dt)

    eqp_inc_history = np.concatenate([
        np.ones(round(1.0 / eqp_inc)) * eqp_inc,
        np.linspace(eqp_inc, 100 * eqp_inc, round(accel_time / eqp_inc)),
        np.ones(round((total_time - 2 * accel_time) / eqp_inc)) * 100 * eqp_inc,
        np.linspace(100 * eqp_inc, eqp_inc, round(accel_time / eqp_inc)),
        np.ones(round(0.5 / eqp_inc)) * eqp_inc,
    ])
    point1 = round(1.0 / eqp_inc)
    point2 = point1 + round(accel_time / eqp_inc)
    point3 = point2 + round((total_time - 2 * accel_time) / eqp_inc)
    point4 = point3 + round(accel_time / eqp_inc)
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
        "accel_time": accel_time,
        "total_time": total_time,
    }


def run_diff_rates_study(
    deformation_mode: str,
    params: MaterialParams,
    accel_times: List[float],
    total_time: float = 2.5,
    eqp_tot: float = 100.0,
    time: float = 100.0,
    base_load_length: int = int(1e5),
    save_npz: bool = True,
    make_summary_plots: bool = True,
    make_comparison_plots: bool = True,
    output_dir: str = ".",
) -> Dict[str, list]:
    """Run the same triaxial servo-controlled path for several acceleration/deceleration
    durations (`accel_times`), reproducing backup/CMCC_evpc_diff_rates.py."""
    mode = check_deformation_mode(deformation_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = []

    for accel_time in accel_times:
        history = make_diff_rates_history(
            accel_time=accel_time,
            eqp_tot=eqp_tot,
            time=time,
            base_load_length=base_load_length,
            total_time=total_time,
        )

        data = initialize(history["load_length"], params)
        data = run_triaxial_path_with_servo(mode, params, history, data)

        input_data, output_data = split_input_output(data, params, mode, history)

        if save_npz:
            save_case_npz(
                output / f"{mode}_diffrate_data_{accel_time:.3f}_{data['OCR']:.3f}.npz",
                input_data,
                output_data,
            )

        if make_summary_plots:
            plot_payload = dict(output_data)
            plot_payload["M"] = float(params.M)
            make_diff_rates_summary_plot(plot_payload, history, output_dir=output_dir)

        results.append({"history": history, "output": output_data, "data": data})

    if make_comparison_plots:
        make_diff_rates_comparison_plots(
            results,
            mode=mode,
            void_ratio_0=float(results[0]["data"]["void_ratio_0"]),
            OCR=float(results[0]["data"]["OCR"]),
            M=float(params.M),
            lambda_val=float(params.lambda_val),
            Delta_Phi=float(params.Delta_Phi),
            output_dir=output_dir,
        )

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="drained", help="deformation mode: drained/undrained or 1/2")
    parser.add_argument("--ocr", type=float, default=1.0, help="OCR value, e.g., 1 or 3")
    parser.add_argument("--accel-times", type=float, nargs="+", default=[0.02, 0.1, 0.5],
                         help="acceleration/deceleration durations [s] to compare")
    parser.add_argument("--total-time", type=float, default=2.5, help="total loading duration [s]")
    parser.add_argument("--output-dir", default=".", help="directory to save npz files and figures")
    args = parser.parse_args()

    mode = check_deformation_mode(args.mode)

    eqp_tot: float = 100.0
    time: float = 100.0
    base_load_length: int = int(1e5)

    params = MaterialParams(pc_0=args.ocr * 150.0)
    run_diff_rates_study(
        mode,
        params,
        accel_times=args.accel_times,
        total_time=args.total_time,
        eqp_tot=eqp_tot,
        time=time,
        base_load_length=base_load_length,
        save_npz=True,
        make_summary_plots=True,
        make_comparison_plots=True,
        output_dir=args.output_dir,
    )
