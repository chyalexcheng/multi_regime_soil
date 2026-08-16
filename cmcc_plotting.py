import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})


def lighten(color, amount):
    """Lighten a color by mixing it with white. amount=0 -> original, amount=1 -> white."""
    try:
        c = np.array(mcolors.to_rgb(color))
    except ValueError:
        c = np.array(mcolors.to_rgb(mcolors.CSS4_COLORS[color]))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


def segmented_plot(ax, x, y, points, **kwargs):
    idx = points
    for i in range(len(idx) - 1):
        i0, i1 = idx[i], idx[i + 1]
        if i1 > i0:
            ax.plot(x[i0:i1], y[i0:i1], **kwargs)


def expand_ylim_for_annotations(ax, top_frac=0.10, bottom_frac=0.03):
    y_min_cur, y_max_cur = ax.get_ylim()
    y_diff = np.abs(y_max_cur - y_min_cur)
    if y_diff == 0:
        y_diff = 1.0
    ax.set_ylim(y_min_cur - bottom_frac * y_diff, y_max_cur + top_frac * y_diff)


def _to_list(values):
    return [] if values is None else list(values)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.abs(den) > 1e-14
    out[mask] = num[mask] / den[mask]
    return out


def _collect_context(data: dict, history: dict) -> dict:
    c = dict(data)
    c.update(history)

    c["mode"] = str(data["deformation_mode"])
    c["e0"] = float(data["void_ratio_0"])
    c["ocr"] = float(data["OCR"])
    c["M"] = float(data["M"])
    c["lambda_val"] = float(data.get("lambda_val", 0.2))

    c["points"] = _to_list(history.get("points"))
    c["markers"] = _to_list(history.get("markers"))
    c["t"] = np.arange(c["load_length"]) * c["dt"]

    # Full text long labels
    c["time_label"] = r'Time ($t$) [s]'
    c["mu_label"] = r'Stress ratio $q/p$ [-]'
    c["pressure_label"] = r'Pressure $p$ [kPa]'
    c["dev_stress_label"] = r'Deviatoric stress $q$ [kPa]'
    c["ev_label"] = r'Volumetric strain $\varepsilon_v$ [-]'
    c["preconsolidation_p_label"] = 'Pre-consolidation pressure [kPa]'
    c["p_ratio_label"] = r'Ratio of dynamic and quasi-static stresses [-]'
    c["e_label"] = r'Void ratio $e$ [-]'

    c["figsize"] = (6, 4.5)
    c["text_box_props"] = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='gray')
    return c


def animation_frame_indices(load_length: int, frame_step: int = 100) -> List[int]:
    """Return regularly spaced animation indices, always including the last step."""
    if load_length <= 0:
        raise ValueError("load_length must be positive")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    indices = list(range(0, load_length, frame_step))
    final_index = load_length - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


def make_animation_frame(
    data: Dict[str, np.ndarray],
    history: Dict[str, np.ndarray],
    step: int,
    output_path: Union[str, Path],
    dpi: int = 100,
) -> Path:
    """Plot the CMCC response through ``step`` as one fixed-size PNG frame.

    The complete histories are drawn in light gray to keep the axes identical
    between frames. Colored curves show the response reached so far.
    """
    required_data = (
        "p",
        "q",
        "p_total",
        "q_total",
        "void_ratio_q",
        "void_ratio_total",
        "pc_history",
    )
    missing = [key for key in required_data if key not in data]
    if missing:
        raise KeyError(f"Missing animation data: {', '.join(missing)}")

    load_length = int(history["load_length"])
    if not 0 <= step < load_length:
        raise IndexError(f"step must be between 0 and {load_length - 1}, got {step}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    dt = float(history["dt"])
    rate = np.asarray(history["eqp_inc_history"]) / dt
    if len(rate) != load_length:
        raise ValueError("eqp_inc_history length does not match load_length")

    arrays = {key: np.asarray(data[key]) for key in required_data}
    mismatched = [key for key, value in arrays.items() if len(value) != load_length]
    if mismatched:
        raise ValueError(f"Data length does not match load_length: {', '.join(mismatched)}")

    p = arrays["p"]
    q = arrays["q"]
    p_total = arrays["p_total"]
    q_total = arrays["q_total"]
    void_ratio_q = arrays["void_ratio_q"]
    void_ratio_total = arrays["void_ratio_total"]
    pc_history = arrays["pc_history"]
    M = float(data["M"])
    if M <= 0:
        raise ValueError("M must be positive")
    mu_q = _safe_ratio(q, p)
    mu_total = _safe_ratio(q_total, p_total)
    t = np.arange(load_length) * dt
    sampled_indices = np.linspace(0, load_length - 1, min(load_length, 2000), dtype=int)
    extrema_indices = []
    for values in (
        rate,
        p,
        q,
        p_total,
        q_total,
        void_ratio_q,
        void_ratio_total,
        pc_history,
        mu_q,
        mu_total,
    ):
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            extrema_indices.extend((finite[np.argmin(values[finite])], finite[np.argmax(values[finite])]))
    reference_indices = np.unique(np.append(sampled_indices, extrema_indices))
    progress_indices = reference_indices[reference_indices <= step]
    if progress_indices[-1] != step:
        progress_indices = np.append(progress_indices, step)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    ax_rate, ax_mu, ax_pq, ax_ep = axes.flat
    reference_color = "0.82"

    ax_rate.plot(t[reference_indices], rate[reference_indices], color=reference_color, linewidth=1.0)
    ax_rate.plot(t[progress_indices], rate[progress_indices], color="black", linewidth=1.5)
    ax_rate.plot(t[step], rate[step], "ko", ms=5)
    time_label = r"Time $t$ [s]"
    ax_rate.set_xlabel(time_label)
    ax_rate.set_ylabel(r"Shear strain rate $\dot{\gamma}$ [1/s]")
    ax_rate.set_xlim(t[0], t[-1] if load_length > 1 else t[0] + dt)

    ax_mu.plot(t[reference_indices], mu_total[reference_indices], color=reference_color, linewidth=1.0)
    ax_mu.plot(t[reference_indices], mu_q[reference_indices], color=reference_color, linewidth=1.0, linestyle="--")
    ax_mu.axhline(1.0, color='red', linestyle='--', label=r'$\mu^{\mathrm{cs}}$')
    ax_mu.axhline(1.5, color='red', linestyle=':', label=r'$\mu^{\mathrm{c}}$')
    ax_mu.plot(t[progress_indices], mu_total[progress_indices], color="green", label="Total")
    ax_mu.plot(t[progress_indices], mu_q[progress_indices], color="blue", label="Quasi-static")
    ax_mu.plot(t[step], mu_total[step], "go", ms=5)
    ax_mu.plot(t[step], mu_q[step], "bo", ms=5)
    ax_mu.set_xlabel(time_label)
    ax_mu.set_ylabel(r"Stress ratio $q/p$ [-]")
    ax_mu.set_xlim(t[0], t[-1] if load_length > 1 else t[0] + dt)
    ax_mu.legend(loc="best", fontsize="small")

    ax_pq.plot(p_total[reference_indices], q_total[reference_indices], color=reference_color, linewidth=1.0)
    ax_pq.plot(p[reference_indices], q[reference_indices], color=reference_color, linewidth=1.0, linestyle="--")
    ax_pq.plot(p_total[progress_indices], q_total[progress_indices], color="green", label="Total")
    ax_pq.plot(p[progress_indices], q[progress_indices], color="blue", label="Quasi-static")
    ax_pq.plot(p_total[step], q_total[step], "go", ms=5)
    ax_pq.plot(p[step], q[step], "bo", ms=5)

    initial_pc = float(data.get("pc_0", float(data.get("OCR", 1.0)) * p[0]))
    current_pc = float(pc_history[step])
    if not np.isfinite(current_pc) or current_pc <= 0:
        current_pc = initial_pc
    yield_p = np.linspace(0.0, current_pc, 300)
    yield_q = M * np.sqrt(np.maximum(yield_p * (current_pc - yield_p), 0.0))

    finite_pc = pc_history[np.isfinite(pc_history) & (pc_history > 0)]
    maximum_pc = max(initial_pc, float(np.max(finite_pc)) if finite_pc.size else initial_pc)
    pressure_values = np.concatenate((p[np.isfinite(p)], p_total[np.isfinite(p_total)]))
    stress_values = np.concatenate((q[np.isfinite(q)], q_total[np.isfinite(q_total)]))
    pressure_min = min(0.0, float(np.min(pressure_values)))
    pressure_max = max(float(np.max(pressure_values)), maximum_pc)
    stress_min = min(0.0, float(np.min(stress_values)))
    stress_max = max(float(np.max(stress_values)), M * maximum_pc / 2.0)
    pressure_margin = max(0.05 * (pressure_max - pressure_min), 1.0)
    stress_margin = max(0.05 * (stress_max - stress_min), 1.0)

    csl_p = np.array([0.0, pressure_max + pressure_margin])
    ax_pq.plot(
        csl_p,
        M * csl_p,
        color='red',
        linewidth=1.5,
        label="Critical state line",
        zorder=0,
    )
    ax_pq.plot(
        yield_p,
        yield_q,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="MCC yield surface",
        zorder=0,
    )
    ax_pq.set_xlabel(r"Pressure $p$ [kPa]")
    ax_pq.set_ylabel(r"Deviatoric stress $q$ [kPa]")
    ax_pq.set_xlim(pressure_min - pressure_margin, pressure_max + pressure_margin)
    ax_pq.set_ylim(stress_min - stress_margin, stress_max + stress_margin)
    ax_pq.legend(loc="best", fontsize="small")

    ax_ep.plot(p_total[reference_indices], void_ratio_total[reference_indices], color=reference_color, linewidth=1.0)
    ax_ep.plot(p[reference_indices], void_ratio_q[reference_indices], color=reference_color, linewidth=1.0, linestyle="--")
    ax_ep.plot(p_total[progress_indices], void_ratio_total[progress_indices], color="green", label="Total")
    ax_ep.plot(p[progress_indices], void_ratio_q[progress_indices], color="blue", label="Quasi-static")
    ax_ep.plot(p_total[step], void_ratio_total[step], "go", ms=5)
    ax_ep.plot(p[step], void_ratio_q[step], "bo", ms=5)
    data_xlim = ax_ep.get_xlim()
    data_ylim = ax_ep.get_ylim()

    positive_pressure_min = data_xlim[0]
    pCSL = np.linspace(positive_pressure_min, data_xlim[1], 300)
    lambda_val = float(data.get("lambda_val", 0.2))
    gamma = void_ratio_q[-1] + lambda_val * np.log(p[-1])
    eCSL = gamma - lambda_val * np.log(pCSL)
    ax_ep.plot(pCSL, eCSL, color="red", label="Critical state line", zorder=0)

    ax_ep.set_xlabel(r"Pressure $p$ [kPa]")
    ax_ep.set_ylabel(r"Void ratio $e$ [-]")
    ax_ep.set_xlim(data_xlim)
    ax_ep.set_ylim(data_ylim)
    ax_ep.legend(loc="best", fontsize="small")

    mode = "Pressure-conserved shear" if data.get("deformation_mode") == "drained" else "Volume-conserved shear"
    ocr = data.get("OCR")
    ocr_text = f", OCR={float(ocr):g}" if ocr is not None else ""
    fig.suptitle(f"{mode.capitalize()}{ocr_text} (t={t[step]:.1f} s)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def make_animation_frames(
    data: Dict[str, np.ndarray],
    history: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    frame_step: int = 100,
    dpi: int = 100,
    progress: Optional[Callable[[int, int, Path], None]] = None,
) -> List[Path]:
    """Render numbered PNG frames at every ``frame_step`` simulation steps."""
    indices = animation_frame_indices(int(history["load_length"]), frame_step)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    width = max(6, len(str(int(history["load_length"]) - 1)))

    frames = []
    for frame_number, step in enumerate(indices, start=1):
        frame_path = output / f"frame_{step:0{width}d}.png"
        frames.append(make_animation_frame(data, history, step, frame_path, dpi=dpi))
        if progress is not None:
            progress(frame_number, len(indices), frame_path)
    return frames


def _natural_path_key(path: Path) -> List[Union[int, str]]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_png_frames(directory: Union[str, Path], pattern: str = "*.png") -> List[Path]:
    """Find PNG files in natural filename order (frame_2 before frame_10)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"PNG frame directory does not exist: {directory}")
    frames = sorted(directory.glob(pattern), key=_natural_path_key)
    if not frames:
        raise ValueError(f"No PNG files matching {pattern!r} in {directory}")
    return frames


def create_gif_from_pngs(
    png_files: Union[str, Path, Sequence[Union[str, Path]], Iterable[Union[str, Path]]],
    output_path: Union[str, Path],
    duration_ms: int = 100,
    loop: int = 0,
    pattern: str = "*.png",
) -> Path:
    """Compile PNG files into a GIF, padding mixed image sizes with white.

    ``png_files`` may be a directory or an iterable of explicit paths. A
    directory is searched non-recursively using ``pattern``.
    """
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if loop < 0:
        raise ValueError("loop must be non-negative")

    if isinstance(png_files, (str, Path)):
        source = Path(png_files)
        paths = find_png_frames(source, pattern) if source.is_dir() else [source]
    else:
        paths = [Path(path) for path in png_files]
    if not paths:
        raise ValueError("At least one PNG frame is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"PNG frame does not exist: {missing[0]}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - matplotlib normally installs Pillow
        raise ImportError("Pillow is required to create GIF files") from exc

    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    canvas_size = (max(size[0] for size in sizes), max(size[1] for size in sizes))

    def load_gif_frame(path: Path):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", canvas_size, "white")
            offset = ((canvas_size[0] - rgba.width) // 2, (canvas_size[1] - rgba.height) // 2)
            canvas.alpha_composite(rgba, offset)
            adaptive = getattr(Image, "Palette", Image).ADAPTIVE
            return canvas.convert("RGB").convert("P", palette=adaptive)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import GifImagePlugin

    # Write one frame at a time. Pillow's high-level save_all API retains every
    # image in memory, which is impractical for long CMCC simulations.
    with output.open("wb") as gif_file:
        for frame_number, path in enumerate(paths):
            frame = load_gif_frame(path)
            try:
                if frame_number == 0:
                    header, _ = GifImagePlugin.getheader(frame, info={"loop": loop})
                    for block in header:
                        gif_file.write(block)
                params = {"duration": duration_ms, "disposal": 2}
                if frame_number > 0:
                    params["include_color_table"] = True
                for block in GifImagePlugin.getdata(frame, **params):
                    gif_file.write(block)
            finally:
                frame.close()
        gif_file.write(b";")
    return output


def _plot_load_history(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    eqp_inc_history = c["eqp_inc_history"]
    points = c["points"]
    markers = c["markers"]

    plt.figure(figsize=c["figsize"])
    plt.plot([-10, 360], [0, 0], '--k')
    plt.plot(t, eqp_inc_history / dt, '-k', label=rf'$|\ddot{{\varepsilon}}_{{zz}}| = 0.02$ s' + r'$^{-2}$')
    x_offset = c["load_length"] * dt * 0.03
    y_offset = max(eqp_inc_history) / dt * 0.03
    for i, (p_i, m_i) in enumerate(zip(points, markers)):
        if p_i >= len(eqp_inc_history):
            continue
        plt.plot(p_i * dt, eqp_inc_history[p_i] / dt, f'k{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, eqp_inc_history[p_i] / dt + y_offset, f"{i+1}", fontsize=14)
    plt.plot(0 * dt, eqp_inc_history[0] / dt, 'k.', ms=10, label='')
    plt.plot(c["load_length"] * dt, eqp_inc_history[-1] / dt, 'kx', ms=10, label='')
    plt.xlabel(c["time_label"])
    plt.ylabel(r'Axial strain rate $\dot{\varepsilon_{zz}}$ [1/s]')
    ymin, ymax = plt.ylim()
    plt.xlim(-10, 360)
    plt.ylim(ymin, ymax + np.ceil(y_offset * 10) / 10)
    plt.legend()
    plt.savefig(output / "load_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_pressure_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    p = c["p"]
    p_total = c["p_total"]

    plt.figure(figsize=c["figsize"])
    plt.plot(t, p_total, '-g', label=r"Total pressure ($p$)")
    plt.plot(t, p, '-b', label=r"Quasi-static pressure ($p^{\mathrm{q}}$)")
    plt.plot(0 * dt, p[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, p_total[0], 'g.', ms=10, label='')
    plt.plot(c["load_length"] * dt, p[-1], 'bx', ms=10, label='')
    plt.plot(c["load_length"] * dt, p_total[-1], 'gx', ms=10, label='')
    x_offset = c["load_length"] * dt * 0.035
    y_offset_p = max(p) * 0.01
    y_offset_p_total = max(p_total) * 0.01
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(p):
            continue
        plt.plot(p_i * dt, p[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_i * dt, p_total[p_i], f'g{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, p[p_i] - y_offset_p, f"{i+1}", fontsize=14, color='blue', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, p_total[p_i] - y_offset_p_total, f"{i+1}", fontsize=14, color='green', bbox=c["text_box_props"])
    plt.xlabel(c["time_label"])
    plt.ylabel(c["pressure_label"])
    if c["mode"] == 'undrained' and c["ocr"] == 3.0:
        plt.ylim(-10, 360)
    elif c["mode"] == 'undrained' and c["ocr"] == 1.0:
        plt.ylim(-5, 165)
    plt.legend()
    plt.savefig(output / f"{c['mode']}_p_and_p_c_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_q_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    q = c["q"]
    q_total = c["q_total"]
    p_total = c["p_total"]

    plt.figure(figsize=c["figsize"])
    plt.plot(t, q_total, '-g', label=r"Total deviatoric stress ($q$)")
    plt.plot(t, q, '-b', label=r"Quasi-static deviatoric stress ($q^{\mathrm{q}}$)")
    plt.plot(0 * dt, q[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, q_total[0], 'g.', ms=10, label='')
    plt.plot(c["load_length"] * dt, q[-1], 'bx', ms=10, label='')
    plt.plot(c["load_length"] * dt, q_total[-1], 'gx', ms=10, label='')
    x_offset = c["load_length"] * dt * 0.035
    y_offset_q = max(q_total) * 0.01
    y_offset_p_total = max(p_total) * 0.01
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(q):
            continue
        plt.plot(p_i * dt, q[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_i * dt, q_total[p_i], f'g{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, q[p_i] - y_offset_q, f"{i+1}", fontsize=12, color='blue', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, q_total[p_i] - y_offset_p_total, f"{i+1}", fontsize=12, color='green', bbox=c["text_box_props"])
    plt.xlabel(c["time_label"])
    plt.ylabel(c["dev_stress_label"])
    if c["mode"] == 'drained':
        plt.ylim(-7, 220)
    elif c["mode"] == 'undrained' and c["ocr"] == 3.0:
        plt.ylim(-10, 360)
    elif c["mode"] == 'undrained' and c["ocr"] == 1.0:
        plt.ylim(-5, 165)
    plt.legend(loc='lower center')
    plt.savefig(output / f"{c['mode']}_q_and_q_c_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_ev_gamma_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    eq = c["eq"]
    ev = c["ev"]

    plt.figure(figsize=c["figsize"])
    ax1 = plt.gca()
    ax1.plot(t, eq, '-k', ms=10, label='')
    ax1.plot(0 * dt, eq[0], 'k.', ms=10, label='')
    ax1.plot(c["load_length"] * dt, eq[-1], 'kx', ms=10, label='')

    x_offset = c["load_length"] * dt * 0.035
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(eq):
            continue
        plt.plot(p_i * dt, eq[p_i], f'k{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, eq[p_i], f"{i+1}", fontsize=14, color='gray', bbox=c["text_box_props"])
    ax1.set_xlabel(c["time_label"])
    ax1.set_ylim(0, 160)
    ax1.set_yticks(np.linspace(0, 160, 6))
    ax1.set_ylabel(r'Deviatoric strain ($\gamma$)', color='black')
    ax1.tick_params(axis='y', labelcolor='k')

    ax2 = ax1.twinx()
    ax2.plot(t, ev, '-', color='gray')
    ax2.plot(0 * dt, ev[0], '.', ms=10, label='', color='gray')
    ax2.plot(c["load_length"] * dt, ev[-1], 'x', ms=10, label='', color='gray')
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(ev):
            continue
        plt.plot(p_i * dt, ev[p_i], f'{m_i}', ms=10, mfc='none', color='gray')
        plt.text(p_i * dt + x_offset, ev[p_i] - 0.02, f"{i+1}", fontsize=14, color='gray', bbox=c["text_box_props"])
    ax2.set_ylabel(r"Volumetric strain ($\varepsilon_v$)", color='gray')
    ax2.set_ylim(bottom=-0.15, top=0.1)
    ax2.tick_params(axis='y', labelcolor='gray')
    plt.savefig(output / f"{c['mode']}_e_v_and_gamma_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_pc_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    pc_history = c["pc_history"]

    plt.figure(figsize=c["figsize"])
    plt.plot(t, pc_history, '-b')
    plt.plot(0 * dt, pc_history[0], 'b.', ms=10, label='')
    plt.plot(c["load_length"] * dt, pc_history[-1], 'bx', ms=10, label='')
    x_offset = c["load_length"] * dt * 0.035
    y_offset = max(pc_history) * 0.01
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(pc_history):
            continue
        plt.plot(p_i * dt, pc_history[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, pc_history[p_i] + y_offset, f"{i+1}", fontsize=14, color='blue', bbox=c["text_box_props"])
    plt.xlabel(c["time_label"])
    plt.ylabel(c["preconsolidation_p_label"])
    plt.savefig(output / f"{c['mode']}_preconsolidate_p_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_dynamic_ratio(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    p = c["p"]
    q = c["q"]
    p_c = c["p_c"]
    q_c = c["q_c"]

    p_ratio = _safe_ratio(p_c[1:], p[1:])
    q_ratio = _safe_ratio(q_c[1:], q[1:])

    plt.figure(figsize=c["figsize"])
    plt.plot(np.arange(c["load_length"] - 1) * dt, p_ratio, '-c', label=r'Pressures ($p^{\mathrm{c}}/p^{\mathrm{q}}$)')
    plt.plot(0 * dt, p_ratio[0], 'c.', ms=10, label='')
    plt.plot(c["load_length"] * dt, p_ratio[-1], 'cx', ms=10, label='')
    plt.plot(np.arange(c["load_length"] - 1) * dt, q_ratio, '-.c', label=r'Deviatoric stresses ($q^{\mathrm{c}}/q^{\mathrm{q}}$)')
    plt.plot(0 * dt, q_ratio[0], 'c.', ms=10, label='')
    plt.plot(c["load_length"] * dt, q_ratio[-1], 'cx', ms=10, label='')

    x_offset = c["load_length"] * dt * 0.035
    y_offset_p = np.nanmax(p_ratio) * 0.01 if np.any(np.isfinite(p_ratio)) else 0.0
    y_offset_q = np.nanmax(q_ratio) * 0.01 if np.any(np.isfinite(q_ratio)) else 0.0
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i <= 0 or p_i >= len(p):
            continue
        y1 = p_c[p_i] / p[p_i] if abs(p[p_i]) > 1e-14 else np.nan
        y2 = q_c[p_i] / q[p_i] if abs(q[p_i]) > 1e-14 else np.nan
        plt.plot(p_i * dt, y1, f'c{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, y1 + y_offset_p, f"{i+1}", fontsize=14, color='cyan', bbox=c["text_box_props"])
        plt.plot(p_i * dt, y2, f'c{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, y2 + y_offset_q, f"{i+1}", fontsize=14, color='cyan', bbox=c["text_box_props"])
    plt.xlabel(c["time_label"])
    plt.ylabel(c["p_ratio_label"])
    plt.legend()
    plt.savefig(output / f"{c['mode']}_p_c_over_p_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_p_q(c: dict, output: Path) -> None:
    p = c["p"]
    q = c["q"]
    p_total = c["p_total"]
    q_total = c["q_total"]
    M = c["M"]

    plt.figure(figsize=c["figsize"])
    plt.plot([0, np.max(p) * 1.5], [0, M * np.max(p) * 1.5], '-r', label='Critical state line')
    plt.plot(p_total, q_total, '-g', label=r"Total path")
    plt.plot(p, q, '-b', markevery=500, label=r"Quasi-static path")
    plt.plot(p[0], q[0], 'b.', ms=10, label='')
    plt.plot(p[-1], q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[0], q_total[0], 'g.', ms=10, label='')
    plt.plot(p_total[-1], q_total[-1], 'gx', ms=10, label='')
    for p_i, m_i in zip(c["points"], c["markers"]):
        if p_i >= len(p):
            continue
        plt.plot(p_total[p_i], q_total[p_i], f'g{m_i}', ms=10, mfc='none')
        plt.plot(p[p_i], q[p_i], f'b{m_i}', ms=10, mfc='none')
    plt.xlabel(c["pressure_label"])
    plt.ylabel(c["dev_stress_label"])
    if c["mode"] == 'drained':
        plt.xlim(0, 160)
        plt.ylim(0, 225)
    elif c["mode"] == 'undrained':
        if c["ocr"] > 1.0:
            plt.xlim(0, 350)
            plt.ylim(0, 400)
        else:
            plt.xlim(0, 160)
            plt.ylim(0, 180)
    plt.legend(loc='upper left')
    plt.savefig(output / f"{c['mode']}_p_vs_q_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_e_p(c: dict, output: Path) -> None:
    p = c["p"]
    p_total = c["p_total"]
    e_q = c["void_ratio_q"]
    e_total = c["void_ratio_total"]
    lambda_val = c["lambda_val"]

    plt.figure(figsize=c["figsize"])
    plt.plot(p_total, e_total, '-g', label='Total void ratio')
    plt.plot(p, e_q, '-b', label=r"Quasi-static void ratio")
    plt.plot(p[0], e_q[0], 'b.', ms=10, label='')
    plt.plot(p_total[0], e_total[0], 'g.', ms=10, label='')
    plt.plot(p[-1], e_q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[-1], e_total[-1], 'gx', ms=10, label='')
    for p_i, m_i in zip(c["points"], c["markers"]):
        if p_i >= len(p):
            continue
        plt.plot(p[p_i], e_q[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_total[p_i], e_total[p_i], f'g{m_i}', ms=10, mfc='none')

    xlim = plt.xlim()
    ylim = plt.ylim()
    pCSL = np.linspace(10, 400, 1000)
    gamma = e_q[-1] + lambda_val * np.log(p[-1])
    eCSL = gamma - lambda_val * np.log(pCSL)

    plt.clf()
    plt.plot(pCSL, eCSL, '-r', label='Critical state line')
    plt.plot(p_total, e_total, '-g', label='Total void ratio')
    plt.plot(p, e_q, '-b', label=r"Quasi-static void ratio")
    plt.plot(p[0], e_q[0], 'b.', ms=10, label='')
    plt.plot(p_total[0], e_total[0], 'g.', ms=10, label='')
    plt.plot(p[-1], e_q[-1], 'bx', ms=10, label='')
    plt.plot(p_total[-1], e_total[-1], 'gx', ms=10, label='')
    for p_i, m_i in zip(c["points"], c["markers"]):
        if p_i >= len(p):
            continue
        plt.plot(p[p_i], e_q[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_total[p_i], e_total[p_i], f'g{m_i}', ms=10, mfc='none')

    plt.xlim(xlim)
    plt.ylim(ylim)
    if c["mode"] == 'drained':
        plt.ylim(0.32, 0.50)
    elif c["mode"] == 'undrained':
        if c["ocr"] > 1.0:
            plt.ylim(0.26, 0.33)
        else:
            plt.ylim(0.44, 0.52)
    plt.xlabel(c["pressure_label"])
    plt.ylabel(c["e_label"])
    plt.savefig(output / f"{c['mode']}_p_vs_void_ratio_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_mu_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    p = c["p"]
    q = c["q"]
    p_total = c["p_total"]
    q_total = c["q_total"]

    mu_total = _safe_ratio(q_total, p_total)
    mu_q = _safe_ratio(q, p)

    plt.figure(figsize=c["figsize"])
    plt.axhline(1.0, color='red', linestyle='--', label=r'$\mu^{\mathrm{cs}}$')
    plt.axhline(1.5, color='red', linestyle=':', label=r'$\mu^{\mathrm{c}}$')
    plt.plot(t, mu_total, '-g', label=r"Total friction ($\mu$)")
    plt.plot(t, mu_q, '-b', label=r"Quasi-static friction ($\mu^{\mathrm{q}}$)")
    plt.plot(0 * dt, mu_q[0], 'b.', ms=10, label='')
    plt.plot(0 * dt, mu_total[0], 'g.', ms=10, label='')
    plt.plot(c["load_length"] * dt, mu_q[-1], 'bx', ms=10, label='')
    plt.plot(c["load_length"] * dt, mu_total[-1], 'gx', ms=10, label='')

    x_offset = c["load_length"] * dt * 0.035
    y_offset_quasi_ratio = np.nanmax(mu_q) * 0.01 if np.any(np.isfinite(mu_q)) else 0.0
    y_offset_total_ratio = np.nanmax(mu_total) * 0.01 if np.any(np.isfinite(mu_total)) else 0.0
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(mu_q):
            continue
        plt.plot(p_i * dt, mu_q[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_i * dt, mu_total[p_i], f'g{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, mu_q[p_i] - y_offset_quasi_ratio, f"{i+1}", fontsize=14, color='blue', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, mu_total[p_i] - y_offset_total_ratio, f"{i+1}", fontsize=14, color='green', bbox=c["text_box_props"])
    plt.xlabel(c["time_label"])
    plt.ylabel(c["mu_label"])
    plt.legend()
    plt.savefig(output / f"{c['mode']}_mu_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_e_components_time(c: dict, output: Path) -> None:
    t = c["t"]
    dt = c["dt"]
    ev = c["ev"]
    ev_qp = c["ev_qp"]
    ev_cp = c["ev_cp"]
    ev_e = c["ev_e"]

    plt.figure(figsize=c["figsize"])
    plt.plot(t, ev, '-', color='gray', label=r"$\varepsilon_v$")
    plt.plot(t, ev_e, '-g', label=r"$\varepsilon_v^{\mathrm{e}}$")
    plt.plot(t, ev_qp, '-b', label=r"$\varepsilon_v^{\mathrm{p,q}}$")
    plt.plot(t, ev_cp, '-m', label=r"$\varepsilon_v^{\mathrm{p,d}}$")

    x_offset = c["load_length"] * dt * 0.04
    y_offset_qp = max(ev_qp) * 0.01
    y_offset_e = max(ev_e) * 0.01
    y_offset_cp = max(ev_cp) * 0.01
    for i, (p_i, m_i) in enumerate(zip(c["points"], c["markers"])):
        if p_i >= len(ev):
            continue
        plt.plot(p_i * dt, ev[p_i], f'{m_i}', color='gray', ms=10, mfc='none')
        plt.plot(p_i * dt, ev_qp[p_i], f'b{m_i}', ms=10, mfc='none')
        plt.plot(p_i * dt, ev_e[p_i], f'g{m_i}', ms=10, mfc='none')
        plt.plot(p_i * dt, ev_cp[p_i], f'm{m_i}', ms=10, mfc='none')
        plt.text(p_i * dt + x_offset, ev[p_i] - y_offset_e, f"{i+1}", fontsize=12, color='gray', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, ev_qp[p_i] + y_offset_qp, f"{i+1}", fontsize=12, color='blue', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, ev_e[p_i] + y_offset_e, f"{i+1}", fontsize=12, color='green', bbox=c["text_box_props"])
        plt.text(p_i * dt + x_offset, ev_cp[p_i] + y_offset_cp, f"{i+1}", fontsize=12, color='magenta', bbox=c["text_box_props"])
    plt.plot(0 * dt, 0, 'b.', ms=10, label='')
    plt.plot(0 * dt, 0, 'g.', ms=10, label='')
    plt.plot(0 * dt, 0, 'm.', ms=10, label='')
    plt.plot(c["load_length"] * dt, ev_qp[-1], 'bx', ms=10, label='')
    plt.plot(c["load_length"] * dt, ev_e[-1], 'gx', ms=10, label='')
    plt.plot(c["load_length"] * dt, ev_cp[-1], 'mx', ms=10, label='')
    if c["mode"] == 'drained':
        plt.ylim(-0.15, 0.1)
    elif c["mode"] == 'undrained':
        plt.ylim(-0.05, 0.05)
        plt.yticks(np.linspace(-0.05, 0.05, 5))
    plt.xlabel(c["time_label"])
    plt.ylabel(c["ev_label"])
    plt.legend()
    plt.savefig(output / f"{c['mode']}_e_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_mu_gamma(c: dict, output: Path) -> None:
    mu_total = _safe_ratio(c["q_total"], c["p_total"])
    plt.figure(figsize=c["figsize"])
    plt.plot(c["eqp_inc_history"] / c["dt"], mu_total)
    plt.xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    plt.ylabel(c["mu_label"])
    plt.ylim(1)
    plt.savefig(output / f"{c['mode']}_mu_gamma_{c['e0']:.3f}_{c['ocr']:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_gamma_phi_transient(c: dict, output: Path) -> None:
    points = c["points"]
    if len(points) < 4:
        return

    p1, p2, p3, p4 = points[:4]
    x = c["eqp_inc_history"] / c["dt"]
    phi_total = 1 / (1 + c["void_ratio_total"])

    plt.figure(figsize=c["figsize"])
    plt.plot(x[p1:p2], phi_total[p1:p2])
    plt.plot(x[p2:p3], phi_total[p2:p3])
    plt.plot(x[p3:p4], phi_total[p3:p4])
    plt.plot(x[p4:], phi_total[p4:])
    x_offset = 0.03
    y_offset = 0.00
    for i, (p_i, m_i) in enumerate(zip(points, c["markers"])):
        if p_i >= len(phi_total):
            continue
        plt.plot(x[p_i], phi_total[p_i], f'k{m_i}', ms=10, mfc='none')
        plt.text(x[p_i] + x_offset, phi_total[p_i] + y_offset, f"{i+1}", fontsize=14)
    plt.xlabel('Shear rate')
    plt.ylabel('Solid volume fraction (total)')
    plt.savefig(output / "gamma_phi_transient.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_plots(data: dict, history: dict, output_dir: str = ".") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    c = _collect_context(data, history)

    # Keep the same plotting order as the original production block.
    _plot_load_history(c, output)
    _plot_pressure_time(c, output)
    _plot_q_time(c, output)
    _plot_ev_gamma_time(c, output)
    _plot_pc_time(c, output)
    _plot_dynamic_ratio(c, output)
    _plot_p_q(c, output)
    _plot_e_p(c, output)
    _plot_mu_time(c, output)
    _plot_e_components_time(c, output)
    _plot_mu_gamma(c, output)
    _plot_gamma_phi_transient(c, output)

    plt.close('all')


def make_diff_rates_summary_plot(data: Dict[str, np.ndarray], history: Dict[str, np.ndarray], output_dir: str = ".") -> None:
    """Per-accel_time 2x4 diagnostic grid, mirroring the original figure saved as
    '{accel:.2f}_summary.png' in backup/CMCC_evpc_diff_rates.py."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dt = history["dt"]
    load_length = history["load_length"]
    t = np.arange(load_length) * dt

    p, q = data["p"], data["q"]
    p_total, q_total = data["p_total"], data["q_total"]
    eq, ev = data["eq"], data["ev"]
    pc_history = data["pc_history"]
    p_c = data["p_c"]

    plt.figure('Pressure controlled simple shear', figsize=(14, 7))

    plt.subplot(2, 4, 1)
    plt.plot(t, p, '-b', label='qstat')
    plt.plot(t, p_total, '-g', label='tot')
    plt.plot(0, p[0], 'bx', ms=10, label='')
    plt.plot(0, p_total[0], 'g+', ms=10, label='')
    plt.plot(t[-1], p[-1], 'bo', ms=10, label='')
    plt.plot(t[-1], p_total[-1], 'g.', ms=10, label='')
    plt.xlabel('t [s]')
    plt.ylabel('p [kPa]')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(t, q, '-b', label='qstat')
    plt.plot(t, q_total, '-g', label='tot')
    plt.plot(0, q[0], 'bx', ms=10, label='')
    plt.plot(0, q_total[0], 'g+', ms=10, label='')
    plt.plot(t[-1], q[-1], 'bo', ms=10, label='')
    plt.plot(t[-1], q_total[-1], 'g.', ms=10, label='')
    plt.xlabel('t [s]')
    plt.ylabel('q [kPa]')
    plt.legend()

    ax1 = plt.subplot(2, 4, 3)
    ax1.plot(t, eq, '-b', label='dev')
    ax1.plot(0, eq[0], 'bx', ms=10, label='')
    ax1.plot(t[-1], eq[-1], 'bo', ms=10, label='')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('Deviatoric Strain', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(t, ev, '--g', label='v')
    ax2.plot(0, ev[0], 'gx', ms=10, label='')
    ax2.plot(t[-1], ev[-1], 'g.', ms=10, label='')
    ax2.set_ylabel('Volumetric Strain', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.subplot(2, 4, 4)
    plt.plot(t, pc_history, '-b')
    plt.plot(0, pc_history[0], 'bx', ms=10, label='')
    plt.plot(t[-1], pc_history[-1], 'b.', ms=10, label='')
    plt.xlabel('t [s]')
    plt.ylabel('pc(t) [kPa]')

    plt.subplot(2, 4, 5)
    plt.plot(t, p_c / p, '-b')
    plt.plot(0, p_c[0] / p[0], 'bx', ms=10, label='')
    plt.plot(t[-1], p_c[-1] / p[-1], 'b.', ms=10, label='')
    plt.xlabel('t [s]')
    plt.ylabel('p_c/p [-]')

    plt.subplot(2, 4, 6)
    m = data["M"]
    plt.plot(p, q, '-b', markevery=500, label='qstat')
    plt.plot(p_total, q_total, '-g', label='tot')
    plt.plot([0, np.max(p)], [0, m * np.max(p)], '-r', label='CSL')
    plt.plot(p[0], q[0], 'bx', ms=10, label='')
    plt.plot(p[-1], q[-1], 'b.', ms=10, label='')
    plt.plot(p_total[0], q_total[0], 'g+', ms=10, label='')
    plt.plot(p_total[-1], q_total[-1], 'g.', ms=10, label='')
    plt.xlabel('p [kPa]')
    plt.ylabel('q [kPa]')
    plt.legend()

    plt.subplot(2, 4, 7)
    void_ratio_q = data["void_ratio_q"]
    void_ratio_total = data["void_ratio_total"]
    plt.plot(p, void_ratio_q, '-b', label='qstat')
    plt.plot(p_total, void_ratio_total, '-g', label='tot')
    plt.plot(p[0], void_ratio_q[0], 'bx', ms=10, label='')
    plt.plot(p_total[0], void_ratio_total[0], 'g+', ms=10, label='')
    plt.plot(p[-1], void_ratio_q[-1], 'b.', ms=10, label='')
    plt.plot(p_total[-1], void_ratio_total[-1], 'g.', ms=10, label='')
    plt.xlabel('p [kPa]')
    plt.ylabel('e [-]')

    plt.subplot(2, 4, 8)
    plt.plot(t, q_total / p_total, 'b')
    plt.plot(0, q_total[0] / p_total[0], 'b+', ms=10, label='')
    plt.plot(t[-1], q_total[-1] / p_total[-1], 'b.', ms=10, label='')
    plt.xlabel('t [s]')
    plt.ylabel('q/p [-]')

    eqp_inc_history = history["eqp_inc_history"]
    accel = max(eqp_inc_history[1:] - eqp_inc_history[:-1]) / dt / dt
    plt.tight_layout()
    plt.savefig(output / f'{round(accel, 3):.2f}_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def make_diff_rates_comparison_plots(
    results: List[Dict],
    mode: str,
    void_ratio_0: float,
    OCR: float,
    M: float,
    lambda_val: float,
    Delta_Phi: float,
    output_dir: str = ".",
) -> None:
    """Overlay each acceleration-rate run on shared load-history, p-q, e-p, gamma-mu,
    and gamma-e figures, mirroring the second loop in backup/CMCC_evpc_diff_rates.py.

    `results` is a list of dicts, each with keys "history" and "output" (as produced
    by cmcc_diff_rates_driver.run_diff_rates_study).
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    n_runs = len(results)
    figsize = (6, 4.5)
    fig_load, ax_load = plt.subplots()
    fig_p_q, ax_p_q = plt.subplots(figsize=figsize)
    fig_e_p, ax_e_p = plt.subplots(figsize=figsize)
    fig_mu, ax_mu = plt.subplots(figsize=figsize)
    fig_void, ax_void = plt.subplots(figsize=figsize)

    color_csl = 'r'
    point1 = None
    p_ref, void_ratio_q_ref = None, None

    for k, res in enumerate(reversed(results)):
        alpha = (1 - k / max(n_runs - 1, 1)) * 0.7
        color_qs = lighten('b', alpha)
        color_load = lighten('k', alpha)

        history = res["history"]
        output_data = res["output"]
        dt = history["dt"]
        eqp_inc_history = history["eqp_inc_history"]
        load_length = history["load_length"]
        points = list(history["points"])
        markers = list(history["markers"]) + ['x']
        points = points + [load_length - 1]

        p = output_data["p"]
        q = output_data["q"]
        void_ratio_q = output_data["void_ratio_q"]
        eq = output_data["eq"]

        if point1 is None:
            point1 = points[0]
            p_ref = p
            void_ratio_q_ref = void_ratio_q

        accel = max(eqp_inc_history[1:] - eqp_inc_history[:-1]) / dt / dt
        label = rf'$|\ddot{{\varepsilon}}_{{zz}}| = {np.ceil(accel * 1e2) / 1e2:.2f}$ s' + r'$^{-2}$'

        # Load history
        ax_load.plot(np.arange(load_length) * dt, eqp_inc_history / dt, label=label, color=color_load)
        for i, (p_i, m_i) in enumerate(zip(points, markers)):
            mfc = 'red' if m_i == '^' else 'none'
            ax_load.plot(p_i * dt, eqp_inc_history[p_i] / dt, f'{m_i}', color=color_load, ms=10, mfc=mfc)
        ax_load.plot(0, eqp_inc_history[0] / dt, '.', color=color_load, ms=10, label='')
        ax_load.plot(load_length * dt, eqp_inc_history[-1] / dt, 'x', color=color_load, ms=10, label='')

        # p-q
        segmented_plot(ax_p_q, p, q, [points[0], len(p) - 1], label=label, color=color_qs)
        for p_i, m_i in zip(points, markers):
            mfc = 'red' if m_i == '^' else 'none'
            ax_p_q.plot(p[p_i], q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)

        # e-p
        segmented_plot(ax_e_p, p, void_ratio_q, [points[0], len(p) - 1], label=label, color=color_qs)
        for p_i, m_i in zip(points, markers):
            mfc = 'red' if m_i == '^' else 'none'
            ax_e_p.plot(p[p_i], void_ratio_q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)

        # gamma_dot vs mu
        gamma_dot = (eq[1:] - eq[:-1]) / dt
        gamma_dot = np.insert(gamma_dot, 0, 0)
        mu_q = np.divide(q, p, out=np.full_like(q, np.nan), where=np.abs(p) > 1e-14)
        segmented_plot(ax_mu, gamma_dot, mu_q, [points[0], len(p) - 1], label=label, color=color_qs)
        for p_i, m_i in zip(points, markers):
            mfc = 'red' if m_i == '^' else 'none'
            ax_mu.plot(gamma_dot[p_i], mu_q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)

        # gamma_dot vs void ratio
        segmented_plot(ax_void, gamma_dot, void_ratio_q, [points[0], len(p) - 1], label=label, color=color_qs)
        for p_i, m_i in zip(points, markers):
            mfc = 'red' if m_i == '^' else 'none'
            ax_void.plot(gamma_dot[p_i], void_ratio_q[p_i], marker=m_i, color=color_qs, ms=10, mfc=mfc)

    ax_load.set_xlabel(r'Time $t$ [s]')
    ax_load.set_ylabel(r'Axial strain rate $\dot{\varepsilon}_{zz}$ [1/s]')
    ax_load.set_xlim(-10, 410)
    ax_load.legend()

    ax_p_q.set_xlabel(r'Pressure $p$ [kPa]')
    ax_p_q.set_ylabel(r'Deviatoric stress $q$ [kPa]')
    ax_e_p.set_xlabel(r'Pressure $p$ [kPa]')
    ax_e_p.set_ylabel(r'Void ratio $e$ [-]')
    ax_mu.set_xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    ax_mu.set_ylabel(r'Stress ratio $q/p$ [-]')
    ax_void.set_xlabel(r'Shear rate $\dot{\gamma}$ [1/s]')
    ax_void.set_ylabel(r'Void ratio $e$')

    x_min, x_max = ax_p_q.get_xlim()
    y_min, y_max = ax_p_q.get_ylim()
    p_line = np.linspace(x_min, x_max, 100)
    ax_p_q.plot(p_line, M * p_line, '-', color=color_csl, label='Critical state line', zorder=0)
    ax_p_q.set_xlim(x_min, x_max)
    ax_p_q.set_ylim(y_min, y_max)
    ax_p_q.legend()

    gamma_ref = void_ratio_q_ref[point1] + lambda_val * np.log(p_ref[point1])
    eCSL = gamma_ref - lambda_val * np.log(p_line)
    ax_e_p.plot(p_line, eCSL, '-', color=color_csl, label='Critical state line', zorder=0)
    ax_e_p.set_xlim(x_min, x_max)
    ax_e_p.legend()

    x_min, x_max = ax_void.get_xlim()
    gamma_line = np.linspace(x_min, x_max, 100)
    dilatancy_line = 1.0 / (1.0 + void_ratio_q_ref[point1]) - Delta_Phi * gamma_line
    ax_void.plot(gamma_line, (1.0 - dilatancy_line) / dilatancy_line, color=color_csl,
                 label='Rate-induced dilatancy', zorder=0)
    ax_void.set_xlim(x_min, x_max)
    ax_void.legend()

    ax_mu.plot(gamma_line, np.ones(100) * M, color=color_csl, label=r'$\mu^{\mathrm{cs}}$', zorder=0)
    ax_mu.set_xlim(x_min, x_max)
    ax_mu.legend()

    expand_ylim_for_annotations(ax_load, top_frac=0.05)
    expand_ylim_for_annotations(ax_p_q, top_frac=0.3)
    expand_ylim_for_annotations(ax_e_p)
    expand_ylim_for_annotations(ax_mu, top_frac=0.05, bottom_frac=0.05)
    expand_ylim_for_annotations(ax_void, top_frac=0.3)

    fig_load.savefig(output / f"{mode}_load_history.png", dpi=300, bbox_inches="tight")
    fig_p_q.savefig(output / f"{mode}_p_vs_q_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
    fig_e_p.savefig(output / f"{mode}_p_vs_void_ratio_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
    fig_mu.savefig(output / f"{mode}_gamma_vs_mu_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
    fig_void.savefig(output / f"{mode}_gamma_vs_phi_{void_ratio_0:.3f}_{OCR:.3f}_diff_rate.png", dpi=300, bbox_inches="tight")
    plt.close('all')
