
"""Berendsen-thermostat project for a Lennard-Jones argon gas."""


import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Avogadro, R
from scipy.spatial import cKDTree
from scipy.stats import gamma, norm


# Particle and simulation data containers
class ParticleSystem:
    """Per-particle parameters and dynamical state."""

    def __init__(self, n_particles):
        if n_particles < 2:
            raise ValueError("At least two particles are required")

        self.n = n_particles

        self.mass = np.zeros(n_particles)
        self.sigma = np.zeros(n_particles)
        self.epsilon = np.zeros(n_particles)

        self.position = np.zeros((n_particles, 3))
        self.velocity = np.zeros((n_particles, 3))
        self.force = np.zeros((n_particles, 3))

        self.current_potential_energy = None

        self.rng = np.random.default_rng()

    def __repr__(self):
        return f"<ParticleSystem with {self.n} particles>"


class SimulationParameters:
    def __init__(self, dt, n_steps, temperature, box_length,
                 tau_thermostat=None, rij_min=0.0, cutoff=None):
        """Store simulation parameters in ps, K, and nm."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if box_length <= 0.0:
            raise ValueError("box_length must be positive")
        if tau_thermostat is not None and tau_thermostat <= 0.0:
            raise ValueError("tau_thermostat must be positive or None")
        if rij_min < 0.0:
            raise ValueError("rij_min must be non-negative")
        if cutoff is not None and (cutoff <= 0.0 or cutoff > box_length / 2.0):
            raise ValueError("cutoff must lie in (0, box_length/2]")
        if cutoff is not None and rij_min >= cutoff:
            raise ValueError("rij_min must be smaller than cutoff")

        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.temperature = float(temperature)
        self.box_length = float(box_length)
        self.tau_thermostat = tau_thermostat
        self.rij_min = float(rij_min)
        self.cutoff = None if cutoff is None else float(cutoff)

        self.xi = None if tau_thermostat is None else 1.0 / tau_thermostat



# Initial positions and velocities
def initialize_positions_grid(ps: ParticleSystem, box_length_in_nm: float):
    """Place particles on a cubic grid, avoiding unrealistically close pairs."""

    n_side = int(np.ceil(ps.n ** (1.0 / 3.0)))

    spacing = box_length_in_nm / n_side

    coordinates = (np.arange(n_side) + 0.5) * spacing

    grid = np.array(
        np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    ).reshape(3, -1).T

    ps.position[:] = grid[:ps.n]


def remove_center_of_mass_velocity(ps: ParticleSystem):
    """Remove translational centre-of-mass motion."""

    v_cm = np.average(ps.velocity, axis=0, weights=ps.mass)

    ps.velocity -= v_cm


def initialize_velocities(ps: ParticleSystem, temperature: float):
    """Draw Maxwell-Boltzmann velocities in nm/ps."""

    molar_mass = ps.mass * 1e-3                 # kg/mol

    standard_deviation = np.sqrt(R * temperature / molar_mass)  # m/s

    velocities_m_s = ps.rng.normal(
        0.0, standard_deviation[:, np.newaxis], size=(ps.n, 3)
    )

    ps.velocity[:] = velocities_m_s * 1e-3     # m/s to nm/ps

    remove_center_of_mass_velocity(ps)


def rescale_temperature(ps: ParticleSystem, temperature: float):
    """Rescale velocities once so their instantaneous temperature is exact."""

    current_temperature = instantaneous_temperature(ps)
    if current_temperature <= 0.0:
        raise ValueError("Cannot rescale a system with zero kinetic energy")
    ps.velocity *= np.sqrt(temperature / current_temperature)



# Thermodynamic quantities
def potential_energy(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """Return the cached force-shifted Lennard-Jones potential energy."""

    if ps.current_potential_energy is None:
        calculate_force(ps, sim)
    return float(ps.current_potential_energy)


def kinetic_energy(ps: ParticleSystem) -> float:
    """Calculate total kinetic energy in kJ/mol."""

    velocity_squared = np.sum(ps.velocity ** 2, axis=1)

    return float(0.5 * np.sum(ps.mass * velocity_squared))


def thermal_degrees_of_freedom(n_particles_value: int) -> int:
    """Return 3N-3 after removing the three centre-of-mass modes."""
    if n_particles_value < 2:
        raise ValueError("At least two particles are required")
    return 3 * n_particles_value - 3


def thermal_kinetic_energy(ps: ParticleSystem) -> float:
    """Kinetic energy relative to the centre-of-mass velocity in kJ/mol."""

    center_of_mass_velocity = np.average(
        ps.velocity, axis=0, weights=ps.mass
    )

    thermal_velocity = ps.velocity - center_of_mass_velocity
    return float(
        0.5 * np.sum(ps.mass * np.sum(thermal_velocity ** 2, axis=1))
    )


def instantaneous_temperature(ps: ParticleSystem) -> float:
    """Calculate thermal temperature using the constrained 3N-3 modes."""

    degrees_of_freedom = thermal_degrees_of_freedom(ps.n)

    return 2.0 * thermal_kinetic_energy(ps) * 1e3 / (
        degrees_of_freedom * R
    )


def ideal_gas_pressure(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """Calculate ideal-gas pressure in Pa using the instantaneous temperature."""

    volume_m3 = sim.box_length ** 3 * 1e-27

    amount_mol = ps.n / Avogadro

    return float(amount_mol * R * instantaneous_temperature(ps) / volume_m3)



# Lennard-Jones forces and integration steps
def calculate_force(ps: ParticleSystem, sim: SimulationParameters):
    """Calculate force-shifted Lennard-Jones forces with periodic boundaries.

    A periodic KD-tree finds only pairs within the cutoff, avoiding an N x N
    distance matrix.  Both potential and force are shifted to zero at the
    cutoff, so pairs crossing it do not introduce discontinuities into the NVE
    energy test.
    """

    cutoff = sim.cutoff
    if cutoff is None:
        cutoff = min(2.5 * ps.sigma[0], sim.box_length / 2.0)

    ps.position[:] = np.mod(ps.position, sim.box_length)

    pairs = cKDTree(ps.position, boxsize=sim.box_length).query_pairs(
        cutoff, output_type="ndarray"
    )

    ps.force.fill(0.0)

    if len(pairs) == 0:
        ps.current_potential_energy = 0.0
        return

    particle_i = pairs[:, 0]
    particle_j = pairs[:, 1]

    displacement = ps.position[particle_i] - ps.position[particle_j]
    displacement -= sim.box_length * np.rint(displacement / sim.box_length)
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("Two particles occupy the same position")
    r = np.maximum(distance, sim.rij_min)

    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    sr6 = (sigma / r) ** 6

    potential = 4.0 * epsilon * (sr6 ** 2 - sr6)

    force_magnitude = 24.0 * epsilon * (2.0 * sr6 ** 2 - sr6) / r

    src6 = (sigma / cutoff) ** 6
    potential_at_cutoff = 4.0 * epsilon * (src6 ** 2 - src6)
    force_at_cutoff = (
        24.0 * epsilon * (2.0 * src6 ** 2 - src6) / cutoff
    )
    shifted_potential = (
        potential - potential_at_cutoff + (r - cutoff) * force_at_cutoff
    )
    shifted_force_magnitude = force_magnitude - force_at_cutoff

    force_vector = (
        shifted_force_magnitude[:, np.newaxis]
        * displacement / distance[:, np.newaxis]
    )
    np.add.at(ps.force, particle_i, force_vector)
    np.add.at(ps.force, particle_j, -force_vector)
    ps.current_potential_energy = float(np.sum(shifted_potential))


def A_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Advance positions for a full or half time step."""

    dt = 0.5 * sim.dt if half_step else sim.dt

    ps.position += ps.velocity * dt

    ps.current_potential_energy = None


def B_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Advance velocities under the deterministic force."""

    dt = 0.5 * sim.dt if half_step else sim.dt

    ps.velocity += (dt * ps.force) / ps.mass[:, np.newaxis]


def O_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Apply the stochastic Ornstein-Uhlenbeck step used by BAOAB."""

    if sim.xi is None:
        raise ValueError("tau_thermostat is required for Langevin dynamics")

    dt = 0.5 * sim.dt if half_step else sim.dt

    random_number = ps.rng.normal(size=(ps.n, 3))

    damping = np.exp(-sim.xi * dt)

    variance = (
        R * sim.temperature * (1.0 - np.exp(-2.0 * sim.xi * dt))
        / (ps.mass * 1e3)
    )
    ps.velocity[:] = (
        damping * ps.velocity
        + np.sqrt(variance)[:, np.newaxis] * random_number
    )


def apply_periodic_boundary(ps: ParticleSystem, sim: SimulationParameters):
    """Wrap all coordinates into [0, L)."""

    ps.position[:] = np.mod(ps.position, sim.box_length)


def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters):
    """Perform one Velocity Verlet step in BAB form."""

    # Velocity Verlet: half kick, drift, force update, half kick
    B_step(ps, sim, half_step=True)

    A_step(ps, sim)

    apply_periodic_boundary(ps, sim)

    calculate_force(ps, sim)

    B_step(ps, sim, half_step=True)


def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters):
    """Perform one Langevin BAOAB step."""
    if sim.tau_thermostat is None:
        raise ValueError("tau_thermostat is required for an NVT simulation")
    # Langevin BAOAB sequence: B-A-O-A-B
    B_step(ps, sim, half_step=True)

    A_step(ps, sim, half_step=True)

    O_step(ps, sim)

    A_step(ps, sim, half_step=True)

    apply_periodic_boundary(ps, sim)

    calculate_force(ps, sim)

    B_step(ps, sim, half_step=True)


def apply_berendsen_thermostat(ps: ParticleSystem,
                               sim: SimulationParameters) -> float:
    """Rescale velocities according to the Berendsen weak-coupling rule.

    lambda = sqrt(1 + dt/tau * (T_target/T_current - 1))

    The returned scale factor is useful for diagnostics.
    """
    if sim.tau_thermostat is None:
        raise ValueError("tau_thermostat is required for Berendsen coupling")

    current_temperature = instantaneous_temperature(ps)
    if current_temperature <= 0.0:
        raise ValueError("Cannot thermostat a system with zero kinetic energy")

    # Berendsen velocity-rescaling factor
    factor = 1.0 + sim.dt / sim.tau_thermostat * (
        sim.temperature / current_temperature - 1.0
    )
    if factor <= 0.0:
        raise ValueError(
            "Berendsen scale factor is not real. Increase tau_thermostat or "
            "decrease dt."
        )

    velocity_scale = np.sqrt(factor)
    ps.velocity *= velocity_scale
    return float(velocity_scale)


def simulate_Berendsen_step(ps: ParticleSystem, sim: SimulationParameters):
    """Perform Velocity Verlet followed by Berendsen velocity rescaling."""

    simulate_NVE_step(ps, sim)

    return apply_berendsen_thermostat(ps, sim)



# VMD-readable XYZ output
def write_xyz_trajectory(filename, trajectory, atom_symbol="Ar"):
    """Write a position trajectory to a VMD-readable multi-frame XYZ file.

    Parameters
    ----------
    filename
        Destination path.
    trajectory
        Array with shape ``(frames, particles, 3)`` in nm.
    atom_symbol
        Element label written for every particle.  XYZ conventionally uses
        Angstrom, so positions are converted from nm by multiplying by ten.

    XYZ files contain positions only.  Full velocity trajectories are saved
    separately in compressed NPZ files by ``export_matched_full_trajectories``.
    """
    filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("trajectory must have shape (frames, particles, 3)")
    trajectory_angstrom = 10.0 * trajectory
    n_frames, n_atoms, _ = trajectory_angstrom.shape
    with open(filename, "w", encoding="utf-8") as output_file:
        for frame_number, frame in enumerate(trajectory_angstrom):
            output_file.write(f"{n_atoms}\n")
            output_file.write(f"Lennard-Jones MD frame {frame_number}\n")
            for position in frame:
                output_file.write(
                    f"{atom_symbol} {position[0]:.8f} "
                    f"{position[1]:.8f} {position[2]:.8f}\n"
                )



# Project settings: argon model and thermostat runs
n_particles = 200
mass_argon = 39.95                         # u
sigma_argon = 0.34                         # nm
epsilon_argon = 120.0 * R * 1e-3          # kJ/mol (epsilon/k_B = 120 K)

initial_temperature = 100.0               # K
target_temperature = 300.0                # K
dt = 0.01                                 # ps; selected by interacting NVE test
box_length = 100.0                        # nm; intentionally dilute gas
rij_min = 1e-2                            # nm; numerical collision safeguard
cutoff = 2.5 * sigma_argon                # nm; standard LJ cutoff
coupling_steps = int(round(100.0 / dt))   # 100 ps Berendsen-only overview

coupling_times = {
    "strong": 0.2,
    "intermediate": 1.0,
    "weak": 10.0,
}
comparison_duration_ps = {0.2: 200.0, 1.0: 200.0, 10.0: 400.0}
comparison_equilibration_ps = {0.2: 50.0, 1.0: 50.0, 10.0: 100.0}

# Seeds used for repeat statistics
replicate_seeds = (2026, 2027, 2028)
random_seed = replicate_seeds[0]

matched_trajectory_tau = 1.0              # ps
matched_trajectory_duration = 100.0       # ps
trajectory_stride = 10                    # save every 0.1 ps for compact VMD files
interacting_box_length = 8.0              # nm; supplementary collision-rich view
interacting_trajectory_duration = 100.0   # ps
interacting_plot_duration = 50.0          # ps shown in the static 3D figure

# Denser NVE system used to validate dt
nve_particles = n_particles
nve_box_length = interacting_box_length   # use the same supplementary box size
nve_duration = 10.0                       # ps
nve_time_steps = (0.005, 0.01, 0.02, 0.05, 0.1)  # ps
nve_error_threshold = 1e-3

# Output folders and comparison line styles
output_directory = Path("project_results_100K")
trajectory_output_directory = Path("VMD_3D_files")
interacting_trajectory_output_directory = (
    trajectory_output_directory / "interacting_8nm"
)
# Figures are always saved. Set this to True only when interactive windows are
# wanted; False lets the complete analysis run without pausing at every plot.
show_figures = False
thermostat_plot_order = ("langevin", "berendsen")
thermostat_styles = {
    "berendsen": {
        "color": "tab:red", "linestyle": "--", "linewidth": 1.5,
        "alpha": 1.0, "zorder": 3,
    },
    "langevin": {
        "color": "tab:cyan", "linestyle": "-", "linewidth": 1.0,
        "alpha": 1.0, "zorder": 1,
    },
}


# General simulation runner
def create_particle_system(n_particles_value, initial_temperature_value,
                           box_length_value, seed):
    """Create the argon system used by all simulations."""

    ps = ParticleSystem(n_particles_value)
    ps.rng = np.random.default_rng(seed)

    ps.mass.fill(mass_argon)
    ps.sigma.fill(sigma_argon)
    ps.epsilon.fill(epsilon_argon)

    initialize_positions_grid(ps, box_length_value)

    initialize_velocities(ps, initial_temperature_value)
    rescale_temperature(ps, initial_temperature_value)
    return ps


def _measure(ps, sim):
    """Return potential, kinetic, total energy, temperature, and pressure."""

    potential = ps.current_potential_energy
    if potential is None:
        potential = potential_energy(ps, sim)
    kinetic = kinetic_energy(ps)

    temperature_value = instantaneous_temperature(ps)
    pressure = ideal_gas_pressure(ps, sim)
    return np.array([
        potential,
        kinetic,
        potential + kinetic,
        temperature_value,
        pressure,
    ])


def run_simulation(thermostat_name="berendsen", *,
                   n_particles_value=n_particles,
                   dt_value=dt,
                   n_steps_value=coupling_steps,
                   target_temperature=target_temperature,
                   initial_temperature_value=initial_temperature,
                   box_length_value=box_length,
                   tau_value=matched_trajectory_tau,
                   rij_min_value=rij_min,
                   cutoff_value=cutoff,
                   seed=random_seed,
                   store_full_trajectory=False,
                   trajectory_stride=1,
                   constrain_center_of_mass=True):
    """Run one case and return all quantities needed for the analysis.

    ``thermostat_name`` must be ``berendsen``, ``langevin``, or ``nve``.
    Energies are stored at every step.  Full particle trajectories are optional;
    particle 0 is always retained for representative trajectory plots.
    """

    thermostat_name = thermostat_name.lower()

    if thermostat_name not in {"berendsen", "langevin", "nve"}:
        raise ValueError("thermostat_name must be berendsen, langevin, or nve")
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be at least 1")

    thermostat_tau = None if thermostat_name == "nve" else tau_value

    sim = SimulationParameters(
        dt=dt_value,
        n_steps=n_steps_value,
        temperature=target_temperature,
        box_length=box_length_value,
        tau_thermostat=thermostat_tau,
        rij_min=rij_min_value,
        cutoff=cutoff_value,
    )
    ps = create_particle_system(
        n_particles_value,
        initial_temperature_value,
        box_length_value,
        seed,
    )
    calculate_force(ps, sim)

    time_ps = np.arange(sim.n_steps + 1) * sim.dt

    energy_trajectory = np.zeros((sim.n_steps + 1, 5))

    selected_position = np.zeros((sim.n_steps + 1, 3))
    selected_velocity = np.zeros((sim.n_steps + 1, 3))

    velocity_scale = np.ones(sim.n_steps + 1)

    energy_trajectory[0] = _measure(ps, sim)
    selected_position[0] = ps.position[0]
    selected_velocity[0] = ps.velocity[0]

    if store_full_trajectory:
        saved_steps = np.arange(0, sim.n_steps + 1, trajectory_stride)

        if saved_steps[-1] != sim.n_steps:
            saved_steps = np.append(saved_steps, sim.n_steps)
        position_trajectory = np.zeros((len(saved_steps), ps.n, 3))
        velocity_trajectory = np.zeros_like(position_trajectory)
        position_trajectory[0] = ps.position
        velocity_trajectory[0] = ps.velocity
        next_frame = 1
    else:
        saved_steps = np.array([], dtype=int)
        position_trajectory = None
        velocity_trajectory = None
        next_frame = 0

    for step in range(1, sim.n_steps + 1):
        if thermostat_name == "berendsen":
            velocity_scale[step] = simulate_Berendsen_step(ps, sim)
        elif thermostat_name == "langevin":
            simulate_NVT_step(ps, sim)
        else:
            simulate_NVE_step(ps, sim)

        if constrain_center_of_mass:
            remove_center_of_mass_velocity(ps)

        energy_trajectory[step] = _measure(ps, sim)
        selected_position[step] = ps.position[0]
        selected_velocity[step] = ps.velocity[0]

        if store_full_trajectory and next_frame < len(saved_steps):
            if step == saved_steps[next_frame]:
                position_trajectory[next_frame] = ps.position
                velocity_trajectory[next_frame] = ps.velocity
                next_frame += 1

    return {
        "thermostat": thermostat_name,
        "sim": sim,
        "particle_system": ps,
        "time_ps": time_ps,
        "energy": energy_trajectory,
        "selected_position": selected_position,
        "selected_velocity": selected_velocity,
        "velocity_scale": velocity_scale,
        "saved_steps": saved_steps,
        "position_trajectory": position_trajectory,
        "velocity_trajectory": velocity_trajectory,
    }


# Statistics and file helpers
def first_time_within_tolerance(time_ps, values, target, tolerance=0.05):
    """Return the first time a series lies within a relative target interval."""

    inside = np.abs(values - target) <= tolerance * target

    indices = np.flatnonzero(inside)

    return float(time_ps[indices[0]]) if len(indices) else np.nan


def ideal_canonical_energy_variance(n_particles_value, temperature_value):
    """Ideal-gas canonical variance after removing COM translation.

    The project formula uses 3N degrees of freedom. This simulation removes
    the three centre-of-mass modes after every step, so its directly comparable
    prediction instead contains 3N-3 degrees of freedom.
    """

    degrees_of_freedom = thermal_degrees_of_freedom(n_particles_value)
    return 0.5 * degrees_of_freedom * (
        R * temperature_value * 1e-3
    ) ** 2


def project_canonical_energy_variance(n_particles_value, temperature_value):
    """Return the assignment formula 3N/2 * (R*T)^2 in (kJ/mol)^2."""

    return 1.5 * n_particles_value * (
        R * temperature_value * 1e-3
    ) ** 2


def production_statistics(result, start_step):
    """Calculate statistics after equilibration."""

    energy = result["energy"][start_step:]
    temperature_values = energy[:, 3]
    total_energy = energy[:, 2]
    theory = ideal_canonical_energy_variance(
        result["particle_system"].n, result["sim"].temperature
    )
    project_theory = project_canonical_energy_variance(
        result["particle_system"].n, result["sim"].temperature
    )
    measured_variance = np.var(total_energy, ddof=1)
    return {
        "mean_temperature_K": float(np.mean(temperature_values)),
        "temperature_std_K": float(np.std(temperature_values, ddof=1)),
        "mean_total_energy_kJmol": float(np.mean(total_energy)),
        "energy_variance_kJmol2": float(measured_variance),
        "theory_variance_kJmol2": float(theory),
        "variance_ratio": float(measured_variance / theory),
        "project_formula_variance_kJmol2": float(project_theory),
        "variance_ratio_project_formula": float(
            measured_variance / project_theory
        ),
    }


def aggregate_replicates(replicate_rows, statistic_names):
    """Return replicate means, standard deviations, and standard errors."""

    summary = {"n_replicates": len(replicate_rows)}
    for name in statistic_names:
        values = np.asarray([row[name] for row in replicate_rows], dtype=float)

        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            mean = standard_deviation = standard_error = np.nan
        else:
            mean = float(np.mean(finite_values))
            standard_deviation = (
                float(np.std(finite_values, ddof=1))
                if len(finite_values) > 1 else 0.0
            )
            standard_error = standard_deviation / np.sqrt(len(finite_values))
        summary[name] = mean
        summary[f"{name}_replicate_sd"] = standard_deviation
        summary[f"{name}_sem"] = standard_error
    return summary


def save_csv(filename, rows, fieldnames):
    """Write a list of dictionaries to CSV with a stable column order."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, filename, tight_crop=True):
    """Write a complete figure atomically without clipping axis labels.

    A tight crop works well for ordinary 2D plots.  Matplotlib does not always
    calculate the outer bounds of 3D labels correctly, so those figures can
    disable the tight crop and use explicit subplot margins instead.
    """

    filename = Path(filename)
    temporary_filename = filename.with_name(filename.stem + ".tmp.png")
    save_options = {
        "dpi": 300,
        "facecolor": "white",
        "pad_inches": 0.25,
    }
    if tight_crop:
        save_options["bbox_inches"] = "tight"
    figure.savefig(temporary_filename, **save_options)
    if temporary_filename.stat().st_size == 0:
        raise OSError(f"Figure output is empty: {temporary_filename}")
    temporary_filename.replace(filename)

    if show_figures:
        plt.show()


def add_figure_legend(figure, axes, ncol=3, bottom_margin=0.10,
                      top_margin=0.95, apply_tight_layout=True):
    """Place one shared legend below the axes, outside the plotted data.

    Labels are collected from every subplot and duplicates are removed while
    preserving their first appearance.  Reserving a separate strip below the
    axes prevents legends from hiding curves, histograms, or trajectories.
    """

    unique_entries = {}
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                unique_entries.setdefault(label, handle)

        existing_legend = axis.get_legend()
        if existing_legend is not None:
            existing_legend.remove()

    if unique_entries:
        figure.legend(
            unique_entries.values(),
            unique_entries.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(ncol, len(unique_entries)),
            frameon=False,
        )

    if apply_tight_layout:
        figure.tight_layout(
            rect=(0.0, bottom_margin, 1.0, top_margin)
        )


# Berendsen coupling-strength analysis
def run_coupling_analysis():
    results = {}
    rows = []

    for regime, tau in coupling_times.items():
        result = run_simulation(
            "berendsen",
            n_particles_value=n_particles,
            dt_value=dt,
            n_steps_value=coupling_steps,
            target_temperature=target_temperature,
            initial_temperature_value=initial_temperature,
            box_length_value=box_length,
            tau_value=tau,
            rij_min_value=rij_min,
            cutoff_value=cutoff,
            seed=random_seed,
        )
        results[regime] = result
        stats = production_statistics(result, coupling_steps // 2)

        equilibration_time = first_time_within_tolerance(
            result["time_ps"],
            result["energy"][:, 3],
            target_temperature,
            tolerance=0.05,
        )
        rows.append({
            "regime": regime,
            "tau_ps": tau,
            "dt_over_tau": dt / tau,
            "equilibration_time_ps_5percent": equilibration_time,
            "maximum_abs_lambda_minus_1": float(
                np.max(np.abs(result["velocity_scale"][1:] - 1.0))
            ),
            **stats,
        })

    # Temperature and Berendsen scaling for the three tau values
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"strong": "tab:red", "intermediate": "tab:blue",
              "weak": "tab:green"}
    for regime, result in results.items():
        tau = coupling_times[regime]
        axes[0].plot(
            result["time_ps"], result["energy"][:, 3],
            label=f"{regime}, tau = {tau:g} ps", color=colors[regime]
        )
        axes[1].plot(
            result["time_ps"][1:], result["velocity_scale"][1:],
            label=f"{regime}, tau = {tau:g} ps", color=colors[regime]
        )
    axes[0].axhline(target_temperature, color="black", ls="--",
                    label="target")
    axes[0].set_ylabel("temperature [K]")
    axes[1].axhline(1.0, color="black", lw=1)
    axes[1].set_ylabel("velocity scale $\\lambda$")
    axes[1].set_xlabel("time [ps]")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Effect of Berendsen coupling strength")
    add_figure_legend(
        figure, axes, ncol=4, bottom_margin=0.10, top_margin=0.94
    )
    save_figure(figure, output_directory / "temperature_coupling.png")
    plt.close(figure)

    taus = np.array([row["tau_ps"] for row in rows])
    temperature_std = np.array([row["temperature_std_K"] for row in rows])
    variance_ratio = np.array([row["variance_ratio"] for row in rows])
    equilibration = np.array([
        row["equilibration_time_ps_5percent"] for row in rows
    ])

    # Compact coupling-statistics summary
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(taus, equilibration, "o-")
    axes[0].set_ylabel("time to within 5% [ps]")
    axes[0].set_xlabel("coupling time $\\tau$ [ps]")
    axes[1].plot(taus, temperature_std, "o-")
    axes[1].set_ylabel("production $\\sigma_T$ [K]")
    axes[1].set_xlabel("coupling time $\\tau$ [ps]")
    axes[2].plot(taus, variance_ratio, "o-")
    axes[2].axhline(1.0, color="black", ls="--", label="canonical")
    axes[2].set_ylabel("measured/constrained theory variance")
    axes[2].set_xlabel("coupling time $\\tau$ [ps]")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    figure.suptitle("Berendsen coupling statistics")
    add_figure_legend(
        figure, axes, ncol=1, bottom_margin=0.15, top_margin=0.90
    )
    save_figure(figure, output_directory / "coupling_statistics.png")
    plt.close(figure)

    save_csv(
        output_directory / "coupling_summary.csv",
        rows,
        list(rows[0].keys()),
    )
    return results, rows


def unwrap_trajectory(position, box_length_value):
    """Remove periodic-boundary jumps from a single-particle trajectory."""

    displacement = np.diff(position, axis=0)

    displacement -= box_length_value * np.rint(displacement / box_length_value)

    unwrapped = np.empty_like(position)
    unwrapped[0] = position[0]
    unwrapped[1:] = position[0] + np.cumsum(displacement, axis=0)
    return unwrapped


def first_rolling_mean_within_tolerance(time_ps, values, target,
                                        tolerance=0.05,
                                        averaging_window_ps=5.0,
                                        required_duration_ps=5.0):
    """Find sustained equilibration using a rolling temperature average.

    A single noisy crossing is not counted.  The rolling average must remain
    inside the tolerance band for ``required_duration_ps``.
    """

    if len(time_ps) < 2:
        return np.nan
    time_step = time_ps[1] - time_ps[0]
    window_steps = max(1, int(round(averaging_window_ps / time_step)))
    kernel = np.ones(window_steps) / window_steps
    rolling_mean = np.convolve(values, kernel, mode="valid")
    inside = np.abs(rolling_mean - target) <= tolerance * target
    persistence_steps = max(
        1, int(round(required_duration_ps / time_step))
    )
    if len(inside) < persistence_steps:
        return np.nan
    sustained = np.convolve(
        inside.astype(int), np.ones(persistence_steps, dtype=int), mode="valid"
    ) == persistence_steps
    indices = np.flatnonzero(sustained)
    if len(indices) == 0:
        return np.nan
    return float(time_ps[indices[0] + window_steps - 1])


# Matched Berendsen-Langevin comparison
def run_thermostat_comparison():
    """Compare thermostats at all tau values using matched replicate seeds."""

    results = {}
    rows = []
    replicate_rows = []
    statistic_names = (
        "rolling_mean_equilibration_time_ps",
        "mean_temperature_K",
        "temperature_std_K",
        "mean_total_energy_kJmol",
        "energy_variance_kJmol2",
        "theory_variance_kJmol2",
        "variance_ratio",
        "project_formula_variance_kJmol2",
        "variance_ratio_project_formula",
    )

    for regime, tau in coupling_times.items():
        n_steps_value = int(round(comparison_duration_ps[tau] / dt))
        equilibration_step = int(round(comparison_equilibration_ps[tau] / dt))

        common_arguments = dict(
            n_particles_value=n_particles,
            dt_value=dt,
            n_steps_value=n_steps_value,
            target_temperature=target_temperature,
            initial_temperature_value=initial_temperature,
            box_length_value=box_length,
            tau_value=tau,
            rij_min_value=rij_min,
            cutoff_value=cutoff,
        )
        for thermostat_name in ("berendsen", "langevin"):
            case_replicates = []
            for seed in replicate_seeds:
                result = run_simulation(
                    thermostat_name, seed=seed, **common_arguments
                )
                if seed == random_seed:
                    results[(thermostat_name, tau)] = result
                replicate_row = {
                    "thermostat": thermostat_name,
                    "regime": regime,
                    "tau_ps": tau,
                    "seed": seed,
                    "rolling_mean_equilibration_time_ps": (
                        first_rolling_mean_within_tolerance(
                            result["time_ps"], result["energy"][:, 3],
                            target_temperature, tolerance=0.05,
                            averaging_window_ps=5.0,
                            required_duration_ps=5.0,
                        )
                    ),
                    **production_statistics(result, equilibration_step),
                }
                replicate_rows.append(replicate_row)
                case_replicates.append(replicate_row)

            rows.append({
                "thermostat": thermostat_name,
                "regime": regime,
                "tau_ps": tau,
                "dt_ps": dt,
                "dt_over_tau": dt / tau,
                "duration_ps": comparison_duration_ps[tau],
                "production_start_ps": comparison_equilibration_ps[tau],
                **aggregate_replicates(case_replicates, statistic_names),
            })

    save_csv(
        output_directory / "thermostat_summary.csv",
        rows,
        list(rows[0].keys()),
    )
    save_csv(
        output_directory / "thermostat_replicates.csv",
        replicate_rows,
        list(replicate_rows[0].keys()),
    )

    # Instantaneous temperature and running average
    figure, axes = plt.subplots(3, 2, figsize=(13, 11))
    for row_index, (regime, tau) in enumerate(coupling_times.items()):
        for thermostat_name in thermostat_plot_order:
            result = results[(thermostat_name, tau)]
            time_ps = result["time_ps"]
            temperature_values = result["energy"][:, 3]
            axes[row_index, 0].plot(
                time_ps, temperature_values,
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
            running_mean = np.cumsum(temperature_values) / np.arange(
                1, len(time_ps) + 1
            )
            axes[row_index, 1].plot(
                time_ps, running_mean,
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
        for column_index in range(2):
            axes[row_index, column_index].axhline(
                target_temperature, color="black", linestyle=":",
                linewidth=1.0, zorder=4, label="target"
            )
            axes[row_index, column_index].grid(alpha=0.25)
            axes[row_index, column_index].set_xlabel("time [ps]")
        axes[row_index, 0].set_ylabel("instantaneous T [K]")
        axes[row_index, 1].set_ylabel("running mean T [K]")
        axes[row_index, 0].set_title(
            f"{regime.capitalize()} coupling: tau = {tau:g} ps"
        )
        axes[row_index, 1].set_title(
            f"Cumulative mean: tau = {tau:g} ps"
        )
    figure.suptitle("Temperature control from the same 100 K initial state")
    add_figure_legend(
        figure, axes, ncol=3, bottom_margin=0.08, top_margin=0.96
    )
    save_figure(
        figure, output_directory / "thermostat_temperature_comparison.png"
    )
    plt.close(figure)

    # Thermostat statistics with replicate error bars
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for thermostat_name in thermostat_plot_order:
        thermostat_rows = sorted(
            (row for row in rows if row["thermostat"] == thermostat_name),
            key=lambda row: row["tau_ps"],
        )
        taus = [row["tau_ps"] for row in thermostat_rows]
        axes[0].errorbar(
            taus,
            [row["rolling_mean_equilibration_time_ps"]
             for row in thermostat_rows],
            yerr=[row["rolling_mean_equilibration_time_ps_sem"]
                  for row in thermostat_rows],
            marker="o", label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
        axes[1].errorbar(
            taus, [row["temperature_std_K"] for row in thermostat_rows],
            yerr=[row["temperature_std_K_sem"] for row in thermostat_rows],
            marker="o", label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
        axes[2].errorbar(
            taus, [row["variance_ratio"] for row in thermostat_rows],
            yerr=[row["variance_ratio_sem"] for row in thermostat_rows],
            marker="o", label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
    axes[0].set_ylabel("rolling-mean equilibration [ps]")
    axes[1].set_ylabel("production temperature std [K]")
    axes[2].set_ylabel("measured/constrained theory variance")
    axes[2].axhline(1.0, color="black", ls="--", label="canonical")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("coupling time tau [ps]")
        axis.grid(alpha=0.25)
    figure.suptitle("Thermostat statistics at matched coupling-time values")
    add_figure_legend(
        figure, axes, ncol=3, bottom_margin=0.14, top_margin=0.90
    )
    save_figure(
        figure, output_directory / "thermostat_statistics_comparison.png"
    )
    plt.close(figure)

    # Representative particle path and speed
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    final_index = int(round(50.0 / dt)) + 1
    for column_index, (regime, tau) in enumerate(coupling_times.items()):
        for thermostat_name in thermostat_plot_order:
            result = results[(thermostat_name, tau)]
            path = unwrap_trajectory(result["selected_position"], box_length)
            path = path[:final_index] - path[0]
            axes[0, column_index].plot(
                path[:, 0], path[:, 1],
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
            speed = np.linalg.norm(result["selected_velocity"], axis=1)
            axes[1, column_index].plot(
                result["time_ps"][:final_index], speed[:final_index],
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
        axes[0, column_index].set_title(
            f"{regime.capitalize()}: tau = {tau:g} ps"
        )
        axes[0, column_index].set_xlabel("particle 0 displacement x [nm]")
        axes[0, column_index].set_ylabel("particle 0 displacement y [nm]")
        axes[0, column_index].set_aspect("equal", adjustable="datalim")
        axes[1, column_index].set_xlabel("time [ps]")
        axes[1, column_index].set_ylabel("particle 0 speed [nm/ps]")
        for row_index in range(2):
            axes[row_index, column_index].grid(alpha=0.25)
    figure.suptitle("Representative particle motion during the first 50 ps")
    add_figure_legend(
        figure, axes, ncol=2, bottom_margin=0.09, top_margin=0.94
    )
    save_figure(figure, output_directory / "representative_trajectories.png")
    plt.close(figure)

    # Final position and velocity distributions
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    velocity_grid = np.linspace(-0.9, 0.9, 400)
    component_std = np.sqrt(
        (1.0 - 1.0 / n_particles)
        * R * target_temperature / (mass_argon * 1e3)
    )
    for column_index, (regime, tau) in enumerate(coupling_times.items()):
        for thermostat_name in thermostat_plot_order:
            final_ps = results[(thermostat_name, tau)]["particle_system"]
            axes[0, column_index].hist(
                final_ps.position[:, 0], bins=20, density=True,
                histtype="step", label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
            axes[1, column_index].hist(
                final_ps.velocity[:, 0], bins=20, density=True,
                histtype="step", label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
        axes[0, column_index].axhline(
            1.0 / box_length, color="black", ls="--", label="uniform"
        )
        axes[1, column_index].plot(
            velocity_grid, norm.pdf(velocity_grid, 0.0, component_std),
            color="black", ls="--", label="Maxwell component",
        )
        axes[0, column_index].set_title(
            f"{regime.capitalize()}: tau = {tau:g} ps"
        )
        axes[0, column_index].set_xlabel("final x position [nm]")
        axes[0, column_index].set_ylabel("probability density [1/nm]")
        axes[1, column_index].set_xlabel("final x velocity [nm/ps]")
        axes[1, column_index].set_ylabel("probability density [ps/nm]")
        for row_index in range(2):
            axes[row_index, column_index].grid(alpha=0.25)
    figure.suptitle("Final particle position and velocity distributions")
    add_figure_legend(
        figure, axes, ncol=4, bottom_margin=0.09, top_margin=0.94
    )
    save_figure(figure, output_directory / "position_velocity_distributions.png")
    plt.close(figure)

    degrees_of_freedom = thermal_degrees_of_freedom(n_particles)
    canonical_shape = 0.5 * degrees_of_freedom
    canonical_scale = R * target_temperature * 1e-3
    theory_variance = canonical_shape * canonical_scale ** 2
    theory_std = np.sqrt(theory_variance)
    # Total-energy distributions after equilibration
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for column_index, (regime, tau) in enumerate(coupling_times.items()):
        equilibration_step = int(round(comparison_equilibration_ps[tau] / dt))
        berendsen = results[("berendsen", tau)]
        langevin = results[("langevin", tau)]
        berendsen_energy = berendsen["energy"][equilibration_step:, 2]
        langevin_energy = langevin["energy"][equilibration_step:, 2]
        potential_offset = np.mean(
            langevin["energy"][equilibration_step:, 0]
        )
        theory_mean = potential_offset + canonical_shape * canonical_scale
        energy_grid = np.linspace(
            theory_mean - 4.0 * theory_std,
            theory_mean + 4.0 * theory_std,
            500,
        )
        common_range = (energy_grid[0], energy_grid[-1])
        energy_by_thermostat = {
            "berendsen": berendsen_energy,
            "langevin": langevin_energy,
        }
        for thermostat_name in thermostat_plot_order:
            axes[0, column_index].hist(
                energy_by_thermostat[thermostat_name],
                bins=60, range=common_range, density=True,
                histtype="step", label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
        canonical_density = gamma.pdf(
            energy_grid - potential_offset,
            a=canonical_shape,
            scale=canonical_scale,
        )
        axes[0, column_index].plot(
            energy_grid, canonical_density,
            color="black", ls="--",
            label="canonical theory (3N-3)",
        )
        axes[0, column_index].set_title(
            f"{regime.capitalize()}: tau = {tau:g} ps"
        )
        axes[1, column_index].hist(
            langevin_energy, bins=50, density=True, histtype="stepfilled",
            label="Langevin", **thermostat_styles["langevin"],
        )
        axes[1, column_index].plot(
            energy_grid, canonical_density,
            color="black", ls="--",
            label="canonical theory (3N-3)",
        )
        axes[1, column_index].set_title("Langevin detail")
        for row_index in range(2):
            axes[row_index, column_index].set_xlabel("total energy [kJ/mol]")
            axes[row_index, column_index].set_ylabel("probability density")
            axes[row_index, column_index].grid(alpha=0.25)
    figure.suptitle("Equilibrium total-energy distribution")
    add_figure_legend(
        figure, axes, ncol=3, bottom_margin=0.09, top_margin=0.94
    )
    save_figure(figure, output_directory / "energy_distribution.png")
    plt.close(figure)

    # Numerical data behind the comparison plots
    comparison_arrays = {}
    for (thermostat_name, tau), result in results.items():
        tau_label = str(tau).replace(".", "p")
        prefix = f"{thermostat_name}_tau_{tau_label}"
        comparison_arrays[f"{prefix}_time_ps"] = result["time_ps"]
        comparison_arrays[f"{prefix}_energy"] = result["energy"]
        comparison_arrays[f"{prefix}_particle_position"] = (
            result["selected_position"]
        )
        comparison_arrays[f"{prefix}_particle_velocity"] = (
            result["selected_velocity"]
        )
        comparison_arrays[f"{prefix}_final_positions"] = (
            result["particle_system"].position
        )
        comparison_arrays[f"{prefix}_final_velocities"] = (
            result["particle_system"].velocity
        )
    # Store all comparison arrays in one compressed file
    np.savez_compressed(
        output_directory / "thermostat_comparison_data.npz",
        **comparison_arrays,
    )
    return results, rows


def run_matched_full_trajectory_pair(box_length_value, duration_ps):
    """Run both thermostats from the same initial state and save all atoms."""

    n_steps_value = int(round(duration_ps / dt))
    common_arguments = dict(
        n_particles_value=n_particles,
        dt_value=dt,
        n_steps_value=n_steps_value,
        target_temperature=target_temperature,
        initial_temperature_value=initial_temperature,
        box_length_value=box_length_value,
        tau_value=matched_trajectory_tau,
        rij_min_value=rij_min,
        cutoff_value=cutoff,
        seed=random_seed,
        store_full_trajectory=True,
        trajectory_stride=trajectory_stride,
    )
    results = {
        thermostat_name: run_simulation(thermostat_name, **common_arguments)
        for thermostat_name in ("berendsen", "langevin")
    }

    # Matched starts make the later trajectory differences meaningful.
    np.testing.assert_allclose(
        results["berendsen"]["position_trajectory"][0],
        results["langevin"]["position_trajectory"][0],
    )
    np.testing.assert_allclose(
        results["berendsen"]["velocity_trajectory"][0],
        results["langevin"]["velocity_trajectory"][0],
    )
    return results


# Full tau=1 ps trajectories for VMD
def export_matched_full_trajectories():
    """Export matched Berendsen/Langevin positions and velocities at tau=1 ps.

    XYZ stores the positions for VMD.  The full velocity vectors are stored in
    one compressed NumPy file because the XYZ format has no velocity fields.
    Both simulations start from the same seeded particle positions and
    velocities; the Langevin trajectory subsequently diverges because of its
    random thermal kicks.
    """

    trajectory_output_directory.mkdir(parents=True, exist_ok=True)

    results = run_matched_full_trajectory_pair(
        box_length, matched_trajectory_duration
    )

    berendsen = results["berendsen"]
    langevin = results["langevin"]

    for thermostat_name, result in results.items():
        unwrapped_positions = unwrap_trajectory(
            result["position_trajectory"], box_length
        )
        write_xyz_trajectory(
            trajectory_output_directory
            / f"{thermostat_name}_single_run_100K_positions.xyz",
            unwrapped_positions,
        )

    # XYZ stores positions; NPZ stores positions, velocities, and metadata
    np.savez_compressed(
        trajectory_output_directory
        / "matched_tau_1ps_trajectories.npz",
        time_ps=berendsen["saved_steps"] * dt,
        box_length_nm=box_length,
        tau_ps=matched_trajectory_tau,
        initial_temperature_K=initial_temperature,
        target_temperature_K=target_temperature,
        berendsen_position_nm=berendsen["position_trajectory"],
        langevin_position_nm=langevin["position_trajectory"],
        berendsen_velocity_nm_per_ps=berendsen["velocity_trajectory"],
        langevin_velocity_nm_per_ps=langevin["velocity_trajectory"],
    )

    component_names = ("x", "y", "z")
    # x, y, and z trajectory components for the representative atom
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for thermostat_name in thermostat_plot_order:
        result = results[thermostat_name]
        time_ps = result["time_ps"]
        unwrapped = unwrap_trajectory(
            result["selected_position"], box_length
        )
        displacement = unwrapped - unwrapped[0]
        velocity = result["selected_velocity"]

        for component_index, component_name in enumerate(component_names):
            axes[0, component_index].plot(
                time_ps, displacement[:, component_index],
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
            axes[1, component_index].plot(
                time_ps, velocity[:, component_index],
                label=thermostat_name.capitalize(),
                **thermostat_styles[thermostat_name],
            )
            axes[0, component_index].set_title(
                f"{component_name}-component"
            )

    for component_index in range(3):
        axes[0, component_index].set_ylabel(
            f"particle 0 displacement {component_names[component_index]} [nm]"
        )
        axes[1, component_index].set_ylabel(
            f"particle 0 velocity {component_names[component_index]} [nm/ps]"
        )
        axes[1, component_index].set_xlabel("time [ps]")
        for row_index in range(2):
            axes[row_index, component_index].grid(alpha=0.25)
    figure.suptitle(
        "Matched Berendsen-Langevin trajectory components: "
        "tau = 1 ps, same 100 K initial state"
    )
    add_figure_legend(
        figure, axes, ncol=2, bottom_margin=0.09, top_margin=0.92
    )
    save_figure(
        figure,
        trajectory_output_directory
        / "trajectory_components_comparison_tau_1ps.png",
    )
    plt.close(figure)
    return results


def export_interacting_full_trajectories():
    """Add an 8 nm trajectory comparison without changing the main analysis.

    The smaller periodic box produces many more Lennard-Jones encounters than
    the 100 nm ideal-gas box.  These trajectories are therefore useful for VMD
    and qualitative discussion, but they are not used for the ideal-gas energy
    variance comparison.
    """

    interacting_trajectory_output_directory.mkdir(parents=True, exist_ok=True)
    results = run_matched_full_trajectory_pair(
        interacting_box_length, interacting_trajectory_duration
    )
    berendsen = results["berendsen"]

    for thermostat_name, result in results.items():
        # Keep atoms inside the periodic box for a compact VMD view.
        write_xyz_trajectory(
            interacting_trajectory_output_directory
            / f"{thermostat_name}_interacting_8nm_wrapped_positions.xyz",
            result["position_trajectory"],
        )

    np.savez_compressed(
        interacting_trajectory_output_directory
        / "matched_interacting_8nm_tau_1ps_trajectories.npz",
        time_ps=berendsen["saved_steps"] * dt,
        box_length_nm=interacting_box_length,
        tau_ps=matched_trajectory_tau,
        initial_temperature_K=initial_temperature,
        target_temperature_K=target_temperature,
        berendsen_position_nm=results["berendsen"]["position_trajectory"],
        langevin_position_nm=results["langevin"]["position_trajectory"],
        berendsen_velocity_nm_per_ps=(
            results["berendsen"]["velocity_trajectory"]
        ),
        langevin_velocity_nm_per_ps=(
            results["langevin"]["velocity_trajectory"]
        ),
    )

    # Static 3D view for presenting the result without VMD.
    plot_end = int(round(interacting_plot_duration / dt)) + 1
    paths = {}
    for thermostat_name, result in results.items():
        path = unwrap_trajectory(
            result["selected_position"][:plot_end], interacting_box_length
        )
        paths[thermostat_name] = path - path[0]
    all_coordinates = np.concatenate(list(paths.values()), axis=0)
    lower_limit = float(np.min(all_coordinates))
    upper_limit = float(np.max(all_coordinates))
    padding = max(0.5, 0.05 * (upper_limit - lower_limit))

    figure = plt.figure(figsize=(13, 6))
    for panel, thermostat_name in enumerate(thermostat_plot_order, start=1):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        path = paths[thermostat_name]
        axis.plot(
            path[:, 0], path[:, 1], path[:, 2],
            label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
        axis.scatter(*path[0], color="green", s=35, label="start")
        axis.scatter(*path[-1], color="black", s=35, label="end")
        axis.set(
            xlim=(lower_limit - padding, upper_limit + padding),
            ylim=(lower_limit - padding, upper_limit + padding),
            zlim=(lower_limit - padding, upper_limit + padding),
            title=thermostat_name.capitalize(),
        )
        # Set the three labels separately so that label padding can be added.
        # This keeps the text clear of the tick labels and inside the canvas.
        axis.set_xlabel("displacement x [nm]", labelpad=12)
        axis.set_ylabel("displacement y [nm]", labelpad=12)
        axis.set_zlabel("displacement z [nm]", labelpad=12)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-58)
    figure.suptitle(
        "Representative 3D trajectories in the interacting 8 nm box "
        f"(first {interacting_plot_duration:g} ps)"
    )
    add_figure_legend(
        figure, figure.axes, ncol=4, apply_tight_layout=False
    )
    # tight_layout/bbox_inches='tight' can cut off labels on 3D axes because
    # Matplotlib does not always report their full rotated bounding boxes.
    # Fixed canvas margins are therefore safer for this particular figure.
    figure.subplots_adjust(
        left=0.03, right=0.84, bottom=0.20, top=0.84, wspace=0.12
    )
    save_figure(
        figure,
        interacting_trajectory_output_directory
        / "interacting_8nm_3D_trajectory_comparison.png",
        tight_crop=False,
    )
    plt.close(figure)

    # Non-zero potential energy demonstrates that particles are interacting.
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    summary_rows = []
    for thermostat_name in thermostat_plot_order:
        result = results[thermostat_name]
        potential = result["energy"][:, 0]
        temperature_values = result["energy"][:, 3]
        axes[0].plot(
            result["time_ps"], potential,
            label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
        axes[1].plot(
            result["time_ps"], temperature_values,
            label=thermostat_name.capitalize(),
            **thermostat_styles[thermostat_name],
        )
        production_start = int(round(50.0 / dt))
        summary_rows.append({
            "thermostat": thermostat_name,
            "box_length_nm": interacting_box_length,
            "tau_ps": matched_trajectory_tau,
            "potential_energy_range_kJmol": float(np.ptp(potential)),
            "potential_energy_std_kJmol": float(np.std(potential, ddof=1)),
            "production_mean_temperature_K": float(
                np.mean(temperature_values[production_start:])
            ),
            "production_temperature_std_K": float(
                np.std(temperature_values[production_start:], ddof=1)
            ),
        })
    axes[0].set_ylabel("potential energy [kJ/mol]")
    axes[1].set_ylabel("temperature [K]")
    axes[1].set_xlabel("time [ps]")
    axes[1].axhline(
        target_temperature, color="black", ls=":", label="target"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Interacting 8 nm supplement (not used for ideal-gas variance)"
    )
    add_figure_legend(
        figure, axes, ncol=3, bottom_margin=0.10, top_margin=0.94
    )
    save_figure(
        figure,
        interacting_trajectory_output_directory
        / "interacting_8nm_energy_temperature_comparison.png",
    )
    plt.close(figure)
    save_csv(
        interacting_trajectory_output_directory
        / "interacting_8nm_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    return results, summary_rows


# Interacting NVE timestep validation
def run_nve_validation():
    results = {}
    rows = []

    for dt_value in nve_time_steps:
        steps = int(round(nve_duration / dt_value))
        result = run_simulation(
            "nve",
            n_particles_value=nve_particles,
            dt_value=dt_value,
            n_steps_value=steps,
            target_temperature=target_temperature,
            initial_temperature_value=target_temperature,
            box_length_value=nve_box_length,
            tau_value=None,
            rij_min_value=rij_min,
            cutoff_value=cutoff,
            seed=random_seed,
        )
        results[dt_value] = result
        total_energy = result["energy"][:, 2]
        relative_error = (total_energy - total_energy[0]) / abs(total_energy[0])
        rows.append({
            "dt_ps": dt_value,
            "n_steps": steps,
            "duration_ps": nve_duration,
            "box_length_nm": nve_box_length,
            "reduced_number_density": (
                nve_particles * sigma_argon ** 3 / nve_box_length ** 3
            ),
            "initial_total_energy_kJmol": total_energy[0],
            "potential_energy_range_kJmol": float(
                np.ptp(result["energy"][:, 0])
            ),
            "maximum_relative_energy_error": float(
                np.max(np.abs(relative_error))
            ),
            "final_relative_energy_error": float(relative_error[-1]),
            "relative_energy_std": float(
                np.std(total_energy, ddof=1) / abs(np.mean(total_energy))
            ),
        })

    # Energy error versus time and versus timestep
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for dt_value, result in results.items():
        total_energy = result["energy"][:, 2]
        relative_error = (total_energy - total_energy[0]) / abs(total_energy[0])
        axes[0].plot(result["time_ps"], relative_error,
                     label=f"dt = {dt_value:g} ps")
    axes[0].set_xlabel("time [ps]")
    axes[0].set_ylabel("relative total-energy error")

    dt_values = np.array([row["dt_ps"] for row in rows])
    maximum_error = np.array([
        row["maximum_relative_energy_error"] for row in rows
    ])
    axes[1].loglog(dt_values, maximum_error, "o-")
    axes[1].set_xlabel("time step [ps]")
    axes[1].set_ylabel("maximum relative energy error")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("NVE time-step validation with Velocity Verlet")
    add_figure_legend(
        figure, axes, ncol=5, bottom_margin=0.13, top_margin=0.90
    )
    save_figure(figure, output_directory / "nve_timestep_validation.png")
    plt.close(figure)

    save_csv(
        output_directory / "nve_timestep_summary.csv",
        rows,
        list(rows[0].keys()),
    )
    return results, rows


def main():
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Running Berendsen coupling-strength analysis...")
    run_coupling_analysis()
    print("Running Berendsen-Langevin comparison...")
    run_thermostat_comparison()
    print("Exporting matched Berendsen-Langevin full trajectories...")
    export_matched_full_trajectories()
    print("Exporting supplementary interacting 8 nm trajectories...")
    export_interacting_full_trajectories()
    print("Running NVE time-step validation...")
    run_nve_validation()
    print(f"\nResults written to {output_directory.resolve()}")
    print(f"VMD trajectories written to {trajectory_output_directory.resolve()}")


if __name__ == "__main__":
    main()
