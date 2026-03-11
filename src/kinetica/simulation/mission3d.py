from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kinetica.models.rocket import Rocket
from kinetica.models.mission import (
    MissionPlan,
    BurnCommand,
    TargetOrbitCommand,
    SOIChangeCommand,
)
from kinetica.models.celestial_body import CelestialBody


G0 = 9.80665


@dataclass
class SimulationResult3D:
    times: np.ndarray
    states: np.ndarray
    dominant_bodies: list[str]
    phase_names: list[str]
    events: list[dict]


@dataclass
class VehicleState:
    active_stage_index: Optional[int]
    propellant_left: list[float]
    dropped: list[bool]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def altitude_relative_to_body(
    t: float,
    position_root: np.ndarray,
    body: CelestialBody,
) -> float:
    body_pos, _ = body.position_velocity_in_root_frame(t)
    return norm(position_root - body_pos) - body.radius


def has_impacted_body(
    t: float,
    position_root: np.ndarray,
    body: CelestialBody,
    tolerance_m: float = 1.0,
) -> bool:
    return altitude_relative_to_body(t, position_root, body) < -tolerance_m


def interpolate_impact_state(
    t: float,
    state_before: np.ndarray,
    state_after: np.ndarray,
    body: CelestialBody,
    h: float,
) -> tuple[float, np.ndarray]:
    alt_before = altitude_relative_to_body(t, state_before[:3], body)
    alt_after = altitude_relative_to_body(t + h, state_after[:3], body)

    if abs(alt_before - alt_after) < 1e-12:
        alpha = 1.0
    else:
        alpha = alt_before / (alt_before - alt_after)

    alpha = max(0.0, min(1.0, alpha))

    impact_state = state_before + alpha * (state_after - state_before)
    impact_time = t + alpha * h

    # Ajuste final para dejar el punto exactamente en la superficie
    body_pos, _ = body.position_velocity_in_root_frame(impact_time)
    rel = impact_state[:3] - body_pos
    rel_norm = norm(rel)
    if rel_norm > 1e-12:
        impact_state = impact_state.copy()
        impact_state[:3] = body_pos + (body.radius / rel_norm) * rel

    return impact_time, impact_state


def append_impact_and_stop(
    *,
    t: float,
    h: float,
    state: np.ndarray,
    next_state: np.ndarray,
    dominant_body: CelestialBody,
    times: list[float],
    states: list[np.ndarray],
    bodies_list: list[str],
    events: list[dict],
) -> tuple[float, np.ndarray, bool]:
    if not has_impacted_body(t + h, next_state[:3], dominant_body):
        return t, next_state, False

    impact_time, impact_state = interpolate_impact_state(
        t=t,
        state_before=state,
        state_after=next_state,
        body=dominant_body,
        h=h,
    )

    events.append({
        "type": "IMPACT",
        "time": impact_time,
        "body": dominant_body.name,
        "altitude_m": 0.0,
    })

    times.append(impact_time)
    states.append(impact_state.copy())
    bodies_list.append(dominant_body.name)

    return impact_time, impact_state, True


def body_fixed_position_from_lat_lon_radius(
    latitude_deg: float,
    longitude_deg: float,
    radius_m: float,
) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)

    x = radius_m * math.cos(lat) * math.cos(lon)
    y = radius_m * math.cos(lat) * math.sin(lon)
    z = radius_m * math.sin(lat)

    return np.array([x, y, z], dtype=float)


def stage_reference_area(stage) -> float:
    diameter = float(getattr(stage, "diameter", 0.0))
    if diameter <= 0.0:
        return 0.0
    radius = 0.5 * diameter
    return math.pi * radius * radius


def orbital_elements_relative(mu: float, r_vec: np.ndarray, v_vec: np.ndarray) -> dict:
    r = norm(r_vec)
    v = norm(v_vec)

    if r < 1e-9:
        return {"a": None, "e": None, "rp": None, "ra": None, "energy": None}

    h_vec = np.cross(r_vec, v_vec)
    h = norm(h_vec)
    energy = 0.5 * v * v - mu / r

    e_vec = ((v * v - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = norm(e_vec)

    if abs(energy) < 1e-12 or energy >= 0.0:
        a = None
        rp = None
        ra = None
    else:
        a = -mu / (2.0 * energy)
        rp = a * (1.0 - e)
        ra = a * (1.0 + e)

    return {
        "a": a,
        "e": e,
        "rp": rp,
        "ra": ra,
        "energy": energy,
        "h": h,
    }


def body_chain_to_root(body: CelestialBody) -> list[CelestialBody]:
    chain = [body]
    while chain[-1].parent is not None:
        chain.append(chain[-1].parent)
    return chain


def dominant_body_for_position(t: float, position_root: np.ndarray, bodies: dict[str, CelestialBody]) -> CelestialBody:
    all_bodies = list(bodies.values())
    all_bodies.sort(key=lambda b: len(body_chain_to_root(b)), reverse=True)

    for body in all_bodies:
        body_pos, _ = body.position_velocity_in_root_frame(t)
        d = norm(position_root - body_pos)
        if d <= body.soi_radius:
            return body

    roots = [b for b in all_bodies if b.parent is None]
    return roots[0]


def compute_nbody_acceleration(t: float, position_root: np.ndarray, bodies: dict[str, CelestialBody]) -> np.ndarray:
    a = np.zeros(3)
    for body in bodies.values():
        body_pos, _ = body.position_velocity_in_root_frame(t)
        rel = position_root - body_pos
        d = norm(rel)
        if d > 1.0:
            a += -body.mu * rel / (d ** 3)
    return a


def launch_site_velocity_due_to_rotation(
    t0: float,
    central_body: CelestialBody,
    launch_lat_deg: float,
    launch_lon_deg: float,
    launch_alt_m: float,
) -> np.ndarray:
    r_site_body = body_fixed_position_from_lat_lon_radius(
        latitude_deg=launch_lat_deg,
        longitude_deg=launch_lon_deg,
        radius_m=central_body.radius + launch_alt_m,
    )

    rot_body_to_root = central_body.rotation_matrix_body_to_root(t0)
    r_site_root_rel = rot_body_to_root @ r_site_body
    omega_root = central_body.angular_velocity_vector_root(t0)

    return np.cross(omega_root, r_site_root_rel)


def launch_site_speed_due_to_rotation(
    t0: float,
    central_body: CelestialBody,
    launch_lat_deg: float,
    launch_lon_deg: float,
    launch_alt_m: float,
) -> float:
    return norm(
        launch_site_velocity_due_to_rotation(
            t0=t0,
            central_body=central_body,
            launch_lat_deg=launch_lat_deg,
            launch_lon_deg=launch_lon_deg,
            launch_alt_m=launch_alt_m,
        )
    )


def launch_state_from_site(
    t0: float,
    central_body: CelestialBody,
    launch_lat_deg: float,
    launch_lon_deg: float,
    launch_alt_m: float,
    launch_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    surface_clearance_m = 1.0

    r_site_body = body_fixed_position_from_lat_lon_radius(
        latitude_deg=launch_lat_deg,
        longitude_deg=launch_lon_deg,
        radius_m=central_body.radius + launch_alt_m + surface_clearance_m,
    )

    body_pos_root, body_vel_root = central_body.position_velocity_in_root_frame(t0)
    rot_body_to_root = central_body.rotation_matrix_body_to_root(t0)

    r_site_root_rel = rot_body_to_root @ r_site_body
    position_root = body_pos_root + r_site_root_rel

    up_body = unit(r_site_body)
    lon = math.radians(launch_lon_deg)

    east_body = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    east_body = unit(east_body)
    north_body = unit(np.cross(up_body, east_body))

    east_root = unit(rot_body_to_root @ east_body)
    north_root = unit(rot_body_to_root @ north_body)

    az = math.radians(launch_azimuth_deg)
    launch_dir_root = unit(math.cos(az) * north_root + math.sin(az) * east_root)

    omega_root = central_body.angular_velocity_vector_root(t0)
    v_site_rotation_root = np.cross(omega_root, r_site_root_rel)

    velocity_root = body_vel_root + v_site_rotation_root + 1.0 * launch_dir_root
    return position_root, velocity_root

def burn_direction(
    cmd: BurnCommand,
    position_root: np.ndarray,
    velocity_root: np.ndarray,
    dominant_body: CelestialBody,
    t: float,
) -> np.ndarray:
    body_pos, body_vel = dominant_body.position_velocity_in_root_frame(t)

    r_rel = position_root - body_pos
    v_rel = velocity_root - body_vel

    r_hat = unit(r_rel)
    v_hat = unit(v_rel)

    h_hat = unit(np.cross(r_rel, v_rel))
    if norm(h_hat) < 1e-12:
        trial = np.cross(r_hat, np.array([0.0, 0.0, 1.0]))
        if norm(trial) < 1e-12:
            trial = np.array([1.0, 0.0, 0.0])
        h_hat = unit(trial)

    if cmd.direction_mode == "prograde":
        return v_hat if norm(v_hat) > 0 else np.array([1.0, 0.0, 0.0])
    if cmd.direction_mode == "retrograde":
        return -v_hat if norm(v_hat) > 0 else np.array([-1.0, 0.0, 0.0])
    if cmd.direction_mode == "radial_out":
        return r_hat
    if cmd.direction_mode == "radial_in":
        return -r_hat
    if cmd.direction_mode == "normal":
        return h_hat
    if cmd.direction_mode == "antinormal":
        return -h_hat
    if cmd.direction_mode == "fixed" and cmd.fixed_direction_eci is not None:
        return unit(np.asarray(cmd.fixed_direction_eci, dtype=float))

    return v_hat if norm(v_hat) > 0 else np.array([1.0, 0.0, 0.0])


def init_vehicle_state(rocket: Rocket) -> VehicleState:
    return VehicleState(
        active_stage_index=0 if len(rocket.stages) > 0 else None,
        propellant_left=[float(s.propellant_mass) for s in rocket.stages],
        dropped=[False for _ in rocket.stages],
    )


def sync_active_stage(vehicle: VehicleState, rocket: Rocket) -> None:
    if vehicle.active_stage_index is None:
        return

    n = len(rocket.stages)
    idx = vehicle.active_stage_index
    while idx < n and vehicle.dropped[idx]:
        idx += 1

    vehicle.active_stage_index = None if idx >= n else idx


def current_stage(vehicle: VehicleState, rocket: Rocket):
    sync_active_stage(vehicle, rocket)
    if vehicle.active_stage_index is None:
        return None
    return rocket.stages[vehicle.active_stage_index]


def current_aero_stage(vehicle: VehicleState, rocket: Rocket):
    for i, stage in enumerate(rocket.stages):
        if not vehicle.dropped[i]:
            return stage
    return None


def dry_mass_floor(rocket: Rocket, vehicle: VehicleState) -> float:
    remaining_dry = 0.0
    for i, s in enumerate(rocket.stages):
        if not vehicle.dropped[i]:
            remaining_dry += float(s.dry_mass)
    return float(rocket.payload_mass + remaining_dry)


def current_stage_thrust_isp(
    rocket: Rocket,
    vehicle: VehicleState,
    thrust_override: Optional[float],
    isp_override: Optional[float],
) -> tuple[float, float]:
    stg = current_stage(vehicle, rocket)
    if stg is None:
        return 0.0, 0.0

    thrust_n = float(thrust_override) if thrust_override is not None else float(stg.thrust_vac)
    isp_s = float(isp_override) if isp_override is not None else float(stg.isp_vac)
    return thrust_n, isp_s


def separate_empty_stage_if_needed(
    rocket: Rocket,
    vehicle: VehicleState,
    state: np.ndarray,
    t: float,
    events: list[dict],
) -> np.ndarray:
    idx = vehicle.active_stage_index
    if idx is None:
        print("DEBUG separate_empty_stage_if_needed -> idx is None at t =", t)
        return state

    print(
        "DEBUG separate_empty_stage_if_needed:",
        "t =", t,
        "idx =", idx,
        "propellant_left =", vehicle.propellant_left[idx],
        "dropped =", vehicle.dropped[idx],
        "mass_before =", state[6],
    )

    if vehicle.propellant_left[idx] > 1e-9:
        print("DEBUG -> stage still has propellant, not separating")
        return state

    if not vehicle.dropped[idx]:
        stage = rocket.stages[idx]
        state = state.copy()
        state[6] -= float(stage.dry_mass)
        vehicle.dropped[idx] = True
        events.append({"type": "STAGE_SEPARATION", "time": t, "stage_index": idx})
        state[6] = max(state[6], dry_mass_floor(rocket, vehicle))

        print(
            "DEBUG -> stage separated:",
            "stage_index =", idx,
            "dry_mass =", stage.dry_mass,
            "mass_after =", state[6],
        )

    sync_active_stage(vehicle, rocket)
    print("DEBUG -> new active_stage_index =", vehicle.active_stage_index)
    return state

def atmospheric_properties(
    t: float,
    position_root: np.ndarray,
    velocity_root: np.ndarray,
    body: CelestialBody,
) -> tuple[float, np.ndarray, float]:
    body_pos, body_vel = body.position_velocity_in_root_frame(t)

    r_rel = position_root - body_pos
    altitude = norm(r_rel) - body.radius

    rho = body.atmospheric_density(max(0.0, altitude))

    omega_root = body.angular_velocity_vector_root(t)
    v_air_root = body_vel + np.cross(omega_root, r_rel)

    v_rel_air = velocity_root - v_air_root
    return rho, v_rel_air, altitude


def aerodynamic_drag_acceleration(
    t: float,
    state: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    body: CelestialBody,
) -> np.ndarray:
    pos = state[:3]
    vel = state[3:6]
    mass = state[6]

    if mass <= 0.0:
        return np.zeros(3)

    aero_stage = current_aero_stage(vehicle, rocket)
    if aero_stage is None:
        return np.zeros(3)

    area = stage_reference_area(aero_stage)
    if area <= 0.0:
        return np.zeros(3)

    rho, v_rel_air, altitude = atmospheric_properties(
        t=t,
        position_root=pos,
        velocity_root=vel,
        body=body,
    )

    if altitude <= 0.0:
        return np.zeros(3)

    v_rel = norm(v_rel_air)
    if rho <= 0.0 or v_rel <= 1e-9:
        return np.zeros(3)

    cd = float(rocket.cd)
    drag_mag = 0.5 * rho * cd * area * v_rel * v_rel
    return -(drag_mag / mass) * unit(v_rel_air)


def acceleration_with_thrust(
    t: float,
    state: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dominant_body: CelestialBody,
    thrust_n: float,
    thrust_dir: np.ndarray,
) -> np.ndarray:
    pos = state[:3]
    vel = state[3:6]
    m = state[6]

    a_grav = compute_nbody_acceleration(t, pos, bodies)
    a_drag = aerodynamic_drag_acceleration(
        t=t,
        state=state,
        rocket=rocket,
        vehicle=vehicle,
        body=dominant_body,
    )

    if thrust_n > 0.0 and m > 0.0:
        a_thrust = (thrust_n / m) * thrust_dir
    else:
        a_thrust = np.zeros(3)

    deriv = np.zeros(7)
    deriv[:3] = vel
    deriv[3:6] = a_grav + a_drag + a_thrust
    deriv[6] = 0.0
    return deriv


def rk4_step(
    t: float,
    state: np.ndarray,
    dt: float,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dominant_body: CelestialBody,
    thrust_n: float,
    thrust_dir: np.ndarray,
) -> np.ndarray:
    k1 = acceleration_with_thrust(t, state, rocket, vehicle, bodies, dominant_body, thrust_n, thrust_dir)
    k2 = acceleration_with_thrust(t + 0.5 * dt, state + 0.5 * dt * k1, rocket, vehicle, bodies, dominant_body, thrust_n, thrust_dir)
    k3 = acceleration_with_thrust(t + 0.5 * dt, state + 0.5 * dt * k2, rocket, vehicle, bodies, dominant_body, thrust_n, thrust_dir)
    k4 = acceleration_with_thrust(t + dt, state + dt * k3, rocket, vehicle, bodies, dominant_body, thrust_n, thrust_dir)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def consume_propellant_and_stage(
    rocket: Rocket,
    vehicle: VehicleState,
    state: np.ndarray,
    t: float,
    burn_time: float,
    mdot: float,
    events: list[dict],
) -> np.ndarray:
    idx = vehicle.active_stage_index
    if idx is None or mdot <= 0.0 or burn_time <= 0.0:
        return state

    burn_mass = mdot * burn_time
    available = vehicle.propellant_left[idx]
    used = min(burn_mass, available)

    vehicle.propellant_left[idx] -= used

    state = state.copy()
    state[6] -= used
    state[6] = max(state[6], dry_mass_floor(rocket, vehicle))

    if vehicle.propellant_left[idx] <= 1e-9:
        vehicle.propellant_left[idx] = 0.0
        state = separate_empty_stage_if_needed(rocket, vehicle, state, t, events)

    return state


def orbit_is_close_to_target(
    dominant_body: CelestialBody,
    t: float,
    state: np.ndarray,
    cmd: TargetOrbitCommand,
) -> bool:
    body_pos, body_vel = dominant_body.position_velocity_in_root_frame(t)
    r_rel = state[:3] - body_pos
    v_rel = state[3:6] - body_vel

    elems = orbital_elements_relative(dominant_body.mu, r_rel, v_rel)
    if elems["rp"] is None or elems["ra"] is None:
        return False

    rp_alt = elems["rp"] - dominant_body.radius
    ra_alt = elems["ra"] - dominant_body.radius

    ok_rp = abs(rp_alt - cmd.target_periapsis_altitude) <= cmd.tolerance_m
    ok_ra = abs(ra_alt - cmd.target_apoapsis_altitude) <= cmd.tolerance_m
    return ok_rp and ok_ra


def propagate_phase_burn(
    t0: float,
    state0: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dt: float,
    dominant_body: CelestialBody,
    cmd: BurnCommand,
) -> tuple[float, np.ndarray, list[np.ndarray], list[float], list[str], list[dict]]:
    t = t0
    state = state0.copy()

    states = [state.copy()]
    times = [t]
    bodies_list = [dominant_body.name]
    events: list[dict] = []

    remaining = float(cmd.duration)

    while remaining > 1e-9:
        sync_active_stage(vehicle, rocket)
        stg = current_stage(vehicle, rocket)
        if stg is None:
            events.append({"type": "NO_ACTIVE_STAGE", "time": t})
            break

        thrust_n, isp_s = current_stage_thrust_isp(
            rocket=rocket,
            vehicle=vehicle,
            thrust_override=cmd.thrust_newtons,
            isp_override=cmd.isp_seconds,
        )

        print("DEBUG phase burn")
        print("  t =", t)
        print("  cmd.duration =", cmd.duration)
        print("  cmd.thrust_newtons =", cmd.thrust_newtons)
        print("  cmd.isp_seconds =", cmd.isp_seconds)
        print("  active_stage_index =", vehicle.active_stage_index)

        if thrust_n <= 0.0 or isp_s <= 0.0:
            print("  DEBUG -> NO_THRUST")
            events.append({"type": "NO_THRUST", "time": t})
            break

        mdot = thrust_n / (isp_s * G0)
        if mdot <= 0.0:
            print("  DEBUG -> INVALID_MDOT")
            events.append({"type": "INVALID_MDOT", "time": t})
            break

        h = min(dt, remaining)

        idx = vehicle.active_stage_index
        prop_left = vehicle.propellant_left[idx]
        max_stage_burn_time = prop_left / mdot if mdot > 0 else 0.0

        print("  thrust_n =", thrust_n)
        print("  isp_s =", isp_s)
        print("  mdot =", mdot)
        print("  dt =", dt)
        print("  remaining =", remaining)
        print("  h =", h)
        print("  prop_left =", prop_left)
        print("  max_stage_burn_time =", max_stage_burn_time)

        if max_stage_burn_time <= 1e-9:
            state = separate_empty_stage_if_needed(rocket, vehicle, state, t, events)
            continue

        h = min(h, max_stage_burn_time)

        dom = dominant_body_for_position(t, state[:3], bodies)
        if dom.name != dominant_body.name:
            events.append({"type": "SOI_CHANGE", "time": t, "from": dominant_body.name, "to": dom.name})
            dominant_body = dom

        direction = burn_direction(cmd, state[:3], state[3:6], dominant_body, t)

        next_state = rk4_step(
            t=t,
            state=state,
            dt=h,
            rocket=rocket,
            vehicle=vehicle,
            bodies=bodies,
            dominant_body=dominant_body,
            thrust_n=thrust_n,
            thrust_dir=direction,
        )

        impact_t, impact_state, impacted = append_impact_and_stop(
            t=t,
            h=h,
            state=state,
            next_state=next_state,
            dominant_body=dominant_body,
            times=times,
            states=states,
            bodies_list=bodies_list,
            events=events,
        )
        if impacted:
            t = impact_t
            state = impact_state
            break

        t += h
        next_state = consume_propellant_and_stage(
            rocket=rocket,
            vehicle=vehicle,
            state=next_state,
            t=t,
            burn_time=h,
            mdot=mdot,
            events=events,
        )

        state = next_state
        times.append(t)
        states.append(state.copy())
        bodies_list.append(dominant_body.name)

        remaining -= h

    return t, state, states, times, bodies_list, events


def propagate_phase_coast(
    t0: float,
    state0: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dt: float,
    dominant_body: CelestialBody,
    duration: float,
) -> tuple[float, np.ndarray, list[np.ndarray], list[float], list[str], list[dict]]:
    t = t0
    state = state0.copy()

    states = [state.copy()]
    times = [t]
    bodies_list = [dominant_body.name]
    events: list[dict] = []

    remaining = float(duration)

    while remaining > 1e-9:
        h = min(dt, remaining)

        dom = dominant_body_for_position(t, state[:3], bodies)
        if dom.name != dominant_body.name:
            events.append({"type": "SOI_CHANGE", "time": t, "from": dominant_body.name, "to": dom.name})
            dominant_body = dom

        next_state = rk4_step(
            t=t,
            state=state,
            dt=h,
            rocket=rocket,
            vehicle=vehicle,
            bodies=bodies,
            dominant_body=dominant_body,
            thrust_n=0.0,
            thrust_dir=np.zeros(3),
        )

        impact_t, impact_state, impacted = append_impact_and_stop(
            t=t,
            h=h,
            state=state,
            next_state=next_state,
            dominant_body=dominant_body,
            times=times,
            states=states,
            bodies_list=bodies_list,
            events=events,
        )
        if impacted:
            t = impact_t
            state = impact_state
            break

        t += h
        state = next_state

        times.append(t)
        states.append(state.copy())
        bodies_list.append(dominant_body.name)

        remaining -= h

    return t, state, states, times, bodies_list, events


def propagate_phase_target_orbit(
    t0: float,
    state0: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dt: float,
    dominant_body: CelestialBody,
    cmd: TargetOrbitCommand,
) -> tuple[float, np.ndarray, list[np.ndarray], list[float], list[str], list[dict]]:
    t = t0
    state = state0.copy()

    states = [state.copy()]
    times = [t]
    bodies_list = [dominant_body.name]
    events: list[dict] = []

    remaining = float(cmd.max_duration)

    while remaining > 1e-9:
        if orbit_is_close_to_target(dominant_body, t, state, cmd):
            events.append({"type": "TARGET_ORBIT_REACHED", "time": t, "body": dominant_body.name})
            break

        sync_active_stage(vehicle, rocket)
        stg = current_stage(vehicle, rocket)
        if stg is None:
            events.append({"type": "NO_ACTIVE_STAGE", "time": t})
            break

        thrust_n, isp_s = current_stage_thrust_isp(
            rocket=rocket,
            vehicle=vehicle,
            thrust_override=cmd.thrust_newtons,
            isp_override=cmd.isp_seconds,
        )

        print("DEBUG phase burn")
        print("  t =", t)
        print("  cmd.duration =", cmd.duration)
        print("  cmd.thrust_newtons =", cmd.thrust_newtons)
        print("  cmd.isp_seconds =", cmd.isp_seconds)
        print("  active_stage_index =", vehicle.active_stage_index)

        if thrust_n <= 0.0 or isp_s <= 0.0:
            print("  DEBUG -> NO_THRUST")
            events.append({"type": "NO_THRUST", "time": t})
            break

        mdot = thrust_n / (isp_s * G0)
        if mdot <= 0.0:
            print("  DEBUG -> INVALID_MDOT")
            events.append({"type": "INVALID_MDOT", "time": t})
            break

        h = min(dt, remaining)

        idx = vehicle.active_stage_index
        prop_left = vehicle.propellant_left[idx]
        max_stage_burn_time = prop_left / mdot if mdot > 0 else 0.0

        print("  thrust_n =", thrust_n)
        print("  isp_s =", isp_s)
        print("  mdot =", mdot)
        print("  dt =", dt)
        print("  remaining =", remaining)
        print("  h =", h)
        print("  prop_left =", prop_left)
        print("  max_stage_burn_time =", max_stage_burn_time)

        if max_stage_burn_time <= 1e-9:
            state = separate_empty_stage_if_needed(rocket, vehicle, state, t, events)
            continue

        h = min(h, max_stage_burn_time)

        dom = dominant_body_for_position(t, state[:3], bodies)
        if dom.name != dominant_body.name:
            events.append({"type": "SOI_CHANGE", "time": t, "from": dominant_body.name, "to": dom.name})
            dominant_body = dom

        burn_cmd = BurnCommand(
            direction_mode=cmd.direction_mode,
            thrust_newtons=thrust_n,
            isp_seconds=isp_s,
            duration=h,
        )
        direction = burn_direction(burn_cmd, state[:3], state[3:6], dominant_body, t)

        next_state = rk4_step(
            t=t,
            state=state,
            dt=h,
            rocket=rocket,
            vehicle=vehicle,
            bodies=bodies,
            dominant_body=dominant_body,
            thrust_n=thrust_n,
            thrust_dir=direction,
        )

        impact_t, impact_state, impacted = append_impact_and_stop(
            t=t,
            h=h,
            state=state,
            next_state=next_state,
            dominant_body=dominant_body,
            times=times,
            states=states,
            bodies_list=bodies_list,
            events=events,
        )
        if impacted:
            t = impact_t
            state = impact_state
            break

        t += h
        next_state = consume_propellant_and_stage(
            rocket=rocket,
            vehicle=vehicle,
            state=next_state,
            t=t,
            burn_time=h,
            mdot=mdot,
            events=events,
        )

        state = next_state
        times.append(t)
        states.append(state.copy())
        bodies_list.append(dominant_body.name)

        remaining -= h

    return t, state, states, times, bodies_list, events


def propagate_phase_soi_change(
    t0: float,
    state0: np.ndarray,
    rocket: Rocket,
    vehicle: VehicleState,
    bodies: dict[str, CelestialBody],
    dt: float,
    dominant_body: CelestialBody,
    cmd: SOIChangeCommand,
) -> tuple[float, np.ndarray, list[np.ndarray], list[float], list[str], list[dict]]:
    t = t0
    state = state0.copy()

    states = [state.copy()]
    times = [t]
    bodies_list = [dominant_body.name]
    events: list[dict] = []

    remaining = float(cmd.max_duration)

    while remaining > 1e-9:
        current_dom = dominant_body_for_position(t, state[:3], bodies)
        if current_dom.name != dominant_body.name:
            events.append({"type": "SOI_CHANGE", "time": t, "from": dominant_body.name, "to": current_dom.name})
            dominant_body = current_dom

        if dominant_body.name == cmd.target_body_name:
            events.append({"type": "TARGET_SOI_REACHED", "time": t, "body": dominant_body.name})
            break

        h = min(dt, remaining)

        next_state = rk4_step(
            t=t,
            state=state,
            dt=h,
            rocket=rocket,
            vehicle=vehicle,
            bodies=bodies,
            dominant_body=dominant_body,
            thrust_n=0.0,
            thrust_dir=np.zeros(3),
        )

        impact_t, impact_state, impacted = append_impact_and_stop(
            t=t,
            h=h,
            state=state,
            next_state=next_state,
            dominant_body=dominant_body,
            times=times,
            states=states,
            bodies_list=bodies_list,
            events=events,
        )
        if impacted:
            t = impact_t
            state = impact_state
            break

        t += h
        state = next_state

        times.append(t)
        states.append(state.copy())
        bodies_list.append(dominant_body.name)

        remaining -= h

    return t, state, states, times, bodies_list, events


def run_mission_3d(
    rocket: Rocket,
    mission: MissionPlan,
    bodies: dict[str, CelestialBody],
    launch_body_name: str = "Earth",
    t0: float = 0.0,
    dt: float = 2.0,
) -> SimulationResult3D:
    launch_body = bodies[launch_body_name]
    launch_site = mission.launch_site

    r0, v0 = launch_state_from_site(
        t0=t0,
        central_body=launch_body,
        launch_lat_deg=launch_site.latitude_deg,
        launch_lon_deg=launch_site.longitude_deg,
        launch_alt_m=launch_site.altitude_m,
        launch_azimuth_deg=launch_site.azimuth_deg,
    )

    m0 = rocket.total_initial_mass()
    state = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], m0], dtype=float)

    vehicle = init_vehicle_state(rocket)

    print("DEBUG total_initial_mass =", rocket.total_initial_mass())
    print("DEBUG payload_mass =", rocket.payload_mass)

    for i, s in enumerate(rocket.stages):
        print(
            f"DEBUG stage {i}: "
            f"dry_mass={getattr(s, 'dry_mass', None)}, "
            f"propellant_mass={getattr(s, 'propellant_mass', None)}, "
            f"thrust_vac={getattr(s, 'thrust_vac', None)}, "
            f"isp_vac={getattr(s, 'isp_vac', None)}"
        )

    print("DEBUG vehicle.propellant_left =", vehicle.propellant_left)

    t = t0
    dominant_body = dominant_body_for_position(t, state[:3], bodies)

    all_times = [t]
    all_states = [state.copy()]
    all_dom = [dominant_body.name]
    all_phase_names = ["INIT"]
    all_events: list[dict] = []

    # Seguridad extra: impacto inmediato en el estado inicial
    if has_impacted_body(t, state[:3], dominant_body):
        all_events.append({
            "type": "IMPACT",
            "time": t,
            "body": dominant_body.name,
            "altitude_m": 0.0,
        })
        return SimulationResult3D(
            times=np.array(all_times, dtype=float),
            states=np.array(all_states, dtype=float),
            dominant_bodies=all_dom,
            phase_names=all_phase_names,
            events=all_events,
        )

    for phase in mission.phases:
        if phase.phase_type == "burn":
            if phase.burn is None:
                raise ValueError(f"La fase '{phase.name}' es burn pero no tiene BurnCommand")

            t, state, states, times, doms, events = propagate_phase_burn(
                t0=t,
                state0=state,
                rocket=rocket,
                vehicle=vehicle,
                bodies=bodies,
                dt=dt,
                dominant_body=dominant_body,
                cmd=phase.burn,
            )

        elif phase.phase_type == "coast":
            t, state, states, times, doms, events = propagate_phase_coast(
                t0=t,
                state0=state,
                rocket=rocket,
                vehicle=vehicle,
                bodies=bodies,
                dt=dt,
                dominant_body=dominant_body,
                duration=phase.coast_duration,
            )

        elif phase.phase_type == "target_orbit":
            if phase.target_orbit is None:
                raise ValueError(f"La fase '{phase.name}' es target_orbit pero no tiene TargetOrbitCommand")

            t, state, states, times, doms, events = propagate_phase_target_orbit(
                t0=t,
                state0=state,
                rocket=rocket,
                vehicle=vehicle,
                bodies=bodies,
                dt=dt,
                dominant_body=dominant_body,
                cmd=phase.target_orbit,
            )

        elif phase.phase_type == "soi_change":
            if phase.soi_change is None:
                raise ValueError(f"La fase '{phase.name}' es soi_change pero no tiene SOIChangeCommand")

            t, state, states, times, doms, events = propagate_phase_soi_change(
                t0=t,
                state0=state,
                rocket=rocket,
                vehicle=vehicle,
                bodies=bodies,
                dt=dt,
                dominant_body=dominant_body,
                cmd=phase.soi_change,
            )
        else:
            raise ValueError(f"Tipo de fase no soportado: {phase.phase_type}")

        dominant_body = dominant_body_for_position(t, state[:3], bodies)

        for k in range(1, len(times)):
            all_times.append(times[k])
            all_states.append(states[k])
            all_dom.append(doms[k])
            all_phase_names.append(phase.name)

        all_events.extend(events)

        if any(event.get("type") == "IMPACT" for event in events):
            break

    return SimulationResult3D(
        times=np.array(all_times, dtype=float),
        states=np.array(all_states, dtype=float),
        dominant_bodies=all_dom,
        phase_names=all_phase_names,
        events=all_events,
    )