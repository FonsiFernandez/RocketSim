# src/kinetica/simulation/trajectory2d.py
import numpy as np
from typing import Tuple, List

from kinetica.models.planet import Planet
from kinetica.models.rocket import Rocket


def thrust_pitch_angle_deg(t: float, pitch_start: float, pitch_end: float, final_pitch_deg: float) -> float:
    """
    Programa de pitch muy simple:
    - Antes de pitch_start: 90º (vertical)
    - Entre pitch_start y pitch_end: transición lineal de 90º a final_pitch_deg
    - Después de pitch_end: final_pitch_deg

    Ángulo medido desde el horizonte local (tangencial):
        90º = totalmente radial (vertical)
         0º = totalmente tangencial (horizontal progrado)
    """
    if t <= pitch_start:
        return 90.0
    if t >= pitch_end:
        return final_pitch_deg
    alpha = (t - pitch_start) / (pitch_end - pitch_start)
    return 90.0 + alpha * (final_pitch_deg - 90.0)


def run_ascent_2d_with_pitch(
    rocket: Rocket,
    planet: Planet,
    t_final: float,
    dt: float,
    pitch_start: float,
    pitch_end: float,
    final_pitch_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulación 2D en un plano orbital, usando integrador tipo RK4.

    Estado y = [x, y, vx, vy, m]

    x, y: posición [m]
    vx, vy: velocidad [m/s]
    m: masa [kg]

    Varios stages con staging y empuje dependiente de altitud.
    El caudal de masa (mdot) y el thrust se consideran constantes dentro de cada paso dt.
    """
    stages: List = rocket.stages
    n_stages = len(stages)

    # Estado inicial: en la superficie, sobre el eje Y
    x0 = 0.0
    y0 = planet.radius          # justo en la superficie
    vx0 = 0.0
    vy0 = 0.0
    m0 = rocket.total_initial_mass()
    y_state = np.array([x0, y0, vx0, vy0, m0], dtype=float)

    # Propelente restante por etapa
    prop_left = [s.propellant_mass for s in stages]
    dropped = [False] * n_stages
    active_stage = 0 if n_stages > 0 else None

    n_steps = int(t_final / dt) + 1
    t_values = np.linspace(0.0, t_final, n_steps)
    y_values = np.zeros((n_steps, 5), dtype=float)

    def acceleration_and_mdot(
        t_local: float,
        state: np.ndarray,
        current_thrust: float,
        current_mdot: float,
        area: float,
    ):
        """
        Calcula aceleración total (gravedad + drag + thrust) y dm/dt para el estado dado.
        current_thrust y current_mdot se mantienen constantes en este paso de integración.
        """
        x, y_pos, vx, vy, m = state
        r_vec = np.array([x, y_pos], dtype=float)
        r = np.linalg.norm(r_vec)

        # Gravedad
        if r > 0.0:
            a_grav = -planet.mu * r_vec / r**3
        else:
            a_grav = np.zeros(2)

        # Base radial / tangencial
        if r > 0:
            e_r = r_vec / r
            e_t = np.array([-e_r[1], e_r[0]])  # tangencial prograde
        else:
            e_r = np.array([0.0, 1.0])
            e_t = np.array([1.0, 0.0])

        # Drag
        v_vec = np.array([vx, vy], dtype=float)
        v = np.linalg.norm(v_vec)
        altitude = r - planet.radius
        rho = planet.density(altitude)

        if v > 0 and area > 0 and rho > 0 and m > 0:
            v_hat = v_vec / v
            drag_mag = 0.5 * rho * rocket.cd * area * v * v
            a_drag = -drag_mag / m * v_hat
        else:
            a_drag = np.zeros(2)

        # Thrust
        if current_thrust > 0 and m > 0:
            gamma_deg = thrust_pitch_angle_deg(t_local, pitch_start, pitch_end, final_pitch_deg)
            gamma_rad = np.deg2rad(gamma_deg)
            thrust_dir = np.cos(gamma_rad) * e_t + np.sin(gamma_rad) * e_r
            a_thrust = current_thrust / m * thrust_dir
        else:
            a_thrust = np.zeros(2)

        a_total = a_grav + a_drag + a_thrust

        # Derivadas: dx/dt = vx, dy/dt = vy, dv/dt = a, dm/dt = -mdot
        deriv = np.array([vx, vy, a_total[0], a_total[1], -current_mdot], dtype=float)
        return deriv

    for i, t in enumerate(t_values):
        y_values[i] = y_state
        x, y_pos, vx, vy, m = y_state

        # Comprobación de impacto con el planeta
        r_vec = np.array([x, y_pos], dtype=float)
        r = np.linalg.norm(r_vec)
        # 🔧 IMPORTANTE: aquí usamos <, NO <=, para permitir despegar desde la superficie
        if r < planet.radius:
            # Proyectar sobre la superficie y frenar
            if r > 0:
                r_vec = r_vec * (planet.radius / r)
            y_state = np.array([r_vec[0], r_vec[1], 0.0, 0.0, m], dtype=float)
            y_values[i] = y_state
            y_values[i + 1 :] = y_state
            break

        # Determinar etapa activa (saltando las ya soltadas)
        if active_stage is not None:
            while active_stage < n_stages and dropped[active_stage]:
                active_stage += 1
            if active_stage >= n_stages:
                active_stage = None

        # Área frontal: primera etapa no soltada
        aero_stage_index = None
        for j in range(n_stages):
            if not dropped[j]:
                aero_stage_index = j
                break
        area = stages[aero_stage_index].area if aero_stage_index is not None else 0.0

        # Calcular thrust y mdot para este paso
        thrust = 0.0
        mdot = 0.0
        stage_just_finished = False

        if active_stage is not None:
            stage = stages[active_stage]
            prop = prop_left[active_stage]

            altitude = r - planet.radius
            mdot_full = stage.mass_flow_rate(planet, altitude)
            thrust_full, _ = stage.performance_at_altitude(planet, altitude)

            if mdot_full > 0 and prop > 0 and m > 0:
                max_burn = mdot_full * dt
                if prop >= max_burn:
                    prop_burn = max_burn
                    mdot = mdot_full
                    thrust = thrust_full
                else:
                    prop_burn = prop
                    mdot = prop_burn / dt
                    thrust = thrust_full * (mdot / mdot_full) if mdot_full > 0 else 0.0
                    stage_just_finished = True

                prop_left[active_stage] -= prop_burn
                if prop_left[active_stage] <= 0 and not stage_just_finished:
                    stage_just_finished = True

        # ===== Integración RK4 en este paso =====
        state = y_state.copy()

        # k1
        k1 = acceleration_and_mdot(t, state, thrust, mdot, area)
        # k2
        k2_state = state + 0.5 * dt * k1
        k2 = acceleration_and_mdot(t + 0.5 * dt, k2_state, thrust, mdot, area)
        # k3
        k3_state = state + 0.5 * dt * k2
        k3 = acceleration_and_mdot(t + 0.5 * dt, k3_state, thrust, mdot, area)
        # k4
        k4_state = state + dt * k3
        k4 = acceleration_and_mdot(t + dt, k4_state, thrust, mdot, area)

        y_next = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Staging: soltar etapa si se ha agotado
        if active_stage is not None and stage_just_finished and not dropped[active_stage]:
            y_next[4] -= stages[active_stage].dry_mass  # restar masa seca
            dropped[active_stage] = True
            active_stage += 1

        # Masa mínima (payload + masas secas restantes)
        min_dry_mass_remaining = rocket.payload_mass + sum(
            stages[k].dry_mass for k in range(n_stages) if not dropped[k]
        )
        if y_next[4] < min_dry_mass_remaining:
            y_next[4] = min_dry_mass_remaining

        # Segunda comprobación de impacto después del paso
        x2, y2, vx2, vy2, m2 = y_next
        r_vec2 = np.array([x2, y2], dtype=float)
        r2 = np.linalg.norm(r_vec2)
        # 🔧 Aquí también usamos < y no <=
        if r2 < planet.radius:
            if r2 > 0:
                r_vec2 = r_vec2 * (planet.radius / r2)
            y_next = np.array([r_vec2[0], r_vec2[1], 0.0, 0.0, m2], dtype=float)
            y_values[i + 1 :] = y_next
            y_state = y_next
            break

        y_state = y_next

    return t_values, y_values
