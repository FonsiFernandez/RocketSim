# src/kinetica/orbits/twobody.py
import numpy as np
from typing import Tuple


def two_body_derivatives(mu: float, state: np.ndarray) -> np.ndarray:
    """
    Ecuaciones del problema de dos cuerpos en 2D:
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r_vec = np.array([x, y], dtype=float)
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros_like(state)

    a = -mu * r_vec / r**3
    return np.array([vx, vy, a[0], a[1]], dtype=float)


def propagate_orbit_rk4(
    mu: float,
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    dt: float,
    body_radius: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propaga una órbita kepleriana (2 cuerpos) mediante RK4 sencillo.
    Devuelve:
      - t: tiempos
      - r_hist: posiciones [N x 2]
      - v_hist: velocidades [N x 2]
    """
    state = np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float)
    n_steps = int(t_final / dt) + 1

    t_values = [0.0]
    r_hist = [np.array([state[0], state[1]], dtype=float)]
    v_hist = [np.array([state[2], state[3]], dtype=float)]

    for i in range(1, n_steps):
        t = i * dt

        k1 = two_body_derivatives(mu, state)
        k2 = two_body_derivatives(mu, state + 0.5 * dt * k1)
        k3 = two_body_derivatives(mu, state + 0.5 * dt * k2)
        k4 = two_body_derivatives(mu, state + dt * k3)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if body_radius is not None:
            r_prev = np.linalg.norm(state[:2]) - body_radius
            r_next = np.linalg.norm(next_state[:2]) - body_radius

            if r_next <= 0.0:
                if abs(r_prev - r_next) < 1e-12:
                    alpha = 1.0
                else:
                    alpha = r_prev / (r_prev - r_next)

                alpha = max(0.0, min(1.0, alpha))
                impact_state = state + alpha * (next_state - state)
                impact_time = (i - 1) * dt + alpha * dt

                t_values.append(impact_time)
                r_hist.append(np.array([impact_state[0], impact_state[1]], dtype=float))
                v_hist.append(np.array([impact_state[2], impact_state[3]], dtype=float))
                break

        state = next_state
        t_values.append(t)
        r_hist.append(np.array([state[0], state[1]], dtype=float))
        v_hist.append(np.array([state[2], state[3]], dtype=float))

    return np.array(t_values), np.array(r_hist), np.array(v_hist)


def apply_delta_v(v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """
    Aplica un impulso instantáneo Δv a la velocidad.
    """
    return v + dv
