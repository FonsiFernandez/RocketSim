# src/rocketsim/orbits/twobody.py
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
    t = np.linspace(0.0, t_final, n_steps)
    r_hist = np.zeros((n_steps, 2), dtype=float)
    v_hist = np.zeros((n_steps, 2), dtype=float)

    for i, ti in enumerate(t):
        x, y, vx, vy = state
        r_hist[i] = [x, y]
        v_hist[i] = [vx, vy]

        k1 = two_body_derivatives(mu, state)
        k2 = two_body_derivatives(mu, state + 0.5 * dt * k1)
        k3 = two_body_derivatives(mu, state + 0.5 * dt * k2)
        k4 = two_body_derivatives(mu, state + dt * k3)

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, r_hist, v_hist


def apply_delta_v(v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """
    Aplica un impulso instantáneo Δv a la velocidad.
    """
    return v + dv
