import numpy as np
from typing import Tuple

from rocketsim.models.planet import Planet
from rocketsim.models.rocket import Rocket


def rocket_ode_vertical(
    t: float,
    y: np.ndarray,
    rocket: Rocket,
    planet: Planet,
) -> np.ndarray:
    """
    Ecuación diferencial 1D (vertical).
    y = [altitud, velocidad, masa]
    """
    h, v, m = y
    r = planet.radius + h

    # Gravedad
    g = planet.gravity_acc(r)

    stages = rocket.stages
    n_stages = len(stages)

    # Por simplicidad: sólo primera etapa con propelente
    stage = stages[0]
    # Aquí podrías usar logic similar a 2D, pero esta función es demo.

    # Empuje e Isp en función de altitud
    thrust, isp = stage.performance_at_altitude(planet, h)
    if thrust <= 0 or isp <= 0:
        mdot = 0.0
        thrust = 0.0
    else:
        mdot = thrust / (isp * 9.80665)

    rho = planet.density(h)
    A = stage.area
    cd = rocket.cd

    drag = 0.5 * rho * cd * A * v * abs(v)

    F_net = thrust - m * g - np.sign(v) * drag

    dhdt = v
    dvdt = F_net / m if m > 0 else 0.0
    dmdt = -mdot

    return np.array([dhdt, dvdt, dmdt], dtype=float)


def run_vertical_ascent(
    rocket: Rocket,
    planet: Planet,
    t_final: float = 300.0,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ascenso vertical simple, opcional. No incluye staging real.
    """
    h0 = 0.0
    v0 = 0.0
    m0 = rocket.total_initial_mass()
    y = np.array([h0, v0, m0], dtype=float)

    n_steps = int(t_final / dt) + 1
    t_values = np.linspace(0, t_final, n_steps)
    y_values = np.zeros((n_steps, 3))

    for i, t in enumerate(t_values):
        y_values[i] = y
        dydt = rocket_ode_vertical(t, y, rocket, planet)
        y = y + dydt * dt
        if y[0] < 0:
            y[0] = 0.0
            y[1] = 0.0
            y_values[i] = y
            y_values[i + 1 :] = y
            break

    return t_values, y_values
