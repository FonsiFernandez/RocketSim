import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import math
import numpy as np
import matplotlib.pyplot as plt

from kinetica.models.planet import PLANETS_BY_NAME
from kinetica.config.rockets import get_rocket_by_name
from kinetica.simulation.trajectory2d import run_ascent_2d_with_pitch


# ================== PARÁMETROS CONFIGURABLES ==================

# Planeta a usar: 'earth', 'mars', 'moon'
PLANET_NAME = "earth"

# Cohete a usar: ver keys en ROCKET_PRESETS (light, medium_f9, miura1, miura5_450, electron, etc.)
ROCKET_NAME = "miura5_450"   # cámbialo aquí para probar otros

# Parámetros de simulación
SIM_T_FINAL = 600.0   # s
SIM_DT = 0.1          # s  -> importante bajarlo para cohetes grandes

# Programa de pitch (simple)
PITCH_START = 10.0       # s -> vertical hasta aquí
PITCH_END = 140.0        # s -> transición hasta aquí
FINAL_PITCH_DEG = 10.0    # 0° = horizontal final

# Altitud umbral para considerar órbita "segura"
SAFE_PERIGEE_ALT = 120_000.0  # 120 km

# ===============================================================


def classify_orbit(planet_radius: float, mu: float, elems: dict) -> str:
    a = elems["a"]
    rp_alt = elems["perigee_altitude"]
    ra_alt = elems["apogee_altitude"]

    if a is None or rp_alt is None or ra_alt is None:
        return "Trayectoria no ligada (escape/parabólica)"

    if rp_alt < 0:
        return "Suborbital (interseca el interior del planeta)"
    if rp_alt < 120_000:
        return "Reentrada atmosférica (perigeo en atmósfera baja)"
    if rp_alt < 2_000_000:
        return "Órbita baja (LEO)"
    if rp_alt < 35_786_000 * 1.2:  # muy aproximado
        return "Órbita media/alta (MEO/HEO)"
    return "Órbita muy alta / casi escape"


def compute_orbital_elements(planet_radius: float, mu: float, x: float, y: float, vx: float, vy: float):
    """
    Calcula los elementos orbitales en un instante dado (x, y, vx, vy).
    No hace 'trampa' especial si estás en la superficie: si la elipse corta el planeta,
    se verá como suborbital (perigeo por debajo de la superficie).
    """
    r_vec = np.array([x, y], dtype=float)
    v_vec = np.array([vx, vy], dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    if r == 0:
        return None

    # Momento angular específico (2D -> componente z)
    h = abs(r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0])

    # Energía específica
    eps = 0.5 * v * v - mu / r

    if eps >= 0:
        # Trayectoria no ligada (parabólica o hiperbólica)
        a = None
        e = None
        rp = None
        ra = None
        T = None
        orbit_type = "no ligada (parabólica/hiperbólica)"
    else:
        a = -mu / (2 * eps)
        e_term = 1.0 - h * h / (a * mu)
        e_term = max(e_term, 0.0)
        e = float(math.sqrt(e_term))
        rp = a * (1.0 - e)
        ra = a * (1.0 + e)
        T = 2 * math.pi * math.sqrt(a**3 / mu)
        orbit_type = "elíptica (ligada)"

    elems = {
        "r": r,
        "v": v,
        "h": h,
        "eps": eps,
        "a": a,
        "e": e if eps < 0 else None,
        "rp": rp,
        "ra": ra,
        "orbit_type": orbit_type,
        "perigee_altitude": rp - planet_radius if rp is not None else None,
        "apogee_altitude": ra - planet_radius if ra is not None else None,
        "period": T,
    }
    elems["classification"] = classify_orbit(planet_radius, mu, elems)
    return elems


def run_demo():
    # Planeta
    planet_factory = PLANETS_BY_NAME.get(PLANET_NAME.lower())
    if planet_factory is None:
        raise ValueError(f"PLANET_NAME desconocido: {PLANET_NAME}. Usa uno de {list(PLANETS_BY_NAME.keys())}")
    planet = planet_factory()

    # Cohete preset
    rocket = get_rocket_by_name(ROCKET_NAME)

    print("===== CONFIGURACIÓN DEL ESCENARIO =====")
    print(f"Planeta: {planet.name}")
    print(f"Radio: {planet.radius/1000:.1f} km, mu: {planet.mu:.3e} m^3/s^2")
    print(f"Cohete: {ROCKET_NAME}")
    print(f"Carga útil: {rocket.payload_mass:.0f} kg")
    print("Etapas:")
    for idx, s in enumerate(rocket.stages, start=1):
        print(
            f"  Etapa {idx}: dry={s.dry_mass:.0f} kg, prop={s.propellant_mass:.0f} kg, "
            f"T_SL={s.thrust_sl/1000:.0f} kN, T_VAC={s.thrust_vac/1000:.0f} kN, "
            f"Isp_SL={s.isp_sl:.0f} s, Isp_VAC={s.isp_vac:.0f} s, "
            f"burn_vac≈{s.burn_time_vac:.1f} s"
        )
    print("Programa de pitch:")
    print(f"  Vertical hasta t={PITCH_START:.1f} s")
    print(f"  Transición hasta t={PITCH_END:.1f} s")
    print(f"  Ángulo final respecto al horizonte: {FINAL_PITCH_DEG:.1f}°")
    print("=======================================\n")

    # Simulación 2D
    t, y = run_ascent_2d_with_pitch(
        rocket=rocket,
        planet=planet,
        t_final=SIM_T_FINAL,
        dt=SIM_DT,
        pitch_start=PITCH_START,
        pitch_end=PITCH_END,
        final_pitch_deg=FINAL_PITCH_DEG,
    )

    x = y[:, 0]
    y_pos = y[:, 1]
    vx = y[:, 2]
    vy = y[:, 3]
    m = y[:, 4]

    r = np.sqrt(x * x + y_pos * y_pos)
    altitude = r - planet.radius
    speed = np.sqrt(vx * vx + vy * vy)

    # --- Resumen del vuelo ---
    alt_max = float(np.max(altitude))
    idx_apogee = int(np.argmax(altitude))
    t_alt_max = float(t[idx_apogee])
    v_max = float(np.max(speed))
    t_v_max = float(t[np.argmax(speed)])

    print("===== RESUMEN DEL VUELO (2D) =====")
    print(f"Altitud máxima: {alt_max:,.0f} m ({alt_max/1000:.2f} km)")
    print(f"Tiempo hasta altitud máxima: {t_alt_max:.1f} s")
    print(f"Velocidad máxima: {v_max:,.1f} m/s")
    print(f"Tiempo hasta velocidad máxima: {t_v_max:.1f} s")
    print(f"Masa inicial: {rocket.total_initial_mass():,.1f} kg")
    print(f"Masa final (último estado): {m[-1]:,.1f} kg")

    # === Elementos orbitales calculados en el APOGEO ===
    x_a = x[idx_apogee]
    y_a = y_pos[idx_apogee]
    vx_a = vx[idx_apogee]
    vy_a = vy[idx_apogee]

    elems = compute_orbital_elements(
        planet_radius=planet.radius,
        mu=planet.mu,
        x=x_a,
        y=y_a,
        vx=vx_a,
        vy=vy_a,
    )

    print("\n===== PARÁMETROS ORBITALES (en el apogeo) =====")
    if elems is None:
        print("No se han podido calcular elementos orbitales.")
    else:
        print(f"Tipo de órbita: {elems['orbit_type']}")
        print(f"Clasificación: {elems['classification']}")
        if elems["a"] is not None:
            print(f"Semi-eje mayor a: {elems['a']/1000:.1f} km")
            if elems["perigee_altitude"] is not None:
                if elems["perigee_altitude"] < 0:
                    print("Perigeo: por debajo de la superficie (suborbital)")
                else:
                    print(f"Perigeo: {elems['perigee_altitude']/1000:.1f} km")
            if elems["apogee_altitude"] is not None:
                print(f"Apogeo:  {elems['apogee_altitude']/1000:.1f} km")
            if elems["period"] is not None:
                print(f"Periodo orbital (si fuera completa): {elems['period']/60:.1f} min")

            if (
                elems["perigee_altitude"] is not None
                and elems["perigee_altitude"] > SAFE_PERIGEE_ALT
            ):
                print(">>> Perigeo por encima de 120 km: órbita 'segura' (sin reentrada inmediata).")
            elif elems["perigee_altitude"] is not None:
                print(">>> Perigeo corta atmósfera / planeta: trayectoria suborbital o reentrada.")
    print("================================================")

    # === Gráficas ===

    alt_km = altitude / 1000.0

    # Altitud + velocidad vs tiempo
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Altitud [km]")
    ax1.plot(t, alt_km, label="Altitud")
    ax1.axvline(t_alt_max, linestyle=":", label="Apogeo")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Velocidad [m/s]")
    ax2.plot(t, speed, linestyle="--", label="Velocidad")

    plt.title(f"Ascent 2D en {planet.name} ({ROCKET_NAME})")
    plt.tight_layout()

    # Trayectoria (distancia horizontal vs altitud)
    fig2, ax_traj = plt.subplots()
    ax_traj.set_xlabel("Distancia horizontal [km]")
    ax_traj.set_ylabel("Altitud [km]")
    ax_traj.plot(x / 1000.0, alt_km)
    ax_traj.scatter(x_a / 1000.0, alt_max / 1000.0, marker="o", label="Apogeo")
    ax_traj.grid(True)
    ax_traj.set_title("Trayectoria aproximada en el plano")
    ax_traj.legend()

    # Órbita kepleriana aproximada SOLO si tiene sentido verla (apogeo alto)
    if (
        elems is not None
        and elems["a"] is not None
        and elems["e"] is not None
        and elems["apogee_altitude"] is not None
        and elems["apogee_altitude"] > 100_000.0  # solo si apogeo > 100 km
    ):
        a = elems["a"]
        e = elems["e"]

        f_vals = np.linspace(0, 2 * np.pi, 400)
        r_orbit = a * (1 - e**2) / (1 + e * np.cos(f_vals))

        x_orb = (r_orbit * np.cos(f_vals)) / 1000.0
        y_orb = (r_orbit * np.sin(f_vals)) / 1000.0

        fig3, ax_orb = plt.subplots()
        ax_orb.set_aspect("equal", "box")

        planet_theta = np.linspace(0, 2 * np.pi, 200)
        x_planet = (planet.radius * np.cos(planet_theta)) / 1000.0
        y_planet = (planet.radius * np.sin(planet_theta)) / 1000.0
        ax_orb.fill(x_planet, y_planet, alpha=0.3, label=planet.name)

        ax_orb.plot(x_orb, y_orb, label="Órbita elíptica (ideal)", linestyle="-")
        ax_orb.set_xlabel("x [km]")
        ax_orb.set_ylabel("y [km]")
        ax_orb.set_title(f"Órbita alrededor de {planet.name}")
        ax_orb.grid(True)
        ax_orb.legend()


    plt.show()
