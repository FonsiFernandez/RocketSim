from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


@dataclass
class OrbitalElements:
    """
    Elementos orbitales keplerianos simplificados respecto al cuerpo padre.

    Unidades:
    - a: m
    - e: adimensional
    - i, raan, argp, nu0: rad
    """
    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu0: float = 0.0


@dataclass
class CelestialBody:
    """
    Cuerpo celeste para simulación 3D simplificada.

    Parámetros:
    - mu: parámetro gravitacional [m^3/s^2]
    - radius: radio medio [m]
    - soi_radius: radio de esfera de influencia [m]
    - rotation_rate_rad_s: velocidad angular alrededor del eje +Z [rad/s]
    - initial_rotation_angle_rad: ángulo de rotación a t=0 [rad]

    Atmósfera simplificada:
    - atmosphere_scale_height_m: altura de escala [m]
    - atmosphere_surface_density_kg_m3: densidad a nivel "superficie" [kg/m^3]
    - atmosphere_limit_altitude_m: altura máxima donde se considera atmósfera
    """
    name: str
    mu: float
    radius: float
    soi_radius: float
    parent: Optional["CelestialBody"] = None
    orbit: Optional[OrbitalElements] = None
    color: str = "blue"
    rotation_rate_rad_s: float = 0.0
    initial_rotation_angle_rad: float = 0.0

    atmosphere_scale_height_m: float = 0.0
    atmosphere_surface_density_kg_m3: float = 0.0
    atmosphere_limit_altitude_m: float = 0.0

    children: list["CelestialBody"] = field(default_factory=list)

    def __post_init__(self):
        if self.parent is not None:
            self.parent.children.append(self)

    def rotation_angle_at_time(self, t: float) -> float:
        return self.initial_rotation_angle_rad + self.rotation_rate_rad_s * t

    def rotation_matrix_body_to_parent(self, t: float) -> np.ndarray:
        theta = self.rotation_angle_at_time(t)
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    def rotation_matrix_body_to_root(self, t: float) -> np.ndarray:
        r_local = self.rotation_matrix_body_to_parent(t)
        if self.parent is None:
            return r_local
        return self.parent.rotation_matrix_body_to_root(t) @ r_local

    def angular_velocity_vector_root(self, t: float) -> np.ndarray:
        omega_local = np.array([0.0, 0.0, self.rotation_rate_rad_s], dtype=float)
        rot_to_root = self.rotation_matrix_body_to_root(t)
        return rot_to_root @ omega_local

    def atmospheric_density(self, altitude_m: float) -> float:
        """
        Modelo exponencial simple:
            rho = rho0 * exp(-h / H)
        """
        if (
            self.atmosphere_surface_density_kg_m3 <= 0.0
            or self.atmosphere_scale_height_m <= 0.0
            or altitude_m < 0.0
            or altitude_m > self.atmosphere_limit_altitude_m
        ):
            return 0.0

        return self.atmosphere_surface_density_kg_m3 * math.exp(
            -altitude_m / self.atmosphere_scale_height_m
        )

    def position_velocity_in_parent_frame(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.parent is None or self.orbit is None:
            return np.zeros(3), np.zeros(3)

        a = self.orbit.a
        e = self.orbit.e
        inc = self.orbit.i
        raan = self.orbit.raan
        argp = self.orbit.argp
        nu0 = self.orbit.nu0

        mu = self.parent.mu
        n = math.sqrt(mu / (a ** 3))

        E0 = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu0 / 2.0),
            math.sqrt(1.0 + e) * math.cos(nu0 / 2.0),
        )
        M0 = E0 - e * math.sin(E0)
        M = M0 + n * t

        E = M
        for _ in range(12):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            E = E - f / fp

        cosE = math.cos(E)
        sinE = math.sin(E)

        r_pf = np.array([
            a * (cosE - e),
            a * math.sqrt(1.0 - e * e) * sinE,
            0.0,
        ])

        r_norm = np.linalg.norm(r_pf)
        factor = math.sqrt(mu * a) / max(r_norm, 1e-9)

        v_pf = np.array([
            -factor * sinE,
            factor * math.sqrt(1.0 - e * e) * cosE,
            0.0,
        ])

        rot = rotation_matrix_3d(raan, inc, argp)
        r_vec = rot @ r_pf
        v_vec = rot @ v_pf
        return r_vec, v_vec

    def position_velocity_in_root_frame(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.parent is None:
            return np.zeros(3), np.zeros(3)

        rp, vp = self.parent.position_velocity_in_root_frame(t)
        r_rel, v_rel = self.position_velocity_in_parent_frame(t)
        return rp + r_rel, vp + v_rel


def rotation_matrix_3d(raan: float, inc: float, argp: float) -> np.ndarray:
    cO = math.cos(raan)
    sO = math.sin(raan)
    ci = math.cos(inc)
    si = math.sin(inc)
    co = math.cos(argp)
    so = math.sin(argp)

    return np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ], dtype=float)