from __future__ import annotations

import math
from kinetica.models.celestial_body import CelestialBody, OrbitalElements


def build_earth_moon_system():
    """
    Sistema simplificado:
    - Sol raíz
    - Tierra orbitando el Sol
    - Luna orbitando la Tierra

    Incluye:
    - rotación
    - atmósfera exponencial simple
    """
    sun_rotation_period_s = 25.38 * 24.0 * 3600.0

    sun = CelestialBody(
        name="Sun",
        mu=1.32712440018e20,
        radius=696_340_000.0,
        soi_radius=1.0e20,
        parent=None,
        orbit=None,
        color="yellow",
        rotation_rate_rad_s=2.0 * math.pi / sun_rotation_period_s,
        initial_rotation_angle_rad=0.0,
        atmosphere_scale_height_m=0.0,
        atmosphere_surface_density_kg_m3=0.0,
        atmosphere_limit_altitude_m=0.0,
    )

    earth_sidereal_day_s = 86164.0905

    earth = CelestialBody(
        name="Earth",
        mu=3.986004418e14,
        radius=6_378_137.0,
        soi_radius=924_000_000.0,
        parent=sun,
        orbit=OrbitalElements(
            a=149_597_870_700.0,
            e=0.0167,
            i=math.radians(0.0),
            raan=0.0,
            argp=0.0,
            nu0=0.0,
        ),
        color="blue",
        rotation_rate_rad_s=2.0 * math.pi / earth_sidereal_day_s,
        initial_rotation_angle_rad=0.0,
        atmosphere_scale_height_m=8500.0,
        atmosphere_surface_density_kg_m3=1.225,
        atmosphere_limit_altitude_m=150_000.0,
    )

    moon_sidereal_rotation_s = 27.321661 * 24.0 * 3600.0

    moon = CelestialBody(
        name="Moon",
        mu=4.9048695e12,
        radius=1_737_400.0,
        soi_radius=66_100_000.0,
        parent=earth,
        orbit=OrbitalElements(
            a=384_400_000.0,
            e=0.0549,
            i=math.radians(5.145),
            raan=0.0,
            argp=0.0,
            nu0=0.0,
        ),
        color="gray",
        rotation_rate_rad_s=2.0 * math.pi / moon_sidereal_rotation_s,
        initial_rotation_angle_rad=0.0,
        atmosphere_scale_height_m=0.0,
        atmosphere_surface_density_kg_m3=0.0,
        atmosphere_limit_altitude_m=0.0,
    )

    return {
        "Sun": sun,
        "Earth": earth,
        "Moon": moon,
    }