from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np

BurnDirectionMode = Literal[
    "prograde",
    "retrograde",
    "radial_out",
    "radial_in",
    "normal",
    "antinormal",
    "fixed",
]

PhaseType = Literal[
    "burn",
    "coast",
    "target_orbit",
    "soi_change",
]


@dataclass
class BurnCommand:
    direction_mode: BurnDirectionMode = "prograde"
    thrust_newtons: Optional[float] = None
    isp_seconds: Optional[float] = None
    duration: float = 0.0
    fixed_direction_eci: Optional[np.ndarray] = None


@dataclass
class TargetOrbitCommand:
    """
    Target orbit alrededor del cuerpo dominante actual.
    """
    target_periapsis_altitude: float
    target_apoapsis_altitude: float
    tolerance_m: float = 10_000.0
    max_duration: float = 7200.0
    direction_mode: BurnDirectionMode = "prograde"
    thrust_newtons: Optional[float] = None
    isp_seconds: Optional[float] = None


@dataclass
class SOIChangeCommand:
    """
    Esperar hasta entrar en la SOI de un cuerpo objetivo.
    """
    target_body_name: str
    max_duration: float = 30 * 24 * 3600.0


@dataclass
class MissionPhase:
    name: str
    phase_type: PhaseType
    burn: Optional[BurnCommand] = None
    target_orbit: Optional[TargetOrbitCommand] = None
    soi_change: Optional[SOIChangeCommand] = None
    coast_duration: float = 0.0


@dataclass
class LaunchSite:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    azimuth_deg: float


@dataclass
class MissionPlan:
    launch_site: LaunchSite
    phases: list[MissionPhase] = field(default_factory=list)