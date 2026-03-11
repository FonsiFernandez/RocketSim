import math

from kinetica.config.rockets import get_rocket_by_name
from kinetica.config.celestial_systems import build_earth_moon_system
from kinetica.models.mission import (
    MissionPlan,
    MissionPhase,
    LaunchSite,
    BurnCommand,
    SOIChangeCommand,
    TargetOrbitCommand,
)
from kinetica.simulation.mission3d import run_mission_3d


def test_mission3d_smoke():
    bodies = build_earth_moon_system()
    rocket = get_rocket_by_name("light")

    mission = MissionPlan(
        launch_site=LaunchSite(
            latitude_deg=28.5,
            longitude_deg=-80.6,
            altitude_m=0.0,
            azimuth_deg=90.0,
        ),
        phases=[
            MissionPhase(
                name="Launch burn",
                phase_type="burn",
                burn=BurnCommand(
                    direction_mode="radial_out",
                    thrust_newtons=2_000_000.0,
                    isp_seconds=320.0,
                    duration=300.0,
                ),
            ),
            MissionPhase(
                name="Coast",
                phase_type="coast",
                coast_duration=1800.0,
            ),
            MissionPhase(
                name="Raise orbit",
                phase_type="target_orbit",
                target_orbit=TargetOrbitCommand(
                    target_periapsis_altitude=180_000.0,
                    target_apoapsis_altitude=300_000.0,
                    tolerance_m=25_000.0,
                    max_duration=1200.0,
                    direction_mode="prograde",
                    thrust_newtons=400_000.0,
                    isp_seconds=340.0,
                ),
            ),
            MissionPhase(
                name="Wait Moon SOI",
                phase_type="soi_change",
                soi_change=SOIChangeCommand(
                    target_body_name="Moon",
                    max_duration=8 * 24 * 3600.0,
                ),
            ),
        ],
    )

    result = run_mission_3d(
        rocket=rocket,
        mission=mission,
        bodies=bodies,
        launch_body_name="Earth",
        t0=0.0,
        dt=10.0,
    )

    assert len(result.times) > 10
    assert result.states.shape[1] == 7