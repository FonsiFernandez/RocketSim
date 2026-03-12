import os
import sys
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kinetica.config.rockets import (
    ROCKET_PRESETS,
    get_rocket_by_name,
    get_rocket_preset_info,
)
from kinetica.config.celestial_systems import build_earth_moon_system
from kinetica.models.rocket import Rocket
from kinetica.models.stage import Stage
from kinetica.models.mission import (
    MissionPlan,
    MissionPhase,
    LaunchSite,
    BurnCommand,
    TargetOrbitCommand,
    SOIChangeCommand,
)
from kinetica.simulation.mission3d import (
    run_mission_3d,
    orbital_elements_relative,
    launch_site_speed_due_to_rotation,
)
from kinetica.ui.i18n import LANGUAGES, get_lang_code, tr


def inject_phase_styles():
    st.markdown(
        """
        <style>
        .phase-shell {
            border-radius: 18px;
            padding: 0.2rem 0.2rem 0.4rem 0.2rem;
            margin: 0.65rem 0 1.15rem 0;
            background: transparent;
        }

        .phase-header {
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            color: inherit;
        }

        .phase-title {
            font-size: 1rem;
            font-weight: 700;
            color: inherit;
            margin: 0;
            line-height: 1.2;
        }

        .phase-subtitle {
            font-size: 0.80rem;
            color: rgba(127, 127, 127, 0.95);
            margin-top: 4px;
        }

        .phase-badge {
            display: inline-block;
            margin-top: 8px;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            color: inherit;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.10);
        }

        .phase-section-title {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(127, 127, 127, 0.95);
            margin: 0.2rem 0 0.5rem 0;
        }

        div[data-testid="stNumberInput"],
        div[data-testid="stSelectbox"],
        div[data-testid="stTextInput"] {
            margin-bottom: 0.30rem;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div.phase-shell) {
            margin-bottom: 0.25rem;
        }

        /* Mejor contraste en tema oscuro */
        [data-baseweb="select"] + div,
        .stApp {
            color: inherit;
        }

        /* Tema oscuro: subtítulos un poco más claros */
        .stApp[data-teststate="dark"] .phase-subtitle,
        .stApp[data-teststate="dark"] .phase-section-title {
            color: rgba(210, 220, 235, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def phase_color_for_type(phase_type: str) -> dict:
    palette = {
        "burn": {
            "bg": "linear-gradient(135deg, rgba(183, 76, 76, 0.30), rgba(86, 30, 30, 0.30))",
            "border": "rgba(255, 120, 120, 0.22)",
            "badge": "#d96b6b",
            "label": "Burn phase",
        },
        "coast": {
            "bg": "linear-gradient(135deg, rgba(76, 114, 183, 0.28), rgba(26, 43, 84, 0.28))",
            "border": "rgba(120, 166, 255, 0.22)",
            "badge": "#5d8fe8",
            "label": "Coast phase",
        },
        "target_orbit": {
            "bg": "linear-gradient(135deg, rgba(76, 183, 145, 0.26), rgba(23, 70, 58, 0.28))",
            "border": "rgba(108, 224, 182, 0.22)",
            "badge": "#4fc79e",
            "label": "Target orbit phase",
        },
        "soi_change": {
            "bg": "linear-gradient(135deg, rgba(166, 104, 196, 0.28), rgba(61, 29, 77, 0.28))",
            "border": "rgba(204, 141, 235, 0.22)",
            "badge": "#b47ce2",
            "label": "SOI transition",
        },
    }
    return palette.get(
        phase_type,
        {
            "bg": "linear-gradient(135deg, rgba(110,110,110,0.25), rgba(40,40,40,0.25))",
            "border": "rgba(180,180,180,0.18)",
            "badge": "#8f8f8f",
            "label": "Phase",
        },
    )


def render_phase_header(idx: int, phase_name: str, phase_type: str):
    colors = phase_color_for_type(phase_type)
    st.markdown(
        f"""
        <div class="phase-shell">
            <div class="phase-header" style="background:{colors['bg']}; border-color:{colors['border']};">
                <div class="phase-title">Phase {idx} · {phase_name}</div>
                <div class="phase-subtitle">{colors['label']}</div>
                <div class="phase-badge" style="border-color:{colors['border']};">
                    {phase_type}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def phase_card_start(idx: int, name: str, phase_type: str, subtitle: str = ""):
    subtitle_block = f'<div class="phase-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="phase-card">
            <div class="phase-header">
                <div>
                    <div class="phase-title">Phase {idx} · {name}</div>
                    {subtitle_block}
                </div>
                <div class="phase-badge">{phase_type}</div>
            </div>
            <div class="phase-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def phase_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.2f} h"
    return f"{seconds/86400:.2f} d"


def format_distance_m(value_m: float) -> str:
    value_m = float(value_m)
    if abs(value_m) < 1000:
        return f"{value_m:.1f} m"
    if abs(value_m) < 1_000_000:
        return f"{value_m/1000:.2f} km"
    return f"{value_m/1_000_000:.3f} Mm"

def format_mass_kg(value: float) -> str:
    value = float(value)
    if abs(value) < 1000:
        return f"{value:.1f} kg"
    if abs(value) < 1_000_000:
        return f"{value / 1000.0:.2f} t"
    return f"{value / 1_000_000.0:.3f} kt"

def downsample_indices(n: int, max_points: int = 2500) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def make_plotly_layout(
    title: str,
    x_title: str,
    y_title: str,
    height: int = 320,
    yaxis_type: str = "linear",
):
    return go.Layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, type=yaxis_type),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )


def body_relative_positions(states: np.ndarray, times: np.ndarray, body) -> np.ndarray:
    rel = []
    for i, t in enumerate(times):
        bp, _ = body.position_velocity_in_root_frame(t)
        rel.append(states[i, :3] - bp)
    return np.array(rel)


def circle_xy(radius_km: float, n: int = 240):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return radius_km * np.cos(theta), radius_km * np.sin(theta)


def sphere_wireframe(radius_km: float, n_u: int = 40, n_v: int = 20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def add_body_sphere(fig: go.Figure, radius_km: float, name: str, colorscale: str = "Blues", opacity: float = 0.25):
    x, y, z = sphere_wireframe(radius_km)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            showscale=False,
            opacity=opacity,
            name=name,
            hoverinfo="skip",
            colorscale=colorscale,
        )
    )


def split_segments_by_label(x_vals, y_vals, labels):
    segments = []
    if len(x_vals) == 0:
        return segments

    start = 0
    current = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((x_vals[start:i], y_vals[start:i], current))
            start = i
            current = labels[i]

    segments.append((x_vals[start:], y_vals[start:], current))
    return segments


def event_marker_data(events: list[dict], times: np.ndarray, states: np.ndarray):
    if len(times) == 0:
        return []

    out = []
    for ev in events:
        ev_t = float(ev.get("time", 0.0))
        idx = int(np.searchsorted(times, ev_t))
        idx = max(0, min(idx, len(times) - 1))
        out.append({
            "event": ev,
            "index": idx,
            "state": states[idx],
            "time": times[idx],
        })
    return out


def phase_template(profile_name: str) -> List[MissionPhase]:
    if profile_name == "LEO directa":
        return [
            MissionPhase(
                name="Launch burn",
                phase_type="burn",
                burn=BurnCommand(
                    direction_mode="radial_out",
                    thrust_newtons=2_000_000.0,
                    isp_seconds=320.0,
                    duration=500.0,
                ),
            ),
            MissionPhase(
                name="Coast",
                phase_type="coast",
                coast_duration=1200.0,
            ),
        ]

    if profile_name == "LEO + ajuste orbital":
        return [
            MissionPhase(
                name="Ascent",
                phase_type="burn",
                burn=BurnCommand(
                    direction_mode="radial_out",
                    thrust_newtons=2_000_000.0,
                    isp_seconds=320.0,
                    duration=350.0,
                ),
            ),
            MissionPhase(
                name="Parking coast",
                phase_type="coast",
                coast_duration=1800.0,
            ),
            MissionPhase(
                name="Target LEO",
                phase_type="target_orbit",
                target_orbit=TargetOrbitCommand(
                    target_periapsis_altitude=180_000.0,
                    target_apoapsis_altitude=220_000.0,
                    tolerance_m=15_000.0,
                    max_duration=1800.0,
                    direction_mode="prograde",
                    thrust_newtons=400_000.0,
                    isp_seconds=340.0,
                ),
            ),
        ]

    if profile_name == "Translunar aproximada":
        return [
            MissionPhase(
                name="Parking orbit insertion",
                phase_type="target_orbit",
                target_orbit=TargetOrbitCommand(
                    target_periapsis_altitude=180_000.0,
                    target_apoapsis_altitude=220_000.0,
                    tolerance_m=20_000.0,
                    max_duration=2400.0,
                    direction_mode="prograde",
                    thrust_newtons=800_000.0,
                    isp_seconds=340.0,
                ),
            ),
            MissionPhase(
                name="Parking orbit coast",
                phase_type="coast",
                coast_duration=3600.0,
            ),
            MissionPhase(
                name="TLI burn",
                phase_type="burn",
                burn=BurnCommand(
                    direction_mode="prograde",
                    thrust_newtons=350_000.0,
                    isp_seconds=345.0,
                    duration=900.0,
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
        ]

    if profile_name == "Personalizado":
        return [
            MissionPhase(
                name="Phase 1",
                phase_type="burn",
                burn=BurnCommand(
                    direction_mode="prograde",
                    thrust_newtons=500_000.0,
                    isp_seconds=330.0,
                    duration=300.0,
                ),
            ),
        ]

    return []


def render_stage_editor(base_rocket: Rocket, lang: str, allow_stage_count_edit: bool = False) -> List[Stage]:
    st.markdown(f"### {tr('stages', lang)}")

    with st.expander(tr("edit_stages", lang), expanded=True):
        default_num_stages = len(base_rocket.stages)

        if allow_stage_count_edit:
            num_stages = st.number_input(
                "Number of stages",
                min_value=1,
                max_value=8,
                value=default_num_stages,
                step=1,
                key="num_stages",
            )
        else:
            num_stages = default_num_stages

        new_stages: List[Stage] = []

        for idx in range(int(num_stages)):
            if idx < len(base_rocket.stages):
                s = base_rocket.stages[idx]
            else:
                s = Stage(
                    dry_mass=1_000.0,
                    propellant_mass=5_000.0,
                    thrust_sl=200_000.0,
                    thrust_vac=220_000.0,
                    isp_sl=280.0,
                    isp_vac=310.0,
                    diameter=2.0,
                )

            st.markdown(f"**{tr('stage', lang)} {idx + 1}**")
            c1, c2, c3 = st.columns(3)

            with c1:
                dry = st.number_input(
                    f"{tr('dry_mass', lang)} {idx + 1} [kg]",
                    min_value=0.0,
                    value=float(s.dry_mass),
                    step=100.0,
                    key=f"dry_{idx}",
                )
                prop = st.number_input(
                    f"{tr('propellant', lang)} {idx + 1} [kg]",
                    min_value=0.0,
                    value=float(s.propellant_mass),
                    step=100.0,
                    key=f"prop_{idx}",
                )

            with c2:
                thrust_sl = st.number_input(
                    f"{tr('thrust_sl', lang)} {idx + 1} [kN]",
                    min_value=0.0,
                    value=float(s.thrust_sl / 1000.0),
                    step=10.0,
                    key=f"tsl_{idx}",
                ) * 1000.0

                thrust_vac = st.number_input(
                    f"{tr('thrust_vac', lang)} {idx + 1} [kN]",
                    min_value=0.0,
                    value=float(s.thrust_vac / 1000.0),
                    step=10.0,
                    key=f"tvac_{idx}",
                ) * 1000.0

            with c3:
                isp_sl = st.number_input(
                    f"{tr('isp_sl', lang)} {idx + 1} [s]",
                    min_value=0.0,
                    value=float(s.isp_sl),
                    step=5.0,
                    key=f"isp_sl_{idx}",
                )
                isp_vac = st.number_input(
                    f"{tr('isp_vac', lang)} {idx + 1} [s]",
                    min_value=0.0,
                    value=float(s.isp_vac),
                    step=5.0,
                    key=f"isp_vac_{idx}",
                )
                diameter = st.number_input(
                    f"{tr('diameter', lang)} {idx + 1} [m]",
                    min_value=0.1,
                    value=float(s.diameter),
                    step=0.1,
                    key=f"diam_{idx}",
                )

            new_stages.append(
                Stage(
                    dry_mass=dry,
                    propellant_mass=prop,
                    thrust_sl=thrust_sl,
                    thrust_vac=thrust_vac,
                    isp_sl=isp_sl,
                    isp_vac=isp_vac,
                    diameter=diameter,
                )
            )

    return new_stages


def default_mission_phase(index: int) -> MissionPhase:
    return MissionPhase(
        name=f"Phase {index + 1}",
        phase_type="coast",
        coast_duration=600.0,
    )


def show_rocket_summary(rocket: Rocket, lang: str):
    with st.expander(tr("rocket_summary", lang), expanded=False):
        st.write(f"**{tr('initial_total_mass', lang)}:** {rocket.total_initial_mass():,.0f} kg")
        st.write(f"**{tr('payload', lang)}:** {rocket.payload_mass:,.0f} kg")
        st.write(f"**{tr('drag_coefficient', lang)}:** {rocket.cd:.2f}")
        for idx, s in enumerate(rocket.stages, start=1):
            st.write(
                f"{tr('stage', lang)} {idx}: dry={s.dry_mass:.0f} kg, prop={s.propellant_mass:.0f} kg, "
                f"Tvac={s.thrust_vac/1000:.0f} kN, Isp_vac={s.isp_vac:.1f} s"
            )


def render_mission_editor(default_phases: List[MissionPhase], lang: str) -> List[MissionPhase]:
    st.markdown(f"### {tr('mission_phases', lang)}")

    with st.expander(tr("edit_mission", lang), expanded=True):
        inject_phase_styles()

        n_phases = st.number_input(
            tr("num_phases", lang),
            min_value=1,
            max_value=12,
            value=max(1, len(default_phases)),
            step=1,
            key="num_phases",
        )

        phases: List[MissionPhase] = []

        phase_options = ["burn", "coast", "target_orbit", "soi_change"]
        direction_options = ["prograde", "retrograde", "radial_out", "radial_in", "normal", "antinormal"]
        soi_options = ["Earth", "Moon", "Sun"]

        for i in range(int(n_phases)):
            base = default_phases[i] if i < len(default_phases) else default_mission_phase(i)

            preview_name = st.session_state.get(f"phase_name_{i}", base.name)
            preview_type = st.session_state.get(f"phase_type_{i}", base.phase_type)

            phase_box = st.container(border=True)
            with phase_box:
                render_phase_header(i + 1, preview_name, preview_type)

                st.markdown('<div class="phase-section-title">Definition</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)

                with c1:
                    phase_name = st.text_input(
                        f"{tr('phase_name', lang)} {i+1}",
                        value=preview_name,
                        key=f"phase_name_{i}",
                    )

                with c2:
                    phase_type = st.selectbox(
                        f"{tr('phase_type', lang)} {i+1}",
                        options=phase_options,
                        index=phase_options.index(preview_type),
                        key=f"phase_type_{i}",
                    )

                st.markdown('<div class="phase-section-title">Parameters</div>', unsafe_allow_html=True)

                if phase_type == "burn":
                    default_burn = base.burn or BurnCommand(
                        direction_mode="prograde",
                        thrust_newtons=500_000.0,
                        isp_seconds=300.0,
                        duration=300.0,
                    )

                    c3, c4, c5, c6 = st.columns(4)

                    with c3:
                        direction_mode = st.selectbox(
                            f"{tr('burn_direction', lang)} {i+1}",
                            options=direction_options,
                            index=direction_options.index(
                                default_burn.direction_mode if default_burn.direction_mode in direction_options else "prograde"
                            ),
                            key=f"burn_dir_{i}",
                        )

                    with c4:
                        thrust_kn = st.number_input(
                            f"{tr('thrust_vac', lang)} {i+1} [kN]",
                            min_value=0.0,
                            value=float((default_burn.thrust_newtons or 0.0) / 1000.0),
                            step=10.0,
                            key=f"burn_thrust_{i}",
                        )

                    with c5:
                        isp_s = st.number_input(
                            f"{tr('isp_vac', lang)} {i+1} [s]",
                            min_value=1.0,
                            value=float(default_burn.isp_seconds or 300.0),
                            step=5.0,
                            key=f"burn_isp_{i}",
                        )

                    with c6:
                        duration_s = st.number_input(
                            f"{tr('burn_duration', lang)} {i+1} [s]",
                            min_value=1.0,
                            value=float(default_burn.duration),
                            step=10.0,
                            key=f"burn_duration_{i}",
                        )

                    phases.append(
                        MissionPhase(
                            name=phase_name,
                            phase_type="burn",
                            burn=BurnCommand(
                                direction_mode=direction_mode,
                                thrust_newtons=thrust_kn * 1000.0,
                                isp_seconds=isp_s,
                                duration=duration_s,
                            ),
                        )
                    )

                elif phase_type == "coast":
                    default_coast = base.coast_duration if base.coast_duration > 0 else 600.0

                    coast_duration = st.number_input(
                        f"{tr('coast_duration', lang)} {i+1} [s]",
                        min_value=1.0,
                        value=float(default_coast),
                        step=60.0,
                        key=f"coast_duration_{i}",
                    )

                    phases.append(
                        MissionPhase(
                            name=phase_name,
                            phase_type="coast",
                            coast_duration=coast_duration,
                        )
                    )

                elif phase_type == "target_orbit":
                    default_target = base.target_orbit or TargetOrbitCommand(
                        target_periapsis_altitude=180_000.0,
                        target_apoapsis_altitude=220_000.0,
                    )

                    c3, c4, c5 = st.columns(3)
                    with c3:
                        target_rp_km = st.number_input(
                            f"{tr('target_periapsis', lang)} {i+1} [km]",
                            min_value=0.0,
                            value=float(default_target.target_periapsis_altitude / 1000.0),
                            step=10.0,
                            key=f"target_rp_{i}",
                        )
                    with c4:
                        target_ra_km = st.number_input(
                            f"{tr('target_apoapsis', lang)} {i+1} [km]",
                            min_value=0.0,
                            value=float(default_target.target_apoapsis_altitude / 1000.0),
                            step=10.0,
                            key=f"target_ra_{i}",
                        )
                    with c5:
                        tolerance_km = st.number_input(
                            f"{tr('tolerance', lang)} {i+1} [km]",
                            min_value=0.1,
                            value=float(default_target.tolerance_m / 1000.0),
                            step=1.0,
                            key=f"target_tol_{i}",
                        )

                    c6, c7, c8, c9 = st.columns(4)
                    with c6:
                        direction_mode = st.selectbox(
                            f"{tr('target_direction', lang)} {i+1}",
                            options=direction_options,
                            index=direction_options.index(
                                default_target.direction_mode if default_target.direction_mode in direction_options else "prograde"
                            ),
                            key=f"target_dir_{i}",
                        )
                    with c7:
                        thrust_kn = st.number_input(
                            f"{tr('thrust_vac', lang)} {i+1} [kN]",
                            min_value=0.0,
                            value=float((default_target.thrust_newtons or 0.0) / 1000.0),
                            step=10.0,
                            key=f"target_thrust_{i}",
                        )
                    with c8:
                        isp_s = st.number_input(
                            f"{tr('isp_vac', lang)} {i+1} [s]",
                            min_value=1.0,
                            value=float(default_target.isp_seconds or 300.0),
                            step=5.0,
                            key=f"target_isp_{i}",
                        )
                    with c9:
                        max_duration = st.number_input(
                            f"{tr('target_max_duration', lang)} {i+1} [s]",
                            min_value=1.0,
                            value=float(default_target.max_duration),
                            step=60.0,
                            key=f"target_max_duration_{i}",
                        )

                    phases.append(
                        MissionPhase(
                            name=phase_name,
                            phase_type="target_orbit",
                            target_orbit=TargetOrbitCommand(
                                target_periapsis_altitude=target_rp_km * 1000.0,
                                target_apoapsis_altitude=target_ra_km * 1000.0,
                                tolerance_m=tolerance_km * 1000.0,
                                max_duration=max_duration,
                                direction_mode=direction_mode,
                                thrust_newtons=thrust_kn * 1000.0,
                                isp_seconds=isp_s,
                            ),
                        )
                    )

                elif phase_type == "soi_change":
                    default_soi = base.soi_change or SOIChangeCommand(target_body_name="Moon")

                    c3, c4 = st.columns(2)
                    with c3:
                        target_body_name = st.selectbox(
                            f"{tr('target_soi', lang)} {i+1}",
                            options=soi_options,
                            index=soi_options.index(
                                default_soi.target_body_name if default_soi.target_body_name in soi_options else "Moon"
                            ),
                            key=f"soi_target_{i}",
                        )
                    with c4:
                        max_duration = st.number_input(
                            f"{tr('soi_wait', lang)} {i+1} [s]",
                            min_value=1.0,
                            value=float(default_soi.max_duration),
                            step=3600.0,
                            key=f"soi_max_duration_{i}",
                        )

                    phases.append(
                        MissionPhase(
                            name=phase_name,
                            phase_type="soi_change",
                            soi_change=SOIChangeCommand(
                                target_body_name=target_body_name,
                                max_duration=max_duration,
                            ),
                        )
                    )

    return phases


def render_events(events: list[dict], lang: str):
    if not events:
        st.info(tr("no_events", lang))
        return

    for ev in events:
        if ev.get("type") == "SOI_CHANGE":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_soi_change', lang)}: "
                f"{ev.get('from_body', '?')} → {ev.get('to_body', '?')}"
            )
        elif ev.get("type") == "TARGET_SOI_REACHED":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_target_soi', lang)} **{ev.get('body', '?')}**"
            )
        elif ev.get("type") == "TARGET_ORBIT_REACHED":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_target_orbit', lang)} **{ev.get('body', '?')}**"
            )
        elif ev.get("type") == "STAGE_SEPARATION":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_stage_sep', lang)} **{ev.get('stage_index', 0) + 1}**"
            )
        elif ev.get("type") == "NO_ACTIVE_STAGE":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_no_active_stage', lang)}"
            )
        elif ev.get("type") == "NO_THRUST":
            st.write(
                f"- **{format_seconds(ev.get('time', 0.0))}** · {tr('event_no_thrust', lang)}"
            )
        else:
            st.write(f"- {ev}")


def build_event_table(events: list[dict]):
    rows = []
    for ev in events:
        rows.append({
            "time_s": float(ev.get("time", 0.0)),
            "time": format_seconds(ev.get("time", 0.0)),
            "type": ev.get("type", ""),
            "details": str(ev),
        })
    return rows


def render_results(result, bodies, phases: List[MissionPhase], lang: str):
    times = result.times
    states = result.states

    idx = downsample_indices(len(times), max_points=2500)

    times_ds = times[idx]
    states_ds = states[idx]

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    vx = states[:, 3]
    vy = states[:, 4]
    vz = states[:, 5]
    m = states[:, 6]

    x_ds = states_ds[:, 0]
    y_ds = states_ds[:, 1]
    z_ds = states_ds[:, 2]
    vx_ds = states_ds[:, 3]
    vy_ds = states_ds[:, 4]
    vz_ds = states_ds[:, 5]
    m_ds = states_ds[:, 6]

    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    speed_ds = np.sqrt(vx_ds * vx_ds + vy_ds * vy_ds + vz_ds * vz_ds)

    final_pos = states[-1, :3]
    final_vel = states[-1, 3:6]
    final_body_name = result.dominant_bodies[-1]
    final_body = bodies[final_body_name]

    body_pos_final, body_vel_final = final_body.position_velocity_in_root_frame(times[-1])
    r_rel = final_pos - body_pos_final
    v_rel = final_vel - body_vel_final
    elems = orbital_elements_relative(final_body.mu, r_rel, v_rel)

    altitudes = []
    for i, t in enumerate(times_ds):
        body_name = result.dominant_bodies[idx[i]]
        body = bodies[body_name]
        body_pos, _ = body.position_velocity_in_root_frame(t)
        altitudes.append(np.linalg.norm(states_ds[i, :3] - body_pos) - body.radius)
    altitudes = np.array(altitudes)

    dist_earth = None
    dist_moon = None
    dist_sun = None

    if "Earth" in bodies:
        dist_earth = []
        for i, t in enumerate(times_ds):
            bp, _ = bodies["Earth"].position_velocity_in_root_frame(t)
            dist_earth.append(np.linalg.norm(states_ds[i, :3] - bp))
        dist_earth = np.array(dist_earth)

    if "Moon" in bodies:
        dist_moon = []
        for i, t in enumerate(times_ds):
            bp, _ = bodies["Moon"].position_velocity_in_root_frame(t)
            dist_moon.append(np.linalg.norm(states_ds[i, :3] - bp))
        dist_moon = np.array(dist_moon)

    if "Sun" in bodies:
        dist_sun = []
        for i, t in enumerate(times_ds):
            bp, _ = bodies["Sun"].position_velocity_in_root_frame(t)
            dist_sun.append(np.linalg.norm(states_ds[i, :3] - bp))
        dist_sun = np.array(dist_sun)

    dominant_map = {"Sun": 0, "Earth": 1, "Moon": 2}
    dominant_numeric = np.array([dominant_map.get(result.dominant_bodies[i], -1) for i in idx])

    unique_phases = []
    prev = None
    for ph in result.phase_names:
        if ph != prev:
            unique_phases.append(ph)
            prev = ph

    st.markdown(f"## {tr('results', lang)}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(tr("total_time", lang), format_seconds(times[-1] - times[0]))
    with c2:
        st.metric(tr("final_velocity", lang), f"{speed[-1]:.1f} m/s")
    with c3:
        st.metric(tr("final_mass", lang), f"{m[-1]:,.1f} kg")
    with c4:
        st.metric(tr("final_body", lang), final_body_name)

    c5, c6, c7 = st.columns(3)
    with c5:
        st.write(f"**{tr('samples', lang)}:** {len(times)}")
    with c6:
        st.write(f"**{tr('distance_to_center', lang)} ({final_body_name}):** {format_distance_m(np.linalg.norm(r_rel))}")
    with c7:
        st.write(f"**{tr('phases', lang)}:** {' → '.join(unique_phases)}")

    st.markdown(f"### {tr('final_relative_orbit', lang)}")
    st.write(f"**{tr('reference_body', lang)}:** {final_body_name}")

    if elems["a"] is None:
        st.write(tr("no_closed_orbit", lang))
    else:
        rp_alt = elems["rp"] - final_body.radius if elems["rp"] is not None else None
        ra_alt = elems["ra"] - final_body.radius if elems["ra"] is not None else None

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write(f"**{tr('semi_major_axis', lang)}:** {format_distance_m(elems['a'])}")
            st.write(f"**{tr('eccentricity', lang)}:** {elems['e']:.6f}")
        with col_b:
            if rp_alt is not None:
                st.write(f"**{tr('periapsis_altitude', lang)}:** {format_distance_m(rp_alt)}")
        with col_c:
            if ra_alt is not None:
                st.write(f"**{tr('apoapsis_altitude', lang)}:** {format_distance_m(ra_alt)}")

    tab_summary, tab_telemetry, tab_trajectory, tab_events = st.tabs(
        [tr("summary", lang), tr("telemetry", lang), tr("trajectories", lang), tr("events", lang)]
    )

    with tab_summary:
        col1, col2 = st.columns(2)

        with col1:
            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(
                x=times_ds,
                y=altitudes / 1000.0,
                mode="lines",
                name=tr("relative_altitude", lang)
            ))
            fig_alt.update_layout(make_plotly_layout(
                tr("relative_altitude", lang),
                "Tiempo / Time [s]",
                "km",
                height=300
            ))
            st.plotly_chart(fig_alt, use_container_width=True)

        with col2:
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(
                x=times_ds,
                y=speed_ds,
                mode="lines",
                name=tr("velocity", lang)
            ))
            fig_speed.update_layout(make_plotly_layout(
                tr("velocity", lang),
                "Tiempo / Time [s]",
                "m/s",
                height=300
            ))
            st.plotly_chart(fig_speed, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_mass = go.Figure()
            fig_mass.add_trace(go.Scatter(
                x=times_ds,
                y=m_ds,
                mode="lines",
                name=tr("vehicle_mass", lang)
            ))
            fig_mass.update_layout(make_plotly_layout(
                tr("vehicle_mass", lang),
                "Tiempo / Time [s]",
                "kg",
                height=300
            ))
            st.plotly_chart(fig_mass, use_container_width=True)

        with col4:
            fig_dom = go.Figure()
            fig_dom.add_trace(go.Scatter(
                x=times_ds,
                y=dominant_numeric,
                mode="lines",
                name=tr("dominant_body", lang)
            ))
            fig_dom.update_layout(make_plotly_layout(
                tr("dominant_body", lang),
                "Tiempo / Time [s]",
                tr("dominant_body", lang),
                height=300
            ))
            fig_dom.update_yaxes(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=[tr("sun", lang), tr("earth", lang), tr("moon", lang)]
            )
            st.plotly_chart(fig_dom, use_container_width=True)

    with tab_telemetry:
        col1, col2 = st.columns(2)

        with col1:
            fig_dist = go.Figure()
            if dist_earth is not None:
                fig_dist.add_trace(go.Scatter(
                    x=times_ds, y=dist_earth / 1000.0, mode="lines", name=tr("earth", lang)
                ))
            if dist_moon is not None:
                fig_dist.add_trace(go.Scatter(
                    x=times_ds, y=dist_moon / 1000.0, mode="lines", name=tr("moon", lang)
                ))
            if dist_sun is not None:
                fig_dist.add_trace(go.Scatter(
                    x=times_ds, y=dist_sun / 1000.0, mode="lines", name=tr("sun", lang)
                ))
            fig_dist.update_layout(make_plotly_layout(
                tr("distance_to_main_bodies", lang),
                "Tiempo / Time [s]",
                "km",
                height=340,
                yaxis_type="log"
            ))
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            fig_combo = go.Figure()
            fig_combo.add_trace(go.Scatter(
                x=times_ds,
                y=altitudes / 1000.0,
                mode="lines",
                name=tr("relative_altitude", lang),
                yaxis="y1"
            ))
            fig_combo.add_trace(go.Scatter(
                x=times_ds,
                y=speed_ds,
                mode="lines",
                name=tr("velocity", lang),
                yaxis="y2"
            ))
            fig_combo.update_layout(
                title=tr("altitude_and_velocity", lang),
                xaxis=dict(title="Tiempo / Time [s]"),
                yaxis=dict(title="km"),
                yaxis2=dict(
                    title="m/s",
                    overlaying="y",
                    side="right"
                ),
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_combo, use_container_width=True)

    with tab_trajectory:
        earth_body = bodies.get("Earth")
        moon_body = bodies.get("Moon")

        earth_rel = body_relative_positions(states_ds, times_ds, earth_body) if earth_body else None
        moon_rel = body_relative_positions(states_ds, times_ds, moon_body) if moon_body else None

        dominant_labels_ds = [result.dominant_bodies[i] for i in idx]
        color_by_body = {
            "Earth": "#1f77b4",
            "Moon": "#7f7f7f",
            "Sun": "#ffbf00",
        }

        event_points = event_marker_data(result.events, times_ds, states_ds)

        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            tr("earth_orbit", lang),
            tr("earth_moon_transfer", lang),
            tr("moon_orbit", lang),
            tr("global_3d_view", lang),
        ])

        with subtab1:
            st.markdown(f"### {tr('earth_centered_view', lang)}")

            if earth_rel is not None and earth_body is not None:
                col1, col2 = st.columns(2)

                with col1:
                    fig_earth_xy = go.Figure()

                    segments = split_segments_by_label(
                        earth_rel[:, 0] / 1000.0,
                        earth_rel[:, 1] / 1000.0,
                        dominant_labels_ds,
                    )
                    for sx, sy, label in segments:
                        fig_earth_xy.add_trace(go.Scatter(
                            x=sx,
                            y=sy,
                            mode="lines",
                            name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                            line=dict(color=color_by_body.get(label, "#444")),
                        ))

                    ex, ey = circle_xy(earth_body.radius / 1000.0)
                    fig_earth_xy.add_trace(go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        fill="toself",
                        name=tr("earth", lang),
                    ))

                    soi_x, soi_y = circle_xy(earth_body.soi_radius / 1000.0)
                    fig_earth_xy.add_trace(go.Scatter(
                        x=soi_x,
                        y=soi_y,
                        mode="lines",
                        line=dict(dash="dot"),
                        name=tr("earth_soi", lang),
                    ))

                    if moon_body is not None:
                        moon_rel_to_earth = []
                        for t in times_ds:
                            moon_pos, _ = moon_body.position_velocity_in_root_frame(t)
                            earth_pos, _ = earth_body.position_velocity_in_root_frame(t)
                            moon_rel_to_earth.append(moon_pos - earth_pos)
                        moon_rel_to_earth = np.array(moon_rel_to_earth)

                        fig_earth_xy.add_trace(go.Scatter(
                            x=moon_rel_to_earth[:, 0] / 1000.0,
                            y=moon_rel_to_earth[:, 1] / 1000.0,
                            mode="lines",
                            line=dict(dash="dot"),
                            name=tr("lunar_orbit_line", lang),
                        ))

                        fig_earth_xy.add_trace(go.Scatter(
                            x=[moon_rel_to_earth[-1, 0] / 1000.0],
                            y=[moon_rel_to_earth[-1, 1] / 1000.0],
                            mode="markers",
                            marker=dict(size=8),
                            name=tr("moon", lang),
                        ))

                    for pt in event_points:
                        body_pos, _ = earth_body.position_velocity_in_root_frame(pt["time"])
                        rel = pt["state"][:3] - body_pos
                        fig_earth_xy.add_trace(go.Scatter(
                            x=[rel[0] / 1000.0],
                            y=[rel[1] / 1000.0],
                            mode="markers",
                            marker=dict(size=8, symbol="x"),
                            name=pt["event"].get("type", "EVENT"),
                            showlegend=False,
                            text=[str(pt["event"])],
                            hovertemplate="%{text}<extra></extra>",
                        ))

                    fig_earth_xy.update_layout(make_plotly_layout(
                        tr("earth_centered_view", lang),
                        "x [km]",
                        "y [km]",
                        height=520
                    ))
                    fig_earth_xy.update_yaxes(scaleanchor="x", scaleratio=1)
                    st.plotly_chart(fig_earth_xy, use_container_width=True)

                with col2:
                    fig_earth_3d = go.Figure()

                    add_body_sphere(
                        fig_earth_3d,
                        radius_km=earth_body.radius / 1000.0,
                        name=tr("earth", lang),
                        colorscale="Blues",
                        opacity=0.35,
                    )

                    for label in ["Earth", "Moon", "Sun"]:
                        mask = np.array([lab == label for lab in dominant_labels_ds])
                        if np.any(mask):
                            fig_earth_3d.add_trace(go.Scatter3d(
                                x=earth_rel[mask, 0] / 1000.0,
                                y=earth_rel[mask, 1] / 1000.0,
                                z=earth_rel[mask, 2] / 1000.0,
                                mode="lines",
                                name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                                line=dict(color=color_by_body.get(label, "#444")),
                            ))

                    fig_earth_3d.add_trace(go.Scatter3d(
                        x=[earth_rel[0, 0] / 1000.0],
                        y=[earth_rel[0, 1] / 1000.0],
                        z=[earth_rel[0, 2] / 1000.0],
                        mode="markers",
                        marker=dict(size=4),
                        name=tr("start", lang),
                    ))

                    fig_earth_3d.add_trace(go.Scatter3d(
                        x=[earth_rel[-1, 0] / 1000.0],
                        y=[earth_rel[-1, 1] / 1000.0],
                        z=[earth_rel[-1, 2] / 1000.0],
                        mode="markers",
                        marker=dict(size=4),
                        name=tr("end", lang),
                    ))

                    fig_earth_3d.update_layout(
                        title=tr("earth_orbit", lang),
                        height=520,
                        margin=dict(l=20, r=20, t=50, b=20),
                        template="plotly_white",
                        scene=dict(
                            xaxis_title="x [km]",
                            yaxis_title="y [km]",
                            zaxis_title="z [km]",
                            aspectmode="data",
                        ),
                    )
                    st.plotly_chart(fig_earth_3d, use_container_width=True)

        with subtab2:
            st.markdown(f"### {tr('cislunar_view', lang)}")

            if earth_body is not None and moon_body is not None and earth_rel is not None:
                fig_cislunar = go.Figure()

                ship_earth_rel = earth_rel / 1000.0

                moon_rel_to_earth = []
                for t in times_ds:
                    moon_pos, _ = moon_body.position_velocity_in_root_frame(t)
                    earth_pos, _ = earth_body.position_velocity_in_root_frame(t)
                    moon_rel_to_earth.append(moon_pos - earth_pos)
                moon_rel_to_earth = np.array(moon_rel_to_earth) / 1000.0

                segments = split_segments_by_label(
                    ship_earth_rel[:, 0],
                    ship_earth_rel[:, 1],
                    dominant_labels_ds,
                )
                for sx, sy, label in segments:
                    fig_cislunar.add_trace(go.Scatter(
                        x=sx,
                        y=sy,
                        mode="lines",
                        name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                        line=dict(color=color_by_body.get(label, "#444")),
                    ))

                fig_cislunar.add_trace(go.Scatter(
                    x=moon_rel_to_earth[:, 0],
                    y=moon_rel_to_earth[:, 1],
                    mode="lines",
                    line=dict(dash="dot"),
                    name=tr("lunar_orbit_line", lang),
                ))

                ex, ey = circle_xy(earth_body.radius / 1000.0)
                fig_cislunar.add_trace(go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    fill="toself",
                    name=tr("earth", lang),
                ))

                earth_soi_x, earth_soi_y = circle_xy(earth_body.soi_radius / 1000.0)
                fig_cislunar.add_trace(go.Scatter(
                    x=earth_soi_x,
                    y=earth_soi_y,
                    mode="lines",
                    line=dict(dash="dot"),
                    name=tr("earth_soi", lang),
                ))

                mx, my = circle_xy(moon_body.radius / 1000.0)
                fig_cislunar.add_trace(go.Scatter(
                    x=mx + moon_rel_to_earth[-1, 0],
                    y=my + moon_rel_to_earth[-1, 1],
                    mode="lines",
                    fill="toself",
                    name=tr("moon", lang),
                ))

                moon_soi_x, moon_soi_y = circle_xy(moon_body.soi_radius / 1000.0)
                fig_cislunar.add_trace(go.Scatter(
                    x=moon_soi_x + moon_rel_to_earth[-1, 0],
                    y=moon_soi_y + moon_rel_to_earth[-1, 1],
                    mode="lines",
                    line=dict(dash="dot"),
                    name=tr("moon_soi", lang),
                ))

                for pt in event_points:
                    earth_pos, _ = earth_body.position_velocity_in_root_frame(pt["time"])
                    rel = pt["state"][:3] - earth_pos
                    fig_cislunar.add_trace(go.Scatter(
                        x=[rel[0] / 1000.0],
                        y=[rel[1] / 1000.0],
                        mode="markers",
                        marker=dict(size=8, symbol="x"),
                        name=pt["event"].get("type", "EVENT"),
                        showlegend=False,
                        text=[str(pt["event"])],
                        hovertemplate="%{text}<extra></extra>",
                    ))

                fig_cislunar.update_layout(make_plotly_layout(
                    tr("earth_moon_transfer", lang),
                    "x [km]",
                    "y [km]",
                    height=640
                ))
                fig_cislunar.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig_cislunar, use_container_width=True)

        with subtab3:
            st.markdown(f"### {tr('moon_centered_view', lang)}")

            if moon_rel is not None and moon_body is not None:
                col1, col2 = st.columns(2)

                with col1:
                    fig_moon_xy = go.Figure()

                    segments = split_segments_by_label(
                        moon_rel[:, 0] / 1000.0,
                        moon_rel[:, 1] / 1000.0,
                        dominant_labels_ds,
                    )
                    for sx, sy, label in segments:
                        fig_moon_xy.add_trace(go.Scatter(
                            x=sx,
                            y=sy,
                            mode="lines",
                            name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                            line=dict(color=color_by_body.get(label, "#444")),
                        ))

                    mx, my = circle_xy(moon_body.radius / 1000.0)
                    fig_moon_xy.add_trace(go.Scatter(
                        x=mx,
                        y=my,
                        mode="lines",
                        fill="toself",
                        name=tr("moon", lang),
                    ))

                    soi_x, soi_y = circle_xy(moon_body.soi_radius / 1000.0)
                    fig_moon_xy.add_trace(go.Scatter(
                        x=soi_x,
                        y=soi_y,
                        mode="lines",
                        line=dict(dash="dot"),
                        name=tr("moon_soi", lang),
                    ))

                    for pt in event_points:
                        body_pos, _ = moon_body.position_velocity_in_root_frame(pt["time"])
                        rel = pt["state"][:3] - body_pos
                        fig_moon_xy.add_trace(go.Scatter(
                            x=[rel[0] / 1000.0],
                            y=[rel[1] / 1000.0],
                            mode="markers",
                            marker=dict(size=8, symbol="x"),
                            name=pt["event"].get("type", "EVENT"),
                            showlegend=False,
                            text=[str(pt["event"])],
                            hovertemplate="%{text}<extra></extra>",
                        ))

                    fig_moon_xy.update_layout(make_plotly_layout(
                        tr("moon_centered_view", lang),
                        "x [km]",
                        "y [km]",
                        height=520
                    ))
                    fig_moon_xy.update_yaxes(scaleanchor="x", scaleratio=1)
                    st.plotly_chart(fig_moon_xy, use_container_width=True)

                with col2:
                    fig_moon_3d = go.Figure()

                    add_body_sphere(
                        fig_moon_3d,
                        radius_km=moon_body.radius / 1000.0,
                        name=tr("moon", lang),
                        colorscale="Greys",
                        opacity=0.35,
                    )

                    for label in ["Earth", "Moon", "Sun"]:
                        mask = np.array([lab == label for lab in dominant_labels_ds])
                        if np.any(mask):
                            fig_moon_3d.add_trace(go.Scatter3d(
                                x=moon_rel[mask, 0] / 1000.0,
                                y=moon_rel[mask, 1] / 1000.0,
                                z=moon_rel[mask, 2] / 1000.0,
                                mode="lines",
                                name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                                line=dict(color=color_by_body.get(label, "#444")),
                            ))

                    fig_moon_3d.add_trace(go.Scatter3d(
                        x=[moon_rel[0, 0] / 1000.0],
                        y=[moon_rel[0, 1] / 1000.0],
                        z=[moon_rel[0, 2] / 1000.0],
                        mode="markers",
                        marker=dict(size=4),
                        name=tr("start", lang),
                    ))

                    fig_moon_3d.add_trace(go.Scatter3d(
                        x=[moon_rel[-1, 0] / 1000.0],
                        y=[moon_rel[-1, 1] / 1000.0],
                        z=[moon_rel[-1, 2] / 1000.0],
                        mode="markers",
                        marker=dict(size=4),
                        name=tr("end", lang),
                    ))

                    fig_moon_3d.update_layout(
                        title=tr("moon_orbit", lang),
                        height=520,
                        margin=dict(l=20, r=20, t=50, b=20),
                        template="plotly_white",
                        scene=dict(
                            xaxis_title="x [km]",
                            yaxis_title="y [km]",
                            zaxis_title="z [km]",
                            aspectmode="data",
                        ),
                    )
                    st.plotly_chart(fig_moon_3d, use_container_width=True)

        with subtab4:
            st.markdown(f"### {tr('global_view_3d', lang)}")

            fig_3d = go.Figure()

            for label in ["Earth", "Moon", "Sun"]:
                mask = np.array([lab == label for lab in dominant_labels_ds])
                if np.any(mask):
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_ds[mask] / 1000.0,
                        y=y_ds[mask] / 1000.0,
                        z=z_ds[mask] / 1000.0,
                        mode="lines",
                        name=f"{tr('spacecraft_trajectory', lang)} ({label})",
                        line=dict(color=color_by_body.get(label, "#444")),
                    ))

            for body_name in ["Sun", "Earth", "Moon"]:
                if body_name in bodies:
                    bp, _ = bodies[body_name].position_velocity_in_root_frame(times_ds[-1])
                    fig_3d.add_trace(go.Scatter3d(
                        x=[bp[0] / 1000.0],
                        y=[bp[1] / 1000.0],
                        z=[bp[2] / 1000.0],
                        mode="markers",
                        marker=dict(size=5),
                        name=body_name
                    ))

            fig_3d.update_layout(
                title=tr("global_view_3d", lang),
                height=580,
                margin=dict(l=20, r=20, t=50, b=20),
                template="plotly_white",
                scene=dict(
                    xaxis_title="x [km]",
                    yaxis_title="y [km]",
                    zaxis_title="z [km]",
                    aspectmode="data",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    with tab_events:
        st.markdown(f"### {tr('mission_events', lang)}")

        render_events_by_phase(result, phases, lang)

        st.markdown("### Raw event table")
        rows = build_event_table(result.events)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Kinetica", layout="wide", page_icon=str("🚀"))

    language_name = st.sidebar.selectbox(
        tr("language", "en"),
        options=list(LANGUAGES.keys()),
        index=0,
    )
    lang = get_lang_code(language_name)

    st.title("🚀 " + tr("app_title", lang))
    st.sidebar.header(tr("configuration", lang))

    bodies = build_earth_moon_system()

    profile_options_map = {
        tr("profile_leo_direct", lang): "LEO directa",
        tr("profile_leo_adjust", lang): "LEO + ajuste orbital",
        tr("profile_tli", lang): "Translunar aproximada",
        tr("profile_custom", lang): "Personalizado",
    }

    rocket_keys = list(ROCKET_PRESETS.keys())

    rocket_name = st.sidebar.selectbox(
        tr("rocket_preset", lang),
        options=rocket_keys,
        index=rocket_keys.index("light") if "light" in rocket_keys else 0,
        format_func=lambda key: ROCKET_PRESETS[key]["label"],
    )

    preset_info = get_rocket_preset_info(rocket_name)
    base_rocket = get_rocket_by_name(rocket_name)

    mission_profile_label = st.sidebar.selectbox(
        tr("mission_profile", lang),
        options=list(profile_options_map.keys()),
        index=1,
    )
    mission_profile = profile_options_map[mission_profile_label]

    launch_body_name = st.sidebar.selectbox(
        tr("launch_body", lang),
        options=["Earth", "Moon"],
        index=0,
    )

    st.sidebar.subheader(tr("payload_and_aero", lang))
    payload = st.sidebar.number_input(
        f"{tr('payload', lang)} [kg]",
        min_value=0.0,
        value=float(base_rocket.payload_mass),
        step=50.0,
    )
    cd_value = st.sidebar.slider(
        "Cd",
        min_value=0.05,
        max_value=1.5,
        value=float(base_rocket.cd),
        step=0.01,
    )

    st.sidebar.subheader(tr("launch_site", lang))
    launch_lat_deg = st.sidebar.slider(f"{tr('latitude', lang)} [°]", -90.0, 90.0, 28.5, 0.1)
    launch_lon_deg = st.sidebar.slider(f"{tr('longitude', lang)} [°]", -180.0, 180.0, -80.6, 0.1)
    launch_alt_m = st.sidebar.number_input(f"{tr('launch_altitude', lang)} [m]", min_value=0.0, value=0.0, step=10.0)
    launch_azimuth_deg = st.sidebar.slider(f"{tr('azimuth', lang)} [°]", 0.0, 360.0, 90.0, 1.0)

    st.sidebar.subheader(tr("integration", lang))
    dt = st.sidebar.select_slider(
        f"{tr('time_step', lang)} [s]",
        options=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
        value=10.0,
    )
    t0 = st.sidebar.number_input(f"{tr('initial_time', lang)} [s]", value=0.0, step=100.0)

    launch_body = bodies[launch_body_name]
    site_rot_speed = launch_site_speed_due_to_rotation(
        t0=t0,
        central_body=launch_body,
        launch_lat_deg=launch_lat_deg,
        launch_lon_deg=launch_lon_deg,
        launch_alt_m=launch_alt_m,
    )

    st.markdown(
        f"""
        **{tr('rocket_preset', lang)}:** `{rocket_name}`  
        **{tr('mission_profile', lang)}:** `{mission_profile_label}`  
        **{tr('launch_from', lang)}:** `{launch_body_name}`  
        **{tr('latitude', lang)}/{tr('longitude', lang)}:** {launch_lat_deg:.2f}°, {launch_lon_deg:.2f}°  
        **{tr('azimuth', lang)}:** {launch_azimuth_deg:.1f}°  
        **{tr('payload', lang)}:** {payload:.1f} kg  
        """
    )

    st.info(
        f"{tr('site_rotation_speed', lang)} {launch_body_name}: **{site_rot_speed:.1f} m/s**"
    )

    st.warning(tr("warning_model", lang))

    st.sidebar.caption(preset_info["description"])

    is_custom = rocket_name == "custom"
    allow_edit = is_custom

    if not is_custom and preset_info.get("editable", False):
        allow_edit = st.sidebar.checkbox("Edit preset", value=False)

    if allow_edit:
        new_stages = render_stage_editor(
            base_rocket,
            lang,
            allow_stage_count_edit=is_custom,
        )
        rocket = Rocket(new_stages, cd=cd_value, payload_mass=payload)
    else:
        rocket = Rocket(
            stages=base_rocket.stages,
            cd=cd_value,
            payload_mass=payload,
        )

    show_rocket_summary(rocket, lang)

    default_phases = phase_template(mission_profile)
    phases = render_mission_editor(default_phases, lang)

    if st.button(tr("simulate", lang)):
        mission = MissionPlan(
            launch_site=LaunchSite(
                latitude_deg=launch_lat_deg,
                longitude_deg=launch_lon_deg,
                altitude_m=launch_alt_m,
                azimuth_deg=launch_azimuth_deg,
            ),
            phases=phases,
        )

        print("=== DEBUG UI -> RUN ===")
        print("rocket_name =", rocket_name)
        print("payload =", rocket.payload_mass)
        print("cd =", rocket.cd)

        for i, s in enumerate(rocket.stages):
            print(
                f"stage {i}: dry={s.dry_mass}, prop={s.propellant_mass}, "
                f"thrust_sl={s.thrust_sl}, thrust_vac={s.thrust_vac}, "
                f"isp_sl={s.isp_sl}, isp_vac={s.isp_vac}, diameter={s.diameter}"
            )

        for i, ph in enumerate(phases):
            print(f"phase {i}: name={ph.name}, type={ph.phase_type}")
            if ph.burn is not None:
                print(
                    f"  burn: dir={ph.burn.direction_mode}, thrust={ph.burn.thrust_newtons}, "
                    f"isp={ph.burn.isp_seconds}, duration={ph.burn.duration}"
                )
            if ph.target_orbit is not None:
                print(
                    f"  target: rp={ph.target_orbit.target_periapsis_altitude}, "
                    f"ra={ph.target_orbit.target_apoapsis_altitude}, "
                    f"tol={ph.target_orbit.tolerance_m}, "
                    f"max_duration={ph.target_orbit.max_duration}, "
                    f"thrust={ph.target_orbit.thrust_newtons}, "
                    f"isp={ph.target_orbit.isp_seconds}"
                )
        print("=======================")

        try:
            result = run_mission_3d(
                rocket=rocket,
                mission=mission,
                bodies=bodies,
                launch_body_name=launch_body_name,
                t0=t0,
                dt=dt,
            )
            render_results(result, bodies, phases, lang)
        except Exception as ex:
            st.error(f"{tr('error_simulation', lang)}: {ex}")
    else:
        st.info(tr("configure_and_simulate", lang))

def build_phase_intervals(times: np.ndarray, phase_names: list[str]):
    if len(times) == 0 or len(phase_names) == 0:
        return []

    intervals = []
    start_idx = 0
    current_phase = phase_names[0]

    for i in range(1, len(phase_names)):
        if phase_names[i] != current_phase:
            intervals.append({
                "phase_name": current_phase,
                "start_time": float(times[start_idx]),
                "end_time": float(times[i - 1]),
                "start_index": start_idx,
                "end_index": i - 1,
            })
            start_idx = i
            current_phase = phase_names[i]

    intervals.append({
        "phase_name": current_phase,
        "start_time": float(times[start_idx]),
        "end_time": float(times[len(times) - 1]),
        "start_index": start_idx,
        "end_index": len(times) - 1,
    })

    return intervals

def attach_events_to_phases(events: list[dict], phase_intervals: list[dict]):
    grouped = []

    for interval in phase_intervals:
        start_t = interval["start_time"]
        end_t = interval["end_time"]

        phase_events = []
        for ev in events:
            ev_t = float(ev.get("time", 0.0))
            if start_t <= ev_t <= end_t:
                phase_events.append(ev)

        grouped.append({
            **interval,
            "events": phase_events,
            "duration": end_t - start_t,
        })

    return grouped

def render_event_propellant_snapshot(ev: dict):
    vehicle_mass = ev.get("vehicle_mass")
    total_remaining = ev.get("total_propellant_remaining")
    total_used = ev.get("total_propellant_used")
    active_stage_index = ev.get("active_stage_index")
    stages = ev.get("stages", [])

    c1, c2, c3 = st.columns(3)

    with c1:
        if vehicle_mass is not None:
            st.write(f"**Vehicle mass:** {format_mass_kg(vehicle_mass)}")

    with c2:
        if total_remaining is not None:
            st.write(f"**Propellant remaining:** {format_mass_kg(total_remaining)}")

    with c3:
        if total_used is not None:
            st.write(f"**Propellant used:** {format_mass_kg(total_used)}")

    if not stages:
        return

    rows = []
    for stage in stages:
        idx = int(stage.get("stage_index", 0))
        dropped = bool(stage.get("dropped", False))
        remaining = float(stage.get("remaining_propellant", 0.0))
        used = float(stage.get("used_propellant", 0.0))
        initial = float(stage.get("initial_propellant", 0.0))

        if initial > 0:
            pct_used = 100.0 * used / initial
            pct_remaining = 100.0 * remaining / initial
        else:
            pct_used = 0.0
            pct_remaining = 0.0

        status = "active" if active_stage_index == idx else ("dropped" if dropped else "pending")

        rows.append({
            "Stage": idx + 1,
            "Status": status,
            "Initial propellant": f"{initial:,.1f} kg",
            "Remaining": f"{remaining:,.1f} kg",
            "Used": f"{used:,.1f} kg",
            "Remaining %": f"{pct_remaining:.1f}%",
            "Used %": f"{pct_used:.1f}%",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

def render_events_by_phase(result, phases: List[MissionPhase], lang: str):
    phase_intervals = build_phase_intervals(result.times, result.phase_names)
    grouped = attach_events_to_phases(result.events, phase_intervals)

    phase_type_map = {ph.name: ph.phase_type for ph in phases}

    if not grouped:
        st.info(tr("no_events", lang))
        return

    for idx, item in enumerate(grouped, start=1):
        phase_name = item["phase_name"]
        phase_type = phase_type_map.get(phase_name, "unknown")
        start_t = item["start_time"]
        end_t = item["end_time"]
        duration = item["duration"]
        events = item["events"]

        with st.expander(
            f"Phase {idx} · {phase_name} · {phase_type} · "
            f"{format_seconds(start_t)} → {format_seconds(end_t)} "
            f"({format_seconds(duration)})",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Start:** {format_seconds(start_t)}")
            with c2:
                st.write(f"**End:** {format_seconds(end_t)}")
            with c3:
                st.write(f"**Duration:** {format_seconds(duration)}")

            if not events:
                st.write("No events in this phase.")
                continue

            for ev in events:
                ev_t = float(ev.get("time", 0.0))
                ev_type = ev.get("type", "EVENT")

                if ev_type == "SOI_CHANGE":
                    desc = f"{ev.get('from_body', '?')} → {ev.get('to_body', '?')}"
                elif ev_type == "TARGET_SOI_REACHED":
                    desc = f"Target SOI reached: {ev.get('body', '?')}"
                elif ev_type == "TARGET_ORBIT_REACHED":
                    desc = f"Target orbit reached: {ev.get('body', '?')}"
                elif ev_type == "STAGE_SEPARATION":
                    desc = f"Stage separation: {ev.get('stage_index', 0) + 1}"
                elif ev_type == "NO_ACTIVE_STAGE":
                    desc = "No active stage"
                elif ev_type == "NO_THRUST":
                    desc = "No thrust"
                elif ev_type == "INVALID_MDOT":
                    desc = f"Invalid mass flow rate: {ev.get('mdot', 0.0):.6f} kg/s"
                elif ev_type == "IMPACT":
                    desc = f"Impact on {ev.get('body', '?')}"
                else:
                    desc = str(ev)

                st.markdown(f"#### {format_seconds(ev_t)} · `{ev_type}`")
                st.write(desc)

                render_event_propellant_snapshot(ev)

                st.markdown("---")

if __name__ == "__main__":
    main()