from rocketsim.models.stage import Stage
from rocketsim.models.rocket import Rocket


# =========================
# COHETES GENÉRICOS
# =========================

def light_launcher() -> Rocket:
    """Cohete pequeño de 2 etapas, genérico."""
    stage1 = Stage(
        dry_mass=8_000.0,
        propellant_mass=60_000.0,
        thrust_sl=1_200_000.0,
        thrust_vac=1_350_000.0,
        isp_sl=280.0,
        isp_vac=310.0,
        diameter=2.5,
    )

    stage2 = Stage(
        dry_mass=2_000.0,
        propellant_mass=15_000.0,
        thrust_sl=0.0,
        thrust_vac=300_000.0,
        isp_sl=0.0,
        isp_vac=340.0,
        diameter=2.5,
    )

    payload_mass = 500.0
    cd = 0.3
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


def medium_f9_like() -> Rocket:
    """Cohete tipo Falcon 9 muy simplificado."""
    stage1 = Stage(
        dry_mass=25_000.0,
        propellant_mass=395_000.0,
        thrust_sl=7_600_000.0,
        thrust_vac=8_000_000.0,
        isp_sl=282.0,
        isp_vac=311.0,
        diameter=3.7,
    )

    stage2 = Stage(
        dry_mass=4_000.0,
        propellant_mass=90_000.0,
        thrust_sl=0.0,
        thrust_vac=934_000.0,
        isp_sl=0.0,
        isp_vac=348.0,
        diameter=3.7,
    )

    payload_mass = 10_000.0
    cd = 0.3
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# CUSTOM
# =========================

def custom_rocket() -> Rocket:
    """
    Cohete base editable para que el usuario cree el suyo.
    """
    stage1 = Stage(
        dry_mass=8_000.0,
        propellant_mass=60_000.0,
        thrust_sl=1_200_000.0,
        thrust_vac=1_350_000.0,
        isp_sl=280.0,
        isp_vac=310.0,
        diameter=2.5,
    )

    stage2 = Stage(
        dry_mass=2_000.0,
        propellant_mass=15_000.0,
        thrust_sl=0.0,
        thrust_vac=300_000.0,
        isp_sl=0.0,
        isp_vac=340.0,
        diameter=2.5,
    )

    return Rocket([stage1, stage2], cd=0.3, payload_mass=500.0)


# =========================
# MIURA 1 (SUBORBITAL)
# =========================

def miura1_suborbital() -> Rocket:
    """
    Modelo aproximado de MIURA 1 (suborbital).
    Supuesto:
      - Masa lanzamiento ≈ 2.6 t
      - Payload ≈ 100 kg
    """
    diameter = 0.7

    stage = Stage(
        dry_mass=1_100.0,
        propellant_mass=1_400.0,
        thrust_sl=30_000.0,
        thrust_vac=35_000.0,
        isp_sl=235.0,
        isp_vac=255.0,
        diameter=diameter,
    )

    payload_mass = 100.0
    cd = 0.4
    return Rocket([stage], cd=cd, payload_mass=payload_mass)


# =========================
# MIURA 5 (ORBITAL, ~450 kg a SSO)
# =========================

def miura5_orbital_450() -> Rocket:
    """
    Modelo aproximado de MIURA 5 versión ~450 kg a ~500 km.
    Masa lanzamiento ~32 t (modelo simplificado).
    """
    diameter = 2.0

    stage1 = Stage(
        dry_mass=6_000.0,
        propellant_mass=24_000.0,
        thrust_sl=5 * 190_000.0,
        thrust_vac=5 * 210_000.0,
        isp_sl=300.0,
        isp_vac=330.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=800.0,
        propellant_mass=800.0,
        thrust_sl=0.0,
        thrust_vac=75_000.0,
        isp_sl=0.0,
        isp_vac=323.0,
        diameter=diameter * 0.9,
    )

    payload_mass = 450.0
    cd = 0.35
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# ELECTRON (Rocket Lab)
# =========================

def electron() -> Rocket:
    """
    Rocket Lab Electron aproximado.
    """
    diameter = 1.2

    stage1 = Stage(
        dry_mass=2_000.0,
        propellant_mass=10_000.0,
        thrust_sl=190_000.0,
        thrust_vac=220_000.0,
        isp_sl=300.0,
        isp_vac=320.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=500.0,
        propellant_mass=1_500.0,
        thrust_sl=0.0,
        thrust_vac=25_000.0,
        isp_sl=0.0,
        isp_vac=345.0,
        diameter=diameter * 0.9,
    )

    payload_mass = 300.0
    cd = 0.35
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# VEGA-C
# =========================

def vega_c() -> Rocket:
    """
    Vega-C aproximado.
    """
    diameter = 3.0

    stage1 = Stage(
        dry_mass=10_000.0,
        propellant_mass=110_000.0,
        thrust_sl=3_000_000.0,
        thrust_vac=3_500_000.0,
        isp_sl=280.0,
        isp_vac=295.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=3_000.0,
        propellant_mass=15_000.0,
        thrust_sl=0.0,
        thrust_vac=250_000.0,
        isp_sl=0.0,
        isp_vac=325.0,
        diameter=diameter * 0.7,
    )

    payload_mass = 2_300.0
    cd = 0.35
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# SOYUZ 2.1b + Fregat
# =========================

def soyuz_2_1b() -> Rocket:
    """
    Modelo muy simplificado de Soyuz 2.1b + Fregat.
    """
    diameter = 2.95

    stage1 = Stage(
        dry_mass=25_000.0,
        propellant_mass=250_000.0,
        thrust_sl=4_100_000.0,
        thrust_vac=4_450_000.0,
        isp_sl=310.0,
        isp_vac=330.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=9_000.0,
        propellant_mass=40_000.0,
        thrust_sl=0.0,
        thrust_vac=800_000.0,
        isp_sl=0.0,
        isp_vac=340.0,
        diameter=diameter * 0.9,
    )

    payload_mass = 7_500.0
    cd = 0.35
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# ARIANE 6.2
# =========================

def ariane6_2() -> Rocket:
    """
    Ariane 6.2 simplificado.
    """
    diameter = 5.4

    stage1 = Stage(
        dry_mass=45_000.0,
        propellant_mass=480_000.0,
        thrust_sl=7_000_000.0,
        thrust_vac=7_500_000.0,
        isp_sl=300.0,
        isp_vac=330.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=10_000.0,
        propellant_mass=100_000.0,
        thrust_sl=0.0,
        thrust_vac=180_000.0,
        isp_sl=0.0,
        isp_vac=445.0,
        diameter=4.5,
    )

    payload_mass = 11_000.0
    cd = 0.32
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# SATURN V (LEO)
# =========================

def saturnv_leo() -> Rocket:
    """
    Saturn V aproximado para LEO.
    """
    diameter = 10.1

    stage1 = Stage(
        dry_mass=130_000.0,
        propellant_mass=2_000_000.0,
        thrust_sl=33_000_000.0,
        thrust_vac=35_000_000.0,
        isp_sl=265.0,
        isp_vac=304.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=40_000.0,
        propellant_mass=500_000.0,
        thrust_sl=0.0,
        thrust_vac=5_000_000.0,
        isp_sl=0.0,
        isp_vac=421.0,
        diameter=diameter * 0.8,
    )

    payload_mass = 118_000.0
    cd = 0.27
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# STARSHIP + SUPER HEAVY
# =========================

def starship_orbital() -> Rocket:
    """
    Modelo muy simplificado de Starship + Super Heavy.
    """
    diameter = 9.0

    stage1 = Stage(
        dry_mass=200_000.0,
        propellant_mass=3_400_000.0,
        thrust_sl=72_000_000.0,
        thrust_vac=76_000_000.0,
        isp_sl=330.0,
        isp_vac=356.0,
        diameter=diameter,
    )

    stage2 = Stage(
        dry_mass=120_000.0,
        propellant_mass=1_200_000.0,
        thrust_sl=0.0,
        thrust_vac=12_000_000.0,
        isp_sl=0.0,
        isp_vac=380.0,
        diameter=diameter,
    )

    payload_mass = 100_000.0
    cd = 0.25
    return Rocket([stage1, stage2], cd=cd, payload_mass=payload_mass)


# =========================
# REGISTRO DE PRESETS
# =========================

ROCKET_PRESETS = {
    "custom": {
        "label": "Custom",
        "factory": custom_rocket,
        "editable": True,
        "description": "Cohete completamente editable por el usuario.",
    },
    "light": {
        "label": "Light launcher",
        "factory": light_launcher,
        "editable": True,
        "description": "Lanzador genérico ligero de 2 etapas.",
    },
    "medium_f9": {
        "label": "Medium F9-like",
        "factory": medium_f9_like,
        "editable": True,
        "description": "Lanzador medio inspirado en Falcon 9.",
    },
    "miura1": {
        "label": "MIURA 1",
        "factory": miura1_suborbital,
        "editable": False,
        "description": "Modelo aproximado suborbital.",
    },
    "miura5_450": {
        "label": "MIURA 5 (~450 kg)",
        "factory": miura5_orbital_450,
        "editable": False,
        "description": "Modelo aproximado orbital ligero.",
    },
    "electron": {
        "label": "Electron",
        "factory": electron,
        "editable": False,
        "description": "Rocket Lab Electron aproximado.",
    },
    "vega_c": {
        "label": "Vega-C",
        "factory": vega_c,
        "editable": False,
        "description": "Vega-C simplificado.",
    },
    "soyuz_2_1b": {
        "label": "Soyuz 2.1b",
        "factory": soyuz_2_1b,
        "editable": False,
        "description": "Soyuz 2.1b + Fregat aproximado.",
    },
    "ariane6_2": {
        "label": "Ariane 6.2",
        "factory": ariane6_2,
        "editable": False,
        "description": "Ariane 6.2 simplificado.",
    },
    "saturnv_leo": {
        "label": "Saturn V (LEO)",
        "factory": saturnv_leo,
        "editable": False,
        "description": "Saturn V aproximado para inserción en LEO.",
    },
    "starship_orbital": {
        "label": "Starship orbital",
        "factory": starship_orbital,
        "editable": False,
        "description": "Starship + Super Heavy muy simplificado.",
    },
}


def get_rocket_by_name(name: str) -> Rocket:
    key = name.lower()
    if key not in ROCKET_PRESETS:
        raise ValueError(f"Cohete '{name}' no reconocido. Usa uno de: {list(ROCKET_PRESETS.keys())}")
    return ROCKET_PRESETS[key]["factory"]()


def get_rocket_preset_info(name: str) -> dict:
    key = name.lower()
    if key not in ROCKET_PRESETS:
        raise ValueError(f"Cohete '{name}' no reconocido. Usa uno de: {list(ROCKET_PRESETS.keys())}")
    return ROCKET_PRESETS[key]