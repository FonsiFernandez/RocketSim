from dataclasses import dataclass
from typing import Callable

from kinetica.models.stage import Stage
from kinetica.models.rocket import Rocket


# ============================================================
# PRESET METADATA
# ============================================================

@dataclass(frozen=True, slots=True)
class RocketPreset:
    key: str
    label: str
    category: str
    factory: Callable[[], Rocket]
    editable: bool
    description: str


# ============================================================
# HELPERS
# ============================================================

def _rocket(stages: list[Stage], payload_mass: float, cd: float) -> Rocket:
    return Rocket(stages=stages, payload_mass=payload_mass, cd=cd)


def _stage(
    *,
    dry_mass: float,
    propellant_mass: float,
    thrust_sl: float,
    thrust_vac: float,
    isp_sl: float,
    isp_vac: float,
    diameter: float,
) -> Stage:
    return Stage(
        dry_mass=dry_mass,
        propellant_mass=propellant_mass,
        thrust_sl=thrust_sl,
        thrust_vac=thrust_vac,
        isp_sl=isp_sl,
        isp_vac=isp_vac,
        diameter=diameter,
    )


# ============================================================
# GENERIC / TRAINING
# ============================================================

def custom_rocket() -> Rocket:
    stage1 = _stage(
        dry_mass=8_000.0,
        propellant_mass=60_000.0,
        thrust_sl=1_200_000.0,
        thrust_vac=1_350_000.0,
        isp_sl=280.0,
        isp_vac=310.0,
        diameter=2.5,
    )
    stage2 = _stage(
        dry_mass=2_000.0,
        propellant_mass=15_000.0,
        thrust_sl=0.0,
        thrust_vac=300_000.0,
        isp_sl=0.0,
        isp_vac=340.0,
        diameter=2.5,
    )
    return _rocket([stage1, stage2], payload_mass=500.0, cd=0.32)


def light_launcher() -> Rocket:
    stage1 = _stage(
        dry_mass=9_000.0,
        propellant_mass=72_000.0,
        thrust_sl=1_500_000.0,
        thrust_vac=1_680_000.0,
        isp_sl=285.0,
        isp_vac=315.0,
        diameter=2.7,
    )
    stage2 = _stage(
        dry_mass=2_300.0,
        propellant_mass=18_000.0,
        thrust_sl=0.0,
        thrust_vac=320_000.0,
        isp_sl=0.0,
        isp_vac=345.0,
        diameter=2.4,
    )
    return _rocket([stage1, stage2], payload_mass=800.0, cd=0.32)


def medium_generic() -> Rocket:
    stage1 = _stage(
        dry_mass=22_000.0,
        propellant_mass=220_000.0,
        thrust_sl=4_200_000.0,
        thrust_vac=4_650_000.0,
        isp_sl=290.0,
        isp_vac=322.0,
        diameter=3.5,
    )
    stage2 = _stage(
        dry_mass=5_000.0,
        propellant_mass=45_000.0,
        thrust_sl=0.0,
        thrust_vac=820_000.0,
        isp_sl=0.0,
        isp_vac=355.0,
        diameter=3.2,
    )
    return _rocket([stage1, stage2], payload_mass=5_000.0, cd=0.31)


def heavy_generic() -> Rocket:
    stage1 = _stage(
        dry_mass=65_000.0,
        propellant_mass=700_000.0,
        thrust_sl=11_500_000.0,
        thrust_vac=12_600_000.0,
        isp_sl=300.0,
        isp_vac=330.0,
        diameter=5.0,
    )
    stage2 = _stage(
        dry_mass=12_000.0,
        propellant_mass=120_000.0,
        thrust_sl=0.0,
        thrust_vac=1_600_000.0,
        isp_sl=0.0,
        isp_vac=440.0,
        diameter=4.5,
    )
    return _rocket([stage1, stage2], payload_mass=18_000.0, cd=0.29)


# ============================================================
# PLD SPACE
# ============================================================

def miura1_suborbital() -> Rocket:
    stage1 = _stage(
        dry_mass=1_150.0,
        propellant_mass=1_370.0,
        thrust_sl=30_000.0,
        thrust_vac=35_000.0,
        isp_sl=235.0,
        isp_vac=255.0,
        diameter=0.7,
    )
    return _rocket([stage1], payload_mass=100.0, cd=0.40)


def miura5_orbital() -> Rocket:
    stage1 = _stage(
        dry_mass=5_800.0,
        propellant_mass=24_000.0,
        thrust_sl=5 * 190_000.0,
        thrust_vac=5 * 210_000.0,
        isp_sl=300.0,
        isp_vac=330.0,
        diameter=2.0,
    )
    stage2 = _stage(
        dry_mass=1_000.0,
        propellant_mass=2_000.0,
        thrust_sl=0.0,
        thrust_vac=75_000.0,
        isp_sl=0.0,
        isp_vac=323.0,
        diameter=1.8,
    )
    return _rocket([stage1, stage2], payload_mass=540.0, cd=0.35)


# ============================================================
# ROCKET LAB
# ============================================================

def electron() -> Rocket:
    stage1 = _stage(
        dry_mass=2_050.0,
        propellant_mass=9_500.0,
        thrust_sl=9 * 24_900.0,
        thrust_vac=9 * 25_800.0,
        isp_sl=311.0,
        isp_vac=327.0,
        diameter=1.2,
    )
    stage2 = _stage(
        dry_mass=500.0,
        propellant_mass=2_000.0,
        thrust_sl=0.0,
        thrust_vac=25_800.0,
        isp_sl=0.0,
        isp_vac=343.0,
        diameter=1.2,
    )
    return _rocket([stage1, stage2], payload_mass=300.0, cd=0.36)


def neutron() -> Rocket:
    stage1 = _stage(
        dry_mass=42_000.0,
        propellant_mass=390_000.0,
        thrust_sl=7_600_000.0,
        thrust_vac=8_300_000.0,
        isp_sl=305.0,
        isp_vac=335.0,
        diameter=7.0,
    )
    stage2 = _stage(
        dry_mass=9_000.0,
        propellant_mass=39_000.0,
        thrust_sl=0.0,
        thrust_vac=1_100_000.0,
        isp_sl=0.0,
        isp_vac=365.0,
        diameter=5.0,
    )
    return _rocket([stage1, stage2], payload_mass=13_000.0, cd=0.28)


# ============================================================
# SPACEX
# ============================================================

def falcon9_block5() -> Rocket:
    stage1 = _stage(
        dry_mass=25_600.0,
        propellant_mass=411_000.0,
        thrust_sl=7_607_000.0,
        thrust_vac=8_227_000.0,
        isp_sl=282.0,
        isp_vac=311.0,
        diameter=3.7,
    )
    stage2 = _stage(
        dry_mass=4_000.0,
        propellant_mass=111_500.0,
        thrust_sl=0.0,
        thrust_vac=981_000.0,
        isp_sl=0.0,
        isp_vac=348.0,
        diameter=3.7,
    )
    return _rocket([stage1, stage2], payload_mass=22_800.0, cd=0.30)


def falcon_heavy() -> Rocket:
    # Modelo 2-stage equivalente: los tres cores quedan absorbidos en una primera etapa compuesta.
    stage1 = _stage(
        dry_mass=77_000.0,
        propellant_mass=1_233_000.0,
        thrust_sl=22_820_000.0,
        thrust_vac=24_680_000.0,
        isp_sl=282.0,
        isp_vac=311.0,
        diameter=3.7,
    )
    stage2 = _stage(
        dry_mass=4_000.0,
        propellant_mass=111_500.0,
        thrust_sl=0.0,
        thrust_vac=981_000.0,
        isp_sl=0.0,
        isp_vac=348.0,
        diameter=3.7,
    )
    return _rocket([stage1, stage2], payload_mass=63_800.0, cd=0.29)


def starship_block2_like() -> Rocket:
    stage1 = _stage(
        dry_mass=230_000.0,
        propellant_mass=3_400_000.0,
        thrust_sl=74_000_000.0,
        thrust_vac=80_000_000.0,
        isp_sl=330.0,
        isp_vac=356.0,
        diameter=9.0,
    )
    stage2 = _stage(
        dry_mass=120_000.0,
        propellant_mass=1_200_000.0,
        thrust_sl=0.0,
        thrust_vac=13_000_000.0,
        isp_sl=0.0,
        isp_vac=378.0,
        diameter=9.0,
    )
    return _rocket([stage1, stage2], payload_mass=100_000.0, cd=0.25)


# ============================================================
# EUROPE
# ============================================================

def vega_c() -> Rocket:
    # Simplificación fuerte: las 3 etapas sólidas + AVUM+ quedan resumidas en 2 etapas equivalentes.
    stage1 = _stage(
        dry_mass=23_000.0,
        propellant_mass=175_000.0,
        thrust_sl=4_500_000.0,
        thrust_vac=4_800_000.0,
        isp_sl=280.0,
        isp_vac=295.0,
        diameter=3.0,
    )
    stage2 = _stage(
        dry_mass=2_200.0,
        propellant_mass=13_000.0,
        thrust_sl=0.0,
        thrust_vac=245_000.0,
        isp_sl=0.0,
        isp_vac=320.0,
        diameter=2.2,
    )
    return _rocket([stage1, stage2], payload_mass=3_300.0, cd=0.34)


def ariane62() -> Rocket:
    # Core + boosters resumidos en una primera etapa equivalente.
    stage1 = _stage(
        dry_mass=92_000.0,
        propellant_mass=760_000.0,
        thrust_sl=11_000_000.0,
        thrust_vac=12_200_000.0,
        isp_sl=285.0,
        isp_vac=330.0,
        diameter=5.4,
    )
    stage2 = _stage(
        dry_mass=15_000.0,
        propellant_mass=31_000.0,
        thrust_sl=0.0,
        thrust_vac=180_000.0,
        isp_sl=0.0,
        isp_vac=457.0,
        diameter=5.4,
    )
    return _rocket([stage1, stage2], payload_mass=10_300.0, cd=0.30)


def ariane64() -> Rocket:
    stage1 = _stage(
        dry_mass=120_000.0,
        propellant_mass=860_000.0,
        thrust_sl=16_000_000.0,
        thrust_vac=17_800_000.0,
        isp_sl=285.0,
        isp_vac=330.0,
        diameter=5.4,
    )
    stage2 = _stage(
        dry_mass=15_000.0,
        propellant_mass=31_000.0,
        thrust_sl=0.0,
        thrust_vac=180_000.0,
        isp_sl=0.0,
        isp_vac=457.0,
        diameter=5.4,
    )
    return _rocket([stage1, stage2], payload_mass=21_600.0, cd=0.30)


# ============================================================
# ULA / BLUE ORIGIN
# ============================================================

def atlas_v_401() -> Rocket:
    stage1 = _stage(
        dry_mass=21_000.0,
        propellant_mass=284_000.0,
        thrust_sl=3_827_000.0,
        thrust_vac=4_150_000.0,
        isp_sl=311.0,
        isp_vac=338.0,
        diameter=3.8,
    )
    stage2 = _stage(
        dry_mass=2_250.0,
        propellant_mass=20_800.0,
        thrust_sl=0.0,
        thrust_vac=106_000.0,
        isp_sl=0.0,
        isp_vac=451.0,
        diameter=3.0,
    )
    return _rocket([stage1, stage2], payload_mass=9_800.0, cd=0.31)


def vulcan_centaur() -> Rocket:
    stage1 = _stage(
        dry_mass=35_000.0,
        propellant_mass=500_000.0,
        thrust_sl=2 * 2_450_000.0,
        thrust_vac=2 * 2_650_000.0,
        isp_sl=310.0,
        isp_vac=340.0,
        diameter=5.4,
    )
    stage2 = _stage(
        dry_mass=8_500.0,
        propellant_mass=54_000.0,
        thrust_sl=0.0,
        thrust_vac=2 * 106_000.0,
        isp_sl=0.0,
        isp_vac=451.0,
        diameter=5.4,
    )
    return _rocket([stage1, stage2], payload_mass=18_000.0, cd=0.29)


def new_glenn() -> Rocket:
    stage1 = _stage(
        dry_mass=100_000.0,
        propellant_mass=1_800_000.0,
        thrust_sl=17_100_000.0,
        thrust_vac=18_500_000.0,
        isp_sl=310.0,
        isp_vac=340.0,
        diameter=7.0,
    )
    stage2 = _stage(
        dry_mass=20_000.0,
        propellant_mass=160_000.0,
        thrust_sl=0.0,
        thrust_vac=2 * 710_000.0,
        isp_sl=0.0,
        isp_vac=450.0,
        diameter=7.0,
    )
    return _rocket([stage1, stage2], payload_mass=45_000.0, cd=0.28)


# ============================================================
# ISRO
# ============================================================

def pslv_xl() -> Rocket:
    # Modelo equivalente simplificado de un lanzador de 4 etapas con strap-ons.
    stage1 = _stage(
        dry_mass=35_000.0,
        propellant_mass=270_000.0,
        thrust_sl=5_200_000.0,
        thrust_vac=5_700_000.0,
        isp_sl=270.0,
        isp_vac=295.0,
        diameter=2.8,
    )
    stage2 = _stage(
        dry_mass=4_500.0,
        propellant_mass=40_000.0,
        thrust_sl=0.0,
        thrust_vac=800_000.0,
        isp_sl=0.0,
        isp_vac=315.0,
        diameter=2.8,
    )
    return _rocket([stage1, stage2], payload_mass=3_800.0, cd=0.33)


def lvm3() -> Rocket:
    # Simplificación equivalente del vehículo con boosters sólidos + core + upper cryogenic.
    stage1 = _stage(
        dry_mass=85_000.0,
        propellant_mass=560_000.0,
        thrust_sl=10_500_000.0,
        thrust_vac=11_400_000.0,
        isp_sl=285.0,
        isp_vac=315.0,
        diameter=4.0,
    )
    stage2 = _stage(
        dry_mass=9_000.0,
        propellant_mass=28_000.0,
        thrust_sl=0.0,
        thrust_vac=200_000.0,
        isp_sl=0.0,
        isp_vac=442.0,
        diameter=4.0,
    )
    return _rocket([stage1, stage2], payload_mass=10_000.0, cd=0.31)


# ============================================================
# HISTORICAL
# ============================================================

def saturn_v() -> Rocket:
    # Modelo 2-stage equivalente para tu simulador.
    stage1 = _stage(
        dry_mass=170_000.0,
        propellant_mass=2_300_000.0,
        thrust_sl=34_000_000.0,
        thrust_vac=36_000_000.0,
        isp_sl=265.0,
        isp_vac=304.0,
        diameter=10.1,
    )
    stage2 = _stage(
        dry_mass=50_000.0,
        propellant_mass=600_000.0,
        thrust_sl=0.0,
        thrust_vac=5_100_000.0,
        isp_sl=0.0,
        isp_vac=421.0,
        diameter=10.1,
    )
    return _rocket([stage1, stage2], payload_mass=118_000.0, cd=0.27)


# ============================================================
# REGISTRY
# ============================================================

_PRESETS: list[RocketPreset] = [
    RocketPreset(
        key="custom",
        label="Custom",
        category="Custom",
        factory=custom_rocket,
        editable=True,
        description="Editable base rocket for user-defined configurations.",
    ),
    RocketPreset(
        key="light",
        label="Light launcher",
        category="Generic",
        factory=light_launcher,
        editable=True,
        description="Generic light two-stage launcher for testing and tuning.",
    ),
    RocketPreset(
        key="medium_generic",
        label="Medium generic",
        category="Generic",
        factory=medium_generic,
        editable=True,
        description="Generic medium two-stage launcher with realistic proportions.",
    ),
    RocketPreset(
        key="heavy_generic",
        label="Heavy generic",
        category="Generic",
        factory=heavy_generic,
        editable=True,
        description="Generic heavy launcher for high-energy mission testing.",
    ),
    RocketPreset(
        key="miura1",
        label="MIURA 1",
        category="PLD Space",
        factory=miura1_suborbital,
        editable=False,
        description="Approximate single-stage suborbital launcher.",
    ),
    RocketPreset(
        key="miura5",
        label="MIURA 5",
        category="PLD Space",
        factory=miura5_orbital,
        editable=False,
        description="Approximate light orbital launcher.",
    ),
    RocketPreset(
        key="electron",
        label="Electron",
        category="Rocket Lab",
        factory=electron,
        editable=False,
        description="Approximate Rocket Lab Electron.",
    ),
    RocketPreset(
        key="neutron",
        label="Neutron",
        category="Rocket Lab",
        factory=neutron,
        editable=False,
        description="Approximate reusable medium launcher.",
    ),
    RocketPreset(
        key="falcon9",
        label="Falcon 9 Block 5",
        category="SpaceX",
        factory=falcon9_block5,
        editable=False,
        description="Approximate Falcon 9 Block 5 in expendable-style performance envelope.",
    ),
    RocketPreset(
        key="falcon_heavy",
        label="Falcon Heavy",
        category="SpaceX",
        factory=falcon_heavy,
        editable=False,
        description="Approximate Falcon Heavy with side boosters folded into stage 1.",
    ),
    RocketPreset(
        key="starship",
        label="Starship + Super Heavy",
        category="SpaceX",
        factory=starship_block2_like,
        editable=False,
        description="Approximate fully reusable super-heavy launcher.",
    ),
    RocketPreset(
        key="vega_c",
        label="Vega-C",
        category="Europe",
        factory=vega_c,
        editable=False,
        description="Equivalent two-stage simplification of Vega-C.",
    ),
    RocketPreset(
        key="ariane62",
        label="Ariane 62",
        category="Europe",
        factory=ariane62,
        editable=False,
        description="Approximate Ariane 62 with boosters folded into stage 1.",
    ),
    RocketPreset(
        key="ariane64",
        label="Ariane 64",
        category="Europe",
        factory=ariane64,
        editable=False,
        description="Approximate Ariane 64 with boosters folded into stage 1.",
    ),
    RocketPreset(
        key="atlas_v_401",
        label="Atlas V 401",
        category="ULA",
        factory=atlas_v_401,
        editable=False,
        description="Approximate Atlas V 401 with single-engine Centaur.",
    ),
    RocketPreset(
        key="vulcan_centaur",
        label="Vulcan Centaur",
        category="ULA",
        factory=vulcan_centaur,
        editable=False,
        description="Approximate Vulcan Centaur with dual-engine Centaur V style upper stage.",
    ),
    RocketPreset(
        key="new_glenn",
        label="New Glenn",
        category="Blue Origin",
        factory=new_glenn,
        editable=False,
        description="Approximate reusable heavy launcher.",
    ),
    RocketPreset(
        key="pslv_xl",
        label="PSLV-XL",
        category="ISRO",
        factory=pslv_xl,
        editable=False,
        description="Equivalent two-stage simplification of PSLV-XL.",
    ),
    RocketPreset(
        key="lvm3",
        label="LVM3",
        category="ISRO",
        factory=lvm3,
        editable=False,
        description="Equivalent two-stage simplification of LVM3.",
    ),
    RocketPreset(
        key="saturn_v",
        label="Saturn V",
        category="Historical",
        factory=saturn_v,
        editable=False,
        description="Historical super-heavy launcher, simplified to 2 equivalent stages.",
    ),
]

ROCKET_PRESETS = {
    preset.key: {
        "label": preset.label,
        "category": preset.category,
        "factory": preset.factory,
        "editable": preset.editable,
        "description": preset.description,
    }
    for preset in _PRESETS
}


# ============================================================
# PUBLIC API
# ============================================================

def list_rocket_presets() -> list[str]:
    return list(ROCKET_PRESETS.keys())


def get_rocket_by_name(name: str) -> Rocket:
    key = name.lower()
    if key not in ROCKET_PRESETS:
        raise ValueError(
            f"Unknown rocket '{name}'. Available presets: {list(ROCKET_PRESETS.keys())}"
        )
    return ROCKET_PRESETS[key]["factory"]()


def get_rocket_preset_info(name: str) -> dict:
    key = name.lower()
    if key not in ROCKET_PRESETS:
        raise ValueError(
            f"Unknown rocket '{name}'. Available presets: {list(ROCKET_PRESETS.keys())}"
        )
    return ROCKET_PRESETS[key]


def get_rocket_presets_grouped() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for key, info in ROCKET_PRESETS.items():
        category = info["category"]
        grouped.setdefault(category, []).append(
            {
                "key": key,
                "label": info["label"],
                "editable": info["editable"],
                "description": info["description"],
            }
        )
    return grouped