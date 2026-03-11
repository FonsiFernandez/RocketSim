# src/rocketsim/models/stage.py
import math
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from rocketsim.models.planet import Planet

G0 = 9.80665  # gravedad estándar [m/s^2]


class Stage:
    """
    Etapa de un cohete con rendimiento distinto en atmósfera y vacío.
    Incluye thrust e Isp a nivel del mar (SL) y en vacío (VAC).
    """

    def __init__(
        self,
        dry_mass: float,
        propellant_mass: float,
        thrust_sl: float,
        thrust_vac: float,
        isp_sl: float,
        isp_vac: float,
        diameter: float,
    ):
        """
        :param dry_mass: masa en seco de la etapa (sin propelente) [kg]
        :param propellant_mass: masa de propelente [kg]
        :param thrust_sl: empuje a nivel del mar [N]
        :param thrust_vac: empuje en vacío [N]
        :param isp_sl: Isp a nivel del mar [s]
        :param isp_vac: Isp en vacío [s]
        :param diameter: diámetro exterior (para área frontal) [m]
        """
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.thrust_sl = thrust_sl
        self.thrust_vac = thrust_vac
        self.isp_sl = isp_sl
        self.isp_vac = isp_vac
        self.diameter = diameter

    @property
    def area(self) -> float:
        """
        Área de referencia para el arrastre (círculo frontal).
        """
        radius = self.diameter / 2.0
        return math.pi * radius**2

    def performance_at_altitude(self, planet: "Planet", altitude: float) -> Tuple[float, float]:
        """
        Devuelve (thrust, Isp) interpolando entre nivel del mar y vacío
        según densidad relativa del aire.

        f_atm = 1 → SL ; f_atm = 0 → vacío
        Si la etapa sólo tiene valores de vacío (upper stage), usa directamente VAC.
        """
        rho = planet.density(altitude)
        rho0 = planet.rho0

        if rho0 <= 0:
            f_atm = 0.0
        else:
            f_atm = max(0.0, min(1.0, rho / rho0))

        # Caso "upper stage puro": sólo datos de vacío
        if self.thrust_sl == 0 and self.isp_sl == 0:
            thrust = self.thrust_vac
            isp = self.isp_vac
        else:
            thrust = f_atm * self.thrust_sl + (1.0 - f_atm) * self.thrust_vac
            isp = f_atm * self.isp_sl + (1.0 - f_atm) * self.isp_vac

        # Evitar valores negativos o cero
        if thrust < 0:
            thrust = 0.0
        if isp < 0:
            isp = 0.0

        return thrust, isp

    def mass_flow_rate(self, planet: "Planet", altitude: float) -> float:
        """
        Caudal de masa de propelente (ṁ) en kg/s, en función del thrust/Isp
        efectivo a esa altitud.
        """
        thrust, isp = self.performance_at_altitude(planet, altitude)
        if thrust <= 0 or isp <= 0:
            return 0.0
        return thrust / (isp * G0)

    @property
    def burn_time_vac(self) -> float:
        """
        Tiempo de quemado aproximado usando rendimiento en vacío (para info).
        """
        if self.thrust_vac <= 0 or self.isp_vac <= 0:
            return 0.0
        mdot_vac = self.thrust_vac / (self.isp_vac * G0)
        return self.propellant_mass / mdot_vac
