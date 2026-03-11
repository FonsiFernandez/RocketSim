import numpy as np


class Planet:
    """
    Representa un cuerpo (Tierra, Marte, Luna...) con gravedad y atmósfera simple.
    """
    def __init__(self, name: str, mu: float, radius: float, rho0: float, H: float):
        """
        :param name: nombre del planeta
        :param mu: parámetro gravitatorio GM [m^3/s^2]
        :param radius: radio del planeta [m]
        :param rho0: densidad del aire a nivel del mar [kg/m^3]
        :param H: altura de escala de la atmósfera [m] (modelo exponencial)
        """
        self.name = name
        self.mu = mu
        self.radius = radius
        self.rho0 = rho0
        self.H = H

    def gravity_acc(self, r: float) -> float:
        """
        Aceleración de la gravedad a distancia r del centro del planeta.
        """
        return self.mu / r**2

    def density(self, altitude: float) -> float:
        """
        Modelo de atmósfera exponencial muy simple.
        Si rho0 == 0, se considera sin atmósfera.
        """
        if altitude < 0:
            altitude = 0.0
        if self.rho0 <= 0 or self.H <= 0:
            return 0.0
        return self.rho0 * np.exp(-altitude / self.H)


def earth_example() -> Planet:
    """
    Tierra: parámetros aproximados.
    """
    mu_earth = 3.986e14      # m^3/s^2
    r_earth = 6_371_000.0    # m
    rho0 = 1.225             # kg/m^3
    H = 8400.0               # m
    return Planet("Earth", mu_earth, r_earth, rho0, H)


def mars_example() -> Planet:
    """
    Marte: atmósfera muy fina.
    """
    mu_mars = 4.2828e13      # m^3/s^2
    r_mars = 3_389_500.0     # m
    rho0 = 0.020             # kg/m^3 (aprox. superficie)
    H = 11_000.0             # m
    return Planet("Mars", mu_mars, r_mars, rho0, H)


def moon_example() -> Planet:
    """
    Luna: sin atmósfera.
    """
    mu_moon = 4.9048695e12   # m^3/s^2
    r_moon = 1_737_400.0     # m
    rho0 = 0.0               # sin atmósfera
    H = 1.0                  # irrelevante porque rho0 = 0
    return Planet("Moon", mu_moon, r_moon, rho0, H)


PLANETS_BY_NAME = {
    "earth": earth_example,
    "mars": mars_example,
    "moon": moon_example,
}
