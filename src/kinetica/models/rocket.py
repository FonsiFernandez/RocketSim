from typing import List
from kinetica.models.stage import Stage


class Rocket:
    """
    Cohete compuesto por varias etapas apiladas.
    """
    def __init__(self, stages: List[Stage], cd: float = 0.3, payload_mass: float = 0.0):
        """
        :param stages: lista de etapas (ordenadas de primera a última)
        :param cd: coeficiente de arrastre global del cohete
        :param payload_mass: masa de la carga útil [kg]
        """
        self.stages = stages
        self.cd = cd
        self.payload_mass = payload_mass

    def total_dry_mass(self) -> float:
        """
        Masa en seco total de todas las etapas (sin propelente) + payload.
        """
        return sum(s.dry_mass for s in self.stages) + self.payload_mass

    def total_propellant_mass(self) -> float:
        """
        Masa total de propelente (suma de todas las etapas).
        """
        return sum(s.propellant_mass for s in self.stages)

    def total_initial_mass(self) -> float:
        """
        Masa total de despegue.
        """
        return self.total_dry_mass() + self.total_propellant_mass()
