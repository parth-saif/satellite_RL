"""
Orbit generation utilities for creating randomized starting conditions for rendezvous training.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from Basilisk.utilities import orbitalMotion
from bsk_rl.utils.orbital import elem2rv, rv2elem, hill2cd


class OrbitGenerator:
    """
    Utility class for generating randomized orbital configurations for training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize orbit generator with configuration.
        
        Args:
            config: Configuration dictionary containing relative positioning parameters
        """
        self.config = config
        self.relative_config = config.get("relative_positioning", {})
        self.mu = config["satellites"]["chief"]["initial_conditions"]["mu"]
        
    def generate_chief_orbit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a randomized chief satellite orbit.
        
        Returns:
            Tuple of (position, velocity) in inertial frame
        """
        if not self.relative_config.get("chief_orbit_randomization", {}).get("enabled", False):
            # Use default orbit from config - convert from orbital elements
            chief_config = self.config["satellites"]["chief"]["initial_conditions"]
            oe_list = chief_config["oe"]
            
            # Create orbital elements object
            oe = orbitalMotion.ClassicElements()
            oe.a = oe_list[0]
            oe.e = oe_list[1]
            oe.i = oe_list[2]
            oe.Omega = oe_list[3]
            oe.omega = oe_list[4]
            oe.f = oe_list[5]
            
            rN, vN = elem2rv(self.mu, oe)
            return rN, vN
        
        # Generate randomized orbit
        orbit_params = self.relative_config["chief_orbit_randomization"]
        
        # Generate orbital elements
        a = np.random.uniform(*orbit_params["altitude_range"])  # Semi-major axis
        e = np.random.uniform(*orbit_params["eccentricity_range"])  # Eccentricity
        i = np.random.uniform(*orbit_params["inclination_range"])  # Inclination
        Omega = np.random.uniform(*orbit_params["raan_range"])  # RAAN
        omega = np.random.uniform(*orbit_params["arg_periapsis_range"])  # Argument of periapsis
        f = np.random.uniform(*orbit_params["true_anomaly_range"])  # True anomaly
        
        # Convert to position and velocity
        oe = orbitalMotion.ClassicElements()
        oe.a = a
        oe.e = e
        oe.i = i
        oe.Omega = Omega
        oe.omega = omega
        oe.f = f
        
        rN, vN = elem2rv(self.mu, oe)
        return rN, vN
    
    def generate_relative_state(self, scenario: Optional[str] = None) -> np.ndarray:
        """
        Generate a randomized relative state (position and velocity) in Hill frame.
        
        Args:
            scenario: Training scenario name ('close_approach', 'medium_range', 'far_range')
                     If None, randomly selects based on weights
        
        Returns:
            6-element array [rho_H, rho_dot_H] in Hill frame
        """
        if scenario is None:
            scenario = self._select_training_scenario()
        
        scenarios = self.relative_config["deputy_relative_states"]["training_scenarios"]
        if scenario not in scenarios:
            scenario = "medium_range"  # Default fallback
        
        scenario_config = scenarios[scenario]
        
        # Generate position in Hill frame
        pos_ranges = scenario_config["position_ranges"]
        rho_H = np.array([
            np.random.uniform(*pos_ranges["radial_range"]),
            np.random.uniform(*pos_ranges["along_track_range"]),
            np.random.uniform(*pos_ranges["cross_track_range"])
        ])
        
        # Generate velocity in Hill frame
        vel_ranges = scenario_config["velocity_ranges"]
        rho_dot_H = np.array([
            np.random.uniform(*vel_ranges["radial_velocity_range"]),
            np.random.uniform(*vel_ranges["along_track_velocity_range"]),
            np.random.uniform(*vel_ranges["cross_track_velocity_range"])
        ])
        
        # Apply constraints
        relative_state = np.concatenate([rho_H, rho_dot_H])
        relative_state = self._apply_constraints(relative_state)
        
        return relative_state
    
    def _select_training_scenario(self) -> str:
        """
        Select training scenario based on weights.
        
        Returns:
            Selected scenario name
        """
        scenarios = self.relative_config["deputy_relative_states"]["training_scenarios"]
        scenario_names = list(scenarios.keys())
        weights = [scenarios[name]["weight"] for name in scenario_names]
        
        return np.random.choice(scenario_names, p=weights)
    
    def _apply_constraints(self, relative_state: np.ndarray) -> np.ndarray:
        """
        Apply safety and training constraints to relative state.
        
        Args:
            relative_state: 6-element relative state vector
            
        Returns:
            Constrained relative state vector
        """
        constraints = self.relative_config.get("constraints", {})
        
        rho_H = relative_state[:3]
        rho_dot_H = relative_state[3:6]
        
        # Distance constraints
        distance = np.linalg.norm(rho_H)
        max_dist = constraints.get("max_relative_distance", 15000.0)
        min_dist = constraints.get("min_relative_distance", 100.0)
        
        if distance > max_dist:
            rho_H = rho_H / distance * max_dist
        elif distance < min_dist:
            rho_H = rho_H / distance * min_dist
        
        # Velocity constraints
        velocity = np.linalg.norm(rho_dot_H)
        max_vel = constraints.get("max_relative_velocity", 15.0)
        
        if velocity > max_vel:
            rho_dot_H = rho_dot_H / velocity * max_vel
        
        return np.concatenate([rho_H, rho_dot_H])
    
    def generate_deputy_orbit(self, chief_rN: np.ndarray, chief_vN: np.ndarray, 
                            relative_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate deputy satellite orbit from chief orbit and relative state.
        
        Args:
            chief_rN: Chief satellite position in inertial frame
            chief_vN: Chief satellite velocity in inertial frame
            relative_state: Relative state in Hill frame (if None, generates random)
            
        Returns:
            Tuple of (deputy_position, deputy_velocity) in inertial frame
        """
        if relative_state is None:
            relative_state = self.generate_relative_state()
        
        rho_H = relative_state[:3]
        rho_dot_H = relative_state[3:6]
        
        # Convert from Hill frame to inertial frame
        deputy_rN, deputy_vN = hill2cd(chief_rN, chief_vN, rho_H, rho_dot_H)
        
        return deputy_rN, deputy_vN
    
    def create_sat_arg_randomizer(self) -> Callable:
        """
        Create a satellite argument randomizer function for bsk_rl.
        
        Returns:
            Function that can be used as sat_arg_randomizer in GeneralSatelliteTasking
        """
        def randomize_satellite_args(satellites):
            """
            Randomize satellite arguments for each episode.
            
            Args:
                satellites: List of satellite objects
                
            Returns:
                Dictionary mapping satellites to their randomized arguments
            """
            args = {}
            
            # Find chief and deputy satellites
            chief_sat = None
            deputy_sat = None
            
            for sat in satellites:
                if sat.name == "ChiefSat":
                    chief_sat = sat
                elif sat.name == "DeputySat":
                    deputy_sat = sat
            
            if chief_sat is None or deputy_sat is None:
                raise ValueError("Both ChiefSat and DeputySat must be present")
            
            # Generate new orbits
            chief_rN, chief_vN = self.generate_chief_orbit()
            relative_state = self.generate_relative_state()
            deputy_rN, deputy_vN = self.generate_deputy_orbit(chief_rN, chief_vN, relative_state)
            
            # Convert to orbital elements
            chief_oe = rv2elem(self.mu, chief_rN, chief_vN)
            deputy_oe = rv2elem(self.mu, deputy_rN, deputy_vN)
            
            # Set up arguments - only provide oe (orbital elements) to avoid conflicts
            args[chief_sat] = {
                "oe": chief_oe
            }
            
            args[deputy_sat] = {
                "oe": deputy_oe
            }
            
            return args
        
        return randomize_satellite_args


def create_orbit_generator(config: Dict) -> OrbitGenerator:
    """
    Factory function to create an OrbitGenerator instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OrbitGenerator instance
    """
    return OrbitGenerator(config)
