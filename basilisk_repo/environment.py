"""
Gym environment for satellite rendezvous task.
Wraps bsk_rl.GeneralSatelliteTasking with configuration-based initialization.
Compatible with AgileRL's flat argument style and YAML configuration.
"""
import yaml
from typing import Any, Optional, Tuple, Dict
import numpy as np
import gymnasium 
from gymnasium import spaces 

import json
from typing import Dict, Any, Optional, Tuple

# Import gymnasium
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# import bsk_rl libraries
from bsk_rl import GeneralSatelliteTasking
from bsk_rl import scene
from bsk_rl.data import ResourceReward

# import satellite types
from satellites import ChiefSat, DeputySat

# import distance and velocity rewards
from rewards import DistanceReward, VelocityReward

# import orbit generation utilities
from orbit_generator import create_orbit_generator


class RendezvousEnv(gym.Env):
    """
    Gymnasium environment for satellite rendezvous task between 2 satellites.
    
    This environment wraps bsk_rl.GeneralSatelliteTasking and provides a 
    configuration-based interface for setting up the simulation.
    Compatible with AgileRL's flat argument passing.
    """
    
    metadata = {"render_modes": [None]}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        flatten_spaces: bool = True,
        # Simulation parameters
        dyn_rate: float = 0.1,
        fsw_rate: float = 1.0,
        max_step_duration: float = 1e9,
        time_limit: float = 1000.0,
        terminate_on_time_limit: bool = False,
        failure_penalty: float = -1.0,
        log_level: str = "WARNING",
        # Reward parameters
        distance_reward_weight: float = -1.0,
        velocity_reward_weight: float = -1.0,
        fuel_penalty_weight: float = -0.1,
        # Other parameters
        use_relative_setup: bool = False,
        **kwargs
    ):
        """
        Initialize satellite and environment parameters.
        
        Args:
            config_path: Path to JSON configuration file. If provided, loads full config.
            flatten_spaces: If True, flatten Tuple action/observation spaces into Box spaces.
            dyn_rate: Dynamics simulation rate [s]
            fsw_rate: FSW update rate [s]
            max_step_duration: Maximum step duration [s]
            time_limit: Episode time limit [s]
            terminate_on_time_limit: Whether to terminate (vs truncate) at time limit
            failure_penalty: Penalty for satellite failure
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            distance_reward_weight: Weight for distance reward
            velocity_reward_weight: Weight for velocity reward
            fuel_penalty_weight: Weight for fuel penalty
            use_relative_setup: Whether to use relative orbital positioning
            **kwargs: Additional arguments (absorbed for compatibility)
        """
        super().__init__()
        
        self.flatten_spaces = flatten_spaces
        
        # If config_path provided, load config and use it
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            # Parse from config
            self._parse_from_config()
        else:
            # Use constructor arguments
            self.dyn_rate = dyn_rate
            self.fsw_rate = fsw_rate
            self.max_step_duration = max_step_duration
            self.time_limit = time_limit
            self.terminate_on_time_limit = terminate_on_time_limit
            self.failure_penalty = failure_penalty
            self.log_level = log_level
            self.distance_reward_weight = distance_reward_weight
            self.velocity_reward_weight = velocity_reward_weight
            self.fuel_penalty_weight = fuel_penalty_weight
            self.use_relative_setup = use_relative_setup
            
            # Set defaults for config-based parameters
            self.distance_target = [0.0, 0.0, 0.0]
            self.velocity_target = [0.0, 0.0, 0.0]
            self.vizard_dir = None
            self.vizard_settings = {}
            self.variable_interval = True
            self.collision_penalty = -100.0
            self.success_reward = 100.0
            self.time_penalty = -0.01
            self.world_type_name = "BasicWorldModel"
            self.world_args = {}
            self.generate_obs_retasking_only = False
            self.dtype = None
            
            # Build minimal config for satellite creation
            self.config = self._build_config_from_args()
        
        # Initialize orbit generator
        self.orbit_generator = create_orbit_generator(self.config)
        
        # Create satellite argument randomizer if using relative setup
        if self.use_relative_setup:
            self.sat_arg_randomizer = self.orbit_generator.create_sat_arg_randomizer()
        else:
            self.sat_arg_randomizer = None
        
        # Create the underlying bsk_rl environment
        self._create_bsk_env()
        
        # For flattened spaces - initialize before _initialize_spaces
        self._action_split_indices = None
        self._original_action_spaces = None
        
        # Initialize spaces eagerly by doing a temporary reset
        self._initialize_spaces()
    
    def _parse_from_config(self):
        """Parse parameters from loaded config dictionary."""
        sim_config = self.config.get("simulation", {})
        self.dyn_rate = sim_config.get("dyn_rate", 0.1)
        self.fsw_rate = sim_config.get("fsw_rate", 1.0)
        self.max_step_duration = sim_config.get("max_step_duration", 1e9)
        self.time_limit = sim_config.get("time_limit", 1000.0)
        self.log_level = sim_config.get("log_level", "WARNING")
        self.terminate_on_time_limit = sim_config.get("terminate_on_time_limit", False)
        self.failure_penalty = sim_config.get("failure_penalty", -1.0)
        self.vizard_dir = sim_config.get("vizard_dir", None)
        self.vizard_settings = sim_config.get("vizard_settings", {})
        self.variable_interval = sim_config.get("variable_interval", True)
        
        reward_config = self.config.get("rewards", {})
        self.distance_reward_weight = reward_config.get("distance_reward_weight", -1.0)
        self.distance_target = reward_config.get("distance_target", [0.0, 0.0, 0.0])
        self.velocity_reward_weight = reward_config.get("velocity_reward_weight", -1.0)
        self.velocity_target = reward_config.get("velocity_target", [0.0, 0.0, 0.0])
        self.fuel_penalty_weight = reward_config.get("fuel_penalty_weight", -0.1)
        self.collision_penalty = reward_config.get("collision_penalty", -100.0)
        self.success_reward = reward_config.get("success_reward", 100.0)
        self.time_penalty = reward_config.get("time_penalty", -0.01)
        
        self.use_relative_setup = self.config.get("relative_positioning", {}).get(
            "use_relative_setup", False
        )
        
        # Parse environment settings
        env_config = self.config.get("environment", {})
        self.world_type_name = env_config.get("world_type", "BasicWorldModel")
        self.world_args = env_config.get("world_args", {})
        self.generate_obs_retasking_only = env_config.get("generate_obs_retasking_only", False)
        self.dtype = env_config.get("dtype", None)
    
    def _build_config_from_args(self) -> Dict[str, Any]:
        """Build config dictionary from constructor arguments."""
        return {
            "simulation": {
                "dyn_rate": self.dyn_rate,
                "fsw_rate": self.fsw_rate,
                "max_step_duration": self.max_step_duration,
                "time_limit": self.time_limit,
                "log_level": self.log_level,
                "terminate_on_time_limit": self.terminate_on_time_limit,
                "failure_penalty": self.failure_penalty,
                "variable_interval": self.variable_interval,
                "vizard_dir": None,
                "vizard_settings": {}
            },
            "rewards": {
                "distance_reward_weight": self.distance_reward_weight,
                "distance_target": self.distance_target,
                "velocity_reward_weight": self.velocity_reward_weight,
                "velocity_target": self.velocity_target,
                "fuel_penalty_weight": self.fuel_penalty_weight,
                "collision_penalty": self.collision_penalty,
                "success_reward": self.success_reward,
                "time_penalty": self.time_penalty,
            },
            "satellites": {
                "chief": {
                    "name": "ChiefSat",
                    "initial_conditions": {
                        "utc_init": 0.0,
                        "oe": [7000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "mu": 398600441800000.0
                    },
                    "physical_properties": {
                        "mass": 300.0,
                        "dragCoeff": 2.2,
                        "width": 1.38,
                        "depth": 1.04,
                        "height": 1.58
                    },
                    "power_system": {
                        "storedCharge_Init": 50000.0,
                        "batteryStorageCapacity": 100000.0,
                        "panelEfficiency": 0.3,
                        "panelArea": 1.0,
                        "basePowerDraw": 0.0
                    },
                    "attitude_control": {
                        "sigma_init": [0.0, 0.0, 0.0],
                        "omega_init": [0.0, 0.0, 0.0],
                        "wheelSpeeds": [0.0, 0.0, 0.0],
                        "maxWheelSpeed": 0.1,
                        "u_max": 0.0
                    }
                },
                "deputy": {
                    "name": "DeputySat",
                    "initial_conditions": {
                        "utc_init": 0.0,
                        "oe": [7001000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "mu": 398600441800000.0
                    },
                    "physical_properties": {
                        "mass": 250.0,
                        "dragCoeff": 2.0,
                        "width": 1.2,
                        "depth": 0.9,
                        "height": 1.3
                    },
                    "power_system": {
                        "storedCharge_Init": 40000.0,
                        "batteryStorageCapacity": 80000.0,
                        "panelEfficiency": 0.3,
                        "panelArea": 0.8,
                        "basePowerDraw": 0.0
                    },
                    "attitude_control": {
                        "sigma_init": [0.0, 0.0, 0.0],
                        "omega_init": [0.0, 0.0, 0.0],
                        "wheelSpeeds": [0.0, 0.0, 0.0],
                        "maxWheelSpeed": 0.1,
                        "u_max": 0.0
                    },
                    "propulsion": {
                        "dv_available_init": 100.0
                    }
                }
            },
            "relative_positioning": {
                "use_relative_setup": self.use_relative_setup
            },
            "environment": {
                "world_type": "BasicWorldModel",
                "world_args": {},
                "communicator": None,
                "sat_arg_randomizer": None,
                "generate_obs_retasking_only": False,
                "dtype": None
            }
        }
    
    def _create_satellites(self):
        """Create satellite instances with configuration."""
        satellites = []
        
        # Create Chief satellite
        chief_config = self.config["satellites"]["chief"]
        chief_sat_args = self._merge_satellite_config(chief_config)
        chief_sat = ChiefSat(
            name=chief_config.get("name", "chief"),
            sat_args=chief_sat_args
        )
        satellites.append(chief_sat)
        
        # Create Deputy satellite
        deputy_config = self.config["satellites"]["deputy"]
        deputy_sat_args = self._merge_satellite_config(deputy_config)
        deputy_sat = DeputySat(
            name=deputy_config.get("name", "deputy"),
            sat_args=deputy_sat_args
        )
        satellites.append(deputy_sat)
        
        return satellites
    
    def _merge_satellite_config(self, satellite_config):
        """Create valid satellite arguments from configuration."""
        # Define valid satellite arguments with defaults
        sat_args = {
            "mass": satellite_config.get("mass", 100.0),  # kg
            "rN": satellite_config.get("r_N", [7000.0, 0.0, 0.0]),  # km
            "vN": satellite_config.get("v_N", [0.0, 7.5, 0.0]),  # km/s
            "sigma_BN": satellite_config.get("sigma_BN", [0.0, 0.0, 0.0]),  # attitude
            "omega_BN_B": satellite_config.get("omega_BN_B", [0.0, 0.0, 0.0])  # angular velocity
        }
        
        # Add propulsion configuration if specified
        if "propulsion" in satellite_config:
            sat_args["dv_available_init"] = satellite_config["propulsion"].get("dv_available_init", 100.0)
        
        return sat_args
    
    def _create_reward_system(self):
        """Create reward system based on configuration."""
        rewards = [
            DistanceReward(
                weight=self.distance_reward_weight,
                target=self.distance_target,
                position_fn=lambda sat: sat.dynamics.r_BN_N
            ), 
            VelocityReward(
                weight=self.velocity_reward_weight,
                target=self.velocity_target,
                velocity_fn=lambda sat: sat.dynamics.v_BN_N
            ),
            ResourceReward(
                reward_weight=self.fuel_penalty_weight,
                resource_fn=lambda sat: sat.fsw.dv_available
            )
        ]
        return rewards
    
    def _flatten_action_space(self, tuple_space: spaces.Tuple) -> spaces.Box:
        """Flatten a Tuple action space into a Box."""
        self._original_action_spaces = tuple_space.spaces
        action_dims = []
        lows = []
        highs = []
        
        for space in self._original_action_spaces:
            if isinstance(space, spaces.Box):
                dim = np.prod(space.shape)
                action_dims.append(dim)
                lows.append(space.low.flatten())
                highs.append(space.high.flatten())
            elif isinstance(space, spaces.Discrete):
                dim = 1
                action_dims.append(dim)
                lows.append(np.array([0.0]))
                highs.append(np.array([float(space.n - 1)]))
            elif isinstance(space, spaces.MultiDiscrete):
                dim = len(space.nvec)
                action_dims.append(dim)
                lows.append(np.zeros(dim))
                highs.append(space.nvec.astype(float) - 1)
            else:
                raise NotImplementedError(f"Cannot flatten space type {type(space)}")
        
        self._action_split_indices = np.cumsum([0] + action_dims)[:-1]
        
        low = np.concatenate(lows)
        high = np.concatenate(highs)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _unflatten_action(self, flat_action: np.ndarray) -> Tuple:
        """Convert flattened action back to tuple."""
        if self._action_split_indices is None or self._original_action_spaces is None:
            return flat_action
        
        action_components = np.split(flat_action, self._action_split_indices[1:])
        actions = []
        
        for component, original_space in zip(action_components, self._original_action_spaces):
            if isinstance(original_space, spaces.Box):
                reshaped = component.reshape(original_space.shape)
                actions.append(reshaped)
            elif isinstance(original_space, spaces.Discrete):
                action_val = int(np.clip(np.round(component[0]), 0, original_space.n - 1))
                actions.append(action_val)
            elif isinstance(original_space, spaces.MultiDiscrete):
                action_vals = np.clip(np.round(component), 0, original_space.nvec - 1).astype(int)
                actions.append(action_vals)
            else:
                raise NotImplementedError(f"Cannot unflatten space type {type(original_space)}")
        
        return tuple(actions)
    
    def _flatten_observation_space(self, tuple_space: spaces.Tuple) -> spaces.Box:
        """Flatten a Tuple observation space into a Box."""
        lows = []
        highs = []
        
        for space in tuple_space.spaces:
            if isinstance(space, spaces.Box):
                lows.append(space.low.flatten())
                highs.append(space.high.flatten())
            else:
                raise NotImplementedError(f"Cannot flatten space type {type(space)}")
        
        low = np.concatenate(lows)
        high = np.concatenate(highs)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _flatten_observation(self, tuple_obs: Tuple) -> np.ndarray:
        """Convert tuple of observations to flattened array."""
        flattened = []
        for obs in tuple_obs:
            flattened.append(np.array(obs).flatten())
        return np.concatenate(flattened).astype(np.float32)
    
    def _create_bsk_env(self):
        """Create the underlying bsk_rl GeneralSatelliteTasking environment."""
        satellites = self._create_satellites()
        rewarder = self._create_reward_system()
        
        env_kwargs = {
            "satellites": satellites,
            "rewarder": rewarder,
            "sim_rate": self.dyn_rate,
            "max_step_duration": self.max_step_duration,
            "time_limit": self.time_limit,
            "terminate_on_time_limit": self.terminate_on_time_limit,
            "failure_penalty": self.failure_penalty,
            "log_level": self.log_level,
        }
        
        # Add optional parameters from config
        if hasattr(self, 'world_args') and self.world_args:
            env_kwargs["world_args"] = self.world_args
        
        if hasattr(self, 'generate_obs_retasking_only'):
            env_kwargs["generate_obs_retasking_only"] = self.generate_obs_retasking_only
        
        if hasattr(self, 'dtype') and self.dtype is not None:
            # Convert string dtype to numpy dtype if needed
            if isinstance(self.dtype, str):
                import numpy as np
                env_kwargs["dtype"] = np.dtype(self.dtype)
            else:
                env_kwargs["dtype"] = self.dtype
        
        if self.sat_arg_randomizer is not None:
            env_kwargs["sat_arg_randomizer"] = self.sat_arg_randomizer
        
        self._bsk_env = GeneralSatelliteTasking(**env_kwargs)
    
    def _initialize_spaces(self):
        """Initialize action and observation spaces by doing a temporary reset."""
        self._bsk_env.reset(seed=0)
        
        original_action_space = self._bsk_env.action_space
        original_obs_space = self._bsk_env.observation_space
        
        if self.flatten_spaces:
            self.action_space = self._flatten_action_space(original_action_space)
            self.observation_space = self._flatten_observation_space(original_obs_space)
        else:
            self.action_space = original_action_space
            self.observation_space = original_obs_space
            self._original_action_spaces = None
            self._action_split_indices = None
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        obs, info = self._bsk_env.reset(seed=seed, options=options)
        
        if self.flatten_spaces:
            obs = self._flatten_observation(obs)
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        if self.flatten_spaces:
            action = self._unflatten_action(action)
        
        obs, reward, terminated, truncated, info = self._bsk_env.step(action)
        
        if self.flatten_spaces:
            obs = self._flatten_observation(obs)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self._bsk_env.render()
    
    def close(self):
        """Clean up environment resources."""
        if hasattr(self, '_bsk_env') and self._bsk_env is not None:
            try:
                self._bsk_env.close()
            except (AttributeError, Exception):
                pass
    
    @classmethod
    def from_config_file(cls, config_path: str, flatten_spaces: bool = True) -> "RendezvousEnv":
        """
        Create RendezvousEnv from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            flatten_spaces: If True, flatten action/observation spaces
            
        Returns:
            RendezvousEnv instance
        """
        return cls(config_path=config_path, flatten_spaces=flatten_spaces)
    
    @property
    def satellites(self):
        """Access to underlying satellites."""
        return self._bsk_env.satellites
    
    @property
    def simulator(self):
        """Access to underlying simulator."""
        return self._bsk_env.simulator