"""
Example usage of the RendezvousEnv with relative positioning configuration.
"""
import numpy as np
from environment import RendezvousEnv


def main():
    """
    Example of how to use the RendezvousEnv with relative positioning.
    """
    # Load configuration from file
    config_path = "config/simulation_config.yaml"
    env_wrapper = RendezvousEnv.from_config_file(config_path)
    
    # Create the gym environment
    env = env_wrapper
    print("Environment created successfully!")
    print(f"Using relative positioning: {env_wrapper.use_relative_setup}")
    
    # Reset environment to get initial observation
    observation, info = env.reset(seed=42)
    
    
    # print the observation space and action space
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Take a few random actions to test
    for step in range(5):
        # Actions for both satellites: ChiefSat gets no action (Drift), DeputySat gets thrust
        chief_action = 0  # Drift action is just a zero
        deputy_action = env.action_space[1].sample()  # Sample from deputy's action space
        action = (chief_action, deputy_action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Reward: {reward}")
        print(f"  Observation: {observation}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("Environment closed.")


def test_orbit_generation():
    """
    Test the orbit generation functionality.
    """
    print("\n=== Testing Orbit Generation ===")
    
    # Load configuration
    config_path = "config/simulation_config.json"
    env_wrapper = RendezvousEnv.from_config_file(config_path)
    
    # Test orbit generator
    orbit_gen = env_wrapper.orbit_generator
    
    # Generate a few different scenarios
    scenarios = ["close_approach", "medium_range", "far_range"]
    
    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} Scenario ---")
        
        # Generate chief orbit
        chief_rN, chief_vN = orbit_gen.generate_chief_orbit()
        print(f"Chief position: {chief_rN}")
        print(f"Chief velocity: {chief_vN}")
        
        # Generate relative state
        relative_state = orbit_gen.generate_relative_state(scenario)
        print(f"Relative state (Hill frame): {relative_state}")
        
        # Generate deputy orbit
        deputy_rN, deputy_vN = orbit_gen.generate_deputy_orbit(chief_rN, chief_vN, relative_state)
        print(f"Deputy position: {deputy_rN}")
        print(f"Deputy velocity: {deputy_vN}")
        
        # Calculate relative distance
        relative_distance = np.linalg.norm(relative_state[:3])
        print(f"Relative distance: {relative_distance:.2f} m")


if __name__ == "__main__":
    main()
    test_orbit_generation()
