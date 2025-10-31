
from bsk_rl import GeneralSatelliteTasking # -> general satellite tasking environment
from bsk_rl import data, scene
from satellites import ChiefSat, DeputySat


# test environments
if __name__ == "__main__":
    # create a target satellite
    target_sat = ChiefSat(name="ChiefSat")

    # create a chaser satellite
    chaser_sat = DeputySat(name="DeputySat", sat_args=dict(dv_available_init=1))

     # Create lists to store data
    target_data = []
    chaser_data = []
    # create environment
    env = GeneralSatelliteTasking(satellites=[target_sat, chaser_sat], scenario=scene.Scenario(), rewarder = data.NoReward(), time_limit=5700, log_level="INFO", terminate_on_time_limit=True, vizard_dir="viz_output")
    obs = env.reset()
    print("Environment created:", env)
    # initialise orbits for both satellites

    # Simulation loop
    done = False
    i=0
    while not done:
        # Take a step with some action
        obs, reward, done, truncated, info = env.step([0, (2,1,0,0)])

    #     if i==0:
    #         target_kepler = obs[0][9:14]
    #         chaser_kepler = obs[1][9:14]

    #         target_data.append(target_kepler.tolist())
    #         chaser_data.append(chaser_kepler.tolist())

    #         i+=1
        
        time = obs[0][-1]*5700
        chaser_obs = obs[1][0:7]  # First 3 elements are position       
        # append to list
    
        chaser_data.append([time] + chaser_obs.tolist())
    

    # Convert to pandas DataFrames
    import pandas as pd
    chaser_df = pd.DataFrame(chaser_data)
    
    # Save to CSV files
    chaser_df.to_csv('chaser_trajectory.csv', index=False)
    
    print("Trajectory data saved to CSV files")