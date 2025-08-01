'''
DOES NOT WORK FOR ELLIPSES. 7th order polynomial segments are NOT good approximations for ellipses, compared to Bezier curves
Runge oscillations are seen.
It does work for manoeuvers between

Drone Trajectory Generation Module
1. Scale trajectories from space-scale to drone-scale.
2. Generate drone trajectories as 7th order polynomial segments from waypoints.
'''
import numpy as np 
packages_folder = '/home/asimov/saif-ESA-BIC/packages/uav_trajectories/scripts' # path to the uav_trajectories package
import sys
sys.path.append(packages_folder) # add the package to the system path
print(sys.path) # print the system path to check if the package is added

from generate_trajectory import generate_trajectory # open source trajectory generation package
from uav_trajectory import Trajectory # import the Trajectory class from the package

class DroneTrajectory(Trajectory):
    """
    DroneTrajectory class to handle drone trajectories.
    Inherits from the Trajectory class to provide additional functionality.
    Adds method to output trajectory as an array.
    """
    def __init__(self, polynomials=None, duration=None):
        super().__init__()
        self.polynomials = polynomials if polynomials is not None else None
        self.duration = duration if duration is not None else None
    def to_array(self):
        """
        Convert the trajectory to a numpy array format.
        :return: Numpy array of the trajectory in the format [[time, x, y, z, yaw], ...]
        """

        data = np.array([
        [p.duration] + list(p.px.p) + list(p.py.p) + list(p.pz.p) + list(p.pyaw.p)
        for p in self.polynomials
        ])
        return data


class TrajectoryGenerator:
    def __init__(self, sat_waypoints=None, num_pieces=5, drone_waypoints=None, dist_sf=1e-6, time_sf=1e-2):
        """
        Initialize the DroneTrajectory with waypoints and number of pieces.
        
        :param waypoints: List of waypoints in the format [[time, x, y, z, yaw], ...]
        :param num_pieces: Number of polynomial pieces to divide the trajectory into
        :param drone_waypoints: Optional list of drone waypoints in the same format as waypoints
        :param drone_traj: Optional pre-generated drone trajectory
        :param dist_sf: Distance scale factor to convert space-scale to drone-scale
        :param time_sf: Time scale factor to convert space-scale to drone-scale
        """
        self.sat_waypoints = np.array(sat_waypoints) if sat_waypoints is not None else None
        self.num_pieces = num_pieces
        
        self.dist_sf = dist_sf  # Distance scale factor
        self.time_sf = time_sf  # Time scale factor
        self.duration = None # init drone trajectory duration
        self.drone_waypoints = self.scale_sat_wp()

        self.trajectory = DroneTrajectory()
        self.traj_array = None  # to store the trajectory as a numpy array
    
    def scale_sat_wp(self):
        """
        Scale the satellite trajectory from space-scale to drone-scale.
        :return: Scaled waypoints in the format [[time, x, y, z, yaw], ...]
        """
        scaled_waypoints = self.sat_waypoints.copy() # create a copy of the satellite waypoints
        # if yaw is not provided, set it to zero
        if scaled_waypoints.shape[1] < 5:
            scaled_waypoints = np.hstack((scaled_waypoints, np.zeros((scaled_waypoints.shape[0], 1))))
        
        scaled_waypoints[:, 1:4] *= self.dist_sf # scale x, y, z coordinates of waypoints by distance scale factor
        scaled_waypoints[:, 0] *= self.time_sf # scale time by time scale factor -> make the trajectory faster
        self.duration = scaled_waypoints[-1,0] # duration of drone trajectory
        self.drone_waypoints = scaled_waypoints
        return scaled_waypoints

    def gen_drone_traj(self):
        """
        Generate a drone trajectory as a 7th order polynomial segments from the provided waypoints.
        Return co-efficents of polynomials.
        """
        if self.sat_waypoints is None:
            raise ValueError("Satellite waypoints are not provided.")
        
        trajectory = generate_trajectory(self.drone_waypoints, self.num_pieces) # this creates a trajectory object with polynomial segments
        self.trajectory = DroneTrajectory(polynomials=trajectory.polynomials, duration=self.duration)
            
        self.traj_array = self.trajectory.to_array()  # convert the trajectory to a numpy array format#
    
        return self.traj_array
    def eval_poly(self):
        """
        Return evaluated polynomial positions in a certain time range.
        """
        time_vec = np.linspace(0, (self.trajectory.duration)-0.1, 500)
    
        # Evaluate the trajectory at each point in the time vector
        # The .eval(t) method returns a state object with a .pos attribute
        eval_points = np.array([self.trajectory.eval(t).pos for t in time_vec])

        return eval_points

if __name__ == "__main__":
    import time
    start = time.time()
    # Example usage
    import pandas as pd
    # Load satellite waypoints from a CSV file 
    sat_waypoints = pd.read_csv('./target_trajectory.csv').values  # Assuming the CSV has columns: time, x, y, z, yaw
    print("Satellite Waypoints:", sat_waypoints)

    # Reduce the number of waypoints to 8
    sat_waypoints = sat_waypoints[::sat_waypoints.shape[0]//8]
    if sat_waypoints.shape[0] > 8:
        sat_waypoints = sat_waypoints[:8]
    print("Satellite Waypoints:", sat_waypoints)
    
    
    drone_gen = TrajectoryGenerator(sat_waypoints=sat_waypoints, num_pieces=4)#, dist_sf=1, time_sf=1)
    drone_waypoints = drone_gen.scale_sat_wp()  # scale the satellite waypoints to drone waypoints
    print(drone_waypoints)
    drone_traj = drone_gen.gen_drone_traj()  # generate the drone trajectory from the scaled waypoints

    print(time.time()-start)

    #print("Drone Trajectory Array:", drone_traj)

    #plot 3d waypoints
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(drone_waypoints[:, 1], drone_waypoints[:, 2], drone_waypoints[:, 3], marker='o', label='Waypoints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # also plot evaluations of poly fit in same plot
    eval_points = drone_gen.eval_poly()
    ax.plot(eval_points[:, 0], eval_points[:, 1], eval_points[:, 2], 
            '-r', lw=2, label='7th Order Polynomial Trajectory',)
    ax.legend()
    plt.show()




