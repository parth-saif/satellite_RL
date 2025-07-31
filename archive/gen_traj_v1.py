'''
This doesnt really work. 7th order polynomial segments are not ideal for ellipse fitting, compared to Bezier curves. Also, it is very numerically difficult with lots of way points to optimise the polynomial segment.

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
        data = np.empty((len(self.polynomials), 8*4+1))

        data = np.array([
        [p.duration] + list(p.px.p) + list(p.py.p) + list(p.pz.p) + list(p.pyaw.p)
        for p in self.polynomials
        ])
        return data


class TrajectoryGenerator:
    def __init__(self, sat_waypoints=None, num_pieces=5, drone_waypoints=None, drone_traj =None, dist_sf=1e-6, time_sf=1e-1):
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

        # optionally drone waypoints and trajectory if already generated
        self.drone_waypoints = np.array(drone_waypoints) if drone_waypoints is not None else None
        self.trajectory = drone_traj 
        self.traj_array = None  # to store the trajectory as a numpy array

        self.dist_sf = dist_sf  # Distance scale factor
        self.time_sf = time_sf  # Time scale factor
    
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
        self.drone_waypoints = scaled_waypoints
        return scaled_waypoints

    def scale_drone_wp(self): # optionally, scale the drone waypoints from drone-scale to space-scale
        """
        Scale the drone trajectory from drone-scale to space-scale.
        :return: Scaled waypoints in the format [[time, x, y, z, yaw], ...]
        """
        if self.trajectory is None:
            raise ValueError("Drone trajectory is not generated yet.")
        
        scaled_waypoints = self.trajectory.to_array()
        scaled_waypoints[:, 1:4] /= self.dist_sf  # scale x, y, z coordinates of waypoints by distance scale factor
        scaled_waypoints[:, 0] /= self.time_sf  # scale time by time scale factor -> make the trajectory slower
        return scaled_waypoints

    def gen_drone_traj(self):
        """
        Generate a drone trajectory as a 7th order polynomial segments from the provided waypoints.
        """
        if self.sat_waypoints is None:
            raise ValueError("Satellite waypoints are not provided.")
        
        scaled_waypoints = self.scale_sat_wp()
        self.trajectory = generate_trajectory(scaled_waypoints, self.num_pieces) # this creates a trajectory object with polynomial segments

        # convert trajectory object to DroneTrajectory class
        if not isinstance(self.trajectory, DroneTrajectory):
            poly = self.trajectory.polynomials
            duration = self.trajectory.duration
            self.trajectory = DroneTrajectory(polynomials=poly, duration=duration)  # convert to DroneTrajectory class
            
        self.traj_array = self.trajectory.to_array()  # convert the trajectory to a numpy array format
        
        return self.traj_array

if __name__ == "__main__":
    import time
    start = time.time()
    # Example usage
    import pandas as pd
    # Load satellite waypoints from a CSV file 
    sat_waypoints = pd.read_csv('./target_trajectory.csv').values  # Assuming the CSV has columns: time, x, y, z, yaw

    # reduce the waypoints to 5 for testing
    if sat_waypoints.shape[0] > 10:
        sat_waypoints = sat_waypoints[::sat_waypoints.shape[0] //10]

    print("Satellite Waypoints:", sat_waypoints)
    
    drone_gen = TrajectoryGenerator(sat_waypoints=sat_waypoints, num_pieces=2)
    drone_waypoints = drone_gen.scale_sat_wp()  # scale the satellite waypoints to drone waypoints
    print(drone_waypoints)
    drone_traj = drone_gen.gen_drone_traj()  # generate the drone trajectory from the scaled waypoints

    print(time.time()-start)

    print("Drone Trajectory Array:", drone_traj)

    #plot 3d waypoints
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()  
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(drone_waypoints[:, 1], drone_waypoints[:, 2], drone_waypoints[:, 3], marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title('Drone Waypoints in 3D')
    # plt.show()
    
    import matplotlib.pyplot as plt
    def calculate_polynomial(t, coeffs):
        """
        Calculates the value of a polynomial at time t.
        coeffs: list of coefficients [c0, c1, ..., c7]
        """
        value = 0
        for i, c in enumerate(coeffs):
            value += c * (t ** i)
        return value

    all_x = []
    all_y = []
    all_z = [] # To store z-coordinates for 3D plot

    # Keep track of the actual current position to ensure continuity
    current_x = 0.0
    current_y = 0.0
    current_z = 0.0

    # Assume the first segment implicitly starts at (0,0,0) or some initial state if not specified explicitly.
    # For these types of polynomial coefficients, the c0 term IS the starting position for that segment
    # IF the segment time `t` starts from 0.0 for each segment.
    # The overall path continuity comes from how these polynomials are derived,
    # i.e., the end point of one polynomial segment should match the start of the next.

    for i, segment in enumerate(drone_traj):
        duration = segment[0]
        x_coeffs = segment[1:9]   # x^0 to x^7
        y_coeffs = segment[9:17]  # y^0 to y^7
        z_coeffs = segment[17:25] # z^0 to z^7

        # Generate time points for the current segment
        # Using 100 points per segment for a smooth curve
        t_segment = np.linspace(0, duration, 100)

        # Calculate x, y, and z values for each time point in the current segment
        # The c0 terms in the coefficients implicitly handle the starting point of each segment.
        # The continuity of the overall path depends on how these polynomials were generated
        # (i.e., the final position of segment N is the initial position of segment N+1).
        x_values = [calculate_polynomial(t, x_coeffs) for t in t_segment]
        y_values = [calculate_polynomial(t, y_coeffs) for t in t_segment]
        z_values = [calculate_polynomial(t, z_coeffs) for t in t_segment]

        # If this is not the first segment, and the first coefficient (c0) of the current segment
        # does not match the last calculated point of the previous segment,
        # there might be an issue with how the data is intended to be used for concatenation.
        # However, usually, trajectory generators ensure c0 of next segment = last point of current segment.
        # For now, we simply append the calculated values directly.

        all_x.extend(x_values)
        all_y.extend(y_values)
        all_z.extend(z_values)

    # --- 2D Plot (X vs Y) ---
    plt.figure(figsize=(10, 8))
    plt.plot(all_x, all_y, label='2D Trajectory (X-Y)', color='blue')
    plt.scatter(all_x[0], all_y[0], color='green', zorder=5, s=100, label='Start Point')
    plt.scatter(all_x[-1], all_y[-1], color='red', zorder=5, s=100, label='End Point')

    plt.title('2D Trajectory from Polynomial Coefficients')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.axis('equal') # Ensures that one unit in x is equal to one unit in y
    plt.legend()
    plt.show()

    # --- 3D Plot (X, Y, Z) ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_x, all_y, all_z, label='3D Trajectory (X-Y-Z)', color='purple')
    ax.scatter(all_x[0], all_y[0], all_z[0], color='green', zorder=5, s=100, label='Start Point')
    ax.scatter(all_x[-1], all_y[-1], all_z[-1], color='red', zorder=5, s=100, label='End Point')

    ax.set_title('3D Trajectory from Polynomial Coefficients')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_zlabel('Z-coordinate')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Print the start and end coordinates for reference
    print(f"Overall Start Point: X={all_x[0]:.4f}, Y={all_y[0]:.4f}, Z={all_z[0]:.4f}")
    print(f"Overall End Point: X={all_x[-1]:.4f}, Y={all_y[-1]:.4f}, Z={all_z[-1]:.4f}")


