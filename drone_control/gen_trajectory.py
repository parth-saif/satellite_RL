"""
---Trajectory Generation modules---

TrajectoryGenerator: Base trajectory generation class
- Contains scaling factor properties.

OrbitGenerator:
- Generates elliptical orbits as Bezier segments.
1. Scale orbit from space-scale to drone-scale.
2. Generate 4 Cubic Bezier Curve segements for elliptical orbit trajectory.
3. Create compressed trajectory format for upload.

ManoeuvreGenerator:
- Generates 7th order polynomial segements for maneuvers.
1. Scale maneuver waypoints to drone-scale.
2. Fit 7th order polynomial segements to waypoints.
3. Convert to Bezier segments.
4. Create compressed Bezier trajectory format for upload.
"""

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
from cflib.crazyflie.mem import CompressedSegment
from cflib.crazyflie.mem import CompressedStart
from typing import Tuple, List

packages_folders = ['./packages/uav_trajectories/scripts', './tests'] # path to the uav_trajectories package
import sys

for packages_folder in packages_folders:
    sys.path.append(packages_folder) # add the package to the system path

# open source scripts for fitting 7th order polynomial segments from waypoints
from generate_trajectory import generate_trajectory # open source trajectory generation package
from uav_trajectory import Trajectory, Polynomial # import the Trajectory class from the package

class PolyTrajectory(Trajectory):
    """
    PolyTrajectory child class to handle polynomial trajectories.
    Inherits from the Trajectory class to provide additional functionality.
    Adds method to output trajectory as an array instead of csv.
    """
    def __init__(self, polynomials: Polynomial = None, duration: float = None):
        super().__init__()
        self.polynomials = polynomials if polynomials is not None else None
        self.duration = duration if duration is not None else None
    def to_array(self) -> np.ndarray:
        """
        Convert the trajectory to a numpy array format.
        :return: Numpy array of the trajectory in the format [[time, x, y, z, yaw], ...]
        """
        data = np.array([
        [p.duration] + list(p.px.p) + list(p.py.p) + list(p.pz.p) + list(p.pyaw.p)
        for p in self.polynomials
        ])
        return data

class TrajectoryGenerator: # parent trajectory generator class
    """
        Base trajectory generator class. Handles shared proprties and static methods.
    """
    def __init__(self, dist_sf: float = 1e-6, speed_up: float = 100):
        self.dist_sf = dist_sf # space-scale to drone-scale scaling factor
        self.speed_up = speed_up # speed up factor

        self.bezier_segs = None # bezier trajectory

    def get_trajectory(self): # extract compressed format trajectory for upload to Crazyflie
        pass

    @staticmethod
    def evaluate_bezier(t: float, control_points: np.ndarray) -> np.ndarray:
        """
        Evaluates a Bézier curve of any order at parameter t.
        
        Args:
            t (float): Parameter value between 0 and 1
            control_points (np.ndarray): Array of shape (n+1, 3) containing control points,
                                       where n is the order of the curve
        
        Returns:
            np.ndarray: The point on the curve at parameter t
        """
        n = len(control_points) - 1  # degree of Bezier curve
        point = np.zeros(3)
        
        for i in range(n + 1):
            # Calculate Bernstein polynomial coefficient using binomial coeffs
            coef = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            point += coef * control_points[i]
            
        return point
    
    @staticmethod
    def comb(a: int, b: int) -> int: # utility function to compute binomial coefficients
        f = math.factorial
        return f(a) / f(b) / f(a - b)
    
    def plot_bez(self, ax):
        t_points = np.linspace(0, 1, 100) # time between 0 and 1 for evaluting Bezier curves
        all_curve_points = []
        
        for i, segment in enumerate(self.bezier_segs):
            control_points = segment  # Control points from remaining rows
            curve_points = np.array([self.evaluate_bezier(t, control_points) for t in t_points])
            all_curve_points.extend(curve_points)
            ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], 'g-', lw=3, label='Bézier Trajectory' if i == 0 else "")
            
            control_polygon = np.array(control_points)
            ax.plot(control_polygon[:, 0], control_polygon[:, 1], control_polygon[:, 2], 'k--', alpha=0.5, label='Control Polygon' if i == 0 else "")
            ax.scatter(control_polygon[:, 0], control_polygon[:, 1], control_polygon[:, 2], c='k', marker='s', s=25, label='Control Points' if i == 0 else "")

class OrbitGenerator(TrajectoryGenerator):
    def __init__(self, orbital_elems: list, dist_sf: float = 1e-6, speed_up: float =100):
        """
        Orbit Generator computes 4 cubic Bezier curves to generate orbital trajectories.
        :param orbital_elems: List of orbital elements excluding true anomaly.
        :param dist_sf: Distance scale factor.
        :param speed_up: Time speed-up factor for drone orbit.
        """
        super().__init__(dist_sf, speed_up)

        self.MU_EARTH = 3.986004418e14 # constant gravitational parameter for Earth

        self.orbital_elems = orbital_elems
        self.a_unscaled = orbital_elems[0] # semi-major axis in m
        self.e = orbital_elems[1] # eccentricity - non-dim
        self.i = orbital_elems[2] # inclination in rad
        self.a_n = orbital_elems[3] # ascending node in rad
        self.ar_p = orbital_elems[4] # argument of periapsis in rad

        self.a = self.a_unscaled * self.dist_sf

        self.ellipse_params = self.orbital_elements_to_ellipse_params() # calculate ellipse params
        self.bezier_segs = self.ellipse_to_bezier_segments_3d # compute Bezier curve segements

        self.segment_durations = np.array(self.calculate_segment_durations()) # Calculate realistic durations
        self.segment_durations_scaled = self.segment_durations/self.speed_up 
    
    def orbital_elements_to_ellipse_params(self) -> Tuple:
        """
        Converts classical orbital elements into 3D geometric ellipse parameters
        needed for Bézier curve generation.

        Args:
            semi_major_axis (float): 'a', in km.
            eccentricity (float): 'e', dimensionless.
            inclination (float): 'i', in radians.
            ascending_node (float): 'Ω' (Omega), in radians.
            argument_of_periapsis (float): 'ω' (omega), in radians.

        Returns:
            tuple: (center_3d, major_axis_vec, minor_axis_vec, normal_vec)
        """

        # Ellipse geometric parameters
        b = self.a * np.sqrt(1 - self.e**2) # semi minor axis
        c = self.a * self.e # disrance from centre to focus

        # Create the rotation matrix from the orbital (perifocal) frame to the Earth-Centered Inertial (ECI) frame.
        # Rotation sequence is Rz(a_n) -> Rx(i) -> Rz(ar_p) : see https://control.asu.edu/Classes/MAE462/462Lecture07.pdf
        cos_a_n = np.cos(self.a_n)
        sin_a_n = np.sin(self.a_n)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        cos_ar_p= np.cos(self.ar_p)
        sin_ar_p = np.sin(self.ar_p)

        # Rotation matrix for Longitude of Ascending Node
        Rz_a_n = np.array([[cos_a_n, -sin_a_n, 0], [sin_a_n, cos_a_n, 0], [0, 0, 1]])
        # Rotation matrix for Inclination 
        Rx_i = np.array([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
        # Rotation matrix for Argument of Periapsis 
        Rz_ar_p = np.array([[cos_ar_p, -sin_ar_p, 0], [sin_ar_p, cos_ar_p, 0], [0, 0, 1]])

        # The full rotation matrix from perifocal to ECI frame
        R = Rz_a_n @ Rx_i @ Rz_ar_p

        # Construct the 3D geometric vectors in perifocal frame.
        # In the perifocal frame, the vector to periapsis is along the x-axis
        # and the vector along the semi-minor axis is along the y-axis.
        p_vec_perifocal = np.array([1, 0, 0])
        q_vec_perifocal = np.array([0, 1, 0])

        # Rotate these vectors to the ECI frame to get their 3D orientation
        p_vec_eci = R @ p_vec_perifocal
        q_vec_eci = R @ q_vec_perifocal

        # The major axis vector points from the center to periapsis
        major_axis_vec = self.a * p_vec_eci
        # The minor axis vector is perpendicular to the major axis
        minor_axis_vec = b * q_vec_eci
        
        # The center of the ellipse is offset from the focus (origin) by 'c'
        # in the direction *opposite* to the periapsis vector.
        center_3d = -c * p_vec_eci
        
        # The normal vector to the orbital plane
        normal_vec = np.cross(major_axis_vec, minor_axis_vec)
        normal_vec /= np.linalg.norm(normal_vec)

        self.ellipse_params = (center_3d, major_axis_vec, minor_axis_vec, normal_vec)
    
    def ellipse_to_bezier_segments_3d(self) -> np.ndarray:
        """
        Converts 3D ellipse parameters into 4 cubic Bézier curve segments - 4 is enough for general ellipse.
        """
        if self.ellipse_params is None:
            return None
            
        center, major_axis, minor_axis, _ = self.ellipse_params
        
        kappa = 0.552284749831 # Magic number for Bézier circle approximation

        # get bezier control points by applying 90 deg rotation matrices
        p = [
            center + major_axis, center + major_axis + kappa * minor_axis, center + kappa * major_axis + minor_axis, center + minor_axis,
            center + minor_axis, center - kappa * major_axis + minor_axis, center - major_axis + kappa * minor_axis, center - major_axis,
            center - major_axis, center - major_axis - kappa * minor_axis, center - kappa * major_axis - minor_axis, center - minor_axis,
            center - minor_axis, center + kappa * major_axis - minor_axis, center + major_axis - kappa * minor_axis, center + major_axis,
        ]

        self.bezier_segs = np.array([p[0:4], p[4:8], p[8:12], p[12:16]])

    def __true_to_eccentric_anomaly(self, nu: float) -> float:
        """
        Converts true anomaly (nu) to eccentric anomaly (E).
        This provides an intermediate step to getting the average duration of a segment.
        """
        # This formula is robust for all quadrants
        E = 2 * np.arctan(np.sqrt((1 - self.e) / (1 + self.e)) * np.tan(nu / 2))
        return E

    def __eccentric_to_mean_anomaly(self, E: float) -> float:
        """
        Converts eccentric anomaly (E) to mean anomaly (M) using Kepler's Equation.
        """
        M = E - self.e * np.sin(E)
        return M
    
    def calculate_segment_durations(self) -> list:
        """
        Computes the time duration for each of the four elliptical quadrant segments
        based on Kepler's Second Law.
        
        Returns:
            list: duration of each segment in seconds.
        """
        # Check for valid elliptical orbit
        if not (0 <= self.e < 1):
            print(f"Warning: Eccentricity e={self.e:.2f} is not valid for an elliptical orbit. Using equal durations.")
            # Fallback to a simple period calculation and equal division
            T = 2 * np.pi * np.sqrt(self.a_unscaled**3 / self.MU_EARTH)
            return [T/4] * 4

        # Calculate Mean Motion (n) using the unscaled semi-major axis - mean motion is average angular speed of a satellite
        # in a circular orbit with the same period. It is used to compute duration from the change in mean anomaly.
        n = np.sqrt(self.MU_EARTH / self.a_unscaled**3)

        # Define the true anomalies (nu) at the boundaries of the four quadrants - true anomaly is position on orbit.
        nu_boundaries = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        
        # Convert true anomalies to mean anomalies
        E_boundaries = [self.__true_to_eccentric_anomaly(nu) for nu in nu_boundaries]
        M_boundaries = [self.__eccentric_to_mean_anomaly(E) for E in E_boundaries]

        # Calculate the time of flight (delta_t) for each segment
        durations = []
        for i in range(4):
            # Handle the wrap-around for the last segment
            delta_M = M_boundaries[i+1] - M_boundaries[i]
            if delta_M < 0:
                delta_M += 2 * np.pi
                
            delta_t = delta_M / n
            durations.append(delta_t)
            
        return durations
    
    def get_drone_orbit(self) -> Tuple[List, List]: # extract bezier segments and durations, both scaled to drone-scale
        self.orbital_elements_to_ellipse_params()
        self.ellipse_to_bezier_segments_3d()

        return self.bezier_segs, self.segment_durations_scaled
    
    def get_trajectory(self) -> list[CompressedStart, CompressedSegment]: # compose trajectory in compressed format for upload. Assume yaw is 0 for now.
        trajectory = [ # duration, elem_x, elem_y, elem_z, elem_yaw -> the elems are Bezier curve control points.
            # use first control point as start?
            CompressedStart(0.0,self.bezier_segs[0, 0, 0], self.bezier_segs[0, 0, 1], self.bezier_segs[0, 0, 2], []),
            # OMIT first control point as it is assumed start = end of last
            CompressedSegment(self.segment_durations_scaled[0], self.bezier_segs[0, 1:, 0], self.bezier_segs[0, 1:, 1], self.bezier_segs[0, 1:, 2], []), # seg 1
            CompressedSegment(self.segment_durations_scaled[1], self.bezier_segs[1, 1:, 0], self.bezier_segs[1, 1:, 1], self.bezier_segs[1, 1:, 2], []), # seg 2
            CompressedSegment(self.segment_durations_scaled[2], self.bezier_segs[2, 1:, 0], self.bezier_segs[2, 1:, 1], self.bezier_segs[2, 1:, 2], []), # seg 3
            CompressedSegment(self.segment_durations_scaled[3], self.bezier_segs[3, 1:, 0], self.bezier_segs[3, 1:, 1], self.bezier_segs[3, 1:, 2], []) # seg4
        ]
        return trajectory

    def plot_orbits(self, ax):
        """
        Visualize the 3D orbit from orbital elements and the Bézier curves.
        """
        ax.set_title('3D Orbit from Elements and Bézier Trajectory')

        # Plot a reference sphere for the central body (e.g., Earth)
        ax.scatter([0], [0], [0], c='b', marker='o', s=150, label='Central Body (Focus)')

        self.plot_bez(ax) # plot bezier curves

        # Set axis labels
        ax.set_xlabel('X (ECI) [m]')
        ax.set_ylabel('Y (ECI) [m]')
        ax.set_zlabel('Z (ECI) [m]')
        ax.legend()

class ManoeuvreGenerator(TrajectoryGenerator):
    def __init__(self, sat_waypoints: list, num_segments: int, max_waypoints: int =None, dist_sf: float =1e-6, speed_up: float =100):
        """
        Manoeuvre generator fits segemnts of 7th order polynomials to create a trajectory with equal durations.
        Converts to 7th order Bezier curves for data compression.
        
        :param sat_waypoints: Satellite time-waypoints.
        :param num_segments: Number of 7th order polynomial segments to fit.
        :param max_waypoints: Maximum number of waypoints to use in polynomial fit.
        :param dist_sf: Distance scale factor.
        :param speed_up: Time speed-up factor for drone orbit.
        """
        super().__init__(dist_sf, speed_up)
    
        self.sat_waypoints = np.array(sat_waypoints)
        self.num_segments = num_segments
        self.max_waypoints = max_waypoints if max_waypoints is not None else None

        # if max_waypoints given, reduce the number of waypoints used for fitting
        if self.max_waypoints is not None and self.sat_waypoints.shape[0] > self.max_waypoints:
            self.sat_waypoints = self.sat_waypoints[::sat_waypoints.shape[0]//self.max_waypoints]

        # Scale waypoints
        self.scaled_pos = self.sat_waypoints[:, 1:4] * self.dist_sf
        self.scaled_time = self.sat_waypoints[:, 0] * self.speed_up
        self.drone_waypoints = np.hstack((self.scaled_time[:, np.newaxis], self.scaled_pos))
        self.duration = self.drone_waypoints[-1, 0] # duration of drone manoeuvre

        # set drone waypoints, set yaw = 0 if not provided
        if self.drone_waypoints.shape[1] < 5:
            zeros = np.zeros((self.drone_waypoints.shape[0], 1))
            self.drone_waypoints = np.hstack((self.drone_waypoints, zeros))
        self.trajectory = PolyTrajectory() # init drone trajectory

        # generate polynomial trajectories
        self.traj_array = self.gen_drone_traj()

        # convert to bezier trajectory
        self.bezier_segs = self.poly2bez()
        
    def gen_drone_traj(self) -> np.ndarray:
        """
        Generate a drone trajectory as a 7th order polynomial segments from the provided waypoints.
        """
        trajectory = generate_trajectory(self.drone_waypoints, self.num_segments) # this creates a trajectory object with polynomial segments
        self.trajectory = PolyTrajectory(polynomials=trajectory.polynomials, duration=self.duration) # convert to DroneTrajectory object
            
        traj_array = self.trajectory.to_array()  # convert the trajectory to a numpy array format
        return traj_array

    def poly2bez(self) -> np.ndarray:
        """
        Convert polynomial segments to Bezier segments for compression.
        Match the position and velocity.

        Returns:
            np.ndarray: An array of shape (num_segments, 8, 3) representing the
                        control points for each 7th order Bezier curve and durat
        """
        matrix = []
        for k in range(0, 8):
            matrixrow = []
            for i in range(0, 8):
                if i > k:
                    matrixrow.append(0)
                else:
                    eff = self.comb(7 - i, k - i) * pow(-1, k - i) * self.comb(7, i)
                    matrixrow.append(eff)
            matrix.append(matrixrow)

        matrixnp = np.array(matrix)
        invmtr = inv(matrixnp)

        bezier_traj = []

        for trj in self.traj_array:
            duration = trj[0]
            multip = 1
            for i in range(8):
                trj[1 + i] *= multip
                trj[9 + i] *= multip
                trj[17 + i] *= multip
                multip *= duration

            xnp = np.transpose(np.array(trj[1:9]))
            ynp = np.transpose(np.array(trj[9:17]))
            znp = np.transpose(np.array(trj[17:25]))

            xctrl = np.matmul(invmtr, xnp).tolist()
            yctrl = np.matmul(invmtr, ynp).tolist()
            zctrl = np.matmul(invmtr, znp).tolist()

            # Stack control points into shape (8,3)
            control_points = np.column_stack((xctrl, yctrl, zctrl))        

            bezier_traj.append(control_points)    

        return np.array(bezier_traj)

    def get_trajectory(self) -> list[CompressedStart, CompressedSegment]: # compose compressed trajectory, assuming yaw is 0 for now.
        trajectory = [CompressedStart(0.0,self.bezier_segs[0, 0, 0], self.bezier_segs[0, 0, 1], self.bezier_segs[0, 0, 2], [])]
        num_segments = self.bezier_segs.shape[0]
        segment_duration = self.duration / num_segments # use equal durations for segments
        for i in range(num_segments):
            trajectory.append(
                CompressedSegment(
                    segment_duration,
                    self.bezier_segs[i, 1:, 0],  # Slice to get points 1 through 7 for x
                    self.bezier_segs[i, 1:, 1],  # Slice to get points 1 through 7 for y
                    self.bezier_segs[i, 1:, 2],  # Slice to get points 1 through 7 for z
                    []  # Assuming yaw is not used here
                )
            )
        return trajectory

    def plot_manoeuvre(self, ax, plot_poly: bool = True, plot_bezier: bool = True):
        """
        Plots waypoints, a 7th-order polynomial trajectory, and Bézier segments on a 3D axis.

        Args:
            ax (Axes3D): The matplotlib 3D axes object to plot on.
            plot_poly (bool): If True, plots the polynomial trajectory.
            plot_bezier (bool): If True, plots the Bézier trajectory.

        """
        time_vec = np.linspace(0, self.trajectory.duration - self.trajectory.duration/500, 500)

        # Plot the reference waypoints
        ax.scatter(self.drone_waypoints[:, 1], self.drone_waypoints[:, 2], self.drone_waypoints[:, 3], 
                color='red', s=100, label='Waypoints', depthshade=False, zorder=10)

        # Plot the 7th-order polynomial trajectory
        if plot_poly:
            poly_points = np.array([self.trajectory.eval(t).pos for t in time_vec])
            ax.plot(poly_points[:, 0], poly_points[:, 1], poly_points[:, 2], 
                    'b-', lw=3, label='7th Order Polynomial')

        # Plot the converted Bézier segments
        if plot_bezier:
            self.plot_bez(ax)

    def eval_poly(self) -> np.ndarray:
        """
        Return evaluated polynomial positions in the experiment duration - for plotting.
        """
        time_vec = np.linspace(0, (self.trajectory.duration)-(self.trajectory.duration)/500, 500)
    
        # Evaluate the trajectory at each point in the time vector
        # The .eval(t) method returns a state object with a .pos attribute
        eval_points = np.array([self.trajectory.eval(t).pos for t in time_vec])

        return eval_points

#test
if __name__ == '__main__':
    import pandas as pd
    # read keplerian elements from csv file
    target_orb_elems = pd.read_csv("./target_trajectory.csv").values[0]
    chaser_orb_elems = pd.read_csv("./chaser_trajectory.csv").values[0]

    from waypoint_gen import generate_waypoints
    start = (1,1,1)
    end = (12, 3, 5)
    num_points = 50
    t = np.linspace(0,10, num_points)

    wp, _ = generate_waypoints(start, end, num_points, method='bezier')
    
    wp = np.hstack([t[:, np.newaxis], wp])
    #print(wp)
    mg = ManoeuvreGenerator(sat_waypoints=wp, num_segments=5, dist_sf=1, speed_up=1)
    #print(mg.get_trajectory())

    target_orb = OrbitGenerator(target_orb_elems, dist_sf=1e-6)
    chaser_orb = OrbitGenerator(chaser_orb_elems, dist_sf=1e-6)

    target_bez, target_dur = target_orb.get_drone_orbit()
    chaser_bez, chaser_dur = chaser_orb.get_drone_orbit()

    

    # plot both trajectories on sample plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mg.plot_manoeuvre(ax, plot_poly=False)


    # target_orb.plot_orbits(ax)
    # chaser_orb.plot_orbits(ax)
    plt.show()
