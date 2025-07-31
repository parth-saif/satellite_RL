"""
Drone Orbit Generation Module
1. Scale orbit from space-scale to drone-scale.
2. Generate 4 Cubic Bezier Curve segements for elliptical orbit trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

class DroneOrbitGenerator:
    def __init__(self, orbital_elems, dist_sf=1e-6, time_sf=1e-2):
        """
        Initialise Orbit generator
        :param orbital_elems: List of orbital elements excluding true anomaly.
        """
        self.orbital_elems = orbital_elems
        self.a = orbital_elems[0] # semi-major axis in m
        self.e = orbital_elems[1] # eccentricity - non-dim
        self.i = orbital_elems[2] # inclination in rad
        self.a_n = orbital_elems[3] # ascending node in rad
        self.ar_p = orbital_elems[4] # argument of periapsis in rad

        self.dist_sf = dist_sf
        self.time_sf = time_sf

        self.ellipse_params = None # initialise 3d ellipse parameters
        self.bezier_segs = None #  initialise Bezier segments

    def scale_trajectory(self):
        self.a *= self.dist_sf # only need to scale the semi-major axis
    
    def orbital_elements_to_ellipse_params(self):
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
    
    def ellipse_to_bezier_segments_3d(self):
        """
        Converts 3D ellipse parameters into 4 cubic Bézier curve segments - 4 is enough for general ellipse.
        """
        self.orbital_elements_to_ellipse_params()

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

    def evaluate_bezier(self, t, p0, p1, p2, p3): # method to evaluate a bezier segment
        return ((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)
    
    def get_drone_orbit(self):
        """
        1. Scale the orbit.
        2. Extract ellipse params.
        3. Return bezier segements.
        """

        self.scale_trajectory()

        self.orbital_elements_to_ellipse_params()

        self.ellipse_to_bezier_segments_3d()

        return self.bezier_segs

    def plot_orbits(self, ax):
        """
        Visualize the 3D orbit from orbital elements and the Bézier curves.
        """
        ax.set_title('3D Orbit from Elements and Bézier Trajectory')

        # Plot a reference sphere for the central body (e.g., Earth)
        ax.scatter([0], [0], [0], c='b', marker='o', s=150, label='Central Body (Focus)')

        # Plot Bézier curves and their control points
        if self.bezier_segs is not None:
            t_points = np.linspace(0, 1, 100) # plotting points
            all_curve_points = []
            
            for i, segment in enumerate(self.bezier_segs):
                p0, p1, p2, p3 = segment
                curve_points = np.array([self.evaluate_bezier(t, p0, p1, p2, p3) for t in t_points])
                all_curve_points.extend(curve_points)
                ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], 'g-', lw=3, label='Bézier Trajectory' if i == 0 else "")
                
                control_polygon = np.array([p0, p1, p2, p3])
                ax.plot(control_polygon[:, 0], control_polygon[:, 1], control_polygon[:, 2], 'k--', alpha=0.5, label='Control Polygon' if i == 0 else "")
                ax.scatter(control_polygon[:, 0], control_polygon[:, 1], control_polygon[:, 2], c='k', marker='s', s=25, label='Control Points' if i == 0 else "")

        # Set axis labels
        ax.set_xlabel('X (ECI) [m]')
        ax.set_ylabel('Y (ECI) [m]')
        ax.set_zlabel('Z (ECI) [m]')
        
        # Auto-scaling axes for better view
        all_points = np.array(all_curve_points)
        max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), all_points[:,1].max()-all_points[:,1].min(), all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        

#test
if __name__ == '__main__':
    import pandas as pd
    # read keplerian elements from csv file
    target_orb_elems = pd.read_csv("./target_trajectory.csv").values[0]
    chaser_orb_elems = pd.read_csv("./chaser_trajectory.csv").values[0]

    target_orb = DroneOrbitGenerator(target_orb_elems, dist_sf=1e-6)
    chaser_orb = DroneOrbitGenerator(chaser_orb_elems, dist_sf=1e-6)

    target_bez = target_orb.get_drone_orbit()
    chaser_bez = chaser_orb.get_drone_orbit()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    target_orb.plot_orbits(ax)
    chaser_orb.plot_orbits(ax)
    plt.show()
