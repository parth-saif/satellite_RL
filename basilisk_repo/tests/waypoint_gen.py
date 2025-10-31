"""
Testing function to generate Bezier curve segements between 2 point.
"""
import numpy as np

def generate_waypoints(start_point, end_point, num_points, **kwargs):
        """
        Generates a set of waypoints between two 3D points following a non-linear path.

        Args:
            start_point (list or np.ndarray): The starting [x, y, z] coordinates.
            end_point (list or np.ndarray): The ending [x, y, z] coordinates.
            num_points (int): The number of waypoints to generate.
            control_point_offset (float): The perpendicular distance of the control
                                        point from the midpoint. Default is 10.

        Returns:
            np.ndarray: An array of shape (num_points, 3) representing the waypoints.
        """
        # Convert points to numpy arrays for vector operations
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        # Create a linear interpolation vector from 0 to 1
        t = np.linspace(0, 1, num_points)

        # --- Calculate the straight-line path (the baseline) ---
        main_vector = end_point - start_point
        straight_path = start_point + t[:, np.newaxis] * main_vector

        # --- Find a vector perpendicular to the main path for the non-linear deviation ---
        # This is needed to define the direction of the curve or wave.
        # We find it using the cross product with an arbitrary "up" vector.
        if np.allclose(main_vector, [0, 0, 0]): # Handle case where start and end are the same
            perp_vector = np.array([1.0, 0, 0])
        else:
            up_vector = np.array([0, 0, 1.0])
            # If the main vector is vertical, the cross product with 'up' would be zero.
            # In that case, use a different vector for the cross product.
            if np.allclose(np.cross(main_vector, up_vector), 0):
                up_vector = np.array([1.0, 0, 0])
            
            perp_vector = np.cross(main_vector, up_vector)
            # Normalize to make it a unit vector
            perp_vector /= np.linalg.norm(perp_vector)

        # For a quadratic Bézier curve, we need one control point.
        # We'll place it at the midpoint, offset by the perpendicular vector.
        control_point_offset = kwargs.get('control_point_offset', 10.0)
        
        midpoint = start_point + 0.5 * main_vector
        control_point = midpoint + control_point_offset * perp_vector
        
        # Quadratic Bézier curve formula: P(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        # P0 is start_point, P1 is control_point, P2 is end_point
        t_col = t[:, np.newaxis]
        waypoints = ( (1 - t_col)**2 * start_point +
                    2 * (1 - t_col) * t_col * control_point +
                    t_col**2 * end_point )
        return waypoints, control_point
