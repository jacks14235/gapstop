import numpy as np
from scipy.spatial.transform import Rotation as srot


def generate_angles_stochastic(
    cone_angle,
    min_cone_sampling,
    max_cone_sampling,
    inplane_angle=360.0,
    min_inplane_sampling=None,
    max_inplane_sampling=None,
    starting_angles=None,
    symmetry=1.0,
    angle_order="zxz",
    seed=None
):
    """Generate Euler angles with stochastic sampling intervals for both cone and inplane rotations.
    
    This function creates a set of Euler angles by sampling the sphere (cone angles) and 
    inplane rotations with random intervals between specified minimum and maximum values.
    This provides more variability compared to the deterministic generate_angles function.

    Parameters
    ----------
    cone_angle : float
        Maximum angle for sampling of cone-angles (refers to range of z-normals of particles).
    min_cone_sampling : float
        Minimum frequency for cone sampling (degrees).
    max_cone_sampling : float
        Maximum frequency for cone sampling (degrees).
    inplane_angle : float, optional
        Desired inplane-angles for particle orientations. Defaults to 360.0.
    min_inplane_sampling : float, optional
        Minimum frequency for sampling of inplane-angles. Defaults to min_cone_sampling.
    max_inplane_sampling : float, optional
        Maximum frequency for sampling of inplane-angles. Defaults to max_cone_sampling.
    starting_angles : ndarray, optional
        Triplet of Euler angles in convention as specified by angle_order. Defaults to None.
    symmetry : float, optional
        Refers to rotational symmetry of particles. Defaults to 1.0.
    angle_order : str, optional
        Convention for Euler angles. Defaults to "zxz".
    seed : int, optional
        Random seed for reproducibility. Defaults to None.

    Returns
    -------
    ndarray
        Sample of Euler angles with shape (N, 3) where N depends on the random sampling.
        Each row contains [phi, theta, psi] in the specified angle_order convention.

    Examples
    --------
    >>> # Generate angles with stochastic sampling between 5 and 15 degrees
    >>> angles = generate_angles_stochastic(360, 5, 15, symmetry=13, seed=42)
    >>> angles.shape
    (M, 3)  # M will vary based on random sampling
    
    >>> # Generate with different cone and inplane sampling ranges
    >>> angles = generate_angles_stochastic(
    ...     cone_angle=180,
    ...     min_cone_sampling=8,
    ...     max_cone_sampling=12,
    ...     min_inplane_sampling=5,
    ...     max_inplane_sampling=10,
    ...     symmetry=6
    ... )

    Notes
    -----
    The stochastic sampling creates a more irregular distribution of angles compared to
    the regular grid-like sampling in generate_angles. This can be useful for:
    - Creating more natural-looking particle distributions
    - Reducing sampling artifacts from regular grids
    - Generating training data with built-in variation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Set default inplane sampling ranges if not provided
    if min_inplane_sampling is None:
        min_inplane_sampling = min_cone_sampling
    if max_inplane_sampling is None:
        max_inplane_sampling = max_cone_sampling
    
    # Validate inputs
    if min_cone_sampling > max_cone_sampling:
        raise ValueError("min_cone_sampling must be <= max_cone_sampling")
    if min_inplane_sampling > max_inplane_sampling:
        raise ValueError("min_inplane_sampling must be <= max_inplane_sampling")
    
    # Generate stochastic cone angle samples
    cone_points = _sample_cone_stochastic(
        cone_angle, 
        min_cone_sampling, 
        max_cone_sampling
    )
    
    # Convert cone points to angles (theta and psi)
    cone_angles = _normals_to_euler_angles(cone_points, output_order=angle_order)
    cone_angles[:, 0] = 0.0  # Set phi to 0 initially
    
    starting_phi = 0.0
    
    # Apply starting rotation if provided
    cone_rotations = srot.from_euler(angle_order, angles=cone_angles, degrees=True)
    
    if starting_angles is not None:
        starting_rot = srot.from_euler(angle_order, angles=starting_angles, degrees=True)
        cone_rotations = starting_rot * cone_rotations
        starting_phi = starting_angles[0]
    
    cone_angles = cone_rotations.as_euler(angle_order, degrees=True)
    cone_angles = cone_angles[:, 1:3]  # Keep only theta and psi
    
    # Generate stochastic phi (inplane) angles
    if inplane_angle != 360.0:
        phi_max = min(360.0 / symmetry, inplane_angle)
    else:
        phi_max = inplane_angle / symmetry
    
    phi_array = _generate_stochastic_samples(
        0, 
        phi_max, 
        min_inplane_sampling, 
        max_inplane_sampling
    )
    
    if phi_array.size == 0:
        phi_array = np.array([0.0])
    
    phi_array = phi_array + starting_phi
    n_phi = len(phi_array)
    
    # Combine phi angles with cone angles
    angular_array = np.concatenate(
        [
            np.tile(phi_array[:, np.newaxis], (cone_angles.shape[0], 1)),
            np.repeat(cone_angles, n_phi, axis=0),
        ],
        axis=1,
    )
    
    return angular_array


def _sample_cone_stochastic(cone_angle, min_sampling, max_sampling, center=None, radius=1.0):
    """Create a stochastic distribution of points on a spherical cone.
    
    Parameters
    ----------
    cone_angle : float
        Angle defining the cone aperture (degrees).
    min_sampling : float
        Minimum angular spacing between samples (degrees).
    max_sampling : float
        Maximum angular spacing between samples (degrees).
    center : ndarray, optional
        Center of sphere. Defaults to origin.
    radius : float, optional
        Radius of sphere. Defaults to 1.0.
    
    Returns
    -------
    ndarray
        Array of points on the sphere with shape (N, 3).
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # Golden angle in radians for better distribution
    phi = np.pi * (3 - np.sqrt(5))
    cone_size = cone_angle / 180.0
    
    # Start with the north pole
    north_pole = np.array([0.0, 0.0, radius]) + center
    sampled_points = [north_pole]
    
    # Stochastically sample along the cone
    current_progress = 0.0
    i = 1
    
    while current_progress < cone_size:
        # Random step size between min and max
        step_size = np.random.uniform(
            min_sampling / 180.0, 
            max_sampling / 180.0
        )
        current_progress += step_size
        
        if current_progress >= cone_size:
            break
        
        # Map progress to z-coordinate
        z = 1 - current_progress
        sp_radius = np.sqrt(max(0, 1 - z * z))
        
        # Golden angle increment for azimuthal distribution
        theta = phi * i
        x = np.cos(theta) * sp_radius
        y = np.sin(theta) * sp_radius
        
        x = x * radius + center[0]
        y = y * radius + center[1]
        z = z * radius + center[2]
        
        sampled_points.append(np.array([x, y, z]))
        i += 1
    
    return np.stack(sampled_points, axis=0)


def _generate_stochastic_samples(start, end, min_spacing, max_spacing):
    """Generate samples in a range with stochastic spacing.
    
    Parameters
    ----------
    start : float
        Start of the range.
    end : float
        End of the range.
    min_spacing : float
        Minimum spacing between samples.
    max_spacing : float
        Maximum spacing between samples.
    
    Returns
    -------
    ndarray
        Array of sample values.
    """
    samples = [start]
    current = start
    
    while current < end:
        # Random step
        step = np.random.uniform(min_spacing, max_spacing)
        current += step
        
        if current < end:
            samples.append(current)
    
    return np.array(samples)


def _normals_to_euler_angles(input_normals, output_order="zxz"):
    """Convert normal vectors to Euler angles.
    
    Parameters
    ----------
    input_normals : ndarray
        Array of normal vectors with shape (N, 3).
    output_order : str, optional
        Euler angle convention. Defaults to "zxz".
    
    Returns
    -------
    ndarray
        Array of Euler angles with shape (N, 3).
    """
    normals = input_normals / np.linalg.norm(input_normals, axis=1)[:, np.newaxis]
    
    theta = np.degrees(np.arctan2(
        np.sqrt(normals[:, 0]**2 + normals[:, 1]**2), 
        normals[:, 2]
    ))
    
    psi = 90 + np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    b_idx = np.where(np.arctan2(normals[:, 1], normals[:, 0]) == 0)
    psi[b_idx] = 0
    
    phi = np.random.rand(normals.shape[0]) * 360
    
    if output_order == "zzx":
        angles = np.column_stack((phi, psi, theta))
    else:
        angles = np.column_stack((phi, theta, psi))
    
    return angles


if __name__ == "__main__":
    # Example usage and comparison
    print("Example 1: Generate stochastic angles with symmetry=13")
    angles_stochastic = generate_angles_stochastic(
        cone_angle=360,
        min_cone_sampling=8,
        max_cone_sampling=12,
        symmetry=13,
        seed=42
    )
    print(f"Generated {len(angles_stochastic)} angle sets")
    print(f"First 5 angles:\n{angles_stochastic[:5]}\n")
    
    print("Example 2: Different cone and inplane sampling ranges")
    angles_mixed = generate_angles_stochastic(
        cone_angle=180,
        min_cone_sampling=5,
        max_cone_sampling=10,
        inplane_angle=180,
        min_inplane_sampling=15,
        max_inplane_sampling=20,
        symmetry=6,
        seed=123
    )
    print(f"Generated {len(angles_mixed)} angle sets")
    print(f"Angle range: phi=[{angles_mixed[:, 0].min():.1f}, {angles_mixed[:, 0].max():.1f}]")
    print(f"            theta=[{angles_mixed[:, 1].min():.1f}, {angles_mixed[:, 1].max():.1f}]")
    print(f"            psi=[{angles_mixed[:, 2].min():.1f}, {angles_mixed[:, 2].max():.1f}]")
