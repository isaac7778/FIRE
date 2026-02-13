import jax
import jax.numpy as jnp

EPS = 1e-8


def project_to_hypersphere(
    x: jnp.array,
    projection_type: str,
    constant: float,
) -> jnp.array:
    if projection_type == "none":
        return x

    elif projection_type == "l2":
        return l2_projection(x)

    elif projection_type == "shift":
        return shift_projection(x, shift=constant)

    elif projection_type == "angular":
        return angular_projection(x, max_abs_x=constant)

    elif projection_type == "hypercube":
        return hypercube_projection(x, max_abs_x=constant)

    elif projection_type == "hemisphere":
        return hemisphere_projection(x, max_abs_x=constant)

    else:
        raise NotImplementedError


def l2normalize(
    x: jnp.ndarray,
    axis: int,
) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    x = x / jnp.maximum(l2norm, EPS)

    return x


def l2_projection(
    x: jnp.array,
) -> jnp.array:
    """
    Args:
        x: A d-dimensional input vector of shape (..., d).
    Returns:
        y A d-dimensional point on the unit hypersphere of shape (..., d).
    """
    return l2normalize(x, -1)


def shift_projection(
    x: jnp.array,
    shift: float = 3.0,
) -> jnp.array:
    """
    Args:
        x: A d-dimensional input vector of shape (..., d).
        shift: The scaling factor to shift toward new axis.
    Returns:
        y (d+1)-dimensional point on the unit hypersphere of shape (..., d+1).
    """
    new_axis = jnp.ones((x.shape[:-1] + (1,))) * shift
    x = jnp.concatenate([x, new_axis], axis=-1)
    y = l2normalize(x, axis=-1)

    return y


def angular_projection(
    x: jnp.array, max_abs_x: float = 6.0, angle_range: float = jnp.pi / 4
) -> jnp.array:
    """
    Args:
        x: A d-dimensional input vector of shape (..., d).
        max_abs_x: maximum absolute value for clipping the input vector.
        angle_scale: The scaling factor for mapping to angles.
    Returns:
        y (d+1)-dimensional point on the unit hypersphere of shape (..., d+1).
    """
    # Step 1: Scale input from [-scale, scale] to angles in [-angle_scale, angle_scale]
    x = jnp.clip(x, -max_abs_x, max_abs_x)  # Shape: (..., d)
    angles = (x / max_abs_x) * angle_range  # Shape: (..., d)

    # Step 2: Compute cosine and sine of all angles
    cos_angles = jnp.cos(angles)  # Shape: (..., d)
    sin_angles = jnp.sin(angles)  # Shape: (..., d)

    # Step 3: Compute the product of cosines up to each dimension
    # We can achieve this by using JAX’s cumulative product function with appropriate padding
    # Prepend a 1 to handle the first dimension’s product
    ones = jnp.ones(angles.shape[:-1] + (1,))
    cos_angles_padded = jnp.concatenate(
        [ones, cos_angles], axis=-1
    )  # Shape: (..., d+1)
    cumprod_cos = jnp.cumprod(cos_angles_padded, axis=-1)[:, :-1]  # Shape: (..., d)

    # Step 4: Compute each component y_i = sin(theta_i) * cumprod_cos[..., i-1]
    y_components = sin_angles * cumprod_cos  # Shape: (..., d)

    # Step 5: Compute the last component y_{d+1} = product of all cos(theta_i)
    y_last = jnp.prod(cos_angles, axis=-1, keepdims=True)  # Shape: (..., 1)

    # Step 6: Concatenate all components to form the final unit vector
    y = jnp.concatenate([y_components, y_last], axis=-1)  # Shape: (..., d+1)

    return y


def hypercube_projection(
    x: jnp.array,
    max_abs_x: float = 6.0,
) -> jnp.array:
    """
    Args:
        x: A d-dimensional input vector of shape (..., d).
        max_abs_x: maximum absolute value for clipping the input vector.
    Returns:
        y (d+1)-dimensional point on the unit hypersphere of shape (..., d+1).
    """
    d = x.shape[-1]

    # Normalize each dimension to make max-l2norm lies in 1
    x = jnp.clip(x, -max_abs_x, max_abs_x)
    x = x / (max_abs_x * jnp.sqrt(d))

    # Calculate the l2 norm
    l2_norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    # Compute the additional z-dimension for hypersphere projection
    # use abs for numerical stability
    z = jnp.sqrt(jnp.abs(1 - l2_norm * l2_norm))

    # Return the projected vector in (d+1) dimensions
    y = jnp.concatenate([x, z], axis=-1)  # Shape: (..., d+1)

    return y


def hemisphere_projection(
    x: jnp.array,
    max_abs_x: float = 6.0,
) -> jnp.array:
    """
    Args:
        x: A d-dimensional input vector of shape (..., d).
        max_abs_x: maximum absolute value for clipping the input vector.
    Returns:
        y (d+1)-dimensional point on the unit hypersphere of shape (..., d+1).
    """
    d = x.shape[-1]

    # Normalize each dimension to make max-l2norm lies in 1
    x = jnp.clip(x, -max_abs_x, max_abs_x)
    x = x / (max_abs_x * jnp.sqrt(d))

    # Find an intersection point between hypercube and stretched vector
    max_abs_val = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    hypercube_intersect_x = x * (1 / jnp.sqrt(d)) * (1 / (max_abs_val + EPS))

    # Find a stretching factor
    stretch_factor = 1 / (
        jnp.linalg.norm(hypercube_intersect_x, ord=2, axis=-1, keepdims=True) + EPS
    )
    x = x * stretch_factor

    # Calculate the l2 norm
    l2_norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    # Compute the additional z-dimension for hypersphere projection
    # use abs for numerical stability
    z = jnp.sqrt(jnp.abs(1 - l2_norm * l2_norm))

    # Return the projected vector in (d+1) dimensions
    y = jnp.concatenate([x, z], axis=-1)  # Shape: (..., d+1)

    return y


if __name__ == "__main__":
    inputs = jnp.array(
        [
            [0.0, 0.0],
            [3.0, 1.0],
            [1.0, -1.0],
        ]
    )
    outputs = l2normalize(inputs, -1)

    projection_types = ["none", "l2", "shift", "angular", "hypercube", "hemisphere"]

    inputs = [
        jnp.array([[-1.0, 1.0]]),
        jnp.array([[6.0, -6.0, 6.0]]),
        jnp.array(
            [
                [3.0, -3.0],
                [-3.0, 0.0],
                [0.0, 0.0],
                [1e-8, 1e-9],
            ]
        ),
        jnp.array(
            [
                [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    ]

    constants = [
        3.0,
        6.0,
    ]

    for projection_type in projection_types:
        print(f"#########################")
        print(f"Testing {projection_type}:")
        for input in inputs:
            for constant in constants:
                result = project_to_hypersphere(
                    x=input,
                    projection_type=projection_type,
                    constant=constant,
                )
                l2_norm = jnp.linalg.norm(
                    result, ord=2, axis=-1
                )  # Measure the norm of the output
                print(f"input: {input}")
                print(f"constant: {constant}")
                print(f"output: {result}")
                print(f"norm: {l2_norm}")
                print()
        print(f"#########################")
