---
title: Inertia Tensor Calculations
---
# Introduction

This document will walk you through the implementation of the inertia tensor calculations feature.

The feature calculates the inertia tensor for a shell structure composed of triangular prisms. This involves creating prisms from triangles, filtering degenerate triangles, and computing the inertia tensor for each prism.

We will cover:

1. How triangular prisms are created from base triangles.
2. How degenerate triangles are filtered out.
3. How the inertia tensor is computed for the shell structure.
4. The mathematical functions used in the calculations.

# Creating triangular prisms

<SwmSnippet path="/src/structures/inertia_tensor.py" line="8">

---

We start by defining a function to create a triangular prism from a base triangle and a specified thickness. This function extends the base triangle along its normal vector.

```

def create_triangle_prism(
    triangle_coordinates: np.ndarray, thickness: float, midsurface: bool = True
) -> np.ndarray:
    """
    Create the coordinates of a triangular prism from a base triangle and a specified thickness.

    This function generates a triangular prism by extending a base triangle along its normal vector by a given thickness.
    The base triangle is defined by its three vertices.
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="17">

---

The function takes the coordinates of the triangle vertices and the thickness as parameters. It calculates the normal vector of the triangle and extends the vertices along this vector to create the prism.

```

    Parameters:
        triangle_coordinates (np.ndarray): An array containing the 3D coordinates of the three vertices of the triangle.
        thickness (float): The distance by which the prism is extended along the triangle's normal vector.

    Returns:
        np.ndarray: An array containing the coordinates of the four vertices of the triangular prism.
    """
    v1, v2, v3 = triangle_coordinates
    normal = triangle_normal(v1, v2, v3)
    v4 = v1 + thickness * normal
    prism = np.array([v1, v2, v3, v4])
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="29">

---

If the <SwmToken path="/src/structures/inertia_tensor.py" pos="30:3:3" line-data="    if midsurface:">`midsurface`</SwmToken> flag is set, the prism is adjusted to be centered around the midsurface.

```

    if midsurface:
        prism = prism - 0.5 * thickness * normal
    return prism


def triangulate_mesh(x, y, z, i, j, k) -> np.ndarray[Any, np.dtype[Any]]:
    triangle_indices = np.vstack((i, j, k)).T
    vertices = np.vstack((x, y, z)).T
    triangles = vertices[triangle_indices]
    triangles = filter_degenerate_triangles(triangles)
    return triangles
```

---

</SwmSnippet>

# Filtering degenerate triangles

<SwmSnippet path="/src/structures/inertia_tensor.py" line="41">

---

Next, we define a function to filter out degenerate triangles, which are triangles with an area close to zero.

```


def filter_degenerate_triangles(triangles: np.ndarray) -> np.ndarray:
    """Filter degenerate triangles (area close to 0.0)

    Parameters
    ----------
     - triangles : np.ndarray
            array of triangle vertices
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="50">

---

The function uses the <SwmToken path="/src/structures/inertia_tensor.py" pos="56:21:21" line-data="    return np.array([*filter(lambda x: ~np.isclose(triangle_area(*x), 0.0), triangles)])">`triangle_area`</SwmToken> function to check each triangle's area and filters out those with an area close to zero.

```

    Returns
    -------
    np.ndarray
        Triangles filtered
    """
    return np.array([*filter(lambda x: ~np.isclose(triangle_area(*x), 0.0), triangles)])
```

---

</SwmSnippet>

# Computing the inertia tensor

<SwmSnippet path="/src/structures/inertia_tensor.py" line="57">

---

We then define the main function to compute the inertia tensor for the shell structure. This function first filters out degenerate triangles.

```


def compute_inertia_tensor_of_shell(
    triangles: np.ndarray, density: float, thickness: float, midsurface: bool = True
) -> np.ndarray:
    """$x^2$"""

    triangles = filter_degenerate_triangles(triangles)
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="65">

---

It then creates triangular prisms for each triangle using the <SwmToken path="/src/structures/inertia_tensor.py" pos="67:1:1" line-data="        create_triangle_prism, thickness=thickness, midsurface=midsurface">`create_triangle_prism`</SwmToken> function.

```

    partial_create_prism = partial(
        create_triangle_prism, thickness=thickness, midsurface=midsurface
    )
    prisms = list(map(partial_create_prism, triangles))

    prisms_coordinates = np.array(prisms)
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="72">

---

The inertia tensor for each prism is computed and summed up to get the total inertia tensor for the shell structure.

```

    inertia_tensor = np.stack(
        list(map(triangle_prism_inertia_tensor, prisms_coordinates)), axis=-1
    )
    inertia_tensor = np.sum(inertia_tensor, axis=-1) * density

    return inertia_tensor
```

---

</SwmSnippet>

# Mathematical functions

<SwmSnippet path="/src/structures/inertia_tensor.py" line="79">

---

Several mathematical functions are used in the calculations. The <SwmToken path="/src/structures/inertia_tensor.py" pos="81:2:2" line-data="def triangle_prism_inertia_tensor(prism_coordinates):">`triangle_prism_inertia_tensor`</SwmToken> function computes the inertia tensor for a single triangular prism.

```


def triangle_prism_inertia_tensor(prism_coordinates):

    x, y, z = prism_coordinates.T

    Ixx = squared_moment_term(y) + squared_moment_term(z)  # $\int_m{y^2 + z^2}dm$
    Iyy = squared_moment_term(x) + squared_moment_term(z)  # $\int_m{x^2 + z^2}dm$
    Izz = squared_moment_term(x) + squared_moment_term(y)  # $\int_m{x^2 + y^2}dm$
    Ixy = product_moment_term(x, y)  # $\int_m{x*y}dm$
    Ixz = product_moment_term(x, z)  # $\int_m{x*z}dm$
    Iyz = product_moment_term(y, z)  # $\int_m{y*z}dm$
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="91">

---

It uses the Jacobian of the transformation from the standard coordinate system to the tetrahedron's coordinate system.

```

    jacobian = np.linalg.det(transformation_jacobian(prism_coordinates))

    return (
        np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]]) * jacobian
    )

```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="98">

---

The <SwmToken path="/src/structures/inertia_tensor.py" pos="99:2:2" line-data="def squared_moment_term(x: np.ndarray) -&gt; float:">`squared_moment_term`</SwmToken> function computes the second moment of mass for a tetrahedron.

```

def squared_moment_term(x: np.ndarray) -> float:
    """
    Compute the second moment of mass (integral of x^2 over the volume) for a tetrahedron defined by its vertices.

    The function calculates the integral of x^2 over the volume of a triangular prism with vertices at coordinates (x1, y1, z1),
    (x2, y2, z2), (x3, y3, z3), and (x4, y4, z4), under a linear transformation mapping a standard coordinate system
    to the a tetrahedron's coordinate system. The result is obtained using the formula derived from integrating the squared
    x-coordinate of the transformation, accounting for the volume of the transformed region.
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="131">

---

The <SwmToken path="/src/structures/inertia_tensor.py" pos="133:2:2" line-data="def product_moment_term(x: np.ndarray, y: np.ndarray):">`product_moment_term`</SwmToken> function computes the product moment of mass for a tetrahedron.

```


def product_moment_term(x: np.ndarray, y: np.ndarray):

    x1, x2, x3, x4 = x
    y1, y2, y3, y4 = y

    return (1 / 24) * (
        -x3 * y1
        - 2 * x4 * y1
        + x3 * y2
        + 2 * x4 * y2
        + 2 * x3 * y3
        + 2 * x4 * y3
        + x1 * (2 * y1 - y2 - y3 - 2 * y4)
        + 2 * x3 * y4
        + 4 * x4 * y4
        + x2 * (-y1 + 2 * y2 + y3 + 2 * y4)
    )
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="150">

---

The <SwmToken path="/src/structures/inertia_tensor.py" pos="152:2:2" line-data="def transformation_jacobian(tetrahedron_coordinates: np.ndarray) -&gt; np.ndarray:">`transformation_jacobian`</SwmToken> function computes the Jacobian matrix of the transformation.

```


def transformation_jacobian(tetrahedron_coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix of the transformation from the standard coordinate system
    to the coordinate system defined by a tetrahedron with specified vertex coordinates.

    This transformation maps a point (ε, η, ζ) in the standard coordinate system to a point
    (x, y, z) in the tetrahedron's coordinate system using the linear combination:
        x = x1 + (x2 - x1) * ε + (x3 - x1) * η + (x4 - x1) * ζ
        y = y1 + (y2 - y1) * ε + (y3 - y1) * η + (y4 - y1) * ζ
        z = z1 + (z2 - z1) * ε + (z3 - z1) * η + (z4 - z1) * ζ
    where (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) are the coordinates of the tetrahedron's vertices.
```

---

</SwmSnippet>

# Example usage

<SwmSnippet path="/src/structures/inertia_tensor.py" line="190">

---

Finally, we provide an example usage of the functions. This example creates a triangular prism and computes its inertia tensor.

```


if __name__ == "__main__":

    triangle = np.array(
        [
            [0.25, 0, 0],
            [0.70315389352, 0, -0.21130913087],
            [0.29357787137, 0, -0.49809734905],
        ]
    )
    density = 1000
    thickness = 0.15
```

---

</SwmSnippet>

<SwmSnippet path="/src/structures/inertia_tensor.py" line="203">

---

The example demonstrates how to create a triangular prism and compute its inertia tensor for different sets of triangle coordinates.

```

    prism = create_triangle_prism(triangle, thickness=thickness)

    density * triangle_prism_inertia_tensor(prism)

    triangle = (
        np.array(
            [
                [250.0, -76.24407719, -50.562258709],
                [703.153893518, 8.015242, -244.345425983],
                [293.577871374, 122.371776976, -507.347450676],
            ]
        )
        / 1000
    )

    prism = create_triangle_prism(triangle, thickness=thickness, midsurface=False)

    density * triangle_prism_inertia_tensor(prism)
```

---

</SwmSnippet>

This concludes the walkthrough of the inertia tensor calculations feature. The code is designed to handle the creation of triangular prisms, filter out degenerate triangles, and compute the inertia tensor for a shell structure efficiently.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBVUFWX01CU0UlM0ElM0FNaWNoYWxsb3Rl" repo-name="UAV_MBSE"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
