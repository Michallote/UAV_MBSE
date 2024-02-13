"""Module for calculating intersections between curves in 2D using NumPy and Matplotlib."""

from __future__ import annotations

from collections import deque

import matplotlib.path as mpath
import numpy as np

# Create a sliding window view of size 2
from numpy.lib.stride_tricks import sliding_window_view


class IntersectionRegistry:
    """
    A registry for managing intersections accessible by two distinct keys.
    """

    registry: dict[str, list[dict]]
    a_keys: dict[str, list[dict]]
    b_keys: dict[str, list[dict]]

    def __init__(self):
        self.registry = {}
        self.a_keys = {}
        self.b_keys = {}

    def add_intersection(self, intersection: dict) -> None:
        """
        Adds an intersection to the registry and updates access dictionaries for 'a' and 'b' keys.

        Parameters:
            intersection (dict): A dictionary representing an intersection, must include 'intersection_point',
                                 'a_initial', and 'b_initial' keys.
        """
        # Assuming 'intersection' is a dict with 'a_initial', 'b_initial', and other keys
        a_key = f"a{intersection['a_initial']}"
        b_key = f"b{intersection['b_initial']}"

        # Add to registry and update a_keys and b_keys dictionaries
        if a_key not in self.registry:
            self.registry[a_key] = []
        if b_key not in self.registry:
            self.registry[b_key] = []
        self.registry[a_key].append(intersection)
        self.registry[b_key].append(intersection)

        # Keep track in a_keys and b_keys
        self.a_keys[a_key] = self.registry[a_key]
        self.b_keys[b_key] = self.registry[b_key]

    def pop_intersection(self, key: str) -> dict | None:
        """
        Removes and returns the first intersection object associated with the provided key.
        Automatically updates the central registry and the corresponding 'a' or 'b' key.

        Parameters:
            key (str): The key ('a' or 'b' followed by a number) of the intersection to remove.

        Returns:
            dict | None: The removed intersection dictionary if the key exists, otherwise None.
        """
        if key not in self.registry:
            return None

        # Remove the first intersection object from the list for this key
        intersection_to_remove = self.registry[key].pop(0)
        if not self.registry[key]:
            del self.registry[key]  # Remove the key if the list is empty

        # Also remove from the corresponding list using the other key
        other_key = (
            f"a{intersection_to_remove['a_initial']}"
            if key.startswith("b")
            else f"b{intersection_to_remove['b_initial']}"
        )
        if other_key in self.registry:
            # Iterate over the list and compare necessary fields excluding 'intersection_point' for equality
            for i, item in enumerate(self.registry[other_key]):
                if (
                    item["a_initial"] == intersection_to_remove["a_initial"]
                    and item["b_initial"] == intersection_to_remove["b_initial"]
                    and item["a_final"] == intersection_to_remove["a_final"]
                    and item["b_final"] == intersection_to_remove["b_final"]
                    and item["t"] == intersection_to_remove["t"]
                    and item["u"] == intersection_to_remove["u"]
                ):
                    del self.registry[other_key][i]

            if not self.registry[other_key]:
                del self.registry[other_key]  # Remove the key if the list is empty

        return intersection_to_remove

    def get_intersections(self, key) -> list:
        """
        Retrieves a list of intersections associated with the provided key.

        Parameters:
            key (str): The key ('a' or 'b' followed by a number) to retrieve intersections for.

        Returns:
            list: A list of intersection dictionaries associated with the key, or an empty list if the key does not exist.
        """
        return self.registry.get(key, [])

    def is_empty(self, key: str | None = None) -> bool:
        """Returns a boolean for the requested registry.
        Parameters:
            key (str): The key ('a' or 'b' followed by a number) to retrieve intersections for.

        Returns:
            bool: True if the key has elements.
        """
        if key is None:
            return not bool(self.registry)

        return not bool(self.get_intersections(key))

    def __repr__(self) -> str:
        return "IntersectionRegistry:" + str(self.registry.keys())


def calculate_intersection_curve(
    curve1: np.ndarray, curve2: np.ndarray, radius: float = 0.00001
) -> np.ndarray:
    """
    Computes the intersection curve between two given curves by identifying overlapping points
    within a specified radius and stitching together segments from each curve based on intersections.

    Args:
        curve1 (np.ndarray): Points of the first curve.
        curve2 (np.ndarray): Points of the second curve.
        radius (float): The proximity radius within which points are considered overlapping.

    Returns:
        np.ndarray: The points of the intersection curve.
    """
    curve1, curve2 = enforce_closed_curve(curve1), enforce_closed_curve(curve2)

    # Identify overlapping points between the two curves
    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=radius)

    # Handle cases where one curve is entirely within the other
    if np.all(mask_1) and np.all(mask_2):
        return curve1
    if np.all(mask_1) and not np.any(mask_2):
        return curve1
    if np.all(mask_2) and not np.any(mask_1):
        return curve2

    # Traverse the curves to construct the intersection curve
    return construct_intersection_curve(curve1, curve2)


def construct_intersection_curve(curve1: np.ndarray, curve2: np.ndarray) -> np.ndarray:
    """
    Constructs the intersection curve by stitching together segments from each curve based on identified intersections.

    Args:
        curve1 (np.ndarray): Points of the first curve.
        curve2 (np.ndarray): Points of the second curve.

    Returns:
        np.ndarray: The points of the constructed intersection curve.
    """
    # Re-calculate without radius tolerance to address edge cases
    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=0.0)

    # Generate a registry mapping points to their intersection details
    intersections_registry = generate_intersection_registry(curve1, curve2)

    intersection_curve = deque()
    pointer_a = mask_1.argmax()  # Returns the first True pointer
    pointer_b = mask_2.argmax()
    current_curve = "a" if np.any(mask_1) else "b"
    n1, n2 = len(curve1), len(curve2)
    # Traverse the curve appending coordinates, intersections trigger switching
    while np.any(mask_1) or np.any(mask_2) or not intersections_registry.is_empty():
        pointer_a = pointer_a % n1
        pointer_b = pointer_b % n2

        if current_curve == "a":
            if mask_1[pointer_a]:
                print(f"curve1 {pointer_a=}")
                intersection_curve.append(curve1[pointer_a])
                mask_1[pointer_a] = False

            while not intersections_registry.is_empty(f"a{pointer_a}"):
                data = intersections_registry.pop_intersection(f"a{pointer_a}")
                intersection_curve.append(data["intersection_point"])
                pointer_b = data["b_final"]
                current_curve = "b"
                # If the next point on curve2 is inside the region, switch curves.
                if not mask_2[pointer_b]:
                    pointer_b = data["b_initial"]

            pointer_a += 1
        else:
            if mask_2[pointer_b % n2]:
                print(f"curve2 {pointer_b=}")
                intersection_curve.append(curve2[pointer_b])
                mask_2[pointer_b] = False

            while not intersections_registry.is_empty(f"b{pointer_b}"):
                data = intersections_registry.pop_intersection(f"b{pointer_b}")
                intersection_curve.append(data["intersection_point"])
                pointer_a = data["a_final"]
                current_curve = "a"
                # If the next point on curve2 is inside the region, switch curves.
                if not mask_1[pointer_a]:
                    pointer_a = data["a_initial"]

            pointer_b += 1

    return np.array(intersection_curve)


def line_segment_intersection(
    segment1: np.ndarray, segment2: np.ndarray, return_params: bool = False
) -> tuple[np.ndarray, float, float] | np.ndarray | None:
    """
    Finds the intersection point of two line segments by solving the system

        A + t(B - A) = C + u(D - C),

    where A, B are points on the first segment, and C, D on the second, with t, u as parameters.

        t(B - A) - u(D - C) = C - A

    Solves using linear algebra (Mx = b).
    Returns None if segments are parallel or intersection is out of bounds.

    Parameters:
    - segment1: Array (2, 2) for the start and end points of the first segment.
    - segment2: Array (2, 2) for the start and end points of the second segment.

    Returns:
    - Intersection point as an array (2,) if within segment bounds, else None.
    """
    a1, b1 = segment1
    c2, d2 = segment2

    # Calculate the direction vectors of the segments
    direction_vector1 = np.diff(segment1, axis=0)
    direction_vector2 = np.diff(segment2, axis=0)

    # Construct the coefficients matrix for the linear system
    coeffs_matrix = np.vstack([direction_vector1, -direction_vector2]).T

    # Compute the vector from the start of segment1 to the start of segment2
    ordinate_vector = c2 - a1

    # Solve the linear system for t and u, the parameters for the intersection point
    try:
        t, u = np.linalg.solve(coeffs_matrix, ordinate_vector)
    except np.linalg.LinAlgError:
        return None  # This catch is for numerical issues or singular matrix

    # Check if the intersection is not within the bounds of the line segments
    if not (0 <= t <= 1 and 0 <= u <= 1):
        return None

    # Calculate the intersection point
    intersection_point = (a1 + t * direction_vector1).flatten()

    if return_params:
        return (intersection_point, t, u)
    return intersection_point


def find_transition_indices(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Find indices in a array where there is a transition from True to False
    or from positive to negative.

    Parameters:
    mask (np.ndarray[bool]): A list of boolean or numeric values.

    Returns:
    list: Indices of elements where the specified transitions occur.
    """

    transitions = []

    for i in range(len(mask) - 1):
        if mask[i] != mask[i + 1]:  # Horizontal change
            transitions.append((i, i + 1))
    return transitions


def identify_overlapping_points(
    curve1: np.ndarray, curve2: np.ndarray, radius: float = 0.000001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determines the points of one curve that lie within a specified radius of the other curve.

    Args:
    curve1 (np.ndarray): The points of the first curve.
    curve2 (np.ndarray): The points of the second curve.
    radius (float): The proximity radius to consider a point of one curve as lying inside the other.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two boolean arrays indicating which points of curve1 are within
                                   curve2 and vice versa.
    """
    # Create paths from the curve points
    path_curve1 = mpath.Path(curve1)
    path_curve2 = mpath.Path(curve2)

    # Determine points within the specified radius
    mask_2 = path_curve1.contains_points(curve2, radius=radius)
    mask_1 = path_curve2.contains_points(curve1, radius=radius)

    return mask_1, mask_2


def find_segment_indices(mask: np.ndarray) -> list[np.ndarray]:
    """
    Finds the start and end indices of continuous 'True' segments in a boolean array.

    Parameters:
    - mask: (np.ndarray) A boolean numpy array.

    Returns: list[tuple(int, int)]
    - A list of tuples, where each tuple contains the start and end indices of
        continuous 'True' segments.
    """
    n = len(mask)
    # Identify indices where the value changes from True to False or vice versa
    transition_indices = np.arange(n - 1, dtype=int)[mask[:-1] != mask[1:]] + 1
    # Add the start and end of the array to handle edge cases
    segment_indices = np.sort(np.hstack([transition_indices, [0, n]]))
    # Use sliding_window_view to pair start and end indices of each segment
    segment_indices = sliding_window_view(segment_indices, window_shape=2)

    # Filter segments to include only those that are entirely True
    valid_segment_indices = [
        (start, end)
        for start, end in segment_indices
        if np.all(mask[start:end])  # and len(mask[start:end]) >= min_length
    ]

    # check if the first and last segment should be a single segment:

    # Check if both 0 and n are in any of the tuples
    has_zero = any(0 in tup for tup in valid_segment_indices)
    has_n = any((n) in tup for tup in valid_segment_indices)

    # Return True if both 0 and n were found, False otherwise
    if has_zero and has_n:
        segments = []

        last = valid_segment_indices[-1]
        first = valid_segment_indices[0]

        # Merge the first and last
        segment = np.hstack(
            [np.arange(start, end, dtype=int) for start, end in (last, first)]
        )
        segments.append(segment)

        for start, end in valid_segment_indices[1:-1]:
            segments.append(np.arange(start, end, dtype=int))

        return segments

    return [np.arange(start, end, dtype=int) for start, end in valid_segment_indices]


def enforce_closed_curve(coordinates: np.ndarray) -> np.ndarray:
    """Closes the curve if first and last coordinate are not
    equal

    Parameters
    ----------
    coordinates : np.ndarray
        coordinates array

    Returns
    -------
    np.ndarray
        closed curve
    """

    if not np.all(np.isclose(coordinates[0], coordinates[-1])):
        return np.vstack([coordinates, coordinates[0]])

    return coordinates


def generate_intersection_registry(
    curve1: np.ndarray, curve2: np.ndarray
) -> IntersectionRegistry:
    """Generates a dictionary that maps two curves to their intersections details.

    Parameters
    ----------
    curve1 : np.ndarray
        curve1 coordinates defined counter clockwise
    curve2 : np.ndarray
        curve2 coordinates defined counter clockwise

    Returns
    -------
    dict.  A dictionary containing the intersection details for each point.
        {f'a{a_initial}' :  'intersection_point' (x,y) np.array, the intersection point
                            'a_initial' int, pointer on curve1 to the first segment point
                            'a_final' int, pointer on curve1 to the second segment point
                            'b_initial' int, pointer on curve2 first segment
                            'b_final' int, pointer on curve2 second segment
                    }
    """

    def get_segments(curve):
        windows = sliding_window_view(np.arange(len(curve), dtype=int), 2)
        return curve[windows]

    intersections_registry = IntersectionRegistry()

    for i, c1_seg in enumerate(get_segments(curve1)):
        for j, c2_seg in enumerate(get_segments(curve2)):
            intersection_point = line_segment_intersection(c1_seg, c2_seg)  # type: ignore

            if intersection_point is not None:

                intersection_point, t, u = line_segment_intersection(
                    c1_seg, c2_seg, return_params=True
                )  # type: ignore

                a_initial = i
                a_final = i + 1
                b_initial = j
                b_final = j + 1

                token = {
                    "intersection_point": intersection_point,
                    "a_initial": a_initial,
                    "a_final": a_final,
                    "b_initial": b_initial,
                    "b_final": b_final,
                    "t": t,
                    "u": u,
                }
                intersections_registry.add_intersection(token)

    return intersections_registry


def legacy_calculate_intersecting_region(
    curve1: np.ndarray, curve2: np.ndarray, radius: float = 0.00001
) -> np.ndarray:

    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=radius)

    # Curves are identical within tolerance
    if np.all(mask_1) and np.all(mask_2):
        return curve1
    # curve1 is completely inside curve2
    if np.all(mask_1) and not np.any(mask_2):
        return curve1
    # curve2 is completely inside curve1
    if np.all(mask_2) and not np.any(mask_1):
        return curve2

    # Edge cases are handled already, removing the tolerance eliminates a bug where points are mistakenly counted inside and intersection lines do not actually intersect
    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=0.0)

    c1_indices = find_transition_indices(mask_1)
    c2_indices = find_transition_indices(mask_2)

    c1_intersections = [curve1[np.array(indices)] for indices in c1_indices]
    c2_intersections = [curve2[np.array(indices)] for indices in c2_indices]

    intersection_curve = []
    # Initialize pointers for both lists
    pointer_a = pointer_b = 0
    current_curve = "a"

    n1 = len(curve1) - 1
    n2 = len(curve2) - 1

    b_first_intersection = True
    starting_b = 0

    while pointer_a < n1 or pointer_b < n2:
        if current_curve == "a":
            # Check if we've reached the end of list_a
            if pointer_a >= n1:
                # Switch to list_b if list_a is exhausted
                current_curve = "b"
                pointer_a += 1
                continue

            # Append from list_a until intersection_point is encountered
            if mask_1[pointer_a]:
                intersection_curve.append(curve1[pointer_a])

                # Append from list_a until switch_value is encountered
                idx, intersection_point = legacy_line_intersection(
                    curve1[[pointer_a, pointer_a + 1]], c2_intersections
                )

                if intersection_point is not None:
                    intersection_curve.append(np.array(intersection_point))
                    current_curve = "b"
                    pointer_a += 1
                    # Retrieve the point in the other curve to continue appending points
                    pointer_b = c2_indices[idx][1]

                    if b_first_intersection:
                        starting_b = pointer_b - 1
                        b_first_intersection = False

                    continue
            pointer_a += 1
        else:
            # Check if we've reached the end of list_b
            if pointer_b >= n2:
                # Set pointer and n2 to new targets
                if n2 != starting_b:
                    pointer_b = 0
                    n2 = starting_b
                else:
                    # Switch to list_a if list_b is exhausted
                    current_curve = "a"
                    pointer_b += 1
                    break
                continue
            # Append from list_b until switch_value is encountered
            if mask_2[pointer_b]:
                intersection_curve.append(curve2[pointer_b])

            idx, intersection_point = legacy_line_intersection(
                curve2[[pointer_b, pointer_b + 1]], c1_intersections
            )
            if intersection_point is not None:
                intersection_curve.append(np.array(intersection_point))
                current_curve = "a"
                pointer_b += 1
                # Retrieve the point in the other curve to continue appending points
                if n2 != starting_b:
                    pointer_a = c1_indices[idx][1]
                continue
            pointer_b += 1
        print(f"{pointer_a=}, {pointer_b=}")

    return np.array(intersection_curve)


def legacy_calculate_curve_intersection(
    curve1: np.ndarray, curve2: np.ndarray, radius: float = 0.00001
) -> np.ndarray:

    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=radius)
    # Handle cases where one curve is entirely within the other
    # Curves are identical within tolerance
    if np.all(mask_1) and np.all(mask_2):
        return curve1
    # curve1 is completely inside curve2
    if np.all(mask_1) and not np.any(mask_2):
        return curve1
    # curve2 is completely inside curve1
    if np.all(mask_2) and not np.any(mask_1):
        return curve2

    # Re-calculate without radius tolerance to address edge cases
    mask_1, mask_2 = identify_overlapping_points(curve1, curve2, radius=0.0)

    intersections_dict = generate_intersections_dict(curve1, curve2)

    intersection_curve = deque()
    # Initialize pointers for both lists
    pointer_a = pointer_b = 0
    current_curve = "a"

    n1 = len(curve1)
    n2 = len(curve2)

    while np.any(mask_1) or np.any(mask_2):
        pointer_a = pointer_a % n1
        pointer_b = pointer_b % n2

        if current_curve == "a":
            # Append from list_a until intersection_point is encountered
            if mask_1[pointer_a]:
                intersection_curve.append(curve1[pointer_a])
                mask_1[pointer_a] = False
                # Append from list_a until switch_value is encountered
                intersect_data = intersections_dict.get(f"a{pointer_a}")

                if intersect_data is not None:

                    intersection_point = intersect_data["intersection_point"]
                    intersection_curve.append(np.array(intersection_point))
                    # Retrieve the point in the other curve to continue the trajectory
                    pointer_b = intersect_data["b_final"]
                    # Switch curve
                    current_curve = "b"

            pointer_a += 1

        else:
            # Append from list_b until switch_value is encountered
            if mask_2[pointer_b]:
                intersection_curve.append(curve2[pointer_b])
                mask_2[pointer_b] = False
                # Append from list_a until switch_value is encountered
                intersect_data = intersections_dict.get(f"b{pointer_b}")

                if intersect_data is not None:
                    intersection_point = intersect_data["intersection_point"]
                    intersection_curve.append(np.array(intersection_point))
                    # Retrieve the point in the other curve to continue the trajectory
                    pointer_a = intersect_data["a_final"]
                    # Switch curve
                    current_curve = "a"

            pointer_b += 1

    return np.array(intersection_curve)


def legacy_line_intersection(
    line_segment, segments
) -> tuple[int, tuple[float, float]] | tuple[None, None]:
    """
    Calculate the intersection point of two line segments, if it exists.

    :param line_segment: A np.ndarray containing the start and end points
                        (x, y) of the first line segment.
    :param segments: A list of np.ndarray containing the start and end points
                        (x, y) of the second line segment.
    :return: The index of the intersection and the intersection point (x, y)
                    if the segments intersect, otherwise None.
    """

    for i, segment in enumerate(segments):
        intersection = legacy_single_line_intersection(line_segment, segment)
        if intersection is not None:
            return i, intersection

    return None, None


def legacy_single_line_intersection(
    segment1: tuple[tuple[float, float], tuple[float, float]],
    segment2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """
    Calculate the intersection point of two line segments, if it exists.

    :param segment1: A tuple containing the start and end points (x, y) of the first line segment.
    :param segment2: A tuple containing the start and end points (x, y) of the second line segment.
    :return: The intersection point (x, y) if the segments intersect, otherwise None.
    """
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Parallel lines case
    if denom == 0:
        return None

    # Calculate numerators
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within the segments
    if not (0 <= t <= 1 and 0 <= u <= 1):
        return None

    # Calculate intersection point
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    return (intersect_x, intersect_y)


def main():

    import numpy as np

    # Define the points
    A = np.array([0, 0])  # Example values
    B = np.array([1, 2])
    C = np.array([0, 2])
    D = np.array([1, 0])

    segment1 = np.array([A, B])
    segment2 = np.array([C, D])

    line_segment_intersection(segment1, segment2)

    theta = np.linspace(0, 2 * np.pi, 193) + 0.05
    theta_2 = np.linspace(0, 2 * np.pi, 123)

    curve2 = (
        (1.05 + 0.5 * np.sin(theta * 7)) * np.array([np.cos(theta), np.sin(theta)])
    ).T
    curve1 = (
        (0.95 + 0.25 * np.sin(theta_2 * 3))
        * np.array([np.cos(theta_2), np.sin(theta_2)])
    ).T

    curve3 = legacy_calculate_intersecting_region(curve1, curve2, radius=0.000001)

    curve3_opt = calculate_intersection_curve(curve1, curve2, radius=0.000001)

    assert np.allclose(enforce_closed_curve(curve3), enforce_closed_curve(curve3_opt))


if __name__ == "__main__":
    main()
