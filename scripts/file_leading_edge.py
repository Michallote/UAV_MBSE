import numpy as np

from src.aerodynamics.airfoil import Airfoil
from src.geometry.transformations import rotation_matrix2d

airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")
airfoil.index_le

airfoil.name = "FX 73-CL2-152_TE"
airfoil.save()


airfoil = Airfoil.from_file("data/databases/airfoil_coordinates_db/s1210.dat")
airfoil.name
airfoil.plot_airfoil()

airfoil.name = "S1210"
airfoil.save()
airfoil.index_le


airfoil.intrados

np.savetxt("S1210_extrados.csv", np.array(airfoil.extrados), delimiter=" ", fmt="%.8f")
np.savetxt("S1210_intrados.csv", airfoil.intrados, delimiter=" ", fmt="%.8f")

airfoil.extrados.shape
airfoil.intrados.shape


# Pad matrix to normalize
def pad_arrays(arr1, arr2):
    max_cols = max(arr1.shape[0], arr2.shape[0])
    mat1_padded = np.pad(
        arr1,
        ((0, max_cols - arr1.shape[0]), (0, 0)),
        mode="constant",
        constant_values=False,
    )
    mat2_padded = np.pad(
        arr2,
        ((0, max_cols - arr2.shape[0]), (0, 0)),
        mode="constant",
        constant_values=-999,
    )

    return np.hstack([mat1_padded, mat2_padded])


H = pad_arrays(airfoil.extrados, airfoil.intrados)
np.savetxt(
    "S1210_extrados_intrados.csv",
    H,
    delimiter=",",
    fmt="%.8f",
    header="extrados_x,extrados_y,intrados_x,intrados_y",
)


# Create an array of 40 points from -pi/2 to pi/2
x = np.linspace(-np.pi / 2, np.pi / 2, 40)

# Apply the sine function and then scale and shift to [0, 1]
x = (np.sin(x) + 1) / 2

np.savetxt(
    "S1210_camber_thickness.csv",
    (np.c_[x, airfoil.thickness(x), airfoil.camber(x)]),
    delimiter=",",
    fmt="%.8f",
    header="x_coords,thickness,camber",
)


x_t = airfoil.max_thickness_position()

[
    *zip(
        (x_t, x_t),
        (
            np.interp(x_t, airfoil.extrados.x, airfoil.extrados.y),
            np.interp(x_t, airfoil.intrados.x, airfoil.intrados.y),
        ),
    )
]


x_t = airfoil.max_camber_position()
(x_t, airfoil.camber(x_t))


import numpy as np


def adjust_trailing_edge(
    xb: np.ndarray,
    yb: np.ndarray,
    gapnew: float,
    blend: float,
    xbte: float,
    xble: float,
    ybte: float,
    yble: float,
    le_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Number of points
    nb = len(xb)

    # Current gap calculations
    dxn = xb[0] - xb[-1]
    dyn = yb[0] - yb[-1]
    gap = np.sqrt(dxn**2 + dyn**2)

    # Direction unit vectors
    if gap > 0.0:
        dxu = dxn / gap
        dyu = dyn / gap
    else:
        dxu = -0.5 * (yb[-1] - yb[0])
        dyu = 0.5 * (xb[-1] - xb[0])

    # Blend constraint
    doc = max(min(blend, 1.0), 0.0)
    dgap = gapnew - gap

    x_add = np.zeros_like(xb)
    y_add = np.zeros_like(yb)

    # Adjust each point's coordinates
    chbsq = (xbte - xble) ** 2 + (ybte - yble) ** 2
    for i in range(nb):
        xoc = ((xb[i] - xble) * (xbte - xble) + (yb[i] - yble) * (ybte - yble)) / chbsq
        if doc == 0.0:
            tfac = 0.0
            if i == 0 or i == nb - 1:
                tfac = 1.0
        else:
            arg = min((1.0 - xoc) * (1.0 / doc - 1.0), 15.0)
            tfac = np.exp(-arg)

        if i > le_index:
            side_m = -1
        else:
            side_m = 1

        x_add[i] = 0.5 * dgap * xoc * tfac * dxu * side_m
        y_add[i] = 0.5 * dgap * xoc * tfac * dyu * side_m

    return x_add, y_add


# Example usage
# Assuming xb, yb are the x and y coordinates of your airfoil
# xbte, xble, ybte, yble need to be defined based on your specific airfoil data
# gapnew and blend are parameters you set
airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152.dat")
airfoil.name
airfoil.plot_airfoil()

xb, yb = airfoil.data.T
xble, yble = airfoil.leading_edge
xbte, ybte = airfoil.trailing_edge
blend = 0.5
gapnew = 0.06

le_index = airfoil.index_le

x_add, y_add = adjust_trailing_edge(
    xb, yb, gapnew, blend, xbte, xble, ybte, yble, le_index
)

import matplotlib.pyplot as plt

plt.plot(
    xb + x_add,
    yb + y_add,
    "r",
    marker=".",
    markeredgecolor="black",
    markersize=3,
)
# plt.axis("equal")
plt.xlim((-0.05, 1.05))
plt.legend(["TE GAP algorithm"])
plt.show()


def trailing_edge_gap(data: np.ndarray, blend: float, gapnew: float, le_index: int):

    blend = 0.5
    gapnew = 0.06
    data = airfoil.data

    trailing_edge = airfoil.trailing_edge
    trailing_edge2 = airfoil.trailing_edge2
    leading_edge = airfoil.leading_edge
    le_index = airfoil.index_le

    mean_trailing_edge = np.mean([trailing_edge, trailing_edge2], axis=0)

    chord = mean_trailing_edge - leading_edge

    epsilon = 1e-3

    dt = trailing_edge - trailing_edge2
    gap = np.linalg.norm(dt)

    if gap > 0:
        dt = dt / gap
    else:
        x_t = np.array([trailing_edge.x - epsilon, trailing_edge.x])
        y_t = airfoil.camber(x_t)
        mean_camber_te = np.diff(np.array([x_t, y_t]))
        # Rotate 90 degrees
        dt = rotation_matrix2d(theta=np.pi / 2) @ mean_camber_te
        dt = dt.flatten()
        dt = dt / np.linalg.norm(dt)

    # Blend constraint
    blend = max(min(blend, 1.0), 0.0)
    dgap = gapnew - gap

    # Normalized Chord Coordinates
    xoc = np.dot((data - leading_edge), chord) / np.linalg.norm(chord)

    arg = np.minimum((1.0 - xoc) * (1.0 / blend - 1.0), 15.0)
    thickness_factor = np.exp(-arg)

    offset_dir = np.where(np.arange(len(data)) <= le_index, 1, -1)

    offsets = 0.5 * offset_dir * dgap * thickness_factor
    offsets = np.c_[offsets] * dt

    return data + offsets
