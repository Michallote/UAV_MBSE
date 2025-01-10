from math import isclose

import numpy as np

from geometry.aircraft_geometry import GeometricCurve, all_different
from geometry.interpolation import resample_curve_equidistant
from geometry.intersection import enforce_closed_curve
from geometry.meshing import compute_3d_planar_mesh
from geometry.planar import (
    centroid_drang,
    curve_area,
    curve_centroid,
    orient_counter_clockwise,
)
from geometry.projections import (
    compute_space_curve_centroid,
    create_projection_and_basis,
    project_points_to_plane,
    transform_vertices_to_plane,
)
from geometry.surfaces import triangle_area
from visualization.plotly_plotter import plot_2d_mesh

CURVE = np.array(
    [
        [2.92431639e-01, 1.14165781e-01, 0.00000000e00],
        [2.92431639e-01, 1.14165781e-01, 9.90000000e-02],
        [2.92431639e-01, 1.14165781e-01, 1.98000000e-01],
        [2.92431639e-01, 1.14165781e-01, 2.97000000e-01],
        [2.92431639e-01, 1.14255604e-01, 3.14381042e-01],
        [2.92431639e-01, 1.14345428e-01, 3.31762085e-01],
        [2.92431639e-01, 1.14435252e-01, 3.49143127e-01],
        [2.92431639e-01, 1.14453840e-01, 3.66296807e-01],
        [2.92431639e-01, 1.14472428e-01, 3.83450488e-01],
        [2.92431639e-01, 1.14491017e-01, 4.00604168e-01],
        [2.92431639e-01, 1.14433799e-01, 4.14888962e-01],
        [2.92431639e-01, 1.14376582e-01, 4.29173756e-01],
        [2.92431639e-01, 1.14319364e-01, 4.43458550e-01],
        [2.92431639e-01, 1.14194929e-01, 4.55305700e-01],
        [2.92431639e-01, 1.14070495e-01, 4.67152850e-01],
        [2.92431639e-01, 1.13946060e-01, 4.79000000e-01],
        [2.92431639e-01, 1.13945752e-01, 4.79188076e-01],
        [2.92431639e-01, 1.13945443e-01, 4.79376151e-01],
        [2.92431639e-01, 1.13945135e-01, 4.79564227e-01],
        [2.92431639e-01, 1.13871698e-01, 4.87224921e-01],
        [2.92431639e-01, 1.13798261e-01, 4.94885614e-01],
        [2.92431639e-01, 1.13724825e-01, 5.02546308e-01],
        [2.92431639e-01, 1.13623041e-01, 5.09590854e-01],
        [2.92431639e-01, 1.13521258e-01, 5.16635400e-01],
        [2.92431639e-01, 1.13419474e-01, 5.23679947e-01],
        [2.92431639e-01, 1.13290240e-01, 5.30199596e-01],
        [2.92431639e-01, 1.13161005e-01, 5.36719245e-01],
        [2.92431639e-01, 1.13031770e-01, 5.43238895e-01],
        [2.92431639e-01, 1.12878316e-01, 5.49299628e-01],
        [2.92431639e-01, 1.12724862e-01, 5.55360362e-01],
        [2.92431639e-01, 1.12571409e-01, 5.61421095e-01],
        [2.92431639e-01, 1.12394634e-01, 5.67075989e-01],
        [2.92431639e-01, 1.12217859e-01, 5.72730883e-01],
        [2.92431639e-01, 1.12041084e-01, 5.78385777e-01],
        [2.92431639e-01, 1.11841757e-01, 5.83670212e-01],
        [2.92431639e-01, 1.11642429e-01, 5.88954646e-01],
        [2.92431639e-01, 1.11443101e-01, 5.94239080e-01],
        [2.92431639e-01, 1.11223121e-01, 5.99184229e-01],
        [2.92431639e-01, 1.11003141e-01, 6.04129379e-01],
        [2.92431639e-01, 1.10783161e-01, 6.09074528e-01],
        [2.92431639e-01, 1.10543165e-01, 6.13696833e-01],
        [2.92431639e-01, 1.10303169e-01, 6.18319138e-01],
        [2.92431639e-01, 1.10063173e-01, 6.22941443e-01],
        [2.92431639e-01, 1.09802295e-01, 6.27265350e-01],
        [2.92431639e-01, 1.09541418e-01, 6.31589256e-01],
        [2.92431639e-01, 1.09280541e-01, 6.35913163e-01],
        [2.92431639e-01, 1.08992610e-01, 6.39975711e-01],
        [2.92431639e-01, 1.08704680e-01, 6.44038258e-01],
        [2.92431639e-01, 1.08416749e-01, 6.48100806e-01],
        [2.92431639e-01, 1.08093990e-01, 6.51955007e-01],
        [2.92431639e-01, 1.07771231e-01, 6.55809209e-01],
        [2.92431639e-01, 1.07448472e-01, 6.59663410e-01],
        [2.92431639e-01, 1.07403620e-01, 6.60108940e-01],
        [2.92431639e-01, 1.07358767e-01, 6.60554470e-01],
        [2.92431639e-01, 1.07313914e-01, 6.61000000e-01],
        [2.92431639e-01, 1.07213228e-01, 6.69928735e-01],
        [2.92431639e-01, 1.07112542e-01, 6.78857469e-01],
        [2.92431639e-01, 1.07011855e-01, 6.87786204e-01],
        [2.92431639e-01, 1.06155217e-01, 7.57617982e-01],
        [2.92431639e-01, 1.05298579e-01, 8.27449761e-01],
        [2.92431639e-01, 1.04441941e-01, 8.97281539e-01],
        [2.92431639e-01, 1.04013408e-01, 9.29521026e-01],
        [2.92431639e-01, 1.03584875e-01, 9.61760513e-01],
        [2.92431639e-01, 1.03156342e-01, 9.94000000e-01],
        [2.92431639e-01, 1.02769488e-01, 1.01701881e00],
        [2.92431639e-01, 1.02382634e-01, 1.04003763e00],
        [2.92431639e-01, 1.01995780e-01, 1.06305644e00],
        [2.92431639e-01, 1.01218429e-01, 1.10605218e00],
        [2.92431639e-01, 1.00441078e-01, 1.14904792e00],
        [2.92431639e-01, 9.96637270e-02, 1.19204366e00],
        [2.92431639e-01, 9.92780666e-02, 1.21202911e00],
        [2.92431639e-01, 9.88924062e-02, 1.23201455e00],
        [2.92431639e-01, 9.85067458e-02, 1.25200000e00],
        [2.92431639e-01, 9.78339606e-02, 1.27635828e00],
        [2.92431639e-01, 9.71611753e-02, 1.30071655e00],
        [2.92431639e-01, 9.64883901e-02, 1.32507483e00],
        [2.92431639e-01, 9.53104109e-02, 1.36747042e00],
        [2.92431639e-01, 9.41324316e-02, 1.40986600e00],
        [2.92431639e-01, 9.29544524e-02, 1.45226159e00],
        [2.92431639e-01, 9.19917769e-02, 1.48706880e00],
        [2.92431639e-01, 9.10291014e-02, 1.52187602e00],
        [2.92431639e-01, 9.00664259e-02, 1.55668323e00],
        [2.92431639e-01, 8.92656705e-02, 1.58573972e00],
        [2.92431639e-01, 8.84649151e-02, 1.61479622e00],
        [2.92431639e-01, 8.76641596e-02, 1.64385271e00],
        [2.92431639e-01, 8.76065362e-02, 1.64590181e00],
        [2.92431639e-01, 8.75489127e-02, 1.64795090e00],
        [2.92431639e-01, 8.74912892e-02, 1.65000000e00],
        [2.92431639e-01, 7.63288680e-02, 1.65000000e00],
        [2.92431639e-01, 6.51664468e-02, 1.65000000e00],
        [2.92431639e-01, 5.40040255e-02, 1.65000000e00],
        [2.92431639e-01, 5.36116992e-02, 1.61332402e00],
        [2.92431639e-01, 5.32193729e-02, 1.57664803e00],
        [2.92431639e-01, 5.28270465e-02, 1.53997205e00],
        [2.92431639e-01, 5.23073606e-02, 1.48349059e00],
        [2.92431639e-01, 5.17876747e-02, 1.42700912e00],
        [2.92431639e-01, 5.12679889e-02, 1.37052766e00],
        [2.92431639e-01, 5.09483659e-02, 1.33101844e00],
        [2.92431639e-01, 5.06287430e-02, 1.29150922e00],
        [2.92431639e-01, 5.03091200e-02, 1.25200000e00],
        [2.92431639e-01, 5.00988603e-02, 1.22926654e00],
        [2.92431639e-01, 4.98886006e-02, 1.20653308e00],
        [2.92431639e-01, 4.96783408e-02, 1.18379962e00],
        [2.92431639e-01, 4.91652281e-02, 1.12694087e00],
        [2.92431639e-01, 4.86521154e-02, 1.07008212e00],
        [2.92431639e-01, 4.81390027e-02, 1.01322337e00],
        [2.92431639e-01, 4.80830031e-02, 1.00681558e00],
        [2.92431639e-01, 4.80270036e-02, 1.00040779e00],
        [2.92431639e-01, 4.79710040e-02, 9.94000000e-01],
        [2.92431639e-01, 4.74470085e-02, 9.18241956e-01],
        [2.92431639e-01, 4.69230130e-02, 8.42483913e-01],
        [2.92431639e-01, 4.63990175e-02, 7.66725869e-01],
        [2.92431639e-01, 4.61646035e-02, 7.31483913e-01],
        [2.92431639e-01, 4.59301896e-02, 6.96241956e-01],
        [2.92431639e-01, 4.56957756e-02, 6.61000000e-01],
        [2.92431639e-01, 4.54981989e-02, 6.59642845e-01],
        [2.92431639e-01, 4.53006221e-02, 6.58285690e-01],
        [2.92431639e-01, 4.51030453e-02, 6.56928535e-01],
        [2.92431639e-01, 4.45790226e-02, 6.53321204e-01],
        [2.92431639e-01, 4.40550000e-02, 6.49713873e-01],
        [2.92431639e-01, 4.35309773e-02, 6.46106543e-01],
        [2.92431639e-01, 4.29871484e-02, 6.42481770e-01],
        [2.92431639e-01, 4.24433195e-02, 6.38856997e-01],
        [2.92431639e-01, 4.18994907e-02, 6.35232224e-01],
        [2.92431639e-01, 4.13267886e-02, 6.31581607e-01],
        [2.92431639e-01, 4.07540865e-02, 6.27930990e-01],
        [2.92431639e-01, 4.01813844e-02, 6.24280373e-01],
        [2.92431639e-01, 3.95670936e-02, 6.20586095e-01],
        [2.92431639e-01, 3.89528029e-02, 6.16891817e-01],
        [2.92431639e-01, 3.83385121e-02, 6.13197539e-01],
        [2.92431639e-01, 3.76746809e-02, 6.09438458e-01],
        [2.92431639e-01, 3.70108497e-02, 6.05679377e-01],
        [2.92431639e-01, 3.63470186e-02, 6.01920296e-01],
        [2.92431639e-01, 3.56232376e-02, 5.98054325e-01],
        [2.92431639e-01, 3.48994566e-02, 5.94188355e-01],
        [2.92431639e-01, 3.41756756e-02, 5.90322385e-01],
        [2.92431639e-01, 3.33802870e-02, 5.86306642e-01],
        [2.92431639e-01, 3.25848984e-02, 5.82290899e-01],
        [2.92431639e-01, 3.17895097e-02, 5.78275156e-01],
        [2.92431639e-01, 3.09176085e-02, 5.74077301e-01],
        [2.92431639e-01, 3.00457072e-02, 5.69879445e-01],
        [2.92431639e-01, 2.91738059e-02, 5.65681590e-01],
        [2.92431639e-01, 2.82194934e-02, 5.61259463e-01],
        [2.92431639e-01, 2.72651809e-02, 5.56837337e-01],
        [2.92431639e-01, 2.63108685e-02, 5.52415210e-01],
        [2.92431639e-01, 2.52618299e-02, 5.47687612e-01],
        [2.92431639e-01, 2.42127914e-02, 5.42960013e-01],
        [2.92431639e-01, 2.31637529e-02, 5.38232415e-01],
        [2.92431639e-01, 2.19903872e-02, 5.33067140e-01],
        [2.92431639e-01, 2.08170215e-02, 5.27901865e-01],
        [2.92431639e-01, 1.96436558e-02, 5.22736590e-01],
        [2.92431639e-01, 1.83062426e-02, 5.16967792e-01],
        [2.92431639e-01, 1.69688295e-02, 5.11198995e-01],
        [2.92431639e-01, 1.56314164e-02, 5.05430198e-01],
        [2.92431639e-01, 1.40864367e-02, 4.98875609e-01],
        [2.92431639e-01, 1.25414570e-02, 4.92321021e-01],
        [2.92431639e-01, 1.09964773e-02, 4.85766433e-01],
        [2.92431639e-01, 1.04595707e-02, 4.83510955e-01],
        [2.92431639e-01, 9.92266410e-03, 4.81255478e-01],
        [2.92431639e-01, 9.38575750e-03, 4.79000000e-01],
        [2.92431639e-01, 6.50834894e-03, 4.68069869e-01],
        [2.92431639e-01, 3.63094039e-03, 4.57139738e-01],
        [2.92431639e-01, 7.53531829e-04, 4.46209607e-01],
        [2.92431639e-01, -4.36922573e-03, 4.27499591e-01],
        [2.92431639e-01, -9.49198329e-03, 4.08789575e-01],
        [2.92431639e-01, -1.46147409e-02, 3.90079559e-01],
        [2.92431639e-01, -2.10332035e-02, 3.67551957e-01],
        [2.92431639e-01, -2.74516662e-02, 3.45024355e-01],
        [2.92431639e-01, -3.38701288e-02, 3.22496754e-01],
        [2.92431639e-01, -3.63409188e-02, 3.13997836e-01],
        [2.92431639e-01, -3.88117088e-02, 3.05498918e-01],
        [2.92431639e-01, -4.12824989e-02, 2.97000000e-01],
        [2.92431639e-01, -4.12824989e-02, 1.98000000e-01],
        [2.92431639e-01, -4.12824989e-02, 9.90000000e-02],
        [2.92431639e-01, -4.12824989e-02, 0.00000000e00],
    ]
)


def test_area_equivalence():
    """Test that all methods provide a single source of truth."""

    gc = GeometricCurve(name="spar", data=CURVE)

    plane_point = gc.data[0]
    plane_normal = gc.normal
    coordinates = gc.data

    projected_coordinates = project_points_to_plane(
        coordinates, plane_point, plane_normal
    )

    area_green = curve_area(projected_coordinates)

    n = len(gc.data) - 1
    indices = np.array(
        [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
    )

    triangles = gc.data[indices]
    area_triangulation = np.sum([triangle_area(*triangle) for triangle in triangles])

    assert np.isclose(area_green, area_triangulation)


def test_centroid_equivalence():
    """Tests that the centroid is equivalent in any method"""

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / 25)
    curve = (
        (1.05 + 0.5 * np.sin(theta * 7))
        * np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    ).T

    offset = np.array([10.125, 12.73, -11.101])

    curve = curve + offset

    # bad quality mesh
    gc = GeometricCurve(name="spar", data=curve)

    coordinates = gc.data

    # Green theorem w/ projection
    projected_coordinates, basis, plane_point = create_projection_and_basis(coordinates)
    centroid_green_projection = curve_centroid(projected_coordinates)
    centroid_green_projection = transform_vertices_to_plane(
        centroid_green_projection, basis, plane_point
    )
    centroid_green_projection = np.squeeze(centroid_green_projection)

    # Delaunay mesh approach
    mesh_dict, boundary_dict = compute_3d_planar_mesh(curve)
    vertices = mesh_dict["vertices"]
    triangles = mesh_dict["vertices"][mesh_dict["triangles"]]
    centroids_mesh = np.sum(triangles, axis=1) / 3

    areas = np.vstack([triangle_area(*triangle) for triangle in triangles])
    centroid_mesh = np.sum(centroids_mesh * areas, axis=0) / np.sum(areas)

    assert np.all(np.isclose(centroid_green_projection, offset))
    assert np.all(np.isclose(gc.centroid, offset))
    assert np.all(np.isclose(centroid_mesh, offset))

    plot_2d_mesh(boundary_dict, mesh_dict, title="")

    centroid_3d = compute_space_curve_centroid(coordinates)

    print(centroid_3d)


def test_centroid_equivalence_on_bad_geometry():
    """The coordinates in CURVE produce different results for the greens algorithm but not for others,
    I believe this is caused by bad geometry."""

    gc = GeometricCurve(name="spar", data=CURVE)

    coordinates = gc.data

    coordinates = resample_curve_equidistant(coordinates, 0.1)

    # Green theorem w/ projection
    projected_coordinates, basis, plane_point = create_projection_and_basis(coordinates)
    centroid_2d = curve_centroid(projected_coordinates)
    centroid_2d_drang = np.array(
        centroid_drang(list(map(tuple, projected_coordinates)))
    )

    assert np.all(np.isclose(centroid_2d, centroid_2d_drang))

    centroid_3d = transform_vertices_to_plane(centroid_2d, basis, plane_point)
    centroid_3d_green = np.squeeze(centroid_3d)

    # Delaunay mesh approach
    mesh_dict, boundary_dict = compute_3d_planar_mesh(coordinates)
    vertices = mesh_dict["vertices"]
    triangles = vertices[mesh_dict["triangles"]]
    centroids_triangles = np.sum(triangles, axis=1) / 3

    areas = np.vstack([triangle_area(*triangle) for triangle in triangles])
    centroid_mesh = np.sum(centroids_triangles * areas, axis=0) / np.sum(areas)

    assert np.all(np.isclose(centroid_3d_green, centroid_mesh))
    assert np.all(np.isclose(centroid_3d_green, gc.centroid))
    assert np.all(np.isclose(centroid_mesh, gc.centroid))

    import plotly.graph_objects as go

    # Create the 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vertices[:, 0],  # X-axis
                y=vertices[:, 1],  # Y-axis
                z=vertices[:, 2],  # Z-axis
                mode="markers",  # Use markers
                marker=dict(
                    size=5,  # Size of markers
                    colorscale="Viridis",  # Colorscale for the markers
                    opacity=0.8,  # Opacity of markers
                ),
            )
        ]
    )
    fig.show()

    assert np.all(np.isclose(centroid_3d_green, offset))
    assert np.all(np.isclose(gc.centroid, offset))
    assert np.all(np.isclose(centroid_mesh, offset))


def test_counter_clockwise_curve_orientation():
    """Test orient_counter_clockwise function"""

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / 25)
    curve = np.c_[np.cos(theta), np.sin(theta)]
    curve = enforce_closed_curve(curve)

    oriented_curve = orient_counter_clockwise(curve)

    assert np.all(curve == oriented_curve)

    curve = np.c_[np.cos(-theta), np.sin(-theta)]
    curve = enforce_closed_curve(curve)

    oriented_curve = orient_counter_clockwise(curve)

    n = len(curve)
    assert np.all([curve[i] == oriented_curve[-i] for i in range(n)])


def test_area_closed_curve():
    """Test that curve_area and centroid is resilient
    of closed or almost closed curves (x[-1] == x[0])"""

    projected_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    area_unclosed = curve_area(projected_coordinates)

    area_closed = curve_area(enforce_closed_curve(projected_coordinates))

    assert area_closed == area_unclosed

    centroid_2d_vec = curve_centroid(projected_coordinates)
    centroid_2d_drang = np.array(
        centroid_drang(list(map(tuple, projected_coordinates)))
    )

    assert np.all(np.isclose(centroid_2d_vec, centroid_2d_drang))
