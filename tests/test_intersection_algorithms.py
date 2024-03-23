import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from src.utils.intersection import calculate_intersection_curve


def plot_curves(*args):

    df_list = []

    for i, arg in enumerate(args):
        # Create a DataFrame for curve1
        df = pd.DataFrame(arg, columns=["x", "y"])
        df["curve"] = f"curve{i}"
        # Add an 'index' column to store the index of each point
        df["index"] = np.arange(len(df))
        df_list.append(df)

    # Combine both DataFrames
    df = pd.concat(df_list)

    # Plot using Plotly Express
    fig = px.line(df, x="x", y="y", color="curve", markers=True, custom_data=["index"])

    # Update hover template to show the index
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>index: %{customdata[0]}")

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    fig.show()


def test_intersection_of_identical_curves():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 100)
    theta_2 = np.linspace(0, 2 * np.pi, 25)

    curve1 = np.array([np.cos(theta), np.sin(theta)]).T

    px.line(x=curve1[:, 0], y=curve1[:, 1]).show()

    assert np.allclose(
        calculate_intersection_curve(curve1, curve1, radius=0.000001), curve1
    )


def test_intersection_of_contained_curves():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 100)

    curve1 = np.array([np.cos(theta), np.sin(theta)]).T
    curve2 = 2 * np.array([np.cos(theta), np.sin(theta)]).T

    px.line(x=curve1[:, 0], y=curve1[:, 1]).show()

    assert np.allclose(
        calculate_intersection_curve(curve1, curve2, radius=0.000001), curve1
    )

    assert np.allclose(
        calculate_intersection_curve(curve2, curve1, radius=0.000001), curve1
    )


def test_intersection_of_contained_curves_dephased():
    """Tests intersection algorithm"""

    theta = np.linspace(0, 2 * np.pi, 27)
    theta_2 = np.linspace(0, 2 * np.pi, 13)

    curve1 = np.array([np.cos(theta), np.sin(theta)]).T
    curve2 = np.array([np.cos(theta_2), np.sin(theta_2)]).T - np.array([0.25, 0.0])

    # Create a DataFrame for curve1
    df1 = pd.DataFrame(curve1, columns=["x", "y"])
    df1["curve"] = "curve1"

    # Create a DataFrame for curve2
    df2 = pd.DataFrame(curve2, columns=["x", "y"])
    df2["curve"] = "curve2"

    curve3 = calculate_intersection_curve(curve1, curve2, radius=0.000001)

    # Create a DataFrame for curve3
    df3 = pd.DataFrame(curve3, columns=["x", "y"])
    df3["curve"] = "intersection"

    # Combine both DataFrames
    df = pd.concat([df1, df2, df3])

    # Plot using Plotly Express
    fig = px.line(df, x="x", y="y", color="curve", markers=True)
    fig.show()


def test_multiple_intersections():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 183) + 0.05
    theta_2 = np.linspace(0, 2 * np.pi, 173)

    curve1 = (
        (1.05 + 0.5 * np.sin(theta * 7)) * np.array([np.cos(theta), np.sin(theta)])
    ).T
    curve2 = (
        (0.95 + 0.25 * np.sin(theta_2 * 3))
        * np.array([np.cos(theta_2), np.sin(theta_2)])
    ).T

    # Create a DataFrame for curve1
    df1 = pd.DataFrame(curve1, columns=["x", "y"])
    df1["curve"] = "curve1"

    # Create a DataFrame for curve2
    df2 = pd.DataFrame(curve2, columns=["x", "y"])
    df2["curve"] = "curve2"

    curve3 = calculate_intersection_curve(curve1, curve2, radius=0.000001)

    # Create a DataFrame for curve3
    df3 = pd.DataFrame(curve3, columns=["x", "y"])
    df3["curve"] = "intersection"

    # Combine both DataFrames
    df = pd.concat([df1, df2, df3])

    # Plot using Plotly Express
    fig = px.line(df, x="x", y="y", color="curve", markers=True)
    fig.show()

    assert np.allclose(
        calculate_intersection_curve(curve1, curve2, radius=0.000001), curve1
    )

    assert np.allclose(
        calculate_intersection_curve(curve2, curve1, radius=0.000001), curve1
    )


def test_optimized_intersections():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 193) + 0.05
    theta_2 = np.linspace(0, 2 * np.pi, 123)

    curve2 = (
        (1.05 + 0.5 * np.sin(theta * 7)) * np.array([np.cos(theta), np.sin(theta)])
    ).T
    curve1 = (
        (0.95 + 0.25 * np.sin(theta_2 * 3))
        * np.array([np.cos(theta_2), np.sin(theta_2)])
    ).T

    # Create a DataFrame for curve1
    df1 = pd.DataFrame(curve1, columns=["x", "y"])
    df1["curve"] = "curve1"

    # Create a DataFrame for curve2
    df2 = pd.DataFrame(curve2, columns=["x", "y"])
    df2["curve"] = "curve2"

    curve3 = calculate_intersection_curve(curve1, curve2, radius=0.000001)

    # Create a DataFrame for curve3
    df3 = pd.DataFrame(curve3, columns=["x", "y"])
    df3["curve"] = "intersection"

    # Combine both DataFrames
    df = pd.concat([df1, df2, df3])

    # Plot using Plotly Express
    fig = px.line(df, x="x", y="y", color="curve", markers=True)
    fig.show()

    assert np.allclose(
        calculate_intersection_curve(curve1, curve2, radius=0.000001),
        curve_intersection(curve1, curve2, radius=0.000001),
    )


def test_non_contained_intersection_curves():

    theta = np.linspace(0, 2 * np.pi, 4)
    theta_2 = np.linspace(0, 2 * np.pi, 4) + np.pi

    curve1 = (np.array([np.cos(theta), np.sin(theta)])).T
    curve2 = (np.array([np.cos(theta_2), np.sin(theta_2)])).T

    plot_curves(curve1, curve2, calculate_intersection_curve(curve1, curve2))


import time


def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


@timing_wrapper
def time_optimized_intersections():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 193) + 0.05
    theta_2 = np.linspace(0, 2 * np.pi, 123)

    curve2 = (
        (1.05 + 0.5 * np.sin(theta * 7)) * np.array([np.cos(theta), np.sin(theta)])
    ).T
    curve1 = (
        (0.95 + 0.25 * np.sin(theta_2 * 3))
        * np.array([np.cos(theta_2), np.sin(theta_2)])
    ).T

    curve3 = curve_intersection(curve1, curve2, radius=0.000001)


@timing_wrapper
def time_unoptimized_intersections():
    """Tests that identical curves return intersecting region as the same curve"""

    theta = np.linspace(0, 2 * np.pi, 193) + 0.05
    theta_2 = np.linspace(0, 2 * np.pi, 123)

    curve2 = (
        (1.05 + 0.5 * np.sin(theta * 7)) * np.array([np.cos(theta), np.sin(theta)])
    ).T
    curve1 = (
        (0.95 + 0.25 * np.sin(theta_2 * 3))
        * np.array([np.cos(theta_2), np.sin(theta_2)])
    ).T

    curve3 = calculate_intersection_curve(curve1, curve2, radius=0.000001)
