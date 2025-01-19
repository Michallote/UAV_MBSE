## A simple Python port - You need: pip install pygame. Note the code here is not efficient but it's made to be educational and easy
import random
from typing import Any

import numpy as np
import pygame

from geometry.interpolation import resample_curve_equidistant
from geometry.meshing import random_points_inside_curve
from src.aerodynamics.airfoil import Airfoil

atoms = []
window_size = 1200
pygame.init()
window = pygame.display.set_mode((window_size, window_size))


def draw(surface, x, y, color, size):
    for i in range(0, size):
        pygame.draw.line(surface, color, (x, y - 1), (x, y + 2), abs(size))


def atom(x, y, c):
    return {"x": x, "y": y, "vx": 0, "vy": 0, "color": c}


def randomxy():
    return round(random.random() * window_size + 1)


def create(number, color):
    group = []
    for i in range(number):
        group.append(atom(randomxy(), randomxy(), color))
        atoms.append((group[i]))
    return group


def create_from_coords(scaled_curve: np.ndarray, color) -> list[dict[str, Any]]:
    group = [atom(float(c[0]), float(c[1]), color) for c in scaled_curve]
    atoms.extend(group)
    return group


def rule(atoms1, atoms2, g):
    for i in range(len(atoms1)):
        fx = 0
        fy = 0
        for j in range(len(atoms2)):
            a = atoms1[i]
            b = atoms2[j]
            dx = a["x"] - b["x"]
            dy = a["y"] - b["y"]
            d = (dx * dx + dy * dy) ** 0.5
            if d > 0 and d < 80:
                F = g / d
                fx += F * dx
                fy += F * dy
        a["vx"] = (a["vx"] + fx) * 0.5
        a["vy"] = (a["vy"] + fy) * 0.5
        a["x"] += a["vx"]
        a["y"] += a["vy"]
        if a["x"] <= 0 or a["x"] >= window_size:
            a["vx"] *= -1
        if a["y"] <= 0 or a["y"] >= window_size:
            a["vy"] *= -1


airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")
airfoil = airfoil.with_trailing_edge_gap(te_gap=0.03, blend_distance=1.0)

curve = airfoil.data
length = np.round(airfoil.trailing_edge_gap, 2)
curve = resample_curve_equidistant(curve, length / 6)

scaled_curve = curve * window_size + np.array([0, window_size / 2])

yellow_points = random_points_inside_curve(scaled_curve, 100)

yellow = create_from_coords(yellow_points, "yellow")
red = create_from_coords(scaled_curve, "red")

run = True
while run:
    window.fill(0)
    # rule(red, red, 0.1)
    rule(yellow, red, 0.1)
    rule(yellow, yellow, 0.1)

    # Update state
    for i in range(len(atoms)):
        draw(window, atoms[i]["x"], atoms[i]["y"], atoms[i]["color"], 3)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
pygame.quit()
exit()
