import random
from typing import Callable
import numpy as np
import math


class DataPoint:
    def __init__(self, x, y, class_type=0):
        self.x = x
        self.y = y
        self.class_type = class_type

    def __str__(self):
        return "(" + str(self.x) + ":" + str(self.y) + ")"

    def __repr__(self):
        return str(self)


def create_circle_of_points(amount_of_points, type_class, circle_x, circle_y, circle_r):
    array = []
    for i in range(0, int(amount_of_points)):
        point = creat_point_in_circle(circle_x, circle_y, circle_r)
        point.class_type = type_class
        array.append(point)
    return array

def normal(lower, upper):
    #lower, upper = 3.5, 6
    mu, sigma = 5, 0.7
    X = np.random.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    N = np.random.norm(loc=mu, scale=sigma)
    return X

def creat_point_in_circle(circle_x, circle_y, circle_r):
    # random angle
    alpha = 2 * math.pi * np.random.normal(1)
    # random radius
    t = np.random.normal(1)
    r = circle_r * math.sqrt(abs(t))
    # calculating coordinates
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    return DataPoint(x, y)


def create_line_of_points(
    amount_of_points: int, type_class: int, func_y: Callable[[float], float]
):
    array = []
    for i in range(0, int(amount_of_points)):
        point = create_point_around_y_line(i, func_y)
        point.class_type = type_class
        array.append(point)
    return array


def create_point_around_y_line(x: int, func_y: Callable[[float], float]):
    disp = np.random.normal(0,4)
    return DataPoint(x, disp + func_y(x))


def normalize(points) -> list[DataPoint]:
    array = []
    max_x = max(point.x for point in points)
    min_x = min(point.x for point in points)
    max_y = max(point.y for point in points)
    min_y = min(point.y for point in points)
    for point in points:
        array.append(
            DataPoint(
                (point.x - min_x) / (max_x - min_x),
                (point.y - min_y) / (max_y - min_y),
                point.class_type,
            )
        )

    return array
