# from data_generation import *
import matplotlib.pyplot as plt
import numpy as np
import math


def linear_func_y(x):
    return -100 * x + 1


class DataPoint:
    def __init__(self, x, y, class_type=0):
        self.x = x
        self.y = y
        self.class_type = class_type

    def __str__(self):
        return "(" + str(self.x) + ":" + str(self.y) + ")"

    def __repr__(self):
        return str(self)


def creat_point_in_circle(circle_x, circle_y, circle_r):
    # random angle
    alpha = 2 * math.pi * np.random.random()
    # random radius
    r = circle_r * math.sqrt(np.random.random())
    # calculating coordinates
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    return DataPoint(x, y)


def create_point_around_y_line(x, func_y):
    disp = np.random.uniform(-1, 1) + 1
    return DataPoint(x, disp * func_y(x))


def create_data(amount_of_points, func_y, interval_start_x=0, interval_end_x=100):
    array = []

    array += create_circle_of_points(int(amount_of_points / 2), -1, 70, 70, 25)
    array += create_circle_of_points(int(amount_of_points / 2), 1, 30, 30, 25)

    return array


def create_circle_of_points(amount_of_points, type_class, circle_x, circle_y, circle_r):
    array = []
    for i in range(0, int(amount_of_points)):
        point = creat_point_in_circle(circle_x, circle_y, circle_r)
        point.class_type = type_class
        array.append(point)
    return array


def normalize(points):
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
                point.class_type

            )
        )

    return array


def create_plot(data):
    x_vals, y_vals, color_vals = zip(*[(float(i.x), float(i.y), i.class_type) for i in data])
    plt.scatter(x_vals, y_vals, c=color_vals)
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.show()


if __name__ == '__main__':
    # np.random.seed(3)
    raw_data = create_data(200, linear_func_y)
    print(raw_data)

    min_lim = 0
    max_lim = 100
    need_normalize = True
    if (need_normalize):
        max_lim = 1
        data = normalize(raw_data)
    else:
        data = raw_data

    create_plot(data)
