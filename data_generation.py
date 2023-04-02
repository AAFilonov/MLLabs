import random


class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def create_data(amount_of_points, func_y, interval_start_x=0, interval_end_x=100):
    array = []
    for i in range(0, amount_of_points):
        x = random.randint(interval_start_x, interval_end_x)
        array.append(DataPoint(x, func_y(x)))
    return array


def normalize(points):
    array = []
    max_x = 100
    min_x = 0
    max_y = 100
    min_y = 0
    for point in points:
        array.append(
            DataPoint(
                (point.x - min_x) / max_x,
                (point.y - min_y) / max_y,
            )
        )

    return points
