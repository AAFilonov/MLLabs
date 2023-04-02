from data_generation import *
import matplotlib.pyplot as plt


def linear_func_y(x):
    return -100 * x + 1



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
