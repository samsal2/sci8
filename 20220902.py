import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

csv = pd.read_csv("20220902_data1.csv")

print(csv)

xi_pfr_cstr = 0.62
xi_cstr_pfr = 0.5


default_x = csv["X"].to_numpy()
default_minvr = csv["minvr"].to_numpy()


def create_pfr_data_points(xi, x, x_list, minvr_list):
    assert xi < x
    assert len(x_list) == len(minvr_list)

    ii, i = np.searchsorted(x_list, [xi, x])

    new_x_list = x_list[ii:i]
    new_minvr_list = minvr_list[ii:i]

    # no exact match for xi was found, just replace the lower bound
    if new_x_list[0] != xi:
        new_x_list[0] = xi
        new_minvr_list[0] = np.interp(xi, x_list, minvr_list)

    # no exact match for x was found, just replace the upper bound
    if new_x_list[-1] != x:
        new_x_list[-1] = x
        new_minvr_list[-1] = np.interp(x, x_list, minvr_list)

    return np.array(new_x_list), np.array(new_minvr_list)


def draw_pfr(xi, x, *args, **kwargs):
    x_list = default_x
    if "x_list" in kwargs:
        x_list = kwargs["x_list"]

    minvr_list = default_minvr
    if "minvr_list" in kwargs:
        minvr_list = kwargs["minvr_list"]

    assert len(x_list) == len(minvr_list)

    x, minvr = create_pfr_data_points(xi, x, x_list, minvr_list)
    plt.fill_between(x, minvr, *args, **kwargs)


def create_cstr_data_points(xi, x, x_list, minvr_list):
    assert xi < x
    assert len(x_list) == len(minvr_list)

    minvr = np.interp(x, x_list, minvr_list)
    return np.array([xi, x]), np.array([minvr, minvr])


def draw_cstr(xi, x, *args, **kwargs):
    x_list = default_x
    if "x_list" in kwargs:
        x_list = kwargs["x_list"]

    minvr_list = default_minvr
    if "minvr_list" in kwargs:
        minvr_list = kwargs["minvr_list"]

    assert len(x_list) == len(minvr_list)

    x, minvr = create_cstr_data_points(xi, x, x_list, minvr_list)
    plt.fill_between(x, minvr, *args, **kwargs)


def draw_cstr_pfr(x0, x1, x, *args, **kwargs):
    x_list = default_x
    if "x_list" in kwargs:
        x_list = kwargs["x_list"]

    minvr_list = default_minvr
    if "minvr_list" in kwargs:
        minvr_list = kwargs["minvr_list"]

    cstr_x, cstr_minvr = create_cstr_data_points(x0, x1, x_list, minvr_list)
    pfr_x, pfr_minvr = create_pfr_data_points(x1, x, x_list, minvr_list)

    px = np.concatenate((cstr_x, pfr_x))
    pminvr = np.concatenate((cstr_minvr, pfr_minvr))

    plt.fill_between(px, pminvr, *args, **kwargs)


def draw_pfr_cstr(x0, x1, x, *args, **kwargs):
    x_list = default_x
    if "x_list" in kwargs:
        x_list = kwargs["x_list"]

    minvr_list = default_minvr
    if "minvr_list" in kwargs:
        minvr_list = kwargs["minvr_list"]

    pfr_x, pfr_minvr = create_pfr_data_points(x0, x1, x_list, minvr_list)
    cstr_x, cstr_minvr = create_cstr_data_points(x1, x, x_list, minvr_list)

    px = np.concatenate((pfr_x, cstr_x))
    pminvr = np.concatenate((pfr_minvr, cstr_minvr))

    plt.fill_between(px, pminvr, *args, **kwargs)


def draw_levenspiel(**kwargs):
    x_list = default_x
    if "x_list" in kwargs:
        x_list = kwargs["x_list"]

    minvr_list = default_minvr
    if "minvr_list" in kwargs:
        minvr_list = kwargs["minvr_list"]

    assert len(x_list) == len(minvr_list)

    plt.plot(x_list, minvr_list, **kwargs)


def main():
    # draw_pfr(0, xi_cstr_pfr, label="$pfr -> cstr$",
    #          facecolor="none", hatch="O", edgecolor="b")
    # draw_cstr(xi_cstr_pfr, default_x[-1],
    #           facecolor="none", hatch="O", edgecolor="b")

    draw_pfr_cstr(0, xi_pfr_cstr, default_x[-1], label="$pfr \\rightarrow cstr$",
                  facecolor="none", hatch="O", edgecolor="b")

    draw_cstr_pfr(0, xi_cstr_pfr, default_x[-1], label="$cstr \\rightarrow pfr$",
                  facecolor="none", hatch="o", edgecolor="r")

    # draw_cstr(0, xi_pfr_cstr, label="$cstr -> pfr$",
    #           facecolor="none", hatch="o", edgecolor="r")
    # draw_pfr(xi_pfr_cstr, default_x[-1],
    #          facecolor="none", hatch="o", edgecolor="r")

    draw_pfr(0, default_x[-1], label="$pfr$",
             facecolor="none", hatch=".", edgecolor="g")

    draw_levenspiel(color="k")

    plt.xticks(np.arange(0, 1, 0.1))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
