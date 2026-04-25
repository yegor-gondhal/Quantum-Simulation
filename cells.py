import cupy as cp
import numpy as np

xp = cp

two = xp.arange(0, 15)
three = xp.arange(0, 9)
two = xp.power(2, two)
three = xp.power(3, three)

TWO, THREE = xp.meshgrid(two, three)
width_values = TWO*THREE
width_values = width_values.ravel()
WIDTH1, WIDTH2 = xp.meshgrid(width_values, width_values)
ratio = WIDTH1/WIDTH2
mask = ((0.56 < ratio) & (ratio < 0.58))
valid_idxs = xp.argwhere(mask)
values = width_values[valid_idxs]
sorted_idxs = values[:, 0].argsort()
sorted_values = values[sorted_idxs]
print(sorted_values)