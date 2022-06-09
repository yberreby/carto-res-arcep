import matplotlib.pyplot as plt
import numpy as np

def coordinate_transform(ref_coords, deriv_coords):
    """
    Returns the least-square affine transform corresponding to a mapping from
    `ref_coords` to `deriv_coords`.
    """

    A = np.zeros((deriv_coords.shape[0], 3))
    A[:, :2] = deriv_coords
    A[:, 2] = 1

    B = ref_coords

    x, res, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return x



def geoseries_to_np_xy(gs):
    n = len(gs)
    a = np.zeros((n, 2))
    a[:, 0] = gs.apply(lambda g: g.x)
    a[:, 1] = gs.apply(lambda g: g.y)
    return a


def plot_im(im):
    plt.figure()
    plt.imshow(im)


def to_shape(a, shape):
    z_, y_, x_ = shape
    z, y, x = a.shape
    z_pad = z_ - z
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        (
            (z_pad // 2, z_pad // 2 + z_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
        ),
        mode="constant",
    )


