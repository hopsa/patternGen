import numpy as np
from PIL import Image
from numpy.matrixlib.defmatrix import matrix


def cart2pol(param, param1):
    (r, theta) = (np.sqrt(param ** 2 + param1 ** 2), np.arctan2(param1, param))
    return r, theta


def gen_circles(polars_c: np.ndarray, feature:int, ts:int) -> np.ndarray:
    shape = (polars_c.shape[0], polars_c.shape[1])
    circles_pixels = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (polars_c[i, j][0] + ts) % feature < feature / 2:
                circles_pixels[i, j] = 255
    return circles_pixels


def gen_spirals(polars_s: np.ndarray, feature:int, ts:float, num_strands:int=1) ->np.ndarray:
    assert (0<=ts<1)
    shape=(polars_s.shape[0], polars_s.shape[1])
    spiral_pixels = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pol=polars_s[i,j]
            sval = pol[0]
            sval += (pol[1]*num_strands/(2 * np.pi))*feature
            sval += ts*feature
            sval = sval%feature/feature
            spiral_pixels[i, j] = sval * 255
    return spiral_pixels

def getPolars(shape: np.shape) -> np.ndarray:
    polars = np.zeros(shape + (2,))
    midpoint = (shape[0] / 2, shape[1] / 2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            polars[i, j,] = cart2pol(i - midpoint[0], j - midpoint[1])
    return polars

def func_to_gif(func, shape:np.shape, steps: int):
    polars = getPolars((shape[0]*3, shape[1]*3))
    n_steps = 16
    image_array = []
    for t in range(0, n_steps):
        pixels = func(polars=polars, feature=60, ts=t / n_steps)
        img = Image.fromarray(pixels)
        small_img = img.resize(shape, Image.Resampling.LANCZOS)
        small_img = small_img.convert('RGB')
        small_img.save(f"spiral{t}.bmp")
        image_array.append(small_img)
        # smallImg.show()
        print(t)
    combined = image_array[0]
    combined.save("spirals.gif", save_all=True, append_images=image_array[1:], duration=100, loop=0)

    return

def main():
    func= lambda polars, feature, ts:gen_spirals(polars, feature, ts, num_strands=3)
    func_to_gif(func=func, shape=(1000,1000), steps=16)

if __name__ == '__main__':
    main()