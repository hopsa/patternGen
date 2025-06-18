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


def gen_spirals(polars_s: np.ndarray, feature:int, ts:float) ->np.ndarray:
    assert (0<=ts<1)
    shape=(polars_s.shape[0], polars_s.shape[1])
    spiral_pixels = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pol=polars_s[i,j]
            sval = pol[0]
            sval += (pol[1]/(2 * np.pi))*feature
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

if __name__ == '__main__':
    # pixels = genCircles((1000, 1000))
    polars=getPolars((3000,3000))
    nSteps=16
    for t in range(0,nSteps):
        pixels = gen_spirals(polars, feature=60, ts=t / nSteps)

        img = Image.fromarray(pixels)
        smallImg = img.resize((1000, 1000), Image.Resampling.LANCZOS)
        smallImg = smallImg.convert('RGB')
        smallImg.save(f"spiral{t}.bmp")
        #smallImg.show()
        print(t)
