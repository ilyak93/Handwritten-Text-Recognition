from PIL.Image import Image, new, AFFINE
import PIL.Image
import numpy as np
from PIL import Image, ImageShow

def add_border(img, width):
    nw = new('RGB', (img.width + 2*width, img.height), (255, 255, 255))
    nw.paste(img, (int(width/2), 0))
    return nw

def shear(img, shear):
    shear = img.transform(img.size, AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(255, 255, 255))
    return shear

def slant_word(img, shear_factor, dir=1):
    img = add_border(img, 200)
    img = shear(img, shear_factor*dir)
    img = np.asarray(img)
    positions = np.nonzero(img-255)

    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()

    img = img[top:bottom, left:right, :]
    return img

