import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from crystal import get_ions
from make_lattice import make_lattice

from PIL import Image, ImageEnhance
from PIL.JpegImagePlugin import JpegImageFile
# import pyexiv2

import random

import piexif
import pickle

"""
Simply run file after altering the following parameters to generate training data with metadata that contains the ion locations and states.
"""


##############################################

# Number of shells to step through (in range(shells))
shells = 5
# Elongations to step through (in range(elongations))
elongations = 2
# How many images per shell/elongation combo. Important because each image gets randomly oriented.
repetitions = 40

filename_prefix = "training_"
output_path = "data/train_data/"


laser_radius = 200 # bright-zone of the image - pixels outside this radius will be dimmed

# Probability that an ion is made dark
dark_prob = 0



##############################################



def normalise_to_pixels(arr, normalise, dim, border):
    return np.floor((arr + normalise)/(2*normalise) * (dim-2*border)) + border


def get_noise_vec(width:int, height:int, noisiness=100, brightness=1.0) -> np.array:
    im = Image.effect_noise((width, height), noisiness)
    enhancer = ImageEnhance.Brightness(im)
    img = enhancer.enhance(brightness)
    img = img.convert(mode="RGB")
    img = np.asarray(img)
    return img.copy()

# Sigmoid to adjust brightness of ions near the edge of the image
# r is the radius of brightness in pixels
def sigmoid(x, r=laser_radius, A=0.53):
    return 1/(1+ np.exp(-A*(x-r)))


def make_bright_spot(n=20, sigma=np.array([0.3, 0.3]), radial=None, image_width=1024, image_border=100):
    n = int(n)
    # Make nxn grid centred at 0 
    x, y = np.mgrid[-1.0:1.0:complex(0,n), -1.0:1.0:complex(0,n)]
    # Convert to an (n, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.0, 0.0])
    
    # Create radial smearing
    if radial is not None:
        # # Create normalised radial vector
        # radial_eigvec = radial/np.linalg.norm(radial)
        # radial_eigval = 0.05

        # # Create azimuthal vector normal to radial vector
        # azimuthal_eigvec = np.array([radial[1], -radial[0]])
        # azimuthal_eigvec = azimuthal_eigvec/np.linalg.norm(azimuthal_eigvec)

        # max_radius = (image_width-2*image_border)/2

        # # Puts azimuthal spread roughly in the range [radial_eigval, 0.3]
        # # spread follows r^2 dependency
        # azimuthal_eigval = (np.linalg.norm(radial)**2/ (max_radius**2)) *(0.2 - radial_eigval) + radial_eigval
        # if azimuthal_eigval < radial_eigval:
        #     azimuthal_eigval = radial_eigval


        # subtle variation
        radial_eigvec = radial/np.linalg.norm(radial)
        azimuthal_eigvec = np.array([radial[1], -radial[0]])
        azimuthal_eigvec = azimuthal_eigvec/np.linalg.norm(azimuthal_eigvec)

        noise = 0.08*np.random.random_sample((1, 2))[0] - 0.04
        radial_eigval = 0.1 + noise[0]
        azimuthal_eigval = 0.1 + noise[1]



        # Create covariance matrix by eigen recomposition
        D = np.array([[radial_eigval, 0], [0, azimuthal_eigval]])
        M = np.array([radial_eigvec, azimuthal_eigvec])
        covariance = np.matmul(np.matmul(M, D), np.linalg.inv(M))
    else:
        covariance = np.diag(sigma**2)

    # Make 3D Gaussian distribution
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    # Reshape back to a (30, 30) grid.
    z = z.reshape(x.shape)

    # Normalise to 0-255
    z =  np.floor(z/np.max(z) * 255)

    # Convert to RGB form
    # bright = np.zeros( (n, n, 3) )
    spot = Image.fromarray(z)
    spot = spot.convert(mode="RGB")
    spot = np.asarray(spot)
    return spot


def make_metadata(ion_locations, states):
    tags = {'ion_location' : ion_locations,
            'state' : states}

    data = pickle.dumps(tags)
    exif_ifd = {piexif.ExifIFD.MakerNote: data}

    exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
             "thumbnail": None, "GPS": {}}

    exif_dat = piexif.dump(exif_dict)

    return exif_dat


def get_labels(image:JpegImageFile):
    raw = image.getexif()[piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags


def make_training_image(x, y, normalise, filename, path="training_examples/", image_width=1024,
    border=100, bg_noisiness=100, bg_brightness=0.3,
    dark_prob=0.0, spot_radius=25):
    """
    image_width             - Set size of image
    border                  - Set border to scale ion lattice accordingly

    ### Randomness Parameters ###
    bg_noisiness            - int that determines how homogeneous the noise is (higher = more coarse)
    bg_brightness           - float between 0 and 1, low=dim, high=intense
    dark_prob               - probability that an ion will be made dark

    spot_radius = 50        - how big the region of the ions will be


    """

    # Create background of dimmed white noise
    image = get_noise_vec(image_width, image_width, noisiness=bg_noisiness, brightness=bg_brightness)

    # Translate the ion coordinates (float from -1 to 1) to pixel coordinates
    xy = np.array(list(zip(x, y)))
    xy_scaled = normalise_to_pixels(xy, normalise, 1024, 100)
    x_scaled, y_scaled = zip(*xy_scaled)

    centre = np.array([image_width//2, image_width//2])

    states = []

    for xpix, ypix in zip(x_scaled, y_scaled):
        xpix = int(xpix)
        ypix = int(ypix)

        # Get vector from centre to ion (used for radial smearing)
        radial = np.array([xpix, ypix]) - centre

        radius = np.linalg.norm(radial)

        dimmer_factor = int(np.floor(sigmoid(radius)*8 + 2))
        # print(dimmer_factor)
        # dimmer_factor = 2
        # Skip if
        if random.random()<dark_prob:
            dimmer_factor = 100
            states.append(False)
            continue
        else:
            states.append(True)

        x_lower = xpix-spot_radius
        x_upper = xpix+spot_radius
        y_lower = ypix-spot_radius
        y_upper = ypix+spot_radius

        # Removed radial spreading in favour of a subtle random spread
        radial = 2*np.random.random_sample((1, 2))[0] - 1
        spot = make_bright_spot(n=spot_radius*2, radial=radial, image_width=image_width, image_border=border)
        spot = spot // dimmer_factor

        image[x_lower:x_upper, y_lower:y_upper]  += spot

    img = Image.fromarray(image, "RGB")

    ion_locations = list(zip(x_scaled, y_scaled))

    data = make_metadata(ion_locations, states)

    # img.show()
    img.save(path+filename, exif=data)

    print("🎉 - Successfully created " + filename)

    # return img

image_id = 0

for shell in range(1, shells+1):
    for elon in range(elongations):
        print(elon, shell)
        for i in range(repetitions):
            if i < 10:
                dark_prob = 0
            elif i < 15:
                dark_prob = 0.25
            else:
                dark_prob = 0.5
            x, y = make_lattice(shell, elon=elon)
            
            xy = np.array(list(zip(x, y)))

            # Rotation matrix
            # Random angle
            theta = random.random()*2*np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))

            x, y = zip(*np.matmul(R, xy.T).T)
            make_training_image(x, y, (shells+elongations), f"{filename_prefix}{image_id}.jpeg", path=output_path, dark_prob=dark_prob)

            image_id += 1




    



# make_training_image(x, y, "testy.jpeg")
