import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageChops

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_height, im_width) = image.size
  if is_greyscale(image)==True:    # convert to RGB if greyscale
      image = image.convert("RGB")

  array = np.array(image.getdata()).reshape(
    (im_width, im_height,3)).astype(np.uint8)
  return array

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return False
    return True

# Load the Label Map
category_index = {
    1: {'id': 1, 'name': 'on'},
}

# load the model from our exported model directory
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('exported_model\saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')


elapsed = []

directory = os.fsencode(r"C:/Users/domin/OneDrive/Desktop/Project/TensorFlow/models/test/test_images") #path to test images
os.chdir(r"C:/Users/domin/OneDrive/Desktop/Project/TensorFlow/models/test") # path contain test image and results directories
images = os.listdir(directory)

for i in range(len(images)):
  file = images[i]
  filename = os.fsdecode(file)
  path = os.path.join('test_images', filename)
  image_np = load_image_into_numpy_array(path)
  input_tensor = np.expand_dims(image_np, 0)
  start_time = time.time()
  detections = detect_fn(input_tensor)
  end_time = time.time()
  elapsed.append(end_time - start_time)

  plt.rcParams['figure.figsize'] = [42, 21]
  label_id_offset = 1
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.7,
        groundtruth_box_visualization_color='white',
        skip_scores=True,
        skip_labels=True,
        agnostic_mode=False)
  plt.figure()
  plt.imshow(image_np_with_detections)
  name = 'results/' + 'result_' + filename
  plt.savefig(name)
  print(name,'is','Done')
  

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

