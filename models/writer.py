import tensorflow.compat.v1 as tf
import io
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
import os
import piexif
import pickle
from object_detection.utils import dataset_util
from random import shuffle

# Function that reads the exif data of our images and outputs tags; ["state"] and ["ion_location"]
def get_labels(image:JpegImageFile):
    raw = image.getexif()[piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags

# Define bounding boxes for each ion and return their vertices and state inside
def extract_boxes(filename, box_radius=14): # specify 1/2 sidelength of bounding boxes to encode
    os.chdir("path_to/train_data") # specify path to images to train with 
    with Image.open(filename) as pil_img:
        tags = get_labels(pil_img)
    #boxes = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    states = []
    for i in range(len(tags["state"])):
        if tags["state"][i] == True: # Currently set so that only 'bright' ions are labelled, remove to include bright and dark
            states.append(1)
            (x, y) = tags["ion_location"][i]
            x = int(x)
            y = int(y)
            xmins.append(int(x-box_radius))
            ymins.append(int(y-box_radius))
            xmaxs.append(int(x+box_radius))
            ymaxs.append(int(y+box_radius))
            #boxes.append([int(y-box_radius), int(x-box_radius), int(y+box_radius), int(x+box_radius)])

    info = np.array([xmins, ymins, xmaxs, ymaxs, states]) 
    return info

# Encodes tfrecord information for 1 image
def create_tf_example(filename):

  info = extract_boxes(filename, box_radius=18)

  #open image
  with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  img = Image.open(encoded_jpg_io)

  label = filename.encode('utf-8') # filename encoded in bytes
  height = img.size[1] # Image height
  width = img.size[0]  # Image width
  image_format = b'jpeg' # b'jpeg' or b'png'
  

  xmins = info[1]/width # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = info[3]/width # List of normalized right x coordinates in bounding box
                        # (1 per box)
  ymins = info[0]/height # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = info[2]/height # List of normalized bottom y coordinates in bounding box
                         # (1 per box)
 
  classes = info[4] # List of integer class id of bounding box (1 per box)
  classes_text = [",".join(item) for item in classes.astype(str)]  # List of string class name of bounding box (1 per box)
  classes_text = [item.encode('utf-8') for item in classes_text]
  
  
  tf_example = tf.train.Example(features=tf.train.Features(feature={   # define and encode features for model to use 
      'image/height': dataset_util.int64_feature(height),              
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(label),
      'image/source_id': dataset_util.bytes_feature(label),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def encode_and_write(dataset, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for file in dataset:
            filename = os.fsdecode(file)
            tf_example = create_tf_example(filename)
            writer.write(tf_example.SerializeToString())
            
        writer.close()

def tfrecord_runner(path):
  
  directory = os.fsencode(r"C:/Users/domin/OneDrive/Desktop/Project/Quantum-Control-Lab-Project/ISPB-master/Penning_training_BD/data/train_data") # path to training images
  images = os.listdir(directory)
  shuffle(images)                                                                               #shuffles (randomises) dataset and splits into
  train_set, eval_set, test_set = np.split(images, [int(.8*len(images)), int(.9*len(images))])  #train, eval and test sets

  sets = [train_set, eval_set, test_set]

  train_set_outpath = os.path.join(path, "train.record")
  eval_set_outpath = os.path.join(path, "val.record")
  test_set_outpath = os.path.join(path, "test.record")


  encode_and_write(train_set, train_set_outpath)
  encode_and_write(eval_set, eval_set_outpath)
  encode_and_write(test_set, test_set_outpath)

if __name__ == '__main__':
  path = r"C:\Users\domin\OneDrive\Desktop\Project\TensorFlow\models\tf_record" # output path
  if not os.path.exists(path):
    os.makedirs(path)
  tfrecord_runner(path)
