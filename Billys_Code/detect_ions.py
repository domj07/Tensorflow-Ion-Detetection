from mrcnn.utils import Dataset, extract_bboxes

from mrcnn.visualize import display_instances

from mrcnn.config import Config

from mrcnn.model import MaskRCNN

import os
os.chdir("/Users/billydodds/Documents/Uni/PHYS3888/ISPB")
# from load_data.QAnalyse import Analyser
from PIL import Image, ImageEnhance, ImageFilter
from PIL.JpegImagePlugin import JpegImageFile

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import re
import sys

import piexif
import pickle
import numpy as np

from mrcnn.config import Config
from mrcnn.utils import Dataset, compute_ap
from mrcnn.model import MaskRCNN, load_image_gt, mold_image

import cv2

mode = sys.argv[1]


# https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

"""
INSTRUCTIONS
Use Python 3.6.10 and install the dependencies (following the link above)

MODES
- python detect_ions.py train 
This is used for training the model. Provide the following directories:
    1. Path to starting weights 'path_to_weight'
    2. Path to write output weights to 'path_to_model_files'
    3. Path to training data 'path_to_images'

- python detect_ions.py predict 
This is used for displaying the predictions of the model on test data. Provide the following parameters:
    1. Path to model weights 'path_to_weight'
    2. Path to test data 'path_to_images'
    3. Output path to write images to 'output_path'
    4. Whether the data is unseen (unlabelled): 'unseen'

- python detect_ions.py evaluate 
This plots the coverage percentage of on ions vs. number of ions for the training data. Directories:
    1. Path to model weights 'path_to_weight'
    2. Path to training data 'path_to_images'
    3. Output path to write plots to 'output_path'
"""


##### Paths #####

path_to_weight = "/Users/billydodds/Documents/Uni/PHYS3888/ISPB/Penning_training_BD/model/best_weight/mask_rcnn_penning_cfg_0004.h5"
path_to_model_files = '/Users/billydodds/Documents/Uni/PHYS3888/ISPB/Penning_training_BD/model/'
output_path = "/Users/billydodds/Documents/Uni/PHYS3888/ISPB/test_results_real/"
path_to_images = "Penning_training_BD/data/real_data/"
unseen = True




def get_labels(image:JpegImageFile):
    raw = image.getexif()[piexif.ExifIFD.MakerNote]
    tags = pickle.loads(raw)
    return tags


class PenningDataset(Dataset):
	# load the dataset definitions
    def load_dataset(self, images_dir, is_train=True, load_all=False):
        # define one class
        self.add_class("dataset", 1, "on")
        self.add_class("dataset", 0, "off")
        # find all images
        for filename in sorted(os.listdir(images_dir), key=lambda x: int(x.split("_")[1].split(".")[0])):
            # extract image id
            # nums = re.findall("[0-9]", filename)
            # image_id = int("".join(nums))
            image_id = int(filename.split("_")[1].split(".")[0])

            if not load_all:
                stratify = image_id%20
                if is_train and (stratify in [0, 9, 12, 18]):
                    continue
                if not is_train and (stratify not in [0, 9, 12, 18]):
                    continue

            print(image_id)

            img_path = images_dir + filename
            # ann_path = images_dir + filename
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path)
    
    def extract_boxes(self, filename, box_radius=13):
        with Image.open(filename) as pil_img:
            tags = get_labels(pil_img)
        boxes = []
        for (x, y) in tags["ion_location"]:
            x = int(x)
            y = int(y)
            boxes.append([int(y-box_radius), int(x-box_radius), int(y+box_radius), int(x+box_radius)])
        return boxes, tags["state"]

    def get_actuals(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['path']
        # load XML
        boxes, states = self.extract_boxes(path)
        return boxes, states
 
	# load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['path']
        # load XML
        boxes, states = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([1024, 1024, len(boxes)], dtype='uint8')
        # create masks
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            state = states[i]
            class_ids.append(int(state))
        return masks, np.asarray(class_ids, dtype='int32')
 
	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']



# define a configuration for the model
class PenningConfig(Config):
    # Give the configuration a recognizable name
    NAME = "penning_cfg"
    # Number of classes (on, off and background)
    NUM_CLASSES = 3
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 160



class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "penning_cfg"
    # number of classes (on, off and background)
    NUM_CLASSES = 3
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def is_inside(point, box):
    y1, x1, y2, x2 = box
    x, y = point
    if x < x2 and x > x1 and y < y2 and y > y1:
        return True
    else: 
        return False


def plot_actual_vs_predicted(dataset, model, cfg, output_path, n_images=25):
    # load image and mask
    for i in range(n_images): #range(0, n_images*4, 5):
        # load the image and mask
        image = dataset.load_image(i)
        boxes, states = dataset.get_actuals(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        plt.figure(i, figsize=(10,5))
        plt.subplot(1, 2, 1)
        actual_im = image.copy()
        # plot raw pixel data
        for (y1, x1, y2, x2), state in zip(boxes, states):
            # calculate width and height of the box
            # cv2.rectangle(actual_im, (y1, x1), (y2, x2), color = (0, 0, 255), thickness=1)
            cv2.rectangle(actual_im, (y1, x1), (y2, x2), color = (0, 255, 0) if state else (255, 0, 0), thickness=2)

            # cv2.putText(actual_im, "On" if state else "Off", (y1-5, x1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if state else (255, 0, 0), 1)

        plt.imshow(actual_im)
        plt.title('Actual')

        # Add legend
        # plt.plot([0], [0], color=(0, 1, 0), label="On")
        # plt.plot([0], [0], color=(1, 0, 0), label="Off")


        # plt.scatter([0], [0], color=(0, 0, 0), label=f"On: {sum(states)}")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"Off: {sum(np.logical_not(np.array(states)))}")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"Total: {len(states)}")


        # plt.legend()    

        # get the context for dårawing boxes
        plt.subplot(1, 2, 2)
        # plot raw pixel data
        # plot each box
        for (x1, y1, x2, y2), state in zip(yhat['rois'], yhat['class_ids']):
            # calculate width and height of the box
            cv2.rectangle(image, (y1, x1), (y2, x2), color = (0, 255, 0) if state else (255, 0, 0), thickness=2)

            # cv2.putText(image, "On" if state else "Off", (y1-5, x1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if state else (255, 0, 0), 1)

        correct_bright_ions = 0
        correct_dark_ions = 0
        false_bright_ions = 0
        false_dark_ions = 0
        missed_bright_ions = 0
        missed_dark_ions = 0


        actuals = list(zip(boxes.copy(), states.copy()))

        for (x1, y1, x2, y2), pred_state in zip(yhat['rois'], yhat['class_ids']):
            centre = ((x2+x1)//2, (y2+y1)//2)
            for box, act_state in actuals:
                # print(centre, box)
                # if the predicted centre is inside an actual location and their classes match, then count it
                if pred_state == act_state and is_inside(centre, box):
                    # print(centre, box)
                    actuals.remove((box, act_state))
                    if act_state:
                        correct_bright_ions += 1
                    else:
                        correct_dark_ions += 1
                    break
            else:
                if pred_state:
                    false_bright_ions += 1
                else:
                    false_dark_ions += 1


        for box, act_state in actuals:
            if act_state:
                missed_bright_ions += 1
            else:
                missed_dark_ions += 1


        plt.imshow(image)
        plt.title('Predicted')
        # plt.plot([0], [0], color=(0, 1, 0), label="On")
        # plt.plot([0], [0], color=(1, 0, 0), label="Off")

        # plt.scatter([0], [0], color=(0, 0, 0), label=f"On Coverage: {round(correct_bright_ions/sum(states) * 100, 2)}%   ({correct_bright_ions}/{sum(states)})")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"Off Coverage: {round(correct_dark_ions/sum(np.logical_not(np.array(states)))*100, 2)}%   ({correct_dark_ions}/{sum(np.logical_not(np.array(states)))})")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"False On: {false_bright_ions}")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"False Off: {false_dark_ions}")
        # plt.scatter([0], [0], color=(0, 0, 0), label=f"Total Preds:{len(yhat['rois'])}")
        # plt.legend() 

        plt.savefig(output_path + f"test_results_{i}")

        
                

    # show the figure
    # plt.show()



def my_evaluation(dataset, model, cfg, n):
    # load image and mask
    on_coverages = []
    num_ions = []
    for i in range(n):
        # load the image and mask
        image = dataset.load_image(i)
        boxes, states = dataset.get_actuals(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot

        correct_bright_ions = 0
        correct_dark_ions = 0
        false_bright_ions = 0
        false_dark_ions = 0
        missed_bright_ions = 0
        missed_dark_ions = 0


        actuals = list(zip(boxes.copy(), states.copy()))

        for (x1, y1, x2, y2), pred_state in zip(yhat['rois'], yhat['class_ids']):
            centre = ((x2+x1)//2, (y2+y1)//2)
            for box, act_state in actuals:

                # print(centre, box)
                # if the predicted centre is inside an actual location and their classes match, then count it
                if pred_state == act_state and is_inside(centre, box):
                    # print(centre, box)
                    actuals.remove((box, act_state))
                    if act_state:
                        correct_bright_ions += 1
                    else:
                        correct_dark_ions += 1
                    break
            else:
                if pred_state:
                    false_bright_ions += 1
                else:
                    false_dark_ions += 1


        for box, act_state in actuals:
            if act_state:
                missed_bright_ions += 1
            else:
                missed_dark_ions += 1

        num_ions.append(len(states))
        on_coverages.append(correct_bright_ions/sum(states))

    return on_coverages, num_ions



def predicted_unseen(dataset, model, cfg, n_images=25, output_path=output_path):
    # load image and mask
    for i in range(n_images): #range(0, n_images*4, 5):
        # load the image and mask
        image = dataset.load_image(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        plt.figure(i, figsize=(5,5))
        # plot raw pixel data
        # plot each box
        for (x1, y1, x2, y2), state in zip(yhat['rois'], yhat['class_ids']):
            # calculate width and height of the box
            cv2.rectangle(image, (y1, x1), (y2, x2), color = (0, 255, 0) if state else (255, 0, 0), thickness=2)

            # cv2.putText(image, "On" if state else "Off", (y1-5, x1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if state else (255, 0, 0), 1)

        plt.imshow(image)
        plt.title('Predicted')
        # plt.plot([0], [0], color=(0, 1, 0), label="On")
        # plt.plot([0], [0], color=(1, 0, 0), label="Off")
        # plt.legend()

        plt.savefig(output_path+f"real_results_{i}")





if mode.lower() == "train":
    train_set = PenningDataset()
    train_set.load_dataset(path_to_images)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))


    test_set = PenningDataset()
    test_set.load_dataset(path_to_images, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))



    # image_id = -1
    # # load the image
    # image = train_set.load_image(image_id)
    # # load the masks and the class ids
    # mask, class_ids = train_set.load_mask(image_id)
    # # extract bounding boxes from the masks
    # bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    # display_instances(image, bbox, mask, class_ids, train_set.class_names)
    
    # prepare config
    config = PenningConfig()


    model = MaskRCNN(mode='training', model_dir=path_to_model_files, config=config)

    model.load_weights(path_to_weight, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')



elif mode.lower() == "predict":
    # path = "Penning_training_BD/artificial_data"

    # # load the train dataset
    # train_set = PenningDataset()
    # train_set.load_dataset(path, load_all=True)
    # train_set.prepare()
    # print('Train: %d' % len(train_set.image_ids))

    # load the test dataset
    test_set = PenningDataset()
    test_set.load_dataset(path_to_images, load_all=True)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir=path_to_model_files, config=cfg)
    # load model weights
    model.load_weights(path_to_weight, by_name=True)
    # evaluate model on training dataset
    # train_mAP = evaluate_model(train_set, model, cfg)
    # print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    # test_mAP = evaluate_model(test_set, model, cfg)
    # print("Test mAP: %.3f" % test_mAP)

    n_images = len(list(os.listdir(path_to_images)))


    if unseen:
        predicted_unseen(test_set, model, cfg, n_images=n_images)
    else:
        plot_actual_vs_predicted(test_set, model, cfg, output_path,  n_images=n_images)


elif mode.lower() == "evaluate":

    main_path = "Penning_training_BD/data/"

    test_sets = [main_path + folder + "/" for folder in ["test_data", "threshold_test_data"]] 

    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir=path_to_model_files, config=cfg)
    # load model weights
    model.load_weights(path_to_weight, by_name=True)

    for set_no, test_path in enumerate(test_sets):
        print(set_no)
        # load the test dataset
        test_set = PenningDataset() 
        test_set.load_dataset(test_path, load_all=True)
        test_set.prepare()
        print('Test: %d' % len(test_set.image_ids))

        n_images = len(list(os.listdir(path_to_images)))


        coverages, x_axis = my_evaluation(test_set, model, cfg, n_images) # 0.9568915038301692 

        # Delete so that we have one no dark, one quarter dark and one half dark
        i = 96
        while i >= 0:
            del coverages[i]
            del x_axis[i]
            i-=4

        coverages = np.array(coverages)
        x_axis = np.array(x_axis)

        plt.figure(set_no, figsize=(20,10))

        # Split into all bright, quarter dark and half dark
        print(len(coverages))
        coverages =  np.reshape(coverages, (3, len(coverages)//3), order='F')
        x_axis = np.reshape(x_axis, (3, len(x_axis)//3), order='F')[0]
        bright = coverages[0]
        quarter_dark = coverages[1]
        half_dark = coverages[2]

        order = np.argsort(x_axis)
        x_axis = x_axis[order]

        plt.plot(x_axis, coverages[0][order], label="0% Dark")
        plt.plot(x_axis, coverages[1][order], label="25% Dark")
        plt.plot(x_axis, coverages[2][order], label="50% Dark")
        plt.legend()

        plt.title("Coverage percentage vs. ions for MRCNN")
        plt.xlabel("number of ions")
        plt.ylabel("Coverage percentage")

        plt.savefig(output_path + f"results_{set_no}")

        # plt.show()