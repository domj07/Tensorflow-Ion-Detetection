## TensorFlow Model Library
This project is based on the model library built and maintained by TensorFlow. It 
would be good to check for performance enhancements, bug fixes and new models. They provide a range of 
object detection models in the research section and each have their own advantages in terms of speed, accuracy and 
other performance metrics. Using the issues forum on this repository is also invaluable for solving bugs etc.

Find the most up to date repository here:
https://github.com/tensorflow/models

## Setup

- Clone this repository or alternatively the most up to date /models/research repository in TensorFlow 
(some code may need to be modified).

- Unzip the large_files folder from https://www.dropbox.com/s/9usb9e9qt5h2wic/large_files.zip?dl=0  into the models directory

- Follow the instructions found here https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
or use the tensorflow2_environment.yml file to create a virtual environment for tensorflow. Note: I found using
conda rather than pip to install as many packages as possible avoided issues with setup.

## Training and running the model

### Creating input data

- create artificial data using Billy's make_training script and vary parameters such as dimmness, lattice
shape and size as required. Other noise/ background scattering could be added at this step.

- Alternatively, use real lab data and label with a tool like 'LabelImg' such as detailed in this tutorial;
https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087

- Create tfrecord files using the writer.py script. This puts our labelled images into a form that can 
be handled by the model. In general, a training batch, evaluation batch and/or test batch should be generated
at this stage. Each should be a different split of the input images so that the model is not evaluated
on images it has been trained on. See https://www.tensorflow.org/tutorials/load_data/tfrecord for more info.

- Check tfrecord files with tfrecord_reader to ensure bboxes etc have been encoded correctly 
(when I was creating my tfrecords the x and y coordinates had been flipped for no apparent reason by the api).
example command:
```
python tfrecord_reader/tfviewer.py /tfrecord/train.record --labels-to-highlight='on'
```

### Configuring the model

- Key file that may need to modified is the pipline.config file. This file determines what model is used, the checkpoint
used to initialise the model weights, the batch size for training, learning rate of the model, directories for
tfrecords and the label map used.

	- The pipline.config file in the exported_model directory is the one I used to train the model.
	This is suitable for the 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8' model, 1 gpu and a CPU with ram around 16gb.
	
	- The label_map is found in the annotations folder and provides string labels for the detected 
	classes e.g 'bright_ion' this can be modified if more classes are included in the tfrecord such
	as dark ions

	- The batch size is set to 4 and the warmup steps, total steps and learning rate should also be modified
	if this is changed. The optimal values for these are not easy to determine and should be investigated for your specific
	model and batch size.

	- 'fine_tune_checkpoint' parameter should be modified to the directory of the checkpoints for the pre-trained
	model you want to use. Using the pretrained checkpoints in \models\exported_model\checkpoint will restore the weights
	to our pretrained model that is already effective at identifying artificial data and so should be a good starting point.
	If you want to train the model from scratch, the checkpoints in models\faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8\checkpoint 
	should be used.

	- The 'input_path' for both evaluation and training should also be adjusted to the location of respective
	tfrecord files.

- Other files that may need to be modified / are worth looking at are model_lib_v2 and main_tf2 in the research/object_detection section.
in main_tf2 you can adjust how often the model outputs a checkpoint of the trained model (which can then be evaluated)
with the 'checkpoint_every_n' flag.

### Training

- The model should now be ready to be trained. If running on the cluster a file similar to train_model.pbs should be submitted
to the queue I found it easiest to start by changing directory into models/.

- Example commands
	- This runs training for 200000 steps, samples every evaluation image, specifies the pipeline.config directory and 
	specifies that tensorboard data should be logged to our model_dir (output directory).
	```
	python research/object_detection/model_main_tf2.py 
		--model_dir="model_data/" 
		--num_train_steps=200000  
		--sample_1_of_n_eval_examples=1  
		--pipeline_config_path=exported_model/pipeline.config  
		--alsologtostder
	```
	- In a seperate terminal, this command will run evaluation of the model simultaneously. checkpoint_dir is 
	where our training script is storing the model outputs, generally the same as we specified for model_dir.
	```
	python /research/object_detection/model_main_tf2.py 
		--model_dir=model_data  
		--checkpoint_dir=model_data 
		--pipeline_config_path=pipeline.config 
		--alsologtostderr
	```
	- This will start tensorboard visualisation of the training and evaluation process. Note, I have included tensorboard data from the previous training so you know 
	what to expect/ what the metrics are for the existing model.
	```
	tensorboard --logdir=model_data
	```
	- To export our trained model from its latest checkpoint, use this command.
	```
	python research/object_detection/exporter_main_v2.py 
		--input_type image_tensor 
		--pipeline_config_path pipeline.config 
		--trained_checkpoint_dir model_data/ 
		--output_directory inference_graph  
		--use_side_inputs False
	```
## Testing and Inference

- To assess the accuracy and performance of the model it is best to look at the coco metrics from the 
evaluation of the model. You can find a summary of the metrics here: 
https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/.

- For a more visual assesment and to test the model on new images without annotating them, use the model_tester script.
This takes an input set of images and saves a results set with bounding boxes drawn, see 'test' in models/
	- The score threshold (minimum confidence for a bounding box to be drawn) can be adjusted by changing the min_score_thresh
	parameter.
	See the eval_util script in research/object_detection/ to see other arguments that can be adjusted.


## CapsNet

I used naturomics' CapsLayer repository as it has good documentation and it would be very simple to modify our writer.py script 
to pass images of single ions to capsnet to classify. Artificial data of single ions or cropped images from the data set would
need to be labelled and passed to the writer. The reader file in naturomics may then need to be slightly modified to match.
see: https://github.com/naturomics/CapsLayer/blob/master/docs/tutorials.md.

The other repositories were sent by Donglin and seem to be updated more regularly and are worth investigating further.

Naturomics: https://github.com/naturomics/CapsLayer

Dynamic Routing between Capsules: https://github.com/Sarasra/models/tree/master/research/capsules

Matrix Capsules with EM Routing: https://github.com/IBM/matrix-capsules-with-em-routing

Stacked Capsule Autoencoders: https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders

