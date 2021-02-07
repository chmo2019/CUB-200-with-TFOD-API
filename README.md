# CUB-200-with-TFOD-API
![alt text](http://www.vision.caltech.edu/visipedia/collage.jpg) <br />
source: vision.caltech.edu

This repository is meant to be quick tutorial train and object detector
on the Caltech-UCSD Birds-200-2011 dataset. In addition to going over
the process of training an single shot detector on the dataset, the
repository includes tools to split the text-formatted dataset which is
different from the usual pascal-VOC-formatted dataset.

# Environment Setup

*NOTE: this was tested on Ubuntu 18.04 LTS with Python 3.6* <br />

*install protobuf with apt*

sudo apt-get install \ <br />
libprotobuf-dev \ <br />
libprotoc-dev \ <br />
protobuf-compiler

*run bash script to set up TFOD API*
./environment_setup.sh

*NOTE: this tutorial used ssd mobilenet v2 but you can use a different model from the model zoo* <br />

*create directory to store models*
mkdir model_dir && cd model_dir

*get ssd mobilenet v2 from Tensorflow 1 model zoo*
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

*untar pretrained ssd*
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

*navigate to root directory*
cd ..

# Data preparation

*NOTE: this tutorial is specifically for CUB 200 2011 wherein the data is provided as text files rather than xml files*

*navigate to data directory*
cd data

*make annotations directory to store data*
mkdir annotations

*get labeled data from Caltech CUB 200 2011 page*
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

*untar labeled data*
tar -xvf CUB_200_2011.tgz

*create COCO dataset csv from labeled data*
python3 create_train_test_split.py --path=$PWD/CUB_200_2011

*generate train tfrecord from csv*
python3 generate_tfrecord.py \ <br />
--csv_input_path=annotations/train.csv \ <br /> 
--output_path=annotations/train.record \ <br />
--image_dir=$PWD/CUB_200_2011/images \ <br />
--classes=$PWD/CUB_200_2011/classes.txt

*generate test tfrecord from csv*
python3 generate_tfrecord.py \ <br />
--csv_input_path=annotations/test.csv \ <br />
--output_path=annotations/test.record \ <br />
--image_dir=$PWD/CUB_200_2011/images \ <br />
--classes=$PWD/CUB_200_2011/classes.txt

*navigate to root directory*
cd ..

# Training

*create directory for fine-tuned model*
mkdir model_dir/CUB_200_model

*run Tensorflow training*
python3 model_main.py \  <br />
--model_dir model_dir/CUB_200_model \ <br />  
--pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--num_classes 200

*NOTE: if you want to monitor training on tensorboard go back to the root directory and run* <br />

tensorboard --logdir model_dir/CUB_200_model/logs 

# Exporting The Model

*create directory for tflite model*
mkdir model_dir/exported

*export fine-tuned checkpoint to frozen inference graph and prototxt*
python3 export_tflite_ssd_graph.py \ <br />
--pipeline_config_path $PWD/model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--trained_checkpoint_prefix $PWD/model_dir/CUB_200_model/model.ckpt-x \ <br />
--output_directory model_dir/exported

*convert frozen inference graph to tflite*
tflite_convert \
--graph_def_file=$PWD/model_dir/exported/tflite_graph.pb \ <br />
--output_file=$PWD/model_dir/exported/detect.tflite \ <br />
--input_shapes=1,300,300,3 \ <br />
--input_arrays=normalized_input_image_tensor \ <br />
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \ <br />
--inference_type=FLOAT \ <br />
--allow_custom_ops 

# Run Inference

*run inference on camera source*
python3 tflite_inference.py \ <br />
--pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--tflite_model_path model_dir/export/detect.tflite

*run inference on video source*
python3 tflite_inference.py \ <br />
--pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--tflite_model_path model_dir/export/detect.tflite \ <br />
--video_source path/to/video_source

# Licenses

https://github.com/datitran/raccoon_dataset/blob/master/LICENSE

https://github.com/tensorflow/models/blob/master/LICENSE

# Citations

Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001. 
