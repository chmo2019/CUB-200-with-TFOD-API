# CUB-200-with-TFOD-API

This repository is meant to be quick tutorial train and object detector
on the Caltech-UCSD Birds-200-2011 dataset. In addition to going over
the process of training an single shot detector on the dataset, the
repository includes tools to split the text-formatted dataset which is
different from the usual pascal-VOC-formatted dataset.

# Environment Setup

./environment_setup.sh

*NOTE: this tutorial used ssd mobilenet v2 but you can use a different model from the model zoo*

mkdir model_dir && cd model_dir

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz' from tensorflow 1 model zoo

tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

cd ..

# Data preparation

*NOTE: this tutorial is specifically for CUB 200 2011 wherein the data is provided as text files rather than xml files*

cd data

mkdir annotations

wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

tar -xvf CUB_200_2011.tgz

python3 create_train_test_split.py --path=$PWD/CUB_200_2011

python3 generate_tfrecord.py \ <br />
--csv_input_path=annotations/train.csv \ <br /> 
--output_path=annotations/train.record \ <br />
--image_dir=$PWD/CUB_200_2011/images \ <br />
--classes=$PWD/CUB_200_2011/classes.txt

python3 generate_tfrecord.py \ <br />
--csv_input_path=annotations/test.csv \ <br />
--output_path=annotations/test.record \ <br />
--image_dir=$PWD/CUB_200_2011/images \ <br />
--classes=$PWD/CUB_200_2011/classes.txt

cd ..

# Training

mkdir model_dir/CUB_200_model

python3 model_main.py \  <br />
--model_dir model_dir/CUB_200_model \ <br />  
--pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--num_classes 200

*NOTE: if you want to monitor training on tensorboard go back to the root directory and run* <br />

tensorboard --logdir model_dir/CUB_200_model/logs 

# Exporting The Model

mkdir model_dir/exported

python3 export_tflite_ssd_graph.py \ <br />
--pipeline_config_path $PWD/model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--trained_checkpoint_prefix $PWD/model_dir/CUB_200_model/model.ckpt-x \ <br />
--output_directory model_dir/exported

tflite_convert \
--graph_def_file=$PWD/model_dir/exported/tflite_graph.pb \ <br />
--output_file=$PWD/model_dir/exported/detect.tflite \ <br />
--input_shapes=1,300,300,3 \ <br />
--input_arrays=normalized_input_image_tensor \ <br />
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \ <br />
--inference_type=FLOAT --allow_custom_ops 

# Run Inference

python3 tflite_inference.py \ <br />
--pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \ <br />
--tflite_model_path model_dir/export/detect.tflite
# Licenses

https://github.com/datitran/raccoon_dataset/blob/master/LICENSE

https://github.com/tensorflow/models/blob/master/LICENSE

# Citations

Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001. 
