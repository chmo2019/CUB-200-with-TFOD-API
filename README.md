# CUB-200-with-TFOD-API

sources: https://github.com/datitran/raccoon_dataset

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
python3 generate_tfrecord.py --csv_input_path=annotations/train.csv --output_path=annotations/train.record --image_dir=$PWD/CUB_200_2011/images --classes=$PWD/CUB_200_2011/classes.txt
python3 generate_tfrecord.py --csv_input_path=annotations/test.csv --output_path=annotations/test.record --image_dir=$PWD/CUB_200_2011/images --classes=$PWD/CUB_200_2011/classes.txt
cd ..

# Training

python3 model_main.py --model_dir model_dir --pipeline_config_path model_dir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config



