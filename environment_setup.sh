#!/bin/bash

DIR = "models/research";

pip install -r requirements.txt

if python -c "import object_detection" &> /dev/null; then
	printf "\n[SETUP] tensorflow object detection installed\n"
else
	if [ ! -d "$DIR" ]; then
		printf "\n[SETUP] tensorflow models directory doesn't exist\n"
		git clone https://github.com/tensorflow/models
		cd models/research
		protoc object_detection/protos/*.proto --python_out=.

		if  python -c "import tensorflow as tf; float(tf.__version__[0]) < 2" &> /dev/null; then
			printf "\n[SETUP] installing object detection for tensorflow 1\n"
			cp object_detection/packages/tf1/setup.py .
		else
			printf "\n[SETUP] install object detection for tensorflow 2\n"
			cp object_detection/packages/tf2/setup.py .
		fi

		python -m pip install .

		if python -c "import object_detection" &> /dev/null; then
			printf "\n[SETUP] tensorflow object detection installed successfully\n"
		else
			printf "\n[SETUP] tensorflow object detection failed to install\n"
		fi
    fi
    	printf "\n[SETUP] now performing unit tests, make sure all tests run 'OK'\n"

		if  python -c "import tensorflow as tf; float(tf.__version__[0]) < 2" &> /dev/null; then
			python object_detection/builders/model_builder_tf1_test.py
		else
			python object_detection/builders/model_builder_tf2_test.py
		fi

fi

printf "\n[SETUP] exiting...\n"
