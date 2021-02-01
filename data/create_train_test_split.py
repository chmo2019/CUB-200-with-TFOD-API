# import necessary libraries
import csv
from PIL import Image
import argparse

# create argument parser with PATH argument 
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
	help='''PATH to CUB_200_2011 folder i.e. folder with CUB 200 csv files 
	(make sure to include full path name for so other scripts can find the data file path(s))''')
args = ap.parse_args()

def create_train_test_split(PATH):
 
 	# open CUB 200 .txt files 
	images = open(PATH + "/images.txt", "r")
	image_class_labels = open(PATH + "/image_class_labels.txt", "r")
	bounding_boxes = open(PATH + "/bounding_boxes.txt", "r")
	split = open(PATH + "/train_test_split.txt", "r")
	classes = open(PATH + "/classes.txt", "r")

	# create csv readers for each .txt file
	tsv = csv.reader(split, delimiter=" ")
	tsv_images = csv.reader(images, delimiter=" ")
	tsv_class_labels = csv.reader(image_class_labels, delimiter=" ")
	tsv_bbox = csv.reader(bounding_boxes, delimiter=" ")
	tsv_classes = csv.reader(classes, delimiter=" ")

	# create dictionary to store data
	train_test = {"0":
			{"filename": [],
			"id": [],
			"width": [],
			"height": [],
			"class": [],
			"x" : [],
			"y": [],
			"img_w": [],
			"img_h": []},
			"1":
			{"filename": [],
			"id": [],
			"width": [],
			"height": [],
			"class": [],
			"x" : [],
			"y": [],
			"img_w": [],
			"img_h": []}} # '0' for test '1' for train

	# write id into dictionary for create train test split
	for row in tsv:
		train_test["{}".format(row[1])]["id"].append(row[0])

	split.close()

	classes_list = {}

	# append class names to dictionary
	for row in tsv_classes:
		classes_list["{}".format(row[0])] = row[1]

	classes.close()

	i = 0
	j = 0

	# add image sizes, labels, and bounding box coordinates to dictionary
	for (image, label, bbox) in zip(tsv_images, tsv_class_labels, tsv_bbox):
		if train_test["0"]["id"][i] == image[0]:
			train_test["0"]["filename"].append(PATH + "/images/" + image[1])
			im = Image.open(PATH + "/images/"+ image[1])
			train_test["0"]["img_w"].append(im.size[0])
			train_test["0"]["img_h"].append(im.size[1])
			train_test["0"]["class"].append(classes_list["{}".format(label[1])])
			train_test["0"]["x"].append(bbox[1])
			train_test["0"]["y"].append(bbox[2])
			train_test["0"]["width"].append(bbox[3])
			train_test["0"]["height"].append(bbox[4])
			i += 1
		else:
			train_test["1"]["filename"].append(PATH + "/images/" + image[1])
			im = Image.open(PATH + "/images/"+ image[1])
			train_test["1"]["img_w"].append(im.size[0])
			train_test["1"]["img_h"].append(im.size[1])
			train_test["1"]["class"].append(classes_list["{}".format(label[1])])
			train_test["1"]["x"].append(bbox[1])
			train_test["1"]["y"].append(bbox[2])
			train_test["1"]["width"].append(bbox[3])
			train_test["1"]["height"].append(bbox[4])
			j += 1
	
	images.close()
	image_class_labels.close()
	bounding_boxes.close()

	# open csv files for coco-formatted data
	f_train = open("./annotations/train.csv", "w")
	f_test = open("./annotations/test.csv", "w")

	# create coco csv header
	f_test.write("{},{},{},{},{},{},{},{}\n".format("filename","width","height","class",
						"xmin", "ymin", "xmax", "ymax"))

	# write coco-formatted data into test split csv
	for k in range(len(train_test["0"]["filename"])):
		f_test.write("{},{},{},{},{},{},{},{}\n".format(train_test["0"]["filename"][k],
							train_test["0"]["img_w"][k],
							train_test["0"]["img_h"][k],
							train_test["0"]["class"][k],
							train_test["0"]["x"][k],
							train_test["0"]["y"][k],
							float(train_test["0"]["x"][k]) +
							float(train_test["0"]["width"][k]),
							float(train_test["0"]["y"][k]) +
							float(train_test["0"]["height"][k])))

	f_train.write("{},{},{},{},{},{},{},{}\n".format("filename","width","height","class",
						"xmin", "ymin", "xmax", "ymax"))

	# write coco-formatted data into train split csv
	for k in range(len(train_test["1"]["filename"])):
		f_train.write("{},{},{},{},{},{},{},{}\n".format(train_test["1"]["filename"][k],
							train_test["1"]["img_w"][k],
							train_test["1"]["img_h"][k],
							train_test["1"]["class"][k],
							train_test["1"]["x"][k],
							train_test["1"]["y"][k],
							float(train_test["1"]["x"][k]) +
							float(train_test["1"]["width"][k]),
							float(train_test["1"]["y"][k]) +
							float(train_test["1"]["height"][k])))

	f_test.close()
	f_train.close()

if __name__ == "__main__":
	# run with command line arguments
	create_train_test_split(args.path)