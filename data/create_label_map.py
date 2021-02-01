import csv

labels = {'id': [], 'name': []}

f_info = open("CUB_200_2011/classes.txt")
f_labels = open("label_map.txt", "w")

tsv = csv.reader(f_info, delimiter=" ")

for row in tsv:
	labels['id'].append(row[0])
	labels['name'].append(row[1])

f_info.close()

for i in range(len(labels['id'])):
	f_labels.write("item {{\n\tid: {}\n\tname: '{}'\n}}\n\n".format(labels['id'][i], labels['name'][i]))
#	print(labels['id'][i])
f_labels.close()
