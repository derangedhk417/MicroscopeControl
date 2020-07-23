import matplotlib.pyplot as plt
import numpy             as np
import sys
import time
import code
import argparse
import os
import cv2
import json
import threading
import math

from Progress import ProgressBar

# Process the command line arguments supplied to the program.
def preprocess(args_specification):
	parser = argparse.ArgumentParser(description=args_specification['description'])

	types = {'str': str, 'int': int, 'float': float}

	for argument in args_specification['arguments']:
		spec = argument['spec']
		if 'type' in spec:
			spec['type'] = types[spec['type']]
		parser.add_argument(
			*argument['names'], 
			**spec
		)

	args = parser.parse_args()

	return args

def getFiles(directory, ext):
	entries = os.listdir(directory)

	results = []
	for entry in entries:
		path = os.path.join(directory, entry)
		
		extension = entry.split(".")[-1].lower()
		if extension == ext:
			results.append(path)

	return results

if __name__ == '__main__':
	# Load the arguments file. 
	with open("ScanSummarize.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# Load a list of json files from the directory.
	meta_files = getFiles(args.image_directory, 'json')

	pb = ProgressBar("Loading", 12, len(meta_files), update_every=5)

	surface_areas   = []
	contrast_means  = []
	contrast_values = [] 
	for idx, file in enumerate(meta_files):
		# Don't process _scan.json
		if os.path.split(file)[-1] == '_scan.json':
			continue

		# Load it into memory, parse it and extract statistics.
		with open(file, 'r') as f:
			data = json.loads(f.read())

		# Load the image with the same name.
		imgpath = ".".join(os.path.split(file)[-1].split('.')[:-1]) + '.png'
		imgpath = os.path.join(args.image_directory, imgpath)
		img     = cv2.imread(imgpath)

		for contour in data['contours']:
			c = np.array(contour)
			c[:, 0] = c[:, 0] * img.shape[1]
			c[:, 1] = c[:, 1] * img.shape[0]
			c = c.astype(np.int32)
			area = cv2.contourArea(c.reshape(-1, 1, 2))
			surface_areas.append(area)

		contrast = np.array(data['contrast_values'])
		contrast_means.append(contrast.mean())
		contrast_values.append(contrast)

		pb.update(idx + 1)

	pb.finish()

	plt.hist(np.log(surface_areas), bins=75)
	plt.xlabel("Surface Area in Square Pixels (log scale)")
	plt.ylabel("Count")
	plt.title("Flake Sizes")
	plt.show()



