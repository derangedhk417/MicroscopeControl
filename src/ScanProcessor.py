import code
import sys
import os
import time
import cv2
import argparse
import json
import numpy             as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from ImageProcessor  import ImageProcessor

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

def getFilesSince(directory, timestamp):
	entries = os.listdir(directory)

	results = []
	for entry in entries:
		try:
			path = os.path.join(directory, entry)
			ts   = os.path.getctime(path)
			ms   = os.path.getmtime(path)
			ts   = max(ms, ts)
		except FileNotFoundError as ex:
			# Its possible for a file to get deleted between the call
			# to os.listdir and the calls to getctime and getmtime.
			continue

		if ts > timestamp:
			ext = entry.split(".")[-1].lower()
			if ext == 'png':
				results.append(path)

	return results

def processFile(fpath, args):
	print(fpath)
	start = time.time_ns()
	while True:
		try:
			img  = cv2.imread(fpath)
			break
		except:
			time.sleep(0.05)
			if (time.time_ns() - start) / 1e9 > 4:
				return
	proc = ImageProcessor(img, downscale=5)

	# This set of filters will produce good edges for the contour
	# algorithm.
	tmp = proc.noStore().denoise(30).laplacian(28).dilate(4).erode(4)
	res = tmp.level().border(5, 0).edge(0, 1).dilate(2).done()


	# Extract contours and bounding rectangles.
	c   = proc.extractContours(res)
	r   = proc.calculateBoundingRectangles(c)
	img = proc.img

	if args.rejection_threshold != 0: 
		filtered_rects    = []
		filtered_contours = []
		for rect, contour in zip(r, c):
			((left, top), (width, height), angle) = rect
			largest = max(width, height)

			left -= 5
			top  -= 5

			contour -= 5

			if largest > args.rejection_threshold:
				filtered_rects.append(((left, top), (width, height), angle))
				filtered_contours.append(contour)

	else:
		filtered_rects    = r
		filtered_contours = c

	if len(filtered_rects) == 0:
		print("Deleting %s"%fpath)
		os.remove(fpath)
		return

	# Now we convert all coordinates to relative coordinates so that 
	# the size of the image they are processed against doesn't matter.

	converted_contours = []
	converted_rects    = []

	for r, c in zip(filtered_rects, filtered_contours):
		# Get the four points that represent the corners of the bounding
		# rectangle and convert them.
		box_points = cv2.boxPoints(r).astype(np.float32)
		box_points[:, 0] = box_points[:, 0] / img.shape[1]
		box_points[:, 1] = box_points[:, 1] / img.shape[0]

		con = c.astype(np.float32)
		# Convert the contour points.
		con[:, 0] = con[:, 0] / img.shape[1]
		con[:, 1] = con[:, 1] / img.shape[0]

		converted_contours.append(con.tolist())
		converted_rects.append(box_points.tolist())

	results = {
		"fname"    : fpath.split("\\")[-1],
		"contours" : converted_contours,
		"rects"    : converted_rects
	}

	outfile = ".".join(fpath.split("\\")[-1].split(".")[:-1]) + '.json'
	outpath = os.path.join("\\".join(fpath.split("\\")[:-1]), outfile)
	print("Writing to %s"%outpath)
	with open(outpath, 'w') as file:
		file.write(json.dumps(results))


if __name__ == '__main__':
	# Load the arguments file. 
	with open("ScanProcessor.json", 'r') as file:
		args_specification = json.loads(file.read())

	args = preprocess(args_specification)

	# We need to create a thread pool and dispatch images to it
	# as soon as they appear in the folder.
	process_pool = Pool(args.n_processes)
	last_time    = 0

	processed = []

	try:
		while True:
			files     = getFilesSince(args.image_directory, last_time)
			last_time = time.time()

			for file in files:
				if file not in processed:
					process_pool.apply_async(processFile, (file, args))
					processed.append(file)
				#processFile(file, args)

	except KeyboardInterrupt as ex:
		print("Exiting . . . ")