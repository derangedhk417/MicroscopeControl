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
from ImageProcessor  import ImageProcessor, FlakeExtractor
from Progress        import ProgressBar

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

def loadAndProcess(fpath, args):
	try:
		img = cv2.imread(fpath)
		return processFile(img, fpath, args)
	except Exception as ex:
		print(ex)
		raise ex


def processFile(img, fpath, args):
	extractor = FlakeExtractor(
		img,
		downscale=args.downscale,
		threshold=args.rejection_threshold,
		contrast_floor=args.contrast_floor
	)

	status, res = extractor.process()

	if status:
		outfile = ".".join(fpath.split("\\")[-1].split(".")[:-1]) + '.json'
		outpath = os.path.join("\\".join(fpath.split("\\")[:-1]), outfile)
		with open(outpath, 'w') as file:
			file.write(json.dumps(res))
		return True, None
	else:
		os.remove(fpath)
		return False, fpath

	# proc = ImageProcessor(img, downscale=args.downscale)

	# # This set of filters will produce good edges for the contour
	# # algorithm.
	# tmp = proc.noStore().denoise(30).laplacian(28).dilate(4).erode(4)
	# res = tmp.level().border(5, 0).edge(0, 1).dilate(2).done()


	# # Extract contours and bounding rectangles.
	# c   = proc.extractContours(res)
	# r   = proc.calculateBoundingRectangles(c)

	# if args.rejection_threshold != 0: 
	# 	filtered_rects    = []
	# 	filtered_contours = []
	# 	for rect, contour in zip(r, c):
	# 		((left, top), (width, height), angle) = rect
	# 		largest = max(width, height)

	# 		# Correct for the border.
	# 		left    -= 5
	# 		top     -= 5
	# 		contour -= 5

	# 		if largest > args.rejection_threshold:
	# 			filtered_rects.append(((left, top), (width, height), angle))
	# 			filtered_contours.append(contour)

	# else:
	# 	filtered_rects    = r
	# 	filtered_contours = c

	# if len(filtered_rects) == 0:
	# 	os.remove(fpath)
	# 	return True, None

	# # Now that the initial processing necessary to subtract the background is
	# # done, we apply a more rigorous method for finding actualy flakes on the
	# # back ground subtracted image.

	# mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

	# for c in filtered_contours:
	# 	con = np.array(c)
	# 	con[:, 0] *= img.shape[1]
	# 	con[:, 1] *= img.shape[0]
	# 	con        = con.astype(np.int32)

	# 	mask = cv2.fillPoly(mask, [con], 1)

	# mask = mask.astype(np.uint8)

	# bg_mask  = mask == 0
	# bg       = img[bg_mask]

	# if bg.shape[0] < 1:
	# 	return False, fpath

	# bg_color = bg.mean(axis=0)

	# bg_subtracted = (img.astype(np.float32) - bg_color)
	# bg_subtracted[bg_subtracted < 3]   = 0
	# bg_subtracted[bg_subtracted > 255] = 255
	# bg_subtracted = bg_subtracted.astype(np.uint8)
	

	# # TODO: Make the cutoff (16) configurable.
	# contrast = cv2.cvtColor(bg_subtracted, cv2.COLOR_BGR2GRAY)
	# contrast[contrast < 16] = 0

	# proc = ImageProcessor(contrast, downscale=1, mode='GS')
	# tmp  = proc.noStore().level().dilate(4).erode(4)
	# res  = tmp.border(5, 0).edge(0, 1).dilate(2).done()

	# # Now we determine bounding boxes and remove everything thats too small.
	# # Extract contours and bounding rectangles.
	# c = proc.extractContours(res)
	# r = proc.calculateBoundingRectangles(c)

	# if args.rejection_threshold != 0: 
	# 	filtered_rects    = []
	# 	filtered_contours = []
	# 	for rect, contour in zip(r, c):
	# 		((left, top), (width, height), angle) = rect
	# 		largest = max(width, height)

	# 		# Correct for the border.
	# 		left    -= 5
	# 		top     -= 5
	# 		contour -= 5

	# 		if largest > args.rejection_threshold:
	# 			filtered_rects.append(((left, top), (width, height), angle))
	# 			filtered_contours.append(contour)
	# else:
	# 	filtered_rects    = r
	# 	filtered_contours = c

	# if len(filtered_rects) == 0:
	# 	os.remove(fpath)
	# 	return True, None

	# # Now we convert all coordinates to relative coordinates so that 
	# # the size of the image they are processed against doesn't matter.

	# converted_contours = []
	# converted_rects    = []

	# for r, c in zip(filtered_rects, filtered_contours):
	# 	# Get the four points that represent the corners of the bounding
	# 	# rectangle and convert them.
	# 	box_points = cv2.boxPoints(r).astype(np.float32)
	# 	box_points[:, 0] = box_points[:, 0] / img.shape[1]
	# 	box_points[:, 1] = box_points[:, 1] / img.shape[0]

	# 	con = c.astype(np.float32)
	# 	# Convert the contour points.
	# 	con[:, 0] = con[:, 0] / img.shape[1]
	# 	con[:, 1] = con[:, 1] / img.shape[0]

	# 	converted_contours.append(con.tolist())
	# 	converted_rects.append([box_points.tolist(), r[2]])

	# results = {
	# 	"fname"    : fpath.split("\\")[-1],
	# 	"contours" : converted_contours,
	# 	"rects"    : converted_rects,
	# 	"bg_color" : bg_color.tolist()
	# }

	




if __name__ == '__main__':
	# Load the arguments file. 
	with open("ScanProcessing.json", 'r') as file:
		args_specification = json.loads(file.read())
	args  = preprocess(args_specification)

	files = getFilesSince(args.image_directory, 0)

	print("Processing %d files using %d processes."%(
		len(files), 
		args.n_processes
	))
	

	pb = ProgressBar("Processing ", 11, len(files), 1, ea=25)

	if args.n_processes > 1:
		pool = Pool(args.n_processes)

		current_in_process = 0
		total_processed    = 0
		idx                = 0
		results            = []
		failures           = []

		while total_processed < len(files):
			while current_in_process < args.n_processes and idx < len(files):
				res = pool.apply_async(loadAndProcess, (files[idx], args))
				results.append(res)
				idx                += 1
				current_in_process += 1

			done = []
			for r in results:
				if r.ready():
					status, fname = r.get(0.01)
					total_processed    += 1
					current_in_process -= 1
					pb.update(total_processed)
					done.append(r)

					if not status:
						failures.append(fname)

			for d in done:
				results.remove(d)

			time.sleep(0.001)
	else:
		for idx, file in enumerate(files):
			img = cv2.imread(file)
			processFile(img, file, args)
			pb.update(idx + 1)

	pb.finish()

	if len(failures) > 0:
		print("The following (%d) files were deleted."%len(failures))
		for f in failures:
			print("     %s"%f)