# Author:      Adam Robinson
# Description: This program takes a directory of images that have already
#              been processed by ScanProcessing.py or Scan.py and summarizes
#              the distribution of flake sizes and flake contrast values.
#              The current plan is to add flake library viewing functionality
#              to this program. This will include allowing a user to specify
#              a range of contrast values and flake sizes to view and then
#              displaying a GUI that allows them to flip through images sorted
#              by some factor of their choosing.


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

from Progress        import ProgressBar
from multiprocessing import Pool
from scipy.stats     import gaussian_kde
from scipy.signal    import ricker, convolve, cwt

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

def processFile(fname, args):
	# Load it into memory, parse it and extract statistics.
	with open(fname, 'r') as f:
		data = json.loads(f.read())

	surface_areas = []
	for contour in data['contours']:
		c = np.array(contour)
		c[:, 0] = c[:, 0] * args.image_dims[0]
		c[:, 1] = c[:, 1] * args.image_dims[1]
		c = c.astype(np.int32)
		area = cv2.contourArea(c.reshape(-1, 1, 2))
		surface_areas.append(area)

	contrast = np.array(data['contrast_values'])
	r        = np.array(data['r_values'])
	g        = np.array(data['g_values'])
	b        = np.array(data['b_values'])
	return surface_areas, contrast, r, g, b

if __name__ == '__main__':
	# Load the arguments file. 
	with open("ScanSummarize.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# Load a list of json files from the directory.
	meta_files = getFiles(args.image_directory, 'json')

	pb = ProgressBar("Loading", 12, len(meta_files), update_every=5)

	surface_areas   = []
	contrast_values = []
	rs = []
	gs = []
	bs = []

	if args.n_processes > 1:
		pool = Pool(args.n_processes)

		current_in_process = 0
		total_processed    = 0
		idx                = 0
		results            = []

		while total_processed < len(meta_files):
			while current_in_process < args.n_processes and idx < len(meta_files):
				if os.path.split(meta_files[idx])[-1] == '_scan.json':
					continue
				res = pool.apply_async(processFile, (meta_files[idx], args))
				results.append(res)
				idx                += 1
				current_in_process += 1

			done = []
			for r in results:
				if r.ready():
					surface_area, contrast, red, g, b = r.get(0.01)
					surface_areas.extend(surface_area)
					contrast_values.append(contrast)
					rs.append(red)
					gs.append(g)
					bs.append(b)
					total_processed    += 1
					current_in_process -= 1
					pb.update(total_processed)
					done.append(r)

			for d in done:
				results.remove(d)

			time.sleep(0.001)
	else:
		for idx, fname in enumerate(meta_files):
			if os.path.split(fname)[-1] == '_scan.json':
				continue
			surface_area, contrast, red, g, b = processFile(fname, args)
			surface_areas.extend(surface_area)
			contrast_values.append(contrast)
			rs.append(red)
			gs.append(g)
			bs.append(b)
			pb.update(idx + 1)

	contrast_values = np.concatenate(tuple(contrast_values))
	rs = np.concatenate(tuple(rs))
	gs = np.concatenate(tuple(gs))
	bs = np.concatenate(tuple(bs))

	pb.finish()

	print("Total flakes identified: %d"%len(surface_areas))

	# surface_areas   = []
	# contrast_values = [] 
	# for idx, file in enumerate(meta_files):
	# 	# Don't process _scan.json
	# 	if os.path.split(file)[-1] == '_scan.json':
	# 		continue

	# 	# Load it into memory, parse it and extract statistics.
	# 	with open(file, 'r') as f:
	# 		data = json.loads(f.read())

	# 	# Load the image with the same name.
	# 	imgpath = ".".join(os.path.split(file)[-1].split('.')[:-1]) + '.png'
	# 	imgpath = os.path.join(args.image_directory, imgpath)
	# 	img     = cv2.imread(imgpath)

	# 	for contour in data['contours']:
	# 		c = np.array(contour)
	# 		c[:, 0] = c[:, 0] * img.shape[1]
	# 		c[:, 1] = c[:, 1] * img.shape[0]
	# 		c = c.astype(np.int32)
	# 		area = cv2.contourArea(c.reshape(-1, 1, 2))
	# 		surface_areas.append(area)

	# 	contrast = np.array(data['contrast_values'])
	# 	contrast_values.append(contrast)

	# 	pb.update(idx + 1)

	# pb.finish()

	# Make a log x-scale plot of the distribution of flake sizes.
	fig, ax = plt.subplots(1, 1)
	ax.hist(np.log(surface_areas), bins=75)
	ax.set_xlabel("Surface Area in Square Pixels (log scale)")
	ax.set_ylabel("Count")
	ax.set_title("Flake Sizes")
	plt.show()

	# Make a histogram of contrast values.
	plt.hist(contrast_values, bins=150)
	plt.xlabel("Optical Contrast")
	plt.ylabel("Count (Pixels)")
	plt.title("Optical Contrast Distribution")
	plt.show()

	# Make a smoothed kernel density estimate.
	contrast_values = np.random.choice(
		contrast_values, 
		min(200000, len(contrast_values)), 
		replace=False
	)
	kde = gaussian_kde(contrast_values, bw_method='silverman')

	n_points = 512
	xrng = np.linspace(contrast_values.min(), contrast_values.max(), n_points)
	y    = kde(xrng)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	ax1.plot(xrng, y)
	ax1.set_xlabel("Contrast")
	ax1.set_ylabel("P(Contrast)")
	ax1.set_title("Normalized Contrast Distribution (Kernel Density Estimate)")

	# # We'll use a Ricker wavelet to enhance the peaks of the 
	# # distribution. This should assist us in eliminating noise
	# # and determining the contrast values that correspond to 
	# # certain thicknesses.

	def RickerWavelet(n, sigma):
		return ricker(n, sigma)




	_rng        = contrast_values.max() - contrast_values.min()
	widths      = np.linspace(
		max(int(_rng / 20), 1), 
		int(_rng / 2), 
		n_points
	)
	transformed = cwt(y, RickerWavelet, widths) 

	ax2.imshow(transformed, cmap='Greys')
	ax2.set_xlabel("Contrast")
	ax2.set_ylabel("Wavelet Width")
	ax2.set_title("Continuous Wavelet Transform with Ricker Wavelet")
	plt.show()

