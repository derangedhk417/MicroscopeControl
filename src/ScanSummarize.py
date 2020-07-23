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
	plt.hist(np.log(surface_areas), bins=75)
	plt.xlabel("Surface Area in Square Pixels (log scale)")
	plt.ylabel("Count")
	plt.title("Flake Sizes")
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

	xrng = np.linspace(contrast_values.min(), contrast_values.max(), 512)
	y    = kde(xrng)

	plt.plot(xrng, y)
	plt.xlabel("Contrast")
	plt.ylabel("P(Contrast)")
	plt.title("Normalized Contrast Distribution (Kernel Density Estimate)")
	plt.show()


	# # Make a multiseries plot of the color values.
	# r_values = np.random.choice(
	# 	rs,
	# 	min(200000, len(rs)),
	# 	replace=False
	# )
	# rkde = gaussian_kde(r_values, bw_method='silverman')

	# b_values = np.random.choice(
	# 	bs,
	# 	min(200000, len(bs)),
	# 	replace=False
	# )
	# bkde = gaussian_kde(b_values, bw_method='silverman')

	# g_values = np.random.choice(
	# 	gs,
	# 	min(200000, len(gs)),
	# 	replace=False
	# )
	# gkde = gaussian_kde(g_values, bw_method='silverman')

	# xrng = np.linspace(
	# 	max(min(r_values.min(), g_values.min(), b_values.min()), 20),
	# 	max(r_values.max(), g_values.max(), b_values.max()),
	# 	512
	# )
	# ry = rkde(xrng)
	# gy = gkde(xrng)
	# by = bkde(xrng)

	# r, = plt.plot(xrng, ry, c='red')
	# g, = plt.plot(xrng, gy, c='green')
	# b, = plt.plot(xrng, by, c='blue')

	# plt.xlabel("Color Value")
	# plt.ylabel("P(Color Value)")
	# plt.title("Flake Color Distribution (Gaussian Kernel Density Estimate)")
	# plt.show()

