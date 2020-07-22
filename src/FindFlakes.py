import code
import sys
import os
import time
import cv2
import argparse
import json
import numpy             as np
import matplotlib.pyplot as plt
import multiprocessing   as mp

from scipy.stats     import gaussian_kde
from multiprocessing import Pool
from ImageProcessor  import ImageProcessor
from Progress        import ProgressBar
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc           import derivative
from scipy.signal         import find_peaks_cwt, gaussian

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

def processFile(file, args, display=False):
	img       = cv2.imread(file)
	json_path = ".".join(file.split(".")[:-1]) + '.json'
	with open(json_path, 'r') as file:
		data = json.loads(file.read())
	
	if display:
		c_img = img.copy()
		# Draw an image with a contour around every flake that we decided was good.
		for con in data['contours']:
			con = np.array(con)
			con[:, 0] = con[:, 0] * img.shape[1]
			con[:, 1] = con[:, 1] * img.shape[0]
			con = con.astype(np.int32)
			c_img = cv2.drawContours(
				c_img, 
				[con.reshape(-1, 1, 2)], 
				0, (255, 0, 0), 2
			)

		plt.imshow(c_img)
		plt.show()	

	img = cv2.resize(img, (0, 0), fx=args.downscale, fy=args.downscale)
	img = cv2.fastNlMeansDenoisingColored(
		img, 
		30
	)

	contrast_data = []
	mask_sum      = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
	for i, c in enumerate(data['contours']):
		mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
		con  = np.array(c)

		con[:, 0] *= img.shape[1]
		con[:, 1] *= img.shape[0]
		con        = con.astype(np.int32)
		mask       = cv2.fillPoly(mask, [con], 1)
		mask       = mask.astype(np.uint8)

		if display:
			mask_sum = mask_sum + mask

		bg       = np.array(data['bg_color'])
		bg_subtracted = img - bg
		contrast      = bg_subtracted / (bg + img)
		contrast      = contrast[mask == 1].sum(axis=1)
		#flakes.append({'file': file, 'flake_idx': i})
		contrast_data.append(contrast.tolist())

	if display:
		plt.imshow(mask_sum)
		plt.show()

	return contrast_data



if __name__ == '__main__':
	# Load the arguments file. 
	with open("FindFlakes.json", 'r') as file:
		args_specification = json.loads(file.read())

	args = preprocess(args_specification)

	# For now, load each image and draw the bounding boxes onto it.
	files = []
	for entry in os.listdir(args.image_directory):
		ext = entry.split(".")[-1].lower()
		if ext == 'png':
			files.append(os.path.join(args.image_directory, entry))

	# Now we go through each flake in each image and extract all of the 
	# contrast values so we can attempt to determine thickness.

	# flakes        = []
	# contrast_data = []

	pb = ProgressBar("Processing ", 20, len(files), 1, ea=30)


	if args.n_processes > 1:
		pool = mp.Pool(args.n_processes)

		contrast_data = []

		current_in_process = 0
		total_processed    = 0
		idx                = 0
		results            = []
		while total_processed < len(files):

			while current_in_process < args.n_processes and idx < len(files):
				res = pool.apply_async(processFile, (files[idx], args))
				results.append(res)
				idx                += 1
				current_in_process += 1

			done = []
			for r in results:
				if r.ready():
					v = r.get(0.01)
					total_processed    += 1
					current_in_process -= 1
					pb.update(total_processed)
					done.append(r)
					contrast_data.append(v)

			for d in done:
				results.remove(d)

			time.sleep(0.005)
	else:
		contrast_data = []
		for idx, file in enumerate(files):
			contrast_data.append(processFile(file, args, False))
			pb.update(idx + 1)


	pb.finish()

	# Here we restructure this as a big flat array.
	arrays = []
	for a in contrast_data:
		arrays.append(np.concatenate(tuple(a)))

	contrast_data = np.concatenate(tuple(arrays))

	contrast_data = np.random.choice(
		contrast_data, 
		min(100000, contrast_data.shape[0]), 
		replace=False
	)
	kde = gaussian_kde(contrast_data, bw_method='silverman')

	plt.hist(contrast_data, bins=200)
	plt.show()

	n = 512
	x = np.linspace(-1, contrast_data.max(), n)
	y = kde(x)

	def wavelet(n, scale, **b):
		return gaussian(n, scale * 0.7)


	peaks     = find_peaks_cwt(y, np.arange(
		5, 
		15
	), wavelet=wavelet, noise_perc=15)
	positions = x[peaks]
	print(peaks)

	# Calculate the mean distance between subsequent peaks.
	distances = []
	last = positions[0]
	for p in positions[1:]:
		distances.append(p - last)
		last = p
	#code.interact(local=locals())

	# positions = x[peaks]

	# dx1 = []
	# dx2 = []
	# for p in x:
	# 	dx1.append(derivative(kde, p, dx=1e-2, n=1))
	#	dx2.append(derivative(kde, p, dx=1e-2, n=2))

	orig, = plt.plot(x, y)
	# d1,   = plt.plot(x, dx1)
	# d2,   = plt.plot(x, dx2)
	# plt.legend([orig, d1, d2], ["Distribution", "First Derivative", "Second Derivative"])
	plt.title("Contrast Distribution")
	plt.xlabel("Constrast")
	plt.ylabel("Frequency")

	for pos in positions:
		plt.axvline(pos)

	plt.show()




	# c_img = bimg.copy()
		# # Draw an image with a contour around every flake that we decided was good.
		# for con in data['contours']:
		# 	con = np.array(con)
		# 	con[:, 0] = con[:, 0] * bimg.shape[1]
		# 	con[:, 1] = con[:, 1] * bimg.shape[0]
		# 	con = con.astype(np.int32)
		# 	c_img = cv2.drawContours(
		# 		c_img, 
		# 		[con.reshape(-1, 1, 2)], 
		# 		0, (255, 0, 0), 1
		# 	)

		# plt.imshow(c_img)
		# plt.show()	



	# c_img = img.copy()
	# # Draw an image with a contour around every flake that we decided was good.
	# for con in data['contours']:
	# 	con = np.array(con)
	# 	con[:, 0] = con[:, 0] * img.shape[1]
	# 	con[:, 1] = con[:, 1] * img.shape[0]
	# 	con = con.astype(np.int32)
	# 	c_img = cv2.drawContours(
	# 		c_img, 
	# 		[con.reshape(-1, 1, 2)], 
	# 		0, (255, 0, 0), 2
	# 	)

	# plt.imshow(c_img)
	# plt.show()	

		
