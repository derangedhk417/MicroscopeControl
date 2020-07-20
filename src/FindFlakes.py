import code
import sys
import os
import time
import cv2
import argparse
import json
import numpy             as np
import matplotlib.pyplot as plt

from scipy.stats     import gaussian_kde
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

	# Load the image and its json metadata.
	for file in files:
		img  = cv2.imread(file)
		json_path = ".".join(file.split(".")[:-1]) + '.json'
		with open(json_path, 'r') as file:
			data = json.loads(file.read())

		rect_img = img.copy()
		# Convert the relative coordinates to absolute coordinates
		# and draw the bounding boxes onto the image.
		for rect in data['rects']:
			rect = np.array(rect)
			rect[:, 0] *= img.shape[1]
			rect[:, 1] *= img.shape[0]
			rect        = rect.reshape(-1, 1, 2)
			box   = np.int0(rect)
			rect_img   = cv2.drawContours(rect_img, [box], 0, (255, 0, 0), 2) 

		plt.imshow(rect_img)
		plt.show()

		
		# Make a separate maske for each contour and print the variance
		# within the contour before showing it on screen.
		
		# for c in data['contours']:
		# 	mask  = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
		# 	c_img = img.copy()
		# 	con = np.array(c)

		# 	con[:, 0] *= img.shape[1]
		# 	con[:, 1] *= img.shape[0]
		# 	con       -= 25 # TMP DELETE THIS
		# 	con        = con.astype(np.int32)


		# 	mask = cv2.fillPoly(mask, [con], 1)
		# 	mask = mask.astype(np.uint8)

		# 	# Erode the mask
		# 	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
		# 	eroded = cv2.erode(mask, kernel)
		# 	eroded[eroded > 0] = 1
		# 	mask   = eroded
			
		# 	contour_img = cv2.drawContours(
		# 		c_img, 
		# 		[con.reshape(-1, 1, 2)], 
		# 		0, (255, 0, 0), 2
		# 	)
		# 	plt.imshow(contour_img)
		# 	plt.show()

		# Make a mask image with all of the contours filled.
		mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

		for c in data['contours']:
			con = np.array(c)
			con[:, 0] *= img.shape[1]
			con[:, 1] *= img.shape[0]
			con        = con.astype(np.int32)

			mask = cv2.fillPoly(mask, [con], 1)

		mask = mask.astype(np.uint8)

		# Select all pixels outside the mask and average them.
		bg_mask  = mask == 0
		bg       = img[bg_mask]
		bg_color = bg.mean(axis=0)
		print(bg_color)

		# Make a background subtracted image and disaply it.
		bg_subtracted = (img.astype(np.float32) - bg_color)
		bg_subtracted[bg_subtracted < 3]   = 0
		bg_subtracted[bg_subtracted > 255] = 255
		bg_subtracted = bg_subtracted.astype(np.uint8)
		plt.imshow(bg_subtracted)
		plt.show()

		# Run a kernel density estimage on the image pixels
		# s = bg_subtracted.shape[0] * bg_subtracted.shape[1]
		# # r = bg_subtracted[:, :, 0].reshape(s)
		# # r = r[r > 10]
		# # rkde = gaussian_kde(r, bw_method='silverman')
		

		# g = bg_subtracted[:, :, 1].reshape(s)
		# g = g[g > 10]
		# gkde = gaussian_kde(g, bw_method='silverman')
		# print(gkde.covariance.shape)
	
		# b = bg_subtracted[:, :, 2].reshape(s)
		# b = b[b > 10]
		# bkde = gaussian_kde(b, bw_method='silverman')
		# print(bkde.covariance.shape)
		
		# rng  = np.linspace(0, 255, 256)

		# #plt.plot(rng, rkde(rng), color='red')
		# plt.plot(rng, gkde(rng), color='green')
		# plt.plot(rng, bkde(rng), color='blue')
		# plt.show()

		
