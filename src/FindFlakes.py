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
from mpl_toolkits.mplot3d import Axes3D

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
		# mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

		# for c in data['contours']:
		# 	con = np.array(c)
		# 	con[:, 0] *= img.shape[1]
		# 	con[:, 1] *= img.shape[0]
		# 	con        = con.astype(np.int32)

		# 	mask = cv2.fillPoly(mask, [con], 1)

		# mask = mask.astype(np.uint8)

		# # Select all pixels outside the mask and average them.
		# bg_mask  = mask == 0
		# bg       = img[bg_mask]
		# bg_color = bg.mean(axis=0)
		# print(bg_color)

		# # Make a background subtracted image and disaply it.
		# bg_subtracted = (img.astype(np.float32) - bg_color)
		# bg_subtracted[bg_subtracted < 3]   = 0
		# bg_subtracted[bg_subtracted > 255] = 255
		# bg_subtracted = bg_subtracted.astype(np.uint8)
		# plt.imshow(bg_subtracted)
		# plt.show()

		# # Compute constrast by summing across all channels and
		# # remove everything below 30

		# contrast = cv2.cvtColor(bg_subtracted, cv2.COLOR_BGR2GRAY)
		# contrast[contrast < 16] = 0

		# proc = ImageProcessor(contrast, downscale=1, mode='GS')
		# res  = proc.noStore().level().dilate(4).erode(4).border(5, 0).edge(0, 1).dilate(2).done()

		# plt.imshow(res)
		# plt.show()

		# # Now we determine bounding boxes and remove everything thats too small.
		# # Extract contours and bounding rectangles.
		# c   = proc.extractContours(res)
		# r   = proc.calculateBoundingRectangles(c)

		
		# filtered_rects    = []
		# filtered_contours = []
		# for rect, contour in zip(r, c):
		# 	((left, top), (width, height), angle) = rect
		# 	largest = max(width, height)

		# 	# Correct for the border.
		# 	left    -= 5
		# 	top     -= 5
		# 	contour -= 5

		# 	if largest > 20:
		# 		filtered_rects.append(((left, top), (width, height), angle))
		# 		filtered_contours.append(contour)

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

		# # Now we essentially perform the flake extraction again and determine
		# # whether or not to keep the image.
		# leveled = contrast.copy()
		# leveled[leveled > 0] = 1

		# bordered = cv2.copyMakeBorder(
		# 	leveled, 
		# 	width, 
		# 	width, 
		# 	width, 
		# 	width, 
		# 	cv2.BORDER_CONSTANT,
		# 	value=color
		# )
		# edged = cv2.Canny(leveled, 0, 1)



		# contrast_small = cv2.resize(bg_subtracted, (0, 0), fx=0.03, fy=0.03)
		# contrast_small = contrast_small.sum(axis=2)
		# contrast_small[contrast_small < 30] = 0

		# X, Y = np.meshgrid(
		# 	np.arange(contrast_small.shape[1]),
		# 	np.arange(contrast_small.shape[0])
		# )

		# Z = contrast_small


		# plt.imshow(contrast)
		# plt.show()

		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False)
		# plt.show()
