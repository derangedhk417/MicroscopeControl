# Author:      Adam Robinson
# Description: This file contains code that breaks down how microscope
#              images are processed by ScanProcessing.py. This allows the
#              user to troubleshoot when the system is not properly 
#              identifying flakes. Often times, this means diagnosing 
#              issues such as incorrect exposure time, image noise, etc.

import code
import sys
import os
import time
import cv2
import argparse
import json
import shutil
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches

from multiprocessing import Pool
from ImageProcessor  import ImageProcessor, FlakeExtractor
from Progress        import ProgressBar
from scipy.stats     import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

colors = [
	"#F0A3FF", "#0075DC", "#993F00",
	"#4C005C", "#191919", "#005C31",
	"#2BCE48", "#FFCC99", "#808080",
	"#94FFB5", "#8F7C00", "#9DCC00",
	"#C20088", "#003380", "#FFA405",
	"#FFA8BB", "#426600", "#FF0010",
	"#5EF1F2", "#00998F", "#E0FF66",
	"#740AFF", "#990000", "#FF5005"
]

def torgb(h):
	return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]

def getFiles(directory):
	entries = os.listdir(directory)

	results = []
	for entry in entries:
		try:
			path = os.path.join(directory, entry)
			
		except FileNotFoundError as ex:
			# Its possible for a file to get deleted between the call
			# to os.listdir and the calls to getctime and getmtime.
			continue

		ext = entry.split(".")[-1].lower()
		if ext == 'png':
			results.append(path)

	return results
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

def debug_show(img, line):
	plt.imshow(img)
	plt.title("Line: %d"%line)
	plt.show()

if __name__ == '__main__':
	# Load the arguments file. 
	with open("ImageAnalyzer.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# The plan is to use this program as a testing ground for better
	# ways to analyze images and more accurately extract flake geometry.
	# This is being done to address the following issue:
	# 
	# BGMM and other clustering methods are good at differentiating
	# different flake thicknesses within the same flake, but they 
	# are slow and it is hard to determine how many clusters the BGMM 
	# should be allowed to fit.
	#
	# The more classical method is fast but does a poor job of differentiating
	# between different thicknesses in the same flake. 

	files = getFiles(args.image_dir)
	img   = cv2.imread(args.image)

	# plt.imshow(img)
	# plt.show()


	start_preprocess = time.time_ns()
	# ===========================================
	# TEST METHOD ONE
	# ===========================================
	# This will use k-means to fit the HSV space of the image without
	# denoising and with a downscaled version of the image. This is purely
	# for the purpose of defining the outer contours of flakes. The contrast
	# method will be used to determine thickness.


	# Determined that this needs to be a multi-pass system. The first pass
	# will use parameter 18 to find the thicker stuff. The system will then mask 
	# out every flake found that way and run the whole thing again with parameter
	# 12. This should allow it to find the fainter stuff, which is also generally
	# smaller.
	ds_factor    = 1 / 10
	bg_tolerance = 12

	# Downscale by a factor of 8 so that the k-means won't take forever.
	img_ds = cv2.resize(img, (0, 0), fx=ds_factor, fy=ds_factor)

	# Convert to HSV
	hsv_img_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2HSV)

	# plt.imshow(hsv_img_ds)
	# plt.show()

	# Run k-means
	# First, reshape the image pixels so that they match the 
	# format expected by sklearn.
	hsv_ds_input = hsv_img_ds.reshape((
		hsv_img_ds.shape[0] * hsv_img_ds.shape[1], 
		3
	))

	# Run k-means with two clusters. This should do a decent job of
	# separating the background from the flakes. We will take the mean
	# of everything classified as background as the background color.
	kmeans_model    = KMeans(n_clusters=2).fit(hsv_ds_input)
	classifications = kmeans_model.predict(hsv_ds_input)


	bg_class = None
	bg_color = None
	c_0      = hsv_ds_input[classifications == 0]
	c_1      = hsv_ds_input[classifications == 1]

	if c_0.shape[0] > c_1.shape[0]:
		bg_class = 0
		bg_color = hsv_ds_input[classifications == 0].mean(axis=0)
	else:
		bg_class = 1
		bg_color = hsv_ds_input[classifications == 1].mean(axis=0)


	bg_color = bg_color.astype(np.uint8)
	print("BG COLOR: [%d, %d, %d]"%(bg_color[0], bg_color[1], bg_color[2]))

	end_preprocess = time.time_ns()

	print("Preprocessing: %fs"%((end_preprocess - start_preprocess) / 1e9))

	# Here we subtract the background from each image in the directory.
	# We will do all of the subtraction at once so that we can time it.

	# First, load all of the image files.
	images = [cv2.imread(imfile) for imfile in files]

	start = time.time_ns()

	bg_removed_images = []
	for image in images:

		# Here we will create a mask that just determines the difference in the hue
		# channel and makes sure it is within bg_tolerance of the background value.
		img_hsv          = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		bg_removed       = img_hsv.reshape((
			img_hsv.shape[0] * img_hsv.shape[1],
			3
		))
		bg_removed_hue   = bg_removed[:, 0]
		# We need to increase the bit depth and make this signed
		# so that we don't get overflow
		bg_removed_hue  = bg_removed_hue.astype(np.int16)
		sub             = np.abs(bg_removed_hue - bg_color[0])
		# Plot the difference for inspection.
		diff_img = sub.reshape((
			img_hsv.shape[0],
			img_hsv.shape[1],
			1
		))
		#plt.imshow(diff_img)
		#plt.show()

		bg_removed_2 = bg_removed.copy()

		# Use 18 for the initial pass.
		mask             = sub < bg_tolerance
		bg_removed[mask] = [0, 0, 0]

		# Use 12 for a secondary pass.
		# bg_tolerance = 12
		# mask         = sub < bg_tolerance
		# bg_removed_2[mask] = [0, 0, 0]

		# Make a mask that is just the stuff from the first pass,
		# dilated. We will use this to cover up everything that was
		# found in the first pass.
		#first_pass_mask = sub < 18


		bg_removed_rgb_img = bg_removed.reshape((
			img_hsv.shape[0],
			img_hsv.shape[1],
			3
		))

		bg_removed_rgb_img = cv2.cvtColor(bg_removed_rgb_img, cv2.COLOR_HSV2RGB)
		bg_removed_images.append(bg_removed_rgb_img)

	end = time.time_ns()

	print("Directory processing: %fs"%((end - start) / 1e9))

	for img, original in zip(bg_removed_images, images):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.imshow(img)

		original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
		ax2.imshow(original)
		plt.show()



	# silhouette_scores = []

	# for n_classes in range(2, 12):

	# 	# n_classes    = 6
	# 	kmeans_model = KMeans(n_clusters=n_classes).fit(hsv_ds_input)

	# 	classifications = kmeans_model.predict(hsv_ds_input)

	# 	class_img_ds = hsv_img_ds.copy()
	# 	class_img_ds = class_img_ds.reshape((
	# 		class_img_ds.shape[0] * class_img_ds.shape[1], 
	# 		3
	# 	))

	# 	# Color each pixel based on its cluster.
	# 	for c in range(n_classes):
	# 		class_img_ds[classifications == c] = torgb(colors[c])

	# 	class_img_ds = class_img_ds.reshape((
	# 		hsv_img_ds.shape[0],
	# 		hsv_img_ds.shape[1], 
	# 		3
	# 	))

	# 	# class_img_us = cv2.resize(
	# 	# 	class_img_ds, 
	# 	# 	(2448, 2048),
	# 	# 	interpolation=cv2.INTER_CUBIC
	# 	# )

	# 	plt.imshow(class_img_ds)
	# 	plt.title(str(n_classes))
	# 	plt.show()

	# 	score = silhouette_score(hsv_ds_input, classifications, sample_size=3000)
	# 	silhouette_scores.append(score)

	# plt.plot(range(3, 12), silhouette_scores)
	# plt.show()
	exit()
	

	if img is None:
		print("Failed to load image at path %s."%args.image)
		exit()

	extractor = FlakeExtractor(
		img,
		downscale=args.downscale,
		threshold=args.rejection_threshold,
		contrast_floor=args.contrast_floor
	)

	status, res = extractor.process(
		DEBUG_DISPLAY=args.debug_display
	)

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

	# Now we display a contrast image.
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	bg_img       = (img - res['bg_color'])
	bg_img[bg_img < 0] = 0
	bg_img[bg_img > 255] = 255
	bg_img = bg_img.astype(np.uint8)

	contrast_vals = np.array(res['contrast_values'])
	# kde = gaussian_kde(contrast_vals, bw_method=1)

	# n_points = 512
	# xrng = np.linspace(
	# 	contrast_vals.min(),
	# 	0.15, 
	# 	n_points
	# )
	# y    = kde(xrng)
	# ax1.plot(xrng, y)
	# ax1.set_title("Contrast Distribution KDE")
	ax1.imshow(bg_img)
	ax1.set_title("Background Subtracted Image")
	contrast_img = (bg_img / (res['bg_color'] + img)).sum(axis=2)
	ax2.imshow(contrast_img)
	ax2.set_title("Contrast Values")

	# Next we draw a rectangle over each identified flake and also print
	# the mean and standard deviation of the contrast for each flake.
	#fig, ax = plt.subplots(1, 1)
	
	ax3.imshow(img)

	for idx in range(len(res['rects'])):
		rectangle = res['rects'][idx]
		mean      = res['contrast_means'][idx]
		std       = res['contrast_stds'][idx]

		# Draw the rectangle using the given rectangle coordinates
		# scaled to the image size.
		rectangle = np.array(rectangle)
		rectangle[:, 0] *= img.shape[1]
		rectangle[:, 1] *= img.shape[0]
		rect             = patches.Polygon(
			rectangle, 
			linewidth=1,
			edgecolor='r',
			facecolor='none'
		)

		ax3.add_patch(rect)
		ax3.text(
			rectangle[0][0], 
			rectangle[0][1], 
			"%1.3f"%mean,
			size=8,
			color='r'
		)

	ax3.set_title("Identified Flakes with Mean Contrast Values")


	
	# Show a histogram of contrast values.
	ax4.hist(res['contrast_values'], bins=400)
	ax4.set_xlabel("Optical Contrast")
	ax4.set_ylabel("Count (Pixels)")
	ax4.set_title("Optical Contrast Distribution")
	fig.tight_layout()
	plt.show()