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
	with open("ImageAnalyzer.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	img = cv2.imread(args.image)

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