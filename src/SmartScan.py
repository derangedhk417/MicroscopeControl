# Author:      Adam Robinson
# Description: This program is similar to Scan.py, except that it performs a scan of the substrate
#              at a much lower zoom in order to determine which regions contain flakes. It then
#              zooms in on only those regions and takes high quality images of the flakes in them.

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

from MicroscopeControl  import MicroscopeController
from scipy.optimize     import curve_fit
from matplotlib.patches import Rectangle
from Progress           import ProgressBar
from multiprocessing    import Pool
from ImageProcessing    import processFile
from datetime           import datetime

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

def calibrateFocus(microscope, args):
	focus_test_points = np.array([
		args.focus_points[i:i + 2] for i in range(0, len(args.focus_points), 2)
	])
	focal_points = []
	for x, y in focus_test_points:
		microscope.stage.moveTo(x, y)
		microscope.autoFocus(
			args.focus_range,
			args.autofocus_parameters[0],
			args.autofocus_parameters[1],
			args.autofocus_parameters[2]
		)
		focal_points.append(microscope.focus.getFocus())

	focal_points = np.array(focal_points)

	def fit(X, mx, my, b):
		x, y = X
		return mx*x + my*y + b

	res, cov = curve_fit(fit, focus_test_points.T, focal_points)

	def interp(X):
		x, y = X
		return res[0]*x + res[1]*y + res[2]

	rmse = np.sqrt(np.square(interp(focus_test_points.T) - focal_points).mean())
	print("The focal points have been fit to a function of the form z = ax + by + c.")
	print("RMSE of fit: %f"%rmse)

	if rmse > args.max_focus_error:
		print("Fit too poor to continue.")
		print("Please ensure that the sample is flat and level.")
		exit()

	return interp

def getRegionsOfInterest(img, args, x0, y0):
	fine_zoom,   fine_w,   fine_h,   fine_exposure   = args.fine_zoom
	coarse_zoom, coarse_w, coarse_h, coarse_exposure = args.coarse_zoom

	# We want to calculate the contrast of every pixel in the image and then subdivide it into 
	# regions with the same size as an image at the fine zoom setting. We'll return an array of
	# points corresponding to the top left corner of any such region that meets the criteria
	# specified in the arguments to this program (negative_contrast, threshold_ratio).
	
	background = max(np.argmax(np.bincount(img.flatten())), 1)
	img        = img.astype(np.float32)
	contrast   = (background - img) / background

	if args.negative_contrast:
		contrast = -contrast

	contrast[contrast < args.contrast_threshold] = 0

	mm_per_pixel_x = coarse_w / contrast.shape[1]
	mm_per_pixel_y = coarse_h / contrast.shape[0]
	mm_per_pixel   = (mm_per_pixel_x + mm_per_pixel_y) / 2

	# Now we iterate over the above mentioned regions of this image and calculate the percentage
	# of each image that has contrast greater than zero.
	x, y    = 0, 0
	regions = [] 
	while x < coarse_w - fine_w:
		while y < coarse_h - fine_h:
			# We need to convert coordinates into indices into the image array so that we can select
			# the correct subset of pixels for the calculation.
			y_low  = int(round(y  / mm_per_pixel))
			y_high = int(round((y + fine_h) / mm_per_pixel)) + 1

			x_low  = int(round(x  / mm_per_pixel))
			x_high = int(round((x + fine_w) / mm_per_pixel)) + 1

			subimage = contrast[y_low:y_high, x_low:x_high]
			ratio    = subimage[subimage > 0].sum() / (subimage.shape[0] * subimage.shape[1])

			if ratio > args.threshold_ratio:
				regions.append([x + x0, y + y0])

			y += fine_h
		y = 0
		x += fine_w

	return regions

def parabolicSubtract(img, n_fit=128):
	def fit(X, mmx, mmy, mx, my, b):
		y, x = X
		return mmx*x**2 + mmy*y**2 + mx*x + my*y + b

	# Select a random subset of the image pixels to perform the fit on. This will be too slow 
	# otherwise.
	fit_x = np.random.randint(0, img.shape[1], (n_fit, ))
	fit_y = np.random.randint(0, img.shape[0], (n_fit, ))
	fit_points = np.stack([fit_y, fit_x], axis=1)

	Z = img[fit_y, fit_x]

	res, cov = curve_fit(fit, fit_points.T, Z)

	def interp(y, x):
		return res[0]*x**2 + res[1]*y**2 + res[2]*x + res[3]*y + res[4]

	Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
	background = interp(Y, X)
	subtracted = img - background.T

	_min = subtracted.min()
	_max = subtracted.max()
	subtracted = (((subtracted - _min) / (_max - _min)) * 255).astype(np.uint8)

	return subtracted

if __name__ == '__main__':
	with open("_smartscan.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# We want to track every part of the scan in a meta data structure so that it can be refered to
	# later if necessary.
	meta_data = {
		'arguments'  : args.__dict__,
		'start_time' : str(datetime.now())
	}

	# Check the output directory.
	if not os.path.exists(args.output_directory):
		os.mkdir(args.output_directory)
	else:
		for f in os.listdir(args.output_directory):
			p = os.path.join(args.output_directory, f)
			if os.path.isfile(p):
				print("The specified output directory is not empty.")
				exit()

	# Estimate the size of the scanned area and process the limits arguments.
	x_min, x_max, y_min, y_max = args.bounds

	scan_area = (x_max - x_min) * (y_max - y_min)
	meta_data['scan_area'] = scan_area

	print("Preparing to scan the region:")
	print("    x = %02.4f to %02.4f mm"%(x_min, x_max))
	print("    y = %02.4f to %02.4f mm"%(y_min, y_max))
	print(" area = %04.4f mm²"%scan_area)

	# Process the fine and coarse zoom setting information.
	fine_zoom,   fine_w,   fine_h,   fine_exposure   = args.fine_zoom
	coarse_zoom, coarse_w, coarse_h, coarse_exposure = args.coarse_zoom

	# Now we figure out how many images will need to be taken to perform the coarse scan.
	n_coarse_images = int(round(1.55 * scan_area / (coarse_w * coarse_h)))
	disk_size       = (n_coarse_images * 5601754) / (1024 * 1024) 
	print("This will produce roughly %d images and occupy %4.2f MB of disk space."%(
		n_coarse_images, disk_size
	))

	if input("Proceed (y/n)? ").lower() != 'y':
		exit()

	# Connect to the microscope hardware.
	microscope = MicroscopeController(verbose=True)
	microscope.camera.startCapture()
	microscope.stage.setMaxSpeed(7.5, 7.5)

	# Now we perform the focus calibration for the coarse scan. 
	microscope.camera.setExposure(coarse_exposure)
	microscope.focus.setZoom(coarse_zoom)
	interp = calibrateFocus(microscope, args)

	# Start a coarse scan. We'll process each image as we acquire it and quickly decide whether or
	# not the region warrants further inspection.
	coarse_progress = ProgressBar("Coarse Scan", 12, n_coarse_images, 1, ea=20)

	regions_of_interest  = []
	image_number         = 0
	x_current, y_current = x_min, y_min

	microscope.stage.moveTo(x_current, y_current)
	microscope.focus.setFocus(
		interp((x_current, y_current)),
		corrected=True
	)

	def avgimg(n, ds):
		imgs = []
		for i in range(n):
			img = microscope.camera.getFrame(downscale=ds)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			imgs.append(img)

		base = np.zeros(imgs[0].shape)
		for i in imgs:
			base += i

		return (base / n).astype(np.uint8)

	while x_current < x_max + coarse_w:
		while y_current < y_max + coarse_h:
			x, y = microscope.stage.getPosition()
			img  = avgimg(args.coarse_averages, args.coarse_downscale)
			img  = parabolicSubtract(img)
			
			# This function will return the coordinates of all of the regions within this image that
			# contain potentially interesting flakes.
			regions_of_interest.extend(getRegionsOfInterest(img, args, x, y))

			y_current += coarse_h
			microscope.stage.moveTo(x_current, y_current)
			microscope.focus.setFocus(
				interp((x_current, y_current)), 
				corrected=args.quality_focus
			)
			image_number += 1
			coarse_progress.update(image_number)
		y_current  = y_min
		x_current += coarse_w
		microscope.stage.moveTo(x_current, y_current)
		microscope.focus.setFocus(
			interp((x_current, y_current)),
			corrected=True
		)

	coarse_progress.finish()

	microscope.focus.setZoom(fine_zoom)
	microscope.camera.setExposure(fine_exposure)
	interp = calibrateFocus(microscope, args)

	fine_progress = ProgressBar("Fine Scan", 12, len(regions_of_interest), 1, ea=20)

	

	def avgimg(n):
		images = []
		for i in range(n):
			images.append(microscope.camera.getFrame(convert=False))
		r, g, b = [], [] ,[]
		for i in images:
			b.append(i[:, :, 0])
			g.append(i[:, :, 1])
			r.append(i[:, :, 2])

		red   = np.stack(r, axis=2).mean(axis=2)
		green = np.stack(g, axis=2).mean(axis=2)
		blue  = np.stack(b, axis=2).mean(axis=2)
		img   = np.stack([b, g, r], axis=2).astype(np.uint8)
		return img

	# We've finished the coarse scan. Now we'll zoom into the regions of interest that we found. 
	for idx, (x, y) in enumerate(regions_of_interest):
		microscope.stage.moveTo(x, y)
		microscope.focus.setFocus(
			interp((x, y)),
			corrected=True
		)
		img = avgimg(args.fine_averages)

		filename = os.path.join(
			args.output_directory,
			"%04d_%2.4f_%2.4f.png"%(idx, x, y)
		)



		cv2.imwrite(filename, img)

		fine_progress.update(idx + 1)

	fine_progress.finish()