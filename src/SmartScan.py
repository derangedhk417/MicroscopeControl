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
from skimage            import data, restoration, util

from mpl_toolkits.mplot3d import Axes3D

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

	

	# This image will have had the background subtracted and will be float32 as a result. 
	# We want to convert it back to a uint8 array.
	_min = img.min()
	_max = img.max()
	img  = (((img - _min) / (_max - _min)) * 255).astype(np.uint8)

	pv = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
	cv2.imshow('Scan Preview', pv)
	cv2.waitKey(250)

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
	pixel_coordinates = [] # DEBUG
	while x < coarse_w:
		while y < coarse_h:
			# We need to convert coordinates into indices into the image array so that we can select
			# the correct subset of pixels for the calculation.
			y_low  = int(round(y  / mm_per_pixel))
			y_high = int(round((y + fine_h) / mm_per_pixel)) + 1

			x_low  = int(round(x  / mm_per_pixel))
			x_high = int(round((x + fine_w) / mm_per_pixel)) + 1

			subimage = contrast[y_low:y_high, x_low:x_high]
			ratio    = subimage[subimage > 0].sum() / (subimage.shape[0] * subimage.shape[1])

			if ratio > args.threshold_ratio:
				pixel_coordinates.append([x_low, x_high, y_low, y_high]) # DEBUG
				regions.append([x + x0, y + y0])

			y += fine_h
		y = 0
		x += fine_w

	# DEBUG
	timg = contrast.copy()
	rect_color = contrast.max() * 2
	for rect in pixel_coordinates:
		timg = cv2.rectangle(
			timg, 
			(rect[0], rect[2]), 
			(rect[1], rect[3]), 
			(rect_color, 0, 0), 
			1
		)
	# plt.imshow(timg)
	# plt.show()
	_min = timg.min()
	_max = timg.max()
	timg  = (((timg - _min) / (_max - _min)) * 255).astype(np.uint8)
	pv = cv2.resize(timg, (0, 0), fx=0.3, fy=0.3)
	cv2.imshow('Scan Preview', pv)
	cv2.waitKey(250)
	# END DEBUG

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

	code.interact(local=locals())

	return subtracted

# Used for debugging purposes. Creates a 3d surface plot of an image.
def plotImage(img, ds=20):
	img = cv2.resize(img, (0, 0), fx=(1 / ds), fy=(1 / ds))
	xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

	fig = plt.figure()
	ax  = fig.gca(projection='3d')
	ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
	plt.show()

def calculateBackground(args, focus_interp, microscope):
	x_min, x_max, y_min, y_max = args.bounds
	# After initial testing I realized that this system produces a background that cannot be 
	# approximated as a simple function when the zoom is not very close to being maxed out. As a 
	# result, it's necessary to determine this background before taking serious images. In order
	# to do this, we take many images at random positions, perform a rolling ball background 
	# subtraction and then take an average of them to get the background. The rolling ball 
	# algorithm tends to produce some irregularities around flakes, so we won't take a regular 
	# average. Instead we will take the standard deviation of each pixel across images and throw
	# out values that are far outside of the mean.
	n_background_images = 20
	bg_img_progress = ProgressBar("Background Images", 18, n_background_images, 1, ea=10)
	positions           = np.array([
		np.random.uniform(x_min, x_max, n_background_images), 
		np.random.uniform(y_min, y_max, n_background_images)
	])
	images = []
	for i, (x, y) in enumerate(positions.T):
		microscope.stage.moveTo(x, y)
		microscope.focus.setFocus(
			focus_interp((x, y)),
			corrected=False
		)
		images.append(avgimg(args.coarse_averages, args.coarse_downscale))
		bg_img_progress.update(i + 1)

	bg_img_progress.finish()

	bg_calc_progress = ProgressBar("Background Calc", 18, n_background_images, 1, ea=10)
	backgrounds      = []
	for i, img in enumerate(images):
		inv = util.invert(img)
		bg  = restoration.rolling_ball(inv, radius=15)
		bg  = util.invert(bg)
		backgrounds.append(bg)
		bg_calc_progress.update(i + 1)

	bg_calc_progress.finish()

	bgs   = np.stack(backgrounds, axis=2)
	means = bgs.mean(axis=2)
	stds  = bgs.std(axis=2)

	weights = []
	for img in backgrounds:
		mask         = np.abs(img - means) < 1.2 * stds
		weight       = np.ones(mask.shape) * 0.01
		weight[mask] = 1.0
		weights.append(weight) 

	weights    = np.stack(weights, axis=2)
	background = np.average(bgs, axis=2, weights=weights)

	# Now we background subtract all of the images and try to perform the same process to remove
	# any background that wasn't picked up by rolling ball.
	subbed = []
	for img in images:
		subbed.append(img - background)

	subbed_s = np.stack(subbed, axis=2)
	means    = subbed_s.mean(axis=2)
	stds     = subbed_s.std(axis=2)

	weights = []
	for img in subbed:
		mask         = np.abs(img - means) < 1.2 * stds
		weight       = np.ones(mask.shape) * 0.01
		weight[mask] = 1.0
		weights.append(weight) 

	weights      = np.stack(weights, axis=2)
	background_s = np.average(subbed_s, axis=2, weights=weights)
	background   = background + background_s

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.imshow(images[0])
	# ax2.imshow(background)
	# ax3.imshow(images[0] - background)
	# plt.show()
	# plotImage(background)
	# code.interact(local=locals())
	return background

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

	coarse_background = calculateBackground(args, interp, microscope)

	# Start a coarse scan. We'll process each image as we acquire it and quickly decide whether or
	# not the region warrants further inspection.
	coarse_progress = ProgressBar("Coarse Scan", 18, n_coarse_images, 1, ea=20)

	regions_of_interest  = []
	image_number         = 0
	x_current, y_current = x_min, y_min

	microscope.stage.moveTo(x_current, y_current)
	microscope.focus.setFocus(
		interp((x_current, y_current)),
		corrected=True
	)

	while x_current < x_max + coarse_w:
		while y_current < y_max + coarse_h:
			x, y = microscope.stage.getPosition()
			img  = avgimg(args.coarse_averages, args.coarse_downscale)
			pv   = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
			cv2.imshow('Scan Preview', pv)
			cv2.waitKey(250)
			img  = img - coarse_background
			
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

	fine_progress = ProgressBar("Fine Scan", 18, len(regions_of_interest), 1, ea=20)

	

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