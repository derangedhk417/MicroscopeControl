# Author:      Adam Robinson
# Description: This program is similar to Scan.py, except that it performs a scan of the substrate
#              at a much lower zoom in order to determine which regions contain flakes. It then
#              zooms in on only those regions and takes high quality images of the flakes in them.

# TODO: Use the four corners for focusing when focus points not specified.


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
import matplotlib.patches as patches

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

def calibrateFocus(microscope, args, gfit=False, debug=False):
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
			args.autofocus_parameters[2],
			debug=debug,
			gfit=gfit
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

	return interp, res[0], res[1]

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
	cv2.waitKey(1)

	# We want to calculate the contrast of every pixel in the image and then subdivide it into 
	# regions with the same size as an image at the fine zoom setting. We'll return an array of
	# points corresponding to the top left corner of any such region that meets the criteria
	# specified in the arguments to this program (negative_contrast, threshold_ratio).
	
	background = max(np.argmax(np.bincount(img.flatten())), 1)
	img        = img.astype(np.float32)
	contrast   = (background - img) / background


	mm_per_pixel_x = coarse_w / contrast.shape[1]
	mm_per_pixel_y = coarse_h / contrast.shape[0]
	mm_per_pixel   = (mm_per_pixel_x + mm_per_pixel_y) / 2

	# Now we iterate over the above mentioned regions of this image and calculate the percentage
	# of each image that has contrast greater than zero.
	x, y    = 0, 0
	regions = []
	pixel_coordinates = [] # DEBUG
	while x < coarse_w - (fine_w / 2):
		while y < coarse_h - (fine_h / 2):
			# We need to convert coordinates into indices into the image array so that we can select
			# the correct subset of pixels for the calculation.
			y_low  = int(round(y  / mm_per_pixel))
			y_high = int(round((y + fine_h) / mm_per_pixel)) + 1

			x_low  = int(round(x  / mm_per_pixel))
			x_high = int(round((x + fine_w) / mm_per_pixel)) + 1

			subimage   = contrast[y_low:y_high, x_low:x_high]
			relevant   = (subimage >= args.contrast_range[0]) & (subimage <= args.contrast_range[1])
			r          = subimage.copy()

			# The halo effect around flakes causes problems when attempting to identify flake 
			# thickness. The halo around a flake tends to smoothly transition between the optical
			# contrast of the flake itself and the substrate. This results in 1-5 pixels around the
			# flake that appear to have lower optical contrast but aren't actually real. The erode
			# and dilate process below tends to eliminate the halo pixels that contain incorrect 
			# contrast values.
			r[relevant]          = 1.0
			r[relevant == False] = 0.0
			ksize   = 4
			kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
			eroded  = cv2.erode(r, kernel)
			dilated = cv2.dilate(eroded, kernel)
			ratio   = dilated.sum() / (subimage.shape[0] * subimage.shape[1])


			if ratio > args.threshold_ratio:
				pixel_coordinates.append([x_low, x_high, y_low, y_high]) # DEBUG
				x_shift = (coarse_w - fine_w) / 2
				y_shift = (coarse_h - fine_h) / 2
				regions.append([x + x0 - x_shift, y + y0 - y_shift])
				# code.interact(local=locals())

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
			3
		)
	# plt.imshow(timg)
	# plt.show()
	_min = timg.min()
	_max = timg.max()
	timg  = (((timg - _min) / (_max - _min)) * 255).astype(np.uint8)
	pv = cv2.resize(timg, (0, 0), fx=0.3, fy=0.3)
	cv2.imshow('Scan Preview', pv)
	cv2.waitKey(1)
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

def calculateBackgroundGreyscale(args, focus_interp, microscope):
	x_min, x_max, y_min, y_max = args.bounds
	# After initial testing I realized that this system produces a background that cannot be 
	# approximated as a simple function when the zoom is not very close to being maxed out. As a 
	# result, it's necessary to determine this background before taking serious images. In order
	# to do this, we take many images at random positions, perform a rolling ball background 
	# subtraction and then take an average of them to get the background. The rolling ball 
	# algorithm tends to produce some irregularities around flakes, so we won't take a regular 
	# average. Instead we will take the standard deviation of each pixel across images and throw
	# out values that are far outside of the mean.
	n_background_images = args.n_background_images
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

	imgs  = np.stack(images, axis=2)
	means = imgs.mean(axis=2)
	stds  = imgs.std(axis=2)

	weights = []
	for img in images:
		mask         = np.abs(img - means) < 1.2 * stds
		weight       = np.ones(mask.shape) * 0.001
		weight[mask] = 1.0
		weights.append(weight) 

	weights    = np.stack(weights, axis=2)
	background = np.average(imgs, axis=2, weights=weights)

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.imshow(images[0])
	# ax2.imshow(background)
	# ax3.imshow(images[0] - background)
	# plt.show()
	# plotImage(background)
	# code.interact(local=locals())
	return background

def calculateBackgroundColored(args, focus_interp, microscope):
	x_min, x_max, y_min, y_max = args.bounds
	# After initial testing I realized that this system produces a background that cannot be 
	# approximated as a simple function when the zoom is not very close to being maxed out. As a 
	# result, it's necessary to determine this background before taking serious images. In order
	# to do this, we take many images at random positions, perform a rolling ball background 
	# subtraction and then take an average of them to get the background. The rolling ball 
	# algorithm tends to produce some irregularities around flakes, so we won't take a regular 
	# average. Instead we will take the standard deviation of each pixel across images and throw
	# out values that are far outside of the mean.
	n_background_images = args.n_background_images
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
		images.append(avgimg(args.fine_averages))
		bg_img_progress.update(i + 1)

	imgs_r_arr = [i[:, :, 2] for i in images]
	imgs_g_arr = [i[:, :, 1] for i in images]
	imgs_b_arr = [i[:, :, 0] for i in images]
	imgs_r = np.stack(imgs_r_arr, axis=2)
	imgs_g = np.stack(imgs_g_arr, axis=2)
	imgs_b = np.stack(imgs_b_arr, axis=2)
	means_r = imgs_r.mean(axis=2)
	means_g = imgs_g.mean(axis=2)
	means_b = imgs_b.mean(axis=2)
	stds_r  = imgs_r.std(axis=2)
	stds_g  = imgs_g.std(axis=2)
	stds_b  = imgs_b.std(axis=2)

	weights_r = []
	weights_g = []
	weights_b = []
	for img_r, img_g, img_b in zip(imgs_r_arr, imgs_g_arr, imgs_b_arr):
		mask_r       = np.abs(img_r - means_r) < 1.2 * stds_r
		mask_g       = np.abs(img_g - means_g) < 1.2 * stds_g
		mask_b       = np.abs(img_b - means_b) < 1.2 * stds_b

		weight_r       = np.ones(mask_r.shape) * 0.001
		weight_r[mask_r] = 1.0
		weights_r.append(weight_r) 

		weight_g       = np.ones(mask_g.shape) * 0.001
		weight_g[mask_g] = 1.0
		weights_g.append(weight_g) 

		weight_b       = np.ones(mask_b.shape) * 0.001
		weight_b[mask_b] = 1.0
		weights_b.append(weight_b) 

	weights_r  = np.stack(weights_r, axis=2)
	background_r = np.average(imgs_r, axis=2, weights=weights_r)

	weights_g  = np.stack(weights_g, axis=2)
	background_g = np.average(imgs_g, axis=2, weights=weights_g)

	weights_b  = np.stack(weights_b, axis=2)
	background_b = np.average(imgs_b, axis=2, weights=weights_b)

	background = np.stack([background_b, background_g, background_r], axis=2).astype(np.uint8)

	#code.interact(local=locals())

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.imshow(images[0])
	# ax2.imshow(background)
	# ax3.imshow(images[0] - background)
	# plt.show()
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

	if args.focus_points is None:
		args.focus_points = [
			x_min, y_min,
			x_min, y_max,
			x_max, y_min,
			x_max, y_max,
			(x_min + x_max) / 2, (y_min + y_max) / 2
		]

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
	n_coarse_images = int(round(1.86 * scan_area / (coarse_w * coarse_h)))
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
	interp, mx, my = calibrateFocus(microscope, args, debug=False, gfit=False)

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

	coarse_background = calculateBackgroundGreyscale(args, interp, microscope)

	
	image_number         = 0
	x_current, y_current = x_min, y_min

	

	# Here we figure out if the scan area is larger on the x or y axis. We can make the scan faster
	# if the inner most loop goes over the largest axis.

	
	if x_max - x_min > y_max - y_min:
		swap = True
		outer_current = y_current
		outer_max     = y_max
		outer_width   = coarse_h
		outer_reset   = x_min

		inner_current = x_current
		inner_max     = x_max
		inner_width   = coarse_w

	else:
		swap = False
		outer_current = x_current
		outer_max     = x_max
		outer_width   = coarse_w
		outer_reset   = y_min

		inner_current = y_current
		inner_max     = y_max
		inner_width   = coarse_h

	# The focus motor performs better when it steps down across the sample instead of up. Here
	# we'll determine the slope of the sample in the chosen inner sweep direction and make sure
	# to move in the direction with a downward slope.
	sign = 1.0
	if swap:
		# inner sweep direction is along the x-axis
		if mx < 0:
			print("Sample slopes upwards, reversing direction.")
			# The focus moves upwards in the positive x-direction, we need to move in the opposite
			# direction.
			outer_reset   = x_max
			sign          = -1.0
			inner_current = x_max
			def inner_condition():
				return inner_current > x_min
		else:
			def inner_condition():
				return inner_current < inner_max + inner_width
	else:
		if my < 0:
			print("Sample slopes upwards, reversing direction.")
			outer_reset   = y_max
			sign          = -1.0
			inner_current = y_max
			def inner_condition():
				return inner_current > y_min
		else:
			def inner_condition():
				return inner_current < inner_max + inner_width
				

	if swap:
		microscope.stage.moveTo(inner_current, outer_current)
		microscope.focus.setFocus(
			interp((inner_current, outer_current)), 
			corrected=True
		)
	else:
		microscope.stage.moveTo(outer_current, inner_current)
		microscope.focus.setFocus(
			interp((outer_current, inner_current)), 
			corrected=True
		)

	# Start a coarse scan. We'll process each image as we acquire it and quickly decide whether or
	# not the region warrants further inspection.
	coarse_progress = ProgressBar("Coarse Scan", 18, n_coarse_images, 1, ea=20)

	regions_of_interest  = []

	while outer_current < outer_max + outer_width:
		row_regions_of_interest = []
		while inner_condition():
			x, y = microscope.stage.getPosition()
			img  = avgimg(args.coarse_averages, args.coarse_downscale)
			pv   = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
			cv2.imshow('Scan Preview', pv)
			cv2.waitKey(1)
			img  = img - coarse_background
			
			# This function will return the coordinates of all of the regions within this image that
			# contain potentially interesting flakes.
			row_regions_of_interest.extend(getRegionsOfInterest(img, args, x, y))

			inner_current += sign*inner_width
			if swap:
				microscope.stage.moveTo(inner_current, outer_current)
				microscope.focus.setFocus(
					interp((inner_current, outer_current)), 
					corrected=args.quality_focus
				)
			else:
				microscope.stage.moveTo(outer_current, inner_current)
				microscope.focus.setFocus(
					interp((outer_current, inner_current)), 
					corrected=args.quality_focus
				)
			image_number += 1
			coarse_progress.update(image_number)

		regions_of_interest.append(row_regions_of_interest)
		inner_current  = outer_reset
		outer_current += outer_width
		if swap:
			microscope.stage.moveTo(inner_current, outer_current)
			microscope.focus.setFocus(
				interp((inner_current, outer_current)),
				corrected=True
			)
		else:
			microscope.stage.moveTo(outer_current, inner_current)
			microscope.focus.setFocus(
				interp((outer_current, inner_current)),
				corrected=True
			)

	coarse_progress.finish()

	# Plot every region that was identified as interesting.
	x_vals = [i[0] for i in regions_of_interest]
	y_vals = [i[1] for i in regions_of_interest]

	fig, ax = plt.subplots(1, 1)
	ax.scatter(x_vals, y_vals, s=1)
	plt.show()
	# for x, y in regions_of_interest:
	# 	rect = patches.Rectangle(
	# 		(50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none'
	# 	)


	microscope.focus.setZoom(fine_zoom)
	microscope.camera.setExposure(fine_exposure)
	interp, mx, my = calibrateFocus(microscope, args, debug=False)

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
		img   = np.stack([blue, green, red], axis=2).astype(np.uint8)
		return img

	# Turning off background subtraction.
	# fine_background = calculateBackgroundColored(args, interp, microscope).astype(np.float32)

	# We've finished the coarse scan. Now we'll zoom into the regions of interest that we found. 
	for row in regions_of_interest:
		x0, y0 = row[0]
		microscope.stage.moveTo(x0, y0)
		microscope.focus.setFocus(
			interp((x0, y0)),
			corrected=True
		)
		for idx, (x, y) in enumerate(row):
			microscope.stage.moveTo(x, y)
			microscope.focus.setFocus(
				interp((x, y)),
				corrected=args.quality_focus
			)
			#code.interact(local=locals())
			# Turning off background subtraction for now. It isn't necessary.
			img = avgimg(args.fine_averages) # .astype(np.float32)
			# img = img - fine_background
			# img_min, img_max = img.min(), img.max()
			# img = (((img - img_min) / (img_min - img_max)) * 255).astype(np.uint8)

			#code.interact(local=locals())

			filename = os.path.join(
				args.output_directory,
				"%04d_%2.4f_%2.4f.png"%(idx, x, y)
			)

			# cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			cv2.imwrite(filename, img)

			fine_progress.update(idx + 1)

	fine_progress.finish()