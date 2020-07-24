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
from ImageProcessor     import ImageProcessor
from ScanProcessing     import processFile
from datetime           import datetime

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

def subdivide(args):
	# First, we'll create an array of locations corresponding to the
	# top-left of each square we are going to scan.
	locations = []
	extents   = []
	x   = args.x_limits[0]
	xl  = args.x_limits[1] - args.x_limits[0]
	y   = args.y_limits[0]
	yl  = args.y_limits[1] - args.y_limits[0]
	n_x = math.ceil(xl / args.square_size)
	n_y = math.ceil(yl / args.square_size)

	for i in range(n_x):
		for j in range(n_y):
			loc = [
				x + args.square_size*i,
				y + args.square_size*j
			]
			locations.append(loc)
			extent = [
				min(loc[0] + args.square_size, args.x_limits[1]),
				min(loc[1] + args.square_size, args.y_limits[1]),
			]
			extents.append(extent)


	locations = np.array(locations)
	extents   = np.array(extents)

	return locations, extents

def calibrate_square(xrng, yrng, microscope):
	xlow  = xrng[0]
	xhigh = xrng[1]

	ylow  = yrng[0]
	yhigh = yrng[1]

	points = np.array([
		[xlow, ylow],
		[xlow, yhigh],
		[xhigh, ylow],
		[xhigh, yhigh]
	])

	print("Performing focus calibration for square %d"%n_squares)
	focal_points = []
	for xp, yp in points:
		print("Position: %f %f"%(xp, yp))
		microscope.stage.moveTo(xp, yp)
		microscope.autoFocus(args.focus_range)
		focal_points.append(microscope.focus.getFocus())

	focal_points = np.array(focal_points)

	def fit(X, mx, my, b):
		x, y = X
		return mx*x + my*y + b

	res, cov = curve_fit(fit, points.T, focal_points)

	def interp(X):
		return res[0]*X[0] + res[1]*X[1] + res[2]

	rmse = np.sqrt(np.square(interp(points.T) - focal_points).mean())
	print("The focal points have been fit to a function of the form z = ax + by + c.")
	print("RMSE of fit: %f"%rmse)

	if rmse > 0.002:
		print("Fit too poor to continue.")
		print("Please ensure that the corners of the ", end='')
		print("scan region contain something to focus on.")
		exit()

	print("Focal Points")
	for p in focal_points:
		print(p)

	return focal_points, interp

if __name__ == '__main__':
	# Load the arguments file. 
	with open("Scan.json", 'r') as file:
		args_specification = json.loads(file.read())

	args = preprocess(args_specification)

	# Setup a structure to write meta information into when
	# scanning.

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

	# Sanity check the parameters of the scan and print the stats to the user.
	if args.x_limits[0] >= args.x_limits[1]:
		print("Your x-limits are nonsensical.")
		exit()

	if args.y_limits[0] >= args.y_limits[1]:
		print("Your y-limits are nonsensical.")
		exit()


	scan_area  = (args.x_limits[1] - args.x_limits[0])
	scan_area *= (args.y_limits[1] - args.y_limits[0])

	meta_data['scan_area'] = scan_area

	print("Preparing to scan the region:")
	print("    x = %02.4f to %02.4f mm"%(args.x_limits[0], args.x_limits[1]))
	print("    y = %02.4f to %02.4f mm"%(args.y_limits[0], args.y_limits[1]))
	print("    w = %02.4f mm"%args.image_width)
	print("    h = %02.4f mm"%args.image_height)
	print(" area = %04.4f mm²"%scan_area)

	# Determine how many images this will produce.
	n_images  = int(round(scan_area / (args.image_width * args.image_height)))
	disk_size = (n_images * 5601754) / (1024 * 1024) 
	print("This will produce roughly %d images and occupy %4.2f MB of disk space."%(
		n_images, disk_size
	))

	if input("Proceed (y/n)? ").lower() != 'y':
		exit()

	# Now that the output directory is ready, we attempt to connect to the 
	# microscope hardware.

	microscope = MicroscopeController(verbose=True)

	y_current = args.y_limits[0]
	x_current = args.x_limits[0]
	n_images  = 1

	microscope.stage.moveTo(x_current, y_current)

	print("Beginning scan.")
	start = time.time_ns()
	
	microscope.camera.startCapture()

	# Divide the search area up into squares. We will focus on 
	# some number of random points in a square and take the average
	# focus and use that for the whole square.
	locations, extents = subdivide(args)

	meta_data['squares'] = {
		"locations" : locations.tolist(),
		"extents"   : extents.tolist()
	}
	
	n_squares = 0
	n_images  = 0

	microscope.stage.setMaxSpeed(7.5, 7.5)

	scanned_locations = []

	# Here we initialize a process pool to delegate image processing to.
	if not args.dont_process:
		process_pool = Pool(args.n_processes)

	statuses = []

	meta_data['image_files'] = []

	# Iterate over the locations.
	for (x, y), (xh, yh) in zip(locations, extents):
		# First we calibrate by focusing at several random
		# points in the square.
		focal_points, interp = calibrate_square(
			[x, xh], 
			[y, yh],
			microscope
		)

		microscope.focus.setFocus(focal_points[0])

		# Now we image the square.
		x_current = x
		y_current = y
		microscope.stage.moveTo(x_current, y_current)

		
		# Estimate the number of images necessary for this square.
		n_total = int(((yh - y) * (xh - x)) / (args.image_height * args.image_width))
		n_total = int(n_total * 1.1)

		# This progress bar will update every time an image is taken and will
		# attempt to estimate the total time remaining after 30 seconds.
		pb1     = ProgressBar("Imaging Square (%d)"%n_squares, 30, n_total, 1, ea=120)
		img_idx = 0

		while x_current < xh:
			while y_current < yh:
				# Take an image
				xc, yc = microscope.stage.getPosition()
				img    = microscope.camera.getFrame(convert=False)

				scanned_locations.append([xc, yc])

				fstr  = '%06d_%2.5f_%2.5f.png'%(
					n_images, xc, yc
				)

				n_images += 1
				img_idx  += 1
				pb1.update(img_idx)
				fname = os.path.join(args.output_directory, fstr)
				meta_data['image_files'].append({
					'path'     : fstr,
					'position' : [xc, yc]	
				})

				decimated = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
				cv2.imshow('Scan Preview', decimated)
				cv2.waitKey(1)

				cv2.imwrite(fname, img)

				if not args.dont_process:
					# Delegate this to a member of the process pool.
					stat = process_pool.apply_async(processFile, (img, fname, args))
					statuses.append(stat)

				y_current += args.image_height
				microscope.stage.moveTo(x_current, y_current)
				microscope.focus.setFocus(
					interp((x_current, y_current)), 
					corrected=False
				)

			y_current = y
			x_current += args.image_width
			microscope.stage.moveTo(x_current, y_current)
			microscope.focus.setFocus(
				interp((x_current, y_current)), 
				corrected=True
			)
		pb1.finish()

		n_squares += 1

	running = False

	scanned_locations = np.array(scanned_locations)
	plt.scatter(scanned_locations[:, 0], scanned_locations[:, 1], s=1)
	plt.title("top left corner of all scanned locations on the sample")
	plt.show()

	end = time.time_ns()

	if not args.dont_process:
		# Wait for all workers to finish.
		while True:
			done = True
			for s in statuses:
				if not s.ready():
					done = False
					time.sleep(0.01)

			if done:
				break

	duration = ((end - start) / 1e9) / 60
	print("Scan took %d minutes"%(int(duration)))

	meta_data['end_time'] = str(datetime.now())

	with open(os.path.join(args.output_directory, "_scan.json"), 'w') as file:
		file.write(json.dumps(meta_data))

	microscope.camera.endCapture()
