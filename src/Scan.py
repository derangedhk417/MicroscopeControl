import matplotlib.pyplot as plt
import numpy             as np
import sys
import time
import code
import argparse
import os
import cv2
import json
import math

#from MicroscopeControl  import MicroscopeController
from scipy.optimize     import curve_fit
from matplotlib.patches import Rectangle
from Progress           import ProgressBar

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
	try:
		with open("Scan.json", 'r') as file:
			args_specification = json.loads(file.read())
	except Exception as ex:
		raise ex

	args = preprocess(args_specification)

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
	
	n_squares = 0
	n_images  = 0

	microscope.stage.setMaxSpeed(7.5, 7.5)

	scanned_locations = []

	# Iterate over the locations.
	for (x, y), (xh, yh) in zip(locations, extents):
		# First we calibrate by focusing at several random
		# points in the square.
		xlow  = x
		xhigh = xh
		x_r   = np.random.uniform(xlow, xhigh, args.n_focus)

		ylow  = y
		yhigh = yh
		y_r   = np.random.uniform(ylow, yhigh, args.n_focus)

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

		microscope.focus.setFocus(focal_points[0])

		# Now we image the square.
		x_current = x
		y_current = y
		microscope.stage.moveTo(x_current, y_current)

		while x_current < xhigh:
			while y_current < yhigh:
				# Take an image
				xc, yc = microscope.stage.getPosition()
				img    = microscope.camera.getFrame(convert=False)

				scanned_locations.append([xc, yc])

				fstr  = '%06d_%2.5f_%2.5f.png'%(
					n_images, xc, yc
				)
				n_images += 1
				fname = os.path.join(args.output_directory, fstr)

				decimated = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
				cv2.imshow('Scan Preview', decimated)
				cv2.waitKey(1)

				cv2.imwrite(fname, img)
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
				corrected=False
			)

		n_squares += 1


	scanned_locations = np.array(scanned_locations)
	plt.scatter(scanned_locations[:, 0], scanned_locations[:, 1], s=1)
	plt.title("top left corner of all scanned locations on the sample")
	plt.show()

	end = time.time_ns()

	duration = ((end - start) / 1e9) / 60
	print("Scan took %d minutes"%(int(duration)))

	microscope.camera.endCapture()
