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

from MicroscopeControl import MicroscopeController

# Process the command line arguments supplied to the program.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Scan a specified area of a sample and save images to a ' +
		'specified directory.'
	)

	parser.add_argument(
		'-o', '--output', dest='output_directory', type=str, required=True,
		help='The directory to write output files to. It will be created if ' +
		'it does not exist. This program will not write into a folder that is ' +
		'not empty.' 
	)

	parser.add_argument(
		'-x', '--x-limits', dest='x_limits', nargs=2, type=float, required=True,
		help='The minimum and maximum x-coordinates to the rectangle to sweep.' 
	)

	parser.add_argument(
		'-y', '--y-limits', dest='y_limits', nargs=2, type=float, required=True,
		help='The minimum and maximum y-coordinates to the rectangle to sweep.' 
	)

	parser.add_argument(
		'-fr', '--focus-range', dest='focus_range', nargs=2, type=float, default=[0.4, 0.6],
		help='The range of values to sweep when focusing the camera.' 
	)

	parser.add_argument(
		'-W', '--width', dest='image_width', type=float, required=True,
		help='The width (along x-direction) of the image at the current zoom (mm).' 
	)

	parser.add_argument(
		'-H', '--height', dest='image_height', type=float, required=True,
		help='The height (along y-direction) of the image at the current zoom (mm).' 
	)

	parser.add_argument(
		'-s', '--square-size', dest='square_size', type=float, required=True,
		help='The length in millimeters of the square subdivisions to use.' 
	)

	parser.add_argument(
		'-n', '--n-focus', dest='n_focus', type=int, required=True,
		help='The number of random positions in each square to focus on when calibrating.' 
	)	

	parser.add_argument(
		'-fps', '--frames-per-second', dest='fps', type=int, default=32,
		help='Assume the camera captures this many frames per second when setting stage velocity.' 
	)

	parser.add_argument(
		'-pi', '--preview-interval', dest='preview_interval', type=int, default=4,
		help='Show a preview image at integer multiples of this number.' 
	)


	args = parser.parse_args()

	return args

if __name__ == '__main__':
	args = preprocess()

	# Check the output directory.
	if not os.path.exists(args.output_directory):
		os.mkdir(args.output_directory)
	else:
		for f in os.listdir(args.output_directory):
			p = os.path.join(args.output_directory, f)
			if os.path.isfile(p):
				print("The specified output directory is not empty.")
				exit()
	

	# Now that the output directory is ready, we attempt to connect to the 
	# microscope hardware.

	microscope = MicroscopeController(verbose=True)

	# Sanity check the parameters of the scan and print the stats to the user.
	if args.x_limits[0] >= args.x_limits[1]:
		print("Your x-limits are nonsensical.")
		exit()

	if args.y_limits[0] >= args.y_limits[1]:
		print("Your y-limits are nonsensical.")
		exit()


	scan_area =  (args.x_limits[1] - args.x_limits[0])
	scan_area *= (args.y_limits[1] - args.y_limits[0])

	print("Preparing to scan the region:")
	print("    x = %02.4f to %02.4f mm"%(args.x_limits[0], args.x_limits[1]))
	print("    y = %02.4f to %02.4f mm"%(args.y_limits[0], args.y_limits[1]))
	print("    w = %02.4f mm"%args.image_width)
	print("    h = %02.4f mm"%args.image_height)
	print(" area = %04.4f mm²"%scan_area)

	# Determine how many images this will produce.
	n_images  = int(round(scan_area / (args.image_width * args.image_height)))
	disk_size = (n_images * (2448 * 2048 * 3)) / (1024 * 1024) 
	print("This will produce roughly %d images and occupy %4.2f MB of disk space."%(
		n_images, disk_size
	))

	if input("Proceed (y/n)? ").lower() != 'y':
		exit()

	y_current = args.y_limits[0]
	x_current = args.x_limits[0]
	n_images  = 1

	microscope.focus.setZoom(0.99)
	microscope.stage.moveTo(x_current, y_current)

	# Based on the assumption that the camera is getting frame_rate
	# frames per second, and an image is args.image_height by 
	# args.image_width, we need to determine an appropriate scan
	# speed on each axis. This will inevitably result in some overlap
	# between images, but that is better than missing data.

	# The default value for fps is slightly lower than the actual one,
	# to ensure that we don't miss any part of the slide.
	x_velocity = min(args.image_width  * args.fps, 7.45)
	y_velocity = min(args.image_height * args.fps, 7.45)

	print("Setting velocity to vx, vy = %1.3f mm/s, %1.3f mm/s"%(
		x_velocity, y_velocity
	))
	microscope.stage.setMaxSpeed(x_velocity, y_velocity)

	print("Beginning scan.")
	start = time.time_ns()
	
	microscope.camera.startCapture()

	# Divide the search area up into squares. We will focus on 
	# some number of random points in a square and take the average
	# focus and use that for the whole square.

	# First, we'll create an array of locations corresponding to the
	# top-left of each square we are going to scan.
	locations = []
	x   = args.x_limits[0]
	xl  = args.x_limits[1] - args.x_limits[0]
	y   = args.y_limits[0]
	yl  = args.y_limits[1] - args.y_limits[0]
	n_x = math.ceil(xl / args.square_size)
	n_y = math.ceil(yl / args.square_size)

	for i in range(n_x):
		for j in range(n_y):
			locations.append([
				x + args.square_size*i,
				y + args.square_size*j
			])

	print("Sample will be broken down into %d square regions."%len(locations))

	n_squares = 0
	n_images  = 0

	scanned_locations = []

	# Iterate over the locations.
	for x, y in locations:
		print("Square %d / %d"%(n_squares + 1, len(locations)))
		# First we calibrate by focusing at several random
		# points in the square.
		xlow  = x
		xhigh = x + args.square_size
		x_r   = np.random.uniform(xlow, xhigh, args.n_focus)

		ylow  = y
		yhigh = y + args.square_size
		y_r   = np.random.uniform(ylow, yhigh, args.n_focus)

		print("Performing focus calibration for square")
		focal_points = []
		for xr, yr in zip(x_r, y_r):
			print("Position: %f %f"%(xr, yr))
			microscope.stage.moveTo(xr, yr)
			microscope.autoFocus(args.focus_range)
			focal_points.append(microscope.focus.getFocus())

		print(focal_points)
		focus = np.array(focal_points).mean()
		print("Selecting focus: %f"%focus)
		microscope.focus.setFocus(focus)

		# Now we image the square.
		x_current = x
		y_current = y
		microscope.stage.moveTo(x_current, y_current)

		while x_current < xhigh:
			microscope.stage.moveTo(x_current, yhigh, return_immediately=True)

			n_captured = 0
			while True:
				# Take an image
				xc, yc = microscope.stage.getPosition()
				img    = microscope.camera.getFrame(convert=False)
				n_captured += 1

				scanned_locations.append([xc, yc])

				fstr  = '%06d_%2.5f_%2.5f.png'%(
					n_images, xc, yc
				)
				print(fstr)
				n_images += 1
				fname = os.path.join(args.output_directory, fstr)

				if n_captured % args.preview_interval == 0:
					decimated = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
					cv2.imshow('Scan Preview', decimated)
					cv2.waitKey(1)

				cv2.imwrite(fname, img)

				if microscope.stage.getAxisStatus('x') or microscope.stage.getAxisStatus('y'):
					continue
				else:
					break

			x_current += args.image_width
			microscope.stage.moveTo(x_current, y_current)

		n_squares += 1


	scanned_locations = np.array(scanned_locations)
	plt.scatter(scanned_locations[:, 0], scanned_locations[:, 1], s=1)
	plt.show()

	end = time.time_ns()

	duration = ((end - start) / 1e9) / 60
	print("Scan took %d minutes"%(int(duration)))

	microscope.camera.endCapture()

	print("Done")

