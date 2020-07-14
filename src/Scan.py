import matplotlib.pyplot as plt
import numpy             as np
import sys
import time
import code
import argparse
import os
import cv2
import json

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
		'-W', '--width', dest='image_width', type=float, required=True,
		help='The width (along x-direction) of the image at the current zoom (mm).' 
	)

	parser.add_argument(
		'-H', '--height', dest='image_height', type=float, required=True,
		help='The height (along y-direction) of the image at the current zoom (mm).' 
	)

	parser.add_argument(
		'-f', '--focus-interval', dest='focus_interval', type=float, required=True,
		help='Refocus the microscope after this many images have been taken.' 
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

	print("Focusing microscope.")
	microscope.autoFocus(
		[0.48, 0.52], 
		mode='auto', 
		n_divisions=100,
		passes=1
	)

	print("Focusing complete. Please check the following image.")
	time.sleep(2)

	microscope.camera.startCapture()
	img = microscope.camera.getFrame(convert=True)
	plt.imshow(img)
	plt.show()
	microscope.camera.endCapture()

	if input("Proceed (y/n)? ").lower() != 'y':
		exit()

	print("Beginning scan.")

	y_current = args.y_limits[0]
	x_current = args.x_limits[0]
	n_images  = 1

	
	microscope.camera.startCapture()
	microscope.stage.moveTo(x_current, y_current)
	while y_current < args.y_limits[1]:
		x_current = args.x_limits[0]
		microscope.stage.moveTo(x_current, y_current)

		while x_current < args.x_limits[1]:
			if n_images % args.focus_interval == 0:
				microscope.autoFocus(
					[0.48, 0.52], 
					mode='auto', 
					n_divisions=100,
					passes=1
				)
				microscope.camera.startCapture()

			x, y = microscope.stage.getPosition()
			img  = microscope.camera.getFrame(convert=False)

			fstr  = '%06d_%2.5f_%2.5f.png'%(
				n_images, x, y
			)
			fname = os.path.join(args.output_directory, fstr)

			decimated = cv2.resize(img, (0, 0), fx=0.333, fy=0.333)
			cv2.imshow('Preview', decimated)
			cv2.waitKey(1)

			cv2.imwrite(fname, img)

			n_images  += 1
			x_current += args.image_width
			microscope.stage.moveTo(x_current, y_current)

		y_current += args.image_height

	microscope.camera.endCapture()

	print("Done")

