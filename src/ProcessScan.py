# Author:      Adam Robinson
# Description: This file contains the logic used by Scan.py to process images
#              as they are acquired. It can also be run as a standalone command
#              line program to process a folder of images that has not yet been
#              processed. This program will move images to a folder called 
#              "deleted", if they don't appear to conatain any flakes whose 
#              largest bounding box dimension is greater than or equal to 
#              the --rejection-threshold parameter (relative to image dimensions).
#
#              This program runs efficiently using multiple subprocesses. The speedup
#              is linearly related to the number of subprocesses up to the number of
#              cores on your systems CPU. 

import code
import sys
import os
import time
import cv2
import argparse
import json
import shutil
import numpy             as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from ImageProcessing import processFile
from Progress        import ProgressBar

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

def getFilesSince(directory, timestamp):
	entries = os.listdir(directory)

	results = []
	for entry in entries:
		try:
			path = os.path.join(directory, entry)
			ts   = os.path.getctime(path)
			ms   = os.path.getmtime(path)
			ts   = max(ms, ts)
		except FileNotFoundError as ex:
			# Its possible for a file to get deleted between the call
			# to os.listdir and the calls to getctime and getmtime.
			continue

		if ts > timestamp:
			ext = entry.split(".")[-1].lower()
			if ext == 'png':
				results.append(path)

	return results


# I know, weird name, but it does accurately and somewhat concisely describe what this code
# does.
class MultiProcessImageProcessor:
	def __init__(self, n_processes=8):
		self.n_processes  = n_processes
		self.pool         = Pool(args.n_processes)

		self.current_in_process = 0
		self.total_processed    = 0
		self.idx                = 0
		self.results            = []
		self.failures           = []

	def addItem(self, item, bg,args=None):
		res = self.pool.apply_async(processFile, (None, item, args))
		self.results.append(res)
		self.idx                += 1
		self.current_in_process += 1

	def waitForCompletion(self):
		pb = ProgressBar("Processing Images", 18, self.current_in_process, 1, ea=25)
		processed = 0
		while self.current_in_process > 0:
			done = []
			for r in self.results:
				if r.ready():
					status, fname            = r.get(0.01)
					self.total_processed    += 1
					self.current_in_process -= 1
					
					processed += 1
					pb.update(processed)
					done.append(r)

					if not status:
						self.failures.append(fname)

			for d in done:
				self.results.remove(d)

			time.sleep(0.001)
		pb.finish()

if __name__ == '__main__':
	# Load the arguments file. 
	with open("_ProcessScan.json", 'r') as file:
		args_specification = json.loads(file.read())
	args  = preprocess(args_specification)

	files = getFilesSince(args.output_directory, 0)

	print("Processing %d files using %d processes."%(
		len(files), 
		args.n_processes
	))
	

	pb = ProgressBar("Processing ", 11, len(files), 1, ea=25)

	if args.n_processes > 1:
		pool = Pool(args.n_processes)

		current_in_process = 0
		total_processed    = 0
		idx                = 0
		results            = []
		failures           = []

		while total_processed < len(files):
			while current_in_process < args.n_processes and idx < len(files):
				res = pool.apply_async(processFile, (None, files[idx], args))
				results.append(res)
				idx                += 1
				current_in_process += 1

			done = []
			for r in results:
				if r.ready():
					status, fname = r.get(0.01)
					total_processed    += 1
					current_in_process -= 1
					pb.update(total_processed)
					done.append(r)

					if not status:
						failures.append(fname)

			for d in done:
				results.remove(d)

			time.sleep(0.001)
	else:
		for idx, file in enumerate(files):
			img = cv2.imread(file)
			processFile(img, file, args)
			pb.update(idx + 1)

	pb.finish()

	if len(failures) > 0:
		print("The following (%d) files were deleted."%len(failures))
		for f in failures:
			print("     %s"%f)