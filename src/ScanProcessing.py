import code
import sys
import os
import time
import cv2
import argparse
import json
import numpy             as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from ImageProcessor  import ImageProcessor, FlakeExtractor
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

def loadAndProcess(fpath, args):
	try:
		img = cv2.imread(fpath)
		return processFile(img, fpath, args)
	except Exception as ex:
		print(ex)
		raise ex


def processFile(img, fpath, args):
	extractor = FlakeExtractor(
		img,
		downscale=args.downscale,
		threshold=args.rejection_threshold,
		contrast_floor=args.contrast_floor
	)

	status, res = extractor.process()

	if status:
		outfile = ".".join(fpath.split("\\")[-1].split(".")[:-1]) + '.json'
		outpath = os.path.join("\\".join(fpath.split("\\")[:-1]), outfile)
		with open(outpath, 'w') as file:
			file.write(json.dumps(res))
		return True, None
	else:
		os.remove(fpath)
		return False, fpath


if __name__ == '__main__':
	# Load the arguments file. 
	with open("ScanProcessing.json", 'r') as file:
		args_specification = json.loads(file.read())
	args  = preprocess(args_specification)

	files = getFilesSince(args.image_directory, 0)

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
				res = pool.apply_async(loadAndProcess, (files[idx], args))
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