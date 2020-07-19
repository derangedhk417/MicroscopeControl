import code
import sys
import os
import time
import cv2
import argparse
import numpy             as np
import matplotlib.pyplot as plt

from ImageProcessor import ImageProcessor

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
	with open("ScanProcessor.json", 'r') as file:
		args_specification = json.loads(file.read())

	args = preprocess(args_specification)