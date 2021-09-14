# Author:      Adam Robinson
# Description: This file defines the code necessary to extract relevant information from a single
#              microcscope image. This generally means supplying contrast values calculated via
#              the FPContrastCalculator.py code.
#              (https://github.com/derangedhk417/MonolayerContrastCalculator)
#              Code in this file will calculate contrast values for an image, contours of flakes and
#              will attempt to categorize the thickness of each flake using the contrast values 
#              calculated by the FPContrastCalculator.py code.

import os
import sys
import code
import time
import argparse
import json

import numpy             as np
import matplotlib.pyplot as plt

import cv2

from scipy.interpolate import interp1d

class ImageProcessor:
	# The material file argument should point to a json file containing information about the camera
	# system and the materials being imaged. The most important information in this file is the 
	# contrast of different thicknesses of the material for each color channel. See _graphene.json
	# for an example file.
	def __init__(self, material_file, **kwargs):
		# Load the material file.
		with open(material_file, 'r') as file:
			raw_file = json.loads(file.read())

		self.layer_contrast = np.array(raw_file['layers'])
		self.thresholds     = raw_file['threshold']

		self._createGaussianScoringFunctions()

		if 'downscale_factor' in kwargs:
			self.downscale_factor = kwargs['downscale_factor']
		else:
			self.downscale_factor = 1

		if "median_blur" in kwargs:
			self.median_blur = kwargs['median_blur']
		else:
			self.median_blur = False

		if "bilateral_filter" in kwargs:
			self.bilateral_filter = kwargs['bilateral_filter']
		else:
			self.bilateral_filter = None

		if "denoise" in kwargs:
			self.denoise = kwargs['denoise']
		else:
			self.denoise = 0

		if "invert_contrast" in kwargs:
			self.invert_contrast = kwargs['invert_contrast']
		else:
			self.invert_contrast = False


	# This will create a gaussian centered on each thickness value for each color channel. The 
	# gaussian will be normalized and it's FWHM will be half the distance between itself and the
	# next closest contrast value.
	def _createGaussianScoringFunctions(self):
		def gaussian(x, fwhm, center):
			s = fwhm / np.sqrt(2 * np.log(2))
			A = (1 / (s * np.sqrt(2*np.pi)))
			return A * np.exp(-np.square(x - center) / (2 * np.square(s)))

		def getGaussianScoringFunctionsForChannel(channel):
			parameters = []
			for idx, layer in enumerate(channel):
				check      = channel.copy()
				check[idx] = -1e6
				diff       = np.abs(layer - check)
				closest    = channel[np.argmin(diff)]
				fwhm       = np.abs(closest - layer) / 2
				parameters.append([layer, fwhm])

			return parameters

		r = getGaussianScoringFunctionsForChannel(self.layer_contrast[:, 2])
		g = getGaussianScoringFunctionsForChannel(self.layer_contrast[:, 3])
		b = getGaussianScoringFunctionsForChannel(self.layer_contrast[:, 4])

		self.score_functions = [r, g, b]

	def getContrastImg(self, img):
		if self.median_blur:
			img = cv2.medianBlur(img, 3)

		if self.bilateral_filter is not None:
			img = cv2.bilateralFilter(img, 5, *self.bilateral_filter)

		if self.denoise != 0:
			img = cv2.fastNlMeansDenoisingColored(
				img, 
				self.denoise, 
				self.denoise
			)

		# Separate the channels.
		B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

		def getMode(channel):
			return np.argmax(np.bincount(channel.flatten()))

		R_mode = max(getMode(R), 1)
		G_mode = max(getMode(G), 1)
		B_mode = max(getMode(B), 1)

		# Now we convert to floating point for the contrast calculation.
		R = R.astype(np.float32)
		G = G.astype(np.float32)
		B = B.astype(np.float32)

		R_con = -(R - R_mode) / R_mode
		G_con = -(G - G_mode) / G_mode
		B_con = -(B - B_mode) / B_mode

		if self.invert_contrast:
			R_con = -R_con
			G_con = -G_con
			B_con = -B_con

		# Set everything less than the threshold value to zero.
		R_con[R_con < self.thresholds['r']] = 0
		G_con[G_con < self.thresholds['g']] = 0
		B_con[B_con < self.thresholds['b']] = 0

		# Return the contrast image.
		return np.stack([R_con, G_con, B_con], axis=2)


	def processImage(self, img):
		# If the supplied image is a string then we load it. Otherwise, if it's an ndarray we
		# assume it's already properly loaded in BGR format.
		if type(img) is np.ndarray:
			pass
		elif type(img) is str:
			img = cv2.imread(img)
		else:
			raise Exception("Supplied 'img' argument was not a string or an image.")

		if self.downscale_factor != 1:
			# Downscale the image before processing.
			img = cv2.resize(img, (0, 0), 
				fx=(1/self.downscale_factor), 
				fy=(1/self.downscale_factor),
				interpolation=cv2.INTER_NEAREST # This should keep overhead at a minimum.
			)


		plt.imshow(img)
		plt.show()

		# Here we calculate the contrast of all values in the image using a simple method.
		contrast_img  = self.getContrastImg(img.copy())

		fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

		ax1.imshow(contrast_img[:, :, 0])
		ax2.imshow(contrast_img[:, :, 1])
		ax3.imshow(contrast_img[:, :, 2])

		plt.show()

	# Attempts to determine the thickness of pixel, in layers. This is done by assigning a gaussian
	# centered on the contrast of each layer for each color. Every pixel will be scored based on 
	# each gaussian and the highest score taken as the thickness of the pixel.
	def categorizeContrastImage(self, contrast_img):
		pass

if __name__ == '__main__':
	p = ImageProcessor(
		"_graphene.json", 
		invert_contrast=False,
		median_blur=True
	)
	p.processImage("test/image2.png")

