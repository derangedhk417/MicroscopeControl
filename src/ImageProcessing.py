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

		if 'output_path' in kwargs:
			self.output_path = kwargs['output_path']
		else:
			self.output_path = "output"

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

		if "sharpen" in kwargs:
			self.sharpen = kwargs['sharpen']
		else:
			self.sharpen = False

		if "dilate" in kwargs:
			self.dilate = kwargs['dilate']
		else:
			self.dilate = 0

		if "erode" in kwargs:
			self.erode = kwargs['erode']
		else:
			self.erode = 0

		if 'debug' in kwargs:
			self.debug = kwargs['debug']
		else:
			self.debug = False

	# This will create a gaussian centered on each thickness value for each color channel. The 
	# gaussian will be normalized and it's FWHM will be half the distance between itself and the
	# next closest contrast value.
	def _createGaussianScoringFunctions(self):
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

		self.gaussianScoringFunctions = [r, g, b]

	def getContrastImg(self, img):
		if self.denoise != 0:
			if self.debug:
				plt.imshow(img)
				plt.title("Before Denoise")
				plt.show()
			img = cv2.fastNlMeansDenoisingColored(
				img, 
				self.denoise, 
				self.denoise
			)
			if self.debug:
				plt.imshow(img)
				plt.title("After Denoise")
				plt.show()

		if self.sharpen:
			if self.debug:
				plt.imshow(img)
				plt.title("Before Sharpen")
				plt.show()
			sharpen_kernel = np.array([
				[-1, -1, -1], 
				[-1, 9,  -1], 
				[-1, -1, -1]
			])
			img = cv2.filter2D(img, -1, sharpen_kernel)
			if self.debug:
				plt.imshow(img)
				plt.title("After Sharpen")
				plt.show()

		if self.median_blur:
			img = cv2.medianBlur(img, 3)

		if self.bilateral_filter is not None:
			if self.debug:
				plt.imshow(img)
				plt.title("Before Bilateral Filter")
				plt.show()
			img = cv2.bilateralFilter(img, 5, *self.bilateral_filter)
			if self.debug:
				plt.imshow(img)
				plt.title("After Bilateral Filter")
				plt.show()


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

		result = np.stack([R_con, G_con, B_con], axis=2)

		if self.erode != 0:
			if self.debug:
				plt.imshow(result)
				plt.title("Before Erode Filter")
				plt.show()
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode, self.erode))
			result    = cv2.erode(result, kernel)
			if self.debug:
				plt.imshow(result)
				plt.title("After Erode Filter")
				plt.show()

		if self.dilate != 0:
			if self.debug:
				plt.imshow(result)
				plt.title("Before Dilate Filter")
				plt.show()
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
			result    = cv2.dilate(result, kernel)
			if self.debug:
				plt.imshow(result)
				plt.title("After Dilate Filter")
				plt.show()

		# Return the contrast image.
		return result

	def processImage(self, img):
		self.img_path = img
		# If the supplied image is a string then we load it. Otherwise, if it's an ndarray we
		# assume it's already properly loaded in BGR format.
		img = cv2.imread(img)

		if self.downscale_factor != 1:
			# Downscale the image before processing.
			img = cv2.resize(img, (0, 0), 
				fx=(1/self.downscale_factor), 
				fy=(1/self.downscale_factor),
				interpolation=cv2.INTER_NEAREST # This should keep overhead at a minimum.
			)

		self.current_img = img

		# Here we calculate the contrast of all values in the image using a simple method.
		contrast_img  = self.getContrastImg(img.copy())

		if self.debug:
			fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

			ax1.imshow(contrast_img[:, :, 0])
			ax2.imshow(contrast_img[:, :, 1])
			ax3.imshow(contrast_img[:, :, 2])

			plt.show()

		# Here we convert the image with shape (h, w, 3) into an image with only one channel,
		# where that channel is an integer corresponding to the index in the list of layers.
		# For example, if a pixels value is 2, that would correspond to 3 layers, if the 
		# supplied data files has data for layers 1-20 in order.

		layers, disagreement, [scores_r, scores_g, scores_b] = self.categorizeContrastImage(
			contrast_img, calc_disagreement=True
		)

		if self.debug:
			disag_img = layers.copy()
			disag_img[:, :] = 0
			disag_img[disagreement] = 1
			fig, (ax1, ax2) = plt.subplots(1, 2)
			ax1.imshow(layers)
			ax2.imshow(disag_img)
			plt.show()

		# Now that we have the layer data, we use it to create a mask where 1 corresponds to flake
		# and 0 corresponds to background. We'll use this to contour each flake. This information
		# will be used later on by the user when they're filtering which flakes they want to look
		# at.

		contours, rectangles = self.getFlakeContours(layers)

		# We now have pretty much all of the useful raw data we can get from the image. The next
		# step is to export this data into a format that can be processed.
		fname = self.img_path.replace("\\", "/").split("/")[-1]
		geometry_path = os.path.join(self.output_path, "%s.geometry.json"%fname)
		layers_path   = os.path.join(self.output_path, "%s.layers.npy"%fname)
		scores_path   = os.path.join(self.output_path, "%s.scores.npy"%fname)

		scores = np.stack([
			scores_r, 
			scores_g,
			scores_b,
		], axis=2)

		# This appears to take way too much disk space.
		# np.save(scores_path, scores)
		np.save(layers_path, layers)
		with open(geometry_path, 'w') as file:
			file.write(json.dumps({
				'contours'   : [c.tolist() for c in contours],
				'rectangles' : rectangles
			}))


	def getFlakeContours(self, layers):
		# In order to make the image ready for processing, we need to level it (make it binary),
		# put a border around it, edge detect it, dilate the edges and run openCVs contour detection
		# rountine.
		leveled = layers.copy()
		mask    = layers > 0
		leveled[mask]  = 1

		if self.debug:
			plt.imshow(leveled)
			plt.title("Leveled")
			plt.show()


		bordered = cv2.copyMakeBorder(
			leveled, 
			4, 
			4, 
			4, 
			4, 
			cv2.BORDER_CONSTANT,
			value=0
		).astype(np.uint8)

		if self.debug:
			plt.imshow(bordered)
			plt.title("Bordered")
			plt.show()

		edged = cv2.Canny(bordered, 0, 1)

		if self.debug:
			plt.imshow(edged)
			plt.title("Edge Detected")
			plt.show()

		kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
		dilated = cv2.dilate(edged, kernel)

		if self.debug:
			plt.imshow(dilated)
			plt.title("dilated")
			plt.show()

		contours, heirarchy = cv2.findContours(
			dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		contours   = [c.reshape(-1, 2) for c in contours]
		rectangles = [cv2.minAreaRect(c.reshape(-1, 1, 2)) for c in contours]

		# Correct the coordinates for the border we added.
		corrected_rects    = []
		corrected_contours = []
		for rect, contour in zip(rectangles, contours):
			((left, top), (width, height), angle) = rect

			# Calculate the relative size of the largest dimension, making
			# sure to use the units of the downscaled image.
			if width > height:
				largest = width / layers.shape[1]
			else:
				largest = height / layers.shape[0]

			# Correct for the border.
			left    -= 4
			top     -= 4
			contour -= 4

			corrected_rects.append(((left, top), (width, height), angle))
			corrected_contours.append(contour)

		if self.debug:
			c_img = self.current_img.copy()
			# code.interact(local=locals())
			for con in contours:
				con = np.array(con)
				con = con.astype(np.int32)
				c_img = cv2.drawContours(
					c_img, 
					[con.reshape(-1, 1, 2)], 
					0, (255, 0, 0), 1
				)

			plt.imshow(c_img)
			plt.title("Contours")
			plt.show()

		return contours, rectangles



	# Attempts to determine the thickness of pixel, in layers. This is done by assigning a gaussian
	# centered on the contrast of each layer for each color. Every pixel will be scored based on 
	# each gaussian and the highest score taken as the thickness of the pixel.
	def categorizeContrastImage(self, contrast_img, nearest=False, calc_disagreement=False):
		def gaussian(x, fwhm, center):
			s = fwhm / np.sqrt(2 * np.log(2))
			A = (1 / (s * np.sqrt(2*np.pi)))
			return A * np.exp(-np.square(x - center) / (2 * np.square(s)))

		def scoreChannel(channel, score_functions):
			# We have N layers specified in self.layer_contrast[:, 2:5]. We need to calculate the
			# value of the gaussian corresponding to each layer, for each pixel.
			scores = np.repeat(channel[:, :, np.newaxis], len(score_functions), axis=2)
			for i, n in enumerate(score_functions):
				center, fwhm    = n
				scores[:, :, i] = gaussian(channel, fwhm, center)

			return scores

		def scoreChannelNearest(channel, contrasts):
			# We have N layers specified in self.layer_contrast[:, 2:5]. We need to calculate the
			# value of the gaussian corresponding to each layer, for each pixel.
			scores = np.repeat(channel[:, :, np.newaxis], len(contrasts), axis=2)
			for i, n in enumerate(contrasts):
				scores[:, :, i] = 1 / np.abs(channel - n)

			return scores

		if nearest:
			scores_r = scoreChannelNearest(contrast_img[:, :, 0], self.layer_contrast[:, 2])
			scores_g = scoreChannelNearest(contrast_img[:, :, 1], self.layer_contrast[:, 3])
			scores_b = scoreChannelNearest(contrast_img[:, :, 2], self.layer_contrast[:, 4])
		else:
			scores_r = scoreChannel(contrast_img[:, :, 0], self.gaussianScoringFunctions[0])
			scores_g = scoreChannel(contrast_img[:, :, 1], self.gaussianScoringFunctions[1])
			scores_b = scoreChannel(contrast_img[:, :, 2], self.gaussianScoringFunctions[2])
		

		# Now we take argmax along the third  dimension for each in order to get an indexed
		# for each pixel.
		cat_r = np.argmax(scores_r, axis=2)
		cat_g = np.argmax(scores_g, axis=2)
		cat_b = np.argmax(scores_b, axis=2)

		# We now have the correct index for each. The next step is to see if they agree. Any that
		# don't agree will be set to -1 in the final array.
		scores       = np.stack([
			np.max(scores_r, axis=2), 
			np.max(scores_g, axis=2),
			np.max(scores_b, axis=2)
		], axis=2)
		best         = np.argmax(scores, axis=2)
		layers       = self.layer_contrast[:, 0]
		
		results = cat_r.copy()
		results[best == 0] = cat_r[best == 0]
		results[best == 1] = cat_g[best == 1]
		results[best == 2] = cat_b[best == 2]
		
		mask          = (contrast_img[:, :, 0] == 0) | (contrast_img[:, :, 1] == 0)
		mask          = mask | (contrast_img[:, :, 2] == 0)
		layers        = layers[results]
		layers[mask]  = 0

		disagreement = None
		if calc_disagreement:
			disagreement = (cat_r != cat_g) | (cat_g != cat_b) | (cat_b != cat_r)



		return [layers, disagreement, [scores_r, scores_g, scores_b]]

def processFile(img, fname, args):
	p = ImageProcessor(
		args.material_file,
		invert_contrast=args.invert_contrast,
		median_blur=False,
		sharpen=True,
		denoise=0,
		erode=3,
		dilate=3,
		downscale_factor=args.downscale
		debug=False,
		output_path=args.output_directory
	)

	p.processImage(fname)
	return True, None

if __name__ == '__main__':
	# p = ImageProcessor(
	# 	"_graphene_on_pdms.json", 
	# 	invert_contrast=False,
	# 	median_blur=True,
	# 	downscale_factor=4
	# )
	# p.processImage("test/000019_-6.02260_0.54630.png")

	# KEEP THIS, IT SEEMS TO WORK WELL
	# p = ImageProcessor(
	# 	"_graphene.json", 
	# 	invert_contrast=False,
	# 	median_blur=False,
	# 	sharpen=True,
	# 	bilateral_filter=(50, 50),
	# 	denoise=15
	# )
	# p.processImage("test/Image2.png")

	# KEEP THIS, IT SEEMS TO WORK WELL
	p = ImageProcessor(
		"_graphene.json", 
		invert_contrast=False,
		median_blur=False,
		sharpen=True,
		denoise=0,
		erode=3,
		dilate=3,
		debug=False
	)
	p.processImage("test/Image2.png")
