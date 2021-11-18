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

		if 'background' in kwargs:
			self.background = kwargs['background']
		else:
			self.background = None

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

		# height, width (mm)
		if 'image_dims' in kwargs:
			self.image_dims = kwargs['image_dims']
		else:
			self.image_dims = None

		if 'min_area' in kwargs:
			self.min_area = kwargs['min_area']
		else:
			self.min_area = 0.0

		if 'skip_channel' in kwargs:
			self.skip_channel = kwargs['skip_channel']
		else:
			self.skip_channel = None

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
		for i in range(10):
			img = cv2.bilateralFilter(img, 5, *self.bilateral_filter)
			

		# Separate the channels.
		B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

		R_bg = self.background[:, :, 2]
		G_bg = self.background[:, :, 1]
		B_bg = self.background[:, :, 0]
		R_con = (R_bg - R) / R_bg
		G_con = (G_bg - G) / G_bg
		B_con = (B_bg - B) / B_bg
		
		if self.invert_contrast:
			R_con = -R_con
			G_con = -G_con
			B_con = -B_con

		# Set everything less than the threshold value to zero.
		R_con[R_con < self.thresholds['r']] = 0
		G_con[G_con < self.thresholds['g']] = 0
		B_con[B_con < self.thresholds['b']] = 0

		result = np.stack([R_con, G_con, B_con], axis=2)

		# Return the contrast image.
		return result

	def processImage(self, img, bg):
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

		layers, _, [scores_r, scores_g, scores_b] = self.categorizeContrastImage(
			contrast_img, calc_disagreement=False
		)

		# code.interact(local=locals())
		plt.imshow(layers)
		plt.show()

		# Now that we have the layer data, we use it to create a mask where 1 corresponds to flake
		# and 0 corresponds to background. We'll use this to contour each flake. This information
		# will be used later on by the user when they're filtering which flakes they want to look
		# at.

		contours, rectangles = self.getFlakeContours(layers)

		# The ultimate product of this function should be a database entry for each flake found in
		# the image. We want to have the following columns for each flake:
		#     area, L1_area, L2_area, ..., LN_area
		# where there is a column for each layer number in the configuration file passed to this
		# function.

		# We need a conversion factor between square pixels and square real units.
		# If we weren't supplied with image area information then we just set this to 1.
		conversion_factor = 1.0
		if self.image_dims is not None:
			h, w = self.image_dims
			h *= 1e-3 # convert to meters
			w *= 1e-3 # convert to meters

			width_per_pixel   = w / self.current_img.shape[1]
			height_per_pixel  = h / self.current_img.shape[0]
			# Pixels should be square so we take the average of the two values.
			# This should result in each area being in square microns.
			conversion_factor = width_per_pixel * height_per_pixel * 1e12

		entries = []
		for idx, (c0, r0) in enumerate(zip(contours, rectangles)):
			# We want to extract a subimage that corresponds to the flake we are dealing with. We'll
			# extract this subimage from the "layers" image and use it to calculate stats for the
			# flake.
			entry             = {}
			entry['file']     = os.path.split(self.img_path)[-1]
			entry['geom_idx'] = idx

			flake_image   = self.subimage(layers, r0)
			area          = (flake_image.flatten() != 0).sum() * conversion_factor
			if area < self.min_area:
				continue
			entry['area'] = area

			# Now we repeat the calculation, but for every layer number.
			counts = np.bincount(flake_image.astype(np.int64).flatten())[1:]
			if len(counts) == 0:
				continue
			for i, count in enumerate(counts):
				entry['L%03d_area'%(i + 1)] = count * conversion_factor

			entries.append(entry)
			# TODO: Calculate quality statistics like the continuity of regions. Number of holes,
			# concavity, etc.

		


		# We now have pretty much all of the useful raw data we can get from the image. The next
		# step is to export this data into a format that can be processed.
		fname = self.img_path.replace("\\", "/").split("/")[-1]
		geometry_path = os.path.join(self.output_path, "%s.geometry.json"%fname)
		layers_path   = os.path.join(self.output_path, "%s.layers.npy"%fname)
		stats_path    = os.path.join(self.output_path, "%s.stats.json"%fname)

		np.save(layers_path, layers)
		with open(geometry_path, 'w') as file:
			file.write(json.dumps({
				'contours'   : [c.tolist() for c in contours],
				'rectangles' : rectangles
			}))

		with open(stats_path, 'w') as file:
			file.write(json.dumps({"flakes": entries}))

		return [geometry_path, layers_path, stats_path]

	def subimage(self, image, rect):
		border_size = int(image.shape[0] / 2)
		image = cv2.copyMakeBorder(
			image,
			border_size,
			border_size,
			border_size,
			border_size,
			cv2.BORDER_CONSTANT,
			value=[0, 0, 0]
		)

		((x, y), (w, h), theta) = rect

		x = x + border_size
		y = y + border_size

		size = (image.shape[1], image.shape[0])

		rotation_matrix = cv2.getRotationMatrix2D(center=(x, y), angle=theta, scale=1)
		new_image       = cv2.warpAffine(image, rotation_matrix, dsize=size)

		x = int(x - w/2)
		y = int(y - h/2)

		w = int(w)
		h = int(h)

		result = new_image[y:y+h, x:x+w]
		#code.interact(local=locals())
		return result

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

def processFile(fname, bg, image_dims, args):
	print("Processing file")
	p = ImageProcessor(
		args.material_file,
		invert_contrast=False,
		bilateral_filter=[10, 50],
		background=bg,
		nearest=False,
		downscale_factor=2,
		debug=False,
		output_path=args.output_directory,
		image_dims=image_dims
	)
	files = p.processImage(fname)
	return True, [fname, files]

def calculateBackgroundColored(images):
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

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	ax1.imshow(images[0])
	ax2.imshow(background)
	ax3.imshow(images[0] - background)
	plt.show()
	return background

if __name__ == '__main__':
	fnames = [
		"0009.png", "0013.png", "0025.png", "0060.png",
		"0091.png", "0098.png", "0123.png", "0142.png"
	]
	imgs = [cv2.imread("data/test/" + f) for f in fnames]

	bg = calculateBackgroundColored(imgs)

	p = ImageProcessor(
		"_graphene_on_90nmSiO2_Silicon.json", 
		invert_contrast=False,
		background=bg,
		bilateral_filter=[10, 50],
		debug=True,
		downscale_factor=2,
		output_path="test",
		nearest=False
	)
	p.processImage("test/0386.png")
