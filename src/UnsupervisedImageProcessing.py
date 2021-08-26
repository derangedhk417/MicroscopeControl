# Author:      Adam Robinson, Luke St. Marie
# Description: This file defines functions for finding and classifying flakes
#              using an unsupervised machine learning method. More specifically,
#              at the time of this writing it uses a Gaussian Mixture Model (GMM)
#              Functionality includes training a model with images that have
#              flakes of known thicknesses, saving a model, loading a model and
#              using it to classify new images of the same material. 
#              This code is meant to be called from the Scan.py tool, but it can
#              also be run directly at the command line in order to debug it,
#              train a new model or classify a set of images that have not 
#              already been classified.

# Attribution: This code is based partially on work done by Luke St. Marie.
# Notes on Luke's Code:
#     The polynomial fit seems to be uneccessary. Every image I tested it on had
#     parameter values on the order of 10^-5 for every parameter except for 
#     the constant offset "a". This means that we can replace the fit with a 
#     simple average for each channel (of the background pixels). This will be a
#     lot faster.

import code
import sys
import json
import os
import time
import cv2
import argparse
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from Colors          import dist_colors     as colors
from Colors          import torgb
from scipy.stats     import gaussian_kde
from scipy.signal    import find_peaks
from scipy.optimize  import curve_fit, least_squares
from copy            import deepcopy

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

# Defines a model that takes an image as an argument and finds the flakes in the
# image. For each image processed, this object should return the following:
#     1) The contour data for each flake.
#     2) The estimated thickness of each flake (an integer).
#
# This information will be returned as a [TBD] object.
#
# This object stores the following:
#     1) Information about the imaging conditions necessary for it to work.
#     2) Information about the material being imaged, the substrate and 
#        any background underneath the substrate (if it's transparent).
#     3) A trained gaussian mixture model.
#     4) Information about how images need to be preprocessed before they can
#        be used. This may include things like bilateral filtering or denoising.
class Model:
	def __init__(self, path=None):
		if path is not None:
			self.loadFromDisk(path)
		else:
			self.gmm                     = None
			self.imagingConditions       = {
				'white_balance'    : (1.0, 1.0, 1.0),
				'gamma'            : 0.44,
				'saturation'       : 100.0,
				'gain'             : 0.0,
				'color_temp'       : 3200,
				'zoom'             : 1.0,
				'objective'        : "Mitutoyo 10X",
				'room_lights'      : 'on',
				'exposure_time'    : 0.030, # Exposure time in ms.
				'source_intensity' : 9.0    # Intensity of the source in ticks 
				                            # on the dial. 

			}
			self.materialCharacteristics = {
				'material'   : '<not specified>',
				'substrate'  : '<not specified>',
				'background' : '<not specified>'
			}
			self.bilateralSettings       = [5, 50, 50]
			self.inputResolution         = (2448, 2048)
			self.downscaleFactor         = 3

# This is the class that actually trains a model. It should be initialized with
# a Model object and then passed a set of images as a training set, when the 
# trainModel function is called.
class ModelTrainer:
	def __init__(self, model, verbose=False):
		self.model   = model
		self.verbose = verbose

	# This function needs to be passed the path to a training set on disk. This
	# training set needs to include the following:
	#     1) one or more image files
	#     2) a configuration file (see example_config.json)
	#
	# The image files are used to train the model and the config file specifies
	# imaging conditions, material information and the thicknesses of each flake
	# in each image. For example, if an image contains flakes of thicknesses
	# 1, 5, and 8, the config file will have an entry for that image file,
	# specifying that flakes of those thicknesses exist in the image. You can 
	# consider very thick flakes as the same thickness when categorizing them.
	def trainModel(self, trainingSetPath):
		if self.verbose:
			print("Loading configuration file . . . ", end='')

		# Check the training set directory and make sure it contains a config
		# file.
		configPath = os.path.join(trainingSetPath, "config.json")

		if not os.path.isfile(configPath):
			raise Exception("configuration file is missing")

		try:
			with open(configPath) as file:
				config = json.loads(file.read())
		except Exception as ex:
			raise Exception("unable to read or parse config file") from ex


		# Now we check for preprocessing parameters in the config file. If they
		# don't exist then we'll set them to defaults.
		self.bilateralSettings = [5, 50, 50]
		self.downscaleFactor   = 3
		self.useHsv            = False

		if 'preprocessing' in config:
			if 'bilateralSettings' in config['preprocessing']:
				self.bilateralSettings = config['preprocessing']['bilateralSettings']
			else:
				self.bilateralSettings = [5, 50, 50]

			if 'downscaleFactor' in config['preprocessing']:
				self.downscaleFactor = config['preprocessing']['downscaleFactor']
			else:
				self.downscaleFactor = 3

			if 'useHsv' in config['preprocessing']:
				self.useHsv = config['preprocessing']['useHsv']
			else:
				self.useHsv = False


		if self.verbose:
			print("done")
			print("Loading image files . . . ", end='')

		# Now we need to read through all of the images that are in the config
		# file, load them and create a structure that matches loaded images to 
		# the list of flake thicknesses in the image.

		imageNames   = []
		loadedImages = []
		imageMeta    = []

		for imageSpec in config['images']:
			imgPath = os.path.join(trainingSetPath, imageSpec['path'])
			imageNames.append(imageSpec['path'])

			try:
				
				img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
				if self.useHsv:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				loadedImages.append(img)
			except Exception as ex:
				raise Exception("Unable to load image %s"%imageSpec['path']) from ex

			if 'meta' in imageSpec:
				imageMeta.append(imageSpec['meta'])
			else:
				raise Exception("No metadata specified for %s"%imageSpec['path'])

		# We've now loaded the images we are going to train the model with. Now
		# we need to process each image into a set of optical contrast values.
		# For the purpose of this system, we will define the optical contrast
		# as (val - bg) / bg where val is the value of each pixel and bg is the
		# background color. This calculation is performed for each channel.

		if self.verbose:
			print("done")
			print("Calculating contrasts . . . ", end='')

		contrastImages = []
		for img in loadedImages:
			if self.useHsv:
				# img = cv2.bilateralFilter(img, 5, 35, 50)
				# img = cv2.fastNlMeansDenoisingColored(
				# 	img, 
				# 	15, 
				# 	15
				# )
				img = img.astype(np.float32)
				img = (img - img.min()) / (img.max() - img.min())
				contrastImages.append(img)
			else:
				contrastImages.append(self.getContrast(img))

		if self.verbose:
			print("done")
			print("Training temporary models . . . ")

		# Now that we have contrast values for each image we can train a GMM on
		# each image. This is necessary to ensure that we can correctly categorize
		# each flake thickness in each image. These GMMs will only be used 
		# temporarily and will not be saved. We use them to ensure that the training
		# set for the final GMM contains enough data from each layer thickness.

		gmms    = []
		results = []
		for n, (cimg, meta) in enumerate(zip(contrastImages, imageMeta)):
			components = len(meta) + 1 # Plus one for the background
			gmm = GaussianMixture(
				n_components    = components,
				covariance_type = 'full'
			)

			# Reshape the image into a dataset for the gmm.
			data = cimg.reshape(cimg.shape[0] * cimg.shape[1], 3)

			# Train the model and categorize all pixels.
			if self.verbose:
				print("%s (%d / %d)"%(imageNames[n], n + 1, len(imageMeta)))

			res = gmm.fit_predict(data)
			results.append(res)

			# If verbose == True, show the image alongside what each pixel was
			# categorized as.
			if self.verbose:
				fig, (ax1, ax2) = plt.subplots(2, 1)
				ax1.imshow(cimg)

				colored_img = cimg.copy().astype(np.uint8)
				for category in range(components):
					mask = (res == category).reshape(
						cimg.shape[0], cimg.shape[1]
					)

					colored_img[mask] = torgb(colors[category])

				ax2.imshow(colored_img)

				ax1.set_title("Contrast Image")
				ax2.set_title("Classification")
				plt.show()

		if self.verbose:
			print("done")
			print("", end='')




	# This uses a very simply process to convert each channel into a contrast
	# value. It assumes that the most common pixel value in each channel is 
	# the background value for that channel.
	def getContrast(self, img, threshold=0.05):
		return getImgContrast(img, threshold)



def getImgContrast(img, threshold=0.05):
	img = cv2.medianBlur(img, 3)
	#img = cv2.bilateralFilter(img, 5, 20, 80)
	# img = cv2.fastNlMeansDenoisingColored(
	# 	img, 
	# 	30, 
	# 	30
	# )
	# Separate the channels.
	R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

	def getMode(channel):
		return np.argmax(np.bincount(channel.flatten()))

	R_mode = getMode(R)
	G_mode = getMode(G)
	B_mode = getMode(B)

	# Now we convert to floating point for the contrast calculation.
	R = R.astype(np.float32)
	G = G.astype(np.float32)
	B = B.astype(np.float32)

	R_con = -(R - R_mode) / R_mode
	G_con = -(G - G_mode) / G_mode
	B_con = -(B - B_mode) / B_mode

	# Set everything less than the threshold value to zero.
	R_con[R_con < threshold] = 0
	G_con[G_con < threshold] = 0
	B_con[B_con < threshold] = 0

	# Return the contrast image.
	return np.stack([R_con, G_con, B_con], axis=2)


class MultiGaussianModel:
	def __init__(self, n, centers, heights, width=0.002):
		self.n_gaussians   = n
		self.centers       = centers
		self.heights       = heights
		self.initial_width = width

	def getFunction(self):
		def fn(args, x):
			result = args[0] * np.exp(-np.square(x - args[1]) / (args[2]))
			for n in range(1, self.n_gaussians):
				a = args[n * 3]
				b = args[n * 3 + 1]
				c = args[n * 3 + 2]
				result += a * np.exp(-np.square(x - b) / c)

			result += args[-1]
			return result
		return fn

	def getErrorFn(self):
		fn = self.getFunction()
		def errFn(args, x, y):
			return fn(args, x) - y

		return errFn

	def fit(self, x, y):
		p0 = []
		for n in range(self.n_gaussians):
			p0.append(self.heights[n])
			p0.append(self.centers[n])
			p0.append(self.initial_width)
		p0.append(0.0)

		errFunc = self.getErrorFn()
		res = least_squares(
			errFunc, x0=p0, bounds=(0, np.inf),
			args=(x, y)
		)

		self.x      = x
		self.y      = y
		self.params = res['x']

	def normalize(self):
		# Once the model has been trained, this calculates the sum of each 
		# gaussian so that they can be used to assign scores to pixels based
		# on their proximity to the gaussian peaks.
		self.normalizations = []
		for g in range(self.n_gaussians):
			# Analytically, the sum of a gaussian of this form must be:
			# a*sqrt(c * pi)
			a = self.params[g * 3]
			c = self.params[g * 3 + 2]
			self.normalizations.append(a * np.sqrt(c * np.pi))

	def score(self, samples):
		self.normalize()
		# Now we evaluate each gaussian (normalized) for the x value "sample"
		# provided. We take this as the "score".
		scores = []
		for n in range(self.n_gaussians):
			A = self.normalizations[n]
			a = self.params[n * 3]
			b = self.params[n * 3 + 1]
			c = self.params[n * 3 + 2]
			score = (a / A) * np.exp(-np.square(samples - b) / c)
			scores.append(score)

		# We should now have an array where each element in the first
		# dimension corresponds to a gaussian and each second dimension
		# corresponds to a sample.
		scores = np.array(scores).T

		# scores is now formatted like:
		# [sample_1_g1_score, sample_1_g2_score ...]
		# .
		# .
		# .
		# [sample_n_g1_score, sample_n_g2_score ...]

		# Now we determine the classification by taking the index of the gaussian
		# for which the score is the highest across each row.
		return np.argmax(scores, axis=1)
		

	def getInitialFunction(self):
		p0 = []
		for n in range(self.n_gaussians):
			p0.append(self.heights[n])
			p0.append(self.centers[n])
			p0.append(self.initial_width)
		p0.append(0.0)

		def fn(x):
			result = p0[0] * np.exp(-np.square(x - p0[1]) / (p0[2]))
			for n in range(1, self.n_gaussians):
				a = p0[n * 3]
				b = p0[n * 3 + 1]
				c = p0[n * 3 + 2]
				result += a * np.exp(-np.square(x - b) / c)

			result += p0[-1]
			return result
		return fn


	def getFinalFunction(self):
		def fn(x):
			result = self.params[0] * np.exp(-np.square(x - self.params[1]) / (self.params[2]))
			for n in range(1, self.n_gaussians):
				a = self.params[n * 3]
				b = self.params[n * 3 + 1]
				c = self.params[n * 3 + 2]
				result += a * np.exp(-np.square(x - b) / c)

			result += self.params[-1]
			return result

		return fn

	def plotIndividualGaussian(self, x, n):
		a = self.params[n * 3]
		b = self.params[n * 3 + 1]
		c = self.params[n * 3 + 2]
		result = a * np.exp(-np.square(x - b) / c)
		return result

	def getRMSE(self):
		fn   = self.getFinalFunction()
		rmse = np.sqrt(np.square(fn(self.x) - self.y).mean())
		return rmse.item()

	def getResidual(self):
		fn   = self.getFinalFunction()
		return fn(self.x) - self.y




if __name__ == '__main__':
	# Load the arguments file. 
	with open("UnsupervisedImageProcessing.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	if args.contrast_data != "":
		img = cv2.imread(args.contrast_data)
		img = cv2.bilateralFilter(img, 5, 50, 50)

		plt.imshow(img)
		plt.show()

		contrast_img = getImgContrast(img, threshold=0.03)

		data = contrast_img.reshape(
			contrast_img.shape[0] * contrast_img.shape[1],
			contrast_img.shape[2]
		)

		# Remove all of the zeroes from the histogram data.
		r_data = data[:, 0]
		r_data = r_data[r_data != 0]

		g_data = data[:, 1]
		g_data = g_data[g_data != 0]

		b_data = data[:, 2]
		b_data = b_data[b_data != 0]


		g_kde = gaussian_kde(g_data)

		x = np.linspace(g_data.min(), g_data.max(), 256)
		y = g_kde(x)

		peaks, props = find_peaks(y, height=[0.02, 1000])

		#code.interact(local=locals())


		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.imshow(contrast_img[:, :, 1])
		ax2.plot(x, y)

		print("%d peaks"%(len(peaks)))
		print(x[peaks])
		print(props['peak_heights'])

		ax2.scatter(
			x[peaks], 
			props['peak_heights'], 
			color="red",
			marker="x"
		)
		# for peak in peaks:
		# 	ax2.axvline(x[peak], color="black", alpha=0.3)

		# for height in props['peak_heights']:
		# 	ax2.axhline(height, color="black", alpha=0.3)

		ax2.set_title("Contrast Value Distribution")
		ax2.set_xlabel("Contrast")
		ax2.set_ylabel("P(contrast)")

		ax1.set_title("Contrast (Green Channel)")
		plt.show()

		halo_img = contrast_img[:, :, 1].copy()
		halo_img[halo_img > 1.6] = 0.0
		plt.imshow(halo_img)
		plt.title("Halo Image")
		plt.show()

		# print(len(peaks))
		# print(x[peaks])
		# print(props['peak_heights'])

		# Attempt a fit.
		# This will add an additional gaussian to the model that should help 
		# fit any broad background.
		centers = x[peaks]
		heights = props['peak_heights']
		centers = list(x[peaks])
		heights = list(props['peak_heights'])
		# centers.append((x[-1] - x[0]) / 2)
		# heights.append(0.5)
		# centers = np.array(centers)
		# heights = np.array(heights)

		rmse       = 10.0
		attempts   = 0
		max_rmse   = 0.03
		last_rmse  = 10
		rmse_step  = 1.5  # If the rmse isn't at least 50% better, stop
		last_model = None

		while rmse > max_rmse and attempts < 5:
			model = MultiGaussianModel(
				len(centers), 
				np.array(centers), 
				np.array(heights)
			)
			model.fit(x, y)
			rmse = model.getRMSE()
			print("RMSE: " + str(rmse))

			if (last_rmse - rmse) / rmse < rmse_step:
				print("Additional gaussian did not improve error significantly")
				print("Stopping optimization")
				model = last_model
				break

			if rmse > max_rmse:
				# calculate the residual and find any peaks in it.
				res = model.getResidual()
				peaks, props = find_peaks(res, height=[0.005, 1000])
				if len(peaks) < 1:
					break
				# Add only the highest peak in the residual to the list.
				highest_arg = np.argmax(props['peak_heights'])
				centers.append(x[highest_arg])
				heights.append(props['peak_heights'][highest_arg])

				last_model = deepcopy(model)
				last_rmse  = rmse
				# for i, p in enumerate(peaks):
				# 	centers.append(x[p])
				# 	heights.append(props['peak_heights'][i])

			attempts += 1

		print("Final Characteristics")
		print("    %d peaks"%(len(model.centers)))
		print("    centers: %s"%str(model.centers))
		print("    heights: %s"%str(model.heights))

		fn  = model.getFinalFunction()
		ifn = model.getInitialFunction()

		s1, = plt.plot(x, y, linewidth=3)
		s2, = plt.plot(x, fn(x), linestyle='dashed')
		s3, = plt.plot(x, ifn(x), linestyle='dashed', alpha=0.3)

		for n in range(len(model.centers)):
			plt.plot(x, model.plotIndividualGaussian(x, n), alpha=0.4)
		plt.legend([s1, s2, s3], ['Data', 'Fit', 'Initial'])
		plt.show()

		width, height = contrast_img.shape[0], contrast_img.shape[1]
		samples = contrast_img[:, :, 1].copy().reshape(width * height)
		scores = model.score(samples)

		fig, (ax1, ax2) = plt.subplots(2, 1)
		colored_img = scores.reshape(width, height)
		colored_img[contrast_img[:, :, 1] == 0] = -1.0
		ax1.imshow(contrast_img[:, :, 1])
		ax2.imshow(colored_img)
		plt.show()

		
	else:
		m = Model()
		t = ModelTrainer(m, verbose=True)
		t.trainModel(args.data_path)

