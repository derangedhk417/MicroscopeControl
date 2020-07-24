import code
import sys
import os
import time
import cv2
import numpy             as np
import matplotlib.pyplot as plt

def debug_show(img, line):
	plt.imshow(img)
	plt.title("Line: %d"%line)
	plt.show()

# Provides all of the functionality necessary to take an image
# and extract flake geometry from it. This class assumes that the
# image passed to it is in BGR format.
class ImageProcessor:
	def __init__(self, img, **kwargs):
		self.raw_img   = img
		self.mode      = kwargs['mode']      if 'mode'      in kwargs else 'HSV'
		self.downscale = kwargs['downscale'] if 'downscale' in kwargs else 1.0

		if self.mode == 'HSV':
			self.img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2HSV)
		else:
			self.img = self.raw_img

		if self.downscale != 1.0:
			self.img = cv2.resize(
				self.img, 
				(0, 0), 
				fx=(1/self.downscale), 
				fy=(1/self.downscale)
			)

		self.current_process = None
		self.names           = None

		# Whether or not to store the result of every operation
		# so that they can be displayed.
		self.no_store        = False

	def noStore(self):
		self.no_store = True
		return self

	def reset(self):
		self.current_process = None
		self.no_store        = False

	def done(self):
		if self.no_store:
			proc = self.current_process[0]
		else:
			proc = self.current_process
		
		self.reset()

		return proc
		

	def denoise(self, strength):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		res = cv2.fastNlMeansDenoisingColored(
			self.current_process[-1], 
			strength
		)

		if self.no_store:
			self.current_process[-1] = res
		else:
			self.current_process.append(res)

		if not self.no_store:
			self.names.append("Fast NL Means Denoised (strength = %d)"%strength)

		return self

	# This attempts to use a laplacian filter, followed by a 
	# thresholding operation to detect the edges in the image.
	def laplacian(self, threshold):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		lpl = cv2.Laplacian(self.current_process[-1], -1, ksize=3).max(axis=2)
		if self.no_store:
			self.current_process[-1] = lpl
		else:
			self.current_process.append(lpl)
			self.names.append("Laplacian Filter (Max Across Channels)")

		mask           = lpl < threshold
		if self.no_store:
			filtered = lpl
		else:
			filtered       = lpl.copy()
		
		filtered[mask] = 0

		if self.no_store:
			self.current_process[-1] = filtered
		else:
			self.current_process.append(filtered)
			self.names.append("Laplacian Filter (Everything Below %d Removed)"%(
				threshold
			))

		return self

	def erode(self, scale):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))

		eroded = cv2.erode(self.current_process[-1], kernel)

		if self.no_store:
			self.current_process[-1] = eroded
		else:
			self.current_process.append(eroded)
			self.names.append("Eroded (%dx%d elliptical filter)"%(scale, scale))

		return self

	def dilate(self, scale):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))


		dilated = cv2.dilate(self.current_process[-1], kernel)
		if self.no_store:
			self.current_process[-1] = dilated
		else:
			self.current_process.append(dilated)
			self.names.append("Dilated (%dx%d elliptical filter)"%(scale, scale))

		return self

	def level(self):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		mask          = self.current_process[-1] > 0.0
		if self.no_store:
			leveled = self.current_process[-1]
		else:
			leveled = self.current_process[-1].copy()

		leveled[mask] = 1

		if self.no_store:
			self.current_process[-1] = leveled
		else:
			self.current_process.append(leveled)
			self.names.append("Leveled")

		return self

	def edge(self, p1, p2):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		edged = cv2.Canny(self.current_process[-1], p1, p2)

		if self.no_store:
			self.current_process[-1] = edged
		else:
			self.current_process.append(edged)
			self.names.append("Edge Detected (%d, %d)"%(p1, p2))

		return self

	def border(self, width, color):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]
		bordered = cv2.copyMakeBorder(
			self.current_process[-1], 
			width, 
			width, 
			width, 
			width, 
			cv2.BORDER_CONSTANT,
			value=color
		)

		if self.no_store:
			self.current_process[-1] = bordered
		else:
			self.current_process.append(bordered)
			self.names.append("Bordered")

		return self

	# Extracts the outermost contours from the images as arrays
	# of 2d coordinates.
	def extractContours(self, img):
		contours, heirarchy = cv2.findContours(
			img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		# The format that opencv uses for contours is strange. Here,
		# we reshape each array into a matrix of shape (n_points, 2).

		contours = [c.reshape(-1, 2) for c in contours]
		return contours

	# Calculates the minimum bounding rectange for each contour.
	# This includes rotation of the rectangle.
	def calculateBoundingRectangles(self, contours):
		# Here we reshape the contours back into the format that
		# opencv expects before passing them to the function.
		return [cv2.minAreaRect(c.reshape(-1, 1, 2)) for c in contours]


	def display(self):
		if self.no_store:
			raise Exception("Current process is in no_store mode.")

		for img, title in zip(self.current_process, self.names):
			plt.imshow(img)
			plt.title(title)
			plt.show()

		return self

class FlakeExtractor:
	def __init__(self, img, **kwargs):
		self.img  = img
		self.proc = ImageProcessor(
			self.img, 
			downscale=kwargs['downscale']
		)
		self.threshold      = kwargs['threshold']
		self.contrast_floor = kwargs['contrast_floor'] 

	def process(self, DEBUG_DISPLAY=False):
		if DEBUG_DISPLAY:
			debug_show(self.img, sys._getframe().f_lineno)
		# This set of filters will produce good edges for the contour
		# algorithm. The numerical parameters are based on the assumption
		# that the image is 2448x2048 and that it is downscaled by a factor
		# of 5.
		if not DEBUG_DISPLAY:
			tmp = self.proc.noStore().denoise(30).laplacian(28).dilate(4).erode(4)
			res = tmp.level().border(5, 0).edge(0, 1).dilate(2).done()
		else:
			tmp = self.proc.denoise(30).laplacian(28).dilate(4).erode(4)
			res = tmp.level().border(5, 0).edge(0, 1).dilate(2).display().done()[-1]

		if DEBUG_DISPLAY:
			debug_show(res, sys._getframe().f_lineno)

		# Extract contours and bounding rectangles.
		c   = self.proc.extractContours(res)
		r   = self.proc.calculateBoundingRectangles(c)

		# Correct the coordinates for the border we added.
		corrected_rects    = []
		corrected_contours = []
		for rect, contour in zip(r, c):
			((left, top), (width, height), angle) = rect

			# Calculate the relative size of the largest dimension, making
			# sure to use the units of the downscaled image.
			if width > height:
				largest = width / self.proc.img.shape[1]
			else:
				largest = height / self.proc.img.shape[0]

			# Correct for the border.
			left    -= 5
			top     -= 5
			contour -= 5


			if largest > self.threshold:
				corrected_rects.append(((left, top), (width, height), angle))
				corrected_contours.append(contour)

		if len(corrected_rects) == 0:
			# Return false to indicate that this image should be thrown out.
			return False, None

		# We now have what we need to subtract the background and compute contrast
		# values. Once the background is subtracted, it becomes easier to determine
		# what is garbage and what is not. This is mostly due to the fact that adhesive
		# residue is usually darker than the background and flakes are almost always 
		# brighter than the background.


		# Mask out the stuff that isn't background to compute the background color.
		mask = np.zeros((
			self.proc.img.shape[0], 
			self.proc.img.shape[1]
		)).astype(np.uint8)

		for c in corrected_contours:
			con  = np.array(c)
			con  = con.astype(np.int32)
			mask = cv2.fillPoly(mask, [con], 1)

		mask = mask.astype(np.uint8)

		if DEBUG_DISPLAY:
			debug_show(mask, sys._getframe().f_lineno)

		bg_mask  = mask == 0
		bg       = self.proc.img[bg_mask]

		bg_color = bg.mean(axis=0)

		bg_subtracted = (self.img.astype(np.float32) - bg_color)
		bg_subtracted[bg_subtracted < 3]   = 0
		bg_subtracted[bg_subtracted > 255] = 255
		bg_subtracted = bg_subtracted.astype(np.uint8)

		bg_removed = cv2.cvtColor(bg_subtracted, cv2.COLOR_BGR2GRAY)
		if DEBUG_DISPLAY:
			debug_show(bg_removed, sys._getframe().f_lineno)
		bg_removed[bg_removed < self.contrast_floor] = 0

		# Now we reprocess flake boundaries.
		proc = ImageProcessor(bg_removed, downscale=1, mode='GS')
		if not DEBUG_DISPLAY:
			tmp  = proc.noStore().level().erode(7)
			res  = tmp.border(5, 0).edge(0, 1).dilate(2).done()
		else:
			tmp  = proc.level().erode(7)
			res  = tmp.border(5, 0).edge(0, 1).dilate(2).display().done()[-1]

		if DEBUG_DISPLAY:
			debug_show(res, sys._getframe().f_lineno)

		# Now we determine bounding boxes and remove everything thats too small.
		# Extract contours and bounding rectangles.
		c = proc.extractContours(res)
		r = proc.calculateBoundingRectangles(c)

		# Correct the coordinates for the border we added.
		corrected_rects    = []
		corrected_contours = []
		for rect, contour in zip(r, c):
			((left, top), (width, height), angle) = rect

			# Calculate the relative size of the largest dimension, making
			# sure to use the units of the downscaled image.
			if width > height:
				largest = width / proc.img.shape[1]
			else:
				largest = height / proc.img.shape[0]

			# Correct for the border.
			left    -= 5
			top     -= 5
			contour -= 5

			if largest > self.threshold:
				corrected_rects.append(((left, top), (width, height), angle))
				corrected_contours.append(contour)

		if len(corrected_rects) == 0:
			# Return false to indicate that this image should be thrown out.
			return False, None


		# Now we convert the contour and rectangle information into
		# relative units.
		converted_contours  = []
		converted_rects     = []
		converted_rot_rects = []

		for r, c in zip(corrected_rects, corrected_contours):
			# Get the four points that represent the corners of the bounding
			# rectangle and convert them.
			box_points = cv2.boxPoints(r).astype(np.float32)
			box_points[:, 0] = box_points[:, 0] / proc.img.shape[1]
			box_points[:, 1] = box_points[:, 1] / proc.img.shape[0]

			((left, top), (width, height), angle) = r
			left = left / proc.img.shape[1]
			top  = top  / proc.img.shape[0]

			width  = width  / proc.img.shape[1]
			height = height / proc.img.shape[0]

			converted_rot_rects.append(((left, top), (width, height), angle))

			con = c.astype(np.float32)
			# Convert the contour points.
			con[:, 0] = con[:, 0] / proc.img.shape[1]
			con[:, 1] = con[:, 1] / proc.img.shape[0]

			converted_contours.append(con.tolist())
			converted_rects.append(box_points.tolist())


		results = {
			"contours"  : converted_contours,
			"rects"     : converted_rects,
			"rot_rects" : converted_rot_rects,
			"bg_color"  : bg_color.tolist()
		}

		bg_downscaled       = cv2.resize(bg_subtracted, (0, 0), fx=0.25, fy=0.25)
		original_downscaled = cv2.resize(self.img, (0, 0), fx=0.25, fy=0.25)

		# Create a mask that contains all of the flakes and extract the contrast
		# values from them.
		contrast_mask = np.zeros((
			bg_downscaled.shape[0], 
			bg_downscaled.shape[1]
		)).astype(np.uint8)

		for contour in results['contours']:
			con  = np.array(contour)
			con[:, 0] *= bg_downscaled.shape[1]
			con[:, 1] *= bg_downscaled.shape[0]
			con  = con.astype(np.int32)
			contrast_mask = cv2.fillPoly(contrast_mask, [con], 1)


		rgb      = bg_downscaled[contrast_mask == 1]
		b_values = rgb[:, 0]
		g_values = rgb[:, 1]
		r_values = rgb[:, 2] 
		results['r_values'] = r_values.tolist()
		results['g_values'] = g_values.tolist()
		results['b_values'] = b_values.tolist()

		contrast_img = bg_downscaled / (bg_color + original_downscaled)
		if DEBUG_DISPLAY:
			debug_show(contrast_img, sys._getframe().f_lineno)

		contrast     = contrast_img[contrast_mask == 1].sum(axis=1)
		results['contrast_values'] = contrast.tolist()

		if DEBUG_DISPLAY:
			# Overlay the contours on the original image.
			c_img = self.img.copy()
			# Draw an image with a contour around every flake that we decided was good.
			for con in results['contours']:
				con = np.array(con)
				con[:, 0] = con[:, 0] * self.img.shape[1]
				con[:, 1] = con[:, 1] * self.img.shape[0]
				con = con.astype(np.int32)
				c_img = cv2.drawContours(
					c_img, 
					[con.reshape(-1, 1, 2)], 
					0, (255, 0, 0), 2
				)

			plt.imshow(c_img)
			plt.title("Line: %d"%sys._getframe().f_lineno)
			plt.show()	

		return True, results