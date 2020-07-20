import code
import sys
import os
import time
import cv2
import numpy             as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
	imgname = sys.argv[1]

	original = cv2.imread(imgname)

	proc = ImageProcessor(original, downscale=5)

	tmp = proc.noStore().denoise(30).laplacian(28).dilate(4).erode(4)
	res = tmp.level().border(5, 0).edge(0, 1).dilate(2).done()


	c   = proc.extractContours(res)
	r   = proc.calculateBoundingRectangles(c)
	img = proc.img

	# Filter out everything smaller than this.
	threshold = 14

	filtered = []
	for rect in r:
		((left, top), (width, height), angle) = rect
		largest = max(width, height)

		left -= 5
		top  -= 5

		if largest > threshold:
			filtered.append(((left, top), (width, height), angle))


	for rect in filtered:
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

	plt.imshow(img)
	plt.show()
