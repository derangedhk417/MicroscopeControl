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

	def reset(self):
		self.current_process = None

	def done(self):
		proc = self.current_process
		self.reset()
		return proc

	def denoise(self, strength):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		self.current_process.append(cv2.fastNlMeansDenoisingColored(
			self.current_process[-1], 
			strength
		))

		self.names.append("Fast NL Means Denoised (strength = %d)"%strength)

		return self

	# This attempts to use a laplacian filter, followed by a 
	# thresholding operation to detect the edges in the image.
	def laplacian(self, threshold):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		lpl = cv2.Laplacian(self.current_process[-1], -1, ksize=3).max(axis=2)
		self.current_process.append(lpl)
		self.names.append("Laplacian Filter (Max Across Channels)")

		mask           = lpl < threshold
		filtered       = lpl.copy()
		filtered[mask] = 0

		self.current_process.append(filtered)
		self.names.append("Laplacian Filter (Everything Below %d Removed)"%threshold)

		return self

	def erode(self, scale):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))

		eroded = cv2.erode(self.current_process[-1], kernel)

		self.current_process.append(eroded)
		self.names.append("Eroded (%dx%d elliptical filter)"%(scale, scale))

		return self

	def dilate(self, scale):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))

		dilated = cv2.dilate(self.current_process[-1], kernel)
		
		self.current_process.append(dilated)
		self.names.append("Dilated (%dx%d elliptical filter)"%(scale, scale))

		return self

	def level(self):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		mask          = self.current_process[-1] > 0.0
		leveled       = self.current_process[-1].copy()
		leveled[mask] = 1
		self.current_process.append(leveled)
		self.names.append("Leveled")

		return self

	def edge(self, p1, p2):
		if self.current_process is None:
			self.current_process = [self.img]
			self.names           = ["Original"]

		edged = cv2.Canny(self.current_process[-1], p1, p2)
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

		self.current_process.append(bordered)
		self.names.append("Bordered")

		return self

	def display(self):
		for img, title in zip(self.current_process, self.names):
			plt.imshow(img)
			plt.title(title)
			plt.show()

		return self

if __name__ == '__main__':
	imgname = sys.argv[1]

	proc = ImageProcessor(cv2.imread(imgname), downscale=5)

	proc.denoise(30).laplacian(28).dilate(4).erode(4).level().border(5, 0).edge(0, 1).dilate(2).display().done()