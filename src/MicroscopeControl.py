# Author:      Adam Robinson
# Description: This class combines the functionality of the CameraController,
#              FocusController and StageController classes into a cohesive
#              class that performs all of the relevant work related to controlling
#              the microscope. 

from CameraControl import CameraController
from FocusControl  import FocusController
from StageControl  import StageController
from FocusControl  import autoConnect     as getFocusController
from StageControl  import autoConnect     as getStageController

import time
import code
import cv2

import matplotlib.pyplot as plt
import numpy             as np


class MicroscopeController:

	# This will initialize a microscope controller object and connect to all
	# of the devices necessary to get things working. The following keyword
	# arguments are valid:
	#     focus_port     : The COM port to use when connecting to the focus controller
	#     focus_baudrate : The baudrate to use when connecting to the focus controller
	#     stage_port     : The COM port to use when connecting to the stage controller
	#     stage_baudrate : The baudrate to use when connecting to the stage controller
	#     verbose        : Print information to the console when an action is being performed
	#
	# If the connection parameters for the focus and stage controllers are not 
	# specified, this class will attempt to connect to them automatically. If they
	# cannot be connected to, an exception will be thrown.
	def __init__(self, **kwargs):
		self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False

		if self.verbose:
			print("Connecting to focus controller . . . ", end='', flush=True)
		if 'focus_port' in kwargs or 'focus_baudrate' in kwargs:
			fp = kwargs['focus_port']
			fb = kwargs['focus_baudrate']
			self.focus = FocusController(fp, fb)
		else:
			self.focus = getFocusController()
			if self.focus is None:
				print("FAIL")
				raise Exception("Failed to connect to focus controller.")

		if self.verbose:
			print("DONE")

		if self.verbose:
			print("Connecting to stage controller . . . ", end='', flush=True)
		if 'stage_port' in kwargs or 'stage_baudrate' in kwargs:
			sp = kwargs['stage_port']
			sb = kwargs['stage_baudrate']
			self.stage = StageController(sp, sb)
		else:
			self.stage = getStageController()
			if self.stage is None:
				print("FAIL")
				raise Exception("Failed to connect to stage controller.")

		if self.verbose:
			print("DONE")

		if 'disable_camera' in kwargs:
			self.disable_camera = kwargs['disable_camera']
		else:
			self.disable_camera = False

		if not self.disable_camera:
			if self.verbose:
				print("Connecting to camera           . . . ", end='', flush=True)
			self.camera = CameraController()
			if self.verbose:
				print("DONE")

	def autoFocus(self, _range, n_divisions=10, passes=3, breakdown=2, mode='max', use_grad=True):
		# Save the current focus motor settings so that we can set them
		# when we are done.
		if n_divisions < 5:
			raise Exception("You must specify n_divisions >= 5.")

		focus_accel     = self.focus.getFocusAcceleration()
		focus_initial_v = self.focus.getFocusInitialVelocity()
		focus_final_v   = self.focus.getFocusMaxVelocity()

		# Set the initial and final velocities to be a very small
		# portion of the limit value, so that the motor will move
		# really slowly.
		z_max, f_max = self.focus.getLimits()

		# I'm guessing that this will cause it to take ten seconds to
		# Scan through the enstire focus range.
		self.focus.setFocusInitialVelocity(50)
		self.focus.setFocusMaxVelocity(500)

		# Seek to zero, then seek to the maximum and take an image 
		# every 50 milliseconds the entire way.

		self.camera.startCapture()
		self.camera.enableAutoExposure()

		current_range = _range
		for idx in range(passes):
			if self.verbose:
				print("Range: [%2.4f, %2.4f]"%(current_range[0], current_range[1]))

			steps = np.linspace(current_range[0], current_range[1], n_divisions)

			self.focus.setFocus(0.0)

			if self.verbose:
				print("Imaging focus range (pass %d)."%(idx + 1))

			images    = []
			positions = []
			for position in steps:
				self.focus.setFocus(position, corrected=False)
				images.append(self.camera.getFrame())
				current_focus = self.focus.getFocus()
				positions.append(current_focus)
				if self.verbose:
					print(".", end='', flush=True)

			

			if use_grad:
				if self.verbose:
					print('')
					print("Calculating gradients.")
				grad = []
				dx_filter = 0.5 * np.array(
					[[ 0, 0, 0],
					 [-1, 0, 1],
					 [ 0, 0, 0]]
				)

				dy_filter = 0.5 * np.array(
					[[ 0,  1, 0],
					 [ 0,  0, 0],
					 [ 0, -1, 0]]
				)

				for img in images:
					dx_img = cv2.filter2D(img, -1, dx_filter) * 2
					dy_img = cv2.filter2D(img, -1, dy_filter) * 2
					mimg   = dx_img + dy_img
					grad.append(mimg.mean())

				best_idx = np.argmax(np.array(grad))

				if self.verbose:
					print('', flush=True)
					plt.scatter(positions, grad, s=1)
					plt.show()

			else:
				if self.verbose:
					print('')
					print("Calculating entropy.")
				ent = []
				for img in images:
					ent.append(self._get_image_entropy(img, breakdown))
					if self.verbose:
						print(".", end='', flush=True)

				if self.verbose:
					print('', flush=True)
					plt.scatter(positions, ent, s=1)
					plt.show()

				if mode == 'max':
					best_idx = np.argmax(np.array(ent))
				elif mode == 'min':
					best_idx = np.argmin(np.array(ent))
				elif mode == 'auto':
					ent = np.array(ent)
					m   = ent.mean()
					mo  = (ent[0] + ent[-1]) / 2
					if mo > m:
						best_idx = np.argmin(np.array(ent))
						mode     = 'min' 
					else:
						best_idx = np.argmax(np.array(ent))
						mode     = 'max'
				else:
					raise Exception("Invalid mode specified, must be 'min' or 'max' or 'auto'")

			if best_idx < 2:
				start_new  = steps[0]
			else:
				start_new  = steps[best_idx - 2]

			if best_idx > len(steps) - 3:
				stop_new   = steps[-1]
			else:
				stop_new   = steps[best_idx + 2]

			current_range = [start_new, stop_new]


		self.camera.endCapture()

		focus_position = positions[best_idx]

		self.focus.setFocus(focus_position) 

		self.focus.setFocusAcceleration(focus_accel)
		self.focus.setFocusInitialVelocity(focus_initial_v)
		self.focus.setFocusMaxVelocity(focus_final_v)



	# This selects one sixteenth of the images pixels systematically
	# and uses them to estimate the entropy of the image in nats.
	def _get_image_entropy(self, img, breakdown):
		img           = img[::breakdown, ::breakdown, :]
		value, counts = np.unique(img, return_counts=True)
		n             = img.shape[0] * img.shape[1] * img.shape[2]
		P             = counts / n
		
		entropy = - (P * np.log(P)).sum()

		return entropy

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import sys
	def showimg():
		microscope.camera.startCapture()
		img = microscope.camera.getFrame(convert=True)
		plt.imshow(img)
		plt.show()
		microscope.camera.endCapture()

	camera_on = True
	if len(sys.argv) > 1:
		if sys.argv[1] == '--no-camera':
			camera_on = False

	microscope = MicroscopeController(
		verbose=True,
		disable_camera=(not camera_on)
	)

	code.interact(local=locals())
