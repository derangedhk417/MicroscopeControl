# Author:      Adam Robinson
# Description: This class combines the functionality of the CameraController,
#              FocusController and StageController classes into a cohesive
#              class that performs all of the relevant work related to controlling
#              the microscope. 

from CameraControl  import CameraController
from FocusControl   import FocusController
from StageControl   import StageController
from FocusControl   import autoConnect     as getFocusController
from StageControl   import autoConnect     as getStageController
from scipy.optimize import curve_fit

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

	# def autoFocus(self, _range, n_divisions=100, passes=1, n_avg=1, display=True):
	# 	# Save the current focus motor settings so that we can set them
	# 	# when we are done.
	# 	if n_divisions < 5:
	# 		raise Exception("You must specify n_divisions >= 5.")

	# 	focus_accel     = self.focus.getFocusAcceleration()
	# 	focus_initial_v = self.focus.getFocusInitialVelocity()
	# 	focus_final_v   = self.focus.getFocusMaxVelocity()

	# 	# Set the initial and final velocities to be a very small
	# 	# portion of the limit value, so that the motor will move
	# 	# really slowly.
	# 	z_max, f_max = self.focus.getLimits()

	# 	# I'm guessing that this will cause it to take ten seconds to
	# 	# Scan through the enstire focus range.
	# 	self.focus.setFocusInitialVelocity(50)
	# 	self.focus.setFocusMaxVelocity(500)

	# 	# Seek to zero, then seek to the maximum and take an image 
	# 	# every 50 milliseconds the entire way.

	# 	self.camera.startCapture()
	# 	self.camera.enableAutoExposure()
	# 	time.sleep(1)
	# 	self.camera.disableAutoExposure()

	# 	current_range = _range
	# 	for idx in range(passes):
	# 		if self.verbose:
	# 			print("Range: [%2.4f, %2.4f]"%(current_range[0], current_range[1]))

	# 		steps = np.linspace(current_range[0], current_range[1], n_divisions)

	# 		self.focus.setFocus(0.0)

	# 		if self.verbose:
	# 			print("Imaging focus range (pass %d)."%(idx + 1))

	# 		images    = []
	# 		positions = []
	# 		for position in steps:
	# 			time.sleep(0.05)
	# 			self.focus.setFocus(position, corrected=False)

	# 			current_images = []
	# 			for i in range(n_avg):
	# 				current_images.append(self.camera.getFrame(downscale=2))
					

	# 			_sum = current_images[0]
	# 			if n_avg > 1:
	# 				for img in current_images[1:]:
	# 					_sum += img

	# 			_sum = _sum / n_avg
	# 			images.append(_sum)
				
	# 			current_focus = self.focus.getFocus()
	# 			positions.append(current_focus)
	# 			if self.verbose:
	# 				print(".", end='', flush=True)

			
	# 		if self.verbose:
	# 			print('')
	# 			print("Calculating gradients.")
	# 		grad = []
	# 		dx_filter = 0.5 * np.array(
	# 			[[ 0, 0, 0],
	# 			 [-1, 0, 1],
	# 			 [ 0, 0, 0]]
	# 		)

	# 		dy_filter = 0.5 * np.array(
	# 			[[ 0,  1, 0],
	# 			 [ 0,  0, 0],
	# 			 [ 0, -1, 0]]
	# 		)

	# 		smooth_filter = (1 / 9) * np.array(
	# 			[[ 1, 1, 1],
	# 			 [ 1, 1, 1],
	# 			 [ 1, 1, 1]]
	# 		)

	# 		for img in images:
	# 			img      = cv2.filter2D(img, -1, smooth_filter)
	# 			grad_img = cv2.filter2D(img, -1, dx_filter + dy_filter)
				
	# 			# Select all gradients 1 std above the mean.
	# 			std  = grad_img.std()
	# 			mean = grad_img.mean()
	# 			idx  = (grad_img - mean) > (std)
	# 			grad.append(grad_img[idx].mean())
	# 			# grad.append(grad_img.mean())
	# 			if self.verbose:
	# 				print(".", end='', flush=True)

	# 		# Calculate a moving average of the data in order to smooth it.
	# 		window_size = 5
	# 		grad = np.pad(grad, (window_size, window_size), mode='edge')
	# 		new_grad = []
	# 		for idx in range(len(grad) - 2*window_size):
	# 			center = idx + window_size
	# 			_range = grad[center - window_size:center + window_size]
	# 			new_grad.append(_range.mean())

	# 		grad      = np.array(new_grad)
	# 		positions = np.array(positions)

	# 		test_fn          = lambda x, m, a, mu, sigma, b: m*x + a*np.exp(-np.square((x - mu) / sigma)) + b
	# 		initial          = [0.1, 0.1, grad.max() - grad.min(), positions[grad.argmax()], 0.02]
	# 		res, cov         = curve_fit(test_fn, positions, grad)
	# 		fn_linear        = lambda x: res[0]*x + res[4]
	# 		fn_gauss         = lambda x: res[1]*np.exp(-np.square((x - res[2]) / res[3]))
	# 		linear_corrected = grad - fn_linear(positions)
	# 		gauss_data       = fn_gauss(positions)

	# 		rmse = np.sqrt(np.square(linear_corrected - gauss_data).mean())

	# 		# Throw out any serious outliers.
	# 		grad     = np.array(grad)
	# 		best_idx = np.argmax(gauss_data)

	# 		print("rmse: %f"%rmse)
	# 		if self.verbose:
	# 			print('', flush=True)
	# 			a, = plt.plot(positions, grad)
	# 			b, = plt.plot(positions, linear_corrected)
	# 			c, = plt.plot(positions, gauss_data)
				
	# 			plt.axvline(positions[best_idx])
	# 			plt.legend([a, b, c], ["Data", "Linear Component Removed", "Gaussian Fit"])
	# 			plt.show()
	# 		# if rmse > 0.1:
	# 		# 	return False

	# 		if self.verbose:
	# 			print('', flush=True)
			

	# 		if best_idx < 2:
	# 			start_new  = positions[0]
	# 		else:
	# 			start_new  = positions[best_idx - 2]

	# 		if best_idx > len(positions) - 3:
	# 			stop_new   = positions[-1]
	# 		else:
	# 			stop_new   = positions[best_idx + 2]

	# 		current_range = [start_new, stop_new]

	# 	focus_position = positions[best_idx]

	# 	self.focus.setFocus(focus_position) 

	# 	self.camera.enableAutoExposure()

	# 	self.focus.setFocusAcceleration(focus_accel)
	# 	self.focus.setFocusInitialVelocity(focus_initial_v)
	# 	self.focus.setFocusMaxVelocity(focus_final_v)

	# 	return True

	# def autoFocus(self, _range, n_divisions=50):
	# 	# Save the current focus motor settings so that we can set them
	# 	# when we are done.
	# 	if n_divisions < 5:
	# 		raise Exception("You must specify n_divisions >= 5.")

	# 	focus_accel     = self.focus.getFocusAcceleration()
	# 	focus_initial_v = self.focus.getFocusInitialVelocity()
	# 	focus_final_v   = self.focus.getFocusMaxVelocity()

	# 	# Set the initial and final velocities to be a very small
	# 	# portion of the limit value, so that the motor will move
	# 	# really slowly.
	# 	z_max, f_max = self.focus.getLimits()

	# 	# I'm guessing that this will cause it to take ten seconds to
	# 	# Scan through the enstire focus range.
	# 	self.focus.setFocusInitialVelocity(50)
	# 	self.focus.setFocusMaxVelocity(500)

	# 	# Seek to zero, then seek to the maximum and take an image 
	# 	# every 50 milliseconds the entire way.

	# 	self.camera.startCapture()
	# 	self.camera.enableAutoExposure()
	# 	time.sleep(1)
	# 	self.camera.disableAutoExposure()

	
	# 	if self.verbose:
	# 		print("Range: [%2.4f, %2.4f]"%(_range[0], _range[1]))

	# 	steps = np.linspace(_range[0], _range[1], n_divisions)

	# 	self.focus.setFocus(0.0)

	# 	def avgimg(n):
	# 		imgs = []
	# 		for i in range(n):
	# 			img = self.camera.getFrame(downscale=3)
	# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 			imgs.append(img)

	# 		base = np.zeros(imgs[0].shape)
	# 		for i in imgs:
	# 			base += i

	# 		return (base / n).astype(np.uint8)

	# 	images    = []
	# 	positions = []
	# 	for position in steps:
	# 		#time.sleep(0.05)
	# 		self.focus.setFocus(position, corrected=False)
	# 		# img = self.camera.getFrame(downscale=3)
	# 		# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 		img = avgimg(3)
	# 		cv2.imshow("preview", img)
	# 		cv2.waitKey(1)
	# 		images.append(img)
	# 		current_focus = self.focus.getFocus()
	# 		positions.append(current_focus)

	# 		if self.verbose:
	# 			print(".", end='', flush=True)

	# 	# Smooth the images with a moving average and calculate the temporal gradient

	# 	smooth_filter = (1 / 9) * np.array(
	# 		[[ 1, 1, 1],
	# 		 [ 1, 1, 1],
	# 		 [ 1, 1, 1]]
	# 	)

	# 	if self.verbose:
	# 		print('')
	# 		print("Denoising.")

	# 	tmp_images   = []
	# 	stds         = []
	# 	max_laplace  = []
	# 	mean_laplace = []
	# 	for img in images:
	# 		#img = cv2.filter2D(img, -1, smooth_filter)
	# 		img = cv2.fastNlMeansDenoising(img, 20)
	# 		lpl = cv2.Laplacian(img, -1, ksize=3)
	# 		max_laplace.append(lpl.max())
	# 		mean_laplace.append(lpl.mean())
	# 		stds.append(img.std())
	# 		cv2.imshow("preview", img)
	# 		cv2.waitKey(1)
	# 		tmp_images.append(img)
	# 		if self.verbose:
	# 			print(".", end='', flush=True)
	# 	images = tmp_images


		
	# 	if self.verbose:
	# 		print('')
	# 		print("Calculating gradients.")

	# 	grad           = []
	# 	grad_upper     = []
	# 	grad_positions = []
	# 	for idx in range(len(images) - 1):
	# 		pos0 = positions[idx]
	# 		pos1 = positions[idx + 1]
	# 		img0 = images[idx]
	# 		img1 = images[idx + 1]


	# 		_grad = np.abs(img1 - img0)
	# 		std   = _grad.std()
	# 		mean  = _grad.mean()
	# 		upper = (_grad - mean) > (std/2)

	# 		grad_upper.append(_grad[upper].mean())
	# 		grad.append(mean)
	# 		grad_positions.append((pos0 + pos1) / 2)
	# 		if self.verbose:
	# 			print(".", end='', flush=True)
		

	# 	grad     = np.array(grad)
	# 	best_idx = np.argmax(max_laplace)

	# 	if self.verbose:
	# 		print('', flush=True)
	# 		a, = plt.plot(grad_positions, grad)
	# 		b, = plt.plot(grad_positions, grad_upper)
	# 		c, = plt.plot(positions, stds)
	# 		d, = plt.plot(positions, max_laplace)
	# 		e, = plt.plot(positions, mean_laplace)
			
	# 		plt.axvline(positions[best_idx])
	# 		plt.legend([a, b, c, d, e], ["Mean Temporal Gradient", "Above 1 STD Temporal Gradient", "STD", "Max Laplace", "Mean Laplace"])
	# 		plt.show()
		

	# 	if self.verbose:
	# 		print('', flush=True)
		

	# 	focus_position = positions[best_idx]

	# 	self.focus.setFocus(focus_position) 

	# 	self.camera.enableAutoExposure()

	# 	self.focus.setFocusAcceleration(focus_accel)
	# 	self.focus.setFocusInitialVelocity(focus_initial_v)
	# 	self.focus.setFocusMaxVelocity(focus_final_v)

	# 	return True

	def autoFocus(self, _range, n_divisions=100, passes=1):
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
		time.sleep(1)
		#self.camera.disableAutoExposure()

	
		current_range = _range
		for p in range(passes):
			if self.verbose:
				print("Range: [%2.4f, %2.4f]"%(current_range[0], current_range[1]))

			steps = np.linspace(current_range[0], current_range[1], n_divisions)

			self.focus.setFocus(0.0)

			def avgimg(n):
				imgs = []
				for i in range(n):
					img = self.camera.getFrame(downscale=3)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					imgs.append(img)

				base = np.zeros(imgs[0].shape)
				for i in imgs:
					base += i

				return (base / n).astype(np.uint8)

			images    = []
			positions = []
			for position in steps:
				#time.sleep(0.05)
				self.focus.setFocus(position, corrected=False)
				# img = self.camera.getFrame(downscale=3)
				# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img = avgimg(1)
				cv2.imshow("Focus Preview", img)
				cv2.waitKey(1)
				images.append(img)
				current_focus = self.focus.getFocus()
				positions.append(current_focus)

				if self.verbose:
					print(".", end='', flush=True)

			# Smooth the images with a moving average and calculate the temporal gradient

			if self.verbose:
				print('')
				print("Denoising and calculating metrics.")

			tmp_images   = []
			stds         = []
			max_laplace  = []
			mean_laplace = []
			for img in images:
				img = cv2.fastNlMeansDenoising(img, 20)
				lpl = cv2.Laplacian(img, -1, ksize=3)

				std  = lpl.std()
				mean = lpl.mean()
				high = np.abs(lpl - mean) > 2*std
				#max_laplace.append(lpl[high].mean())
				max_laplace.append(lpl.max())
				mean_laplace.append(lpl.mean())
				cv2.imshow("Lacplacian", lpl)
				cv2.imshow("Focus Preview", img)
				cv2.waitKey(1)
				tmp_images.append(img)
				if self.verbose:
					print(".", end='', flush=True)
			images = tmp_images


			best_idx = np.argmax(max_laplace)
			_max = np.array(max_laplace).max()
			
			focus_position = positions[best_idx]
			print(focus_position)
			# if self.verbose:
			# 	print('', flush=True)
			# 	d, = plt.plot(positions, max_laplace)
			# 	e, = plt.plot(positions, mean_laplace)
				
			# 	plt.axvline(positions[best_idx])
			# 	plt.legend([d, e], ["Max Laplace", "Mean Laplace"])
			# 	plt.show()
			

			if self.verbose:
				print('', flush=True)
			

			width          = current_range[1] - current_range[0]
			current_range  = (focus_position - width / 5, focus_position + width / 5)
			

		self.focus.setFocus(focus_position) 

		self.camera.enableAutoExposure()

		self.focus.setFocusAcceleration(focus_accel)
		self.focus.setFocusInitialVelocity(focus_initial_v)
		self.focus.setFocusMaxVelocity(focus_final_v)

		return True



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
