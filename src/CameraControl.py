from pixelinkWrapper import *
from ctypes          import *

import numpy as np
import cv2
import os
import code
import time
import atexit

# Default resolution for our camera is: 2448 x 2048
class CameraController:
	def __init__(self):
		ret = PxLApi.initialize(0)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Unable to initialize a camera.")

		self.camera_handle    = ret[1]
		self.pixel_type       = 0      # PT_COLOR
		self.raw_image_buffer = create_string_buffer(5000 * 5000 * 2)

	# This needs to be called before capturing images from the camera.
	def startCapture(self):
		ret = PxLApi.setStreamState(
			self.camera_handle, 
			PxLApi.StreamState.START
		)

		if not PxLApi.apiSuccess(ret[0]):
			PxLApi.uninitialize(self.camera_handle)
			raise Exception("Unable to start the stream on the camera.")

	# This should be called when you are done capturing images.
	def endCapture(self):
		ret = PxLApi.setStreamState(
			self.camera_handle, 
			PxLApi.StreamState.STOP
		)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to end capture state.")

	# Turns on automatic exposure adjustment.
	def enableAutoExposure(self):
		exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
		params = [exposure]

		ret = PxLApi.getFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not read auto-exposure status.")

		params = ret[2]
		flags  = PxLApi.FeatureFlags.AUTO

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE, 
			flags, 
			params
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not set auto-exposure to on.")

	# Disabled automatic exposure adjustment.
	def disableAutoExposure(self):
		exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
		params = [exposure]

		ret = PxLApi.getFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not read auto-exposure status.")

		params = ret[2]
		flags  = PxLApi.FeatureFlags.MANUAL

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE, 
			flags, 
			params
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not set auto-exposure to on.")


	def __del__(self):
		self.cleanup()

	def cleanup(self):
		ret = PxLApi.uninitialize(self.camera_handle)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to disconnect from camera.")


	# Returns a frame. By default, this will be in the BGR pixel format.
	# Set convert=True to get an RGB frame.
	def getFrame(self, convert=False):
		ret = PxLApi.getNextFrame(
			self.camera_handle, 
			self.raw_image_buffer
		)
		frameDesc = ret[1]

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to capture frame.")

		ret = PxLApi.formatImage(
			self.raw_image_buffer, 
			frameDesc, 
			PxLApi.ImageFormat.RAW_RGB24
		)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to convert frame to 24 bit RGB.")

		formatedImage = ret[1]

		np_img = np.full_like(
			formatedImage, 
			formatedImage, 
			order="C"
		)
		np_img.dtype = np.uint8

		imageHeight = int(frameDesc.Roi.fHeight)
		imageWidth  = int(frameDesc.Roi.fWidth)
		newShape    = (imageHeight, imageWidth, 3)

		np_img = np.reshape(np_img, newShape)

		if convert:
			np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

		return np_img


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	camera = CameraController()
	code.interact(local=locals())




			
			


