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

import code

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
			print("Connecting to focus controller . . . ", end='')
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
			print("Connecting to stage controller . . . ", end='')
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

		if self.verbose:
			print("Connecting to camera           . . . ", end='')
		self.camera = CameraController()
		if self.verbose:
			print("DONE")

if __name__ == '__main__':
	microscope = MicroscopeController(verbose=True)

	code.interact(local=locals())
