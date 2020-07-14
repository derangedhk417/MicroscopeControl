# Author:      Adam Robinson
# Description: This class handles zooming and focusing of the microscope
#              via the connected servo-motors. This file also contains a
#              helper function that attempts to detect and automatically 
#              connect to the motor controller.

import serial
import sys
import time
import threading
import code

# This handles focusing and zooming of the microscope.
# All calls that perform a long running operation have a keyword
# argument called "callback". When specified, the function will 
# return automatically, and will call the specified callback function
# when the operation is complete. For functions that normally return 
# a value, this callback function should take that value as its first
# argument.
class FocusController:
	# The timeout argument controls how long the serial connection
	# will wait for data when reading. If this value is too high, function
	# calls will take a long time to return because the class will spend so
	# much time waiting for more data on the serial port.
	#
	# retries: The number of times to retry an operation if it fails. 
	#          Unfortunately, the controller seems to fail randomly,
	#          which makes this kind of thing necessary.
	def __init__(self, port="COM4", baud=38400, timeout=0, retries=4):
		self.connection = serial.Serial(port, baud, timeout=timeout)
		self.retries    = retries

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

		# If an operation appears to cause an invalid response from the
		# controller, but does not result in invalid behavior, this can
		# be set in order to prevent exceptions from being generated. Use
		# at your own risk.
		self._ignore_response = False

		# Motor 1 is zoom.
		# Motor 2 is focus.
		# Read the motor limits from the controller so that this class
		# can limit itself. This should prevent the motors from attempting
		# to go past their limits.
		try:
			z, f = self._read_limits()

			self.zoom_range  = (0, z)
			self.focus_range = (0, f)

			self._wait_for_idle('zoom')
			self._wait_for_idle('focus')
		except Exception as ex:
			self.cleanup()
			raise Exception("Error reading motor limits.") from ex

		

	def __del__(self):
		self.cleanup()

	def cleanup(self):
		try:
			if self.connection.is_open:
				self.connection.close()
		except:
			pass

	# "motor" should be either "zoom" or "focus". This function
	# will return true if the motor is idle and false otherwise.
	def _is_idle(self, motor):
		if motor == 'zoom':
			command_str = b"read status_1\n"
		elif motor == 'focus':
			command_str = b"read status_2\n"
		else:
			raise Exception("Unrecognized value for \'motor\' paramter. (%s)"%motor)

		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		# Despite what the documentation says, the idle state only
		# seems to actually be reached when every bit except for bit
		# 15 is unset. 
		response_code = int(value[1])
		idle_bit      = (response_code & 0b100000000000000) >> 14
		other_bits    = (response_code & 0b011111111111111)

		return (idle_bit == 1) and (other_bits == 0)


	# This function will block until the specified motor is in the idle state.
	# It will user the serial connection to check the motor status every
	# "interval" seconds and will throw an exception after "timeout" seconds,
	# if the motor has not reached the idle state.
	def _wait_for_idle(self, motor, interval=0.05, timeout=30):
		start = time.time_ns()

		while True:
			if not self._is_idle(motor):
				time.sleep(interval)
			else:
				break

			if (time.time_ns() - start) / 1e9 >= timeout:
				msg = """Timeout of %f seconds exceeded while waiting for motor 
				to reach idle state."""
				raise Exception(msg%timeout)

	# Read the limit values for both of the motors.
	def _read_limits(self):
		command_str = b"read setup_limit_1\n"
		self.connection.write(command_str)

		response   = self.readResponse()
		value      = self._parse_response(response, command_str)
		zoom_limit = int(value[1])

		command_str = b"read setup_limit_2\n"
		self.connection.write(command_str)

		response   = self.readResponse()
		value      = self._parse_response(response, command_str)
		focus_limit = int(value[1])

		return zoom_limit, focus_limit

	def _set_ignore_response(self, value):
		self._ignore_response = value

	# Returns zoom_limit, focus_limit
	def getLimits(self):
		return self._read_limits()

	# Read data from the serial port until a complete response has been
	# sent. The response is terminated with "$ ", so this function returns
	# when it reads that. It does NOT return the final "$ ".
	def readResponse(self, timeout=2):
		start           = time.time_ns()
		response_string = ""
		char            = self.connection.read(1).decode("ascii")
		while char != '$':
			response_string += char
			char             = self.connection.read(1).decode("ascii")
			if (time.time_ns() - start) / 1e9 > timeout:
				raise IOError("Controller took too long to respond.")

		char = self.connection.read(1).decode("ascii")

		while char == '':
			char = self.connection.read(1).decode("ascii")
			if (time.time_ns() - start) / 1e9 > timeout:
				raise IOError("Controller took too long to respond.")

		if char != ' ' and not self._ignore_response:
			raise IOError("Incomplete response from controller. (%s)")

		return response_string

	# NOTE: I don't know what units the velocity and acceleration values are in.
	# I suspect that they are in motor steps per second and motor steps per second
	# squared.

	# Retrieve the acceleration of the zoom motor. 
	def getZoomAcceleration(self):
		command_str = b"read setup_accel_1\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the initial velocity of the zoom motor.
	def getZoomInitialVelocity(self):
		command_str = b"read setup_initv_1\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the maximum velocity of the zoom motor.
	def getZoomMaxVelocity(self):
		command_str = b"read setup_maxv_1\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the acceleration of the zoom motor.
	def setZoomAcceleration(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_accel_1 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the initial velocity of the zoom motor.
	def setZoomInitialVelocity(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_initv_1 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the maximum velocity of the zoom motor.
	def setZoomMaxVelocity(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_maxv_1 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the acceleration of the focus motor.
	def getFocusAcceleration(self):
		command_str = b"read setup_accel_2\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the initial velocity of the focus motor.
	def getFocusInitialVelocity(self):
		command_str = b"read setup_initv_2\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the maximum velocity of the focus motor.
	def getFocusMaxVelocity(self):
		command_str = b"read setup_maxv_2\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the acceleration of the focus motor.
	def setFocusAcceleration(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_accel_2 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the initial velocity of the focus motor.
	def setFocusInitialVelocity(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_initv_2 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Set the maximum velocity of the focus motor.
	def setFocusMaxVelocity(self, value):
		if not isinstance(value, int):
			raise Exception("This function requires an integer argument.")

		command_str = b"write setup_maxv_2 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		return int(value[1])

	# Retrieve the current position of the zoom motor.
	def getZoom(self):
		command_str = b"read current_1\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)
		zoom     = int(value[1])

		return (zoom - self.zoom_range[0]) / (self.zoom_range[1] - self.zoom_range[0])

	# Retrieve the current position of the focus motor.
	def getFocus(self):
		command_str = b"read current_2\n"
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)
		focus    = int(value[1])

		return (focus - self.focus_range[0]) / (self.focus_range[1] - self.focus_range[0])

	# Change the current zoom by the specified value.
	def incrementZoom(self, value, callback=None):
		if value < -1.0 or value > 1.0:
			raise Exception("Values must be in the range [-1.0, 1.0]")

		# Read the current position and modify the increment to ensure that
		# we don't overrun the motor limit.

		current_zoom = self.getZoom()

		if current_zoom + value > 1.0:
			value = 1.0 - current_zoom
		elif current_zoom + value < 0.0:
			value = -current_zoom

		value  = value * (self.zoom_range[1] - self.zoom_range[0])
		value += self.zoom_range[0]
		value  = int(round(value))

		command_str = b"write increment_1 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)


		if callback is None:
			self._wait_for_idle('zoom')
			return int(value[1])
		else:
			def complete_operation():
				self._wait_for_idle('zoom')
				callback(int(value[1]))
			wait_thread = threading.Thread(target=complete_operation)
			wait_thread.start()
			return None
		
		

	# Change the current focus by the specified value.
	def incrementFocus(self, value, callback=None):
		if value < -1.0 or value > 1.0:
			raise Exception("Values must be in the range [-1.0, 1.0]")


		# Read the current position and modify the increment to ensure that
		# we don't overrun the motor limit.

		current_focus = self.getFocus()

		if current_focus + value > 1.0:
			value = 1.0 - current_focus
		elif current_focus + value < 0.0:
			value = -current_focus

		value  = value * (self.focus_range[1] - self.focus_range[0])
		value += self.focus_range[0]
		value  = int(round(value))

		command_str = b"write increment_2 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		if callback is None:
			self._wait_for_idle('focus')
			return int(value[1])
		else:
			def complete_operation():
				self._wait_for_idle('focus')
				callback(int(value[1]))
			wait_thread = threading.Thread(target=complete_operation)
			wait_thread.start()
			return None

	# Set the position of the zoom motor. By default, this will home the
	# zoom motor to position zero before moving to the requested position. 
	# This results in better repeatability.
	def setZoom(self, value, corrected=True, callback=None):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_zoom(0.0)
			if callback is None:
				self._wait_for_idle('zoom')
				self._set_zoom(value)
				self._wait_for_idle('zoom')
			else:
				def complete_operation():
					self._wait_for_idle('zoom')
					self._set_zoom(value)
					self._wait_for_idle('zoom')
					callback()
				wait_thread = threading.Thread(target=complete_operation)
				wait_thread.start()
		else:
			self._set_zoom(value)
			if callback is None:
				self._wait_for_idle('zoom')
			else:
				def complete_operation():
					self._wait_for_idle('zoom')
					callback()
				wait_thread = threading.Thread(target=complete_operation)
				wait_thread.start()

	def _set_zoom(self, value):
		value  = value * (self.zoom_range[1] - self.zoom_range[0])
		value += self.zoom_range[0]
		value  = int(round(value))

		command_str = b'write target_1 %d\n'%value

		self.connection.write(command_str)

		response = self.readResponse()

		value = self._parse_response(response, command_str)

		return value

	# Set the position of the focus motor. By default, this will home the
	# focus motor to position zero before moving to the requested position. 
	# This results in better repeatability.
	def setFocus(self, value, corrected=True, callback=None):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_focus(0.0)
			if callback is None:
				self._wait_for_idle('focus')
				self._set_focus(value)
				self._wait_for_idle('focus')
			else:
				def complete_operation():
					self._wait_for_idle('focus')
					self._set_focus(value)
					self._wait_for_idle('focus')
					callback()
				wait_thread = threading.Thread(target=complete_operation)
				wait_thread.start()
		else:
			self._set_focus(value)
			if callback is None:
				self._wait_for_idle('focus')
			else:
				def complete_operation():
					self._wait_for_idle('focus')
					callback()
				wait_thread = threading.Thread(target=complete_operation)
				wait_thread.start()

	def _set_focus(self, value):
		value  = value * (self.focus_range[1] - self.focus_range[0]) 
		value += self.focus_range[0]
		value  = int(round(value))

		command_str = b'write target_2 %d\n'%value

		self.connection.write(command_str)

		response = self.readResponse()

		value = self._parse_response(response, command_str)

		return value

	# This ensures that the controller responded with a correct copy of
	# the command that we issued to it and then returns all parts of the
	# response split up by "\r\n"
	def _parse_response(self, response, command):
		responses = response.split("\r\n")

		if not self._ignore_response:
			if responses[0].strip() != command.decode('ascii').strip():
				raise IOError("Controller received corrupted command. (%s != %s)"%(
					command.decode("ascii").strip(), 
					responses[0].strip()
				))

		return responses

# This class will attempt to connect to the motor controller, assuming
# that it is plugged in. It does this by querying every serial controller
# it can find and attempting to issue an innocuous command to it. If the
# controller responds appropriately, it is assumed to be the motor controller.
# This function will return a connected and ready to go FocusController object
# if it succeeds and None otherwise.
def autoConnect():
	# Build a list of ports to try.
	ports = ["COM4", "COM1", "COM2", "COM3", "COM0"]

	for i in range(5, 10):
		ports.append("COM%d"%i)

	for p in ports:
		try:
			# This class always reads it's limits from the controller upon
			# initialization, so it will throw an exception if we have 
			# connected to something that isn't the controller.
			controller = FocusController(p, 115200)
			return controller
		except:
			continue

	return None

if __name__ == '__main__':
	port       = sys.argv[1]
	baud       = int(sys.argv[2])
	controller = FocusController(port, baud)
	code.interact(local=locals())