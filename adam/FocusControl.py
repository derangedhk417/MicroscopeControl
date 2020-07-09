import serial
import sys
import code
import time

# This handles focusing and zooming of the microscope.
# TODO: Add a mode to this class that has non-blocking queued commands and
# callback functions for handling errors.
class FocusController:
	# The timeout argument controls how long the serial connection
	# will wait for data when reading. If this value is too high, function
	# calls will take a long time to return because the class will spend so
	# much time waiting for more data on the serial port.
	def __init__(self, port="COM4", baud=38400, timeout=0.01):
		self.connection = serial.Serial(port, baud, timeout=timeout)
		self.position   = None

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

		# Motor 1 is zoom.
		# Motor 2 is focus.

		# Read the motor limits from the controller so that this class
		# can limit itself. This should prevent the motors from attempting
		# to go past their limits.
		z, f = self._read_limits()

		self.zoom_range  = (0, z)
		self.focus_range = (0, f)

		self._wait_for_idle('zoom')
		self._wait_for_idle('focus')

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

	# Read data from the serial port until a complete response has been
	# sent. The response is terminated with "$ ", so this function returns
	# when it reads that. It does NOT retur the final "$ ".
	def readResponse(self):
		response_string = ""
		char            = self.connection.read(1).decode("ascii")
		while char != '$':
			response_string += char
			char             = self.connection.read(1).decode("ascii")

		char = self.connection.read(1).decode("ascii")

		if char != ' ':
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
	def incrementZoom(self, value):
		if value < -1.0 or value > 1.0:
			raise Exception("Values must be in the range [-1.0, 1.0]")

		value  = value * (self.zoom_range[1] - self.zoom_range[0])
		value += self.zoom_range[0]
		value  = int(round(value))

		command_str = b"write increment_1 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		self._wait_for_idle('zoom')
		return value

	# Change the current focus by the specified value.
	def incrementFocus(self, value):
		if value < -1.0 or value > 1.0:
			raise Exception("Values must be in the range [-1.0, 1.0]")

		value  = value * (self.focus_range[1] - self.focus_range[0])
		value += self.focus_range[0]
		value  = int(round(value))

		command_str = b"write increment_2 %d\n"%value
		self.connection.write(command_str)

		response = self.readResponse()
		value    = self._parse_response(response, command_str)

		self._wait_for_idle('focus')
		return value

	# Set the position of the zoom motor. By default, this will home the
	# zoom motor to position zero before moving to the requested position. 
	# This results in better repeatability.
	def setZoom(self, value, corrected=True):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_zoom(0.0)
			self._wait_for_idle('zoom')
			self._set_zoom(value)
			
		else:
			self._set_zoom(value)

		self._wait_for_idle('zoom')

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
	def setFocus(self, value, corrected=True):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_focus(0.0)
			self._wait_for_idle('focus')
			self._set_focus(value)
		else:
			self._set_focus(value)

		self._wait_for_idle('focus')

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

		if responses[0].strip() != command.decode('ascii').strip():
			raise IOError("Controller received corrupted command. (%s != %s)"%(
				command.decode("ascii").strip(), 
				responses[0].strip()
			))

		return responses



if __name__ == '__main__':
	port       = sys.argv[1]
	baud       = int(sys.argv[2])
	controller = FocusController(port, baud)
	#serial = serial.Serial(port, baud, timeout=0.5)
	code.interact(local=locals())