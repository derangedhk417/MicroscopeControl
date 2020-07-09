import serial
import sys
import code
import time

# This handles focusing and zooming of the microscope.
class FocusController:
	def __init__(self, port="COM4", baud=38400, timeout=0.01):
		self.connection = serial.Serial(port, baud, timeout=timeout)
		self.position   = None

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

		# Motor 1 is zoom.
		# Motor 2 is focus.

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
	def _wait_for_idle(self, motor, interval=0.05, timeout=8):
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

	

	def setZoom(self, value, corrected=True):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_zoom(0.0)
			self._wait_for_idle('zoom')
			return self._set_zoom(value)
		else:
			return self._set_zoom(value)		

	def _set_zoom(self, value):
		value  = value * (self.zoom_range[1] - self.zoom_range[0])
		value += self.zoom_range[0]
		value  = int(round(value))

		command_str = b'write target_1 %d\n'%value

		self.connection.write(command_str)

		response = self.readResponse()

		value = self._parse_response(response, command_str)

		return value

	def setFocus(self, value, corrected=True):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_focus(0.0)
			self._wait_for_idle('focus')
			return self._set_focus(value)
		else:
			self._set_focus(value)

	def _set_focus(self, value):
		value  = value * (self.focus_range[1] - self.focus_range[0]) 
		value += self.focus_range[0]
		value  = int(round(value))

		command_str = b'write target_2 %d\n'%value

		self.connection.write(command_str)

		response = self.readResponse()

		value = self._parse_response(response, command_str)

		return value


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