import serial
import sys
import code

# This handles focusing and zooming of the microscope.
class FocusController:
	def __init__(self, port="COM3", baud=9600, timeout=0.01):
		self.connection = serial.Serial(port, baud, timeout=timeout)
		self.position   = None

		# Motor 1 is zoom.
		# Motor 2 is focus.

		z, f = self._read_limits()

		self.zoom_range  = (0, z)
		self.focus_range = (0, f)

		print("Limits:")
		print("\tZoom:  %d"%self.zoom_range[1])
		print("\tFocus: %d"%self.focus_range[1]) 

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

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
			raise IOError("Incomplete response from controller.")

		return response_string

	

	def setZoom(self, value, corrected=False):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_zoom(0.0)
			return self._set_zoom(value)
		else:
			return self._set_zoom(value)		

	def _set_zoom(self, value):
		value = value * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
		value = int(round(value))

		command_str = b'write target_1 %d\n'%value

		self.connection.write(command_str)

		response = self.readResponse()

		value = self._parse_response(response, command_str)

		return value

	def setFocus(self, value, corrected=False):
		if value < 0.0 or value > 1.0:
			raise Exception("Values must be in the range [0.0, 1.0]")

		if corrected:
			self._set_focus(0.0)
			return self._set_focus(value)
		else:
			self._set_focus(value)

	def _set_focus(self, value):
		value = value * (self.focus_range[1] - self.focus_range[0]) + self.focus_range[0]
		value = int(round(value))

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