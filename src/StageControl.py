import serial
import sys
import code


# This handles the motion of the motorized stage as well as reading
# data from the stage (in order to determine stage position).
class StageController:
	def __init__(self, port="COM3", baud=9600, timeout=0.01):
		self.connection = serial.Serial(port, baud, timeout=timeout)
		self.position   = None

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

		# This will prevent the system from hitting the supports that hold
		# the optics.
		controller.setLimits([-50, 50], [-50, 37])

	def readResponse(self):
		response_string = ""
		char            = self.connection.read(1).decode("ascii")
		while char != '\r':
			response_string += char
			char             = self.connection.read(1).decode("ascii")

		char = self.connection.read(1).decode("ascii")

		if char != '\n':
			raise IOError("Incomplete response from controller.")

		return response_string

	# Returns True if the axis is moving and false otherwise.
	def getAxisStatus(self, axis):
		if axis == 'x':
			command_string = b"RB X \r"
		elif axis == 'y':
			command_string = b"RB Y \r"
		else:
			raise Exception("Unrecognized axis specified.")

		self.connection.write(command_string)

		response = self.readResponse()
		status   = ord(response[1])

		moving = (status & 0b00000001)

		return moving == 1

	# Move the stage position to the given value in millimeters.
	def moveTo(self, x, y):
		x = int(round(x * 10000))
		y = int(round(y * 10000))

		command_string = b"MOVE X=%d Y=%d \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	def moveDelta(self, x, y):
		command_string = b"R X=%d Y=%d Z \r"%(x * 10000, y * 10000)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Set the amount of time in milliseconds that it should take for the
	# axis to go from the start velocity to the maximum velocity.
	def setAcceleration(self, x, y):
		if not isinstance(x, int) or not isinstance(y, int):
			raise Exception("This function takes integers (milliseconds) as arguments.")

		command_string = b"AC X=%d Y=%d \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Sets the sensitivity of the hardware joystick attached to the controller.
	# The coarse argument applies when fine control has been toggled off with the
	# button in the center of the joystick. The fine argument applies when it has
	# been toggled on. Arguments are in percent of the normal sensitivity.
	def setJoystickSensitivity(self, coarse, fine):
		if coarse < 0.1 or coarse > 100 or fine < 0.1 or fine > 100:
			raise Exception("Sensitivity values must be in the range [0.1, 100]")


		command_string = b"JS X=%2.2f Y=%2.2f \r"%(coarse, fine)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Sets the maximum speed of the stage on each axis.
	def setMaxSpeed(self, x, y):
		if x > 7.5 or x < 0.0 or y > 7.5 or y < 0.0:
			raise Exception("Values must be in the range [0, 7.5] mm/s")

		command_string = b"S X=%1.6f Y=%1.6f \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Given a range of valid values on the x and y axes,
	# this function will set limits on the motion of the stage.
	# These limits will be enforced by the controller.
	def setLimits(self, x, y):
		x = [int(round(x[0])), int(round(x[1]))]
		y = [int(round(y[0])), int(round(y[1]))]

		command_string = b"SL X=%d Y=%d \r"%(x[0], y[0])

		self.connection.write(command_string)
		response      = self.readResponse()
		response_data = self._parse_response(response)

		command_string = b"SU X=%d Y=%d \r"%(x[1], y[1])

		self.connection.write(command_string)
		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Resets the position of the stage.
	def home(self):
		self.connection.write(b'MOVE X=0 Y=0 \r')
		response      = self.readResponse()
		response_data = self._parse_response(response)


	# Returns the X and Y positions in millimeters.
	def getPosition(self):
		command_string = b"WHERE X Y Z \r"

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

		X_str, Y_str = response_data.split(" ")

		return float(X_str) / 10000, float(Y_str) / 10000

	def _parse_response(self, response):
		if response[0] != ':':
			raise IOError("Malformed response from controller. (%s)"%(response))

		if response[1] == 'A':
			response_data = ""
			idx           = 2
			while idx < len(response):
				response_data += response[idx]
				idx           += 1

			return response_data.strip()
		elif response[1] == 'N':
			valid_numbers       = '0123456789'
			response_error_code = ""
			if response[3] in valid_numbers:
				response_error_code += response[3]
				if response[4] in valid_numbers:
					response_error_code += response[4]
			else:
				raise IOError("Controller responded with error but did not specify error code. (%s)"%(response))

			response_error_code = int(response_error_code)
			raise Exception("Controller responded with error code %d (%s)"%(response_error_code, response))
		else:
			raise IOError("Malformed response from controller. (%s)"%response)



if __name__ == '__main__':
	port       = sys.argv[1]
	baud       = int(sys.argv[2])
	controller = StageController(port, baud)

	code.interact(local=locals())