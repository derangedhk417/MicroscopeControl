import serial
import sys
import code

class StageController:
    def __init__(self, port=COM3, baud=9600, timeout=0.01):
        self.connection = serial.Serial(port, baud, timeout=1)
        self.position   = None
    
    if not self.connection.is_open:
        raise Exception("Failed to open serial connection.")
        
    def readResponse(self):
        response_string = ""
        char = self.connection.read(1).decode("ascii")
        while char != '\r':
            response_string += char
            char = self.connection.read(1).decode("ascii")
            
        char = self.connection.read(1).decode("ascii")
        
        if char != '\n':
            raise IOError("Incomplete response from controller.")
            
        return response_string
        
    # Move the stage position to the given value in milimeters
    def moveTo(self, x, y):
        x = int(round(x * 10000))
        y = int(round(x * 10000))
        
        command_string = b"MOVE X=%d Y=%d \r"%(X,Y)
        
        self.connection.write(command_string)
        
        response = self.connection.read(100).decode("ascii")
        response_data = self._parse_response(response)
    
    # Move the stage to the center
    def home(self):
        self.connection.write(b'MOVE X=0 Y=0 \r')
        response = self.connection.read(100).decode("ascii")
        response_data = self._parse_response(response)
        
    def getPosition(self):
        command_string = b"WHERE X Y Z \r"
        
        self.connection.write(command_string)
        
        response = self.connection.read(100).decode("ascii")
        response_data = self._parse_response(response)
        
        X_str, Y_str = response_data.split(" ")
        
        return float(X_str) / 10000, float(Y_str) / 10000
        
    def _parse_response(self, response):
        if response[0] != ':':
            raise IOError("Malformed response from controller. (%s)"%(response))
        
        if response[1] == 'A':
            response_data = ""
            idx           = 2
            while response[idx] != '\r' and idx < len(response):
                response_data   += response[idx]
                idx             += 1
                
            if response[-2:] != '\r\n':
                raise IOError("Incomplete response from controller. (%s)"%response)
                
            return response_data.strip()
        elif response[1] == 'N':
            valid_numbers = '0123456789'
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
            raise IOError("Malformed response from controller. (%s)"%(response))

if __name__ == '__main__':
    port = sys.argv[1]
    baud = int(sys.argv[2])
    connection = serial.Serial(port, baud, timeout=1)
    
    if not connection.is_open:
        print("Connection failed...")
        exit()
        
    code.interact(local=locals())
    
    