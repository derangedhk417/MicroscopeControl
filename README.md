# MicroscopeControl

This repository contains the code necessary to control all of the components of the automated microscope being built at Dr. Patrick Vora's lab at George Mason University. This includes control of the motorized stage, zoom and focus motor and the camera attached to the optics. 

**See the [Wiki](https://github.com/derangedhk417/MicroscopeControl/wiki) for details on each of the code files in this repo**

## Dependencies

In order to run this code, you need openCV, NumPy, SciPy, Matplotlib, pixelinkWrapper, kivy and pyserial installed. You should be able to accomplish this with the following.

You will need to run the installer in `dependencies/` in order to install the build tools for windows, before you run the user interface programs in this repo and before you install kivy (second command below). Everything should run on Linux without issue.

```
pip install numpy scipy opencv-python matplotlib pixelinkWrapper pyserial --user
python -m pip install https://github.com/kivy/kivy/archive/master.zip
```
**Note:**
Sometimes kivy doesn't automatically install all of its dependencies correctly. If you run into an error when running one of the GUI applications in this repo, run the following.

```
python -m pip install docutils pygments pypiwin32 kivy_deps.sdl2==0.1.* kivy_deps.glew==0.1.*
python -m pip install kivy_deps.gstreamer==0.1.*
python -m pip install kivy_deps.angle==0.1.*
python -m pip install kivy==1.11.1
```

In addition to the Python libraries required, you will also need to install the Pixelink capture software. This comes with the drivers and .dll files necessary for the Python wrapper to interface with the camera. The software can be found [here](https://pixelink.com/products/software/pixelink-capture-software/pixelink-capture-software-download/). The repository that contains sample code necessary to use the Python wrapper can be found [here](https://github.com/pixelink-support/pixelinkPythonWrapper). Detailed documentation of the protocols necessary to communicate with the stage and the focus and zoom motors can be found in the `documentation/` folder of this repository. An introduction to the Pixelink API is available [here](https://support.pixelink.com/support/solutions/articles/3000044964-basic-principles).

## Example Usage

Make sure that the `src/` folder is either in your python library search path or that every file in the `src/` folder is in your current working directory. The following is an example of how to use this code. See the [Wiki](https://github.com/derangedhk417/MicroscopeControl/wiki) for a full listing of available functionality.

```Python
from MicroscopeControl import MicroscopeController

# Setting verbose=True will cause connection information to print
# to the console while the controller is attempting to connect.
microscope = MicroscopeController(verbose=True) 

# After this call, the x and y axes of the stage will take 1500ms to
# Accelerate to their maximum speed.
microscope.stage.setAcceleration(1500, 1500)

# Move the stage 10mm in the positive direction on both
# the x and y axes.
microscope.stage.moveTo(10, 10)

# Move the stage back to the home position (0, 0)
microscope.stage.home()

# Set the focus motor position to half of its maximum value.
microscope.stage.setFocus(0.5)

# Set the zoom motor position to its maximum value.
microscope.stage.setZoom(1.0)

# Put the camera in capture mode.
microscope.camera.startCapture()

# Turn on auto-exposure adjustment for the camera.
microscope.camera.enableAutoExposure()

# Take an image as an RGB NumPy ndarray.
# The default (convert=False) will return a BGR array.
img = microscope.camera.getFrame(convert=True)

# Turn off auto exposure.
microscope.camera.disableAutoExposure()

# Turn off capture mode.
microscope.camera.endCapture()
```

The `CameraController` class will automatically ensure that everything disconnects properly when it is freed. You can do this manually with

```Python
microscope = MicroscopeController(verbose=True)
del microscope
```

If you are on a Unix system, you'll probably need to manually specify the port and baudrate for each device.

```Python
from MicroscopeControl import MicroscopeController

microscope = MicroscopeController(
    focus_port='/dev/tty0',
    focus_baudrate=38400,
    stage_port='/dev/tty1',
    stage_baudrate=9600
)
```

The camera will be recognized automatically. Obviously the ports they show up on in your computer will vary. So far, a baud rate of 9600 appears to be the correct value for the stage controller. Physical switches on the back control this value. See the documentation pdf file for details. A baud rate of 38400 appears to be the correct value for the focus/zoom motor controller. I didn't find this anywhere in the documentation, I discovered it through testing.

**Note:**

On Windows, you can use the `mode` command at the command prompt to list all connected serial devices.
