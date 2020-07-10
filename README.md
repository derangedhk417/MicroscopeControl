# MicroscopeControl

This repository contains the code necessary to control all of the components of the automated microscope being built at Dr. Patrick Vora's lab at George Mason University. This includes control of the motorized stage, zoom and focus motor and the camera attached to the optics. 

## Dependencies

In order to run this code, you need openCV, NumPy, SciPy, Matplotlib, pixelinkWrapper, kivy and pyserial installed. You should be able to accomplish this with the following.

You will need to run the installer in `dependencies/` in order to install the build tools for windows, before you run the user interface programs in this repo and before you install kivy (second command below). Everything should run on Linux without issue.

```
pip install numpy scipy opencv-python matplotlib pixelinkWrapper pyserial --user
python -m pip install https://github.com/kivy/kivy/archive/master.zip
```

In addition to the Python libraries required, you will also need to install the Pixelink capture software. This comes with the drivers and .dll files necessary for the Python wrapper to interface with the camera. The software can be found [here](https://pixelink.com/products/software/pixelink-capture-software/pixelink-capture-software-download/). The repository that contains sample code necessary to use the Python wrapper can be found [here](https://github.com/pixelink-support/pixelinkPythonWrapper). Detailed documentation of the protocols necessary to communicate with the stage and the focus and zoom motors can be found in the `documentation/` folder of this repository. An introduction to the Pixelink API is available [here](https://support.pixelink.com/support/solutions/articles/3000044964-basic-principles).
