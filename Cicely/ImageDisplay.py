# Author:      Adam Robinson
# Description: This file contains a class that displays an image while
#              handling resizing, events, zooming, panning and drawing
#              geometry on the image.

from kivy.uix.label        import Label
from kivy.uix.image        import Image
from kivy.uix.gridlayout   import GridLayout
from kivy.uix.behaviors    import ButtonBehavior
from kivy.core.image       import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.graphics         import Rectangle, Color, Line, InstructionGroup
from kivy.uix.scrollview   import ScrollView
from kivy.uix.button       import Button
from kivy.uix.floatlayout  import FloatLayout
from kivy.core.window      import Window

from kivy.graphics.stencil_instructions import StencilPush, StencilUse
from kivy.graphics.stencil_instructions import StencilPop, StencilUnUse
from kivy.utils                         import get_color_from_hex as hex_color

from CustomBoxLayout   import CustomBoxLayout

import numpy as np
import code

# Relative Coordinates: Coordinates in the range [0.0, 1.0], that correspond
# to positions in the image being displayed.
# Example: With an image that has dimensions (height: 300, width: 600), 
# relative coordinates (0.5, 0.5) would correspond to x_pixel: 150,
# y_pixel: 300. 

# Handles the acttual sizing and drawing of the image rectangle, as well as 
# drawing the zoom rectangle and cropping the image when zooming. The click
# and drag events that take place to zoom to a rectangle are handled by this
# class. The parent object (ImageDisplay), handles the setting and unsetting
# of the is_zooming flag, which this class will read when determining how to 
# interpret user inputs.
class ImageManager(ButtonBehavior, CustomBoxLayout):
	def __init__(self, *args, **kwargs):
		# Read the padding argument from the user and adhere to it.
		# It will interfere with normal operation of the FloatLayout constructor,
		# so we remove the kay from kwargs before calling it.
		if 'padding' in kwargs:
			self.padding = kwargs['padding']
			del kwargs['padding']
		else:
			self.padding = [0, 0, 0, 0]

		super(ImageManager, self).__init__(*args, **kwargs)

		# Make sure the background is white. When you don't, images draw
		# with a tint (no idea why).
		with self.canvas:
			Color(1, 1, 1, 1)
			self.image_rect = Rectangle(
				pos=self.pos, 
				size=self.size
			)

		# Flag for whether or not an image is currently loaded. 
		# Zooming events are invalid when this is false.
		self.is_loaded = False

		# These events will be used to ensure that the image draws in bounds
		# and has the correct aspect ratio.
		self.bind(size=self._update_dims, pos=self._update_dims)

		# These events are necessary to track the clicking and dragging action
		# that a user takes to zoom into a particular rectangle.
		self.bind(
			on_touch_down=self._down,
			on_touch_up=self._up,
			on_touch_move=self._move
		)
		Window.bind(mouse_pos=self._mouse_pos)

		# These keep track of where the user started clicking and dragging
		# to zoom, the width and height of the zoom rectangle and the graphics
		# object that draws a dashed line to indicate where the user is zooming.
		self.click_down            = False
		self.drag_start            = (0, 0)
		self.drag_w                = 0
		self.drag_h                = 0
		self.current_zoom_rect_obj = None

		# These are, respectively, the current subsection of the image
		# data array that we are zoomed to and the last coordinates that
		# were zoomed to, within the array. These are stored as actual
		# indices, not float32 coordinates.
		self.current_zoom_subarray = None

		# These are the position and dimensions of the rectangle
		# that displays the image. They are updated every time the
		# pos and size events fire.
		self.display_x      = None
		self.display_y      = None
		self.display_width  = None
		self.display_height = None

		# These store the colors and geometry for all of the currently
		# drawn lines. The instruction_groups list breaks each line
		# up into an instruction group so that it can be added and 
		# removed as a single object.
		self.lines              = []
		self.colors             = []
		self.instruction_groups = []

		# This is used to ensure that contours do not draw outside
		# of the image when the user zooms in.
		self.stencil_group = None

		self.last_zoom = None


	# Used to determine if a coordinate position lies within the specified
	# rectangle.
	def posInRect(self, pos, rpos, rsize):
		if pos[0] > rpos[0] and pos[0] < rpos[0] + rsize[0]:
			if pos[1] > rpos[1] and pos[1] < rpos[1] + rsize[1]:
				return True
		return False

	# Fired when the mouse moves. This is used to change the cursor to a 
	# crosshair when the parent object has is_zooming == True
	def _mouse_pos(self, inst, val):

		in_rect = self.posInRect(
			Window.mouse_pos,
			(self.display_x, self.display_y),
			(self.display_width, self.display_height)
		)

		if in_rect:
			if self.parent.is_zooming:
				Window.set_system_cursor('crosshair')
		else:
			Window.set_system_cursor('arrow')


	# Initiates a zoom action and stores the coordinates where the zoom began.
	def _down(self, inst, val):
		in_rect = self.posInRect(
			Window.mouse_pos,
			(self.display_x, self.display_y),
			(self.display_width, self.display_height)
		)

		if in_rect and self.is_loaded:
			self.click_down = True
			self.drag_start = Window.mouse_pos


	# If appropriate, executes a zoom and removes the dashed line from the
	# canvas so that it will no longer draw.
	def _up(self, inst, val):
		self.click_down = False

		if self.current_zoom_rect_obj is not None:
			self.canvas.remove(self.current_zoom_rect_obj)

		in_rect = self.posInRect(
			Window.mouse_pos,
			(self.display_x, self.display_y),
			(self.display_width, self.display_height)
		)

		# See if we need to execute a zoom.
		if in_rect and self.parent.is_zooming:
			# See if they selected a reasonably large area.
			if np.abs(self.drag_w * self.drag_h) > 16:
				# The selected at least a 4x4 region, consider this a zoom in.
				# Calculate the coordinates in the image, of the zoom.
				zoom_x0 = self.drag_start[0] - self.display_x
				zoom_y0 = self.drag_start[1] - self.display_y

				zoom_x1 = val.pos[0] - self.display_x
				zoom_y1 = val.pos[1] - self.display_y

				# If the user dragged backwards on any axis, the 
				# coordinates may need to be exhanged.
				if zoom_x0 > zoom_x1:
					t = zoom_x1
					zoom_x1 = zoom_x0
					zoom_x0 = t

				if zoom_y0 > zoom_y1:
					t = zoom_y1
					zoom_y1 = zoom_y0
					zoom_y0 = t

				self.zoomTo(zoom_x0, zoom_x1, zoom_y0, zoom_y1)
				self.drag_start            = (0, 0)
				self.drag_w                = 0
				self.drag_h                = 0
		# If a zoom operation hasn't overriden the functionality, see if we need
		# to place a contour point.
		elif in_rect and False:# and self.parent.is_editing_contour and False:
			# Calculate where the contour is in image coordinates (cartesian)
			x, y = self.screenCoordinatesToRelativeCoordinates(val.pos[0], val.pos[1])

			# Add this to the parent objects current contour (it will then)
			# in turn, call this classes popLine method on the current contour
			# and then this classes pushLine method to add the newly updates contour.
			self.parent.addPointToContour(x, y)

	# Converts coordinates on the screen (relative to the bottom left corner
	# of the window) to relative coordinates in the image. See: Relative Coordinates
	# at the top of this file.
	def screenCoordinatesToRelativeCoordinates(self, x, y):
		if self.last_zoom is not None:
			x0_idx, x1_idx, y0_idx, y1_idx = self.last_zoom
			xr  = ((x - self.display_x) / self.display_width) * self.img_size[0]
			xr += x0_idx
			xr /= self.img.shape[1]

			y0_idx = self.img.shape[0] - y0_idx
			yr  = ((y - self.display_y) / self.display_height) * self.img_size[1]
			yr += y0_idx
			yr /= self.img.shape[0]
		else:
			xr = (x - self.display_x) / self.display_width
			yr = (y - self.display_y) / self.display_height

		return xr, yr

	# Given a set of coordinates from 0.0 to 1.0, scaled to the width
	# and height of the currently loaded image (relative coordinates), this 
	# function will return where on the screen those coordinates lie.
	def relativeCoordinatesToScreenCoordinates(self, x, y):
		if self.last_zoom is not None:
			x0_idx, x1_idx, y0_idx, y1_idx = self.last_zoom
			xs = (x * self.img.shape[1]) - x0_idx
			xs = (xs / self.img_size[0]) * self.display_width + self.display_x

			y0_idx = self.img.shape[0] - y0_idx
			ys = (y * self.img.shape[0]) - y0_idx
			ys = (ys / self.img_size[1]) * self.display_height + self.display_y
		else:
			xs = (x * self.display_width)  + self.display_x
			ys = (y * self.display_height) + self.display_y

		return xs, ys

	
		


	# Handles updating of the zoom rectangle (dashed line) and updating
	# of the coordinates that indicate the current zoom rectangle.
	def _move(self, inst, val):
		if self.click_down:
			if self.parent.is_zooming:
				pos    = Window.mouse_pos
				self.drag_w = pos[0] - self.drag_start[0]
				self.drag_h = pos[1] - self.drag_start[1] 

				if self.current_zoom_rect_obj is not None:
					self.canvas.remove(self.current_zoom_rect_obj)

				self.current_zoom_rect_obj = InstructionGroup()
				self.current_zoom_rect_obj.add(Color(0, 0, 0, 1))
				self.current_zoom_rect_obj.add(Line(
					rectangle=(
						self.drag_start[0],  
						self.drag_start[1], 
						self.drag_w, 
						self.drag_h
					),
					dash_length=5,
					dash_offset=5,
					width=1
				))
				self.canvas.add(self.current_zoom_rect_obj)
	
	# This ensures that the image being displayed will maintain its native 
	# aspect ratio and will draw in bounds so that the user can see all of 
	# it. Very importantly, display_x, display_y, display_width and display_height
	# are set during this event.
	def _resize(self, size, pos):
		# We need to honor the desired padding for the image display, as 
		# well as the aspect ratio of the loaded image.

		# Take the larger dimension of the image and figure out what it 
		# needs to be to fit in the window.

		if self.is_loaded:
			largest   = max(self.img_size[0], self.img_size[1])
			w_largest = True if size[0] > size[1] else False

			if w_largest:
				display_height = size[1] - self.padding[1] - self.padding[3]
				display_width  = int(self.aspect * display_height)

				if display_width > size[0] - self.padding[0] - self.padding[2]:
					display_width  = size[0] - self.padding[0] - self.padding[2]
					display_height = int((1 / self.aspect) * display_width)
			else:
				display_width  = size[0] - self.padding[0] - self.padding[2]
				display_height = int((1 / self.aspect) * display_width)

				if display_height > size[1] - self.padding[1] - self.padding[3]:
					display_height = size[1] - self.padding[1] - self.padding[3]
					display_width  = int(self.aspect * display_height)
			
			# Calculate how to center the image.
			self.display_x = pos[0] + (size[0] // 2) - (display_width // 2)
			self.display_y = pos[1] + (size[1] // 2) - (display_height // 2)

			self.image_rect.size = (
				display_width, 
				display_height
			)

			self.image_rect.pos  = (
				self.display_x, 
				self.display_y
			)

			self.display_width  = display_width
			self.display_height = display_height

		else:
			self.image_rect.size = (
				size[0] - self.padding[0] - self.padding[2], 
				size[1] - self.padding[1] - self.padding[3]
			)

			self.display_width  = size[0] - self.padding[0] - self.padding[2]
			self.display_height = size[1] - self.padding[1] - self.padding[3]

			self.image_rect.pos  = (
				pos[0] + self.padding[0], 
				pos[1] + self.padding[1]
			)

			self.display_x = pos[0] + self.padding[0]
			self.display_y = pos[1] + self.padding[1]

		#self.updateContoursForZoom()

	def _update_dims(self, inst, val):
		self._resize(inst.size, inst.pos)

	# Given a set of coordinates relative to the lower left corner of the 
	# displayed image, this class will calculate appropriate indices into
	# the underlying array that stores the image, retrieve the correct
	# sub array and create a new texture object to display the zoomed image.
	# This call stacks properly onto previous zoom calls, allowing the user
	# to make multiple successive zooms.
	def zoomTo(self, x0, x1, y0, y1):
		# First, select the part of the image data that corresponds to the zoom
		# rectangle.
		if self.current_zoom_subarray is not None:
			to_zoom = self.current_zoom_subarray
		else:
			to_zoom = self.img

		# Convert screen coordinates to array indices
		x0_idx = int(round((x0 / self.display_width) * to_zoom.shape[1]))
		x1_idx = int(round((x1 / self.display_width) * to_zoom.shape[1]))
		y0_idx = int(round((y0 / self.display_height) * to_zoom.shape[0]))
		y1_idx = int(round((y1 / self.display_height) * to_zoom.shape[0]))

		# Account for the different coordinate system. Arrays grow
		# from top to bottom in their first dimension, while kivy screen
		# coordinates are in the first cartesian quandrant.
		y0_idx = to_zoom.shape[0] - y0_idx - 1
		y1_idx = to_zoom.shape[0] - y1_idx - 1

		self.current_zoom_subarray = to_zoom[
			y1_idx:y0_idx + 1,
			x0_idx:x1_idx + 1,
			:
		]

		# This is ugly, but necessary to ensure that successive zoom operations
		# are tracked. The self.last_zoom variable produced by this block is used
		# to calculate conversions between screen and relative coordinates.
		if self.last_zoom is not None:
			x0_idx = int(round((x0 / self.display_width)  * to_zoom.shape[1]))
			y0_idx = int(round((y0 / self.display_height) * to_zoom.shape[0]))
			old_x0_idx, old_x1_idx, old_y0_idx, old_y1_idx = self.last_zoom

			x0_idx += old_x0_idx
			y0_idx  = y0_idx + (self.img.shape[0] - old_y0_idx)
			y0_idx  = self.img.shape[0] - y0_idx
			self.last_zoom = [
				x0_idx, x1_idx, y0_idx, y1_idx
			]
		else:
			x0_idx = int(round((x0 / self.display_width)  * to_zoom.shape[1]))
			y0_idx = int(round((y0 / self.display_height) * to_zoom.shape[0]))
			y0_idx = to_zoom.shape[0] - y0_idx
			self.last_zoom = [
				x0_idx, x1_idx, y0_idx, y1_idx
			]

		# Modify some variables that are necessary for the resize event to
		# run properly.
		self.img_size = (
			self.current_zoom_subarray.shape[1],
			self.current_zoom_subarray.shape[0]
		)
		self.aspect     = self.current_zoom_subarray.shape[1]
		self.aspect    /= self.current_zoom_subarray.shape[0]
		self.img_buffer = np.fliplr(np.rot90(np.rot90(self.current_zoom_subarray)))
		self.img_buffer = memoryview(self.img_buffer.flatten())

		# Load a new texture object into graphics memory so it can be displayed.
		self.image_texture = Texture.create(
			size=self.img_size, 
			colorfmt='bgr'
		)
		self.image_texture.blit_buffer(
			self.img_buffer, 
			colorfmt='bgr', 
			bufferfmt='ubyte'
		)

		self.image_rect.texture = self.image_texture

		# Here we force a resize. This will ensure that proper aspect ratio 
		# is maintained.
		self._resize(self.size, self.pos)

	# Undo all zooming operations.
	def reset(self):
		if self.is_loaded:
			self.current_zoom_subarray = None
			self.last_zoom             = None
			self.setImage(self.img)


	# Takes a numpy/cv2 image and displays it.
	def setImage(self, img):
		self.is_loaded  = True
		self.img_size   = (img.shape[1], img.shape[0])
		self.aspect     = img.shape[1] / img.shape[0]
		self.img        = img
		self.img_buffer = np.fliplr(np.rot90(np.rot90(img)))
		self.img_buffer = memoryview(self.img_buffer.flatten())

		self.image_texture = Texture.create(size=self.img_size, colorfmt='bgr')
		self.image_texture.blit_buffer(
			self.img_buffer, 
			colorfmt='bgr', 
			bufferfmt='ubyte'
		)

		self.image_rect.texture = self.image_texture

		# Here we force a resize. This will ensure that the image does not get
		# cut off at the edges of the layout it is in.
		self._resize(self.size, self.pos)


# High level class that contains buttons for zooming and reseting the current
# view of an image. Also contains the more complicated child object that
# handles drawing, zooming and contouring of an image.
class ImageDisplay(ButtonBehavior, CustomBoxLayout):
	def __init__(self, *args, **kwargs):
		super(ImageDisplay, self).__init__(*args, **kwargs)

		self.toolbar = CustomBoxLayout(
			orientation='horizontal',
			size_hint_y=None,
			height=50,
			padding=[5, 5, 5, 5],
			spacing=5,
			border=(0, 0, 1, 0)
		)

		self.zoom_button = Button(
			text='Zoom',
			size_hint_x=None,
			size_hint_y=1,
			width=70
		)

		self.reset_button = Button(
			text='Reset Zoom',
			size_hint_x=None,
			size_hint_y=1,
			width=120
		)

		

		self.toolbar.add_widget(self.zoom_button)
		self.toolbar.add_widget(self.reset_button)

		self.add_widget(self.toolbar)

		self.image_manager = ImageManager(
			border=(0, 0, 1, 0),
			padding=(3, 3, 4, 3)
		)

		self.add_widget(self.image_manager)

		# These events just call the relevent function in this classes
		# image_manager instance and/or set flags that it reads.
		self.zoom_button.bind(on_press=self._zoom_pressed)
		self.reset_button.bind(on_press=self._reset_pressed)

		self.is_zooming = False

	def _reset_pressed(self, inst):
		self.image_manager.reset()

	
	# This handles toggling of the zoom state and coloring of the 
	# button.
	def _zoom_pressed(self, inst):
		if self.is_zooming:
			self.is_zooming = False
			self.zoom_button.background_color = self.saved_color
		else:
			self.saved_color = self.zoom_button.background_color
			self.zoom_button.background_color = hex_color('#616161')
			self.is_zooming = True

	def setImage(self, img):
		self.image_manager.setImage(img)


# # Adds a new line onto the stack.
	# def pushLine(self, color, geometry):
	# 	isg = InstructionGroup()
	# 	converted_coordinates = []
	# 	for x, y in geometry:
	# 		converted_coordinates.extend(
	# 			self.relativeCoordinatesToScreenCoordinates(x, y)
	# 		)

	# 	# The stencil operations referenced here ensure that the contours do not
	# 	# draw outside the image. Without these, the contours would be all over the 
	# 	# place when zooming in (unless manual clipping was implemented).
	# 	isg.add(StencilPush())
	# 	isg.add(Rectangle(
	# 		pos=(self.display_x, self.display_y),
	# 		size=(self.display_width, self.display_height)
	# 	))
	# 	isg.add(StencilUse())		
	# 	isg.add(Color(*color))
	# 	isg.add(Line(
	# 		points=converted_coordinates,
	# 		width=1.0
	# 	))
	# 	isg.add(StencilUnUse())
	# 	isg.add(Rectangle(
	# 		pos=(self.display_x, self.display_y),
	# 		size=(self.display_width, self.display_height)
	# 	))
	# 	isg.add(StencilPop())

	# 	self.lines.append(geometry)
	# 	self.colors.append(color)
	# 	self.instruction_groups.append(isg)
	# 	self.canvas.add(isg)

	# Pops the most recent line off of the stack.
	# def popLine(self):
	# 	self.canvas.remove(self.instruction_groups[-1])
	# 	self.instruction_groups.remove(self.instruction_groups[-1])
	# 	self.colors.remove(self.colors[-1])
	# 	self.lines.remove(self.lines[-1])

	# def clearLines(self):
	# 	# Get rid of the existing graphics objects.
	# 	for g in range(len(self.instruction_groups)):
	# 		self.canvas.remove(self.instruction_groups[-1])

	# 	self.instruction_groups = []
	# 	self.colors             = []
	# 	self.lines              = []

	# def setContourColor(self, index, color):
	# 	self.colors[index] = color
	# 	self.updateContoursForZoom()

	# Updates the graphics objects used to render the contours. Does not 
	# update the underlying "lines" array. This is meant to store ground
	# truth coordinates, not on screen coordinates.
	# def updateContoursForZoom(self):
	# 	# We need to update all of the rectangles in the canvas that pertain
	# 	# to drawing contours.

	# 	# Get rid of the existing graphics objects.
	# 	for g in range(len(self.instruction_groups)):
	# 		self.canvas.remove(self.instruction_groups[-1])
	# 		self.instruction_groups.remove(self.instruction_groups[-1])

	# 	# Add new graphics instructions with appropriate coordinates.
		
	# 	# Redraw them normally.
	# 	for geo, color in zip(self.lines, self.colors):
	# 		isg = InstructionGroup()
	# 		converted_coordinates = []
	# 		for x, y in geo:
	# 			converted_coordinates.extend([
	# 				self.relativeCoordinatesToScreenCoordinates(x, y)
	# 			])

	# 		# The stencil operations referenced here ensure that the contours 
	# 		# do not draw outside the image. Without these, the contours would 
	# 		# be all over the place when zooming in (unless manual clipping was
	# 		# implemented).
	# 		isg.add(StencilPush())
	# 		isg.add(Rectangle(
	# 			pos=(self.display_x, self.display_y),
	# 			size=(self.display_width, self.display_height)
	# 		))
	# 		isg.add(StencilUse())		
	# 		isg.add(Color(*color))
	# 		isg.add(Line(
	# 			points=converted_coordinates,
	# 			width=1.0
	# 		))
	# 		isg.add(StencilUnUse())
	# 		isg.add(Rectangle(
	# 			pos=(self.display_x, self.display_y),
	# 			size=(self.display_width, self.display_height)
	# 		))
	# 		isg.add(StencilPop())
	# 		self.instruction_groups.append(isg)
	# 		self.canvas.add(isg)