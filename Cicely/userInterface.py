# Authors:     Cicely Motamedi, Adam Robinson
# Description: This file contains the main code for the microscope user interface.
#              Some of the more complicated functionality is found in other files.



from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.uix.slider import Slider
from kivy.config     import Config
from kivy.core.window import Window
from kivy.clock           import Clock
from threading import Thread

import sys
sys.path.append("..\\src")
from MicroscopeControl import MicroscopeController
from CustomBoxLayout   import CustomBoxLayout
from ImageDisplay      import ImageDisplay

import numpy as np

class accordionWidget(Accordion):
    def __init__(self, *args, **kwargs):
        kwargs['orientation'] = 'vertical'
        kwargs['size_hint_x'] = 2
        super(accordionWidget, self).__init__(*args, **kwargs)

        #root = Accordion(orientation='vertical', size_hint_x=2)

        item1 = AccordionItem(title='Camera')
        item1.add_widget(Slider(min=-100, max=100, value=25))
        self.add_widget(item1)

        item2 = AccordionItem(title='Zoom and Focus')

        box2 = BoxLayout(orientation='vertical')

        label1 = BoxLayout(orientation='horizontal')
        zoomLabel = Label(text='Zoom',size_hint_x=1)
        zoomSlider = Slider(min=0.005, max=0.995, value=0.5,size_hint_x=10)
        box2.add_widget(label1)
        label1.add_widget(zoomLabel)
        label1.add_widget(zoomSlider)
        zoomSlider.bind(value=self.setZoom)

        label2 = BoxLayout(orientation='horizontal')
        focusLabel = Label(text='Focus',size_hint_x=1)
        focusSlider = Slider(min=0, max=100, value=0.5,size_hint_x=10)
        box2.add_widget(label2)
        label2.add_widget(focusLabel)
        label2.add_widget(focusSlider)
        focusSlider.bind(value=self.setFocus)

        item2.add_widget(box2)

        self.add_widget(item2)

        item3 = AccordionItem(title='Stage Control')
        item3.add_widget(Slider(min=-100, max=100, value=25))
        self.add_widget(item3)

        item4 = AccordionItem(title='Image Settings')
        item4.add_widget(Slider(min=-100, max=100, value=25))
        self.add_widget(item4)

        self.microscope  = None
        self.zooming     = False
        self.zoom_value  = 0.5
        self.closing     = False
        self.zoom_thread = Thread(target=self.adjustZoom)
        self.zoom_thread.start()

        self.focusing     = False
        self.focus_value  = 0.5
        self.focus_thread = Thread(target=self.adjustFocus)
        self.focus_thread.start()
        

    def close(self):
        self.closing = True

    def adjustZoom(self):
        while not self.closing:
            if self.microscope is not None and not self.zooming:
                current = self.microscope.focus.getZoom()
                if np.abs(current - self.zoom_value) > 0.005 and not self.zooming:
                    def done():
                        self.zooming = False
                    self.zooming = True
                    self.microscope.focus.setZoom(self.zoom_value, corrected=False, callback=done)

    def setZoom(self, object, value):
        self.zoom_value = value

    def adjustFocus(self):
        while not self.closing:
            if self.microscope is not None and not self.focusing:
                current = self.microscope.focus.getFocus()
                if np.abs(current - self.focus_value) > 0.005 and not self.focusing:
                    def done():
                        self.focusing = False
                    self.focusing = True
                    self.microscope.focus.setFocus(self.focus_value, corrected=False, callback=done)

    def setFocus(self, object, value):
        self.focus_value = value

    def setMicroscope(self, ms):
        self.microscope = ms



class userInterface(BoxLayout):
    
    def initializeMicroscope(self):
        self.microscope = MicroscopeController()

    def close(self):
        self.accordion.close()

    def __init__(self, **kwargs):
        kwargs['orientation'] = 'horizontal'
        super(userInterface, self).__init__(**kwargs)
        self.accordion = accordionWidget()
        self.add_widget(self.accordion)
        self.display = BoxLayout(orientation='vertical', size_hint_x=4)
        self.microscope = None

        Thread(target=self.initializeMicroscope).start()

        def checkMicroscope(a):
            if self.microscope is not None:
                self.accordion.setMicroscope(self.microscope)
                self.microscope.camera.startCapture()
                img = self.microscope.camera.getFrame()
                img = np.rot90(img, 3, axes=(0, 1))
                img = np.flipud(img)
                self.image_display.setImage(img)
        Clock.schedule_interval(checkMicroscope, 1/25)


        # def _check_process(b):
        #         self.load_progress.value = self.current_progress
        #         if not t.is_alive():
        #             Clock.unschedule(_check_process)
        #             self.load_progress.opacity = 0
        #             self._parent_obj.interface.preview_pane.loadThumbnails(
        #                 self._parent_obj.dataset
        #             )

        #     Clock.schedule_interval(_check_process, .025)

        # self.display.add_widget(
        #     AsyncImage(source="https://images.squarespace-cdn.com/content/v1/5a5906400abd0406785519dd/1552662149940-G6MMFW3JC2J61UBPROJ5/ke17ZwdGBToddI8pDm48kLkXF2pIyv_F2eUT9F60jBl7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z4YTzHvnKhyp6Da-NYroOW3ZGjoBKy3azqku80C789l0iyqMbMesKd95J-X4EagrgU9L3Sa3U8cogeb0tjXbfawd0urKshkc5MgdBeJmALQKw/baelen.jpg?format=1500w",size_hint_y=10)
        # )
        self.image_display = ImageDisplay(orientation='vertical', size_hint_y=10)
        self.display.add_widget(self.image_display)
        img = np.random.normal(0.0, 127, (1024, 1024, 3)).astype(np.uint8)
        self.image_display.setImage(img)
        self.input1 = BoxLayout(orientation='horizontal',size_hint_y=1)
        self.input1.add_widget(Label(text='X=',size_hint_x=1))
        self.input1.add_widget(TextInput(size_hint_x=4))
        self.input2 = BoxLayout(orientation='horizontal',size_hint_y=1)
        self.input2.add_widget(Label(text='Y=',size_hint_x=1))
        self.input2.add_widget(TextInput(size_hint_x=4))
        self.add_widget(self.display)
        self.display.add_widget(self.input1)
        self.display.add_widget(self.input2)



class userInterfaceApp(App):
    def on_request_close(self, *args):
        print("close called")
        self.interface.close()
        return False

    def build(self):
        self.interface = userInterface()
        Window.bind(on_request_close=self.on_request_close)
        return self.interface
        
if __name__=="__main__":
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    userInterfaceApp().run()