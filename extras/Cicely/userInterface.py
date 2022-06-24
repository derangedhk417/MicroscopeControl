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
from kivy.uix.gridlayout import GridLayout

import sys
sys.path.append("../../src")
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

        instructions = Label(text='Enter a value between 0 and 1',size_hint_y=1)
        box2.add_widget(instructions)

        label1       = BoxLayout(orientation='horizontal',size_hint_y=1)
        zoomLabel    = Label(text='Zoom',size_hint_x=None,width=100, size_hint_y=None, height=40)
        ZminusButton = Button(text='-',size_hint_x=None,width=30,size_hint_y=None,height=40)
        self.zoomInput    = TextInput(text='0.005',multiline=False,size_hint_x=None,width=100,size_hint_y=None,height=40)
        ZplusButton  = Button(text='+',size_hint_x=None,width=30,size_hint_y=None,height=40)
        label1.add_widget(zoomLabel)
        label1.add_widget(ZminusButton)
        label1.add_widget(self.zoomInput)
        label1.add_widget(ZplusButton)
        box2.add_widget(label1)
        self.zoomInput.bind(on_text_validate=self.setZoom)
        ZminusButton.bind(on_release=self.incrementZoomMinus)
        ZplusButton.bind(on_release=self.incrementZoomPlus)

        instructions2 = Label(text='Enter a value between 0 and 1',size_hint_y=1)
        box2.add_widget(instructions2)

        label2 = BoxLayout(orientation='horizontal',size_hint_y=1)
        focusLabel = Label(text='Focus',size_hint_x=None, width=100,size_hint_y=None,height=40)
        FminusButton = Button(text='-',size_hint_x=None,width=30,size_hint_y=None,height=40)
        self.focusInput = TextInput(text='0.005',multiline=False,size_hint_x=None,width=100,size_hint_y=None,height=40)
        FplusButton = Button(text='+',size_hint_x=None,width=30,size_hint_y=None,height=40)
        box2.add_widget(label2)
        label2.add_widget(focusLabel)
        label2.add_widget(FminusButton)
        label2.add_widget(self.focusInput)
        label2.add_widget(FplusButton)
        self.focusInput.bind(on_text_validate=self.setFocus)
        FminusButton.bind(on_release=self.incrementFocusMinus)
        FplusButton.bind(on_release=self.incrementFocusPlus)

        item2.add_widget(box2)

        self.add_widget(item2)

        item3 = AccordionItem(title='Stage Control')

        gridLayout = GridLayout(cols=3)
        gridLayout.add_widget(Button(opacity=0))
        moveUp = Button(text='Up')
        gridLayout.add_widget(moveUp)
        gridLayout.add_widget(Button(opacity=0))
        moveLeft = Button(text='Left')
        gridLayout.add_widget(moveLeft)
        gridLayout.add_widget(Button(opacity=0))
        moveRight = Button(text='Right')
        gridLayout.add_widget(moveRight)
        gridLayout.add_widget(Button(opacity=0))
        moveDown = Button(text='Down')
        gridLayout.add_widget(moveDown)
        gridLayout.add_widget(Button(opacity=0))

        moveUp.bind(on_press=self.clockMoveUp)
        moveUp.bind(on_release=self.stopClock)

        item3.add_widget(gridLayout)
        self.add_widget(item3)

        item4 = AccordionItem(title='Image Settings')
        item4.add_widget(Slider(min=-100, max=100, value=25))
        self.add_widget(item4)

        self.microscope  = None
        # self.zooming     = False
        # self.zoom_value  = 0.5
        # self.closing     = False
        # self.zoom_thread = Thread(target=self.adjustZoom)
        # self.zoom_thread.start()

        # self.focusing     = False
        # self.focus_value  = 0.5
        # self.focus_thread = Thread(target=self.adjustFocus)
        # self.focus_thread.start()
        

    def close(self):
        pass

    # def adjustZoom(self):
    #     while not self.closing:
    #         if self.microscope is not None and not self.zooming:
    #             current = self.microscope.focus.getZoom()
    #             if np.abs(current - self.zoom_value) > 0.005 and not self.zooming:
    #                 def done():
    #                     self.zooming = False
    #                 self.zooming = True
    #                 self.microscope.focus.setZoom(self.zoom_value, corrected=False, callback=done)

    def setZoom(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting zoom")

        #print(dir(object))
        value = object.text
        print(value)
        try:
            value = float(value)
        except Exception as ex:
            return



        if value >= 0.005 and value <= 0.995: 
            print(value)
            self.microscope.focus.setZoom(value, corrected=False, cb=done)
        else:
            print("Invalid input")


    # def adjustFocus(self):
    #     while not self.closing:
    #         if self.microscope is not None and not self.focusing:
    #             current = self.microscope.focus.getFocus()
    #             if np.abs(current - self.focus_value) > 0.005 and not self.focusing:
    #                 def done():
    #                     self.focusing = False
    #                 self.focusing = True
    #                 self.microscope.focus.setFocus(self.focus_value, corrected=False, callback=done)

    def setFocus(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting focus")

        value = object.text 
        print(value)
        try:
            value=float(value)
        except Exception as ex:
            return

        if value >= 0.005 and value <= 0.995:
            self.microscope.focus.setFocus(value, corrected=False, cb=done)
        else:
            print("Invalid input")

    def setMicroscope(self, ms):
        self.microscope = ms

    def incrementZoomMinus(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting focus")

        value = float(self.zoomInput.text)
        value -= 0.1
        self.zoomInput.text = "%1.4f"%value

        if value >= 0.005 and value <= 0.995:
            self.microscope.focus.setZoom(value, corrected=False, cb=done)
        else:
            print("Invalid input")

    def incrementZoomPlus(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting focus")

        value = float(self.zoomInput.text)
        value += 0.1
        self.zoomInput.text = "%1.4f"%value

        if value >= 0.005 and value <= 0.995:
            self.microscope.focus.setZoom(value, corrected=False, cb=done)
        else:
            print("Invalid input")

    def incrementFocusMinus(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting focus")

        value = float(self.focusInput.text)
        value -= 0.1
        self.focusInput.text = "%1.4f"%value

        if value >= 0.005 and value <= 0.995:
            self.microscope.focus.setFocus(value, corrected=False, cb=done)
        else:
            print("Invalid input")

    def incrementFocusPlus(self, object):
        def done(error, val):
            if error is None:
                print(val)
            else:
                print("Error setting focus")

        value = float(self.focusInput.text)
        value += 0.1
        self.focusInput.text = "%1.4f"%value

        if value >= 0.005 and value <= 0.995:
            self.microscope.focus.setFocus(value, corrected=False, cb=done)
        else:
            print("Invalid input")

    def moveIncrementUp(self,a):
        def done(error, value):
            if error is None:
                print(value)
            else:
                print(error)

        self.microscope.stage.moveDelta(0, 0.01, cb=done)

    def clockMoveUp(self,a):
        Clock.schedule_interval(self.moveIncrementUp,0.01)

    def stopClock(self, a):
        Clock.unschedule(self.moveIncrementUp)


class userInterface(BoxLayout):
    
    def initializeMicroscope(self):
        self.microscope = MicroscopeController()

    def close(self):
        self.accordion.close()
        if self.microscope is not None:
            self.microscope.cleanup()

    def __init__(self, **kwargs):
        kwargs['orientation'] = 'horizontal'
        super(userInterface, self).__init__(**kwargs)
        self.accordion = accordionWidget()
        self.add_widget(self.accordion)
        self.display = BoxLayout(orientation='vertical', size_hint_x=4)
        self.microscope = None

        Thread(target=self.initializeMicroscope).start()

        self.microscope_loaded = False

        def checkMicroscope(a):
            if not self.microscope_loaded and self.microscope is not None:
                self.microscope_loaded = True
                self.microscope.camera.enableLowRes()
                self.accordion.setMicroscope(self.microscope)
                self.microscope.camera.start_capture()

            if self.microscope_loaded:
                img = self.microscope.camera.getFrame()
                img = np.rot90(img, 3, axes=(0, 1))
                img = np.flipud(img)
                self.image_display.setImage(img)
        Clock.schedule_interval(checkMicroscope, 1 / 10)


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

        self.input1 = BoxLayout(orientation='horizontal',size_hint_y=None, height=30)

        xLabel = Label(text='X=',size_hint_x=1, size_hint_y=None, height=30)
        self.xInput = TextInput(multiline=False,size_hint_x=4, size_hint_y=None, height=30)
        yLabel = Label(text='Y=',size_hint_x=1, size_hint_y=None, height=30)
        self.yInput = TextInput(multiline=False,size_hint_x=4, size_hint_y=None, height=30)


        self.input1.add_widget(xLabel)
        self.input1.add_widget(self.xInput)

        #self.input2 = BoxLayout(orientation='horizontal',size_hint_y=1)

        self.input1.add_widget(yLabel)
        self.input1.add_widget(self.yInput)

        self.add_widget(self.display)
        self.display.add_widget(self.input1)
        #self.display.add_widget(self.input2)

        self.yInput.bind(on_text_validate=self.moveTo)
        self.xInput.bind(on_text_validate=self.moveTo)

    def moveTo(self, object):
        def done():
            if error is None:
                print(val)
            else:
                print("Error moving stage")


        xvalue = self.xInput.text
        yvalue = self.yInput.text
        print(xvalue, yvalue)

        if xvalue.strip() == "" or xvalue is None:
            xvalue = "0.0"

        if yvalue.strip() == "" or yvalue is None:
            yvalue = "0.0"

        try:
            xvalue=float(xvalue)
            yvalue=float(yvalue)
        except Exception as ex:
            return

        print(xvalue, yvalue)

        if xvalue >= -50 and xvalue <= 50:
            if yvalue >= -44 and yvalue <= 37:
                self.microscope.stage.moveTo(xvalue, yvalue, callback=done)
        else:
            print("Invalid input")



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