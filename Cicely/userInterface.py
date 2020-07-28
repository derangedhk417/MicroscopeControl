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

def accordionWidget():
    root = Accordion(orientation='vertical')
    item1 = AccordionItem(title='Camera')
    item1.add_widget(Slider(min=-100, max=100, value=25))
    root.add_widget(item1)
    item2 = AccordionItem(title='Zoom')
    item2.add_widget(Slider(min=-100, max=100, value=25))
    root.add_widget(item2)
    item3 = AccordionItem(title='Focus')
    item3.add_widget(Slider(min=-100, max=100, value=25))
    root.add_widget(item3)
    item3 = AccordionItem(title='Stage Control')
    item3.add_widget(Slider(min=-100, max=100, value=25))
    root.add_widget(item3)
    item3 = AccordionItem(title='Image Settings')
    item3.add_widget(Slider(min=-100, max=100, value=25))
    root.add_widget(item3)
    return root

class userInterface(BoxLayout):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(accordionWidget())
        s = Scatter()
        i = AsyncImage(source="http://test-ipv6.com/images/hires_ok.png")
        self.add_widget(s)
        s.add_widget(i)
        self.add_widget(TextInput(text='X=',size_hint_y=None))
        self.add_widget(TextInput(text='Y=',size_hint_y=None))

class userInterfaceApp(App):
    def build(self):
        return userInterface()
        
if __name__=="__main__":
	userInterfaceApp().run()