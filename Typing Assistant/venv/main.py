from pynput import keyboard
from pynput.keyboard import Key, Controller
import pyperclip
import httpx
from string import Template
import time

controller = Controller()

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_CONFIG = {"model": "mistral:7b-instruct-v0.2-q4_K_S",
                "keep_alive": "5m",
                "stream": False
}

PROMPT_TEMPLATE = Template(
    """
    Fix all the typos and casing and punctuation in this text, but preseve all new line characters
    
    $text
    
    Return only corrected text, don't inlcude a preamble
    """
)

def fix_text(text):
    prompt =  PROMPT_TEMPLATE.substitute(text=text)
    response = httpx.post(OLLAMA_ENDPOINT,
                        json={"prompt": prompt, **OLLAMA_CONFIG},
                        headers={"Content-Type": "application/json"},
                        timeout=10)
    if response.status_code != 200:
        return None
    return response.json()["response"].strip()

def fix_current_line():
    controller.pressed(Key.ctrl)
    controller.pressed(Key.shift)
    controller.pressed(Key.left)
    
    controller.release(Key.ctrl)
    controller.release(Key.shift)
    controller.release(Key.left)
    
    fix_selection()



def fix_selection():
    # *1. Copy to the clipboard
    with controller.pressed(Key.ctrl):
        controller.tap('c')
        
    # *2. get the text from the clipboard
    text = pyperclip.paste()
    print(text)
    
    # *3. fixing the text
    print("Fixing text....")
    fixed_text = fix_text(text)
    print(fixed_text)    
    
    # *4. copy back to the clipboard
    pyperclip.copy(fixed_text)
    print("Text recieved")
    
    # *5. inserting the text
    with controller.pressed(Key.ctrl):
        controller.tap('v')

def on_f9():
    fix_current_line()

def on_f10():
    fix_selection()

with keyboard.GlobalHotKeys({
        '<120>': on_f9,
        '<121>': on_f10}) as h:
    h.join()