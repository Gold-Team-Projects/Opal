from json import load, dump
from os import path
from importlib import import_module

VERSION     = "0.0.0"
DEFAULTS    = {
    "name":       f"Opal v{VERSION}",
    "modules":    [{ "name": "core", "path": "modules/core.py" }]
}

def main():
    data = {}
    first = False
    
    if path.isfile("data.json"):
        # Not first
        data = load(open("data.json", "r"))
        
    else:
        # First
        open("data.json", "x").close()
        data = DEFAULTS
        dump(data, open("data.json", "w"))
        first = True
    
    if first == True: print("First time!")
    if first == False: print("Not First time!")

main()