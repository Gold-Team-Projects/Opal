from json import load, dump
from os import path
from importlib import import_module

VERSION     = "0.0.0"
DEFAULTS    = {
    "name":       f"Opal v{VERSION}",
    "modules":    [{ "name": "core", "path": "modules/core.py" }]
}

def main():
    if path.isfile("data.json"):
        # Not first
        data = load("data.json")
    else:
        # First
        open("data.json", "x").close()
        data = DEFAULTS

main()