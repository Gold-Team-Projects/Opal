from module import Module
from context import Context
import importlib

def add_module(modules: dict, path: str):
    data = importlib.import_module(path.replace("/", "."))
    modules[data.NAME] = data.MODULE 

def process(x: str, ctx: Context):
    print(x)