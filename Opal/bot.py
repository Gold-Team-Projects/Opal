from module import Module
import importlib

def add_module(modules: dict, path: str):
    data = importlib.import_module(path.replace("/", "."))
    modules[data.NAME] = data.MODULE 