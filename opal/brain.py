from opal.types import Module

class Brain:
    modules = {}
    
    def add_module(self, x: Module):
        self.modules[x.id] = x
        self.modules[x.id].set_brain(self)