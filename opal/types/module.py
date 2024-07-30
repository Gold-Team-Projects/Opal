import  torch
import  threading
import  torch.nn                as nn
from    opal.types.message      import Message
from    queue                   import PriorityQueue

class Module(nn.Module):
    name: str               = ""
    """The name of the module (used for debugging)"""
    id: str                 = ""
    """An unique ID that other modules use to access this module."""
    continuous: bool        = False
    """`True` if the module should always be running."""
    stop_on_error: bool     = False
    """`True` if `opal` should stop when errors occur. This should be enabled for critical modules. """
    queue: PriorityQueue    = PriorityQueue()
    """Queue for messages (By priority)."""
    brain                   = None
    """The brain holding this module."""
    
    def __init__(self):
        super(Module, self).__init__()
    
    def activate(self):
        """
        Called on the activation of a continuous module.
        This should not be overwritten.
        """
        if self.continuous:
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            
    def forward(self, *args, **kwargs):
        """
        Ran on continuous module activation or when a non-continuous module is invoked.
        This should be overwritten.
        """
        raise NotImplementedError

    def enqueue(self, x, priority):
        self.queue.put((x, priority))
        self.on_message()
        
    def set_brain(self, brain):
        self.brain = brain
        
    def on_message(self):
        """
        Ran on message reception.
        This should be overwritten.
        """
        raise NotImplementedError