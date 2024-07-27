import  torch
import  threading
import  torch.nn        as nn
from    types.message   import Message
from    queue           import PriorityQueue

class Module(nn.Module):
    name: str               = ""
    """The name of the module (used for debugging)"""
    continuous: bool        = False
    """`True` if the module should always be running."""
    stop_on_error: bool     = False
    """`True` if `opal` should not handle errors."""
    queue: PriorityQueue    = PriorityQueue()
    """Queue for messages (By priority)."""
    on_message: function    = None
    """Run on message reception"""
    
    def activate(self):
        """
        Called on the activation of a continuous module.
        This should not be overwritten.
        """
        if self.continuous:
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            
    def run(self, *args, **kwargs):
        """
        Ran on continuous module activation or when a non-continuous module is invoked.
        This should be overwritten.
        """
        raise NotImplementedError

    def enqueue(self, x, priority):
        self.on_message()
        self.queue.put((x, priority))