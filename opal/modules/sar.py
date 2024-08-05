import  opal
import  torch
import  torch.nn    as nn
from    json        import load
from    time        import sleep
from    io          import StringIO

class SARModule(opal.Module):
    name            = "State-Action-Reward Module"
    id              = "controller-sar"
    continuous      = True
    stop_on_error   = True
    
    def setup(self, instream: StringIO, interval, input_size, output_size, hidden_size=256, hidden_layers=4, trainer=opal.modules.trainers.PPOModule()):
        self.input_layer    = nn.Linear(input_size, hidden_size)
        self.output_layer   = nn.Linear(hidden_size, output_size)
        self.hidden_layers  = [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        self.instream       = instream
        self.outstream      = StringIO()
        self.interval       = interval / 100
        self.trainer        = trainer
        self.trainer.setup()

    def foward(self):
        if self.training == False:
            while True:
                sleep(self.interval)
                state = load(self.instream)
                
                x = self.input_layer.forward(torch.tensor(state))
                
                for layer in self.hidden_layers:
                    x = layer.forward(x)
                    
                x = self.output_layer.forward(x)
                self.outstream.write(x)
        else:
            try: self.brain.modules[self.trainer.id].forward("ping")
            except: self.brain.add_module(self.trainer)
            
            while True:
                sleep(self.interval)
                