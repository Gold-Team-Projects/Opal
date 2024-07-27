import opal
import torch
import torch.nn as nn
import enum

class LanguageMemoryModule(opal.Module):
    name = "Language Memory Module"
    continuous = False
    stop_on_error = False
    
    class ReqTypes(enum.Enum):
        GET_TRANSLATION = "get-translate"
        """Gets the embeds of a sentence."""
    
    def __init__(self, embedding_dim):
        self.w2i = {}
        self.i2w = {}
        self.embedding = nn.Embedding(0, embedding_dim)
    
    def run(self, req, *args):
        if req == "get-translate":
            return self.embedding(torch.tensor([self.w2i[token] for token in args[0]]))
        elif req == "add":
            self.w2i[args[0]]= args[1]
            