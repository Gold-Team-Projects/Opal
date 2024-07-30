import  re
import  opal
import  torch
import  spacy
import  string
import  threading
import  importlib
import  torch.nn        as nn
import  torch.optim     as optim

class LanguageMemoryModule(opal.Module):
    name            = "Language Processor Module"
    id              = "memory-lang"
    continuous      = False
    stop_on_error   = False
    
    vocab_forward   = {}
    vocab_backward  = {}
    
    new_dims        = {
        100: 50,
        500: 100,
        1000: 150,
        5000: 200,
        10000: 250,
        50000: 300,
        100000: 350
    }
    
    lock = threading.Lock()
    
    def __init__(self, brain, lang="en"):
        super(LanguageMemoryModule, self).__init__()
        self.set_brain(brain)
        
        self.lang = lang
        
        self.lock            = threading.Lock()
        self.nlp             = spacy.load(f"{lang}_core_web_sm")
        self.tokenizer       = self.nlp.tokenizer
        self.removals        = list(importlib.import_module(f"spacy.lang.{self.lang}").stop_words.STOP_WORDS)

        self.vocab_forward = {word: i for i, word in enumerate(self.nlp.vocab.strings)}
        self.vocab_backward = {i: word for i, word in enumerate(self.nlp.vocab.strings)}
        
        self.embeddings = nn.Embedding(len(self.vocab_forward), 50)
        
    def expand(self, word: str):
        try: return self.vocab_forward[word]
        except: pass
        self.lock.acquire()
        old_dim = self.embeddings.embedding_dim
        old_num = self.embeddings.num_embeddings
        old_data = self.embeddings.weight.data
        self.embeddings = nn.Embedding(old_num + 1, old_dim)
        self.embeddings.weight.data[:old_num] = old_data
        self.vocab_forward[word] = old_num + 1
        self.vocab_backward[old_num + 1] = word
        self.lock.release()
        
    def forward(self, type, *args, **kwargs):
        if type == "expand": # expand vocab
            self.expand(args[0])
        if type == "translate-forward": # str -> list<int>
            return self.prepare(args[0])
        if type == "translate-backward": # list<int> -> str
            return " ".join([self.vocab_backward[int(idx)] for idx in args[0]])
        if type == "train": pass
            
    def prepare(self, x: str):
        x = x.lower()
        x = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x)
        x = " ".join([word for word in x.split() if word not in self.removals])
        
        out = []
        
        for token in self.tokenizer(x):
            try: out.append(self.vocab_forward[token.text])
            except:
                self.expand(token.text)
                out.append(self.vocab_forward[token.text])
        
        return out
        
class LanguageGeneratorModule(opal.Module):
    
    class Encoder(nn.Module): pass
    
    class Decoder(nn.Module): pass
    
    name            = "Language Generator Module"
    id              = "lang-gen"
    continuous      = False
    stop_on_error   = False
    
    default_window_size = 6
    
    def __init__(self, brain, lang: str="en"):
        super(LanguageGeneratorModule, self).__init__()
        self.set_brain(brain)
        
        try: self.brain.modules["memory-lang"].forward("ping")
        except: self.brain.add_module(LanguageMemoryModule(brain, lang))
        
        self.lstm
    
    def forward(self, type, *args, **kwargs):
        if type == "train": 
            try: window = args[1]
            except: window = self.default_window_size
            
            if kwargs["file"] == True: 
                for path in args[0]:
                    with open(path, "r") as file:
                        x = []
                        y = []
                        
                        data = self.brain.modules["memory-lang"].forward("translate-forward", file.read())
                        
                        for i in range(len(data)):
                            try:
                                self.brain.modules["memory-lang"].forward("expand", data[i])
                                seqin = data[i:i + window]
                                seqout = data[i + window]
                                x.append(seqin)
                                y.append(seqout)
                            except: break

                        self.brain.modules["memory-lang"].forward("train", x, y, asynchronous=True)
                        
                        
                        
                        
            
            else:
                pass