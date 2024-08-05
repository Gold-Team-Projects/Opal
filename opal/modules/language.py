import  re
import  csv
import  opal
import  torch
import  spacy
import  string
import  threading
import  importlib
import  torch.nn            as nn
import  torch.optim         as optim
import  torch.nn.functional as F
import  torch.nn.utils      as utils

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
    
    def setup(self, brain, lang="en"):
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
    name            = "Language Generator Module"
    id              = "gen-lang"
    continuous      = False
    stop_on_error   = False
    
    default_window_size = 6
    
    def setup(self, brain, lang: str="en", hidden_size=256, dropout=0.2, num_layers=2, bidirectional=True):
        super(LanguageGeneratorModule, self).__init__()
        self.set_brain(brain)
        
        try: self.brain.modules["memory-lang"].forward("ping")
        except: self.brain.add_module(LanguageMemoryModule(brain, lang))
        
        self.lstm = nn.LSTM(input_size=0, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
    
    async def forward(self, type, *args, **kwargs):
        if type == "train": 
            try: window = args[1]
            except: window = self.default_window_size
            try: kwargs["file"] = kwargs["file"]
            except: kwargs["file"] = False
            try: kwargs["asynchronous"] = kwargs["asynchronous"]
            except: kwargs["asynchronous"] = False
            
            if kwargs["file"] == True: 
                def fn():
                    for path in args[0]:
                        x = {"predict": [], "respond": []}
                        y = {"predict": [], "respond": []}
                        
                        if path.find("csv") != -1:
                            f = open(path, "r")
                            reader = csv.reader(f)
                            reader.next()
                            
                            for row in reader:
                                row[0] = self.brain.modules["memory-lang"].forward("translate-forward", row[0])
                                row[1] = self.brain.modules["memory-lang"].forward("translate-forward", row[1])
                                x["respond"].append(row[0])
                                y["respond"].append(row[1])
                                
                                def fn2(tokens):
                                    for i in range(len(tokens)):
                                        window_tokens = tokens[window:i] if i > 6 else tokens[:i]
                                        x["predict"].append(window_tokens)
                                        y["predict"].append(tokens[i])
                                            
                                fn2(self.brain.modules["memory-lang"].forward("translate-forward", row[0]))
                                fn2(self.brain.modules["memory-lang"].forward("translate-forward", row[1]))
                            
                            self.brain.modules["memory-lang"].forward("train", f.read())
                            
                            f.close()
                            
                        x["predict"] = torch.tensor(x["predict"], torch.float64)
                        x["respond"] = torch.tensor(x["respond"], torch.float64)
                        y["predict"] = torch.tensor(y["predict"], torch.float64)
                        y["respond"] = torch.tensor(y["respond"], torch.float64)
                        
                        
                if kwargs["asynchronous"] == True:
                    thread = threading.Thread(target=fn)
                    thread.daemon = True
                    thread.start()
                    thread.join()
                else: fn()
                
            
            else:
                pass