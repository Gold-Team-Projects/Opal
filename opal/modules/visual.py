import  opal
import  torch
import  torchvision
import  torch.nn    as nn
import  torchvision.transforms as transforms

class ImageClassificationModule(opal.Module):
    name        = "Image Classification Module"
    id          = "classifier-img"
    continuous  = False

class ImageGeneratorModule(opal.Module):
    name        = "Image Generator Module"
    id          = "gen-img"
    continuous  = False