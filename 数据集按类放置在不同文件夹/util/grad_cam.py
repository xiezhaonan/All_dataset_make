import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import trycv
import torch
import numpy as np
from models import GoogleNet9


class InfoHolder():

    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()


def generate_heatmap(weighted_activation):
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()


def superimpose(input_img, heatmap):
    img = trycv.cvtColor(input_img, trycv.COLOR_BGR2RGB)
    heatmap = trycv.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap) #h w c
    eyeheatmap = heatmap[:15,:]
    faceheatmap = heatmap[15:,:]
    trycv.normalize(eyeheatmap, eyeheatmap, 0, 255, trycv.NORM_MINMAX)
    trycv.normalize(faceheatmap, faceheatmap, 0, 255, trycv.NORM_MINMAX)
    heatmap =np.concatenate((eyeheatmap,faceheatmap),axis = 0)
    print("debug",eyeheatmap.shape,faceheatmap.shape)
    print("debug2",heatmap)
    heatmap = trycv.applyColorMap(heatmap, trycv.COLORMAP_JET)#生成热力图
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    pil_img = trycv.cvtColor(superimposed_img, trycv.COLOR_BGR2RGB)
    return pil_img


def to_RGB(tensor):
    tensor = (tensor - tensor.min())
    tensor = tensor / (tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.numpy(), (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image

def grad_cam(model, input_tensor, heatmap_layer, truelabel=None):
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)

    output = model(input_tensor.unsqueeze(0))[0]
    truelabel = truelabel if truelabel else torch.argmax(output)

    output[truelabel].backward()

    weights = torch.mean(info.gradient, [0, 2, 3])
    activation = info.activation.squeeze(0)

    weighted_activation = torch.zeros(activation.shape)
    for idx, (weight, activation) in enumerate(zip(weights, activation)):
        weighted_activation[idx] = weight * activation

    heatmap = generate_heatmap(weighted_activation)
    input_image = to_RGB(input_tensor)
    return superimpose(input_image, heatmap)





