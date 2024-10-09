import urllib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ace_network import Encoder
from ace_network import Regressor
import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct

use_homogeneous = True

input_tensor = torch.zeros(1,480,640)
input_batch = input_tensor.unsqueeze(0)
encoder_weights = torch.load('ace_encoder_pretrained.pt', map_location=torch.device('cpu'))
print(input_batch.shape)

mean = torch.zeros((3,))
regressor = Regressor.create_from_encoder(encoder_weights, mean, 1, use_homogeneous)
trace = torch.jit.trace(regressor, input_batch)
print('traced')

std_dev = 0.25
mean = 0.4

mlmodel = ct.convert(
    trace,
    inputs=[ct.ImageType(name="input",
			 shape=input_batch.shape,
                         scale=1/(255.0*std_dev),
                         bias=-mean/std_dev,
                         color_layout=ct.colorlayout.GRAYSCALE,
                         channel_first=True)],
)
mlmodel.save("ace_regressor_image.mlpackage")

# save the weights so we can compare
state_dict = regressor.state_dict()
torch.save(state_dict, 'testing_snapshot.pt')
