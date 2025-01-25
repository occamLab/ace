import urllib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from ace_network import Encoder
from ace_network import Regressor
import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct

# note: works with use_homogeneous = False (not with True)
use_homogeneous = True

input_tensor = torch.zeros(1, 448, 448)
input_batch = input_tensor.unsqueeze(0)
encoder_weights = torch.load(
    "ace_encoder_pretrained.pt", map_location=torch.device("cpu")
)
print(input_batch.shape)

mean = torch.zeros((3,))
regressor = Regressor.create_from_encoder(encoder_weights, mean, 1, use_homogeneous)
trace = torch.jit.trace(regressor, input_batch)
print("traced")

mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)
mlmodel.save("ace_regressor.mlpackage")
