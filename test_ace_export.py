import urllib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from ace_network import Encoder
import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct

input_tensor = torch.zeros(1, 448, 448)
input_batch = input_tensor.unsqueeze(0)
encoder_weights = torch.load(
    "ace_encoder_pretrained.pt", map_location=torch.device("cpu")
)
encoder = Encoder()
encoder.load_state_dict(encoder_weights)
print(input_batch.shape)
trace = torch.jit.trace(encoder, input_batch)

mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)
mlmodel.save("ace_encoder.mlpackage")
