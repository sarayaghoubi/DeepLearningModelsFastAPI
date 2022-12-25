import os.path as pth
import torch
from loguru import logger
from model_deployment_fastapi.models import UNet
from model_deployment_fastapi.models.ExchangeDtType import InputDT, Output
import numpy as np

class predictor(object):
    def __init__(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = {
            'u': UNet(n_channels=1, n_classes=1, bilinear=False),
            'd': UNet(n_channels=1, n_classes=1, bilinear=False)
        }
        self.nets['u'].to(device=device)
        self.nets['d'].to(device=device)
        self.nets['u'].load_state_dict(
            torch.load(pth.join(path, 'u.pth'), map_location=torch.device('cpu')))
        self.nets['d'].load_state_dict(
            torch.load(pth.join(path, 'd.pth'), map_location=torch.device('cpu')))
        self.nets['u'].eval()
        self.nets['d'].eval()
        self.device = device

    def _preprocess(self, features: InputDT) -> torch.Tensor:
        logger.debug("Pre-processing features.")

        x = torch.unsqueeze(torch.from_numpy(features), dim=0)
        return torch.unsqueeze(x, dim=0)

    def _post_processing(self, outputs)->(float,float):
        [u, d] = outputs
        return Output(u.item(), d.item())

    def predict(self, features: InputDT) -> Output:
        x = self._preprocess(features)
        x = x.to(device=self.device, dtype=torch.float32)
        u, d = self.nets['u'](x), self.nets['d'](x)
        logger.info(f"model predicted: {[u, d]}")
        return self._post_processing([u, d])
