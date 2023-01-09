import os.path as pth
import torch
from loguru import logger
from DeepLearningModelsFastAPI.model_deployment_fastapi.models.UNet import UNet30
from DeepLearningModelsFastAPI.model_deployment_fastapi.models.ExchangeDtType import InputDT, Output
from pathlib import Path


class Predictor(object):
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = {
            'u': UNet30(n_channels=1, n_classes=1),
            'd': UNet30(n_channels=1, n_classes=1)
        }
        self.nets['u'].to(device=device)
        self.nets['d'].to(device=device)
        base_dir = Path(__file__).resolve(strict=True).parent
        self.nets['u'].load_state_dict(
            torch.load(pth.join(base_dir, 'u.pth'), map_location=torch.device('cpu')))
        self.nets['d'].load_state_dict(
            torch.load(pth.join(base_dir, 'd.pth'), map_location=torch.device('cpu')))
        self.nets['u'].eval()
        self.nets['d'].eval()
        self.device = device

    @staticmethod
    def _preprocess(features: InputDT) -> torch.Tensor:
        logger.debug("Pre-processing features.")

        x = torch.unsqueeze(torch.from_numpy(features), dim=0)
        return torch.unsqueeze(x, dim=0)

    @staticmethod
    def _post_processing(outputs) -> (float, float):
        [u, d] = outputs
        return Output(u.item(), d.item())

    def predict(self, features: InputDT) -> Output:
        x = self._preprocess(features)
        x = x.to(device=self.device, dtype=torch.float32)
        u, d = self.nets['u'](x), self.nets['d'](x)
        logger.info(f"model predicted: {[u, d]}")
        return self._post_processing([u, d])
