from ..models.interface import SISRInterface
from ...shared_framework.training.base_handler import BaseTrainingHandler


class SISRTrainingHandler(BaseTrainingHandler):
    def __init__(self, *args, **kwargs):
        super(SISRTrainingHandler, self).__init__(*args, **kwargs)

    def setup_model(self, **kwargs):
        return SISRInterface(**kwargs)
        
