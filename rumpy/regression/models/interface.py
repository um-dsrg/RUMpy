import time
import torch
from rumpy.shared_framework.models.base_interface import ImageModelInterface


class RegressionInterface(ImageModelInterface):
    """
    Main interface for regression tasks (image in, vector out)
    """
    def __init__(self, *args, **kwargs):
        super(RegressionInterface, self).__init__(*args, **kwargs)

    def train_batch(self, lr, target_metadata, **kwargs):
        """
        LR image input, GT data is target metadata (or other regression vector)
        """
        return self.model.run_train(x=lr, y=target_metadata, **kwargs)

    def net_run_and_process(self, lr=None, target_metadata=None, *args, **kwargs):
        out_vector, loss, timing = self.model.run_eval(x=lr, y=target_metadata, **kwargs)
        return out_vector, loss, timing
