import torch
import numpy as np
import time

from rumpy.shared_framework.models.base_architecture import MultiModel


class RcanSplitCelebHandler(MultiModel):
    """
    This model allows for training multiple sub-models, where each model is only fed a specific subset of the data selected.
    """
    def __init__(self, split_variable='gender', multi_params=None, **kwargs):
        if len(multi_params) > 2 or multi_params is None:
            raise RuntimeError(
                'Exactly two models must be specified for this multi-model system '
                '(one for each possible binary attribute).')

        self.default_allocations = ['positive', 'negative']
        self.model_targets = {}
        for model_name, model in multi_params.items():
            if model['allocation'] not in self.default_allocations or model['allocation'] is None:
                raise RuntimeError('All sub-models needs to have either a "negative" '
                                   'or "positive" allocation attribute.')
            self.model_targets[model['allocation']] = model_name

        super(RcanSplitCelebHandler, self).__init__(multi_params=multi_params, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.model_name = 'rcansplitceleb'
        self.split_variable = split_variable

    def _partition_input(self, metadata_keys, metadata):
        """
        Partitions data according to selected variable.
        :param metadata_keys: Position key of metadata.
        :param metadata: Actual metadata.
        :return: Indices of positive/negative locations.
        """

        split_pos = int(np.where([self.split_variable in m for m in metadata_keys])[0])

        positive_indices = np.where([m[split_pos] == 1 for m in metadata])[0]
        negative_indices = np.where([m[split_pos] == 0 for m in metadata])[0]
        return [positive_indices, negative_indices]

    def run_train(self, x, y, tag=None, mask=None, extra_channels=None, metadata=None,
                  metadata_keys=None, *args, **kwargs):
        splits = self._partition_input(metadata_keys, metadata)
        loss_package = {}
        sum_loss = 0
        full_sr = torch.zeros_like(y)
        for ind, alloc in enumerate(self.default_allocations):
            if len(splits[ind]) == 0:
                loss_package['%s-loss' % alloc] = np.nan
                continue
            loss, output = self.child_models[self.model_targets[alloc]].run_train(x=x[splits[ind]], y=y[splits[ind]],
                                                                                  metadata=metadata,
                                                                                  metadata_keys=metadata_keys, **kwargs)
            full_sr[splits[ind]] = output
            loss_package['%s-loss' % alloc] = loss
            sum_loss += loss
        loss_package['train-loss'] = sum_loss

        return loss_package, full_sr

    def run_eval(self, x, y=None, request_loss=False, tag=None, metadata=None, metadata_keys=None, timing=False,
                 *args, **kwargs):

        splits = self._partition_input(metadata_keys, metadata)
        if request_loss:
            sum_loss = 0
        else:
            sum_loss = None
        full_sr = torch.zeros_like(y)
        if timing:
            tic = time.perf_counter()
        for ind, alloc in enumerate(self.default_allocations):
            if len(splits[ind]) == 0:
                continue
            output, loss, _ = self.child_models[self.model_targets[alloc]].run_eval(x=x[splits[ind]], y=y[splits[ind]],
                                                                                    timing=False,
                                                                                    request_loss=request_loss,
                                                                                    metadata=metadata,
                                                                                    metadata_keys=metadata_keys,
                                                                                    **kwargs)
            full_sr[splits[ind]] = output
            if request_loss:
                sum_loss += loss
        if timing:
            toc = time.perf_counter()
        return full_sr, sum_loss, toc-tic if timing else None

    def extra_diagnostics(self):
        print('This is a multi-model system.')
        for idx, (key, model) in enumerate(self.child_models.items()):
            model_target = list(self.model_targets.keys())[list(self.model_targets.values()).index(key)]
            print('Model %d (targeting %s attributes) has the %s architecture (with %d parameters), '
                  'and was loaded at local epoch %d.' % (
                      idx, model_target, model.model_name, model.print_parameters(), model.curr_epoch))
