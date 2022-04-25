import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to

class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.mixcut = config.mixcut

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        if self.mixcut > 0  and self.is_training:
            targets, tmp, tmp2 = y_true.chunk(3)
            targets = torch.squeeze(targets, 0)
        else:
            targets = y_true
        x = move_to(x, self.device)
        targets = move_to(targets, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        outputs = self.get_model_output(x, targets)

        results = {
            'g': g,
            'y_true': targets,
            'y_pred': outputs,
            'metadata': metadata,
            'mixcut_y': y_true
        }
        return results

    def objective(self, results):
        loss = self.loss['loss'] if self.is_training else self.loss['eval_loss']
        results_y = results['mixcut_y'] if self.is_training else results['y_true']
        labeled_loss = loss.compute(results['y_pred'], results_y, return_dict=False)
        return labeled_loss
