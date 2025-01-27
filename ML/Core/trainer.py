import importlib.util
import json
import psutil
import torch
import os
import torchinfo
import random
import numpy as np
import pandas as pd

from typing import Any, Literal
from datetime import datetime
from ML.Core.custom_loss import LossFactory
from Utilities import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger
from ML.Core.normalizer import NormalizerFactory


class Trainer:
    """
    The Trainer class takes in a torch model, alongside some training parameters, and performs testing,
    evaluating and prediction.
    """

    def __init__(
            self,
            features_path: str,
            targets_path: str,
            train_percent: float,
            val_percent: float,
            feature_win: int,
            target_win: int,
            intersect: int,
            batch_size: int,
            model_class_name: str,
            model_args: dict,
            exp_name: str,
            optimizer_name: str,
            criterion_name: str,
            patience: int,
            patience_tolerance: float,
            n_epochs: int,
            seed: int,
            features_norm_method: str,
            targets_norm_method: str,
            features_global_normalizer: bool,
            targets_global_normalizer: bool,
            flip_history: bool,
            regularization_factor: float,
            hyperparams: dict[str:Any] | None = None
    ):
        self._set_seed(seed)

        self.init_args = {key: val for key, val in locals().copy().items() if key != 'self'}
        for key, val in self.init_args.items():
            setattr(self, key, val)

        self._stop_training: bool = False
        self._early_stopping: int = 0
        self._best_val_loss: float | torch.Tensor = float('inf')
        self._tb_writer: SummaryWriter | None = None

        self.model_dir: str = ""
        self.best_model_path: str = ""
        self.info_path = ""

        self.features_normalizer = NormalizerFactory.create(features_norm_method, features_global_normalizer)
        self.targets_normalizer = NormalizerFactory.create(targets_norm_method, targets_global_normalizer)

        self.features: torch.Tensor | list[torch.Tensor] = torch.load(features_path)
        self.targets: torch.Tensor | list[torch.Tensor] = torch.load(targets_path)
        self._handle_tensor_types_and_flip_history(flip_history)

        self.device: torch.cuda.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module = self._construct_model()
        self.loss_fn: torch.nn.modules.loss
        self.regularization_fn: None | callable
        self._construct_criterion()  # setting the loss and regularization functions
        self.optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters())
        self._create_model_dir()
        use_var_len = True if (isinstance(self.features, list) and isinstance(self.targets, list)) else False
        self.data_dict = utils.train_val_test_split(
            use_variable_length_dataset=use_var_len,
            features=self.features,
            targets=self.targets,
            train_percent=self.train_percent,
            val_percent=self.val_percent,
            feature_window_size=self.feature_win,
            target_window_size=self.target_win,
            intersect=self.intersect,
            batch_size=self.batch_size,
            features_normalizer=self.features_normalizer,
            targets_normalizer=self.targets_normalizer
        )
        self.train_loader = self.data_dict['train']['loader']
        self.val_loader = self.data_dict['val']['loader']
        self.hyperparams = hyperparams
        logger.add(os.path.join(self.model_dir, "trainer.log"), level="INFO", rotation="500 MB", compression="zip")
        logger.info(f"\nHyperparams:\n {json.dumps(hyperparams, sort_keys=True, indent=4)}")
        logger.info(
            f"\nModel:\n {torchinfo.summary(self.model, input_size=(self.batch_size, self.feature_win, self.features[0].shape[-1]), depth=5, verbose=0)}")

    def _handle_tensor_types_and_flip_history(self, flip_history: bool) -> None:
        """
        for both the features and the targets, sets the tensors as float types and flips the history dimension,
        if needed. handles both tensors with shape (N,H,F) or list of tensors, each with shape (Hi,F) features and
        targets. To flip the 3d tensor we use `torch.fliplr` - an alternative to [:,::-1,:]. to flip the list of 2D
        tensor we use torch.flip on the first dim.
        :param flip_history: boolean indicating whether to keep the history dim intact or flip it
        """
        if isinstance(self.features, torch.Tensor):
            self.features = torch.fliplr(self.features.float()) if flip_history else self.features.float()
        elif isinstance(self.features, list):
            self.features = [torch.flip(t.float(), [0]) if flip_history else t.float() for t in self.features]

        if isinstance(self.targets, torch.Tensor):
            self.targets = torch.fliplr(self.targets.float()) if flip_history else self.targets.float()
        elif isinstance(self.targets, list):
            self.targets = [torch.flip(t.float(), [0]) if flip_history else t.float() for t in self.targets]

    def _construct_criterion(self):
        """ populates the loss and regularization functions based on the provided criterion name """
        if '+' in self.criterion_name:
            loss_name, regularization_name = self.criterion_name.replace(' ', '').split('+')
            self.loss_fn = LossFactory.get_loss(loss_name)
            self.regularization_fn = LossFactory.get_loss(regularization_name)
        else:  # no regularization
            self.loss_fn = LossFactory.get_loss(self.criterion_name)
            self.regularization_fn = lambda x: 0  # just a function that does nothing

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _construct_model(self):
        """ uses the provided model name and model args and returns a suitable model object """
        file_path = os.path.join('Zoo', f'{self.model_class_name}.py')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The model file '{file_path}' does not exist")
        # importing the relevant model dynamically:
        spec = importlib.util.spec_from_file_location(self.model_class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_class_name = os.path.basename(self.model_class_name)  # extract the last component of the path
        model_class = getattr(module, model_class_name)
        model = model_class(**self.model_args)
        return model.to(self.device)

    def _create_model_dir(self):
        """ called when a trainer is initialized and creates a model dir with relevant information txt file(s) """
        init_timestamp = datetime.now().strftime(utils.TIME_FORMAT)
        self.model_dir = os.path.join(os.getcwd(), 'saved_models', f"{self.model}_{self.exp_name}_{init_timestamp}")
        self.info_path = os.path.join(self.model_dir, 'trainer_info.yaml')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        utils.update_json(self.info_path, self.init_args)

    def _calc_criterion(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        calculates the final loss using the loss and regularization functions.
        Assumes reg term takes only the predictions
        """
        loss_term = self.loss_fn(predictions, targets)
        regularization_term = self.regularization_fn(predictions)
        return loss_term + self.regularization_factor * regularization_term

    def _reset_loss_state(self):
        if hasattr(self.regularization_fn, 'reset_state'):
            self.regularization_fn.reset_state()
        if hasattr(self.loss_fn, 'reset_state'):
            self.loss_fn.reset_state()

    def _train_one_epoch(self, epoch: int) -> float:
        """ performs a training process for one epoch """
        self.model.train(True)
        self._reset_loss_state()
        total_loss = 0.0
        tqdm_loader = tqdm(self.train_loader)
        for inputs, targets in tqdm_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            predictions = self.model(inputs)
            loss = self._calc_criterion(predictions, targets)
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            tqdm_loader.set_postfix({'Train Loss': f"{loss.item():.5f}",
                                     'RAM_%': psutil.virtual_memory().percent,
                                     'GPU_%': utils.get_gpu_usage_percentage(),
                                     'Epoch': epoch})
        total_loss /= len(self.train_loader)
        return total_loss

    def _evaluate(self, epoch: int) -> float:
        """ evaluates the model on the validation set """
        self.model.train(False)
        self._reset_loss_state()

        total_loss = 0.0
        with torch.no_grad():
            tqdm_loader = tqdm(self.val_loader)
            for inputs, targets in tqdm_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                loss = self._calc_criterion(predictions, targets)
                total_loss += loss
                tqdm_loader.set_postfix({'Val Loss': f"{loss.item():.5f}",
                                         'RAM_%': psutil.virtual_memory().percent,
                                         'Epoch': epoch})
            total_loss /= len(self.val_loader)
        return total_loss

    def _finish_epoch(self, val_loss, prev_val_loss, time, epoch):
        """ saves the best model and performs early stopping if needed """
        if val_loss < self._best_val_loss:  # if val is improved, we update the best model
            self._best_val_loss = val_loss
            if self.best_model_path:
                os.remove(self.best_model_path)
            self.best_model_path = os.path.join(self.model_dir, f"{self.model}_{self.exp_name}_{time}.pt")
            torch.save(self.model.state_dict(), self.best_model_path)
        elif torch.abs(val_loss - prev_val_loss) <= self.patience_tolerance:  # if val doesn't improve, counter += 1
            self._early_stopping += 1
        if self._early_stopping >= self.patience:  # if patience value is reached, the training process halts
            self._stop_training = True
            logger.info(f"[Early Stopping] at epoch #{epoch}.\n"
                        f"Best model path: {self.best_model_path}\n"
                        f"Best val loss: {self._best_val_loss.item():.4f}")

    def fit(self) -> None:
        """ fits the model to the training data, with early stopping """
        # add hparams with the su
        fit_time = datetime.now().strftime(utils.TIME_FORMAT)
        self._tb_writer = SummaryWriter(os.path.join('tb_runs', os.path.basename(self.model_class_name),
                                                     f"{self.model}_{self.exp_name}_{fit_time}"))

        prev_val_loss = float('inf')
        for epoch in range(self.n_epochs):
            train_avg_loss = self._train_one_epoch(epoch)
            val_avg_loss = self._evaluate(epoch)
            self._tb_writer.add_scalar('Loss/Train', train_avg_loss, epoch)
            self._tb_writer.add_scalar('Loss/Val', val_avg_loss, epoch)
            logger.info(
                f'[Epoch {epoch}/{self.n_epochs}] '
                f'Train: {train_avg_loss.item():.4f} | '
                f'Val: {val_avg_loss.item():.4f}'
            )
            self._finish_epoch(val_avg_loss, prev_val_loss, fit_time, epoch)
            if self._stop_training:
                break
            prev_val_loss = val_avg_loss

        self._tb_writer.add_hparams(
            hparam_dict={
                **{k: v for k, v in self.hyperparams.items() if not isinstance(v, tuple)},
                **{'params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}
            },
            metric_dict={
                'hparam/best_val_loss': self._best_val_loss.item()
            }
        )

    def predict(
            self,
            dataset_name: Literal['ours', 'prssm'],
            split: str,
            add_inputs: bool
    ) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
        """
        creates a prediction for every loader in our loaders list based on the desired dataset
        :param dataset_name: the name of the dataset,
        :param split: either train val or test
        :param add_inputs: if true, merges the input alongside the predictions
        :return: A tuple with the following:
                    - dictionary with dataset index (int) keys and dataframe of predicted values, with possible inputs
                    - a dataframe of loss value per dataset.
        """
        merge_helper = lambda df_to_merge, name, idx: pd.merge(
            merged_df, df_to_merge.add_prefix(f'[{name}] [#{idx}] '), how='outer', left_index=True, right_index=True
        )
        all_loaders = self.data_dict[split]['all_dataloaders']
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.model.train(False)
        predictions = dict()
        losses = dict()
        with torch.no_grad():
            for j, data_loader in enumerate(all_loaders):
                merged_df = pd.DataFrame()
                if add_inputs:
                    curr_inputs = pd.DataFrame(
                        self.features_normalizer.inverse_transform(
                            self.data_dict[split]['all_datasets'][j].normalized_features.squeeze(0)
                        ),
                        columns=['F1', 'F2', 'F3', 'F4'] if dataset_name == 'ours' else ['Fx', 'Fy', 'Fz', 'My', 'Mz']
                    )
                    merged_df = merge_helper(curr_inputs, 'input', j)
                curr_preds, curr_trues = [], []
                for inputs_i, true_i in tqdm(data_loader, desc=f"Predicting on {split} loader #{j}"):
                    inputs_i = inputs_i.to(self.device)
                    pred_i = self.model(inputs_i).to('cpu')
                    pred_i = self.targets_normalizer.inverse_transform(pred_i)
                    curr_preds.append(pred_i.squeeze())
                    curr_trues.append(true_i.squeeze())
                if self.target_win == 1:  # only supports target_win == 1 for prediction df
                    merged_df = merge_helper(pd.DataFrame(curr_preds), 'pred', j)
                    merged_df = merge_helper(pd.DataFrame(curr_trues), 'true', j)
                    merged_df = merged_df.map(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
                    predictions[j] = merged_df
                losses[j] = self.loss_fn(torch.stack(curr_preds), torch.stack(curr_trues)).item()
        losses_df = pd.DataFrame(losses, index=[0]).rename(columns={k: f"data #{k}" for k in losses.keys()})
        losses_df['mean'] = losses_df.mean(axis=1)
        logger.info(f"Results for {dataset_name} [{split} set]:\n {losses_df.to_string()}")
        return predictions, losses_df

    @classmethod
    def from_model_dirname(cls, model_dirname):
        """
        Initializes a trainer instance using the trainer_info defined with the model provided in model_dirname.
        This way, a trained model can be loaded with the trainer that was specifically instantiated when it was trained.
        :param model_dirname: path to the dirname containing the model.pt file and trainer_info.yaml file
        :return: the trainer object
        """
        model_pt_path = ""
        trainer_yaml_path = ""
        for file_name in os.listdir(model_dirname):
            if file_name.endswith('.pt'):
                model_pt_path = os.path.join(model_dirname, file_name)
            elif file_name.endswith('.yaml'):
                trainer_yaml_path = os.path.join(model_dirname, file_name)

        with open(trainer_yaml_path, 'r') as f:
            trainer_args = json.load(f)

        trainer = cls(**trainer_args)
        trainer.best_model_path = model_pt_path
        trainer.model.load_state_dict(torch.load(model_pt_path, map_location=trainer.device))
        return trainer
