import random
from abc import ABC, abstractmethod

import pandas as pd
import torch

from Utilities import utils


class Normalizer(ABC):
    def __init__(self, global_normalizer: bool = True):
        """
        initializes a normalizer object.
        :param global_normalizer: boolean value that governs the normalization dimension of an input
                                  tensor with shape (N,H,F):
                                    - If True: casts to (N*H,F) and normalizes over dim=0
                                    - If False: keeps as (N,H,F) and normalizes over dim=1
        """
        self.global_normalizer = global_normalizer
        self.norm_dim = 0 if self.global_normalizer else 1  # for dimensions (N*H,F) or (N,H,F)

    def _handle_shape(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """ casts an input data to the required dimension given the global norm boolean"""
        if not self.global_normalizer:  # TODO may not work for lists right now
            return data

        if isinstance(data, torch.Tensor):
            n_datasets, history_size, input_size = data.shape
            return data.reshape(n_datasets * history_size, input_size)

        elif isinstance(data, list):
            return torch.cat(data)

    @abstractmethod
    def fit(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        """ computes relevant statistics to be used for later scaling. """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """
        normalizes (feature-wise) a tensor with shape (N,H,F), or a list of N tensors, each with variable shape (Hi,F)
        assumes the statistics are already given
        :return: tensor t such that for every i=0,1,2,...,N-1, the matrix t[i,:,:] columns are normalized
        """
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """ calls fit and then transform one after the other """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """ converts a normalized data to its original values. assumes the statistics are already given. """
        raise NotImplementedError


class ZScore(Normalizer):
    """ zscore scaler, maps x -> [ x - mean(x) ] / [ std(x) ]"""

    def __init__(self, global_normalizer: bool = True):
        super().__init__(global_normalizer)
        self.mean_val = None
        self.std_val = None

    def fit(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        data = self._handle_shape(data)
        self.mean_val = data.mean(self.norm_dim, keepdim=True)
        self.std_val = data.std(self.norm_dim, keepdim=True)

    def transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return (data - self.mean_val) / self.std_val
        elif isinstance(data, list):
            return [(tensor - self.mean_val) / self.std_val for tensor in data]

    def fit_transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data * self.std_val + self.mean_val
        elif isinstance(data, list):
            return [tensor * self.std_val + self.mean_val for tensor in data]


class MinMax(Normalizer):
    """ min-max scaler, maps x -> [ x - min(x) ] / [ max(x) - min(x) ] """

    # TODO: add list of tensors support like in zscore
    def __init__(self, global_normalizer: bool = True):
        super().__init__(global_normalizer)
        self.min_val = None
        self.max_val = None

    def fit(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._handle_shape(tensor)
        self.min_val = tensor.min(self.norm_dim, keepdim=True).values
        self.max_val = tensor.max(self.norm_dim, keepdim=True).values

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        minmax_tensor = (tensor - self.min_val) / (self.max_val - self.min_val)
        return minmax_tensor

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        self.fit(tensor)
        return self.transform(tensor)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        inv_minmax_tensor = tensor * (self.max_val - self.min_val) + self.min_val
        return inv_minmax_tensor


class Identity(Normalizer):
    """ An identity 'normalizer' to be used when no normalization is wanted """

    def __init__(self, global_normalizer: bool = True):
        super().__init__(global_normalizer)

    def fit(self, tensor: torch.Tensor) -> None:
        return

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class NormalizerFactory:
    """ a normalizers factory class """

    @staticmethod
    def create(normalizer_type: str, global_normalizer: bool = True):
        if normalizer_type.lower() == 'zscore':
            return ZScore(global_normalizer)
        elif normalizer_type.lower() == 'minmax':
            return MinMax(global_normalizer)
        elif normalizer_type.lower() == 'identity' or normalizer_type is None:
            return Identity(global_normalizer)
        else:
            raise ValueError(f"Invalid normalizer type: {normalizer_type}")


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    zscaler = StandardScaler()
    mmscaler = MinMaxScaler()
    kinematics, forces = utils.load_data_from_prssm_paper()
    N, H, F = forces.shape
    exp = random.randint(0, N - 1)
    for global_normalizer in [False]:
        Z = NormalizerFactory.create('zscore', global_normalizer)
        MM = NormalizerFactory.create('minmax', global_normalizer)

        z_forces = Z.fit_transform(forces)
        invz_forces = Z.inverse_transform(z_forces)

        mm_forces = MM.fit_transform(forces)
        invmm_forces = MM.inverse_transform(mm_forces)

    # sk_forces = forces.reshape(N * H, F).numpy() if global_normalizer else forces.numpy()

    # z_sk_forces = torch.from_numpy(zscaler.fit_transform(sk_forces)).reshape(N, H, F)
    # mm_sk_forces = torch.from_numpy(mmscaler.fit_transform(sk_forces)).reshape(N, H, F)

    df_f = pd.DataFrame(forces[exp], columns=[f'Original_{i}' for i in range(F)])
    # df_fskz = pd.DataFrame(z_sk_forces[exp], columns=[f"Sklearn_Zcore_{i}" for i in range(F)])
    # df_fskmm = pd.DataFrame(mm_sk_forces[exp], columns=[f"Sklearn_MinMax_{i}" for i in range(F)])
    df_fz = pd.DataFrame(z_forces[exp], columns=[f"My_Zcore_{i}" for i in range(F)])
    df_finvz = pd.DataFrame(invz_forces[exp], columns=[f"My_Inv_Zscore_{i}" for i in range(F)])
    df_fmm = pd.DataFrame(mm_forces[exp], columns=[f"My_MinMax_{i}" for i in range(F)])
    df_finvmm = pd.DataFrame(invmm_forces[exp], columns=[f"My_Inv_MinMax_{i}" for i in range(F)])

    df = pd.concat([df_f, df_fz, df_fmm, df_finvz, df_finvmm], axis=1)  # , df_fskz, df_fskmm], axis=1)
# utils.plot(df, title=f"global_norm={global_normalizer}, experiment#={exp}")
