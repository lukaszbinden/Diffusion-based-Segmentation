
from typing import (Generic, TypeVar, Callable)

from torch.utils.data import Dataset

__all__ = [
    "TransformedDataset"
]

Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class TransformedDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 source_dataset: Dataset,
                 transform_func: Callable[..., Tout]
                 ) -> None:
        self.source_dataset = source_dataset
        self.transform_func = transform_func

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Tout:

        value = self.source_dataset[idx]

        if isinstance(value, tuple):
            return self.transform_func(*value)

        return self.transform_func(value)
