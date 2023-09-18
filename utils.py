import torch.nn as nn
from torch import Tensor
from argparse import Namespace
from torch_geometric.loader import DataLoader
from typing import Tuple


class TwoLayerMLP(nn.Module):
    """Basic two layer perceptron."""

    def __init__(self, num_hidden) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == 'qm9':
        num_input = 15
        num_out = 1
    else:
        raise ValueError(f'Do not recognize dataset {args.dataset}.')

    if args.model_name == 'egnn':
        from models.egnn import EGNN
        model = EGNN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers
        )
    else:
        raise ValueError(f'Model type {args.model_name} not recognized.')

    return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == 'qm9':
        from qm9.utils import generate_loaders_qm9
        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized.')

    return train_loader, val_loader, test_loader