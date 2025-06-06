import torch

from utilis.train.train_on_automated_labels import train_on_automated_labels
from utilis.train.train_on_basic_labels import train_on_basic_labels
from utilis.val.validate_on_automated_labels import validate_automated_labels
from utilis.val.validate_on_basic_labels import validate_on_basic_labels


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch, 'has_mps', False) and torch.has_mps:
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    train_on_basic_labels(device)
    # train_on_automated_labels(device)

    #validate_on_basic_labels(device)
    #validate_automated_labels(device)




if __name__ == '__main__':
    main()