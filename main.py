import glob
import sys

from args import get_args_parser
from data import get_loaders, generate_data, split_data
from model import eegt
from engine import prepare_training, train_model
from torchsummary import summary

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args(args=[])
    sys.stdout = open('logs/exp_4000_drop_5e-6.txt', 'w')
    model, optimizer, lr_scheduler, criterion, device, _ = prepare_training(args)
    print(summary(model, (59, 4000)))

    calib_files = glob.glob('data/*.mat')
    X, y = generate_data(calib_files)
    train_X, train_y, val_X, val_y, test_X, test_y = split_data(X, y)
    dataloaders = get_loaders(train_X, train_y, val_X, val_y, test_X, test_y)
    
    best_model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders)
