import sys, time, copy
import torch
import torch.nn as nn
import tqdm

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

def prepare_training(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model('eegt').to(device)
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, loss_scaler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args={'dataset_sizes': {'train': 1000, 'val': 197, 'test':200}}):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(50):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 50))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device).squeeze(1)
                # print(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects.double() / args['dataset_sizes'][phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
