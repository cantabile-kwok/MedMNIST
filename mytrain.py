import os
import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from medmnist.mymodel import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO

# from spacecutter.models import OrdinalLogisticModel
# from skorch import NeuralNet

# from spacecutter.callbacks import AscensionCallback
# from spacecutter.losses import CumulativeLinkLoss


def train(model, optimizer, criterion, train_loader, device, task):
    '''
    :param train_loader: DataLoader for train set
    :param device: cuda or cpu
    :param task: what task the dataset is for (multi-class/binary-class/multi-label-binary-class)
    '''
    model.train()
    for batch, (X_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(X_train.to(device))
        if "multi-label" in task:
            # use BCEWithLogitsLoss and with pos_weight
            # first we have to calculate the pos_weight in the y_train
            num_pos = y_train.sum(0).float()

            num_pos+=1e-6
            num_neg = (y_train.shape[0] - num_pos).float()
            pos_weight = num_neg / num_pos
            y_pred = y_pred.float().to(device)
            y_train = y_train.float().to(device)
            # try to make the pos weight equal to 2 (5-25)
            pos_weight = [2 for _ in range(y_train.shape[1])]
            pos_weight = torch.FloatTensor(pos_weight)
            loss = F.binary_cross_entropy_with_logits(y_pred.cpu(), y_train.cpu(), pos_weight=pos_weight).to(device)
        else:
            y_train = y_train.squeeze().long().to(device)
            loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    


def val(model, val_loader, device, val_auc_list, task, ckpt_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    # y_true= [], y_score = []
    with torch.no_grad():
        for batch, (X_val, y_val) in enumerate(val_loader):
            outputs = model(X_val.to(device))
            if task == 'multi-label, binary-class':
                y_val = y_val.float().to(device)
                sigmoid = nn.Sigmoid()
                y_pred = sigmoid(outputs).to(device)
            else:
                y_val = y_val.squeeze().long().to(device)
                sigmoid = nn.Softmax(dim=1)
                y_pred = sigmoid(outputs).to(device)
                # targets = targets.float().resize_(len(targets), 1)
            y_true = torch.cat((y_true, y_val.float()), 0)
            y_score = torch.cat((y_score, y_pred), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(ckpt_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, which_set, data_loader, device, dataset_name, task, output_root):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(data_loader):
            outputs = model(X_test.to(device))

            if task == 'multi-label, binary-class':
                y_test = y_test.to(torch.float32).to(device)
                sigmoid = nn.Sigmoid()
                y_pred = sigmoid(outputs).to(device)
            else:
                y_test = y_test.squeeze().long().to(device)
                sigmoid = nn.Softmax(dim=1)
                y_pred = sigmoid(outputs).to(device)
                # targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, y_test.float()), 0)
            y_score = torch.cat((y_score, y_pred), 0)

        if y_true.ndim == 1:
            y_true.unsqueeze_(1).long()
        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (which_set, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, dataset_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (which_set))
            save_results(y_true, y_score, output_path)


def main(dataset_name, data_root, output_root, model_name, start_epoch, end_epoch, batch_size, lr=0.001,
         download=False):
    '''
    :param dataset_name: any of the ten
    :param data_root: root of dataset
    :param start_epoch: if 0, train from scratch; else warm start
    :param download: whether download the dataset
    '''

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    try:
        DataClass = flag_to_class[dataset_name]
    except KeyError:
        print('ERROR: Dataset name not found. Supported ones are:')
        print(
            'pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, '
            'breastmnist, organmnist_axial, organmnist_coronal, organmnist_sagittal.')
        exit(1)

    info = INFO[dataset_name]
    task = info['task']
    num_channels = info['n_channels']
    num_classes = len(info['label'])

    print('*********************CONFIGURATIONS*********************')
    print("Dataset: \t", dataset_name)
    print('Task: \t\t', task)
    print('Model: \t\t',model_name)
    print('Learning rate=\t\t',lr)
    print('batch size=\t\t',batch_size)
    print('start epoch=\t\t',start_epoch)
    print('end epoch=\t\t',end_epoch)
    print('********************************************************')

    val_auc_list = []
    ckpt_path = os.path.join(output_root, '%s_checkpoints' % (dataset_name))
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=input_root,
                              split='train',
                              transform=train_transform,
                              download=download)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=input_root,
                            split='val',
                            transform=val_transform,
                            download=download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=input_root,
                             split='test',
                             transform=test_transform,
                             download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "18" in model_name:
        model = ResNet18(in_channels=num_channels, num_classes=num_classes).to(device)
    elif "50" in model_name:
        model = ResNet50(in_channels=num_channels, num_classes=num_classes).to(device)
    else:
        print('ERROR: Unknown model name!')
        exit(1)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if start_epoch != 0:
        if os.listdir(ckpt_path):
            if not str(start_epoch - 1) in ''.join(os.listdir(ckpt_path)):
                print(f'ERROR: start_epoch ({start_epoch}) not valid')
                exit(1)
            for ckpt in os.listdir(ckpt_path):
                if 'ckpt_' + str(start_epoch - 1) + "_" in ckpt:
                    print(f'===Starting from checkpoint with auc={ckpt.split("_")[-1]}')
                    model.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt))['net'])
                    break
        else:
            print(f'Not trained to this epoch yet! ({start_epoch})')
            exit(1)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, ckpt_path, epoch)

    val_auc_list = np.array(val_auc_list)
    best_idx = int(np.argmax(val_auc_list))
    print('epoch %s is the best model' % (best_idx + start_epoch))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        ckpt_path, 'ckpt_%d_auc_%.5f.pth' % (best_idx+start_epoch, val_auc_list[best_idx]))
    model.load_state_dict(torch.load(restore_model_path)['net'])

    test(model, 'train', train_loader, device, dataset_name, task,
         output_root=output_root)
    test(model, 'val', val_loader, device, dataset_name, task,
         output_root=output_root)
    test(model, 'test', test_loader, device, dataset_name, task,
         output_root=output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='breastmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--model_name',
                        default='resnet18',
                        help='which model to use',
                        type=str)
    parser.add_argument('--input_root',
                        default='./dataset',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--start_epoch',
                        default=0,
                        help='if 0, train from scratch; else warm start',
                        type=int)
    parser.add_argument('--end_epoch',
                        default=100,
                        help='E.g. if 200, then train from start_epoch to 199 epoch',
                        type=int)
    parser.add_argument('--download',
                        default=False,
                        help='whether download the dataset or not',
                        type=bool)
    parser.add_argument('--lr',
                        default=0.001,
                        help='the learning rate',
                        type=float)
    parser.add_argument('--batch_size',
                        default=128,
                        help='batch size',
                        type=int)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = os.path.join(args.output_root, args.model_name)
    end_epoch = args.end_epoch
    download = args.download

    main(data_name,
         input_root,
         output_root,
         model_name=args.model_name,
         start_epoch=args.start_epoch,
         end_epoch=end_epoch,
         lr=args.lr,
         batch_size=args.batch_size,
         download=download)
