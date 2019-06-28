import matplotlib
matplotlib.use('Agg')

import os
import json
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from utils.dataset_csv import CSVDataset, CSVDatasetValidate
from utils.utils import init_seeds, compute_loss, non_max_suppression, test_model
from utils.torch_utils import select_device
from utils.plot_utils import plot_images
from utils.augs import *

from models.yolov3_tiny import YOLOv3Tiny
from models.yolov3_tiny_mobilenet import YOLOv3TinyMobile as YOLOv3TinyMobile
from models.yolov3_tiny_squeeze import YOLOv3TinySqueeze as YOLOv3TinySqueeze
from models.yolov3_tiny_shuffle import YOLOv3TinyShuffle as YOLOv3TinyShuffle

import tensorboardX

import shutil
from datetime import datetime


def save_log(logs_writer: tensorboardX.SummaryWriter, model, mean_loss, imgs,
             epoch, labels, metrics):
    with torch.no_grad():
        model.eval()
        p, _ = model(imgs)
    model.train()
    det = non_max_suppression(p, 0.1, 0.1)[0]
    logs_writer.add_scalar('xy', mean_loss[0], epoch)
    logs_writer.add_scalar('wh', mean_loss[1], epoch)
    logs_writer.add_scalar('conf', mean_loss[2], epoch)
    logs_writer.add_scalar('cls', mean_loss[3], epoch)
    logs_writer.add_scalar('total', mean_loss[4], epoch)

    if metrics is not None:
        logs_writer.add_scalar('mAP', metrics['AP'], epoch)
        logs_writer.add_scalar('AP', metrics['AP50'], epoch)
        logs_writer.add_scalar('IOU', metrics['IOU'], epoch)

    if det is None:
        logs_writer.add_image_with_boxes('example_result', imgs[0], np.array([]),
                                         epoch)
    else:
        labels = labels[det[:, -1].cpu().numpy().astype(int)]
        logs_writer.add_image_with_boxes('example_result', imgs[0], det[:, :4],
                                         epoch, labels=labels)


def check_paths(save_dir, name, data_path: str, val_path, logs_path, no_save):
    save_dir = os.path.join(save_dir, name)

    data_dir, data_name = os.path.split(data_path)
    labels_path = ''
    if not val_path:
        prefix = 'train_'
        if data_name.startswith(prefix):
            val_path =  'val_' + data_name[len(prefix):]
            val_path = os.path.join(data_dir, val_path)
            labels_path = 'labels_' + data_name[len(prefix):]
            labels_path = os.path.join(data_dir, labels_path)

    if not no_save:
        if os.path.exists(save_dir) and name == 'test':
            shutil.rmtree(save_dir)

        if os.path.exists(save_dir):
            if not os.path.isdir(save_dir):
                raise ValueError('Save dir is not a dir: ' + save_dir)

            shutil.move(save_dir, f'{save_dir}_{datetime.now()}')
        os.makedirs(save_dir)

        data_dir, data_name = os.path.split(data_path)
        new_data_path = os.path.join(save_dir, data_name)
        shutil.copy(data_path, new_data_path)
        data_path = new_data_path

        if val_path:
            _, val_name = os.path.split(val_path)
            new_val_path = os.path.join(save_dir, val_name)
            shutil.copy(val_path, new_val_path)
            val_path = new_val_path

        if os.path.exists(labels_path):
            new_labels_path = os.path.join(save_dir, 'labels_' + data_name)
            shutil.copy(labels_path, new_labels_path)

        labels_path = os.path.join(data_dir, 'labels_' + name)
        new_labels_path = os.path.join(save_dir, 'labels_' + name)
        if os.path.exists(labels_path):
            shutil.copy(labels_path, new_labels_path)
        else:
            pass

    logs_path = os.path.join(logs_path, name)
    if os.path.exists(logs_path) and name == 'test':
        shutil.rmtree(logs_path)
    if os.path.exists(logs_path):
        shutil.move(logs_path, f'{logs_path}_{datetime.now()}')
    os.makedirs(logs_path, exist_ok=True)

    if not os.path.isdir(save_dir):
        raise ValueError('Save dir is not a dir: ' + save_dir)
    if not os.path.isdir(logs_path):
        raise ValueError('Logs dir is not a dir: ' + logs_path)
    if not os.path.isfile(data_path):
        raise ValueError('Data path is not a file: ' + data_path)
    if val_path and not os.path.isfile(val_path):
        raise ValueError('Validate path is not a dir: ' + val_path)

    return save_dir, data_path, val_path, logs_path


def train(opt):
    config_path = opt.config_path
    hyper_params = None

    if config_path:
        with open(config_path, 'r') as file:
            hyper_params = json.load(file)

        for key, item in hyper_params.items():
            setattr(opt, key, item)

        opt.hyper_params = config_path

    name = opt.name
    hyper_params_path = opt.hyper_params
    data_csv_path = opt.data_csv
    encoder_weights_path = opt.encoder_weights
    img_size = opt.img_size
    epochs = opt.epochs
    batch_size = opt.batch_size
    multi_scale = opt.multi_scale
    augment = not opt.no_aug
    mixed = opt.mixed
    num_workers = opt.num_workers
    no_save = opt.no_save
    in_channels = opt.in_channels
    save_dir = opt.save_dir
    kernels_divider = opt.kernels_divider
    logs_path = opt.logs_path
    val_path = opt.validate
    encoder = opt.encoder
    use_class_weights = opt.use_class_weights

    if hyper_params is None:
        with open(hyper_params_path, 'r') as file:
            hyper_params = json.load(file)

    save_dir, data_csv_path, val_path, logs_path = check_paths(save_dir, name,
                                                               data_csv_path,
                                                               val_path,
                                                               logs_path,
                                                               no_save)
    logs_writer = tensorboardX.SummaryWriter(logs_path)

    latest_weights = os.path.join(save_dir, f'{name}_latest.pt')
    best_weights = os.path.join(save_dir, f'{name}_best.pt')
    print('Latest weights path: ', latest_weights)
    print('Best weights path: ', best_weights)

    init_seeds()
    device = select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size

        if num_workers != 0:
            print('In multi scale mode forced num_workers=0', num_workers)
            num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    if augment:
        transform = Compose([
            # RandomHSVYOLO(),
            RandomAffineYOLO(),
            RandomFlip(),
        ])
    else:
        transform = None


    # Dataset
    dataset = CSVDataset(data_csv_path, img_size, transform=transform,
                         in_channels=in_channels)
    if val_path:
        val_dataset = CSVDatasetValidate(val_path, img_size,
                                         in_channels=in_channels)
    else:
        val_dataset = None
    labels = dataset.labels

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Initialize model
    if encoder == 'mobile':
        yolo_class = YOLOv3TinyMobile
    elif encoder == 'squeeze':
        yolo_class = YOLOv3TinySqueeze
    elif encoder == 'shuffle':
        yolo_class = YOLOv3TinyShuffle
    else:
        yolo_class = YOLOv3Tiny
    model = yolo_class(in_channels=in_channels,
                       n_class=dataset.cls_number,
                       anchors=hyper_params['anchors'],
                       hyper_params=hyper_params,
                       kernels_divider=kernels_divider,)
    model = model.to(device)
    if encoder_weights_path:
        model.load_darknet_weights(encoder_weights_path, warnings=False)

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=hyper_params['init_lr'],
                          momentum=hyper_params['sgd_momentum'],
                          weight_decay=hyper_params['weight_decay'])
    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: 1 - 10 ** (
                hyper_params['final_lr'] * (1 - x / epochs))  # inverse exp ramp
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Mixed precision training https://github.com/NVIDIA/apex
    # install help: https://github.com/NVIDIA/apex/issues/259
    # if mixed:
    #     from apex import amp
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    best_loss = float('inf') if val_dataset is None else float('-inf')

    n_batches = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(n_batches / 5 + 1), 1000)  # burn-in batches

    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None

    class_weights = torch.from_numpy(dataset.class_weight).to(device) if use_class_weights else None

    # Start training
    t, t0 = time.time(), time.time()
    for epoch in range(epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step(epoch)

        mean_loss = torch.zeros(5).to(device)
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            n_targets = len(targets)

            # Plot images with bounding boxes
            if epoch == 0 and i <= 1:
                plot_images(imgs=imgs, targets=targets, fname=f'train_batch{i}.jpg')

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyper_params['init_lr'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model,
                                            class_weight=class_weights)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            t = time.time()

            # Update running mean of tracked metrics
            mean_loss = (mean_loss * i + loss_items) / (i + 1)

            if epoch == 0 and i == 0:
                if val_dataset is not None:
                    metrics = test_model(model, val_dataset, batch_size,
                                         num_workers, device)
                else:
                    metrics = None
                save_log(logs_writer, model, mean_loss, imgs, -1,
                         labels, metrics)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, n_batches - 1), *mean_loss, n_targets, time.time() - t)
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        if val_dataset is not None:
            metrics = test_model(model, val_dataset, batch_size,
                                 num_workers, device)
        else:
            metrics = None
        save_log(logs_writer, model, mean_loss, imgs, epoch, labels, metrics)

        # Update best_weights loss
        test_loss = results[4]

        if metrics is not None:
            if metrics['AP'] > best_loss:
                best_loss = metrics['AP']
        else:
            if test_loss < best_loss:
                best_loss = test_loss

        # Save training results
        save = (not no_save) or (epoch == epochs - 1)
        if save:
            # Create checkpoint
            model_params = {'n_class': model.n_class,
                            'onxx': model.onnx,
                            'in_channels': in_channels,
                            'kernels_divider': kernels_divider,
                            'in_shape': model.in_shape,
                            'anchors': model.anchors}

            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'hyper_params': hyper_params,
                     'model_params': model_params,
                     'model_cls': model.__class__.__name__}

            # Save latest_weights checkpoint
            torch.save(chkpt, latest_weights)

            # Save best_weights checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best_weights)

            print('Latest saved:', latest_weights)
            print('Best saved:', best_weights)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, os.path.join(save_dir , f'backup{epoch}.pt'))

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print(f'{epoch} epochs completed in {dt:.3f} hours.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='experiment name (need for save weights)', default='test')
    parser.add_argument('--data_csv', type=str, help='csv file path', default='')
    parser.add_argument('--config_path', type=str, help='config json file path', default='')
    parser.add_argument('--hyper_params', type=str, help='hyper parameters json file path', default='')
    parser.add_argument('-w', '--encoder_weights', type=str, help='weights path', default='')
    parser.add_argument('-s', '--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('-e', '--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='size of each image batch')
    parser.add_argument('-ms', '--multi_scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--no_aug', action='store_true', help='No augments')
    parser.add_argument('--mixed', action='store_true', help='Mixed precision')
    parser.add_argument('-nw', '--num_workers', type=int, default=6, help='number of Pytorch DataLoader workers')
    parser.add_argument('--no_save', action='store_true', help='do not save training results')
    parser.add_argument('--in_channels', type=int, help='image color channels', default=3)
    parser.add_argument('--save_dir', type=str, help='Results save dir', default='dumps')
    parser.add_argument('--kernels_divider', type=int, help='kernels count divider', default=1)
    parser.add_argument('--logs_path', type=str, help='path to tensorboard logs', default='train_logs')
    parser.add_argument('-v', '--validate', type=str, help='path to validate dataset', default='')
    parser.add_argument('--encoder', type=str, help='encoder type', default='darknet', choices=['darknet', 'mobile', 'squeeze', 'shuffle'])
    parser.add_argument('--use_class_weights', action='store_true', help='use class weights')
    opt = parser.parse_args()
    print(opt)

    results = train(opt)
