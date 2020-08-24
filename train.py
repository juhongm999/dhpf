r"""Dynamic Hyperpixel Flow training (validation) code"""
import argparse

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from common.evaluation import Evaluator
from common.logger import AverageMeter
from common.logger import Logger
from common import supervision as sup
from common import utils
from model.base.geometry import Geometry
from model.objective import Objective
from model import dhpf
from data import download


def train(epoch, model, dataloader, strategy, optimizer, training):
    r"""Code for training DHPF"""
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    for idx, batch in enumerate(dataloader):

        # 1. DHPF forward pass
        src_img, trg_img = strategy.get_image_pair(batch, training)
        correlation_matrix, layer_sel = model(src_img, trg_img)

        # 2. Transfer key-points (nearest neighbor assignment)
        prd_kps = Geometry.transfer_kps(strategy.get_correlation(correlation_matrix), batch['src_kps'], batch['n_pts'])

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(prd_kps, batch)

        # 4. Compute loss to update weights
        loss = strategy.compute_loss(correlation_matrix, eval_result, layer_sel, batch)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_meter.update(eval_result, layer_sel.detach(), batch['category'], loss.item())
        average_meter.write_process(idx, len(dataloader), epoch)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = utils.mean(average_meter.buffer['pck'])
    return avg_loss, avg_pck


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Dynamic Hyperpixel Flow Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_DHPF')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'])
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--supervision', type=str, default='strong', choices=['weak', 'strong'])
    parser.add_argument('--selection', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--bsz', type=int, default=8)
    args = parser.parse_args()
    Logger.initialize(args)
    utils.fix_randseed(seed=0)

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dhpf.DynamicHPF(args.backbone, device)
    Objective.initialize(args.selection, args.alpha)
    strategy = sup.WeakSupStrategy() if args.supervision == 'weak' else sup.StrongSupStrategy()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

    # Dataset download & initialization
    download.download_dataset(args.datapath, args.benchmark)
    trn_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn')
    val_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val')
    trn_dl = DataLoader(trn_ds, batch_size=args.bsz, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.bsz, shuffle=False)
    Evaluator.initialize(args.benchmark, args.alpha)

    # Train DHPF
    best_val_pck = float('-inf')
    for epoch in range(args.niter):

        trn_loss, trn_pck = train(epoch, model, trn_dl, strategy, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_pck = train(epoch, model, val_dl, strategy, optimizer, training=False)

        # Save the best model
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            Logger.save_model(model, epoch, val_pck)
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/pck', {'trn_pck': trn_pck, 'val_pck': val_pck}, epoch)

    Logger.tbd_writer.close()
    Logger.info('==================== Finished training ====================')
