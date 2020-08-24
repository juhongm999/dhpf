r"""Dynamic Hyperpixel Flow testing code"""
import argparse

from torch.utils.data import DataLoader
import torch

from common.evaluation import Evaluator
from common.logger import AverageMeter
from common.logger import Logger
from common import utils
from model.base.geometry import Geometry
from model import dhpf
from data import download


def test(model, dataloader):
    r"""Code for testing DHPF"""
    average_meter = AverageMeter(dataloader.dataset.benchmark)

    for idx, batch in enumerate(dataloader):

        # 1. DHPF forward pass
        correlation_matrix, layer_sel = model(batch['src_img'], batch['trg_img'])

        # 2. Transfer key-points (nearest neighbor assignment)
        prd_kps = Geometry.transfer_kps(correlation_matrix, batch['src_kps'], batch['n_pts'])

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(prd_kps, batch)
        average_meter.update(eval_result, layer_sel.detach(), batch['category'])
        average_meter.write_process(idx, len(dataloader))

    # Write evaluation results
    Logger.visualize_selection(average_meter.sel_buffer)
    average_meter.write_result('Test')


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Dynamic Hyperpixel Flow Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_DHPF')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'])
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'pfwillow', 'caltech', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--load', type=str, default='')
    args = parser.parse_args()
    Logger.initialize(args)
    utils.fix_randseed(seed=0)

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dhpf.DynamicHPF(args.backbone, device)
    model.load_state_dict(torch.load(args.load))
    model.eval()

    # Dataset download & initialization
    download.download_dataset(args.datapath, args.benchmark)
    test_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test')
    test_dl = DataLoader(test_ds, batch_size=args.bsz, shuffle=False)
    Evaluator.initialize(args.benchmark, args.alpha)

    # Test DHPF
    with torch.no_grad(): test(model, test_dl)
