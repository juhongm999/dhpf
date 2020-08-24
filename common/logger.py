r"""Logging"""
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath

        cls.logpath = os.path.join('logs', logpath + logtime + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n+=========== Dynamic Hyperpixel Flow ============+')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('+================================================+\n')

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_model(cls, model, epoch, val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. PCK: %5.2f.\n' % (epoch, val_pck))

    @classmethod
    def visualize_selection(cls, catwise_sel):
        r"""Visualize (class-wise) layer selection frequency"""
        if cls.benchmark == 'pfpascal':
            sort_ids = [17, 8, 10, 19, 4, 15, 0, 3, 6, 5, 18, 13, 1, 14, 12, 2, 11, 7, 16, 9]
        elif cls.benchmark == 'pfwillow':
            sort_ids = np.arange(10)
        elif cls.benchmark == 'caltech':
            sort_ids = np.arange(101)
        elif cls.benchmark == 'spair':
            sort_ids = np.arange(18)

        for key in catwise_sel:
            catwise_sel[key] = torch.stack(catwise_sel[key]).mean(dim=0).cpu().numpy()

        category = np.array(list(catwise_sel.keys()))[sort_ids]
        values = np.array(list(catwise_sel.values()))[sort_ids]
        cols = list(range(values.shape[1]))
        df = pd.DataFrame(values, index=category, columns=cols)

        plt.pcolor(df, vmin=0.0, vmax=1.0)
        plt.gca().set_aspect('equal')
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 5), df.columns[::5])
        plt.tight_layout()

        plt.savefig('%s/selected_layers.jpg' % cls.logpath)


class AverageMeter:
    r"""Stores loss, evaluation results, selected layers"""
    def __init__(self, benchamrk):
        r"""Constructor of AverageMeter"""
        if benchamrk == 'caltech':
            self.buffer_keys = ['ltacc', 'iou']
        else:
            self.buffer_keys = ['pck']

        self.buffer = {}
        for key in self.buffer_keys:
            self.buffer[key] = []
        self.sel_buffer = {}

        self.loss_buffer = []

    def update(self, eval_result, layer_sel, category, loss=None):
        for key in self.buffer_keys:
            self.buffer[key] += eval_result[key]

        for sel, cls in zip(layer_sel, category):
            if self.sel_buffer.get(cls) is None:
                self.sel_buffer[cls] = []
            self.sel_buffer[cls] += [sel]

        if loss is not None:
            self.loss_buffer.append(loss)

    def write_result(self, split, epoch=-1):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch > -1 else ''

        if len(self.loss_buffer) > 0:
            msg += 'Loss: %5.2f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += '%s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch=-1):
        msg = '[Epoch: %02d] ' % epoch if epoch > -1 else ''
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        if len(self.loss_buffer) > 0:
            msg += 'Loss: %6.2f  ' % self.loss_buffer[-1]
            msg += 'Avg Loss: %6.5f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += 'Avg %s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        Logger.info(msg)
