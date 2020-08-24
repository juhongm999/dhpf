r"""Superclass for semantic correspondence datasets"""
import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

from model.base.geometry import Geometry


class CorrespondenceDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, Caltech, and SPair"""
    def __init__(self, benchmark, datapath, thres, device, split):
        r"""CorrespondenceDataset constructor"""
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech':  ('Caltech-101',
                         'test_pairs_caltech_with_category.csv',
                         '101_ObjectCategories',
                         '',
                         ''),
            'spair':   ('SPair-71k',
                        'Layout/large',
                        'JPEGImages',
                        'PairAnnotation',
                        'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        if benchmark == 'caltech':
            self.max_pts = 400
        else:
            self.max_pts = 40
        self.split = split
        self.device = device
        self.imside = 240
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres
        self.transform = transforms.Compose([transforms.Resize((self.imside, self.imside)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        r"""Constructs and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        # Image as tensor
        batch['src_img'] = self.transform(src_pil).to(self.device)
        batch['trg_img'] = self.transform(trg_pil).to(self.device)

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size)
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        r"""Reads PIL image from path"""
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def get_pckthres(self, batch, imsize):
        r"""Computes PCK threshold"""
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float().to(self.device)

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image with size of (240,240)"""
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1
        x_crds = pts_list[idx][0] * (self.imside / org_imsize[0])
        y_crds = pts_list[idx][1] * (self.imside / org_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1).to(self.device)

        return kps, n_pts

    def match_idx(self, kps, n_pts):
        r"""Samples the nearst feature (receptive field) indices"""
        nearest_idx = find_knn(Geometry.rf_center, kps.t())
        nearest_idx -= (self.range_ts >= n_pts).to(self.device).long()

        return nearest_idx


def find_knn(db_vectors, qr_vectors):
    r"""Finds K-nearest neighbors (Euclidean distance)"""
    db = db_vectors.unsqueeze(1).repeat(1, qr_vectors.size(0), 1)
    qr = qr_vectors.unsqueeze(0).repeat(db_vectors.size(0), 1, 1)
    dist = (db - qr).pow(2).sum(2).pow(0.5).t()
    _, nearest_idx = dist.min(dim=1)

    return nearest_idx
