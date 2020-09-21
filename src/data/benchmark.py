import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data
import glob
import pdb

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            if "HR" in filename:
                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
        #pdb.set_trace()
        for entry in os.scandir(self.dir_lr):
            filename = os.path.splitext(entry.name)[0]
            if "LR" in filename:
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(
                        self.dir_lr, filename + self.ext))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.all_files = glob.glob(os.path.join(self.apath, 'HR', "*.png"))
        #self.dir_lr = os.path.join(dir_data, self.name, 'Test/3')
        #self.dir_hr = os.path.join(dir_data, self.name, 'Test/3')
        self.dir_lr = os.path.join(dir_data, self.name, 'LR','X4')
        self.dir_hr = os.path.join(dir_data, self.name, 'HR')
        #self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png'