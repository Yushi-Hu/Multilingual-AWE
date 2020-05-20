import logging as log
import os
import re
import json
import signal
import subprocess
import torch
import torch.nn as nn

import utils.cache as cache


class Saver:

    def set_savepath(self, save_dir, name, keep=5):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.savepath = os.path.join(self.save_dir, name + ".{}.pth")
        self.pattern = re.compile(self.savepath.format("[0-9]+"))
        self.keep = keep

    def save(self, tag, best=False):
        savepath = self.savepath.format(tag)
        torch.save(self.state_dict(), savepath)
        log.info(f"saved {os.path.relpath(savepath)}")

        if best:
            best_savepath = self.savepath.format("best")
            if os.path.exists(best_savepath):
                os.remove(best_savepath)
            os.symlink(savepath, best_savepath)
            log.info(f"linked {os.path.relpath(best_savepath)}")

            self.clean()

    def clean(self):
        files = {}
        for base in os.listdir(self.save_dir):
            file_ = os.path.join(self.save_dir, base)
            if re.match(self.pattern, file_):
                files[file_] = os.path.getmtime(file_)

        if len(files) > self.keep:
            for old_file in sorted(files, key=files.get)[:-self.keep]:
                if os.path.exists(old_file):
                    os.remove(old_file)
                log.info(f"removed {os.path.relpath(old_file)}")

    def load_from_fullpath(self, loadpath):
        self.load_state_dict(torch.load(loadpath))
        log.info(f"loaded {os.path.relpath(loadpath)}")
        log.info(f"note: relative to {os.getcwd()}")

    def load(self, tag):
        loadpath = self.savepath.format(tag)
        self.load_from_fullpath(loadpath)


class NetSaver(nn.Module):

    def __init__(self):
        super(NetSaver, self).__init__()
        self.net = nn.ModuleDict()

    def __iter__(self):
        return iter(self.net)

    def __getitem__(self, key):
        return self.net[key]

    def __setitem__(self, key, value):
        self.net[key] = value

    def set_savepath(self, save_dir, name):
        for key in self:
            self[key].set_savepath(save_dir, f"{name}-{key}")

    def save(self, tag, best=False):
        for key in self:
            self[key].save(tag, best=best)

    def load(self, tag):
        for key in self:
            self[key].load(tag)


class TrainerSaver:

    def __init__(self, cache=False):
        self.cache = None
        if "SLURM_JOB_ID" in os.environ:
            if cache:
                self.cache = cache.Cache()
            self.slurm_job_id = os.environ["SLURM_JOB_ID"]
            signal.signal(signal.SIGUSR1, self._slurm_handler)

    def set_savepaths(self, save_dir):
        self.save_dir = save_dir
        for attr in self.savable:
            getattr(self, attr).set_savepath(save_dir=self.save_dir, name=attr)

    def save(self, tag=None, best=False):
        if tag is None:
            tag = self.global_step
        for attr in self.savable:
            getattr(self, attr).save(tag=tag, best=best)

    def load(self, tag=None):
        if tag is None:
            tag = self.global_step
        for attr in self.savable:
            getattr(self, attr).load(tag=tag)

    @property
    def best_so_far(self):
        return self.global_step == self.best_global_step

    def _slurm_handler(self, sig_num, frame):
        self.save()
        with open(self.config_file, "w") as f:
            json.dump(vars(self.config), f)
        subprocess.call(f"scontrol requeue {self.slurm_job_id}".split())
