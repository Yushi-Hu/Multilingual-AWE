import random
import numpy as np
import torch.utils.data as tud

import utils.saver


class StatefulBatchSampler:

    def __init__(self, examples, batch_size, shuffle=None):
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.init_iter()

    def init_iter(self):
        self.iter = 0
        if self.shuffle in ["both", "examples"]:
            random.shuffle(self.examples)
        self.batches = []
        batch = []
        for ex in self.examples:
            if len(batch) < self.batch_size:
                batch.append(ex)
            else:
                self.batches.append(batch)
                batch = [ex]
        if len(batch) > 0:
            self.batches.append(batch)
        if self.shuffle in ["both", "batches"]:
            random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while self.iter < len(self):
            batch = self.batches[self.iter]
            self.iter += 1
            yield batch
        self.init_iter()

    def state_dict(self, itr):
        return {
            "iter": self.iter - (itr._send_idx - itr._rcvd_idx),
            "batches": np.array(self.batches)
        }

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]
        self.batches = state_dict["batches"].tolist()


class MultilangStatefulBatchSampler:

    def __init__(self, all_examples, batch_size=1, shuffle=None):
        self.all_examples = all_examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter = 0
        self.batches = []
        self.init_iter()

    def init_iter(self):
        self.iter = 0
        self.batches = []

        for lang_id, examples in enumerate(self.all_examples):
            if self.shuffle in ["both", "examples"]:
                random.shuffle(examples)
            batch = []
            for ex in examples:
                cs, uid, i = ex
                ex = (lang_id, cs, uid, i)
                if len(batch) < self.batch_size:
                    batch.append(ex)
                else:
                    self.batches.append(batch)
                    batch = [ex]
            if len(batch) > 0:
                self.batches.append(batch)

        # shuffle batches across all languages
        if self.shuffle in ["both", "batches"]:
            random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while self.iter < len(self):
            batch = self.batches[self.iter]
            self.iter += 1
            yield batch
        self.init_iter()

    def state_dict(self, itr):
        return {
            "iter": self.iter - (itr._send_idx - itr._rcvd_idx),
            "batches": np.array(self.batches)
        }

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]
        self.batches = state_dict["batches"].tolist()


class StatefulDataset(tud.Dataset, utils.saver.Saver):

    def __init__(self):
        super(StatefulDataset, self).__init__()

    @property
    def iter(self):
        return self.loader.batch_sampler.iter

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self.iterator

    def __len__(self):
        return len(self.loader)

    def state_dict(self):
        return self.loader.batch_sampler.state_dict(self.iterator)

    def load_state_dict(self, state_dict):
        self.loader.batch_sampler.load_state_dict(state_dict)
