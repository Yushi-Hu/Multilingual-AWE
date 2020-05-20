import logging as log
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.utils.tensorboard

import net
import loss
import metric
import data
import vocab

import optim
import sched
import utils.saver


class Trainer(utils.saver.TrainerSaver):

  savable = ["net", "optim", "sched", "data"]

  def __init__(self, config_file, config):

    super(Trainer, self).__init__()

    self.data = data.Dataset(feats=config.train_feats,
                             align=config.train_align,
                             vocab=config.vocab,
                             subwords=config.subwords,
                             min_occ_count=config.train_min_occ_count,
                             min_seg_dur=config.train_min_seg_dur,
                             stack_frames=config.stack_frames,
                             batch_size=config.train_batch_size,
                             shuffle=config.shuffle,
                             cache=self.cache)

    self.data_dev = data.Dataset(feats=config.dev_feats,
                                 align=config.dev_align,
                                 vocab=config.vocab,
                                 subwords=config.subwords,
                                 word_counts=self.data.word_counts,
                                 min_occ_count=config.dev_min_occ_count,
                                 min_seg_dur=config.dev_min_seg_dur,
                                 stack_frames=config.stack_frames,
                                 batch_size=config.dev_batch_size,
                                 cache=self.cache)

    self.vocab = vocab.Vocab(words_to_ids=self.data_dev.words_to_ids,
                             word_to_subwords=self.data_dev.word_to_subwords,
                             subwords_to_ids=self.data_dev.subwords_to_ids)

    loss_fun = loss.Obj02(margin=config.loss_margin,
                          k=config.loss_k)

    self.net = net.MultiViewRNN(config=config,
                                feat_dim=self.data.feat_dim,
                                num_subwords=self.data.num_subwords,
                                loss_fun=loss_fun,
                                use_gpu=True)

    self.optim = optim.Adam(params=self.net.parameters(), lr=config.adam_lr)

    self.sched = sched.RevertOnPlateau(network=self.net,
                                       optimizer=self.optim,
                                       mode=config.mode,
                                       factor=config.factor,
                                       patience=config.patience,
                                       min_lr=config.min_lr)


    expt_dir = os.path.dirname(config_file)
    save_dir = os.path.join(expt_dir, "save")
    self.set_savepaths(save_dir=save_dir)

    self.config_file = config_file
    self.config = config

  @property
  def global_step(self):
    return self.config.global_step


if __name__ == "__main__":
  log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")
  torch.backends.cudnn.enabled = False

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", help="configuration filename")
  args = parser.parse_args()

  config_file = args.config
  with open(config_file, "r") as f:
    config = argparse.Namespace(**json.load(f))

  random.seed(config.global_step)
  np.random.seed(config.global_step)
  torch.manual_seed(config.global_step)

  trainer = Trainer(config_file, config)
  if trainer.global_step > 0:
    trainer.load()

  writer = torch.utils.tensorboard.SummaryWriter(
      log_dir=os.path.join(trainer.save_dir, "tensorboard"))

  eval_interval = trainer.config.eval_interval or len(trainer.data)

  while not trainer.optim.converged:

    for iter_, batch in enumerate(trainer.data, trainer.data.iter):

      trainer.net.train()
      trainer.optim.zero_grad()

      ids = batch.pop("ids")
      inv = batch.pop("inv")

      out1, out2 = trainer.net.forward(batch)
      loss_val = trainer.net.loss_fun(out1, out2, inv)

      grad_norm = trainer.net.backward(loss_val)
      trainer.optim.step()

      log.info(f"batch {iter_}) "
               f"global_step={trainer.global_step}, "
               f"loss={loss_val.data.item():.3f}, "
               f"grad_norm={grad_norm:.2f}, "
               f"segments={len(inv)}, "
               f"words={len(ids)}")

      trainer.config.global_step += 1

      if trainer.global_step % eval_interval == 0:

        trainer.net.eval()

        embs1, ids1 = [], []
        embs2, ids2 = [], []

        with torch.no_grad():

          for batch in trainer.data_dev.loader:
            ids = batch.pop("ids")
            inv = batch.pop("inv")
            out = trainer.net.forward_view("view1", batch, numpy=True)
            ids1.append(ids[inv.numpy()])
            embs1.append(out)

          for batch in trainer.vocab.loader:
            ids = batch.pop("ids")
            out = trainer.net.forward_view("view2", batch, numpy=True)
            ids2.append(ids)
            embs2.append(out)

        ids1 = np.hstack(ids1)
        embs1 = np.vstack(embs1)
        ids2 = np.hstack(ids2)
        embs2 = np.vstack(embs2)

        acoustic_ap = metric.acoustic_ap(embs1, ids1,
                                         k=1e7, low_mem=True)
        crossview_ap = metric.crossview_ap(embs1, ids1, embs2, ids2,
                                           k=1e7, low_mem=True)

        best_so_far = trainer.sched.step(crossview_ap, trainer.global_step)

        log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                 f"global_step={trainer.global_step}, "
                 f"acoustic_ap={acoustic_ap:.2f}, "
                 f"crossview_ap={crossview_ap:.2f}, "
                 f"{'(best)' if best_so_far else ''}")

        if best_so_far:
          trainer.save(best=True)
          writer.add_embedding(embs1, metadata=ids1.tolist(),
                               global_step=trainer.global_step,
                               tag="view1_embs")
          writer.add_embedding(embs2, metadata=ids2.tolist(),
                               global_step=trainer.global_step,
                               tag="view2_embs")
