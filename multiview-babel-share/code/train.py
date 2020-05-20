import logging as log
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard

import net
import loss
import metric
import data

import optim
import sched
import utils.saver


class Trainer(utils.saver.TrainerSaver):
    savable = ["net", "optim", "sched", "data"]

    def __init__(self, config_file, config):
        super().__init__(cache=False)

        self.train_language_list = config.train_language_list
        self.n_train_lang = len(config.train_language_list)

        self.main_dev_list = config.main_dev_language_list
        self.n_main_dev_lang = len(config.main_dev_language_list)

        self.add_dev_list = config.add_dev_language_list
        self.n_add_dev_lang = len(config.add_dev_language_list)

        #self.subwords_to_ids = data.combine_subwords_to_ids(config.all_vocab, config.subwords)
        with open(config.subwords_to_ids, 'r') as f:
            self.subwords_to_ids = json.load(f)

        self.data = data.MultilangDataset(feats_fns=config.train_feats,
                                          align_fns=config.train_align,
                                          vocab_fns=config.train_vocab,
                                          subwords=config.subwords,
                                          subwords_to_ids=self.subwords_to_ids,
                                          min_occ_count=config.train_min_occ_count,
                                          min_seg_dur=config.train_min_seg_dur,
                                          stack_frames=config.stack_frames,
                                          batch_size=config.train_batch_size,
                                          shuffle=config.shuffle,
                                          cache=self.cache)

        # statistics
        train_subwords = set(data.combine_subwords_to_ids(config.train_vocab, config.subwords).keys())
        log.info(f"Using {len(train_subwords)} subwords in training")

        # dev sets for all training languages
        self.dev_datasets = []

        for i in range(self.n_main_dev_lang + self.n_add_dev_lang):
            data_dev = data.DevDataset(feats=config.dev_feats[i],
                                       align=config.dev_align[i],
                                       vocab=config.dev_vocab[i],
                                       subwords=config.subwords,
                                       min_occ_count=config.dev_min_occ_count,
                                       min_seg_dur=config.dev_min_seg_dur,
                                       stack_frames=config.stack_frames,
                                       batch_size=config.dev_batch_size,
                                       cache=self.cache,
                                       subwords_to_ids = self.data.unify_subwords_to_ids)

            self.dev_datasets.append(data_dev)

            # statistics
            if i < self.n_main_dev_lang:
                this_lang = self.main_dev_list[i]
            else:
                this_lang = self.add_dev_list[i-self.n_main_dev_lang]

            this_subwords = set(data.combine_subwords_to_ids([config.dev_vocab[i]], config.subwords))
            log.info(f"language {this_lang} has {len(this_subwords)} subwords, "
                     f"intersect {len(train_subwords.intersection(this_subwords))} subwords")


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

    @property
    def fine_tune(self):
        return self.config.fine_tune == "true"


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

    if trainer.fine_tune:
        trainer.net.load(tag='ft')

    if trainer.global_step > 0:
        trainer.load()

    writer = tensorboard.SummaryWriter(
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

                # average ap of languages
                main_acoustic_aps = np.zeros(trainer.n_main_dev_lang)
                main_crossview_aps = np.zeros(trainer.n_main_dev_lang)

                for i in range(trainer.n_main_dev_lang + trainer.n_add_dev_lang):

                    embs1, ids1 = [], []
                    embs2, ids2 = [], []

                    with torch.no_grad():

                        for batch in trainer.dev_datasets[i].loader:
                            ids = batch.pop("ids")
                            inv = batch.pop("inv")
                            out1, out2 = trainer.net.forward(batch, numpy=True)

                            ids1.append(ids[inv.numpy()])
                            ids2.append(ids)
                            embs1.append(out1)
                            embs2.append(out2)

                    ids1 = np.hstack(ids1)
                    ids2, ind = np.unique(np.hstack(ids2), return_index=True)
                    embs1 = np.vstack(embs1)
                    embs2 = np.vstack(embs2)[ind]

                    acoustic_ap = metric.acoustic_ap(embs1, ids1)
                    crossview_ap = metric.crossview_ap(embs1, ids1, embs2, ids2)

                    if i < trainer.n_main_dev_lang:
                        this_lang = trainer.main_dev_list[i]
                        main_acoustic_aps[i] = acoustic_ap
                        main_crossview_aps[i] = crossview_ap

                        log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                 f"global_step={trainer.global_step}, "
                                 f"language={this_lang},"
                                 f"acoustic_ap={acoustic_ap:.2f}, "
                                 f"crossview_ap={crossview_ap:.2f} ")

                        """
                        # save embeddings. Don't save for now
                        words1 = map(trainer.data_dev.ids_to_words.get, ids1)
                        writer.add_embedding(embs1, metadata=list(words1), tag="view1_embs")
                        words2 = map(trainer.data_dev.ids_to_words.get, ids2)
                        writer.add_embedding(embs2, metadata=list(words2), tag="view2_embs")
                        """

                        if i == (trainer.n_main_dev_lang - 1):
                            avg_cross_ap = np.mean(main_crossview_aps)
                            avg_acoustic_ap = np.mean(main_acoustic_aps)
                            log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                     f"global_step={trainer.global_step}, "
                                     f"avg_acoustic_ap={avg_acoustic_ap:.2f}, "
                                     f"avg_crossview_ap={avg_cross_ap:.2f} ")

                            best_so_far = trainer.sched.step(avg_cross_ap, trainer.global_step)

                            if best_so_far:
                                log.info("crossview_ap best")
                                trainer.save(best=True)

                    # additional evaluated language
                    else:
                        this_lang = trainer.add_dev_list[i-trainer.n_main_dev_lang]
                        log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                 f"global_step={trainer.global_step}, "
                                 f"language={this_lang},"
                                 f"acoustic_ap={acoustic_ap:.2f}, "
                                 f"crossview_ap={crossview_ap:.2f} ")
