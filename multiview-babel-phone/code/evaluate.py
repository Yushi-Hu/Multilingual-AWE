import logging as log
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
import shutil

import net
import loss
import metric
import data

import optim
import sched
import utils.saver


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration filename")
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = argparse.Namespace(**json.load(f))

    # transfer to absolute path
    expt_dir = os.path.dirname(config_file)
    config.dev_feats = [os.path.join(expt_dir, i) for i in config.dev_feats]
    config.dev_align = [os.path.join(expt_dir, i) for i in config.dev_align]
    config.dev_vocab = [os.path.join(expt_dir, i) for i in config.dev_vocab]
    config.feature_fn = os.path.join(expt_dir, config.feature_fn)
    config.subwords_to_ids = os.path.join(expt_dir, config.subwords_to_ids)
    config.ckpt_dir = os.path.join(expt_dir, config.ckpt_dir)

    random.seed(config.global_step)
    np.random.seed(config.global_step)
    torch.manual_seed(config.global_step)

    with open(config.subwords_to_ids, 'r') as f:
        subwords_to_ids = json.load(f)

    # get ipa for showing results
    def get_xsampa_to_ipa(feature_fn):
        with open(feature_fn, 'r') as f:
            all_dict = json.load(f)

        phones = all_dict['phones']

        xsampa_to_ipa_dict = {}
        for xsampa, (ipa, this_dict) in phones.items():
            xsampa_to_ipa_dict[xsampa] = ipa

        xsampa_to_ipa_dict['"'] = 'ˈ'
        xsampa_to_ipa_dict['%'] = 'ˌ'
        xsampa_to_ipa_dict['#'] = ' '
        xsampa_to_ipa_dict['_1'] = '˥'
        xsampa_to_ipa_dict['_2'] = '˧˥'
        xsampa_to_ipa_dict['_3'] = '˧'
        xsampa_to_ipa_dict['_4'] = '˨˩'
        xsampa_to_ipa_dict['_5'] = '˩˧'
        xsampa_to_ipa_dict['_6'] = '˨'

        supps = all_dict['supps']
        for xsampa in supps:
            if not xsampa in xsampa_to_ipa_dict:
                xsampa_to_ipa_dict[xsampa] = xsampa

        return xsampa_to_ipa_dict


    xsampa_to_ipa = get_xsampa_to_ipa(config.feature_fn)

    # prepare evaluate dataset
    n_lang = len(config.dev_feats)
    datasets = []

    min_occ_count = config.dev_min_occ_count,
    min_seg_dur = config.dev_min_seg_dur,

    for lang_id in range(n_lang):
        this_set = data.DevDataset(feats=config.dev_feats[lang_id],
                                   align=config.dev_align[lang_id],
                                   vocab=config.dev_vocab[lang_id],
                                   subwords="phones",
                                   min_occ_count=config.dev_min_occ_count,
                                   min_seg_dur=config.dev_min_seg_dur,
                                   stack_frames=config.stack_frames,
                                   batch_size=config.dev_batch_size,
                                   subwords_to_ids=subwords_to_ids)
        datasets.append(this_set)

    net = net.MultiViewRNN(config=config,
                           feat_dim=datasets[0].feat_dim,
                           num_subwords=len(subwords_to_ids.keys()),
                           loss_fun=None,
                           use_gpu=True)
    # load net
    net.set_savepath(config.ckpt_dir, "net")
    net.load(tag='ft')
    net.eval()

    # get language scores and embeddings

    for lang_id in range(n_lang):

        this_lang = config.main_dev_language_list[lang_id]

        embs1, ids1 = [], []
        embs2, ids2 = [], []

        with torch.no_grad():

            for batch in datasets[lang_id].loader:
                ids = batch.pop("ids")
                inv = batch.pop("inv")
                out1, out2 = net.forward(batch, numpy=True)

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

        log.info(f"language = {this_lang},"
                 f"acoustic_ap = {acoustic_ap:.3f} ,"
                 f"crossview_ap = {crossview_ap:.3f}")

        with open(config.dev_vocab[lang_id], 'r') as f:
            this_vocab = json.load(f)

        words_to_ids = this_vocab["words_to_ids"]
        ids_to_words = {v:k for k,v in words_to_ids.items()}
        word_to_phones = this_vocab["word_to_phones"]

        def id_to_label(idx):
            word = ids_to_words[idx]
            xsampas = word_to_phones[ids_to_words[idx]][0]
            ipas = [xsampa_to_ipa[p] for p in xsampas]
            return f"{word} , /{''.join(ipas)}/ , {xsampas}"

        words1 = [id_to_label(idx) for idx in ids1]
        words2 = [id_to_label(idx) for idx in ids2]

        # write csv
        with open(f"{expt_dir}/{this_lang}-acoustic-labels.tsv", 'w') as f:
            for word in words1:
                f.write(f"{word}\n")

        with open(f"{expt_dir}/{this_lang}-acoustic-vectors.tsv", 'w') as f:
            for vec in embs1:
                for i in vec:
                    f.write(f"{i}\t")
                f.write('\n')

        with open(f"{expt_dir}/{this_lang}-word-labels.tsv", 'w') as f:
            for word in words2:
                f.write(f"{word}\n")

        with open(f"{expt_dir}/{this_lang}-word-vectors.tsv", 'w') as f:
            for vec in embs2:
                for i in vec:
                    f.write(f"{i}\t")
                f.write('\n')
