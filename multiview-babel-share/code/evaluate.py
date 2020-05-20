import os
import json
import argparse
import random
import numpy as np
import torch
import logging as log
from collections import namedtuple
import net
import metric
import data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", help="configuration filename")
    parser.add_argument("--mode", help="single, all, unseen")
    parser.add_argument("--dir", help="directory")
    args = parser.parse_args()

    idx = int(args.idx)

    # single
    dev_idx_list = [idx]
    test_idx_list = [idx]
    ph_idx = [idx]
    ph_train_idx = [idx]

    if args.mode == "all":
        dev_idx_list = list(range(12))
        test_idx_list = list(range(12))
        ph_idx = list(range(12))
        ph_train_idx = list(range(12))

    if args.mode == "unseen":
        dev_idx_list = [idx]
        test_idx_list = [idx]
        ph_idx = [idx]
        ph_train_idx = list(range(0, idx)) + list(range(idx+1, 12))

    dev_dir = '/share/data/speech/Data/yushihu/multiview-babel/dataset'
    test_dir = '/share/data/speech/Data/yushihu/multiview-babel/testset'

    lang_list = ['101-cantonese', '102-assamese', '103-bengali', '104-pashto', '105-turkish',
                 '106-tagalog', '204-tamil', '206-zulu', '304-lithuanian', '305-guarani', '306-igbo', '401-english']

    my_config = {
        'feats': [f"{dev_dir}/{lang_list[idx]}/fbank_pitch_feats_with_cmvn.dev.hdf5" for idx in dev_idx_list] + \
                 [f"{test_dir}/{lang_list[idx]}/fbank_pitch_feats_with_cmvn.test.hdf5" for idx in test_idx_list],
        'aligns': [f"{dev_dir}/{lang_list[idx]}/align.dev.hdf5" for idx in dev_idx_list] + \
                  [f"{test_dir}/{lang_list[idx]}/align.test.hdf5" for idx in test_idx_list],
        'vocabs': [f"{dev_dir}/{lang_list[idx]}/vocab.json" for idx in dev_idx_list] + \
                  [f"{dev_dir}/{lang_list[idx]}/vocab.json" for idx in test_idx_list],
        'langs': [lang_list[idx] for idx in dev_idx_list + test_idx_list],
        'subwords_to_ids': '/share/data/speech/Data/yushihu/multiview-babel-share/subwords_to_ids.json',
        'feature_fn': '/share/data/speech/Data/yushihu/multiview-babel-phonetic/xsampa_phonetic_features.json'
    }

    my_config = namedtuple("Config", my_config.keys())(*my_config.values())

    config_file = f"{args.dir}/train_config.json"
    with open(config_file, "r") as f:
        config = argparse.Namespace(**json.load(f))

    with open(my_config.subwords_to_ids, 'r') as f:
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


    xsampa_to_ipa = get_xsampa_to_ipa(my_config.feature_fn)

    # for logging
    log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")

    n_lang = len(my_config.langs)
    datasets= []


    for lang_id in range(n_lang):
        this_set = data.DevDataset(feats=my_config.feats[lang_id],
                                   align=my_config.aligns[lang_id],
                                   vocab=my_config.vocabs[lang_id],
                                   subwords="phones",
                                   min_occ_count=0,
                                   min_seg_dur=50,
                                   stack_frames=True,
                                   batch_size=500,
                                   subwords_to_ids = subwords_to_ids)
        datasets.append(this_set)

    net = net.MultiViewRNN(config=config,
                           feat_dim=datasets[0].feat_dim,
                           num_subwords=len(subwords_to_ids.keys()),
                           loss_fun=None,
                           use_gpu=True)

    net.set_savepath(f"{args.dir}/save", "net")
    net.load(tag='best')
    net.eval()

    embed = net["view2"].emb.weight.detach().cpu().cuda()[1:, :]

    # get language scores and embeddings

    for lang_id in range(n_lang):

        this_lang = lang_list[(dev_idx_list + test_idx_list)[lang_id]]

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

        data_type = 'dev'
        if lang_id >= len(dev_idx_list):
            data_type = 'test'

        log.info(f"language = {this_lang}-{data_type} ,"
                 f"acoustic_ap = {acoustic_ap:.3f} ,"
                 f"crossview_ap = {crossview_ap:.3f}")

        with open(my_config.vocabs[lang_id], 'r') as f:
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

        # record the embeddings of dev set
        if lang_id < len(dev_idx_list):
            data_type = "dev"

            # write csv
            with open(f"{args.dir}/{this_lang}-{data_type}-acoustic-labels.tsv", 'w') as f:
                for word in words1:
                    f.write(f"{word}\n")

            with open(f"{args.dir}/{this_lang}-{data_type}-acoustic-vectors.tsv", 'w') as f:
                for vec in embs1:
                    for i in vec:
                        f.write(f"{i}\t")
                    f.write('\n')

            with open(f"{args.dir}/{this_lang}-{data_type}-phone-labels.tsv", 'w') as f:
                for word in words2:
                    f.write(f"{word}\n")

            with open(f"{args.dir}/{this_lang}-{data_type}-phone-vectors.tsv", 'w') as f:
                for vec in embs2:
                    for i in vec:
                        f.write(f"{i}\t")
                    f.write('\n')

    # get phoneme embeddings
    train_vocabs = [f"{dev_dir}/{lang_list[idx]}/vocab.json" for idx in ph_train_idx]
    train_subwords = set(data.combine_subwords_to_ids(train_vocabs, 'phones').keys())

    this_vocabs = [f"{dev_dir}/{lang_list[idx]}/vocab.json" for idx in ph_idx]
    this_subwords = set(data.combine_subwords_to_ids(this_vocabs, 'phones').keys())

    labels = []
    vectors = []
    for subword in this_subwords:
        if subword in ['<noise>', '<hes>', '<v-noise>']:
            continue

        with torch.no_grad():
            emb = embed[subwords_to_ids[subword]]

        if subword in train_subwords:
            labels.append(f"{subword} , {xsampa_to_ipa[subword]}")
        else:
            labels.append(f"{subword} , {xsampa_to_ipa[subword]}(unseen)")

        vectors.append(list(emb))

    # write csv
    with open(f"{args.dir}/phoneme-pf-labels.tsv",'w') as f:
        for subword in labels:
            f.write(f"{subword}\n")

    with open(f"{args.dir}/phoneme-pf-vectors.tsv",'w') as f:
        for vec in vectors:
            for i in vec:
                f.write(f"{i}\t")
            f.write('\n')




