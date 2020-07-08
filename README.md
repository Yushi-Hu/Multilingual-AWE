# Multilingual-AWE

This recipe trains multilingual acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) described in [Hu et al., 2020](https://arxiv.org/pdf/2006.14007.pdf).

The training objective is based on the multiview triplet loss functions
of [He et al., 2016](https://arxiv.org/pdf/1611.04496.pdf).

The two directories correspond to two different architecture we used. 

The `multiview-babel-share` folder contains the multiview model trained with phone sequence. 

The `multiview-babel-phonetic` folder contains the multiview model trained with phonetic features.

### Dependencies
python 3, pytorch 1.4, h5py, numpy, scipy

### Phone Set
The json file for all the X-SAMPA phones are in `multiview-babel-share/subwords_to_ids.json`

The json file for all the X-SAMPA phones and their corresponding IPA phone and distinctive features are in `multiview-babel-phonetic/xsampa_phonetic_features.json`

### Training
In any one of the folder, run the following:
```
python train.py --config config.json
```

### Data
The model is trained by Babel language packs. To train from scratch, please email the author for pre-processed data.
