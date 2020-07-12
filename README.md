# Multilingual Acoustic Word Embeddings

This is the code base for multilingual acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) in [Hu et al., 2020](https://arxiv.org/pdf/2006.14007.pdf).

The `multiview-babel-phone` folder contains the multiview model trained with phone sequence. 

The `multiview-babel-feature` folder contains the multiview model trained with distinctive features.

### Dependencies
python 3, pytorch 1.4, h5py, numpy, scipy, tensorboard

### Phone Set
The json file for all the X-SAMPA phones are in `subwords_to_ids.json`

The json file for all the X-SAMPA phones, corresponding IPA phones, and distinctive features are in `xsampa_phonetic_features.json`

### Data
The model is trained by Babel language packs (Cantonese, Assamese, Bengali, Pashto, Turkish, Tagalog, Tamil, Zulu, Lithuanian, Guarani, Igbo) and Switchboard dataset (English). To train from scratch, please contact the authors for all the data. A sample processed dataset `sample_dataset` is given in this repo.

### Quick Start
Multiview model trained with X-SAMPA phone sequence

All the checkpoints, output embeddings will be saved in `multiview-babel-phone/expt/sample/`
```
# evaluate the pre-trained model on sample dataset and get word embeddings

cd multiview-babel-phone/code
python evaluate.py --config ../expt/sample/config.json

# train the model with sample dataset
# fine tune on pre-trained model 
# (to train from scratch, change fine_tune in config.json to False)

cd multiview-babel-phone/code
python train.py --config ../expt/sample/config.json
```

Multiview model trained with distinctive features:

All the checkpoints, output embeddings will be saved in `multiview-babel-feature/expt/sample/`
```
# evaluate the pre-trained model on sample dataset and get word embeddings

cd multiview-babel-feature/code
python evaluate.py --config ../expt/sample/config.json

# train the model with sample dataset
# fine tune on pre-trained model 
# (to train from scratch, change fine_tune in config.json to False)

cd multiview-babel-feature/code
python train.py --config ../expt/sample/config.json
```
