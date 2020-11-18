# Multilingual Acoustic Word Embeddings

This is the code base for multilingual acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) in [Multilingual Jointly Trained Acoustic and Written Word Embeddings](https://arxiv.org/pdf/2006.14007.pdf) in INTERSPEECH 2020.

```
@inproceedings{hu2020multilingual,
    title={Multilingual Jointly Trained Acoustic and Written Word Embeddings},
    author={Yushi Hu and Shane Settle and Karen Livescu},
    year={2020},
    booktitle=interspeech
}
```

The `multiview-babel-phone` folder contains the multiview model trained with phone sequence. 

The `multiview-babel-feature` folder contains the multiview model trained with distinctive features.

### Dependencies
python 3.7, pytorch 1.3, h5py, numpy, scipy, tensorboard

(not supported in python 3.8 so far, will fix later)

### Phone Set
The json file for all the X-SAMPA phones are in `subwords_to_ids.json`

The json file for all the X-SAMPA phones, corresponding IPA phones, and distinctive features are in `xsampa_phonetic_features.json`

### Data
The model is trained by Babel language packs (Cantonese, Assamese, Bengali, Pashto, Turkish, Tagalog, Tamil, Zulu, Lithuanian, Guarani, Igbo) and Switchboard dataset (English). To train from scratch, please contact the authors for all the data. A sample processed dataset `sample_dataset` is given in this repo. We used FilterBank features with pitch in the sample dataset. The code also works for other kinds of acoustic features.

### Word Embedding Examples
Visualization of word embeddings via Tensorflow projector:

[acoustic word embeddings visualization](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/Yushi-Hu/Multilingual-AWE/master/emb-examples/awe-projector-config.json)

[acoustically grounded word embeddings visualization](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/Yushi-Hu/Multilingual-AWE/master/emb-examples/projector-config.json)

[AWE and AGWE](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/Yushi-Hu/Multilingual-AWE/master/emb-examples/awe-agwe-config.json)

### Quick Start
Multiview model trained with X-SAMPA phone sequence:

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


