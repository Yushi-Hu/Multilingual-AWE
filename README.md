# Multilingual-AWE

The two directories correspond to two different architecture we used. the multiview-babel-share folder contains the multiview model trained with phone sequence. The multiview-babel-phonetic folder contains the multiview model trained with phonetic features.

## Phone Set
The json file for all the X-SAMPA phones are in multiview-babel-share/subwords_to_ids.json
The json file for all the X-SAMPA phones and their corresponding IPA phone and phonetic features are in multiview-babel-phonetic/xsampa_phonetic_features.json

## Training
In any one of the folder, run the following:
python train.py --config config.json

## Data
The data is processed from Babel language packs. Please ask in issues for processed data.
