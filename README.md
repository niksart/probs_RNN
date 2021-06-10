# probs_RNN
Code for extracting a probability distribution ...

Code adapted from: https://github.com/facebookresearch/colorlessgreenRNNs/
Training of the model was not done by me. I adapted and changed this file https://github.com/facebookresearch/colorlessgreenRNNs/blob/master/src/language_models/evaluate_target_word.py
Link to the license: https://creativecommons.org/licenses/by-nc/4.0/

1- prepare environment: python3.7 + pytorch 1.3 + pandas
2- download the model: https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
3- download the training data based on wikipedia on this page https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/data for English and place the 4 files inside a folder (e.g. "English")
4- execute the script: `python find_probs_each_word.py --data English/ --checkpoint model.pt --corpus sample.text --output sample.probs`
