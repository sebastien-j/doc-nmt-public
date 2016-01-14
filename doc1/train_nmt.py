import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=1000,
                     maxlen=1000,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
                     datasets=['/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok.text.bpe10000', 
                               '/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.fr.tok.text.bpe10000',
                               '/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok.text.bpe10000'],
                     valid_datasets=['/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok.text.bpe10000.first1000', 
                               '/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.fr.tok.text.bpe10000.first1000',
                               '/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok.text.bpe10000.first1000'],
                     dictionaries=['/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.en.tok.text.bpe10000.pkl', 
                                   '/misc/kcgscratch1/WIT3/en-fr/train.tags.en-fr.fr.tok.text.bpe10000.pkl'],
                     use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/misc/kcgscratch1/WIT3/en-fr/models/debug_model.npz'],
        'dim_word': [167],
        'dim': [243],
        'n-words': [10000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
