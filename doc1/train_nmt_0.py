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
                     maxlen=400,
                     batch_size=96,
                     valid_batch_size=96,
                     validFreq=1000,
                     dispFreq=100,
                     saveFreq=2000,
                     sampleFreq=500,
                     datasets=['/misc/kcgscratch1/WIT3/en-fr/debug/train.tags.en-fr.en.tok.text.bpe10000.shuf', 
                               '/misc/kcgscratch1/WIT3/en-fr/debug/train.tags.en-fr.fr.tok.text.bpe10000.shuf',
                               '/misc/kcgscratch1/WIT3/en-fr/debug/train.tags.en-fr.en.tok.context.bpe10000.shuf'],
                     valid_datasets=['/misc/kcgscratch1/WIT3/en-fr/debug/IWSLT15.TED.dev2010.en-fr.en.out.xml.tok.text.bpe10000', 
                               '/misc/kcgscratch1/WIT3/en-fr/debug/IWSLT15.TED.dev2010.en-fr.fr.out.xml.tok.text.bpe10000',
                               '/misc/kcgscratch1/WIT3/en-fr/debug/IWSLT15.TED.dev2010.en-fr.en.out.xml.tok.context.bpe10000'],
                     dictionaries=['/misc/kcgscratch1/WIT3/en-fr/debug/train.tags.en-fr.en.tok.text.bpe10000.pkl', 
                                   '/misc/kcgscratch1/WIT3/en-fr/debug/train.tags.en-fr.fr.tok.text.bpe10000.pkl'],
                     use_dropout=params['use-dropout'][0],
                     rng=1234,
                     trng=1234,
                     save_inter=True,
                     encoder='gru',
                     decoder='gru_cond_legacy')
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/misc/kcgscratch1/WIT3/en-fr/models/debug_model.0.npz'],
        'dim_word': [500],
        'dim': [10000],
        'n-words': [10234], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
