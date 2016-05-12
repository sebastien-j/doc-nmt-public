import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, target, context,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 shuffle=True,
                 tc=False):
        self.target = fopen(target, 'r')
        self.context = fopen(context, 'r')

        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.target_buffer = []
        self.context_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False
        self.shuffle = shuffle
        self.tc = tc

        if not tc:
            assert '|||' not in source_dict
            self.source_dict['|||'] = 0
        else:
            assert '|||' not in target_dict
            self.target_dict['|||'] = 0

    def __iter__(self):
        return self

    def reset(self):
        self.target.seek(0)
        self.context.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        target = []
        context = []

        # fill buffer, if it's empty
        len(self.target_buffer) == len(self.context_buffer), 'Buffer size mismatch!'

        if len(self.context_buffer) == 0:
            for k_ in xrange(self.k):
                tt = self.target.readline()
                if tt == "":
                    break
                cc = self.context.readline()
                if cc == "":
                    break

                self.target_buffer.append(tt.strip().split())
                self.context_buffer.append(cc.strip().split())

            # sort by target buffer
            if self.shuffle:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
            else:
                tidx = range(len(self.target_buffer))[::-1]

            _tbuf = [self.target_buffer[i] for i in tidx]
            _cbuf = [self.context_buffer[i] for i in tidx]
            
            self.target_buffer = _tbuf
            self.context_buffer = _cbuf

        if len(self.target_buffer) == 0 or len(self.context_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    tt = self.target_buffer.pop()
                except IndexError:
                    break
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                # read from source context file and map to word index
                if not self.tc:
                    cc_ = self.context_buffer.pop()
                    cc_ = [self.source_dict[w] if w in self.source_dict else 1
                          for w in cc_]
                    if self.n_words_source > 0:
                        cc_ = [w if w < self.n_words_source else 1 for w in cc_]
                    cc = []
                    tmp = []
                    for word_id in cc_:
                        if word_id != 0:
                            tmp.append(word_id)
                        else:
                            cc.append(tmp)
                            tmp = []
                    cc.append(tmp)
                else:
                    cc_ = self.context_buffer.pop()
                    cc_ = [self.target_dict[w] if w in self.target_dict else 1
                          for w in cc_]
                    if self.n_words_target > 0:
                        cc_ = [w if w < self.n_words_target else 1 for w in cc_]
                    cc = []
                    tmp = []
                    for word_id in cc_:
                        if word_id != 0:
                            tmp.append(word_id)
                        else:
                            cc.append(tmp)
                            tmp = []
                    cc.append(tmp)

                if len(tt) > self.maxlen:
                    continue

                target.append(tt)
                context.append(cc)

                if len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return target, context
