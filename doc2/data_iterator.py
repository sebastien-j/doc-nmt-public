import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, source_context,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 shuffle=True):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.source_context = fopen(source_context, 'r')

        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.source_context_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False
        self.shuffle = shuffle

        assert '|||' not in source_dict
        self.source_dict['|||'] = 0

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.source_context.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        source_context = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer) == len(self.source_context_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                cc = self.source_context.readline()
                if cc == "":
                    break

                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
                self.source_context_buffer.append(cc.strip().split())

            # sort by target buffer
            if self.shuffle:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
            else:
                tidx = range(len(self.target_buffer))[::-1]

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]
            _cbuf = [self.source_context_buffer[i] for i in tidx]
            
            self.source_buffer = _sbuf
            self.target_buffer = _tbuf
            self.source_context_buffer = _cbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.source_context_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                # read from source context file and map to word index
                cc_ = self.source_context_buffer.pop()
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

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
                source_context.append(cc)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, source_context
