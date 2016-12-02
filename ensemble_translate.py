'''
Translates a source file using a translation model.
'''
import argparse
import ipdb

import numpy
import theano
import cPickle as pkl
import copy

#import (build_sampler, gen_sample, load_params,
#                 init_params, init_tparams)
import session2.nmt
import doc2.nmt

from multiprocessing import Process, Queue

def gen_sample(tparams_list, f_init_list, f_next_list, type_list, x, xc, xc_mask, xc_mask_2, xc_mask_3, options_list, trng_list=None, k=1, maxlen=30):

    num_models = len(options_list)

    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')

    ret_list = [[]] * num_models

    next_state_list = [None] * num_models
    ctx0_list = [None] * num_models
    sc0_list = [None] * num_models
    next_memory_list = [None] * num_models

    for jj, options in enumerate(options_list):
        if type_list[jj] == 0:
            ins = [x]
        else:
            # get initial state of decoder rnn and encoder context
            if options['kwargs'].get('context_birnn', False) or options['kwargs'].get('sentence_dep_context', False):
                ins = [x, xc, xc_mask, xc_mask_2]
            else:
                ins = [x, xc, xc_mask]
            if not options['kwargs'].get('rnn_over_context', False):
                ins.append(xc_mask_3)

        ret_list[jj] = f_init_list[jj](*ins)

        if type_list[jj] == 0:
            if options['decoder'].startswith('lstm'):
                next_state_list[jj], ctx0_list[jj], next_memory_list[jj] = ret_list[jj][0], ret_list[jj][1], ret_list[jj][2]
            else:
                next_state_list[jj], ctx0_list[jj] = ret_list[jj][0], ret_list[jj][1]
        else:
            if options['decoder'].startswith('lstm'):
                next_state_list[jj], ctx0_list[jj], sc0_list[jj], next_memory_list[jj] = ret_list[jj][0], ret_list[jj][1], ret_list[jj][2], ret_list[jj][3]
            else:
                next_state_list[jj], ctx0_list[jj], sc0_list[jj] = ret_list[jj][0], ret_list[jj][1], ret_list[jj][2]

    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    xc_mask_2_0 = xc_mask_2[:]
    next_p_list = [[]] * num_models
    next_w_list = [[]] * num_models

    for ii in xrange(maxlen):
        for jj, options in enumerate(options_list):
            if type_list[jj] == 0:
                ctx = numpy.tile(ctx0_list[jj], [live_k, 1])
                if options['decoder'].startswith('lstm'):
                    inps = [next_w, ctx, next_state_list[jj], next_memory_list[jj]]
                else:
                    inps = [next_w, ctx, next_state_list[jj]]
            else:
                ctx = numpy.tile(ctx0_list[jj], [live_k, 1])
                sc = numpy.tile(sc0_list[jj], [live_k, 1])
                xc_mask_2 = numpy.tile(xc_mask_2_0, [live_k])
                if options['decoder'].startswith('lstm'):
                    if options['kwargs'].get('sentence_dep_context', False) or options['kwargs'].get('concat_context', False):
                        inps = [next_w, ctx, next_state_list[jj], next_memory_list[jj]]
                    else:
                        inps = [next_w, ctx, next_state_list[jj], sc, next_memory_list[jj], xc_mask_2]
                else:
                    if options['kwargs'].get('sentence_dep_context', False) or options['kwargs'].get('concat_context', False):
                        inps = [next_w, ctx, next_state_list[jj]]
                    else:
                        inps = [next_w, ctx, next_state_list[jj], sc, xc_mask_2]

            ret = f_next_list[jj](*inps)
            if options['decoder'].startswith('lstm'):
                next_p_list[jj], next_w_list[jj], next_state_list[jj], next_memory_list[jj] = ret[0], ret[1], ret[2], ret[3]
            else:
                next_p_list[jj], next_w_list[jj], next_state_list[jj] = ret[0], ret[1], ret[2]

        avg_log_probs =  numpy.mean(numpy.log(next_p_list), axis=0)
        cand_scores = hyp_scores[:, None] - avg_log_probs
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]

        voc_size = avg_log_probs.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')

        new_hyp_states_list = []
        new_hyp_memories_list = []
        for jj in xrange(num_models):
            new_hyp_states_list.append([])
            new_hyp_memories_list.append([])

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            for jj, options in enumerate(options_list):
                new_hyp_states_list[jj].append(copy.copy(next_state_list[jj][ti]))
                if options['decoder'].startswith('lstm'):
                    new_hyp_memories_list[jj].append(copy.copy(next_memory_list[jj][ti]))
        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states_list = []
        hyp_memories_list = []
        for jj in xrange(num_models):
            hyp_states_list.append([])
            hyp_memories_list.append([])

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                for jj, options in enumerate(options_list):
                    hyp_states_list[jj].append(new_hyp_states_list[jj][idx])
                    if options['decoder'].startswith('lstm'):
                        hyp_memories_list[jj].append(new_hyp_memories_list[jj][idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = numpy.array([w[-1] for w in hyp_samples])
        for jj, options in enumerate(options_list):
            next_state_list[jj] = numpy.array(hyp_states_list[jj])
            if options['decoder'].startswith('lstm'):
                next_memory_list[jj] = numpy.array(hyp_memories_list[jj])

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score
######

def translate_model(queue, rqueue, pid, model_list, options_list, type_list, k, normalize, maxlen):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng_list = []
    for ii, options in enumerate(options_list):
        trng_list.append(RandomStreams(options['trng']))
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    # load model parameters and set theano shared variables
    params_list = []
    tparams_list = []
    f_init_list = []
    f_next_list = []
    for ii, options in enumerate(options_list):
        if type_list[ii] == 0:
            params_ = session2.nmt.init_params(options)
            params_list.append(session2.nmt.load_params(model_list[ii], params_))
            tparams_list.append(session2.nmt.init_tparams(params_list[ii]))
            f_init, f_next = session2.nmt.build_sampler(tparams_list[ii], options, trng_list[ii], use_noise)
            f_init_list.append(f_init)
            f_next_list.append(f_next)
        else:
            params_ = doc2.nmt.init_params(options)
            params_list.append(doc2.nmt.load_params(model_list[ii], params_))
            tparams_list.append(doc2.nmt.init_tparams(params_list[ii]))
            f_init, f_next = doc2.nmt.build_sampler(tparams_list[ii], options, trng_list[ii], use_noise)
            f_init_list.append(f_init)
            f_next_list.append(f_next)

    def _translate(seq, seq_context, xc_mask, xc_mask_2, xc_mask_3):
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams_list, f_init_list, f_next_list, type_list,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   xc, xc_mask, xc_mask_2, xc_mask_3,
                                   options_list, trng_list=trng_list, k=k, maxlen=maxlen)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x, xc, xc_mask, xc_mask_2, xc_mask_3 = req[0], req[1], req[2], req[3], req[4], req[5]
        print pid, '-', idx
        seq = _translate(x, xc, xc_mask, xc_mask_2, xc_mask_3)

        rqueue.put((idx, seq))

    return


def main(model_list, type_list, dictionary, dictionary_target, source_file, source_context_file, saveto, k=5,
         normalize=False, n_process=5, maxlen=100):

    # load model model_options
    options_list = []
    for model in model_list:
        with open('%s.pkl' % model, 'rb') as f:
            options_list.append(pkl.load(f))

    #print options['kwargs']
    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model_list, options_list, type_list, k, normalize, maxlen))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname, gname):
        with open(fname, 'r') as f:
            with open(gname, 'r') as g:
                for idx, line in enumerate(f):
                    words = line.strip().split()
                    x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                    x = map(lambda ii: ii if ii < options_list[0]['n_words_src'] else 1, x)
                    x += [0]

                    gline = g.readline()
                    context_words = gline.strip().split(' ||| ') # with |||
                    
                    num_sent = len(context_words)
                    maxlen_ = max(len(sent.strip().split()) for sent in context_words) + 1
                    xc = numpy.zeros((num_sent, 1, maxlen_), dtype=numpy.int64)
                    xc_mask = numpy.zeros((num_sent, 1, maxlen_), dtype=numpy.float32)
                    xc_mask_2 = numpy.zeros((num_sent, 1), dtype=numpy.float32)
                    xc_mask_3 = numpy.zeros((num_sent, 1), dtype=numpy.float32)

                    for ii, sent in enumerate(context_words):
                        sent = sent.strip().split()
                        tmp = map(lambda w: word_dict[w] if w in word_dict else 1, sent)
                        tmp = map(lambda ii: ii if ii < options_list[0]['n_words_src'] else 1, tmp)
                        tmp += [0]
                        xc[ii,0,:len(tmp)] = tmp
                        xc_mask[ii,0,:len(tmp)] = 1.0
                        xc_mask_2[ii,0] = 1.0
                        xc_mask_3[ii,0] = float(len(tmp))
                    
                    queue.put((idx, x, xc, xc_mask, xc_mask_2, xc_mask_3))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans

    print 'Translating ', source_file, 'with context from', source_context_file, '...'
    n_samples = _send_jobs(source_file, source_context_file)
    trans = _seqs2words(_retrieve_jobs(n_samples))
    _finish_processes()
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('--models', type=str, nargs='+')
    parser.add_argument('--types', type=int, nargs='+')
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('source_context', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('maxlen', type=int, default=100)
    args = parser.parse_args()

    main(args.models, args.types, args.dictionary, args.dictionary_target, args.source, args.source_context,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p, maxlen=args.maxlen)
