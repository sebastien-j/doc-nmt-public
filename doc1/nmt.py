'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_legacy': ('param_init_gru_cond_legacy', 'gru_cond_legacy_layer'),
          'gru_simple_sc': ('param_init_gru_simple_sc', 'gru_simple_sc_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_simple_sc': ('param_init_lstm_simple_sc', 'lstm_simple_sc_layer'),
          'lstm_late_sc': ('param_init_lstm_late_sc', 'lstm_late_sc_layer'),
          'gru_cond_legacy_simple_sc':  ('param_init_gru_cond_legacy_simple_sc', 'gru_cond_legacy_simple_sc_layer'),
          'gru_cond_simple_sc':  ('param_init_gru_cond_simple_sc', 'gru_cond_simple_sc_layer'),
          'lstm_cond_legacy': ('param_init_lstm_cond_legacy', 'lstm_cond_legacy_layer'),
          'lstm_cond_legacy_simple_sc': ('param_init_lstm_cond_legacy_simple_sc', 'lstm_cond_legacy_simple_sc_layer'),
          'lstm_cond_legacy_late_sc': ('param_init_lstm_cond_legacy_late_sc', 'lstm_cond_legacy_late_sc_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim, rng=None):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_y, seqs_xc, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    lengths_xc = [len(s) for s in seqs_xc]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_seqs_xc = []
        new_lengths_x = []
        new_lengths_y = []
        new_lengths_xc = []
        for l_x, s_x, l_y, s_y, l_xc, s_xc in zip(lengths_x, seqs_x, lengths_y, seqs_y, lengths_xc, seqs_xc):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_seqs_xc.append(s_xc)
                new_lengths_xc.append(l_xc)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        lengths_xc = new_lengths_xc
        seqs_xc = new_seqs_xc

    if len(lengths_x) < 1 or len(lengths_y) < 1:
        return None, None, None, None, None, None, lengths_x, lengths_y, lengths_xc

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1
    maxlen_xc = numpy.max(lengths_xc) + 1 # Use EOS token (similar to bias, and makes sampling a bit easier to code)

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    xc = numpy.zeros((maxlen_xc, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    xc_mask = numpy.zeros((maxlen_xc, n_samples)).astype('float32')
    for idx, [s_x, s_y, s_xc] in enumerate(zip(seqs_x, seqs_y, seqs_xc)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
        xc[:lengths_xc[idx], idx] = s_xc
        xc_mask[:lengths_xc[idx]+1, idx] = 1. # EOS (bias)

    return x, x_mask, y, y_mask, xc, xc_mask, lengths_x, lengths_y, lengths_xc


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho, rng=rng)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                           norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              init_state=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    W = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                           norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim, rng=rng),
                              ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim,)).astype('float32')

    Ux_nl = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim, rng=rng)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru_cond',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond_legacy(options, params, prefix='gru_cond_legacy',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim, rng=rng)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_legacy_layer(tparams, state_below, options, prefix='gru_cond_legacy',
                   mask=None, context=None, one_step=False, init_state=None,
                   context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]
    # projected x into gru gates
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional gru layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        # hidden state proposal, leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0])],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

# GRU layer
def param_init_gru_simple_sc(options, params, prefix='gru_simple_sc', nin=None, dim=None, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                           norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux')] = Ux

    Wsc = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'Wsc')] = Wsc

    Wscx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wscx')] = Wscx

    return params


def gru_simple_sc_layer(tparams, state_below, options, prefix='gru_simple_sc', mask=None,
              init_state=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')]) # Will be broadcasted over # of steps
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscx')])

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    """
     Stack the weight matrices for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = numpy.concatenate([norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng)], axis=1)
    params[_p(prefix,'W')] = W
    # for the previous hidden activation
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')
    params[_p(prefix,'b')][dim:2*dim] = 1.0

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, init_state=None, init_memory=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    assert state_below.ndim == 3
    n_samples = state_below.shape[1]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        assert _x.ndim == 2
        return _x[:, n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    shared_vars = [tparams[_p(prefix, 'U')]]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory],
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, strict=True)
    return rval

# LSTM layer
def param_init_lstm_simple_sc(options, params, prefix='lstm_simple_sc', nin=None, dim=None, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    """
     Stack the weight matrices for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = numpy.concatenate([norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng)], axis=1)
    params[_p(prefix,'W')] = W
    # for the previous hidden activation
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')
    params[_p(prefix,'b')][dim:2*dim] = 1.0

    Wsc = numpy.concatenate([norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng)], axis=1)
    params[_p(prefix, 'Wsc')] = Wsc

    return params

# This function implements the lstm fprop
def lstm_simple_sc_layer(tparams, state_below, options, prefix='lstm_simple_sc', mask=None, init_state=None, init_memory=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    assert state_below.ndim == 3
    n_samples = state_below.shape[1]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        assert _x.ndim == 2
        return _x[:, n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')] + \
                  tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])

    shared_vars = [tparams[_p(prefix, 'U')]]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory],
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, strict=True)
    return rval


# LSTM layer
def param_init_lstm_late_sc(options, params, prefix='lstm_late_sc', nin=None, dim=None, rng=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    """
     Stack the weight matrices for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = numpy.concatenate([norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng),
                           norm_weight(nin,dim, rng=rng)], axis=1)
    params[_p(prefix,'W')] = W
    # for the previous hidden activation
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')
    params[_p(prefix,'b')][dim:2*dim] = 1.0

    Wsc = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wsc')] = Wsc

    Wr = norm_weight(dim, dim, rng=rng)
    params[_p(prefix, 'Wr')] = Wr
    Wrr = norm_weight(dim, dim, rng=rng)
    params[_p(prefix, 'Wrr')] = Wrr
    params[_p(prefix,'br')] = numpy.zeros((dim,)).astype('float32')


    return params

# This function implements the lstm fprop
def lstm_late_sc_layer(tparams, state_below, options, prefix='lstm_late_sc', mask=None, init_state=None, init_memory=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    assert state_below.ndim == 3
    n_samples = state_below.shape[1]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        assert _x.ndim == 2
        return _x[:, n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_, tsc, U, Wr, Wrr, br):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        r = tensor.nnet.sigmoid(tensor.dot(tsc, Wr) + tensor.dot(c, Wrr) + br)

        h = o * tensor.tanh(c + r * tsc)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    tsc = tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])

    shared_vars = [tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Wr')], tparams[_p(prefix, 'Wrr')], tparams[_p(prefix, 'br')]]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory],
                                non_sequences=[tsc]+shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, strict=True)
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond_legacy(options, params, prefix='gru_cond_legacy',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim, rng=rng)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_legacy_layer(tparams, state_below, options, prefix='gru_cond_legacy',
                   mask=None, context=None, one_step=False, init_state=None,
                   context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]
    # projected x into gru gates
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional gru layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        # hidden state proposal, leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0])],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

########
########


# Conditional GRU layer with Attention
def param_init_gru_cond_legacy_simple_sc(options, params, prefix='gru_cond_legacy_simple_sc',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim, rng=rng)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    Wsc = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'Wsc')] = Wsc

    Wscx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wscx')] = Wscx

    Wscc = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wscc')] = Wscc

    return params


def gru_cond_legacy_simple_sc_layer(tparams, state_below, options, prefix='gru_cond_legacy_simple_sc',
                   mask=None, context=None, one_step=False, init_state=None,
                   context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscx')])
    # projected x into gru gates
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')]) + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscc')]) # Should a bias be added here?

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional gru layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        # hidden state proposal, leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0])],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

# Conditional GRU layer with Attention
def param_init_gru_cond_simple_sc(options, params, prefix='gru_cond_simple_sc',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    W = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                           norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim, rng=rng),
                           ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim, rng=rng),
                              ortho_weight(dim, rng=rng)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim,)).astype('float32')

    Ux_nl = ortho_weight(dim, rng=rng)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim, rng=rng)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    Wsc = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'Wsc')] = Wsc

    Wscx = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wscx')] = Wscx

    return params


def gru_cond_simple_sc_layer(tparams, state_below, options, prefix='gru_cond_simple_sc',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscx')])
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond_legacy(options, params, prefix='lstm_cond_legacy',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_lstm(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*4, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def lstm_cond_legacy_layer(tparams, state_below, options, prefix='lstm_cond_legacy',
                   mask=None, context=None, one_step=False, init_state=None, init_memory=None,
                   context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'
        assert init_memory, 'previous cell must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix,'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into lstm
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xc_, h_, ctx_, alpha_, c_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional lstm layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T, c

    seqs = [mask, state_below_, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, init_memory, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0]),
                          init_memory],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond_legacy_simple_sc(options, params, prefix='lstm_cond_legacy_simple_sc',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_lstm(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*4, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    Wsc = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng),
                             norm_weight(nin, dim, rng=rng)], axis=1)
    params[_p(prefix, 'Wsc')] = Wsc

    Wscc = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wscc')] = Wscc

    return params


def lstm_cond_legacy_simple_sc_layer(tparams, state_below, options, prefix='lstm_cond_legacy_simple_sc',
                   mask=None, context=None, one_step=False, init_state=None, init_memory=None,
                   context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'
        assert init_memory, 'previous cell must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix,'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into lstm
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')] + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')]) + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscc')]) # Should a bias be added here?

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xc_, h_, ctx_, alpha_, c_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional lstm layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T, c

    seqs = [mask, state_below_, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, init_memory, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0]),
                          init_memory],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond_legacy_late_sc(options, params, prefix='lstm_cond_legacy_late_sc',
                        nin=None, dim=None, dimctx=None, rng=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_lstm(options, params, prefix, nin=nin, dim=dim, rng=rng)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*4, rng=rng)
    params[_p(prefix, 'Wc')] = Wc

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, rng=rng)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx, rng=rng)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1, rng=rng)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    Wscc = norm_weight(nin, dimctx, rng=rng)
    params[_p(prefix, 'Wscc')] = Wscc

    Wsc = norm_weight(nin, dim, rng=rng)
    params[_p(prefix, 'Wsc')] = Wsc

    Wr = norm_weight(dim, dim, rng=rng)
    params[_p(prefix, 'Wr')] = Wr
    Wrr = norm_weight(dim, dim, rng=rng)
    params[_p(prefix, 'Wrr')] = Wrr
    params[_p(prefix,'br')] = numpy.zeros((dim,)).astype('float32')

    return params


def lstm_cond_legacy_late_sc_layer(tparams, state_below, options, prefix='lstm_cond_legacy_late_sc',
                   mask=None, context=None, one_step=False, init_state=None, init_memory=None,
                   context_mask=None, **kwargs):

    assert 'sc' in kwargs # sc: n_samples x dim_sc

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'
        assert init_memory, 'previous cell must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix,'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into lstm
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')]) + \
        tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wscc')]) # Should a bias be added here?

    tsc = tensor.dot(kwargs['sc'], tparams[_p(prefix, 'Wsc')])

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xc_, h_, ctx_, alpha_, c_, pctx_, cc_, tsc,
                    U, Wc, Wd_att, U_att, c_tt, Wr, Wrr, br):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional lstm layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        r = tensor.nnet.sigmoid(tensor.dot(tsc, Wr) + tensor.dot(c, Wrr) + br)

        h = o * tensor.tanh(c + r * tsc)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T, c

    seqs = [mask, state_below_, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Wr')],
                   tparams[_p(prefix, 'Wrr')],
                   tparams[_p(prefix, 'br')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, init_memory, pctx_, context, tsc]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0]),
                          init_memory],
            non_sequences=[pctx_,
                           context, tsc]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    rng = numpy.random.RandomState(options['rng'])

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'], rng=rng)
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'], rng=rng)

    params = get_layer('ff')[0](options, params, prefix='f_context_emb',
                                nin=options['dim_word'], nout=options['dim'],
                                ortho=False, rng=rng)

    params = get_layer('ff')[0](options, params, prefix='r_context_emb',
                                nin=options['dim_word'], nout=options['dim'],
                                ortho=False, rng=rng)

    if options['encoder'].startswith('lstm'):
        params = get_layer('ff')[0](options, params, prefix='f_init_memory',
                                    nin=options['dim_word'], nout=options['dim'],
                                    ortho=False, rng=rng)

        params = get_layer('ff')[0](options, params, prefix='r_init_memory',
                                    nin=options['dim_word'], nout=options['dim'],
                                    ortho=False, rng=rng)
    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'], rng=rng)
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'], rng=rng)
    ctxdim = 2 * options['dim']

    # init_state, init_memory
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'], rng=rng)

    if options['decoder'].startswith('lstm'):
        params = get_layer('ff')[0](options, params, prefix='ff_cell',
                                    nin=ctxdim, nout=options['dim'], rng=rng)    
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim, rng=rng)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False, rng=rng)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False, rng=rng)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False, rng=rng)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_sc',
                                nin=options['dim_word'], nout=options['dim_word'],
                                ortho=False, rng=rng)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'], rng=rng)

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(options['trng'])
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    xc = tensor.matrix('xc', dtype='int64')
    xc_mask = tensor.matrix('xc_mask', dtype='float32')    

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_timesteps_context = xc.shape[0]
    n_samples = x.shape[1]

    context_emb = tparams['Wemb'][xc.flatten()]
    context_emb = context_emb.reshape([n_timesteps_context, n_samples, options['dim_word']])

    # tensor.clip is not strictly necessary anymore. (EOS always there)
    context_emb = (context_emb * xc_mask[:,:,None]).sum(0) / tensor.clip(xc_mask.sum(0), 1.0, numpy.inf)[:, None] # n_samples x options['dim_word'], n_samples x 1
    f_context_emb = get_layer('ff')[1](tparams, context_emb, options,
                                    prefix='f_context_emb', activ='tanh')
    r_context_emb = get_layer('ff')[1](tparams, context_emb, options,
                                    prefix='r_context_emb', activ='tanh')

    f_init_memory = None
    r_init_memory = None
    if options['encoder'].startswith('lstm'):
        f_init_memory = get_layer('ff')[1](tparams, context_emb, options,
                        prefix='f_init_memory', activ='tanh')
        r_init_memory = get_layer('ff')[1](tparams, context_emb, options,
                prefix='r_init_memory', activ='tanh')

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            init_state=f_context_emb,
                                            init_memory=f_init_memory,
                                            sc=context_emb)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             init_state=r_context_emb,
                                             init_memory=r_init_memory,
                                             sc=context_emb)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')
    init_memory = None
    if options['decoder'].startswith('lstm'):
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options,
                                        prefix='ff_cell', activ='tanh')
    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            init_memory=init_memory,
                                            sc=context_emb)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit_sc = get_layer('ff')[1](tparams, context_emb, options,
                                   prefix='ff_logit_sc', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx+logit_sc)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, 
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, xc, xc_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    xc = tensor.matrix('xc', dtype='int64')

    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    n_timesteps_context = xc.shape[0]

    context_emb = tparams['Wemb'][xc.flatten()]
    context_emb = context_emb.reshape([n_timesteps_context, n_samples, options['dim_word']])

    context_emb = context_emb.mean(0) # Will probably crash here for no context # FIXME
    f_context_emb = get_layer('ff')[1](tparams, context_emb, options,
                                    prefix='f_context_emb', activ='tanh')
    r_context_emb = get_layer('ff')[1](tparams, context_emb, options,
                                    prefix='r_context_emb', activ='tanh')

    f_init_memory = None
    r_init_memory = None
    if options['encoder'].startswith('lstm'):
        f_init_memory = get_layer('ff')[1](tparams, context_emb, options,
                        prefix='f_init_memory', activ='tanh')
        r_init_memory = get_layer('ff')[1](tparams, context_emb, options,
                        prefix='r_init_memory', activ='tanh')

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder', init_state=f_context_emb,
                                            init_memory=f_init_memory, sc=context_emb)
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r', init_state=r_context_emb,
                                             init_memory=r_init_memory, sc=context_emb)

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')
    if options['decoder'].startswith('lstm'):
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options,
                                        prefix='ff_cell', activ='tanh')
    print 'Building f_init...',
    if options['decoder'].startswith('lstm'):
        outs = [init_state, ctx, context_emb, init_memory]
    else:
        outs = [init_state, ctx, context_emb]
    f_init = theano.function([x, xc], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    if not options['decoder'].startswith('lstm'):
        init_memory = None
    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            init_memory=init_memory,
                                            sc=context_emb)
    # get the next hidden state
    next_state = proj[0]
    if options['decoder'].startswith('lstm'):
        next_memory = proj[3]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    print logit_ctx.ndim
    logit_sc = get_layer('ff')[1](tparams, context_emb, options,
                                   prefix='ff_logit_sc', activ='linear')
    print logit_sc.ndim
    #ipdb.set_trace()
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx+logit_sc)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    if options['decoder'].startswith('lstm'):
        inps = [y, ctx, init_state, context_emb, init_memory]
        outs = [next_probs, next_sample, next_state, next_memory]
    else:
        inps = [y, ctx, init_state, context_emb]
        outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, xc, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if options['decoder'].startswith('lstm'):
        hyp_memories = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x, xc)
    #print 'A', x.shape, xc.shape
    if options['decoder'].startswith('lstm'):
        next_state, ctx0, sc0, next_memory = ret[0], ret[1], ret[2], ret[3]
    else:
        next_state, ctx0, sc0 = ret[0], ret[1], ret[2]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    #print 'B', next_w.shape, next_state.shape, ctx0.shape, sc0.shape, next_memory.shape

    for ii in xrange(maxlen):
        #print 'C', ctx0.shape, sc0.shape
        ctx = numpy.tile(ctx0, [live_k, 1])
        sc = numpy.tile(sc0, [live_k, 1])
        if options['decoder'].startswith('lstm'):
            inps = [next_w, ctx, next_state, sc, next_memory]
        else:
            inps = [next_w, ctx, next_state, sc]
        #print 'D', next_w.shape, ctx.shape, next_state.shape, sc.shape, next_memory.shape
        ret = f_next(*inps)
        if options['decoder'].startswith('lstm'):
            next_p, next_w, next_state, next_memory = ret[0], ret[1], ret[2], ret[3]
        else:
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            if options['decoder'].startswith('lstm'):
                new_hyp_memories = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                if options['decoder'].startswith('lstm'):
                    new_hyp_memories.append(copy.copy(next_memory[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            if options['decoder'].startswith('lstm'):
                hyp_memories = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if options['decoder'].startswith('lstm'):
                        hyp_memories.append(new_hyp_memories[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            if options['decoder'].startswith('lstm'):
                next_memory = numpy.array(hyp_memories)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y, xc in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask, xc, xc_mask, lengths_x, lengths_y, lengths_xc = prepare_data(x, y, xc,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask, xc, xc_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(rng=123,
          trng=1234,
          dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          reload_=False,
          save_inter=False,
          **kwargs):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1], datasets[2],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, xc, xc_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, xc, xc_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y, xc in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask, xc, xc_mask, lengths_x, lengths_y, lengths_xc = prepare_data(x, y, xc, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask, xc, xc_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))

                cur_saveto = saveto[:-3]+str(uidx)+".npz"
                print cur_saveto,
                cur_params = unzip(tparams)
                numpy.savez(cur_saveto, history_errs=history_errs, **cur_params)
                pkl.dump(model_options, open('%s.pkl' % cur_saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:lengths_x[jj]+1, jj][:, None], xc[:lengths_xc[jj]+1, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
