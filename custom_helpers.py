import tensorflow as tf
import texar as tx
## MODIFIED TO ALLOW CLASS EMBEDDING APPENDING
class ContextSoftmaxEmbeddingHelper(tf.contrib.seq2seq.Helper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).
    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.
    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, context, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
        if isinstance(embedding, tx.modules.EmbedderBase):
            embedding = embedding.embedding

        if callable(embedding):
            raise ValueError("`embedding` must be an embedding tensor or an "
                             "instance of subclass of `EmbedderBase`.")
        else:
            self._embedding = embedding
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        self.context = context
        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._start_inputs = tf.concat([self._start_inputs, self.context], axis=-1)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_dist = tf.nn.softmax(outputs / self._tau)
        sampler = tf.distributions.Categorical(logits=sample_dist)
        sample_ids = sampler.sample()
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = self._embedding_fn(sample_ids)
        ## Modified
        next_inputs = tf.concat([next_inputs, self.context], axis=-1)
        return (finished, next_inputs, state)


class Helper(object):
  """Interface for implementing sampling in seq2seq decoders.
  Helper instances are used by `BasicDecoder`.
  """

  def batch_size(self):
    """Batch size of tensor returned by `sample`.
    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  def sample_ids_shape(self):
    """Shape of tensor returned by `sample`, excluding the batch dimension.
    Returns a `TensorShape`.
    """
    raise NotImplementedError("sample_ids_shape has not been implemented")

  def sample_ids_dtype(self):
    """DType of tensor returned by `sample`.
    Returns a DType.
    """
    raise NotImplementedError("sample_ids_dtype has not been implemented")

  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass

class ContextGreedyEmbeddingHelper(Helper):
  """A helper for use during inference.
  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    if isinstance(embedding, tx.modules.EmbedderBase):
        embedding = embedding.embedding

    if callable(embedding):
        raise ValueError("`embedding` must be an embedding tensor or an "
                         "instance of subclass of `EmbedderBase`.")
    else:
        self._embedding = embedding
        self._embedding_fn = (
            lambda ids: tf.nn.embedding_lookup(embedding, ids))
    self.context = context
    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = tf.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = self._embedding_fn(self._start_tokens)
    self._start_inputs = tf.concat([self._start_inputs, self.context], axis = -1)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    finished = tf.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    sample_ids = tf.argmax(outputs, axis=-1, output_type=dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""
    del time, outputs  # unused by next_inputs_fn
    finished = tf.equal(sample_ids, self._end_token)
    all_finished = tf.reduce_all(finished)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: tf.concat([self._embedding_fn(sample_ids), self.context], axis=-1))
    return (finished, next_inputs, state)


class ContextSampleEmbeddingHelper(ContextGreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token, 
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(ContextSampleEmbeddingHelper, self).__init__(
        embedding, context, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.context = context

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature
    
    sample_id_sampler = tf.distributions.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    #p = tf.print(sample_ids)
    p2 = tf.print(sample_ids.shape)
    return sample_ids
