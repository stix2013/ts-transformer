import tensorflow as tf

from tf_transformer.transformer import Transformer
from tf_transformer.layer import MultiHeadAttention, PositionwiseFeedforward
from tf_transformer.block import TransformerBlock
from tf_transformer.encoder import Encoder, positional_encoding
from tf_transformer.decoder import Decoder


def test_positional_encoding():
    pos = 50
    d_model = 512
    pos_enc = positional_encoding(pos, d_model)
    assert pos_enc.shape == (1, pos, d_model)
    assert pos_enc.dtype == tf.float32


class TestMultiHeadAttention:
    def test_mha_shapes(self):
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        batch_size = 64
        seq_len = 10

        q = tf.random.uniform((batch_size, seq_len, d_model))
        k = tf.random.uniform((batch_size, seq_len, d_model))
        v = tf.random.uniform((batch_size, seq_len, d_model))

        output = mha(v, k, q, mask=None)
        assert output.shape == (batch_size, seq_len, d_model)


class TestPositionwiseFeedforward:
    def test_ffn_shapes(self):
        d_model = 512
        dff = 2048
        ffn = PositionwiseFeedforward(d_model, dff)

        batch_size = 64
        seq_len = 10

        x = tf.random.uniform((batch_size, seq_len, d_model))
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    def test_transformer_block_shapes(self):
        d_model = 512
        num_heads = 8
        dff = 2048
        transformer_block = TransformerBlock(d_model, num_heads, dff)

        batch_size = 64
        seq_len = 10

        x = tf.random.uniform((batch_size, seq_len, d_model))
        output = transformer_block(x, training=False, mask=None)
        assert output.shape == (batch_size, seq_len, d_model)


class TestEncoder:
    def test_encoder_shapes(self):
        num_layers = 2
        d_model = 512
        num_heads = 8
        dff = 2048
        input_vocab_size = 1000
        maximum_position_encoding = 100

        encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
        )

        batch_size = 64
        seq_len = 10

        x = tf.random.uniform(
            (batch_size, seq_len), dtype=tf.int32, minval=0, maxval=input_vocab_size
        )
        output = encoder(x, training=False, mask=None)
        assert output.shape == (batch_size, seq_len, d_model)


class TestDecoder:
    def test_decoder_shapes(self):
        num_layers = 2
        d_model = 512
        num_heads = 8
        dff = 2048
        target_vocab_size = 1200
        maximum_position_encoding = 100

        decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            maximum_position_encoding,
        )

        batch_size = 64
        seq_len = 10

        x = tf.random.uniform(
            (batch_size, seq_len), dtype=tf.int32, minval=0, maxval=target_vocab_size
        )
        enc_output = tf.random.uniform((batch_size, seq_len, d_model))

        output, _ = decoder(
            x, enc_output, training=False, look_ahead_mask=None, padding_mask=None
        )
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformer:
    def test_transformer_shapes(self):
        num_layers = 2
        d_model = 512
        num_heads = 8
        dff = 2048
        input_vocab_size = 1000
        target_vocab_size = 1200
        maximum_position_encoding = 100

        transformer = Transformer(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            target_vocab_size,
            maximum_position_encoding,
        )

        batch_size = 64
        seq_len = 10

        inp = tf.random.uniform(
            (batch_size, seq_len), dtype=tf.int32, minval=0, maxval=input_vocab_size
        )
        tar = tf.random.uniform(
            (batch_size, seq_len), dtype=tf.int32, minval=0, maxval=target_vocab_size
        )

        output = transformer((inp, tar))

        assert output.shape == (batch_size, seq_len, target_vocab_size)
