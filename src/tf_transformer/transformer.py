from keras import Model
from keras.layers import Dense
from .encoder import Encoder
from .decoder import Decoder


class Transformer(Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        maximum_position_encoding,
        dropout_rate=0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
            dropout_rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            maximum_position_encoding,
            dropout_rate,
        )
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        inp, tar = inputs
        enc_output = self.encoder(inp, training=training, mask=padding_mask)
        dec_output, _ = self.decoder(
            tar,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=padding_mask,
        )
        final_output = self.final_layer(dec_output)

        return final_output


if __name__ == "__main__":
    print("This is the transformer with tensorflow module.")
