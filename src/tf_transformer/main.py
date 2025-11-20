import tensorflow as tf
from transformer import Transformer

# Defining Custom Parameters.
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 8500
target_vocab_size = 8000
maximum_position_encoding = 10000
dropout_rate = 0.1

transformer = Transformer(
    num_layers,
    d_model,
    num_heads,
    dff,
    input_vocab_size,
    target_vocab_size,
    maximum_position_encoding,
    dropout_rate,
)


def main() -> None:
    print("This is the transformer with tensorflow module.")
    inputs = tf.random.uniform(
        (64, 50), dtype=tf.int64, minval=0, maxval=input_vocab_size
    )
    targets = tf.random.uniform(
        (64, 50), dtype=tf.int64, minval=0, maxval=target_vocab_size
    )

    look_ahead_mask = None
    padding_mask = None

    output = transformer(
        (inputs, targets),
        training=True,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )
    print(output.shape)


if __name__ == "__main__":
    main()