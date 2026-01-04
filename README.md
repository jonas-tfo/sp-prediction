## t5_lstm_cnn_2

- only one cnn layer for simplicity
- lstm layer after cnn layer
- using t5 tokenizer and encoder and embedding

## t5_lstm_cnn_3

- added two more cnn layers after the first cnn layer for more complex feature extraction that are concatenated on the channel dimension
- lstm layer after the three cnn layers (reduced layers to 2)
- using gelu activation function to attempt better non-linearity when working with transformers compared to relu
- moved dropout before the classifier at the end
- increased patience to stop early stopping

## t5_lstm_cnn_4

-
