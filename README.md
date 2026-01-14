# Signal Peptide Prediction

This project focuses on **Signal Peptide Prediction** using various deep learning architectures and protein sequence encoding methods. The goal is to accurately identify and classify signal peptides in protein sequences, exploring both binary (2-state) and multi-class (6-state) classification tasks.

### t5_lstm_cnn_2

- **Architecture**: Simple hybrid model.
- Only one CNN layer for simplicity.
- LSTM layer follows the CNN layer.
- Uses T5 tokenizer, encoder, and embeddings.

### t5_lstm_cnn_3

- **Changes**: Enhanced feature extraction.
- Added two more CNN layers after the first CNN layer; these are concatenated on the channel dimension.
- LSTM layer processes the output of the three CNN layers (number of LSTM layers reduced to 2).
- Switched to **GELU** activation function to attempt better non-linearity with transformers (compared to ReLU).
- Moved dropout to be before the final classifier.
- Increased patience for early stopping.
- _Result_: Only very barely improved performance.

### t5_lstm_cnn_4

- **Changes**: Architectural reordering.
- Changed the order of CNN layers to more closely resemble the **SignalP 5.0** architecture.

### t5_lstm_cnn_5

- **Changes**: Data sampling and optimization.
- Trained on an **oversampled dataset** to attempt to fix class imbalance issues.
- Added **weight decay** to the optimizer.
- _Result_: Did not work as expected; performance decreased slightly.

### t5_lstm_cnn_6

- **Changes**: Dual loss function and structured prediction.
- **Combined Loss**: Uses a weighted sum of Cross-Entropy Loss and CRF Loss.
- **Architecture**:
    - Input Embeddings (T5) -> Conv1d (kernel 5) -> GELU -> LSTM (Bidirectional)
    - LSTM Output -> Parallel Conv1d (kernel 7) and Conv1d (kernel 9)
    - Concatenation -> BatchNorm -> GELU -> Dropout -> Linear Classifier
    - CRF and Cross-Entropy Loss for training.
    - CRF Decoding for inference.
- Aims to improve the structural coherence of the predicted signal peptide regions.

