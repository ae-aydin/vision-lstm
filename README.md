# Vision-LSTM Traversals

Extension of [Vision-LSTM](https://github.com/NX-AI/vision-lstm) comparing patch traversal orders. Vision-LSTM is a recurrent vision model that replaces transformer self-attention with linear-memory LSTM processing. Image patches are processed in sequence; the order depends on the chosen traversal strategy.

Implemented 5 structured patch traversals for ViL-T and trained from scratch on Tiny ImageNet.

## Results

No clear accuracy winner across structured traversals for classification. Traversal strategy affects training dynamics but does not improve generalization. Positional embedding benefit increases with traversal complexity.

## License

MIT License, except `vision_lstm/vision_lstm.py` and `vision_lstm/vision_lstm2.py` which are Apache-2.0.
