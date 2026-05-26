# Vision-LSTM

[[`Webpage`](https://nx-ai.github.io/vision-lstm)]
[[`Paper`](https://arxiv.org/abs/2406.04303)]

*PyTorch implementation of Vision-LSTM (ViL), an adaptation of xLSTM to computer vision.*

---

## About this Fork

This fork extends Vision-LSTM to study the effect of patch traversal order on model performance. Five traversal strategies are compared by training ViL-T from scratch on Tiny ImageNet:

- **Row-wise** (baseline)
- **Diagonal Zig-Zag**
- **Spiral (Outward)**
- **Hilbert Curve**
- **Random Fixed** (seed 42)

---

## License

This project is licensed under the MIT License, except the following folders/files which are licensed under the Apache-2.0 license:

- `src/vislstm/modules/xlstm`
- `vision_lstm/vision_lstm.py`
- `vision_lstm/vision_lstm2.py`

## Citation

Original Vision-LSTM paper:

```bibtex
@article{alkin2024visionlstm,
  title={{Vision-LSTM}: {xLSTM} as Generic Vision Backbone},
  author={Benedikt Alkin and Maximilian Beck and Korbinian P{\"o}ppel and Sepp Hochreiter and Johannes Brandstetter},
  journal={arXiv preprint arXiv:2406.04303},
  year={2024}
}
```
