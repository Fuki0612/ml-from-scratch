# ML from Scratch

機械学習・ニューラルネットワークの主要アルゴリズムを、フレームワーク（PyTorch/TensorFlow等）に頼らず NumPy のみで実装するリポジトリ。仕組みへの理解の深さを示すことが目的。

## 実装予定リスト

### 基礎
- [ ] 線形回帰
- [ ] ロジスティック回帰
- [ ] k-means
- [ ] 決定木

### ニューラルネットワーク
- [ ] 全結合層・活性化関数（ReLU / Sigmoid / Softmax）
- [ ] 誤差逆伝播法（Backpropagation）
- [ ] MLP（多層パーセプトロン） + MNIST での動作確認
- [ ] CNN（畳み込み層・プーリング層）
- [ ] RNN / LSTM
- [ ] Attention機構
- [ ] 簡易Transformer

### 最適化・学習まわり
- [ ] SGD
- [ ] Momentum
- [ ] Adam
- [ ] 各種損失関数（MSE, CrossEntropy）

## 構成

```
src/
  layers/       # 各層の実装（linear, activations, conv, rnn, attention）
  losses.py     # 損失関数
  optim/        # 最適化アルゴリズム
  models/       # 上記を組み合わせたモデル（mlp, cnn, rnn, transformer）
tests/          # 各実装の単体テスト
examples/       # MNIST等での学習・動作確認スクリプト
```

## 方針

- 外部MLライブラリは使用しない（NumPyのみ）
- 各実装には対応するテスト（勾配チェック等）を用意し、正しさを担保する
- README内のチェックリストで進捗を管理する
