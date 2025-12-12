# Configuration Guide

このプロジェクトは[Hydra](https://hydra.cc/)を使用して設定を管理しています。

## 基本的な使い方

### デフォルト設定で実行

```bash
python main.py
```

### 設定をコマンドラインでオーバーライド

```bash
python main.py main.epochs=10 main.seed=1234
```

### カスタム設定ファイルを使用

```bash
python main.py --config-path=/path/to/config --config-name=my_config
```

### 複数の設定を組み合わせる

```bash
python main.py main.model_path="stabilityai/stable-diffusion-xl-base-1.0" trainer.lr=1e-4
```

## 設定ファイルの場所

- デフォルト設定: `conf/config.yaml`
- 古い形式の例: `config/example.yaml`

## Hydraの利点

- コマンドラインから簡単に設定をオーバーライド可能
- 自動的に実行ごとの出力ディレクトリを作成
- 設定の履歴管理が容易
- マルチラン（ハイパーパラメータサーチ）のサポート

## マルチラン例

```bash
python main.py -m main.seed=1,2,3,4,5 trainer.lr=1e-3,1e-4
```

これにより、異なるシードと学習率の組み合わせで複数回実行されます。
