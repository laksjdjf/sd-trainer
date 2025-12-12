# Configuration Guide

このプロジェクトは[Hydra](https://hydra.cc/)を使用して設定を管理しています。

## ⚠️ 重要な変更

**このプロジェクトは Hydra に対応しました！**

- **新しい設定ファイルの場所**: `conf/` ディレクトリ
- **古い設定ファイル**: `config/example.yaml` は参考用として残されています

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
# conf/example.yaml を使用
python main.py --config-name=example

# カスタムパスから設定を読み込む
python main.py --config-path=/path/to/config --config-name=my_config
```

### 複数の設定を組み合わせる

```bash
python main.py main.model_path="stabilityai/stable-diffusion-xl-base-1.0" trainer.lr=1e-4
```

## 設定ファイルの場所

- **デフォルト設定**: `conf/config.yaml`
- **例の設定**: `conf/example.yaml`
- **古い形式の例**: `config/example.yaml` （参考用）

## Hydraの利点

- ✅ コマンドラインから簡単に設定をオーバーライド可能
- ✅ 自動的に実行ごとの出力ディレクトリを作成 (`outputs/YYYY-MM-DD/HH-MM-SS/`)
- ✅ 設定の履歴管理が容易
- ✅ マルチラン（ハイパーパラメータサーチ）のサポート
- ✅ 設定ファイルの合成と継承が可能

## マルチラン例

```bash
# 異なるシードと学習率の組み合わせで複数回実行
python main.py -m main.seed=1,2,3,4,5 trainer.lr=1e-3,1e-4
```

これにより、シード5つ × 学習率2つ = 合計10回の実行が行われます。

## 設定ファイルの作成

### 新しい設定ファイルを作成する場合

1. `conf/` ディレクトリに新しいYAMLファイルを作成
2. ベース設定を継承する場合：

```yaml
defaults:
  - config  # ベース設定を継承
  - _self_

# カスタマイズする項目だけを記述
main:
  epochs: 20
  seed: 12345
```

3. 実行：

```bash
python main.py --config-name=your_config
```

## トラブルシューティング

### 出力ディレクトリについて

Hydraはデフォルトで `outputs/YYYY-MM-DD/HH-MM-SS/` に出力を保存します。
これを変更したい場合：

```bash
python main.py hydra.run.dir=./my_output_dir
```

### 設定ファイルが見つからない場合

- `conf/` ディレクトリが存在することを確認
- `--config-name` で拡張子なしのファイル名を指定（`.yaml` は不要）
- カスタムパスを使用する場合は `--config-path` を指定

