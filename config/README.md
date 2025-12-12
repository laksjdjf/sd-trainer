# Configuration Guide

このプロジェクトは[Hydra](https://hydra.cc/)を使用して設定を管理しています。

## ⚠️ 重要な変更

**このプロジェクトは Hydra に対応しました！**

- **新しい設定ファイルの場所**: `conf/` ディレクトリ
- **設定グループによる分割**: main, trainer, dataset, dataloader, network ごとにファイルを分けることが可能
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

### 設定グループを切り替える

```bash
# 異なるmain設定を使用（quick = 1エポックのみ）
python main.py main=quick

# 異なるtrainer設定を使用（高い学習率）
python main.py trainer=high_lr

# 複数の設定グループを同時に切り替え
python main.py main=quick trainer=high_lr
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

## 設定ファイルの構造

### メイン設定ファイル

- **`conf/config.yaml`**: デフォルト設定（設定グループを組み合わせる）
- **`conf/example.yaml`**: カスタム設定の例

### 設定グループ（カテゴリ別のファイル分割）

以下のディレクトリに、各カテゴリの設定を分けて配置できます：

- **`conf/main/`**: メイン設定（モデルパス、エポック数など）
  - `default.yaml` - 標準設定
  - `quick.yaml` - 1エポックのクイックトレーニング
  
- **`conf/trainer/`**: トレーナー設定（学習率、最適化など）
  - `base.yaml` - 標準トレーナー
  - `high_lr.yaml` - 高学習率トレーナー
  
- **`conf/dataset/`**: データセット設定
  - `base.yaml` - 標準データセット
  
- **`conf/dataloader/`**: データローダー設定
  - `default.yaml` - 標準データローダー
  
- **`conf/network/`**: ネットワーク設定
  - `lora.yaml` - LoRAネットワーク

### 古い形式

- **`config/example.yaml`**: 旧形式の設定ファイル（参考用）

## Hydraの利点

- ✅ コマンドラインから簡単に設定をオーバーライド可能
- ✅ 設定を論理的なグループに分割して管理可能
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

### 新しい設定グループを作成する場合

設定グループを使用すると、目的別に設定を整理できます。

**例1: 新しいmain設定を追加**

1. `conf/main/my_config.yaml` を作成：

```yaml
# カスタムメイン設定
model_path: "stabilityai/stable-diffusion-xl-base-1.0"
output_path: "output_custom"
seed: 12345
epochs: 20
log_level: "DEBUG"
# その他のフィールドはdefault.yamlから継承される
```

2. 実行：

```bash
python main.py main=my_config
```

**例2: 新しいtrainer設定を追加**

1. `conf/trainer/custom.yaml` を作成：

```yaml
# カスタムトレーナー設定
module: modules.trainer.BaseTrainer
lr: "5e-4"
lr_scheduler: "linear"
gradient_checkpointing: false
# その他の設定...
```

2. 実行：

```bash
python main.py trainer=custom
```

**例3: 複数の設定グループを組み合わせる**

```bash
# クイックトレーニング + 高学習率
python main.py main=quick trainer=high_lr
```

### 新しい全体設定ファイルを作成する場合

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

## 設定グループの詳細

### 設定グループとは？

設定グループは、関連する設定を論理的にまとめる仕組みです。
各ディレクトリ（`main/`, `trainer/` など）には、複数の設定バリエーションを配置できます。

### 利点

- 🎯 **モジュール化**: 設定を小さな部品に分割
- 🔄 **再利用性**: 共通設定を複数の構成で再利用
- 🎛️ **柔軟性**: 実行時に簡単に切り替え可能
- 📊 **実験管理**: 異なる設定の組み合わせを簡単にテスト

### ディレクトリ構造の例

```
conf/
├── config.yaml          # メイン設定（defaults で各グループを指定）
├── example.yaml         # カスタム設定例
├── main/
│   ├── default.yaml     # 標準設定
│   ├── quick.yaml       # クイックトレーニング
│   └── my_config.yaml   # あなたのカスタム設定
├── trainer/
│   ├── base.yaml        # 標準トレーナー
│   ├── high_lr.yaml     # 高学習率
│   └── adamw8bit.yaml   # 8-bit Adam (例)
├── dataset/
│   └── base.yaml
├── dataloader/
│   └── default.yaml
└── network/
    ├── lora.yaml        # LoRA
    └── full.yaml        # Full fine-tuning (例)
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

