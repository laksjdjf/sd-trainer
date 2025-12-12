# Configuration Directory

このディレクトリにはHydraの設定ファイルが含まれています。

## ディレクトリ構造

```
conf/
├── config.yaml          # メイン設定ファイル（デフォルト）
├── example.yaml         # カスタム設定の例
├── main/                # メイン設定グループ
│   ├── default.yaml     # 標準設定
│   └── quick.yaml       # クイックトレーニング（1エポック）
├── trainer/             # トレーナー設定グループ
│   ├── base.yaml        # 標準トレーナー
│   └── high_lr.yaml     # 高学習率トレーナー
├── dataset/             # データセット設定グループ
│   └── base.yaml        # 標準データセット
├── dataloader/          # データローダー設定グループ
│   └── default.yaml     # 標準データローダー
└── network/             # ネットワーク設定グループ
    └── lora.yaml        # LoRAネットワーク
```

## 使い方

### デフォルト設定で実行

```bash
python main.py
```

### 設定グループを切り替える

```bash
# クイックトレーニング設定を使用
python main.py main=quick

# 高学習率トレーナーを使用
python main.py trainer=high_lr

# 複数の設定グループを組み合わせ
python main.py main=quick trainer=high_lr
```

### 新しい設定グループを追加

例: 新しいmain設定を追加する場合

1. `conf/main/my_config.yaml` を作成
2. 必要な設定を記述（他のフィールドは継承される）
3. `python main.py main=my_config` で実行

## 詳細

設定の詳細については、`config/README.md` を参照してください。
