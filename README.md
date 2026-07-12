# sd-trainer
れあどめ作成中

## 使い方

依存関係はuvで管理している。設定はHydraによるcomposition方式(`conf/`参照)。

```bash
uv sync                                          # 環境構築
uv run python main.py experiment=example         # 実験configで学習
uv run python main.py model=sdxl trainer.lr=1e-4 # CLIオーバーライド
uv run python main.py -m trainer.lr=1e-4,1e-5    # multirunでスイープ
```

- `conf/config.yaml`: 共通デフォルト
- `conf/{model,trainer,dataset,network}/`: 部品グループ
- `conf/experiment/*.yaml`: 実験ごとの差分(gitignore対象、`example.yaml`が見本)
- 実行ごとの解決済みconfigは `outputs/<日付>/<時刻>/.hydra/` に自動保存される

## 前処理

データセットの前処理は `preprocess` パッケージに統一されている(`--help`で各引数を確認)。

```bash
uv run python -m preprocess buckets -i raw_images -o dataset      # bucketへ振り分け+buckets.json
uv run python -m preprocess tags -d dataset -o dataset/captions   # WD Taggerでタグ付け
uv run python -m preprocess latents -d dataset -o dataset/latents -m <model> -t sdxl
uv run python -m preprocess text -d dataset/captions -o dataset/text_emb -m <model> -t sdxl
uv run python -m preprocess masks -d dataset -o dataset/mask      # 顔マスク(要lbpcascade_animeface.xml)
```

各処理はライブラリ関数(進捗コールバック対応)としても呼び出せる。旧単発スクリプトのうち
現役でないもの(PFG関連・旧tagger等)は `preprocess/legacy/` にある。


# 参考リポジトリ
https://github.com/harubaru/waifu-diffusion

https://github.com/kohya-ss/sd-scripts

https://github.com/KohakuBlueleaf/LyCORIS

