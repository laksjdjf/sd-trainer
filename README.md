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

## Mission Control (Web UI)

学習の起動・監視用のコックピットUI。ブラウザで `http://localhost:7860` を開く。

```bash
uv run uvicorn ui.server:app --port 7860
```

- Config Editor: 構成グループを選んで合成された全設定をフォーム編集、差分だけをオーバーライド化
- 学習をSTART、PAUSE(SIGSTOP)/STOP(SIGTERM)に加え、SAMPLE NOW(SIGUSR1)/
  SAVE NOW(SIGUSR2)でステップの切れ目に割り込みでサンプル生成・保存
- loss EMA・lr・samples/s・ETA・進捗・GPU(VRAM/使用率)をWebSocketでリアルタイム表示
- validationサンプル画像(`image_logs/`)のギャラリーとログのイベントフィード
- PIPELINE: オーバーライドから合成したconfigが実際に使うデータセットを自動スキャンし、
  キャッシュ(caption/latent/text_emb)の充足状況を表示

メトリクスは環境変数 `SD_TRAINER_METRICS` にパスが設定されたときだけ
`trainer.fit()` がJSONLで書き出す(UIサーバーが自動設定。通常のCLI実行には影響なし)。

GPUなしでUIを試すデモモード:

```bash
SD_TRAINER_UI_CMD=".venv/bin/python ui/dev_fake_train.py" uv run uvicorn ui.server:app --port 7860
```


# 参考リポジトリ
https://github.com/harubaru/waifu-diffusion

https://github.com/kohya-ss/sd-scripts

https://github.com/KohakuBlueleaf/LyCORIS

