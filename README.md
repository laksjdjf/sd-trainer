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

## MOSS-Image（実験的対応）

[OpenMOSSの公式モデル](https://github.com/OpenMOSS/MOSS-Image)のTransformer LoRA学習に対応。
`model=moss_image` でBaseをロードする。SFTは
`main.model_path=OpenMOSS-Team/MOSS-Image-SFT` で選択できる。
Diffusers形式のHubリポジトリまたはローカルディレクトリを指定する。

以下は `dataset/images/*.png`、`dataset/captions/*.caption`、
`dataset/buckets.json` を用意した例。画像の縦横は16の倍数にする。
画像ディレクトリにbuckets.jsonがない場合、latent前処理は1枚ずつ処理する。

```bash
uv sync --locked
uv run python -m preprocess latents -d dataset/images -o dataset/latents_moss_image \
  -m OpenMOSS-Team/MOSS-Image-Base -t moss_image --dtype fp32 -b 1
uv run python -m preprocess text -d dataset/captions -o dataset/text_emb_moss_image \
  -m OpenMOSS-Team/MOSS-Image-Base -t moss_image --dtype bf16 --save_dtype fp32 -b 1
uv run python main.py model=moss_image +dataset.args.path=dataset \
  main.output_path=moss_image_lora main.steps=1000 main.wandb=null \
  trainer.validation_args.prompt="A red ceramic teapot on a wooden table."
```

専用latentキャッシュはFLUX.2 VAE出力を2×2パッチ化した128ch・空間16倍圧縮の
未正規化テンソル。学習時に128chそれぞれのBN統計量で正規化する。
既存モデルのlatentやテキストキャッシュは流用しない。
Qwen3.5は各プロンプトをパディングせずに処理し、最終hidden stateとattention maskを保存する。
テキストをキャッシュしない場合は `dataset.args.text_emb=null` を指定する。
その場合、毎ステップTEを使うため `trainer.te_device=cuda` または別GPUへの配置を指定する。

設定はbf16のTransformer・TE、fp32のVAE、gradient checkpointingを使用する。
TE/VAEはCPUに退避し、サンプルは1024×1024・50ステップ・CFG 3。
公式の最近傍グリッドスケジュール（推論shift 5、学習shift 1）と、
負条件側の22番ブロックのスキップ、fp32でのCFG・Euler更新を使う。
LoRAは `trained/networks/<output_path>.safetensors` に保存され、
`network.resume=...` で重みを再読込できる。本リポジトリのNetworkManager形式で保存する。

ComfyUIでは別ファイルに変換して `models/loras/` に置き、MOSSモデルに
「Load LoRA (Model Only)」で適用する。

```bash
uv run python -m tools.convert_moss_image_lora \
  trained/networks/moss_image_lora.safetensors \
  trained/networks/moss_image_lora_comfy.safetensors
```

FFN用LoRAはComfyUIローダーで90モジュールの対応と重み形状を確認済み。
ComfyUIでの実画像生成は未検証。再学習後、同じ変換先を更新する場合は `--overwrite` を付ける。

検証は `uv run python -m tools.verify_moss_image`。
小型の実Qwen3.5・FLUX.2 VAE・MOSS Transformerで、コンポーネントロード、
キャッシュ経由の学習、LoRA更新・再読込、bf16サンプル計算を確認する。
公開済みの全重みでの学習完走・生成品質・必要VRAMは未検証。
現在の対象はBaseTrainer・BaseDatasetでのT2I Transformer LoRA。
フルモデル／TE学習、NF4/FP8、ControlNet、動画は対象外。
Transformerと推論スケジューラは公式コードの固定リビジョンを同梱しており、
実行時にリモートのPythonコードをロードしない。出典と変更点は
`modules/moss_image_model.py`、Apache-2.0ライセンスは `licenses/` に記載。

## Qwen-Image-2.1（実験的対応）

通常画像のテキスト条件付きTransformer LoRA学習に対応。モデル指定は
`model=qwenimage21`。Diffusers・Transformersの更新を含むので、最初に
`uv sync --locked` を実行する。

このモデル専用のlatentキャッシュを作る。既存モデルのlatentは流用不可。
テキスト埋め込みはキャッシュせず、毎ステップ、キャプションから計算する（text_emb: null）。
VAEはRGBA入力、空間16倍圧縮・64ch。RGB画像には不透明なアルファを追加する。
画像サイズは縦横とも32の倍数にする（既存のbucket前処理は64刻み）。
可変長テキストのattention maskも学習時に渡す。

以下は `dataset/images/*.png`、`dataset/captions/*.caption`、
`dataset/buckets.json` を用意した場合の例。空きGPUを指定して実行する。

```bash
uv run python -m preprocess latents -d dataset/images -o dataset/latents_qwenimage21 \
  -m Qwen/Qwen-Image-2.1 -t qwenimage21 --dtype bf16 -b 1
uv run python main.py model=qwenimage21 +dataset.args.path=dataset \
  main.output_path=qwenimage21_lora main.steps=1000 main.wandb=null \
  trainer.validation_args.prompt="A red ceramic teapot on a wooden table." \
  trainer.validation_args.negative_prompt=""
```

モデル設定はbf16、gradient checkpointing、TEのGPU実行・VAEのCPU退避、
1024×1024・40ステップ・guidance scale 1のサンプル生成を指定する。
TEも毎ステップ使うため、Transformerと別のGPUに配置する場合はtrainer.te_device=cuda:1を指定する。
CUDA_VISIBLE_DEVICESで両方のGPUを見せる必要がある。
公開済みの全重みを使った学習の完走は未検証。

LoRAは既存の `trained/networks/<output_path>.safetensors` に保存される。
`network.resume=...` でアダプターの重みを再ロードできる（optimizer状態の再開ではない）。
保存形式は本リポジトリのNetworkManager形式。ComfyUI向けには別ファイルに変換する。

```bash
uv run python -m tools.convert_qwenimage21_lora \
  trained/networks/qwenimage21_lora.safetensors \
  trained/networks/qwenimage21_lora_comfy.safetensors
```

変換先は既存ファイルを上書きしないため、再変換時は新しいファイル名を指定する。
Qwen-Image-2.1対応のComfyUIローダーで224モジュールの対応と、
融合MLPのgate/projへの割り当てを確認済み。ComfyUIでの実画像生成は未検証。

現在の対象はBaseTrainer・BaseDatasetでのT2I LoRA。
画像編集・参照画像条件、フルモデル学習、テキストエンコーダー学習、
NF4/FP8、ControlNetは対象外。RGBデータから学習経路を確認しており、
透過画像の学習品質も未検証。重みの利用条件は
[公式モデルのライセンス](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)を参照。

ConvRotは `trainer.convrot=forward` または `trainer.convrot=backward` で有効にできる。
小型モデルで14層のINT8化、gradient checkpointing下のLoRA更新、RGBA生成を確認済み。
実モデル全体での学習品質・速度・必要VRAMは未検証。
VRAM削減を優先する場合はforwardから試す。backwardは勾配計算用の転置INT8重みも保持する。
対象はTransformerのブロック内Linearで、TE/VAEは量子化しない。
量子化前にはbf16のTransformerをGPUへ配置するため、ロード時のピークVRAMは残る。

検証コマンド：

```bash
# ダウンロード不要。勾配、checkpointing、LoRA保存・再ロード、キャッシュ、scheduler
uv run python -m tools.verify_qwenimage21
# 公式processorのみ取得。小型のランダム重みでGPU学習2ステップとRGBA生成
CUDA_VISIBLE_DEVICES=1 uv run python -m tools.smoke_qwenimage21
CUDA_VISIBLE_DEVICES=1 uv run python -m tools.smoke_qwenimage21 --convrot forward
CUDA_VISIBLE_DEVICES=1 uv run python -m tools.smoke_qwenimage21 --convrot backward
```

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

