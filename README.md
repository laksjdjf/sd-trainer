# sd-trainer
このコードはStable Diffusionのファインチューニング用コードです。
データセット、設定ファイルを用意して、```python main.py <config_file>```で実行できます。
# Preprocess
画像データとそのキャプションデータが必要です。
データセットは以下のようなファイル構造を前提にしています。
```
dataset_directory
  |-buckets.json
  |-latents
  ||-hoge.npy
  |-captions
  ||-hoge.caption
  |-control #controlnetの学習の場合（設定できるように改善予定）
  ||-hoge.png
  |-mask #maskの場合、create_mask.pyで作れる
  ||-hoge.npz
  |-pfg #pfgの場合、tagger_control.pyで作れる
  ||-hoge.npz
```

bucketing.pyでAspect ratio bucketingによる解像度調整およびメタデータの作成を行います。
--resolutionで最大解像度、--min_lengthで最小幅、--max_lengthで最大幅を決定できます。デフォルトの設定は768×768を基準にしています。
```
python3 preprocess/bucketing.py -d "<image_directory>" -o "<dataset_directory>"
```

latent.pyで潜在変数にあらかじめ変換します。モデルは基本的には学習対象を選んでください。
```
python3 preprocess/latent.py -d "<dataset_directory>" -o "<dataset_directory>/latents" -m "<diffusers_model>"
```

説明文を```.caption```という拡張子で`<dataset_directory>/captions`に入れてください。

PFGの学習をする場合は以下のコードでwd14taggerによる特徴量の変換を行います。```.npz```形式で保存されます。使うにはwd14taggerのダウンロードが必要です。
PFGの学習ではcaptionファイルは必要ありません。
```
python3 preprocess/tagger_control.py -d "<dataset_directory>" -o "<dataset_directory>/pfg"
```

# Config
[説明](config/README.md)

# 参考リポジトリ
訓練ループなどの中核コード：https://github.com/harubaru/waifu-diffusion

何を参考にしたか忘れたくらい色々参考にした：https://github.com/kohya-ss/sd-scripts

lohaやlocon:https://github.com/KohakuBlueleaf/LyCORIS

