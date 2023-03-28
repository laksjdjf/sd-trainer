# sd-trainer
このコードはStable Diffusionのファインチューニング用コードです。

# Preprocess
画像データとそのキャプションデータが必要です。

bucketing.pyでAspect ratio bucketingによる解像度調整およびメタデータの作成を行います。
--resolutionで最大解像度、--min_lengthで最小幅、--max_lengthで最大幅を決定できます。デフォルトの設定は768×768を基準にしています。
```
python3 preprocess/bucketing.py -d <image_directory> -o <dataset_directory>
```

latent.pyで潜在変数にあらかじめ変換します。モデルは基本的には学習対象を選んでください。
```
python3 preprocess/latent.py -d <dataset_directory> -o <dataset_directory> -m "<diffusers_model>"
```

説明文を```.caption```という拡張子で画像を同じファイル名にして同じディレクトリに入れてください。

最終的に<dataset_directory>に画像分の"hoge.npy"と"hoge.caption"、"buckets.json"があれば学習できます。

PFGの学習をする場合は以下のコードでwd14taggerによる特徴量の変換を行います。```.npz```形式で保存されます。使うにはwd14taggerのダウンロードが必要です。
PFGの学習ではcaptionファイルは必要ありません。
```
python3 preprocess/tagger_control.py -d <dataset_directory> -o <dataset_directory>
```

# Config
configに設定例をあげているのでなんとなく察してください。大項目だけ軽く説明します。
+ model 学習先モデルおよび出力先、v2やv_predictionを設定します。v2は現状なにもかわりませんが＾＾
+ dataset 基本的にはargs.pathをデータセットがあるパスに変えればいいだけです。
+ save モデルのセーブや検証画像の生成に関わる設定項目です。
+ train 訓練に関わる設定です。
+ feature 私のオリジナル設定たちです。設定非推奨＾＾
+ network LoRA等の設定です
+ pfg PFGの設定です
+ optimizer 最適化関数の設定です。たとえばtorch.optim⇒bitsandbytes.optim、AdamW⇒AdamW8bitで8bitAdamになります。

# 参考リポジトリ
訓練ループなどの中核コード：https://github.com/harubaru/waifu-diffusion

何を参考にしたか忘れたくらい色々参考にした：https://github.com/kohya-ss/sd-scripts

lohaやlocon:https://github.com/KohakuBlueleaf/LyCORIS

