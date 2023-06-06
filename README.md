# sd-trainer
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laksjdjf/sd-trainer/blob/main/sd_trainer_tutorial.ipynb)

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
  |-pfg #pfgの場合、create_pfg_feature.pyで作れる
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
学習設定は全て設定ファイルに入力します。設定例が```config```にあります。[説明](config/README.md)

# Feature
実装した機能
+ xformers、AMP、gradient_checkpointing等のメモリ効率化手法
+ Aspect ratio bucketingによる複数のアスペクト比の学習
+ LoRA、LoCon、Lohaの学習
+ Maskを使った学習
+ ControlNetの学習
+ PFGの学習
+ wandbによるログチェック
+ 設定ファイルからOptimizerを自由に選べる
+ データセット、セーブ機能の拡張性（クラスの定義のみで使えるはず）
+ LoRAとControlNetの同時学習・LoRAを適用だけして学習対象としないといったこともできる（はず）。ただそれぞれ別々の学習率を設定できないのでできるようにしたいがあまり複雑にしたくもない。
+ ミニバッチリピート（データセットが小規模のとき、ミニバッチを同一データで拡大することで大きなバッチサイズを確保できます。同一ステップに同一データがあってもtimestepが違うので多少意味はあると思います。）
+ ステップ範囲指定（timestepの範囲を限定します。"0.0,0.5"とかにすればスタイルを、"0.5,1.0"とかにすれば構図を学びやすくなるかもしれない。）
+ UNetのup_blocksのみの学習（LoRAにも対応）
+ キャプションの先頭に文字列（"anime, "とか？）を追加するやつ
+ キャプションをランダムにドロップアウトする機能（大規模学習する場合以外には必要ないと思います。）
+ [tomesd](https://github.com/dbolya/tomesd)を使った学習

実装してない機能
+ Dreambooth（データセットを継承すれば割と簡単に実装できそうな気がする）
+ hypernetworks
+ Textual Inversion
+ Latentをcacheしないで学習（訓練ループ自体には機能がありますが、対応するデータセットがない）
+ データ拡張（Latentにしてしまうためありません。）
+ トークン長の拡張（個人的に学習段階でトークン長の拡張を行うのはよくないと思うのですがどうでしょうか。）

実装したい機能
+ ckpt, safetensors(compvis？)モデルの対応
+ LoRA等のsafetensorsによる保存
+ 複数GPU対応
+ gradient accumulation
+ DeepFloyd IFへの対応（検証画像生成周りをもうちょっとなんとかしたい）
+ Lokr
+ [Min-SNR-weight](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training)
+ [noise_offset](https://www.crosslabs.org/blog/diffusion-with-offset-noise)
+ VAEの学習（誰か損失関数教えて）


# 参考リポジトリ
訓練ループなどの中核コード：https://github.com/harubaru/waifu-diffusion

何を参考にしたか忘れたくらい色々参考にした：https://github.com/kohya-ss/sd-scripts

lohaやlocon:https://github.com/KohakuBlueleaf/LyCORIS

Token Merging:https://github.com/dbolya/tomesd

