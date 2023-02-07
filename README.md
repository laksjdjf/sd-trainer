[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/laksjdjf/sd-trainer/blob/main/sd_trainer_for_colab.ipynb)

# sd-trainer
九龍城砦と化した訓練コードをなんとかするため、一から訓練コードを自分で書いてみようと思う。勉強にもなりそうだしな。
かなりの部分を[waifu-diffusion](https://github.com/harubaru/waifu-diffusion)から援用するはず。
また様々な機能を[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)を参考にして九龍城砦の建築を目指す。あれ？

# Usage
use_bucketのコマンドライン引数で、SimpleDataset(すべて同じ解像度のデータ)かAspectDataset(異なるアスペクト比を使うデータ)か選べる。AspectDatasetを使う場合、前処理がいっぱい必要である。また個人的な事情でAspectDatasetを使う場合キャプションファイルの拡張子は.captionとする。

preprocessディレクトリには前処理用のコードがあり、基本的にはbucketing.pyでAspect ratio bucketingして、latent.pyで潜在変数に変えてセーブしておけば動くかもしれない。tagger.pyとcaption_preprocessor.pyはdanbooruのメタデータがあることが前提であり、自分以外が使うことを想定していない。

loraもつくったけど、kohya-ss氏のコードを簡略化して実装している。互換性は今のところあると思うけど。。。

mask学習にはcreate_mask.pyでマスクを作る必要がある。重みのダウンロードが必要。

# Features

他の人が考えたすごい機能と自分が思い付きで作った謎の機能を紹介します。

+ SDの訓練コードに最低限必要な機能はそろっていると思います。まだ足りないものとしては、optimizerの状態を保存することや、勾配蓄積、重みのEMAくらいだが、個人的にはあまり欲しい機能ではないのでやる気が出ない。

+ Aspect ratio bucketing: 
異なるアスペクト比のデータセットを学習する手法、[これ](https://github.com/NovelAI/novelai-aspect-ratio-bucketing)を参考にした。

+ latent cache: 潜在変数をあらかじめ計算しておくことで、計算時間を削減する手法。私はkohya氏の実装を参考にしたが、元々は何かのDreamboothにあった発想のはず。

+ wandbによる訓練lossや生成画像のログ監視: ほとんど[waifu-diffusion](https://github.com/harubaru/waifu-diffusion)にあったもの。

+ lora: トレーニング対象のパラメータを大幅削減するすごい手法、最初にSDに適用したのは、cloneofsimo氏だが実装はkohya氏のものを少し簡略化したものになっている。

+ loraの重み監視：　loraの重み絶対値平均を監視できる機能。これもwandbを使っている。

+ masked score estimation: [これ](https://github.com/cloneofsimo/lora/discussions/96)を実装した。ただしcloneofsimo氏の実装と違い、maskに正規化をほどこしていない（mask / mask.mean()をすると、一部の要素が異常に大きくなりオーバーフローを起こす可能性がある）

+ up_blocksのみ学習：　私の思い付き。単に出力側の一部に限定すれば逆伝搬を途中までしか計算せずに済み、効率がいいのではという発想だよ。

+ minibatch repeat：　bucketingを行うと、特に小さいデータセットではバッチサイズに届かないbucketが増えてしまい、バッチサイズを増やしづらくなる。そこでミニバッチをそのまま拡大することで、バッチサイズを上げたことと同じような効果が得られる。通常の学習では同一ステップに同一データを重複させても意味がないが、拡散モデルの場合ノイズやtimestepもランダムに決定されるため、問題がないと思われる。

+ step range: 学習対象のサンプリングステップを制限する。これにより、低ステップのみに制限すればスタイルを、高ステップのみに制限すれば構図を学習できるのではないかと期待している。特にmasked score estimationとの相性がバッチグーだと思うんだけどどうでしょう。



# Update
2023/02/07 謎のネットワーク「EH」

2023/01/26 学習率スケジューラー、ミニバッチリピート

2023/01/25 mask学習、sampling step制限、loraの重みを学習中に監視してみる

2023/01/24 aspect bucketing+latent cache,前処理コード,lora,wandbログ等々

2023/01/23 計画開始
