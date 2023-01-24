# sd-trainer
九龍城砦と化した訓練コードをなんとかするため、一から訓練コードを自分で書いてみようと思う。勉強にもなりそうだしな。
かなりの部分を[waifu-diffusion](https://github.com/harubaru/waifu-diffusion)から援用するはず。
また様々な機能を[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)を参考にして九龍城砦の建築を目指す。あれ？

# Usage
使う人いないだろうけど自分用に書いておく、テストもしてない）

use_bucketのコマンドライン引数で、SimpleDataset(すべて同じ解像度のデータ)かAspectDataset(異なるアスペクト比を使うデータ)か選べる。AspectDatasetを使う場合、前処理がいっぱい必要である。また個人的な事情でAspectDatasetを使う場合キャプションファイルの拡張子は.captionとする。

preprocessディレクトリには前処理用のコードがあり、基本的にはbucketing.pyでAspect ratio bucketingして、latent.pyで潜在変数に変えてセーブしておく。tagger.pyとcaption_preprocessor.pyはdanbooruのメタデータがあることが前提であり、自分以外が使うことを想定していない。

loraもつくったけど、kohya-ss氏のコードを簡略化して実装している。互換性は今のところあると思うけど。。。


# Update
2023/01/24 aspect bucketing+latent cache,前処理コード,wandbログ等々
2023/01/23 計画開始
