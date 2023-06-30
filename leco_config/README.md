# LECO
platdev氏の[https://github.com/p1atdev/LECO](LECO)を参考に、色々変更を加えて実装したものになります。

解説記事はこちらになります。https://zenn.dev/aics/articles/lora_for_erasing_concepts_from_diffusion_models

originalのリポジトリはhttps://github.com/rohitgandikota/erasingになります。

# 簡単な説明
targetのワードの意味をpositiveの意味に近づける（または遠ざける）ように学習します。
target_guidance_scaleを大きくすると、より近づくようになり、マイナスにすると遠ざかるようになります。

例１：
target: "real life"

positive: null # nullにすると自動的にtargetと同じワードになります。

target_guidance_scale: -3

この例ではtargetが自分自身の意味から離れていきます。つまり意味が消失して、"real life"という単語で実写画像が生成できなくなります。

例２：

target: "1girl"

positive: "1girl futanari"

target_guidance_scale: 3

この例では"1girl"の意味が"1girl futanari"に近づいていきます。これによって特に指定しなくても勝手におちｎ****

# 変更点

+ LECOでは学習用の画像をつくるのではなく、学習中に画像を生成します。
学習時にランダムにステップ数を決めて、そのステップ数分ノイズ除去した画像を使います。この処理に時間がかかるため、私の実装ではノイズ除去ループ中の途中結果も学習に使うようにしました。1回のループで何回分の画像を使うかがnum_samplesで設定できます。あげれば学習速度が上がりますが、精度はさがるかもしれません。

+ neutralはよく分からなかったので廃止しました。

+ LoRAの実装はKohya氏のものと少し違うので、なんかでエラーが起こるかもしれません。


