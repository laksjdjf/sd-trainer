# config
設定ファイルはimportlibを利用したモジュール読み込みによって拡張性をあげています。そのためオリジナルのデータセットやらネットワークを定義して使うこと等もできます。

## model
input_path：学習対象のモデルパスです。diffusers限定です。

output_path：出力先です。loraは"output_path.pt"、pfgは"output_path-n{tokens}.pt"という形式に変わります。

v2：SDv1かv2かの設定なんですが、今のところ関係ないので参照してません。何であるんだ？

v_prediction：SDv2(768)系の場合trueにする必要があります。

## dataset
上にあるようにオリジナルデータセット等も読み込めます。
argsが重要で、pathにはデータセットのパス、maskはmask学習の場合true、controlはpfgの場合trueです。

## save
セーブや検証画像の生成に関わる設定です。

wandb_nameはnullにすると使わなくなります。

over_writeはfalseにすると上書きせず各チェックポイントを残しますが使ったことないので動くか分かりません。

あとはまあなんとなくわかるでしょう。

## train
訓練に関わる設定です。見た通りの設定しかないので説明はありません。

## feature
わたしのオリジナル実装です。

minibatch_repeatは私ですらなかったことにしています。

up_onlyはUNetのup_blocksのみを学習するオプションです。

step_rangeは拡散過程の学習範囲を制限するものです。

## network
LoRAやLocon等のネットワークに関する設定です。
resumeに学習対象のLoRAファイルを設定できます。nullなら一から学習します。

argsにはrankとconv_rankを設定できます。

conv_rankを設定するとLoconになります。

module:"loha"でlohaになります。また"dynamic"を設定すると自動でrankを決めます。

## pfg
PFGの設定です。cross_attention_dimはv1なら768にしてください。num_tokensは任意です。

## optimizer
最適化関数を設定します。8bitAdamの場合は、module: bitsandbytes.optim、attribute：AdamW8bit とすればできます。他にもlionやらなんやら多分できます。なんで一番したなんだ？
