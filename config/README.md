# config
設定ファイルはimportlibを利用したモジュール読み込みによって拡張性をあげています。そのためオリジナルのデータセットやらネットワークを定義して使うこと等もできます。

## model
input_path：学習対象のモデルパスです。diffusers限定です。

output_name：出力先は`sd-trainer/trained/hoge/output_name`になります。絶対パスは想定していません。

v2：SDv1かv2かの設定なんですが、今のところ関係ないので参照してません。

v_prediction：SDv2(768)系の場合trueにする必要があります。

## dataset
上にあるようにオリジナルデータセット等も読み込めます。
argsが重要で、pathにはデータセットのパス、maskはmask学習の場合true、controlはpfgの場合trueです。promptでpfg用の共通プロンプト、prefixで全データの最小に同じ文字列を加えることができます。

## save
セーブや検証画像の生成に関わる設定です。

wandb_nameはnullにすると使わなくなります。

over_writeはfalseにすると上書きせず各チェックポイントを残します。

あとはまあなんとなくわかるでしょう。

## train
訓練に関わる設定です。anoはfalseでfloat16、"bfloat16"でbfloat16、falseでfloat32になります。

## feature
わたしのオリジナル実装です。

minibatch_repeat：ミニバッチを拡大することで、少量のデータセットを学習するときにバッチサイズを確保できるようにします。

up_only：UNetのup_blocksのみを学習するオプションです。

step_range：拡散過程の学習範囲を制限するものです。

test_steps：1以上にするとそのステップでプログラムが終了します。つまりテスト用です。

## optimizer
最適化関数を設定します。8bitAdamの場合は、module: bitsandbytes.optim.AdamW8bit とすればできます。他にもlionやらなんやら多分できます。

## network
LoRAやLoCon等のネットワークに関する設定です。
resumeに学習対象のLoRAファイルを設定できます。nullなら一から学習します。

argsにはrankとconv_rankを設定できます。

conv_rankを設定するとLoconになります。

module:"loha"でlohaになります。また"dynamic"を設定すると自動でrankを決めます(loha用)。

## pfg
PFGの設定です。cross_attention_dimはv1なら768にしてください。num_tokensは任意です。
