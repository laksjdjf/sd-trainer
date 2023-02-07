EHは私が考えた謎のネットワークです。Transformersの全結合層(to_q等)はトークンごとに埋め込みベクトルの全結合を行います。
EHでは埋め込みベクトルを $n$ グループに分けて全結合を行うことで、パラメータ数を $n^2$ 分の $1$ に抑えられます。

Original Linear:
$x_{out} = W_{org}x_{in}$

EH Linear:
$x_{out} = W_{org}x_{in} + \[W_{eh}x_{in}^{1} \cdots W_{eh}x_{in}^{n}\]^{T} \ \ \ (x_{in} = \[x_{in}^{1} \cdots  x_{in}^{n}\]^{T})$

$W_{out}$のサイズを(row,col)とするとき、$W_{eh}$のサイズは(row/n,col/n)となります。$W_{out}$をfrozenにして、$W_{eh}$のみ学習します。

$W_{eh}$の初期重みは0です。これにより学習の開始時点ではモデルの出力が同じになります。(LoRAやHypernetworksと同様)

ただの思い付きですが、そういえばmulti head attentionの分割数と同じにすればなんかすっきりするようなしないような気もしてきます。ただ実装が（特にv2では）めんどくさいのでやっていない。
