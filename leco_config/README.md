# LECO
platdev氏の[https://github.com/p1atdev/LECO](LECO)を参考に、色々変更を加えて実装したものになります。

originalのリポジトリはhttps://github.com/rohitgandikota/erasingになります。

# 簡単な説明
ある概念を消去するようなLoRAを作る学習法です。
たとえば"real"というワードを消したいとき、"real"というプロンプトによる生成結果が、
LoRA適用前のモデルで"real"をネガティブプロンプトにして生成した結果に近づくよう学習します。
つまりプロンプトとネガティブプロンプトの関係を反転させるように学習します。

# 変更点
