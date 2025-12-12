# sd-trainer

Stable Diffusion用のトレーニングフレームワーク（Hydra対応）

## 使い方

```bash
# デフォルト設定で実行
python main.py

# 設定をコマンドラインでオーバーライド
python main.py main.epochs=10 main.seed=1234

# カスタム設定ファイルを使用
python main.py --config-path=/path/to/config --config-name=my_config
```

詳細は[config/README.md](config/README.md)を参照してください。

れあどめ作成中


# 参考リポジトリ
https://github.com/harubaru/waifu-diffusion

https://github.com/kohya-ss/sd-scripts

https://github.com/KohakuBlueleaf/LyCORIS

