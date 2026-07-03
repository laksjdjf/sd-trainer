# sd-trainer
れあどめ作成中

## 使い方

依存関係はuvで管理している。設定はHydraによるcomposition方式(`conf/`参照)。

```bash
uv sync                                          # 環境構築
uv run python main.py experiment=example         # 実験configで学習
uv run python main.py model=sdxl trainer.lr=1e-4 # CLIオーバーライド
uv run python main.py -m trainer.lr=1e-4,1e-5    # multirunでスイープ
```

- `conf/config.yaml`: 共通デフォルト
- `conf/{model,trainer,dataset,network}/`: 部品グループ
- `conf/experiment/*.yaml`: 実験ごとの差分(gitignore対象、`example.yaml`が見本)
- 実行ごとの解決済みconfigは `outputs/<日付>/<時刻>/.hydra/` に自動保存される


# 参考リポジトリ
https://github.com/harubaru/waifu-diffusion

https://github.com/kohya-ss/sd-scripts

https://github.com/KohakuBlueleaf/LyCORIS

