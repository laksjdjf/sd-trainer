# UI開発・デモ用の偽トレーナー。本物のmain.pyと同じ形式でメトリクスJSONL・ログ・
# サンプル画像を出力する。GPU不要。
#
#   SD_TRAINER_UI_CMD=".venv/bin/python ui/dev_fake_train.py" uv run uvicorn ui.server:app --port 7860
import json
import math
import os
import random
import time

from PIL import Image, ImageDraw

TOTAL = int(os.environ.get("FAKE_TOTAL", 600))
SAMPLE_EVERY = TOTAL // 5


def write_metrics(f, record):
    if f:
        f.write(json.dumps(record) + "\n")


def fake_image(step, idx):
    img = Image.new("RGB", (208, 304), (12 + step % 20, 16, 40))
    draw = ImageDraw.Draw(img)
    rnd = random.Random(step * 7 + idx)
    for _ in range(9):
        x, y = rnd.randint(0, 208), rnd.randint(0, 304)
        r = rnd.randint(20, 90)
        color = rnd.choice([(255, 111, 165), (95, 217, 232), (255, 180, 84), (150, 120, 255)])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color + (0,), outline=None)
    return img


def main():
    metrics_path = os.environ.get("SD_TRAINER_METRICS")
    f = open(metrics_path, "a", buffering=1) if metrics_path else None
    os.makedirs("image_logs", exist_ok=True)

    print("シードは4545だよ！")
    print("モデルをfake/animagine-xl-3.0からロードしちゃうよ！")
    print("データセットのサイズは842だよ！")
    print(f"トータルのステップ数は{TOTAL}だよ！")
    write_metrics(f, {"event": "start", "total_steps": TOTAL, "total_epochs": 24, "output_path": "fake_run", "time": time.time()})

    loss_ema = 0.14
    for step in range(TOTAL):
        t = step / TOTAL
        loss = 0.135 * math.exp(-2.2 * t) + 0.066 + (random.random() - 0.5) * 0.006
        loss_ema = loss_ema * 0.99 + loss * 0.01
        lr = 1e-4 * 0.5 * (1 + math.cos(math.pi * t))
        logs = {"loss_ema": round(loss_ema, 5), "samples_per_second": round(2.4 + random.random() * 0.3, 2), "lr": lr}
        write_metrics(f, {"event": "step", "step": step, "epoch": step * 24 // TOTAL, "time": time.time(), **logs})

        if step % SAMPLE_EVERY == 0 or step == TOTAL - 1:
            print("サンプルを生成するよ！")
            files = []
            for i in range(2):
                name = f"{step}_{i}.png"
                fake_image(step, i).save(f"image_logs/{name}")
                files.append(name)
            write_metrics(f, {"event": "sample", "step": step, "files": files, "time": time.time()})
            print("モデルを保存します！")
            write_metrics(f, {"event": "save", "step": step, "time": time.time()})

        if step % (TOTAL // 24 or 1) == 0 and step > 0:
            print(f"エポック{step * 24 // TOTAL}が終わったよ！")
        time.sleep(0.08)

    print("トレーニングが終わったよ！")
    write_metrics(f, {"event": "end", "step": TOTAL, "time": time.time()})


if __name__ == "__main__":
    main()
