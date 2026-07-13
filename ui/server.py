# Mission Control UIのバックエンド。学習subprocessの起動/停止、メトリクス・ログ・
# サンプル画像・GPU状態のWebSocket配信を行う。
#
#   uv run uvicorn ui.server:app --port 7860
#
# 学習コマンドはデフォルトで `python main.py <overrides>`。SD_TRAINER_UI_CMD で
# 差し替えられる(デモ: SD_TRAINER_UI_CMD="python ui/dev_fake_train.py")。
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import BaseModel

from modules.config import Config

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "image_logs"
METRICS_PATH = ROOT / "ui" / "run_metrics.jsonl"
CONF_DIR = ROOT / "conf"
CONFIG_GROUPS = ["model", "trainer", "dataset", "network", "experiment"]

# main.pyと同じスキーマ登録(conf/config.yamlのdefaults先頭のbase_schemaが解決できるように)
ConfigStore.instance().store(name="base_schema", node=Config)
_compose_lock = threading.Lock()

app = FastAPI(title="sd-trainer mission control")

# ---------------------------------------------------------------------------
# 状態とブロードキャスト
# ---------------------------------------------------------------------------
clients: set = set()
loop: asyncio.AbstractEventLoop | None = None

state = {
    "status": "idle",  # idle | training | paused | finished | failed | stopped
    "run_info": None,  # metricsのstartイベント
    "overrides": [],
    "steps": deque(maxlen=2000),   # stepイベント履歴
    "logs": deque(maxlen=300),     # stdout行
    "samples": deque(maxlen=24),   # {"url", "name", "step"}
    "gpu": None,
}
proc: subprocess.Popen | None = None


async def _send_all(msg):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def broadcast(msg):
    # ワーカースレッドから呼べるthread-safeなブロードキャスト
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_send_all(msg), loop)


def set_status(status):
    state["status"] = status
    broadcast({"type": "status", "status": status})


# ---------------------------------------------------------------------------
# ワーカースレッド
# ---------------------------------------------------------------------------
def _stdout_reader(p):
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        state["logs"].append(line)
        broadcast({"type": "log", "line": line})
    code = p.wait()
    if state["status"] in ("training", "paused"):
        set_status("finished" if code == 0 else "failed")


def _metrics_tail(p):
    # 学習プロセスが書くJSONLを追いかけて配信する
    while not METRICS_PATH.exists():
        if p.poll() is not None:
            return
        time.sleep(0.2)
    with open(METRICS_PATH, "r") as f:
        while True:
            line = f.readline()
            if not line:
                if p.poll() is not None:
                    return
                time.sleep(0.2)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "start":
                state["run_info"] = record
            elif record.get("event") == "step":
                state["steps"].append(record)
            broadcast({"type": "metrics", **record})


def _image_watcher(p, started_at):
    seen = set()
    while p.poll() is None or len(seen) == 0:
        if IMAGE_DIR.exists():
            for f in sorted(IMAGE_DIR.glob("*.png")):
                if f.name in seen or f.stat().st_mtime < started_at:
                    continue
                seen.add(f.name)
                step = f.name.split("_")[0]
                sample = {"url": f"/images/{f.name}", "name": f.name, "step": step}
                state["samples"].append(sample)
                broadcast({"type": "sample", **sample})
        if p.poll() is not None:
            return
        time.sleep(1.5)


def _gpu_watcher():
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().splitlines()[0]
            used, total, util = [float(x) for x in out.split(",")]
            gpu = {"used_gb": round(used / 1024, 1), "total_gb": round(total / 1024, 1), "util": util}
        except Exception:
            gpu = None
        state["gpu"] = gpu
        broadcast({"type": "gpu", "gpu": gpu})
        time.sleep(2)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    overrides: list[str] = []


@app.post("/api/train/start")
def train_start(req: StartRequest):
    global proc
    if proc is not None and proc.poll() is None:
        raise HTTPException(409, "already running")

    METRICS_PATH.unlink(missing_ok=True)
    state["steps"].clear()
    state["logs"].clear()
    state["samples"].clear()
    state["run_info"] = None
    state["overrides"] = req.overrides

    base = os.environ.get("SD_TRAINER_UI_CMD")
    cmd = (base.split() if base else [sys.executable, "main.py"]) + req.overrides
    env = {**os.environ, "SD_TRAINER_METRICS": str(METRICS_PATH), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,  # シグナルをプロセスグループ単位で送るため
    )
    started_at = time.time()
    for target, args in [(_stdout_reader, (proc,)), (_metrics_tail, (proc,)), (_image_watcher, (proc, started_at))]:
        threading.Thread(target=target, args=args, daemon=True).start()

    set_status("training")
    broadcast({"type": "log", "line": f"$ {' '.join(cmd)}"})
    return {"ok": True, "pid": proc.pid, "cmd": cmd}


def _signal_run(sig):
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "not running")
    os.killpg(os.getpgid(proc.pid), sig)


@app.post("/api/train/pause")
def train_pause():
    _signal_run(signal.SIGSTOP)
    set_status("paused")
    return {"ok": True}


@app.post("/api/train/resume")
def train_resume():
    _signal_run(signal.SIGCONT)
    set_status("training")
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop():
    if state["status"] == "paused":  # 停止シグナルを受け取れるように再開してから送る
        _signal_run(signal.SIGCONT)
    _signal_run(signal.SIGTERM)
    set_status("stopped")
    return {"ok": True}


def _signal_main_process(sig):
    # dataloaderのworker(デフォルトのSIGUSRで死ぬ)を巻き込まないよう、グループではなく本体だけに送る
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "not running")
    os.kill(proc.pid, sig)


@app.post("/api/train/sample")
def train_sample():
    _signal_main_process(signal.SIGUSR1)
    return {"ok": True}


@app.post("/api/train/save")
def train_save():
    _signal_main_process(signal.SIGUSR2)
    return {"ok": True}


@app.get("/api/status")
def status():
    return {
        "status": state["status"],
        "run_info": state["run_info"],
        "overrides": state["overrides"],
        "gpu": state["gpu"],
    }


def _scan_dataset(path, caption="captions", latent="latents", text_emb="text_emb"):
    # データセットディレクトリを走査してパイプラインの充足状況を返す。
    # サブディレクトリ名はBaseDatasetのデフォルトと同じ。
    d = Path(path)
    if not d.exists():
        raise HTTPException(404, f"{path} が見つからないよ")
    info = {"path": str(path), "dirs": {"caption": caption, "latent": latent, "text_emb": text_emb}}
    buckets_path = d / "buckets.json"
    if buckets_path.exists():
        with open(buckets_path) as f:
            buckets = json.load(f)
        info["buckets"] = {"num_buckets": len(buckets), "num_images": sum(len(v) for v in buckets.values())}
    info["images"] = len(list(d.glob("*.png")))
    for key, sub, pattern in [("captions", caption, "*.caption"), ("latents", latent, "*.npy"), ("text_emb", text_emb, "*.npz")]:
        p = d / sub  # 絶対パスを指定した場合はそちらが優先される(pathlibの/演算子の仕様)
        info[key] = len(list(p.glob(pattern))) if p.exists() else 0
    return info


@app.get("/api/dataset")
def dataset_info(path: str, caption: str = "captions", latent: str = "latents", text_emb: str = "text_emb"):
    return _scan_dataset(path, caption, latent, text_emb)


@app.get("/api/pipeline")
def pipeline_info(override: list[str] = Query(default=[])):
    # LAUNCHのオーバーライドでconfigを合成し、学習が実際に使うデータセットを走査する
    with _compose_lock:
        try:
            with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
                cfg = compose(config_name="config", overrides=override)
        except Exception as e:
            raise HTTPException(400, f"composeに失敗したよ: {e}")
    args = OmegaConf.to_container(cfg.dataset.args, resolve=True) if cfg.dataset.args is not None else {}
    path = args.get("path")
    if not path:
        raise HTTPException(400, "dataset.args.path が設定されてないよ")
    info = _scan_dataset(
        path,
        caption=args.get("caption", "captions") or "captions",
        latent=args.get("latent", "latents") or "latents",
        text_emb=args.get("text_emb") or "text_emb",
    )
    # configが実際にどのソースを使うか(BaseDatasetのデフォルトに準拠)
    info["uses"] = {
        "images": args.get("image") is not None,
        "latents": args.get("latent", "latents") is not None and args.get("image") is None,
        "text_emb": args.get("text_emb") is not None,
    }
    return info


@app.get("/api/config/groups")
def config_groups():
    # conf/配下の構成グループと選択肢(yamlファイル名)を返す
    out = {}
    for group in CONFIG_GROUPS:
        p = CONF_DIR / group
        out[group] = sorted(f.stem for f in p.glob("*.yaml")) if p.exists() else []
    return out


@app.get("/api/config")
def config_compose(override: list[str] = Query(default=[])):
    # 指定オーバーライドでHydra configを合成して返す(Config Editorのベースライン用)
    with _compose_lock:
        try:
            with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
                cfg = compose(config_name="config", overrides=override)
        except Exception as e:
            raise HTTPException(400, f"composeに失敗したよ: {e}")
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    await ws.send_json({
        "type": "snapshot",
        "status": state["status"],
        "run_info": state["run_info"],
        "overrides": state["overrides"],
        "steps": list(state["steps"])[-600:],
        "logs": list(state["logs"])[-100:],
        "samples": list(state["samples"]),
        "gpu": state["gpu"],
    })
    try:
        while True:
            await ws.receive_text()  # クライアントからのping等は読み捨て
    except WebSocketDisconnect:
        clients.discard(ws)


@app.on_event("startup")
async def on_startup():
    global loop
    loop = asyncio.get_running_loop()
    threading.Thread(target=_gpu_watcher, daemon=True).start()


IMAGE_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.mount("/", StaticFiles(directory=ROOT / "ui" / "static", html=True), name="static")
