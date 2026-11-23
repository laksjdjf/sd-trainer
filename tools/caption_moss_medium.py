#!/usr/bin/env python3
"""Write one medium MOSS-style caption per image into a separate caption folder.

Uses an OpenAI-compatible VLM server (llama.cpp). Existing tag captions are sent
as a reference and are not copied into the output. Each output file is only the
medium paragraph, which this trainer reads as a single .caption.
"""
import argparse
import base64
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

PROMPT = """You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "medium": "a one-paragraph image description"
  }
}

Rules:
- caption.medium is one paragraph describing the composition, key subjects, actions, setting, colors, lighting, and artistic style.
- Include only information visible in the image.
- Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
- Do not describe watermarks.
- The medium field must be a non-empty string.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations.
Here is the original caption reference:
{{reference}}
"""


def parse_args():
    p = argparse.ArgumentParser(description="Create medium-only captions for MOSS training.")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="directory of existing .caption files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None, help="buckets.json: caption only listed stems")
    p.add_argument("--knowledge", default="", help="world knowledge appended to the prompt for all images")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="a")
    p.add_argument("--max-image-size", type=int, default=1024)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=4, help="concurrent requests; match llama-server -np")
    p.add_argument("--limit", type=int, default=0, help="caption at most this many missing images; 0 means all")
    return p.parse_args()


def encode_image(path: Path, max_size: int, quality: int) -> str:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if max_size > 0 and max(w, h) > max_size:
            scale = max_size / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(content: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the response")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("the response is not a JSON object")
    medium = obj.get("caption", {}).get("medium") if isinstance(obj.get("caption"), dict) else None
    if not isinstance(medium, str) or not medium.strip():
        raise ValueError("caption.medium is missing or empty")
    return obj


def chat(args, prompt: str, image_url: str) -> dict:
    body = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def build_prompt(reference: str, knowledge: str) -> str:
    prompt = PROMPT.replace("{{reference}}", reference or "None.")
    if knowledge.strip():
        block = (
            "Here is additional knowledge about the image. Use it only to identify what is visible "
            "and refer by these names in the caption:\n"
            f"{knowledge.strip()}\n\n"
        )
        prompt = prompt.replace(
            "Here is the original caption reference:",
            block + "Here is the original caption reference:",
        )
    return prompt


def list_images(images_dir: Path, metadata: Path | None) -> list[Path]:
    if metadata is None:
        return sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    buckets = json.loads(metadata.read_text(encoding="utf-8"))
    stems = [stem for samples in buckets.values() for stem in samples]
    images = []
    missing = 0
    for stem in stems:
        path = images_dir / f"{stem}.png"
        if path.is_file():
            images.append(path)
        else:
            missing += 1
    if missing:
        print(f"warning: {missing} metadata entries have no .png in {images_dir}", flush=True)
    return sorted(images)


def caption_one(args, image: Path, reference: str) -> str:
    prompt = build_prompt(reference, args.knowledge)
    image_url = encode_image(image, args.max_image_size, args.jpeg_quality)
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            response = chat(args, prompt, image_url)
            choice = response["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if not content:
                raise ValueError(f"empty content (finish_reason={choice.get('finish_reason')})")
            return extract_json(content)["caption"]["medium"].strip()
        except Exception as e:
            last = e
            print(f"  attempt {attempt} failed: {e}", flush=True)
            time.sleep(1)
    raise RuntimeError(last)


def main():
    args = parse_args()
    images = list_images(args.images, args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [p for p in images if not (args.output / f"{p.stem}.caption").is_file()]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"{len(images)} images, {len(pending)} to caption -> {args.output} ({args.workers} at a time)", flush=True)
    lock = threading.Lock()
    done = {"n": 0, "failed": 0}

    def work(image: Path) -> None:
        ref_path = args.reference / f"{image.stem}.caption"
        reference = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else ""
        started = time.time()
        try:
            medium = caption_one(args, image, reference)
        except Exception as e:
            with lock:
                done["n"] += 1
                done["failed"] += 1
                print(f"[{done['n']}/{len(pending)}] {image.name} FAILED: {e}", flush=True)
            return
        out = args.output / f"{image.stem}.caption"
        tmp = out.with_suffix(".caption.tmp")
        tmp.write_text(medium + "\n", encoding="utf-8")
        tmp.replace(out)
        with lock:
            done["n"] += 1
            print(f"[{done['n']}/{len(pending)}] {image.name} {time.time() - started:.1f}s  {medium[:160]}", flush=True)

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, image) for image in pending]
        for future in as_completed(futures):
            future.result()
    print(f"done, failed {done['failed']}", flush=True)
    return 1 if done["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
