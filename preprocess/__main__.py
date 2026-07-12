# 前処理パイプラインの統一CLI。リポジトリ直下で `python -m preprocess <サブコマンド>` として実行する。
# 重い依存(torch/timm等)はサブコマンド実行時に遅延importする。
import argparse


def _cmd_buckets(args):
    from preprocess.buckets import bucket_images
    meta = bucket_images(
        args.input_dir, args.output_dir,
        resolution=args.resolution, min_length=args.min_length,
        max_length=args.max_length, max_ratio=args.max_ratio,
        num_workers=args.num_workers,
    )
    for key in meta:
        print(f"{key}: {len(meta[key])}個")


def _cmd_metadata(args):
    from preprocess.buckets import make_metadata
    meta = make_metadata(args.directory, num_workers=args.num_workers)
    for key in meta:
        print(f"{key}: {len(meta[key])}個")


def _cmd_original_size(args):
    from preprocess.buckets import make_original_size_metadata
    dic = make_original_size_metadata(args.metadata, args.output_path)
    print(f"{len(dic)}件書き出したよ")


def _cmd_latents(args):
    from preprocess.latents import encode_latents
    encode_latents(
        args.directory, args.output_dir, args.model,
        model_type=args.model_type, dtype=args.dtype,
        batch_size=args.batch_size, metadata=args.metadata,
        skip_existing=not args.no_skip_existing,
    )


def _cmd_text(args):
    from preprocess.text_emb import encode_text
    encode_text(
        args.directory, args.output_dir, args.model,
        model_type=args.model_type, dtype=args.dtype,
        batch_size=args.batch_size, clip_skip=args.clip_skip,
        save_dtype=args.save_dtype,
    )


def _cmd_tags(args):
    from preprocess.tagging import tag_images
    tag_images(
        args.directory, args.output_dir, caption_dir=args.caption_dir,
        repo_id=args.repo_id, batch_size=args.batch_size,
        threshold=args.threshold, character_threshold=args.character_threshold,
        extension=args.extension, num_workers=args.num_workers,
    )


def _cmd_masks(args):
    from preprocess.masks import create_face_masks
    create_face_masks(args.directory, args.output_dir, cascade_path=args.cascade, num_workers=args.num_workers)


def main():
    parser = argparse.ArgumentParser(prog="python -m preprocess", description="学習用データセットの前処理パイプライン")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("buckets", help="画像をアスペクト比bucketへリサイズ+クロップしbuckets.jsonを作る")
    p.add_argument("--input_dir", "-i", required=True, help="元のデータセット")
    p.add_argument("--output_dir", "-o", required=True, help="保存先のディレクトリ")
    p.add_argument("--resolution", "-r", type=int, default=1024, help="bucketの解像度(64の倍数を推奨)")
    p.add_argument("--min_length", type=int, default=512, help="bucketの最小長")
    p.add_argument("--max_length", type=int, default=2048, help="bucketの最大長")
    p.add_argument("--max_ratio", type=float, default=2.0, help="最大アスペクト比(逆数が最小アスペクト比)")
    p.add_argument("--num_workers", "-p", type=int, default=12)
    p.set_defaults(func=_cmd_buckets)

    p = sub.add_parser("metadata", help="リサイズ済みデータセットからbuckets.jsonだけを作り直す")
    p.add_argument("--directory", "-d", required=True)
    p.add_argument("--num_workers", "-p", type=int, default=8)
    p.set_defaults(func=_cmd_metadata)

    p = sub.add_parser("original-size", help="スクレイピングメタデータからSDXL size condition用jsonを作る")
    p.add_argument("--metadata", "-m", required=True, help="image_width/image_heightを含むjson")
    p.add_argument("--output_path", "-o", required=True)
    p.set_defaults(func=_cmd_original_size)

    p = sub.add_parser("latents", help="VAEでlatentキャッシュ(.npy)を作る")
    p.add_argument("--directory", "-d", required=True)
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--model", "-m", required=True, help="Diffusers形式のモデルパスかHubリポジトリ")
    p.add_argument("--model_type", "-t", default="sdxl")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch_size", "-b", type=int, default=8)
    p.add_argument("--metadata", default="buckets.json")
    p.add_argument("--no_skip_existing", action="store_true", help="既存の.npyも作り直す")
    p.set_defaults(func=_cmd_latents)

    p = sub.add_parser("text", help="テキストエンコーダ出力のキャッシュ(.npz)を作る")
    p.add_argument("--directory", "-d", required=True, help=".captionファイルのあるディレクトリ")
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--model_type", "-t", default="sdxl")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch_size", "-b", type=int, default=32)
    p.add_argument("--clip_skip", type=int, default=None, help="未指定ならモデルタイプのデフォルト")
    p.add_argument("--save_dtype", default="fp16", choices=["fp16", "fp32"])
    p.set_defaults(func=_cmd_text)

    p = sub.add_parser("tags", help="WD Taggerでタグ付けして.captionを作る")
    p.add_argument("--directory", "-d", required=True, help="入力画像のディレクトリ")
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--caption_dir", "-c", default=None, help="既存captionのディレクトリ(指定するとcaptionの後ろにタグを追記)")
    p.add_argument("--repo_id", "-r", default="SmilingWolf/wd-eva02-large-tagger-v3")
    p.add_argument("--batch_size", "-b", type=int, default=16)
    p.add_argument("--threshold", "-t", type=float, default=0.35, help="generalタグの閾値")
    p.add_argument("--character_threshold", type=float, default=0.75, help="characterタグの閾値")
    p.add_argument("--extension", default="caption")
    p.add_argument("--num_workers", type=int, default=4)
    p.set_defaults(func=_cmd_tags)

    p = sub.add_parser("masks", help="アニメ顔検出で学習用マスク(.npz)を作る")
    p.add_argument("--directory", "-d", required=True)
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--cascade", default="lbpcascade_animeface.xml")
    p.add_argument("--num_workers", "-p", type=int, default=12)
    p.set_defaults(func=_cmd_masks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
