from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--if-encoder-adapter", action="store_true", default=False)
    parser.add_argument("--if-mask-decoder-adapter", action="store_true", default=False)
    parser.add_argument("--encoder-adapter-depths", type=int, nargs="*", default=[])
    parser.add_argument("--decoder-adapt-depth", type=int, default=0)
    parser.add_argument("--thd", action="store_true", default=False)
    parser.add_argument("--encoder-depth-layer", type=int, nargs="*", default=[])
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--arch", default="vit_b")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-cls", type=int, default=1)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.encoder_adapter_depths is None:
        args.encoder_adapter_depths = []
    if args.encoder_depth_layer is None:
        args.encoder_depth_layer = []
    return args
