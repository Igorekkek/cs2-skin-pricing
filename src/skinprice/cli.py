"""Command line interface: ``skinprice <command>``.

    skinprice data build [--db PATH]          SQLite -> parquet dataset
    skinprice data sample [--per-mille 50]     refresh data/sample from the dataset
    skinprice collect --items FILE             fetch sales history from CSFloat
    skinprice features build [--dataset DIR]   feature table with the time split
    skinprice train MODEL [--set k=v ...]      train, evaluate on test, store
    skinprice evaluate                         compare the latest trained models
    skinprice snapshot build                   serving assets for predict/serve
    skinprice predict --json FILE | --url URL | --name NAME [...]
    skinprice serve [--port 8000]              minimal web UI + JSON API
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from skinprice import paths


def _parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _overrides(pairs: list[str] | None) -> dict:
    result = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        result[key.strip()] = _parse_value(value.strip())
    return result


def cmd_data(args: argparse.Namespace) -> None:
    if args.action == "build":
        from skinprice.data.build import build_dataset

        db = Path(args.db) if args.db else paths.source_db()
        out = Path(args.out) if args.out else paths.workspace().dataset_dir
        print(json.dumps(build_dataset(db, out), indent=2))
    else:
        from skinprice.data.io import load_tables
        from skinprice.data.sample import write_sample

        manifest = write_sample(load_tables(args.dataset), paths.SAMPLE_DIR, args.per_mille)
        print(json.dumps(manifest, indent=2))


def cmd_features(args: argparse.Namespace) -> None:
    from skinprice.features.build import FeatureConfig, build_and_save

    config = FeatureConfig(max_sticker_reference_lag_days=args.max_sticker_lag)
    manifest = build_and_save(Path(args.dataset) if args.dataset else None, args.name, config)
    print(json.dumps({k: v for k, v in manifest.items() if k != "dataset"}, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> None:
    from skinprice.features.build import load_features
    from skinprice.training import train_and_evaluate

    result = train_and_evaluate(args.model, load_features(args.features), params=_overrides(args.set), version=args.version)
    print(json.dumps(result.metrics, indent=2))
    print(f"saved to {result.directory}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    import pandas as pd

    from skinprice.models import REGISTRY
    from skinprice.models.base import read_manifest
    from skinprice.research.experiment import to_markdown
    from skinprice.training import latest_dir

    rows = []
    for name in args.models or sorted(REGISTRY):
        try:
            manifest = read_manifest(latest_dir(name))
        except FileNotFoundError:
            continue
        m = manifest["metrics"]["test"]
        rows.append(
            {
                "model": name,
                "version": manifest["version"],
                "median_ape": m["median_ape"],
                "p90_ape": m["p90_ape"],
                "mae_usd": m["mae_usd"],
                "coverage": m.get("interval_coverage"),
                "size_mb": manifest["size_mb"],
            }
        )
    print(to_markdown(pd.DataFrame(rows)))


def cmd_snapshot(args: argparse.Namespace) -> None:
    from skinprice.inference.assets import build_serving_assets

    print(json.dumps(build_serving_assets(args.features, Path(args.dataset) if args.dataset else None), indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from skinprice.inference.predictor import Predictor

    if args.json:
        payload = {"listing": json.loads(Path(args.json).read_text(encoding="utf-8"))}
    elif args.url:
        payload = {"url": args.url}
    elif args.name:
        stickers = []
        for text in args.sticker or []:
            name, _, wear = text.partition("@")
            stickers.append({"name": name.strip(), "wear": wear.strip() or None})
        payload = {
            "manual": {
                "name": args.name,
                "float_value": args.float,
                "paint_seed": args.seed,
                "stickers": stickers,
                "keychains": args.charm or [],
                "price_usd": args.price,
            }
        }
    else:
        sys.exit("Give --json, --url or --name")
    predictor = Predictor.from_workspace(args.models)
    print(json.dumps(predictor.predict(payload), indent=2, ensure_ascii=False))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from skinprice.server.app import create_app

    uvicorn.run(create_app(models=args.models), host=args.host, port=args.port)


def cmd_collect(args: argparse.Namespace) -> None:
    from skinprice.data.collect import collect

    names = [line.strip() for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    db = Path(args.db) if args.db else paths.source_db()
    collect(names, db, delay_seconds=args.delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skinprice", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="build the dataset or the public sample")
    data.add_argument("action", choices=["build", "sample"])
    data.add_argument("--db", help="SQLite database (default: SKINPRICE_SOURCE_DB)")
    data.add_argument("--out", help="output directory (default: workspace/dataset)")
    data.add_argument("--dataset", help="dataset to sample from (default: workspace/dataset)")
    data.add_argument("--per-mille", type=int, default=50, help="share of skins kept in the sample")
    data.set_defaults(func=cmd_data)

    collect = sub.add_parser("collect", help="collect sales history from CSFloat")
    collect.add_argument("--items", required=True, help="text file with one market hash name per line")
    collect.add_argument("--db", help="SQLite database to append to")
    collect.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    collect.set_defaults(func=cmd_collect)

    features = sub.add_parser("features", help="build the feature table")
    features.add_argument("action", choices=["build"])
    features.add_argument("--dataset", help="dataset directory (e.g. data/sample)")
    features.add_argument("--name", default="features")
    features.add_argument("--max-sticker-lag", type=float, default=None, help="mask sticker prices updated N days after the sale")
    features.set_defaults(func=cmd_features)

    train = sub.add_parser("train", help="train and evaluate one model")
    train.add_argument("model")
    train.add_argument("--features", default="features")
    train.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a model parameter")
    train.add_argument("--version")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="test metrics of the latest trained models")
    evaluate.add_argument("models", nargs="*")
    evaluate.set_defaults(func=cmd_evaluate)

    snapshot = sub.add_parser("snapshot", help="build serving assets")
    snapshot.add_argument("action", choices=["build"])
    snapshot.add_argument("--features", default="features")
    snapshot.add_argument("--dataset")
    snapshot.set_defaults(func=cmd_snapshot)

    predict = sub.add_parser("predict", help="price one listing")
    predict.add_argument("--json", help="CSFloat listing JSON file")
    predict.add_argument("--url", help="CSFloat item link or listing id")
    predict.add_argument("--name", help="market hash name, e.g. 'AK-47 | Redline (Field-Tested)'")
    predict.add_argument("--float", type=float)
    predict.add_argument("--seed", type=int)
    predict.add_argument("--sticker", action="append", help="sticker name, optionally '@wear' (0..1)")
    predict.add_argument("--charm", action="append")
    predict.add_argument("--price", type=float, help="asking price in USD")
    predict.add_argument("--models", nargs="*")
    predict.set_defaults(func=cmd_predict)

    serve = sub.add_parser("serve", help="run the web UI and JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--models", nargs="*")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
