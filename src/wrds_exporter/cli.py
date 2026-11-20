from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from wrds_exporter.config import ConfigError, load_config
from wrds_exporter.connection import close_connection, connect_wrds
from wrds_exporter.exporter import WRDSExporter
from wrds_exporter.logging_config import configure_logging
from wrds_exporter.validation import validate_identifier


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"wrds-export: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrds-export",
        description="Stream WRDS PostgreSQL queries into partitioned Parquet files.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a configured WRDS export.")
    run_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        action="append",
        help="YAML config path. Repeat --config to run several exports with one WRDS connection.",
    )
    run_parser.add_argument("--force", action="store_true")
    run_parser.set_defaults(func=_run)

    inspect_parser = subparsers.add_parser(
        "inspect", help="List WRDS schema tables or describe a table."
    )
    inspect_parser.add_argument(
        "--schemas",
        action="store_true",
        help="List visible WRDS libraries/schemas for the current account.",
    )
    inspect_parser.add_argument("--schema")
    inspect_parser.add_argument("--table")
    inspect_parser.add_argument("--username-env", default="WRDS_USER")
    inspect_parser.set_defaults(func=_inspect)

    test_parser = subparsers.add_parser(
        "test-connection", help="Test WRDS connection and account permissions."
    )
    test_parser.add_argument("--config", type=Path)
    test_parser.add_argument("--username-env")
    test_parser.set_defaults(func=_test_connection)
    return parser


def _run(args: argparse.Namespace) -> int:
    configs = [load_config(path) for path in args.config]
    logger = configure_logging()
    username_envs = {config.connection.username_env for config in configs}
    if len(username_envs) != 1:
        raise ValueError("All configs in one run must use the same connection.username_env.")

    db = connect_wrds(configs[0].connection.username_env)
    totals = {"completed": 0, "skipped": 0, "failed": 0}
    try:
        for config in configs:
            summary = WRDSExporter(config, logger=logger).run(force=args.force, db=db)
            for key in totals:
                totals[key] += summary[key]
    finally:
        close_connection(db)

    if totals["failed"]:
        return 1
    print(
        "Export complete: "
        f"completed={totals['completed']} skipped={totals['skipped']} failed={totals['failed']}"
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    if not args.schemas and not args.schema:
        raise ValueError("inspect requires --schema, or use --schemas to list libraries.")
    schema = (
        validate_identifier(args.schema, label="schema")
        if args.schema is not None
        else None
    )
    table = validate_identifier(args.table, label="table") if args.table else None

    db = connect_wrds(args.username_env)
    try:
        if args.schemas:
            for name in db.list_libraries():
                print(name)
            return 0

        assert schema is not None
        if table:
            description = db.describe_table(library=schema, table=table)
            _print_table(description)
        else:
            for name in db.list_tables(library=schema):
                print(name)
    finally:
        close_connection(db)
    return 0


def _test_connection(args: argparse.Namespace) -> int:
    username_env = args.username_env
    if args.config and not username_env:
        username_env = load_config(args.config).connection.username_env
    db = connect_wrds(username_env or "WRDS_USER")
    try:
        libraries = db.list_libraries()
        print(f"WRDS connection OK. Visible libraries: {len(libraries)}")
    finally:
        close_connection(db)
    return 0


def _print_table(value: Any) -> None:
    to_string = getattr(value, "to_string", None)
    if callable(to_string):
        print(to_string(index=False))
    else:
        print(value)


if __name__ == "__main__":
    raise SystemExit(main())
