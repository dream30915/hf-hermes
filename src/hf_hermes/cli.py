"""Command-line entry point for the ``hf hermes`` extension."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class CatalogModel:
    """One model returned by the HF OpenAI-compatible catalog."""

    model_id: str
    providers: tuple[str, ...] = ()


class CatalogError(RuntimeError):
    """Raised when the Hugging Face model catalog cannot be loaded."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse redirects so bearer credentials never cross origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _catalog_endpoint(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise CatalogError("HF_BASE_URL is not a valid URL.") from exc
    if parsed.scheme.lower() != "https":
        raise CatalogError("HF_BASE_URL must use HTTPS.")
    if not hostname:
        raise CatalogError("HF_BASE_URL must include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise CatalogError("HF_BASE_URL must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise CatalogError("HF_BASE_URL must not contain a query or fragment.")
    return f"{base_url.rstrip('/')}/models"


def _open_catalog(request: Request, timeout: int):
    return build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hf hermes",
        description="Launch Hermes Agent with Hugging Face Inference Providers.",
    )
    parser.add_argument("--model", help="Hugging Face model ID")
    parser.add_argument(
        "--hf-provider",
        help="Optional concrete HF inference provider (default: automatic routing)",
    )
    parser.add_argument(
        "--hermes-bin",
        default="hermes",
        help="Hermes executable to launch (default: hermes)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available to the active Hugging Face token",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Hermes command without executing it",
    )
    parser.add_argument(
        "hermes_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to `hermes chat`",
    )
    return parser


def _model_argument(model: str, provider: str | None) -> str:
    return f"{model}:{provider}" if provider else model


def _read_hf_token(environ: Mapping[str, str]) -> str | None:
    token = environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    hf = shutil.which("hf", path=environ.get("PATH"))
    if not hf:
        return None
    try:
        result = subprocess.run(
            [hf, "auth", "token"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def fetch_catalog(token: str, base_url: str = DEFAULT_HF_BASE_URL) -> list[CatalogModel]:
    """Fetch the account-visible Hugging Face inference model catalog."""
    request = Request(
        _catalog_endpoint(base_url),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with _open_catalog(request, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise CatalogError(
                "Hugging Face rejected the token. Run `hf auth login` with Inference Providers permission."
            ) from exc
        raise CatalogError(f"Hugging Face model catalog returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise CatalogError("Could not load the Hugging Face model catalog.") from exc

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise CatalogError("Hugging Face returned an invalid model catalog.")

    models: list[CatalogModel] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        providers: list[str] = []
        for item in row.get("providers") or []:
            name = item.get("provider") if isinstance(item, dict) else None
            status = item.get("status") if isinstance(item, dict) else None
            if (
                isinstance(name, str)
                and name
                and status == "live"
                and name not in providers
            ):
                providers.append(name)
        models.append(CatalogModel(row["id"], tuple(providers)))
    return models


def _format_command(command: Sequence[str], platform_name: str | None = None) -> str:
    """Render argv for diagnostics using the current platform's conventions."""
    if (platform_name or os.name) == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _choose(
    heading: str,
    options: Sequence[tuple[str, str | None]],
    input_fn: Callable[[str], str],
    stderr: TextIO,
) -> str | None:
    print(heading, file=stderr)
    for index, (label, _) in enumerate(options, start=1):
        print(f"  {index}. {label}", file=stderr)
    while True:
        try:
            answer = input_fn("Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][1]
        print(f"Enter a number from 1 to {len(options)}, or q to cancel.", file=stderr)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    token_reader: Callable[[], str | None] | None = None,
    catalog_loader: Callable[[str], list[CatalogModel]] | None = None,
    input_fn: Callable[[str], str] | None = None,
    executable_finder: Callable[[str], str | None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[object]] | None = None,
) -> int:
    """Run the extension and return its process exit code."""
    env = dict(os.environ if environ is None else environ)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)

    token = env.get("HF_TOKEN", "").strip()
    read_token = token_reader or (lambda: _read_hf_token(env))
    if not token:
        token = read_token() or ""
    if token:
        env["HF_TOKEN"] = token

    if not token:
        print(
            "Error: no Hugging Face token found. Run `hf auth login` or set HF_TOKEN.",
            file=err,
        )
        return 3

    load_catalog = catalog_loader or (
        lambda current_token: fetch_catalog(
            current_token, env.get("HF_BASE_URL", DEFAULT_HF_BASE_URL)
        )
    )

    models: list[CatalogModel] | None = None
    if args.list_models or not args.model:
        try:
            models = load_catalog(token)
        except CatalogError as exc:
            print(f"Error: {exc}", file=err)
            return 4

    if args.list_models:
        assert models is not None
        for model in models:
            print(
                f"{model.model_id}\t{','.join(model.providers) or 'auto'}",
                file=out,
            )
        return 0

    if not args.model:
        if input_fn is None and not sys.stdin.isatty():
            print("Error: --model is required when stdin is not interactive.", file=err)
            return 2
        assert models is not None
        if not models:
            print("Error: Hugging Face returned no available models.", file=err)
            return 4
        choose_input = input_fn or input
        selected_model = _choose(
            "Select a model:",
            [(model.model_id, model.model_id) for model in models],
            choose_input,
            err,
        )
        if not selected_model:
            print("Cancelled.", file=err)
            return 130
        args.model = selected_model
        selected = next(model for model in models if model.model_id == selected_model)
        if args.hf_provider is None and selected.providers:
            selected_provider = _choose(
                f"Select a provider for {selected_model}:",
                [("automatic", ""), *[(name, name) for name in selected.providers]],
                choose_input,
                err,
            )
            if selected_provider is None:
                print("Cancelled.", file=err)
                return 130
            args.hf_provider = selected_provider or None

    forwarded = list(args.hermes_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command = [
        args.hermes_bin,
        "chat",
        "--provider",
        "huggingface",
        "--model",
        _model_argument(args.model, args.hf_provider),
        *forwarded,
    ]
    if args.dry_run:
        print(_format_command(command), file=out)
        return 0

    finder = executable_finder or (lambda name: shutil.which(name, path=env.get("PATH")))
    executable = finder(args.hermes_bin)
    if not executable:
        print(
            f"Error: Hermes executable '{args.hermes_bin}' was not found in PATH. "
            "Install Hermes Agent or pass --hermes-bin PATH.",
            file=err,
        )
        return 127
    command[0] = executable

    run = runner or subprocess.run
    try:
        result = run(command, env=env, check=False)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: could not launch Hermes: {exc}", file=err)
        return 127
    return int(result.returncode)


def entrypoint() -> None:
    raise SystemExit(main(os.sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
