# hf-hermes

[![CI](https://github.com/dream30915/hf-hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/dream30915/hf-hermes/actions/workflows/ci.yml)

An [`hf` CLI extension](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-extensions)
that launches [Hermes Agent](https://github.com/NousResearch/hermes-agent) with
[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers).

It uses Hermes' native `huggingface` provider. The extension can list the models
available to your Hugging Face account, lets you select a model and concrete
inference provider, and forwards the remaining arguments to `hermes chat`.

## Requirements

- Python 3.10 or newer
- The `hf` CLI
- Hermes Agent with the `huggingface` provider
- A Hugging Face token with **Make calls to Inference Providers** permission

Authenticate once:

```bash
hf auth login
```

The extension first checks `HF_TOKEN`, then falls back to the active token from
`hf auth token`. It never saves or prints the token.

## Install

```bash
hf extensions install dream30915/hf-hermes
```

## Use

Select a model and provider interactively:

```bash
hf hermes
```

Launch a specific model with automatic provider routing:

```bash
hf hermes --model Qwen/Qwen3.5-397B-A17B
```

Pin a concrete Hugging Face inference provider:

```bash
hf hermes \
  --model deepseek-ai/DeepSeek-V3.2 \
  --hf-provider fireworks-ai
```

Forward arguments to `hermes chat` after `--`:

```bash
hf hermes \
  --model Qwen/Qwen3.5-397B-A17B \
  -- --oneshot -q "Review this repository"
```

List the model catalog visible to your token:

```bash
hf hermes --list-models
```

Preview a platform-appropriate diagnostic rendering of the Hermes command
without launching it:

```bash
hf hermes --model Qwen/Qwen3.5-397B-A17B --dry-run -- --quiet
```

If `hermes` is not on `PATH`, provide its executable explicitly:

```bash
hf hermes --hermes-bin /path/to/hermes --model Qwen/Qwen3.5-397B-A17B
```

On Windows, quote a path containing spaces:

```powershell
hf hermes --hermes-bin "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" --model Qwen/Qwen3.5-397B-A17B
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Hugging Face access token; preferred over `hf auth token` |
| `HF_BASE_URL` | HTTPS Hugging Face router override; redirects and embedded credentials are rejected |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | Invalid/non-interactive invocation |
| `3` | No Hugging Face token available for catalog access |
| `4` | Hugging Face model catalog failure |
| `127` | Hermes executable could not be launched |
| other | Exit code returned by Hermes |

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## License

MIT
