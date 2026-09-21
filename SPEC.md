# hf-hermes specification

## Goal

Provide a cross-platform Python extension named `hf-hermes` that appears as
`hf hermes` and launches Hermes Agent through its native Hugging Face inference
provider.

## Required behavior

1. The repository and console entry point are both named `hf-hermes`.
2. The package supports Python 3.10 and newer and has no runtime dependencies.
3. `hf hermes` can fetch the authenticated account's `/v1/models` catalog and
   interactively select a model and an optional concrete inference provider.
4. `--model` bypasses interactive model selection.
5. `--hf-provider` appends the provider suffix expected by the Hugging Face
   router.
6. Arguments after `--` are forwarded to `hermes chat` unchanged.
7. Authentication uses `HF_TOKEN`, falling back to `hf auth token`.
8. Tokens are never printed, persisted, or embedded in the command line.
9. `--list-models` prints an agent-friendly tab-separated catalog.
10. `--dry-run` prints a platform-appropriate diagnostic rendering of the
    Hermes argument vector without executing it.
11. The Hermes child process exit code is returned unchanged.
12. The package works on Windows, macOS, and Linux.

## Out of scope

- Installing or upgrading Hermes Agent.
- Persistently changing Hermes configuration.
- Storing Hugging Face credentials.
- Making paid inference calls during tests or CI.
