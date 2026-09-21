# Security

## Token handling

`hf-hermes` reads a token from `HF_TOKEN` or from `hf auth token`. It passes the
token to the Hermes child process through its environment. It does not write the
token to disk, include it in command previews, or intentionally log it.

Catalog requests require HTTPS. `HF_BASE_URL` values containing embedded
credentials, queries, or fragments are rejected, and HTTP redirects are not
followed, so the authorization header is never forwarded to another origin.

Use a fine-grained Hugging Face token with only the permissions you need. For
inference, enable **Make calls to Inference Providers**.

## Reporting a vulnerability

Please open a private security advisory in this repository. Do not include live
tokens, credentials, or other secrets in an issue, log, screenshot, or test case.
