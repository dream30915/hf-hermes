import io
import os
import subprocess
import unittest
from unittest import mock
from urllib.request import Request

from hf_hermes import cli


class CliTests(unittest.TestCase):
    def test_catalog_rejects_plain_http_before_sending_token(self):
        with self.assertRaisesRegex(cli.CatalogError, "HTTPS"):
            cli.fetch_catalog("hf_super_secret", "http://router.huggingface.co/v1")

    def test_catalog_rejects_base_url_with_embedded_credentials(self):
        with self.assertRaisesRegex(cli.CatalogError, "credentials"):
            cli.fetch_catalog(
                "hf_super_secret",
                "https://username:password@router.huggingface.co/v1",
            )

    def test_catalog_redirect_handler_never_forwards_authorization(self):
        request = Request(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": "Bearer hf_super_secret"},
        )

        redirected = cli._NoRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.example/models",
        )

        self.assertIsNone(redirected)

    def test_command_rendering_uses_windows_quoting_on_windows(self):
        rendered = cli._format_command(
            ["hermes", "chat", "--prompt", "it's complicated"],
            platform_name="nt",
        )

        self.assertEqual(rendered, 'hermes chat --prompt "it\'s complicated"')

    def test_catalog_only_exposes_live_providers(self):
        payload = io.BytesIO(
            b'{"data":[{"id":"org/model","providers":['
            b'{"provider":"working","status":"live"},'
            b'{"provider":"broken","status":"error"}]}]}'
        )

        with mock.patch.object(cli, "_open_catalog", return_value=payload):
            models = cli.fetch_catalog("hf_super_secret")

        self.assertEqual(models, [cli.CatalogModel("org/model", ("working",))])

    def test_model_launch_requires_token_before_resolving_hermes(self):
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--model", "Qwen/Qwen3.5-397B-A17B"],
            environ={},
            stdout=io.StringIO(),
            stderr=stderr,
            token_reader=lambda: None,
            executable_finder=lambda _: self.fail("Hermes must not be resolved"),
            runner=lambda *args, **kwargs: self.fail("Hermes must not be launched"),
        )

        self.assertEqual(exit_code, 3)
        self.assertIn("hf auth login", stderr.getvalue())

    def test_dry_run_builds_native_huggingface_command_without_exposing_token(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = cli.main(
            [
                "--model",
                "Qwen/Qwen3.5-397B-A17B",
                "--dry-run",
                "--",
                "--oneshot",
                "-q",
                "hello",
            ],
            environ={"HF_TOKEN": "hf_super_secret"},
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            "hermes chat --provider huggingface --model Qwen/Qwen3.5-397B-A17B --oneshot -q hello",
        )
        self.assertNotIn("hf_super_secret", stdout.getvalue() + stderr.getvalue())

    def test_executes_hermes_with_token_and_returns_child_exit_code(self):
        calls = []

        def fake_run(command, *, env, check):
            calls.append((command, env, check))
            return subprocess.CompletedProcess(command, 7)

        exit_code = cli.main(
            ["--model", "Qwen/Qwen3.5-397B-A17B", "--", "--quiet"],
            environ={"PATH": os.environ.get("PATH", "")},
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            token_reader=lambda: "hf_from_login",
            executable_finder=lambda _: "/opt/hermes/bin/hermes",
            runner=fake_run,
        )

        self.assertEqual(exit_code, 7)
        self.assertEqual(
            calls[0][0],
            [
                "/opt/hermes/bin/hermes",
                "chat",
                "--provider",
                "huggingface",
                "--model",
                "Qwen/Qwen3.5-397B-A17B",
                "--quiet",
            ],
        )
        self.assertEqual(calls[0][1]["HF_TOKEN"], "hf_from_login")
        self.assertFalse(calls[0][2])

    def test_list_models_prints_catalog_and_does_not_launch_hermes(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--list-models"],
            environ={"HF_TOKEN": "hf_catalog_token"},
            stdout=stdout,
            stderr=stderr,
            catalog_loader=lambda _: [
                cli.CatalogModel("Qwen/Qwen3.5-397B-A17B", ("cerebras", "groq")),
                cli.CatalogModel("deepseek-ai/DeepSeek-V3.2", ()),
            ],
            runner=lambda *args, **kwargs: self.fail("Hermes must not be launched"),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                "Qwen/Qwen3.5-397B-A17B\tcerebras,groq",
                "deepseek-ai/DeepSeek-V3.2\tauto",
            ],
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_interactive_selection_adds_concrete_hf_provider_suffix(self):
        answers = iter(["2", "2"])
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--dry-run"],
            environ={"HF_TOKEN": "hf_catalog_token"},
            stdout=stdout,
            stderr=stderr,
            catalog_loader=lambda _: [
                cli.CatalogModel("Qwen/Qwen3.5-397B-A17B", ("cerebras",)),
                cli.CatalogModel("deepseek-ai/DeepSeek-V3.2", ("fireworks-ai",)),
            ],
            input_fn=lambda _: next(answers),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            "hermes chat --provider huggingface --model deepseek-ai/DeepSeek-V3.2:fireworks-ai",
        )
        self.assertIn("Select a model", stderr.getvalue())
        self.assertIn("automatic", stderr.getvalue())

    def test_cancelling_provider_selection_does_not_launch_hermes(self):
        answers = iter(["1", "q"])
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--dry-run"],
            environ={"HF_TOKEN": "hf_catalog_token"},
            stdout=stdout,
            stderr=stderr,
            catalog_loader=lambda _: [
                cli.CatalogModel("Qwen/Qwen3.5-397B-A17B", ("cerebras",)),
            ],
            input_fn=lambda _: next(answers),
        )

        self.assertEqual(exit_code, 130)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Cancelled", stderr.getvalue())

    def test_list_models_requires_token_without_leaking_credentials(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--list-models"],
            environ={},
            stdout=stdout,
            stderr=stderr,
            token_reader=lambda: None,
        )

        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("hf auth login", stderr.getvalue())

    def test_missing_hermes_binary_returns_127(self):
        stderr = io.StringIO()

        exit_code = cli.main(
            ["--model", "Qwen/Qwen3.5-397B-A17B"],
            environ={"HF_TOKEN": "hf_secret"},
            stdout=io.StringIO(),
            stderr=stderr,
            executable_finder=lambda _: None,
        )

        self.assertEqual(exit_code, 127)
        self.assertIn("was not found", stderr.getvalue())
        self.assertNotIn("hf_secret", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
