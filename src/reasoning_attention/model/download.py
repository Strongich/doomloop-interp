"""Download the model snapshot via the Hugging Face `hf` CLI.

Per spec we use the `hf` CLI rather than the Python `huggingface_hub` API. The
download is idempotent: `hf download` skips files already present in the cache
(or `local_dir`), so calling this repeatedly is cheap.
"""

from __future__ import annotations

import shutil
import subprocess

from reasoning_attention.config import ModelConfig


def download_model(config: ModelConfig | None = None) -> str:
    """Download `config.model_id` using `hf download`.

    Returns the path the snapshot was downloaded to (stdout of the CLI, which is
    the resolved local path).

    Raises:
        RuntimeError: if the `hf` CLI is not on PATH.
        subprocess.CalledProcessError: if the download fails.
    """
    config = config or ModelConfig()

    if shutil.which("hf") is None:
        raise RuntimeError(
            "The `hf` CLI was not found on PATH. Install it with:\n"
            "  curl -LsSf https://hf.co/cli/install.sh | bash -s"
        )

    cmd = ["hf", "download", config.model_id, "--revision", config.revision]
    if config.local_dir is not None:
        cmd += ["--local-dir", config.local_dir]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    # `hf download` prints the resolved local path on the last stdout line.
    path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return path or (config.local_dir or config.model_id)


if __name__ == "__main__":
    print(download_model())
