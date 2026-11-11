"""llama.cpp process discovery, capability inspection, and local HTTP calls."""
from __future__ import annotations

import hashlib
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, Self, cast

import httpx


class RuntimeErrorDetail(RuntimeError):
    """Actionable runtime or model validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_runtime(binary: str) -> dict[str, Any]:
    path = Path(binary).expanduser()
    if not path.is_file():
        import shutil
        found = shutil.which(binary)
        if not found:
            raise RuntimeErrorDetail(f"llama-server executable not found: {binary}")
        path = Path(found)
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10, check=False)
        help_result = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeErrorDetail(f"Could not inspect llama-server: {exc}") from exc
    output = result.stdout + result.stderr
    help_text = help_result.stdout + help_result.stderr
    if result.returncode != 0 and not output.strip():
        raise RuntimeErrorDetail(f"llama-server --version failed (exit {result.returncode})")
    flags = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", help_text))
    version_lower = output.lower()
    backends = [backend for backend in ("metal", "cuda", "vulkan", "sycl", "opencl")
                if backend in version_lower]
    return {"path": str(path.resolve()), "version": output.strip() or "version unavailable",
            "capabilities": sorted(flags), "backends": backends, "help": help_text}


class LlamaServer:
    def __init__(self, binary: str, model: Path, *, context: int, cache_k: str, cache_v: str,
                 gpu_layers: int = 99, flash_attn: str = "on", timeout: float = 60) -> None:
        self.runtime = inspect_runtime(binary)
        self.model = model.expanduser().resolve()
        if not self.model.is_file():
            raise RuntimeErrorDetail(f"GGUF model not found: {self.model}")
        self.model_sha256 = sha256_file(self.model)
        self.context, self.cache_k, self.cache_v = context, cache_k, cache_v
        self.gpu_layers, self.flash_attn, self.timeout = gpu_layers, flash_attn, timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None
        self.base_url = ""
        self.log: list[str] = []

    def _args(self) -> list[str]:
        caps = set(self.runtime["capabilities"])
        required = ["--model", "--ctx-size", "--cache-type-k", "--cache-type-v", "--host", "--port", "--parallel"]
        missing = [flag for flag in required if flag not in caps]
        if missing:
            raise RuntimeErrorDetail("llama-server lacks required options: " + ", ".join(missing))
        args = [self.runtime["path"], "--model", str(self.model), "--ctx-size", str(self.context),
                "--cache-type-k", self.cache_k, "--cache-type-v", self.cache_v,
                "--host", "127.0.0.1", "--port", "0", "--parallel", "1"]
        # Current llama.cpp supports port 0 in some versions; reserve a free port when required.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args[args.index("0", args.index("--port") + 1)] = str(port)
        self.base_url = f"http://127.0.0.1:{port}"
        if "--n-gpu-layers" in caps:
            args += ["--n-gpu-layers", str(self.gpu_layers)]
        if "--flash-attn" in caps:
            args += ["--flash-attn", self.flash_attn]
        for flag in ("--no-cache-prompt", "--no-context-shift", "--no-webui"):
            if flag in caps:
                args.append(flag)
        if "--cache-reuse" in caps:
            args += ["--cache-reuse", "0"]
        if "--cache-ram" in caps:
            args += ["--cache-ram", "0"]
        if "--kv-offload" in caps:
            args.append("--kv-offload")
        if "--fit" in caps:
            args += ["--fit", "off"]
        return args

    def start(self) -> None:
        args = self._args()
        # A file avoids deadlocking the child if startup logging fills an unread pipe.
        self._log_file = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - retained until stop()
        process = subprocess.Popen(args, stdout=self._log_file, stderr=subprocess.STDOUT)
        self.process = process
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self._log_file.seek(0)
                tail = self._log_file.read().decode(errors="replace")[-4000:]
                return_code = process.returncode
                self.stop()
                raise RuntimeErrorDetail(f"llama-server exited during startup ({return_code}):\n{tail}")
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise RuntimeErrorDetail(f"llama-server readiness timed out after {self.timeout:g}s")

    def tokenize(self, text: str) -> list[int]:
        return self._tokenize_response(text, with_pieces=False)[0]

    def tokenize_with_pieces(self, text: str) -> tuple[list[int], list[str]]:
        response = httpx.post(f"{self.base_url}/tokenize", json={"content": text, "add_special": False,
                            "parse_special": True, "with_pieces": True}, timeout=self.timeout)
        return self._tokens_from_response(response)

    def _tokenize_response(self, text: str, *, with_pieces: bool) -> tuple[list[int], list[str]]:
        response = httpx.post(f"{self.base_url}/tokenize", json={"content": text, "add_special": False,
                            "parse_special": True, "with_pieces": with_pieces}, timeout=self.timeout)
        return self._tokens_from_response(response)

    def _tokens_from_response(self, response: httpx.Response) -> tuple[list[int], list[str]]:
        response.raise_for_status()
        data = cast(dict[str, Any], response.json())
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeErrorDetail("/tokenize returned no token list")
        ids = [int(token["id"] if isinstance(token, dict) else token) for token in tokens]
        pieces: list[str] = []
        for token in tokens:
            piece = token.get("piece", "") if isinstance(token, dict) else ""
            pieces.append(bytes(piece).decode(errors="replace") if isinstance(piece, list) else str(piece))
        return ids, pieces

    def token_position(self, text: str, marker: str) -> int:
        marker_offset = text.find(marker)
        if marker_offset < 0:
            return 0
        try:
            ids, pieces = self.tokenize_with_pieces(text)
        except (httpx.HTTPError, RuntimeErrorDetail, ValueError):
            return len(self.tokenize(text[:marker_offset]))
        if not ids or "".join(pieces) != text:
            return len(self.tokenize(text[:marker_offset]))
        cursor = 0
        for index, piece in enumerate(pieces):
            end = cursor + len(piece)
            if end > marker_offset:
                return index
            cursor = end
        # Older or unusual tokenizers may return no pieces; use the exact serialized prefix.
        return len(self.tokenize(text[:marker_offset]))

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(f"{self.base_url}/apply-template", json={"messages": messages}, timeout=self.timeout)
        response.raise_for_status()
        prompt = response.json().get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeErrorDetail("/apply-template returned no serialized prompt")
        return prompt

    def prepare_prompt(self, document: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "Answer using only the supplied document. If the information is absent, answer NOT_FOUND. Return only the answer value."},
            {"role": "user", "content": f"DOCUMENT:\n{document}\n\nQUESTION:\n{question}"},
        ]
        return self.apply_template(messages)

    def complete(self, prompt: str, max_tokens: int = 16, *, seed: int = 1234,
                 temperature: float = 0) -> dict[str, Any]:
        response = httpx.post(f"{self.base_url}/completion", json={"prompt": prompt, "n_predict": max_tokens,
                            "temperature": temperature, "top_k": 1, "seed": seed, "cache_prompt": False,
                            "stream": False}, timeout=self.timeout)
        response.raise_for_status()
        data = cast(dict[str, Any], response.json())
        if not isinstance(data.get("content"), str):
            raise RuntimeErrorDetail("/completion response omitted string content")
        return data

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def read_logs(self) -> str:
        if self._log_file is None:
            return ""
        self._log_file.flush()
        self._log_file.seek(0)
        value = self._log_file.read().decode(errors="replace")
        self._log_file.seek(0, 2)
        return value

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def build_prompt(document: str, question: str) -> str:
    return ("Answer using only the supplied document. If the information is absent, answer NOT_FOUND. "
            "Return only the answer value.\n\nDOCUMENT:\n" + document + "\n\nQUESTION:\n" + question + "\n\nANSWER:")
