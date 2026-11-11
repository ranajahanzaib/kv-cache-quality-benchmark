from pathlib import Path

import httpx
import pytest

from kv_cache_quality_benchmark import runtime


def fake_server_binary(tmp_path: Path, flags: str | None = None) -> Path:
    binary = tmp_path / "llama-server"
    help_flags = flags or "--model --ctx-size --cache-type-k --cache-type-v --host --port --parallel --flash-attn --fit"
    binary.write_text("#!/usr/bin/env python3\nimport sys\n"
                      f"if '--version' in sys.argv: print('llama.cpp test build'); raise SystemExit(0)\n"
                      f"print({help_flags!r})\n")
    binary.chmod(0o755)
    return binary


def test_runtime_detection_inspects_real_help_output(tmp_path: Path):
    binary = fake_server_binary(tmp_path)
    info = runtime.inspect_runtime(str(binary))
    assert "--cache-type-k" in info["capabilities"]
    assert info["version"] == "llama.cpp test build"


def test_runtime_missing_binary_has_actionable_error():
    with pytest.raises(runtime.RuntimeErrorDetail, match="not found"):
        runtime.inspect_runtime("no-such-kv-cache-quality-benchmark-server")


def test_tokenize_supports_integer_and_object_wire_formats(monkeypatch):
    runner = runtime.LlamaServer.__new__(runtime.LlamaServer)
    runner.base_url, runner.timeout = "http://127.0.0.1:1234", 1

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"tokens": [{"id": 19, "piece": "a"}, 23]}

    def fake_post(url, **kwargs):
        assert url.endswith("/tokenize")
        assert kwargs["json"]["parse_special"] is True
        return Response()

    monkeypatch.setattr(runtime.httpx, "post", fake_post)
    assert runner.tokenize("hello") == [19, 23]


def test_template_and_completion_validate_response(monkeypatch):
    runner = runtime.LlamaServer.__new__(runtime.LlamaServer)
    runner.base_url, runner.timeout = "http://127.0.0.1:1234", 1

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    responses = [Response({"prompt": "<chat>exact</chat>"}),
                 Response({"content": "READY", "timings": {"prompt_ms": 1}})]
    monkeypatch.setattr(runtime.httpx, "post", lambda *args, **kwargs: responses.pop(0))
    assert runner.prepare_prompt("doc", "question") == "<chat>exact</chat>"
    assert runner.complete("<chat>exact</chat>", 3, seed=7)["content"] == "READY"
    monkeypatch.setattr(runtime.httpx, "post", lambda *args, **kwargs: Response({"oops": True}))
    with pytest.raises(runtime.RuntimeErrorDetail, match="omitted string"):
        runner.complete("prompt")


def test_server_requires_cli_capabilities_before_launch(tmp_path: Path):
    binary = fake_server_binary(tmp_path, "--model")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    server = runtime.LlamaServer(str(binary), model, context=1024, cache_k="f16", cache_v="f16")
    with pytest.raises(runtime.RuntimeErrorDetail, match="lacks required options"):
        server._args()


def test_server_startup_health_and_clean_shutdown(monkeypatch, tmp_path: Path):
    binary = fake_server_binary(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    created = []
    def fake_popen(args, **kwargs):
        process = Process()
        created.append((args, process))
        return process

    class Health:
        status_code = 200

    class LocalPort:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def bind(self, _address):
            return None

        def getsockname(self):
            return ("127.0.0.1", 43210)

    server = runtime.LlamaServer(str(binary), model, context=1024, cache_k="f16", cache_v="f16")
    monkeypatch.setattr(runtime.socket, "socket", LocalPort)
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime.httpx, "get", lambda *args, **kwargs: Health())
    server.start()
    assert created[0][0][created[0][0].index("--host") + 1] == "127.0.0.1"
    assert created[0][0][created[0][0].index("--port") + 1] != "0"
    server.stop()
    assert server.process is None


def test_readiness_timeout_cleans_up_process(monkeypatch, tmp_path: Path):
    binary = fake_server_binary(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    proc = Process()

    class LocalPort:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def bind(self, _address):
            return None

        def getsockname(self):
            return ("127.0.0.1", 43210)

    server = runtime.LlamaServer(str(binary), model, context=1024, cache_k="f16", cache_v="f16", timeout=0.05)
    monkeypatch.setattr(runtime.socket, "socket", LocalPort)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(runtime.httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("not ready")))
    with pytest.raises(runtime.RuntimeErrorDetail, match="readiness timed out"):
        server.start()
    assert server.process is None
    assert proc.returncode == 0
