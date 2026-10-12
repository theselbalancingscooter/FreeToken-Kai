"""``ft daemon`` is upstream's supervisor; ``ft mgr`` is the same thing plus kai's web console.

The split is the point: a daemon serves none of the console routes, so upstream's contract is
untouched and everything this fork added lives under one command.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.app import build_app
from freetoken.daemon.logring import LogRing
from freetoken.daemon.profiles import ProfileStore

CONSOLE_ROUTES = ["/ui/env.json", "/profiles", "/models", "/host", "/engine/config", "/engine/external", "/engine/view/stats"]
UPSTREAM_ROUTES = ["/health", "/engine/status", "/engine/metrics", "/engine/health"]
# /engine/logs is an endless SSE stream: ask the route table for it rather than opening it
STREAMING_ROUTES = ["/engine/logs"]


class _Probe:
    """The console's read-only views proxy the serve; with none running they still must answer."""

    def health(self, port):
        return {"reachable": False, "status": "unreachable"}

    stats = health

    def get(self, path, port):
        return {"reachable": False, "status": "unreachable"}


class _Manager:
    def status(self):
        return {"running": False, "port": None}

    def current_pid(self):
        return None

    def serve_args(self):
        return []


def _client(tmp_path, console):
    app = build_app(
        manager=_Manager(), ring=LogRing(), probe=_Probe(), footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1), proxy_pool=ThreadPoolExecutor(1),
        profiles=ProfileStore(str(tmp_path / "profiles.json")) if console else None,
        console=console, console_cache_dir=str(tmp_path / "console"),
    )
    return TestClient(app)


@pytest.mark.parametrize("path", CONSOLE_ROUTES)
def test_daemon_does_not_serve_the_console(tmp_path, path):
    assert _client(tmp_path, console=False).get(path).status_code == 404


@pytest.mark.parametrize("path", CONSOLE_ROUTES)
def test_mgr_serves_the_console(tmp_path, path):
    assert _client(tmp_path, console=True).get(path).status_code != 404


@pytest.mark.parametrize("path", UPSTREAM_ROUTES)
def test_both_keep_the_upstream_routes(tmp_path, path):
    for console in (False, True):
        assert _client(tmp_path, console=console).get(path).status_code != 404


@pytest.mark.parametrize("path", STREAMING_ROUTES)
def test_both_keep_the_streaming_routes(tmp_path, path):
    for console in (False, True):
        paths = {r.path for r in _client(tmp_path, console=console).app.routes}
        assert path in paths


def test_the_page_knows_which_command_served_it(tmp_path):
    assert _client(tmp_path, console=True).get("/ui/env.json").json()["mode"] == "mgr"


def test_the_benchmark_routes_take_the_experts_choice(tmp_path):
    import json

    model = tmp_path / "Flash-Next-NVFP4"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (tmp_path / "tune").mkdir()
    (tmp_path / "tune" / "Flash-Next-NVFP4.json").write_text(json.dumps({"args": ["--pp-size", "2"], "finished": 1.0}))
    (tmp_path / "tune" / "Flash-Next-NVFP4.q2_0.json").write_text(json.dumps({"args": ["--moe-expert-gguf", "x.gguf"], "finished": 2.0}))
    c = _client(tmp_path, console=True)
    last = lambda q="": c.get(f"/tune/last?model={model}{q}").json()["result"]  # noqa: E731
    assert last()["finished"] == 1.0 and last("&experts=checkpoint")["finished"] == 1.0 and last("&experts=q2")["finished"] == 2.0
    assert c.get(f"/tune/pending?model={model}&experts=q2").json()["reason"] == "last_incomplete"
    # the recommendation follows the result measured with the experts it serves; here no GGUF sits
    # next to the checkpoint, so that is the checkpoint's, whichever is asked for
    for q in ("", "&experts=checkpoint", "&experts=q2"):
        r = c.get(f"/recommend?model={model}{q}")
        assert r.status_code == 200 and r.json()["flags"] == ["--pp-size", "2"], r.text
    # and the GGUF cannot be benchmarked where there is none (starting is an operation: from this PC)
    local = TestClient(c.app, base_url="http://127.0.0.1:1901", client=("127.0.0.1", 50000))
    r = local.post("/tune/start", json={"model": str(model), "experts": "q2"})
    assert r.status_code == 409 and r.json()["code"] == "no_q2_gguf"


def test_mgr_remembers_which_profile_the_server_was_started_from(tmp_path):
    """Two profiles with the same model and port (the checkpoint's experts and the Q2_0 GGUF's):
    starting one must not make the other look as if its server ran, edited."""

    class Running(_Manager):
        cfg = None

        def status(self):
            return {"running": self.cfg is not None, "model": self.cfg and self.cfg[0], "port": self.cfg and self.cfg[1]}

        def serve_args(self):
            return list(self.cfg[2]) if self.cfg else []

        def switch(self, model, port, args, force=False):
            self.cfg = (model, port, list(args))
            return {"pid": 1}

    model = tmp_path / "Flash-Next-NVFP4"
    model.mkdir()
    (model / "config.json").write_text("{}")
    m = Running()
    app = build_app(manager=m, ring=LogRing(), probe=_Probe(), footprint_fn=lambda pid: {}, lifecycle_pool=ThreadPoolExecutor(1),
                    proxy_pool=ThreadPoolExecutor(1), profiles=ProfileStore(str(tmp_path / "profiles.json")), console=True,
                    console_cache_dir=str(tmp_path / "console"))
    c = TestClient(app, base_url="http://127.0.0.1:1901", client=("127.0.0.1", 50000))
    assert c.put("/profiles/fast", json={"model": str(model), "port": 1919, "args": ["--moe-expert-gguf", "x.gguf"]}).status_code == 200
    assert c.put("/profiles/plain", json={"model": str(model), "port": 1919, "args": ["--pp-size", "2"]}).status_code == 200
    assert c.get("/engine/config").json()["profile"] is None
    assert c.post("/profiles/nope/start", json={}).status_code == 404
    assert c.post("/profiles/fast/start", json={}).status_code == 200
    assert m.cfg == (str(model), 1919, ["--moe-expert-gguf", "x.gguf"])
    cfg = c.get("/engine/config").json()
    assert cfg["profile"] == "fast" and cfg["args"] == ["--moe-expert-gguf", "x.gguf"]
    # the profile is edited: the server still runs what the start gave it, so it is still that profile's
    c.put("/profiles/fast", json={"model": str(model), "port": 1919, "args": ["--moe-expert-gguf", "x.gguf", "--spec-mtp", "5"]})
    assert c.get("/engine/config").json()["profile"] == "fast"
    # something else replaced the server (the benchmark, another client): nobody's until a profile starts it
    m.switch(str(model), 1919, ["--other"])
    assert c.get("/engine/config").json()["profile"] is None
    m.cfg = None
    assert c.get("/engine/config").json()["profile"] is None


def test_defaults_do_not_collide():
    from freetoken.daemon import server

    assert server.DEFAULT_PORT == 1900 and server.MGR_PORT == 1901
    assert server._default_state_dir(False) != server._default_state_dir(True)
