"""T00b-only hooks: explicit trainer phases and fresh Unity FIFO RGB packets.

Copied to a per-run import overlay. Never patches shared installed packages.
METADATA_PATCH reuses cached image bytes and is explicitly NOT a new RGB frame.
"""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
import urllib.request

_lock = threading.RLock()
_installed = False
_active_requests = 0
_event_id = 0
_scope = {"phase": "initialization", "training_step": None, "validation_step": None}
_validation_stack = []
_writer = None
_controller_class = None
_runtime = None
_context = threading.local()


def phase_token():
    """Read the private scratch credential; only its path enters Ray metadata."""
    path = Path(os.environ["T00B_PHASE_TOKEN_FILE"])
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("Phase credential file must be owned by this user with mode 0600")
    value = path.read_text().strip()
    if len(value) != 48 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("Invalid phase credential file")
    return value


def _emit(kind, **fields):
    global _event_id, _writer
    if not _installed:
        raise RuntimeError("Fresh RGB instrumentation is not installed in the environment server")
    # The caller holds _lock: all phases and packets have one total event order.
    if _writer is None:
        _writer = open(os.environ["T00B_FRAME_EVENTS"], "x", buffering=1)
    _event_id += 1
    row = {"event_id": _event_id, "kind": kind, "time_ns": time.time_ns(),
           "monotonic_ns": time.monotonic_ns(), "server_pid": os.getpid(),
           "scope": dict(_scope), **fields}
    _writer.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _writer.flush()
    return row


def fifo_packet(server, field_name, body):
    """Called only on a real FIFO wire field, before cached files are reused."""
    if field_name not in ("RGB_IMAGE", "THIRD_PARTY_IMAGE", "METADATA_PATCH"):
        return
    action = None
    if server._last_action_message:
        try:
            action = json.loads(server._last_action_message).get("action")
        except (ValueError, TypeError, AttributeError):
            pass
    unity = getattr(server, "unity_proc", None)
    identity = getattr(server, "_t00b_gpu_identity", getattr(_context, "device", None))
    with _lock:
        _emit("rgb" if field_name != "METADATA_PATCH" else "cached_metadata",
              field_type=field_name, payload_bytes=len(body),
              payload_sha256=hashlib.sha256(body).hexdigest(),
              stream_id=hashlib.sha256(server.server_pipe_path.encode()).hexdigest()[:24],
              unity_pid=getattr(unity, "pid", None), sequence_id=server.sequence_id,
              action=action, active_requests=_active_requests, gpu_identity=identity)


def install_server(app):
    """Install a job-token-protected phase barrier in this private Flask server."""
    from flask import g, jsonify, request
    global _installed
    if _installed:
        raise RuntimeError("T00b server hooks installed more than once")
    _installed = True
    with _lock:
        _emit("session_start", hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

    @app.before_request
    def request_begin():
        global _active_requests
        g.t00b_active = not request.path.startswith("/t00b/") and request.path != "/health"
        if g.t00b_active:
            with _lock:
                _active_requests += 1

    @app.teardown_request
    def request_end(error):
        global _active_requests
        if getattr(g, "t00b_active", False):
            with _lock:
                g.t00b_active = False
                _active_requests -= 1
                if error is not None:
                    _emit("request_error", error=f"{type(error).__name__}: {error}")

    @app.route("/t00b/phase", methods=["POST"])
    def phase_endpoint():
        global _scope
        if request.headers.get("X-T00b-Token") != phase_token():
            return jsonify({"error": "wrong job token"}), 403
        data = request.get_json(force=True)
        operation, step = data.get("operation"), data.get("step")
        with _lock:
            if _active_requests:
                # A response can be sent just before Flask teardown runs. The
                # client retries only this busy barrier; no ambiguous transition.
                return jsonify({"error": "environment requests still active"}), 409
            previous = dict(_scope)
            if operation == "train_begin":
                if type(step) is not int or step < 1 or _scope["training_step"] is not None or _validation_stack:
                    return jsonify({"error": "invalid train_begin"}), 400
                _scope = {"phase": "training", "training_step": step, "validation_step": None}
            elif operation == "train_end":
                if _scope != {"phase": "training", "training_step": step, "validation_step": None} or _validation_stack:
                    return jsonify({"error": "invalid train_end"}), 400
                _scope = {"phase": "idle", "training_step": None, "validation_step": None}
            elif operation == "validation_begin":
                if type(step) is not int or step < 0:
                    return jsonify({"error": "invalid validation step"}), 400
                _validation_stack.append(previous)
                _scope = {"phase": "validation", "training_step": previous["training_step"], "validation_step": step}
            elif operation == "validation_end":
                if not _validation_stack or _scope["validation_step"] != step:
                    return jsonify({"error": "invalid validation_end"}), 400
                _scope = _validation_stack.pop()
            else:
                return jsonify({"error": "unknown phase operation"}), 400
            row = _emit("phase", operation=operation, step=step, previous_scope=previous,
                        failed=bool(data.get("failed", False)))
            return jsonify({"ok": True, "event_id": row["event_id"], "scope": _scope})

    @app.route("/t00b/status", methods=["GET"])
    def status_endpoint():
        if request.headers.get("X-T00b-Token") != phase_token():
            return jsonify({"error": "wrong job token"}), 403
        with _lock:
            return jsonify({"ok": True, "event_id": _event_id, "scope": _scope,
                            "active_requests": _active_requests,
                            "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def phase(operation, step, failed=False):
    """Synchronous barrier called from the one trainer driver, not log parsing."""
    import urllib.error
    payload = json.dumps({"operation": operation, "step": int(step), "failed": failed}).encode()
    request = urllib.request.Request(os.environ["T00B_SERVER_URL"] + "/t00b/phase", data=payload,
        headers={"Content-Type": "application/json", "X-T00b-Token": phase_token()})
    deadline = time.monotonic() + 30
    while True:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if not result.get("ok"):
                raise RuntimeError("Environment server rejected instrumentation phase")
            return result
        except urllib.error.HTTPError as error:
            if error.code != 409 or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def validation_phase(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        step = int(self.global_steps)
        phase("validation_begin", step)
        failed = True
        try:
            result = function(self, *args, **kwargs)
            failed = False
            return result
        finally:
            phase("validation_end", step, failed=failed)
    return wrapped


def navigation_controller(**kwargs):
    """Use the allocation adapter; never touch HOME or shared mapping caches."""
    global _controller_class, _runtime
    with _lock:
        if _controller_class is None:
            from src.envs.thor_allocation import make_controller_class
            _runtime = json.loads(Path(os.environ["T00B_RUNTIME_JSON"]).read_text())
            _controller_class = make_controller_class(_runtime, role="navigation")
    device = next(row for row in _runtime["devices"] if row["logical_index"] == kwargs.get("gpu_device", 0))
    _context.device = device
    try:
        controller = _controller_class(**kwargs)
        controller.server._t00b_gpu_identity = device
        return controller
    finally:
        del _context.device


def ray_environment():
    """Explicitly carry instrumentation into Ray without overriding GPU masks."""
    result = {key: os.environ[key] for key in ("T00B_SERVER_URL", "T00B_PHASE_TOKEN_FILE")}
    result["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return result
