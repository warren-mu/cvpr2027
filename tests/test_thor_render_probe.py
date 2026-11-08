"""CPU fixtures for probe transport compatibility; never start Unity or a GPU."""
import ast
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from src.envs.thor_render_probe import capture_event, observe_fifo_receives


class ProbeTransportTests(unittest.TestCase):
    def test_probe_constructor_keeps_default_server_class(self):
        path = Path(__file__).resolve().parents[1] / "src/envs/thor_render_probe.py"
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "__init__"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "controller"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("server_class", [argument.arg for argument in calls[0].keywords])
        self.assertFalse(any(isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id == "FifoServer" for base in node.bases)
            for node in ast.walk(tree)))

    def test_installed_pinned_build_server_requires_exact_class_identity(self):
        # Read the installed source WITHOUT importing ai2thor or any graphics
        # library. Execute only the extracted constructor-selection method with
        # standard-library fake classes; no real server is ever constructed.
        try:
            distribution = importlib.metadata.distribution("ai2thor")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("ai2thor source is not installed; run this check in the pinned environment")
        self.assertEqual(distribution.version, "5.0.0")
        path = Path(distribution.locate_file("ai2thor/controller.py"))
        controller_node = next(node for node in ast.parse(path.read_text()).body
                               if isinstance(node, ast.ClassDef) and node.name == "Controller")
        method = next(node for node in controller_node.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_build_server")

        class FakeFifo:
            server_type = "FIFO"
            def __init__(self, **kwargs):
                self.parameters = kwargs
            def receive(self, timeout=None):
                return "fixture_event"

        class FakeWsgi(FakeFifo):
            server_type = "WSGI"

        class UnsupportedSubclass(FakeFifo):
            pass

        namespace = {"ai2thor": SimpleNamespace(fifo_server=SimpleNamespace(FifoServer=FakeFifo),
                      wsgi_server=SimpleNamespace(WsgiServer=FakeWsgi))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        def controller(server_class):
            return SimpleNamespace(server=None, server_class=server_class, server_timeout=1,
                depth_format="fixture", add_depth_noise=False,
                _build=SimpleNamespace(server_types=["FIFO", "WSGI"], url="fixture"))

        invalid = controller(UnsupportedSubclass)
        namespace["_build_server"](invalid, "localhost", 0, 2, 2)
        self.assertIsNone(invalid.server)  # Reproduces the old probe's blocker.
        seen = []
        with observe_fifo_receives(FakeFifo, lambda server, event, previous: seen.append(event)):
            valid = controller(FakeFifo)
            namespace["_build_server"](valid, "localhost", 0, 2, 2)
            self.assertIs(type(valid.server), FakeFifo)
            self.assertEqual(valid.server.receive(), "fixture_event")
        self.assertEqual(seen, ["fixture_event"])

    def test_wrapper_preserves_events_timeouts_and_cached_file_identity(self):
        event = SimpleNamespace(metadata={"lastActionSuccess": True})
        old_files, new_files = {"image": [b"old"]}, {"image": [b"new"]}
        class FakeServer:
            def __init__(self):
                self.raw_files = old_files
                self.calls = []
            def receive(self, timeout=None):
                self.calls.append(timeout)
                self.raw_files = new_files
                return event
        original = FakeServer.receive
        observations = []
        server = FakeServer()
        with observe_fifo_receives(FakeServer, lambda server, result, previous:
                                   observations.append((result, previous, server.raw_files))):
            self.assertIs(server.receive(timeout=3), event)
            self.assertIs(server.receive(timeout=None), event)
        self.assertIs(FakeServer.receive, original)
        self.assertEqual(server.calls, [3, None])
        self.assertIs(observations[0][1], old_files)
        self.assertIs(observations[0][2], new_files)
        self.assertIs(observations[1][1], new_files)

    def test_method_restored_when_original_observer_or_context_fails(self):
        for failure in ("receive", "observer", "context"):
            with self.subTest(failure=failure):
                class FakeServer:
                    def receive(self, timeout=None):
                        if failure == "receive":
                            raise RuntimeError("fixture failure")
                        return object()
                original = FakeServer.receive
                def observer(server, event, previous):
                    if failure == "observer":
                        raise RuntimeError("fixture failure")
                with self.assertRaises(RuntimeError):
                    with observe_fifo_receives(FakeServer, observer):
                        FakeServer().receive()
                        if failure == "context":
                            raise RuntimeError("fixture failure")
                self.assertIs(FakeServer.receive, original)

    def test_receive_wrapper_really_saves_black_fixture_frame(self):
        import numpy as np
        from PIL import Image
        event = SimpleNamespace(metadata={"lastAction": "Pass", "lastActionSuccess": True},
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
            instance_segmentation_frame=np.zeros((2, 2, 3), dtype=np.uint8),
            instance_masks={"fixture": np.array([[True, False], [False, False]])})
        class FakeServer:
            def receive(self, timeout=None):
                return event
        with tempfile.TemporaryDirectory(prefix="thor_probe_fixture_") as temporary:
            observations = []
            with observe_fifo_receives(FakeServer, lambda server, value, previous:
                    observations.append(capture_event(value, Path(temporary), 0, "fixture", {"action": "Pass"}))):
                self.assertIs(FakeServer().receive(), event)
            row = observations[0]
            self.assertEqual(row["rgb"]["nonzero_values"], 0)
            self.assertEqual(row["rgb"]["min"], 0)
            self.assertEqual(row["rgb"]["max"], 0)
            self.assertTrue(Path(row["rgb"]["png_path"]).is_file())
            with Image.open(row["rgb"]["png_path"]) as image:
                np.testing.assert_array_equal(np.asarray(image), event.frame)
            with np.load(row["raw_arrays_path"]) as arrays:
                np.testing.assert_array_equal(arrays["rgb"], event.frame)
            self.assertEqual(json.loads(Path(row["metadata_path"]).read_text()), event.metadata)
            self.assertEqual(row["instance_masks"]["pixels_by_id"], {"fixture": 1})


if __name__ == "__main__":
    unittest.main()
