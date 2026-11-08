"""Explicit scripted CPU fixtures. This is neither the L nor the P backend."""
from dataclasses import replace
import uuid

from src.envs.paired_env import BackendFrame, parse_action


class MockObservationBackend:
    """Each world needs its own frames; no opposite-world fallback or pixel reuse.

    frames: {(pair_id, world_id, state_id): BackendFrame}
    starts: {(pair_id, world_id): state_id}
    transitions: {(pair_id, world_id, state_id, canonical_action): next_state_id}
    Fixture RGB, labels and evidence remain synthetic even when exact equality
    holds. Tests of a real backend will need independent render provenance.
    """
    def __init__(self, frames, starts, transitions):
        self.frames, self.starts, self.transitions = dict(frames), dict(starts), dict(transitions)
        for key, frame in self.frames.items():
            if (not isinstance(frame, BackendFrame) or key != (frame.capture.pair_id, frame.capture.world_id, frame.capture.state_id)
                    or not frame.capture.synthetic or frame.capture.backend_kind != "mock"):
                raise ValueError("Mock backend accepts only explicitly synthetic, exactly keyed frames")
        for pair_world, state in self.starts.items():
            if (*pair_world, state) not in self.frames:
                raise ValueError("Missing independently supplied world start")
        for (pair, world, state, action), following in self.transitions.items():
            if ((pair, world, state) not in self.frames or (pair, world, following) not in self.frames
                    or parse_action(action) not in self.frames[(pair, world, state)].observation_actions):
                raise ValueError("Invalid scripted observation transition")
        self.key = None
        self.closed = False
        self.requests = []

    def _capture(self):
        frame = self.frames[self.key]
        capture = replace(frame.capture, capture_id="mock-capture-" + uuid.uuid4().hex)
        comparison = frame.evidence
        if comparison is not None:
            side = "left_capture" if comparison.left_capture == frame.capture else "right_capture"
            comparison = replace(comparison, **{side: capture})
        return replace(frame, capture=capture, evidence=comparison)

    def reset(self, pair_id, world):
        if self.closed:
            raise RuntimeError("Mock backend is closed")
        self.key = (pair_id, world, self.starts[(pair_id, world)])
        self.requests.append(("reset", pair_id, world))
        return self._capture()

    def step(self, action):
        if self.closed or self.key is None:
            raise RuntimeError("Call reset on an open mock backend first")
        action = parse_action(action)
        following = self.transitions[(*self.key, action.text())]
        self.requests.append(("step", *self.key, action.text()))
        self.key = (*self.key[:2], following)
        return self._capture()

    def verify_capture(self, frame):
        return False  # Synthetic IDs can never attest to production rendering.

    def verify_evidence(self, comparison):
        return False

    def close(self):
        self.closed = True


def production_backend(kind, **kwargs):
    """No implicit/mock fallback while D1 and real backend work are pending."""
    if kind not in ("L", "P"):
        raise ValueError("Production backend must be explicitly selected by D1")
    raise NotImplementedError("D1-dependent " + kind + " backend is not implemented")
