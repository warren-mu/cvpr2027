"""Pure T01 schemas and deterministic planning primitives.

There are no simulator imports, file writes, or implicit experiment settings.
Scene membership, seeds, relation geometry policy, tie tolerance, path evidence,
and movement-heading semantics are supplied by the caller. A graph witness is
only a candidate: replay, collision/penetration checks, rendered prefix equality,
and the remaining METRICS_SPEC validity checks belong to the observation backend.
Distances between navigation points use 3-D Euclidean metres. RGB visibility is
represented by nonempty instance-mask pixels, never metadata's ``visible`` flag.
"""
from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, order=True)
class Point3:
    x: float
    y: float
    z: float

    def __post_init__(self):
        for axis in ("x", "y", "z"):
            _finite(getattr(self, axis), axis)

    def distance(self, other):
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    name: str
    object_type: str
    position: Point3
    rotation: Point3
    moveable: bool = False
    pickupable: bool = False
    openable: bool = False
    receptacle: bool = False

    def __post_init__(self):
        for field in ("object_id", "name", "object_type"):
            _nonempty(getattr(self, field), field)
        if not isinstance(self.position, Point3) or not isinstance(self.rotation, Point3):
            raise ValueError("Object position and rotation must be Point3 values")
        for field in ("moveable", "pickupable", "openable", "receptacle"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a bool")


@dataclass(frozen=True)
class WorldSnapshot:
    """Object inventory/poses, not a claim to serialize Unity's entire state.

    ``world_key`` must identify this particular realized world, not just its
    scene. The adapter must use a fresh key after any relevant state change.
    """
    scene: str
    world_key: str
    objects: Tuple[ObjectState, ...]
    complete_object_inventory: bool = False

    def __post_init__(self):
        _nonempty(self.scene, "scene")
        _nonempty(self.world_key, "world_key")
        if not isinstance(self.objects, tuple) or not all(isinstance(o, ObjectState) for o in self.objects):
            raise ValueError("objects must be a tuple of ObjectState records")
        for attribute in ("object_id", "name"):
            if len({getattr(o, attribute) for o in self.objects}) != len(self.objects):
                raise ValueError(f"Duplicate object {attribute}")
        if type(self.complete_object_inventory) is not bool:
            raise ValueError("complete_object_inventory must be a bool")

    def full_object_poses(self):
        """Export all moveable OR pickupable poses from a complete inventory.

        This checks inventory bookkeeping, not absence of physical penetration.
        """
        if not self.complete_object_inventory:
            raise ValueError("Cannot export full poses from an incomplete object inventory")
        return [{"objectName": o.name, "position": o.position.as_dict(), "rotation": o.rotation.as_dict()}
                for o in sorted(self.objects, key=lambda obj: obj.name) if o.moveable or o.pickupable]


@dataclass(frozen=True)
class RelationalTask:
    target_type: str
    relation: str
    anchor_type: str

    def __post_init__(self):
        _nonempty(self.target_type, "target_type")
        _nonempty(self.anchor_type, "anchor_type")
        if self.relation not in ("next to", "in", "on"):
            raise ValueError("relation must be 'next to', 'in', or 'on'")

    def instruction(self, type_names: Mapping[str, str]):
        """Use explicit public category names, never a privileged target ID."""
        target, anchor = type_names[self.target_type], type_names[self.anchor_type]
        _nonempty(target, "target category name")
        _nonempty(anchor, "anchor category name")
        return f"Select the {target} {self.relation} the {anchor}."


def make_scene_split(scene_rooms: Mapping[str, str], test_scenes: Iterable[str]):
    """Record an explicit whole-scene split; do not choose a split ratio/seed."""
    rooms = dict(scene_rooms)
    test = tuple(sorted(test_scenes))
    if not rooms or len(test) != len(set(test)):
        raise ValueError("Scene inventory must be nonempty and test scenes unique")
    for scene, room in rooms.items():
        _nonempty(scene, "scene")
        if room not in ("kitchen", "living_room"):
            raise ValueError(f"Unsupported T01 room type for {scene}: {room}")
    if not set(test) <= set(rooms):
        raise ValueError("Unknown held-out scene")
    train = tuple(sorted(set(rooms) - set(test)))
    if not train or not test:
        raise ValueError("Both train and held-out test scenes are required")
    return {"policy": "explicit whole-scene holdout", "train_scenes": list(train),
            "test_scenes": list(test), "test_fraction": len(test) / len(rooms),
            "scene_rooms": {scene: rooms[scene] for scene in sorted(rooms)}}


@dataclass(frozen=True)
class ViewpointSample:
    points: Tuple[Point3, ...]
    requested_count: int
    seed: int
    minimum_spacing_m: float
    complete: bool
    reason: str


def farthest_point_sample(reachable: Iterable[Point3], *, count: int, seed: int,
                         minimum_spacing_m: float = 1.0):
    """Seeded first point, then farthest nearest-neighbor distance, stable ties.

    Input order and duplicate points do not affect the result. Insufficient
    spacing returns an explicitly incomplete sample; it never pads the set.
    GetReachablePositions provenance must be retained by the caller.
    """
    if type(count) is not int or not 12 <= count <= 20:
        raise ValueError("T01 requests 12–20 viewpoints")
    if type(seed) is not int:
        raise ValueError("An explicit integer seed is required")
    _finite(minimum_spacing_m, "minimum_spacing_m", 1.0)
    points = tuple(reachable)
    if not all(isinstance(point, Point3) for point in points):
        raise ValueError("Reachable points must be Point3 values")
    remaining = sorted(set(points))
    selected = []
    if remaining:
        selected.append(remaining.pop(random.Random(seed).randrange(len(remaining))))
    while remaining and len(selected) < count:
        distances = [min(point.distance(chosen) for chosen in selected) for point in remaining]
        # remaining is sorted, so equal distances have a geometry-only tie break.
        index = max(range(len(remaining)), key=lambda i: distances[i])
        if distances[index] < minimum_spacing_m:
            break
        selected.append(remaining.pop(index))
    complete = len(selected) == count
    return ViewpointSample(tuple(selected), count, seed, minimum_spacing_m, complete,
                           "complete" if complete else "insufficient_reachable_points_at_required_spacing")


@dataclass(frozen=True)
class Viewpoint:
    viewpoint_id: str
    position: Point3

    def __post_init__(self):
        _nonempty(self.viewpoint_id, "viewpoint_id")
        if not isinstance(self.position, Point3):
            raise ValueError("Viewpoint position must be Point3")


@dataclass(frozen=True)
class PathEvidence:
    """A backend's complete path check between these exact viewpoints.

    Euclidean proximity, an arbitrary list of path corners, and successful
    endpoint teleports alone are not sufficient evidence for ``feasible=True``.
    ``context_key`` must cover the world AND current container-state geometry.
    """
    source: str
    destination: str
    context_key: str
    feasible: Optional[bool]
    evidence_id: str


@dataclass(frozen=True)
class NavigationGraph:
    context_key: str
    viewpoints: Tuple[Viewpoint, ...]
    edges: Tuple[PathEvidence, ...]
    opened_containers: Tuple[str, ...]
    rejected_edges: Tuple[Tuple[str, str, str], ...]

    def __post_init__(self):
        _nonempty(self.context_key, "context_key")
        if not isinstance(self.viewpoints, tuple) or not self.viewpoints:
            raise ValueError("Graph viewpoints must be a nonempty tuple")
        nodes = {node.viewpoint_id: node.position for node in self.viewpoints}
        if len(nodes) != len(self.viewpoints) or len(set(nodes.values())) != len(nodes):
            raise ValueError("Graph viewpoint IDs and positions must be unique")
        if not isinstance(self.edges, tuple):
            raise ValueError("Graph edges must be a tuple")
        keys = set()
        for edge in self.edges:
            key = (edge.source, edge.destination)
            if (key in keys or edge.source not in nodes or edge.destination not in nodes
                    or edge.source == edge.destination or edge.feasible is not True
                    or edge.context_key != self.context_key
                    or nodes[edge.source].distance(nodes[edge.destination]) > 1.5):
                raise ValueError("Graph contains a duplicate, unverified, or invalid edge")
            _nonempty(edge.evidence_id, "verified path evidence_id")
            keys.add(key)
        if (not isinstance(self.opened_containers, tuple)
                or tuple(sorted(set(self.opened_containers))) != self.opened_containers):
            raise ValueError("Graph open-container IDs must be a sorted unique tuple")
        for object_id in self.opened_containers:
            _nonempty(object_id, "open-container ID")

    def neighbors(self, viewpoint_id):
        return tuple(sorted(edge.destination for edge in self.edges if edge.source == viewpoint_id))


def build_navigation_graph(viewpoints: Sequence[Viewpoint], evidence: Mapping[Tuple[str, str], PathEvidence],
                           *, context_key: str, opened_containers: Iterable[str],
                           maximum_edge_distance_m: float = 1.5):
    """Keep only near, explicitly verified directed edges for one world state.

    Both directions need evidence; reverse reachability is never inferred.
    Graph construction does not certify the T01 viewpoint count or physical
    validity of a world edit. Those are independent generator checks.
    """
    _nonempty(context_key, "context_key")
    _finite(maximum_edge_distance_m, "maximum_edge_distance_m", 0)
    if not 0 < maximum_edge_distance_m <= 1.5:
        raise ValueError("T01 graph edges must be no longer than 1.5m")
    nodes = tuple(sorted(viewpoints, key=lambda node: node.viewpoint_id))
    if not nodes or len({node.viewpoint_id for node in nodes}) != len(nodes):
        raise ValueError("Navigation viewpoints must be nonempty with unique IDs")
    if len({node.position for node in nodes}) != len(nodes):
        raise ValueError("Duplicate viewpoint positions")
    opened = tuple(sorted(opened_containers))
    if len(opened) != len(set(opened)):
        raise ValueError("Duplicate open-container IDs")
    for object_id in opened:
        _nonempty(object_id, "open-container ID")
    edges, rejected = [], []
    for source in nodes:
        for destination in nodes:
            if source == destination or source.position.distance(destination.position) > maximum_edge_distance_m:
                continue
            key = (source.viewpoint_id, destination.viewpoint_id)
            check = evidence.get(key)
            if check is None:
                rejected.append((*key, "path_not_checked"))
                continue
            if (check.source, check.destination, check.context_key) != (*key, context_key):
                raise ValueError(f"Path evidence has the wrong endpoints/world context: {key}")
            if check.feasible is not None and type(check.feasible) is not bool:
                raise ValueError("Path feasibility must be True, False, or None")
            if check.feasible is True:
                _nonempty(check.evidence_id, "verified path evidence_id")
                edges.append(check)
            else:
                rejected.append((*key, "path_infeasible" if check.feasible is False else "path_unknown"))
    return NavigationGraph(context_key, nodes, tuple(edges), opened, tuple(rejected))


@dataclass(frozen=True)
class Mark:
    label: int
    object_id: str
    roles: Tuple[str, ...]
    bounding_box: Tuple[int, int, int, int]  # inclusive xmin, ymin, xmax, ymax
    anchor_pixel: Tuple[int, int]
    pixel_count: int


def visible_marks(snapshot: WorldSnapshot, masks: Mapping[str, Iterable[Tuple[int, int]]],
                  *, target_type: str, width: int, height: int):
    """Order marks by mask geometry, independent of metadata/ID iteration order.

    Pixels are (x, y), with (0, 0) at the upper left. Structural mask IDs that
    have no ObjectState are ignored. No distance/metadata-visible filtering is
    applied: any nonempty target-category or openable-receptacle mask counts.
    """
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Positive integer image dimensions required")
    entries, occupied = [], set()
    for obj in snapshot.objects:
        roles = tuple(role for role, matches in
                      (("candidate", obj.object_type == target_type),
                       ("container", obj.openable and obj.receptacle)) if matches)
        if not roles or obj.object_id not in masks:
            continue
        pixels = set()
        for pixel in masks[obj.object_id]:
            if (len(pixel) != 2 or any(type(value) is not int for value in pixel)
                    or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
                raise ValueError(f"Invalid mask pixel for {obj.object_id}")
            pixels.add(tuple(pixel))
        if not pixels:
            continue
        if occupied.intersection(pixels):
            raise ValueError("Instance masks overlap; cannot assign unambiguous marks")
        occupied.update(pixels)
        xs, ys = zip(*pixels)
        box = (min(xs), min(ys), max(xs), max(ys))
        ordered_pixels = tuple(sorted(pixels, key=lambda pixel: (pixel[1], pixel[0])))
        count, sx, sy = len(pixels), sum(xs), sum(ys)
        # Integer arithmetic: nearest mask pixel to centroid, then row/column.
        anchor = min(pixels, key=lambda p: ((p[0] * count - sx) ** 2 + (p[1] * count - sy) ** 2, p[1], p[0]))
        entries.append(((box[1], box[0], box[3], box[2], ordered_pixels), obj.object_id,
                        roles, box, anchor, count))
    entries.sort(key=lambda entry: entry[0])
    return tuple(Mark(index, object_id, roles, box, anchor, count)
                 for index, (_, object_id, roles, box, anchor, count) in enumerate(entries, 1))


@dataclass(frozen=True)
class RelationEvidence:
    """Explicit world-specific facts; absence/None means UNKNOWN, not false.

    For 'next to', values are measured distances in metres under the named
    method. For 'in'/'on', they are verified booleans. parentReceptacles alone
    does not distinguish containment from support; the adapter must establish
    the requested relation. Neither centre distance nor AABB overlap is silently
    substituted for semantic/physical evidence by this module.
    """
    world_key: str
    relation: str
    method: str
    values: Mapping[Tuple[str, str], object]


@dataclass(frozen=True)
class TargetResolution:
    target_id: Optional[str]
    anchor_id: Optional[str]
    candidate_ids: Tuple[str, ...]
    reason: str
    evidence_method: str


def resolve_target(task: RelationalTask, snapshot: WorldSnapshot, evidence: RelationEvidence,
                   *, distance_tie_tolerance_m: Optional[float] = None):
    """Resolve independently in each world; ambiguity/unknown facts reject.

    A category-only anchor description requires exactly one such anchor in the
    full world inventory. Next-to means uniquely nearest under the caller's
    explicit metric and tie tolerance; there is no invented adjacency cutoff.
    """
    if not snapshot.complete_object_inventory:
        raise ValueError("Unique-target resolution requires a complete object inventory")
    if (evidence.world_key, evidence.relation) != (snapshot.world_key, task.relation):
        raise ValueError("Relation evidence belongs to a different world or relation")
    _nonempty(evidence.method, "relation evidence method")
    anchors = sorted(obj.object_id for obj in snapshot.objects if obj.object_type == task.anchor_type)
    candidates = tuple(sorted(obj.object_id for obj in snapshot.objects if obj.object_type == task.target_type))
    if len(anchors) != 1:
        return TargetResolution(None, None, candidates, "anchor_not_unique", evidence.method)
    anchor = anchors[0]
    candidates = tuple(object_id for object_id in candidates if object_id != anchor)
    if not candidates:
        return TargetResolution(None, anchor, candidates, "no_candidate", evidence.method)
    values = {object_id: evidence.values.get((object_id, anchor)) for object_id in candidates}
    if any(value is None for value in values.values()):
        return TargetResolution(None, anchor, candidates, "unknown_relation_evidence", evidence.method)
    if task.relation == "next to":
        if distance_tie_tolerance_m is None:
            raise ValueError("Next-to requires an explicit distance tie tolerance")
        _finite(distance_tie_tolerance_m, "distance_tie_tolerance_m", 0)
        for value in values.values():
            _finite(value, "relation distance", 0)
        nearest = min(values.values())
        matches = [object_id for object_id, distance in values.items()
                   if distance - nearest <= distance_tie_tolerance_m]
    else:
        if any(type(value) is not bool for value in values.values()):
            raise ValueError("In/on evidence must contain verified booleans or None")
        matches = [object_id for object_id, matches in values.items() if matches]
    unique = len(matches) == 1
    return TargetResolution(matches[0] if unique else None, anchor, candidates,
                            "unique" if unique else "target_not_unique", evidence.method)


@dataclass(frozen=True, order=True)
class ViewState:
    viewpoint_id: str
    heading_index: int  # Unity yaw = index * 45 degrees, horizon fixed at zero
    opened_containers: Tuple[str, ...] = ()

    def __post_init__(self):
        _nonempty(self.viewpoint_id, "viewpoint_id")
        if type(self.heading_index) is not int or not 0 <= self.heading_index < 8:
            raise ValueError("heading_index must be an integer from 0 through 7")
        if not isinstance(self.opened_containers, tuple) or tuple(sorted(set(self.opened_containers))) != self.opened_containers:
            raise ValueError("opened_containers must be a sorted unique tuple")
        for object_id in self.opened_containers:
            _nonempty(object_id, "open-container ID")


@dataclass(frozen=True)
class Action:
    kind: str
    argument: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("move", "turn_left", "turn_right", "open", "select"):
            raise ValueError("Unknown discrete action")
        if self.kind in ("move", "open", "select"):
            if type(self.argument) is not int or self.argument < 1:
                raise ValueError("Indexed actions require a positive integer argument")
        elif self.argument is not None:
            raise ValueError("Turn actions have no argument")

    def text(self):
        return self.kind if self.argument is None else f"{self.kind} {self.argument}"


def neighbor_descriptions(graph: NavigationGraph, state: ViewState):
    """One-based neighbor indices, sorted by stable viewpoint ID, relative yaw."""
    if graph.opened_containers != state.opened_containers:
        raise ValueError("Graph paths were checked under different container states")
    nodes = {node.viewpoint_id: node.position for node in graph.viewpoints}
    origin = nodes[state.viewpoint_id]
    descriptions = []
    directions = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
    for index, destination in enumerate(graph.neighbors(state.viewpoint_id), 1):
        point = nodes[destination]
        yaw = math.degrees(math.atan2(point.x - origin.x, point.z - origin.z))
        relative = (yaw - state.heading_index * 45) % 360
        direction = directions[int(math.floor((relative + 22.5) / 45)) % 8]
        descriptions.append({"index": index, "viewpoint_id": destination, "direction": direction,
                             "relative_bearing_degrees": (relative + 180) % 360 - 180})
    return tuple(descriptions)


@dataclass(frozen=True)
class Transition:
    action: Action
    next_state: ViewState


def _validate_transition(state, transition):
    if not isinstance(transition, Transition) or not isinstance(transition.next_state, ViewState):
        raise ValueError("Successors must contain Transition records with ViewState destinations")
    action, following = transition.action, transition.next_state
    if not isinstance(action, Action) or action.kind == "select":
        raise ValueError("Select belongs in terminal_action, not observation successors")
    if action.kind in ("turn_left", "turn_right"):
        delta = -1 if action.kind == "turn_left" else 1
        expected = ViewState(state.viewpoint_id, (state.heading_index + delta) % 8, state.opened_containers)
        if following != expected:
            raise ValueError("A turn must change only the heading by exactly 45 degrees")
    elif action.kind == "move":
        if following.opened_containers != state.opened_containers or following.viewpoint_id == state.viewpoint_id:
            raise ValueError("A move must change viewpoints without changing container states")
    elif action.kind == "open":
        before, after = set(state.opened_containers), set(following.opened_containers)
        if (following.viewpoint_id != state.viewpoint_id or following.heading_index != state.heading_index
                or not before < after or len(after - before) != 1):
            raise ValueError("Open must add exactly one open container without moving or turning")


def navigation_transitions(graph: NavigationGraph, state: ViewState,
                           *, destination_heading: Callable[[ViewState, str], int]):
    """Movement heading policy is explicit; each observation action is one step."""
    moves = [Transition(Action("move", item["index"]),
                        ViewState(item["viewpoint_id"], destination_heading(state, item["viewpoint_id"]), state.opened_containers))
             for item in neighbor_descriptions(graph, state)]
    return tuple(moves + [Transition(Action("turn_left"), ViewState(state.viewpoint_id, (state.heading_index - 1) % 8, state.opened_containers)),
                          Transition(Action("turn_right"), ViewState(state.viewpoint_id, (state.heading_index + 1) % 8, state.opened_containers))])


def selection_action(target_id: str, marks: Sequence[Mark], *, measured_distance_m: Optional[float]):
    """Candidate terminal action from actual mask visibility and measured range."""
    if len({mark.label for mark in marks}) != len(marks):
        raise ValueError("Mark labels must be unique before selecting a target")
    if measured_distance_m is None:
        return None
    _finite(measured_distance_m, "measured_distance_m", 0)
    matches = [mark for mark in marks if mark.object_id == target_id and "candidate" in mark.roles and mark.pixel_count > 0]
    if len(matches) != 1 or measured_distance_m > 1.5:
        return None
    return Action("select", matches[0].label)


@dataclass(frozen=True)
class WitnessPlan:
    actions: Tuple[Action, ...]
    states: Tuple[ViewState, ...]
    found: bool
    reason: str
    expanded_states: int
    requires_simulator_replay: bool = True


def bounded_witness(start: ViewState, successors: Callable[[ViewState], Iterable[Transition]],
                    terminal_action: Callable[[ViewState], Optional[Action]], *,
                    maximum_expanded_states: int, maximum_actions: int = 12):
    """Deterministic BFS including final select within the 12-action ceiling.

    Callbacks must use world-specific path/visibility/open-action evidence.
    Open transitions must include the actual currently visible mark label and
    a corresponding changed container state; this core does not infer that an
    opening is collision-free or available. A found plan is NOT an executed
    witness and must not set witness validity until successful simulator replay.
    """
    if type(maximum_actions) is not int or not 1 <= maximum_actions <= 12:
        raise ValueError("The witness budget must include select and be at most 12 actions")
    if type(maximum_expanded_states) is not int or maximum_expanded_states < 1:
        raise ValueError("An explicit positive search-state budget is required")
    queue = deque([(start, (), (start,))])
    visited, expanded, depth_limited = {start}, 0, False
    while queue:
        if expanded >= maximum_expanded_states:
            return WitnessPlan((), (), False, "search_state_budget_exhausted", expanded)
        state, actions, states = queue.popleft()
        expanded += 1
        select = terminal_action(state)
        if select is not None:
            if not isinstance(select, Action) or select.kind != "select":
                raise ValueError("The only terminal action is select")
            return WitnessPlan(actions + (select,), states, True, "candidate_requires_replay", expanded)
        if len(actions) >= maximum_actions - 1:
            depth_limited = True
            continue
        transitions = list(successors(state))
        by_action = {}
        for transition in transitions:
            _validate_transition(state, transition)
            key = transition.action.text()
            if key in by_action and by_action[key] != transition.next_state:
                raise ValueError("One action has conflicting next states")
            by_action[key] = transition.next_state
        for action_text, next_state in sorted(by_action.items()):
            if next_state not in visited:
                action = next(item.action for item in transitions if item.action.text() == action_text)
                visited.add(next_state)
                queue.append((next_state, actions + (action,), states + (next_state,)))
    return WitnessPlan((), (), False, "action_budget_exhausted" if depth_limited else "no_goal_in_supplied_graph", expanded)
