"""Discrete-event-style model of AI request placement across edge and cloud nodes.

All quantities are synthetic. This is an experiment in placement and queues,
not a packet-level network simulator or a benchmark of real inference hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Node:
    name: str
    location: str  # "edge" or "cloud"
    slots: int
    tokens_per_second: float
    rtt_ms: float
    bandwidth_mbps: float
    active_watts: float
    link_watts: float
    phase: float = 0.0


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    payload_kb: float
    deadline_s: float
    urgent: bool


@dataclass(frozen=True)
class Outcome:
    policy: str
    request_id: int
    urgent: bool
    node: str
    location: str
    arrival_s: float
    completion_s: float
    latency_s: float
    queue_s: float
    deadline_s: float
    missed_deadline: bool
    energy_proxy_j: float


@dataclass(frozen=True)
class _Candidate:
    node: Node
    slot: int
    compute_end_s: float
    completion_s: float
    queue_s: float
    energy_proxy_j: float

    def latency_s(self, request: Request) -> float:
        return self.completion_s - request.arrival_s


def default_nodes() -> tuple[Node, ...]:
    """Illustrative devices, not measurements from physical servers."""
    return (
        Node("edge-a", "edge", 1, 110.0, 9.0, 38.0, 43.0, 2.0, 0.2),
        Node("edge-b", "edge", 1, 145.0, 13.0, 55.0, 50.0, 2.0, 1.4),
        Node("edge-c", "edge", 1, 190.0, 18.0, 42.0, 61.0, 2.0, 2.7),
        Node("cloud", "cloud", 2, 440.0, 60.0, 24.0, 215.0, 5.0, 0.8),
    )


def generate_requests(
    count: int = 400, seed: int = 42, arrival_rate: float = 1.2
) -> list[Request]:
    """Seeded synthetic Poisson arrivals, with urgent and normal requests."""
    if count < 1:
        raise ValueError("count must be positive")
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")

    rng = random.Random(seed)
    t = 0.0
    requests = []
    for i in range(count):
        t += rng.expovariate(arrival_rate)
        urgent = rng.random() < 0.23
        prompt = rng.randint(80, 640)
        output = rng.randint(24, 170)
        payload = rng.uniform(35.0, 340.0 if urgent else 640.0)
        deadline = rng.uniform(1.1, 2.7) if urgent else rng.uniform(3.0, 7.0)
        requests.append(
            Request(i, t, prompt, output, payload, deadline, urgent)
        )
    return requests


def bandwidth_at(node: Node, time_s: float) -> float:
    """A deterministic changing link, including occasional cloud congestion."""
    wave = 0.83 + 0.25 * math.sin(time_s / 21.0 + node.phase)
    if node.location == "cloud" and int(time_s // 65) % 3 == 1:
        wave *= 0.48
    return max(0.25 * node.bandwidth_mbps, node.bandwidth_mbps * wave)


class Simulator:
    """Place arrivals onto execution slots; track waits and completions.

    Each slot has a next-available time. Network transfers are estimated using
    the link at request arrival, and model computation is non-preemptive.
    """

    POLICIES = ("edge_round_robin", "min_latency", "deadline_energy")

    def __init__(self, nodes: Iterable[Node] | None = None):
        self.nodes = tuple(nodes) if nodes is not None else default_nodes()
        if not self.nodes or any(
            n.slots <= 0 or n.tokens_per_second <= 0
            or n.bandwidth_mbps <= 0 or n.rtt_ms < 0
            for n in self.nodes
        ):
            raise ValueError("nodes need positive capacities and valid links")
        if not any(n.location == "edge" for n in self.nodes):
            raise ValueError("the edge-only baseline requires an edge node")

    @staticmethod
    def _estimate(node: Node, request: Request, free_times: list[float]) -> _Candidate:
        bw = bandwidth_at(node, request.arrival_s)
        rtt_half_s = node.rtt_ms / 2000.0

        # Mbps to kilobytes per second: 1 Mbps = 125 kB/s.
        upload_s = rtt_half_s + request.payload_kb / (125.0 * bw)
        output_kb = request.output_tokens * 0.025
        download_s = rtt_half_s + output_kb / (125.0 * bw)

        # Prompt processing is discounted relative to autoregressive output.
        work_tokens = 0.22 * request.prompt_tokens + request.output_tokens
        compute_s = work_tokens / node.tokens_per_second
        slot = min(range(len(free_times)), key=lambda j: (free_times[j], j))
        compute_start = max(request.arrival_s + upload_s, free_times[slot])
        queue_s = compute_start - (request.arrival_s + upload_s)
        compute_end = compute_start + compute_s
        completion = compute_end + download_s

        # Active compute + transfer power, not actual electrical measurements.
        energy = compute_s * node.active_watts + (upload_s + download_s) * node.link_watts
        return _Candidate(node, slot, compute_end, completion, queue_s, energy)

    def run(self, requests: Iterable[Request], policy: str) -> list[Outcome]:
        if policy not in self.POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected {self.POLICIES}")
        slots = {node.name: [0.0] * node.slots for node in self.nodes}
        edges = [node.name for node in self.nodes if node.location == "edge"]
        outcomes = []

        for request in sorted(requests, key=lambda r: (r.arrival_s, r.request_id)):
            candidates = [
                self._estimate(node, request, slots[node.name]) for node in self.nodes
            ]
            if policy == "edge_round_robin":
                target = edges[request.request_id % len(edges)]
                chosen = next(c for c in candidates if c.node.name == target)
            elif policy == "min_latency":
                chosen = min(
                    candidates,
                    key=lambda c: (c.completion_s, c.node.name),
                )
            else:
                feasible = [
                    c for c in candidates
                    if c.latency_s(request) <= request.deadline_s
                ]
                chosen = (
                    min(feasible, key=lambda c: (c.energy_proxy_j, c.completion_s))
                    if feasible else
                    min(candidates, key=lambda c: (c.completion_s, c.energy_proxy_j))
                )

            slots[chosen.node.name][chosen.slot] = chosen.compute_end_s
            latency = chosen.latency_s(request)
            outcomes.append(
                Outcome(
                    policy=policy,
                    request_id=request.request_id,
                    urgent=request.urgent,
                    node=chosen.node.name,
                    location=chosen.node.location,
                    arrival_s=request.arrival_s,
                    completion_s=chosen.completion_s,
                    latency_s=latency,
                    queue_s=chosen.queue_s,
                    deadline_s=request.deadline_s,
                    missed_deadline=latency > request.deadline_s,
                    energy_proxy_j=chosen.energy_proxy_j,
                )
            )
        return outcomes
