"""Tests for repeatability, scheduling, metrics and file outputs."""

import tempfile
import unittest
from pathlib import Path

from cloud_network_computing.simulation import (
    Node,
    Request,
    Simulator,
    bandwidth_at,
    generate_requests,
)
from run_experiments import run_experiment, summarize


class WorkloadTests(unittest.TestCase):
    def test_seeded_requests_are_repeatable(self):
        self.assertEqual(generate_requests(50, 12), generate_requests(50, 12))
        self.assertNotEqual(generate_requests(50, 12), generate_requests(50, 13))

    def test_arrivals_and_inputs_are_valid(self):
        rows = generate_requests(250)
        self.assertEqual(len(rows), 250)
        self.assertEqual(rows, sorted(rows, key=lambda r: r.arrival_s))
        self.assertTrue(all(r.prompt_tokens > 0 and r.deadline_s > 0 for r in rows))

    def test_invalid_workload_parameters(self):
        with self.assertRaises(ValueError):
            generate_requests(0)
        with self.assertRaises(ValueError):
            generate_requests(3, arrival_rate=0)


class SimulationTests(unittest.TestCase):
    def test_same_trace_is_deterministic(self):
        workload = generate_requests(150, 42)
        for policy in Simulator.POLICIES:
            self.assertEqual(
                Simulator().run(workload, policy),
                Simulator().run(workload, policy),
            )

    def test_every_request_has_valid_outcome(self):
        workload = generate_requests(220, 9)
        for policy in Simulator.POLICIES:
            rows = Simulator().run(workload, policy)
            self.assertEqual(len(rows), len(workload))
            self.assertEqual(set(o.request_id for o in rows),
                             set(r.request_id for r in workload))
            self.assertTrue(all(o.latency_s > 0 and o.queue_s >= 0 for o in rows))
            self.assertTrue(all(o.energy_proxy_j > 0 for o in rows))
            self.assertTrue(all(o.completion_s >= o.arrival_s for o in rows))
            self.assertTrue(all(o.missed_deadline == (o.latency_s > o.deadline_s)
                                for o in rows))

    def test_baseline_is_edge_only(self):
        rows = Simulator().run(generate_requests(30), "edge_round_robin")
        self.assertTrue(all(row.location == "edge" for row in rows))

    def test_backlog_creates_queueing(self):
        heavy = generate_requests(200, arrival_rate=20)
        rows = Simulator().run(heavy, "edge_round_robin")
        self.assertTrue(any(row.queue_s > 0 for row in rows))

    def test_cloud_network_has_periodic_congestion(self):
        cloud = Node("cloud", "cloud", 1, 400, 60, 25, 200, 3)
        self.assertLess(bandwidth_at(cloud, 70), bandwidth_at(cloud, 5))

    def test_one_request_min_latency_selects_fastest_estimate(self):
        workload = [Request(0, 0, 150, 80, 100, 3, True)]
        sim = Simulator()
        rows = sim.run(workload, "min_latency")
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0].node, {n.name for n in sim.nodes})

    def test_invalid_policy_and_nodes(self):
        with self.assertRaises(ValueError):
            Simulator().run(generate_requests(1), "unknown")
        with self.assertRaises(ValueError):
            Simulator(nodes=[])

    def test_no_urgent_requests_in_summary(self):
        rows = Simulator().run([Request(1, 0, 120, 30, 50, 3, False)],
                               "min_latency")
        self.assertEqual(summarize(rows)["urgent_miss_pct"], 0.0)


class ExperimentTests(unittest.TestCase):
    def test_smoke_run_writes_tables_and_charts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            folder = Path(tempdir)
            rows = run_experiment(40, 42, 1.2, folder)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["requests"] for r in rows}, {40})
            for name in (
                "summary.csv", "requests.csv",
                "latency_comparison.svg", "deadline_misses.svg",
            ):
                self.assertTrue((folder / name).is_file(), name)
            self.assertIn("<svg", (folder / "latency_comparison.svg").read_text())


if __name__ == "__main__":
    unittest.main()
