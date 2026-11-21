# Edge/cloud inference scheduling simulator

A small Python simulation for studying where AI inference requests should run when edge and cloud resources have different compute speeds, queue lengths, and network conditions.

I built this as a reproducible experiment in resource-aware scheduling. It does not execute real language models or reproduce a production mobile network. The requests and network conditions are synthetic, and the power values are illustrative.

## What it simulates

Requests arrive over time with a prompt length, expected output length, input payload size, deadline, and priority. Three edge servers and one cloud server serve the workload. Each server has a fixed number of execution slots, a processing rate, a network round-trip time, and a time-varying bandwidth. Queues form when all slots are busy.

Three scheduling policies are compared using the **same requests and network trace**:

- **Edge-only round robin:** send requests to edge servers in a fixed rotation.
- **Minimum predicted latency:** select the server with the earliest estimated response time, considering its queue.
- **Deadline-aware / energy-proxy:** among servers predicted to meet the request deadline, prefer the lowest estimated energy proxy; otherwise choose the earliest completion.

The experiment reports response latency, 95th-percentile latency, missed deadlines, urgent-request deadline misses, edge placement, and an illustrative compute/network energy proxy.

## Run

Python 3.10 or later; no external packages required.

```bash
python run_experiments.py --requests 400 --seed 42 --arrival-rate 1.2
```

The script prints a comparison and writes these files into `results/`:

- `summary.csv`: aggregate statistics for each policy
- `requests.csv`: individual request outcomes for inspection
- `latency_comparison.svg`: mean and 95th-percentile latency
- `deadline_misses.svg`: total and urgent deadline-miss rates

To run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Try changing the arrival rate to see how queueing changes the comparison:

```bash
python run_experiments.py --requests 400 --seed 42 --arrival-rate 0.4
python run_experiments.py --requests 400 --seed 42 --arrival-rate 2.0
```

## Repository structure

```text
cloud_network_computing/
    __init__.py
    simulation.py       # Workload, nodes, queueing and scheduling logic
run_experiments.py      # CLI, statistics, CSV and SVG outputs
tests/
    test_simulation.py
.github/workflows/
    tests.yml
results/                # Generated on demand; not committed
```

## Model and assumptions

The simulator uses an arrival-driven discrete-event approach: each request is placed in the earliest available execution slot at its assigned server. A request incurs input transfer time, queueing, compute time, and output transfer time. Network conditions are sampled at arrival and remain fixed for that request. Edge and cloud capacity and workload sizes can be changed in the source code.

The *deadline-aware* policy has access to the simulator's own service-time estimates. In real deployments these would need to be learned from noisy telemetry, and inference times would vary by model, batch size, GPU and input. The energy estimate is a simple product of modelled active power and compute/transfer duration, **not measured electrical energy**.

This project intentionally leaves out radio interference, packet-level routing, real-time preemption, failures, GPU memory management, batching and actual model accuracy. It is a scheduling experiment, not a validated public-safety communications simulator or a performance claim about real foundation models. For serious research, the next steps would be calibration against measurements, more realistic traffic traces, and tests under prediction error and changing network conditions.

## Why this project

My earlier research involved LLM orchestration, model evaluation and constrained recommendation. I wanted to explore the systems side: how different placement decisions affect response times when compute and networking are both limited. This simulation is a starting point for that investigation.

## License

MIT. See `LICENSE`.
