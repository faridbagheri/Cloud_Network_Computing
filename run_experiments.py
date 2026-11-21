"""Run comparable placement experiments and save the outcomes.

Only the Python standard library is used. See README.md for model limitations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from html import escape
from pathlib import Path
from statistics import fmean

from cloud_network_computing.simulation import (
    Outcome,
    Simulator,
    generate_requests,
)


DISPLAY_NAMES = {
    "edge_round_robin": "Edge round robin",
    "min_latency": "Minimum latency",
    "deadline_energy": "Deadline / energy",
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize(outcomes: list[Outcome]) -> dict[str, float | int | str]:
    if not outcomes:
        raise ValueError("cannot summarize empty outcomes")
    urgent = [o for o in outcomes if o.urgent]
    return {
        "policy": outcomes[0].policy,
        "requests": len(outcomes),
        "mean_latency_s": round(fmean(o.latency_s for o in outcomes), 4),
        "p95_latency_s": round(percentile([o.latency_s for o in outcomes], 95), 4),
        "deadline_miss_pct": round(100 * sum(o.missed_deadline for o in outcomes) / len(outcomes), 2),
        "urgent_miss_pct": round(
            100 * sum(o.missed_deadline for o in urgent) / len(urgent), 2
        ) if urgent else 0.0,
        "edge_placement_pct": round(
            100 * sum(o.location == "edge" for o in outcomes) / len(outcomes), 2
        ),
        "total_energy_proxy_j": round(sum(o.energy_proxy_j for o in outcomes), 2),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(
    path: Path,
    title: str,
    rows: list[dict],
    first: tuple[str, str],
    second: tuple[str, str],
    unit: str,
) -> None:
    """A lightweight grouped bar chart without plotting dependencies."""
    width, height, baseline, top = 900, 345, 267, 75
    first_values = [float(row[first[0]]) for row in rows]
    second_values = [float(row[second[0]]) for row in rows]
    maximum = max(first_values + second_values + [1.0]) * 1.18
    chart_h = baseline - top
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="38" font-family="sans-serif" font-weight="bold" '
        f'font-size="20" fill="#192338">{escape(title)}</text>',
        f'<line x1="70" y1="{baseline}" x2="870" y2="{baseline}" '
        'stroke="#56657a" stroke-width="1.5"/>',
    ]
    colors = ("#2363ac", "#dc7f36")
    for index, row in enumerate(rows):
        center_x = 194 + index * 265
        for value, offset, color in (
            (float(row[first[0]]), -46, colors[0]),
            (float(row[second[0]]), 6, colors[1]),
        ):
            bar_h = value / maximum * chart_h
            x = center_x + offset
            y = baseline - bar_h
            svg += [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="40" height="{bar_h:.1f}" '
                f'fill="{color}" rx="2"/>',
                f'<text x="{x + 20:.1f}" y="{max(top + 5, y - 8):.1f}" '
                'font-family="sans-serif" font-size="12" text-anchor="middle" '
                f'fill="#192338">{value:.1f}</text>',
            ]
        name = DISPLAY_NAMES[str(row["policy"])]
        svg.append(
            f'<text x="{center_x:.1f}" y="290" font-family="sans-serif" '
            f'font-size="14" text-anchor="middle" fill="#24344c">{escape(name)}</text>'
        )
    for i, (label, color) in enumerate(((first[1], colors[0]), (second[1], colors[1]))):
        x = 72 + i * 275
        svg += [
            f'<rect x="{x}" y="310" width="15" height="11" fill="{color}"/>',
            f'<text x="{x + 22}" y="320" font-family="sans-serif" '
            f'font-size="12" fill="#354156">{escape(label)} ({escape(unit)})</text>',
        ]
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def run_experiment(
    count: int, seed: int, arrival_rate: float, output_dir: Path
) -> list[dict]:
    workload = generate_requests(count=count, seed=seed, arrival_rate=arrival_rate)
    summary_rows: list[dict] = []
    request_rows: list[dict] = []

    for policy in Simulator.POLICIES:
        outcomes = Simulator().run(workload, policy)
        summary_rows.append(summarize(outcomes))
        request_rows.extend(asdict(o) for o in outcomes)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "requests.csv", request_rows)
    write_svg(
        output_dir / "latency_comparison.svg",
        "Simulated request latency by scheduler",
        summary_rows,
        ("mean_latency_s", "Mean"),
        ("p95_latency_s", "95th percentile"),
        "seconds",
    )
    write_svg(
        output_dir / "deadline_misses.svg",
        "Simulated deadline misses by scheduler",
        summary_rows,
        ("deadline_miss_pct", "All requests"),
        ("urgent_miss_pct", "Urgent requests"),
        "percent",
    )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=1.2,
                        help="Mean request arrivals per simulated second")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        rows = run_experiment(args.requests, args.seed, args.arrival_rate, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Synthetic workload: {args.requests} requests; seed={args.seed}; "
          f"arrival rate={args.arrival_rate}/s")
    print("Energy is an illustrative proxy, not an electrical measurement.\n")
    print(f'{"Policy":<21} {"Mean(s)":>9} {"P95(s)":>9} '
          f'{"Miss(%)":>9} {"Urgent(%)":>11} {"Edge(%)":>9} {"Energy(J)":>12}')
    for row in rows:
        print(f'{DISPLAY_NAMES[str(row["policy"])]:<21} '
              f'{row["mean_latency_s"]:>9.3f} {row["p95_latency_s"]:>9.3f} '
              f'{row["deadline_miss_pct"]:>9.2f} {row["urgent_miss_pct"]:>11.2f} '
              f'{row["edge_placement_pct"]:>9.2f} {row["total_energy_proxy_j"]:>12.1f}')
    print(f"\nCSV results and SVG figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
