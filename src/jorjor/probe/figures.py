from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact


FIGURE_FILENAMES = {
    "figure_2": "figure_2_evidence_to_claim_trajectory",
    "figure_3": "figure_3_composition_gate_per_seed",
    "figure_4": "figure_4_matched_composition_decisive",
    "figure_5": "figure_5_oracle_validation_ordered_vs_shortcut",
}
SEED_ORDER = ("seed_k", "seed_l", "seed_m")
FIGURE3_COMPARISONS = (
    (
        "composition_nuisance:ordered_residual_vs_action_value",
        "ordered residual vs action/value",
        "#3972b8",
    ),
    (
        "composition_nuisance:action_value_composition_only_vs_action_value",
        "action/value+composition vs action/value",
        "#d07a24",
    ),
    (
        "composition_nuisance:composition_residual_vs_action_value_composition",
        "residual vs action/value+composition",
        "#8f4ca6",
    ),
)


def build_paper_figure_pack(
    *,
    belief_shortcut_report_path: Path,
    final_tables_path: Path,
    matched_composition_report_path: Path,
    synthetic_oracle_report_path: Path,
    learned_synthetic_report_path: Path,
    toy_oracle_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build the v1.16 paper figure pack from archived diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "belief_shortcut_report": belief_shortcut_report_path,
        "final_tables": final_tables_path,
        "matched_composition_report": matched_composition_report_path,
        "synthetic_oracle_report": synthetic_oracle_report_path,
        "learned_synthetic_report": learned_synthetic_report_path,
        "toy_oracle_report": toy_oracle_report_path,
    }
    sources = {name: _source_record(path) for name, path in source_paths.items()}

    belief_report = _read_json_object(belief_shortcut_report_path)
    final_tables = _read_json_object(final_tables_path)
    matched_report = _read_json_object(matched_composition_report_path)
    synthetic_report = _read_json_object(synthetic_oracle_report_path)
    learned_report = _read_json_object(learned_synthetic_report_path)
    toy_report = _read_json_object(toy_oracle_report_path)

    figure2 = _build_figure2_data(belief_report, final_tables, matched_report)
    figure3 = _build_figure3_data(final_tables)
    figure4 = _build_figure4_data(matched_report)
    figure5 = _build_figure5_data(synthetic_report, learned_report, toy_report)

    figures = [
        _write_figure(output_dir, "figure_2", figure2, _render_figure2_svg(figure2)),
        _write_figure(output_dir, "figure_3", figure3, _render_figure3_svg(figure3)),
        _write_figure(output_dir, "figure_4", figure4, _render_figure4_svg(figure4)),
        _write_figure(output_dir, "figure_5", figure5, _render_figure5_svg(figure5)),
    ]
    payload: dict[str, object] = {
        "version": "v1.16",
        "artifact_type": "paper_figure_pack",
        "sources": sources,
        "figures": figures,
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "figure_pack.json", payload)
    (output_dir / "figure_pack.md").write_text(_figure_pack_markdown(payload), encoding="utf-8")
    return payload


def _build_figure2_data(
    belief_report: dict[str, object],
    final_tables: dict[str, object],
    matched_report: dict[str, object],
) -> dict[str, object]:
    case = _first_case(belief_report)
    metrics = _dict(case.get("key_metrics"))
    final_rows = _list(final_tables.get("rows"))
    matched_rows = _list(matched_report.get("rows"))
    matched_min_lower = min(_matched_lower_delta(_dict(row)) for row in matched_rows)
    action_value_rows = [
        _find_final_table_row(final_rows, seed, "no_range_action_value_residual_ci") for seed in SEED_ORDER
    ]
    action_value_mean_delta = sum(float(row["delta"]) for row in action_value_rows) / len(action_value_rows)
    action_value_min_lower = min(float(row["ci_lower"]) for row in action_value_rows)

    gates = [
        {
            "gate": "naive probe",
            "supported_claim_level": 4.0,
            "diagnosis": str(case.get("naive_probe_status", "positive")),
            "source_metric": None,
            "source_value": None,
        },
        {
            "gate": "label control",
            "supported_claim_level": 4.0,
            "diagnosis": str(case.get("label_control_status", "passes")),
            "source_metric": "label_control_delta_min",
            "source_value": _optional_float(metrics.get("label_control_delta_min")),
        },
        {
            "gate": "no-observation",
            "supported_claim_level": 3.0,
            "diagnosis": str(case.get("presence_last_action_status", "passes")),
            "source_metric": "presence_last_action_min_lower_delta",
            "source_value": _optional_float(metrics.get("presence_last_action_min_lower_delta")),
        },
        {
            "gate": "history destruction",
            "supported_claim_level": 3.0,
            "diagnosis": str(case.get("history_control_status", "passes")),
            "source_metric": "history_delta_ci_min_lower_delta",
            "source_value": _optional_float(metrics.get("history_delta_ci_min_lower_delta")),
        },
        {
            "gate": "action/value control",
            "supported_claim_level": 3.0,
            "diagnosis": "nuisance-controlled positive",
            "source_metric": "locked_action_value_residual_mean_delta",
            "source_value": action_value_mean_delta,
            "source_detail": {
                "min_ci_lower": action_value_min_lower,
                "seed_count": len(action_value_rows),
            },
        },
        {
            "gate": "composition control",
            "supported_claim_level": 1.0,
            "diagnosis": str(case.get("composition_nuisance_status", "composition_shortcut_supported")),
            "source_metric": "composition_residual_vs_action_value_composition_min_lower_delta",
            "source_value": _optional_float(metrics.get("composition_residual_vs_action_value_composition_min_lower_delta")),
        },
        {
            "gate": "matched-composition",
            "supported_claim_level": 1.0,
            "diagnosis": str(case.get("final_diagnostic_label", "composition_shortcut_supported")),
            "source_metric": "matched_bucket_min_ci_lower",
            "source_value": matched_min_lower,
        },
    ]
    return {
        "figure_id": "Figure 2",
        "title": "Stronger gates change the supported claim",
        "y_axis": "supported_claim_level",
        "claim_level_legend": {
            "4": "belief-like probe signal",
            "3": "standard nuisance-controlled signal",
            "2": "order evidence weak or inconclusive",
            "1": "composition shortcut supported",
        },
        "gates": gates,
    }


def _build_figure3_data(final_tables: dict[str, object]) -> dict[str, object]:
    rows = _list(final_tables.get("rows"))
    plotted_rows = []
    for seed in SEED_ORDER:
        for claim_gate, label, color in FIGURE3_COMPARISONS:
            row = _find_final_table_row(rows, seed, claim_gate)
            plotted_rows.append(
                {
                    "seed": seed,
                    "comparison": label,
                    "claim_gate": claim_gate,
                    "delta": float(row["delta"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                    "layer": str(row.get("layer", "")),
                    "layer_role": str(row.get("layer_role", "")),
                    "color": color,
                }
            )
    return {
        "figure_id": "Figure 3",
        "title": "Composition gate per seed",
        "metric": "top10_correct_delta",
        "rows": plotted_rows,
    }


def _build_figure4_data(matched_report: dict[str, object]) -> dict[str, object]:
    rows = []
    for raw in _list(matched_report.get("rows")):
        rows.append(
            {
                "seed": str(raw["seed"]),
                "delta": float(raw["bucket_bootstrap_mean_delta"]),
                "ci_lower": _matched_lower_delta(raw),
                "ci_upper": _matched_upper_delta(raw),
                "point_delta": float(raw.get("delta_residual_minus_prior", raw["bucket_bootstrap_mean_delta"])),
                "residual_top10": float(raw["residual_top10"]),
                "action_value_composition_top10": float(raw["action_value_composition_top10"]),
                "matched_bucket_count": int(raw["matched_bucket_count"]),
                "matched_point_count": int(raw["matched_point_count"]),
            }
        )
    rows.sort(key=lambda row: SEED_ORDER.index(str(row["seed"])) if str(row["seed"]) in SEED_ORDER else 999)
    return {
        "figure_id": "Figure 4",
        "title": "Residual hidden states do not beat composition priors inside matched buckets",
        "metric": "bucket_bootstrap_delta_residual_minus_action_value_composition",
        "rows": rows,
    }


def _build_figure5_data(
    synthetic_report: dict[str, object],
    learned_report: dict[str, object],
    toy_report: dict[str, object],
) -> dict[str, object]:
    rows = []
    rows.extend(_oracle_rows("analytic", synthetic_report))
    rows.extend(_oracle_rows("learned", learned_report))
    rows.extend(_oracle_rows("toy", toy_report))
    return {
        "figure_id": "Figure 5",
        "title": "Oracle validation ordered vs shortcut",
        "metric": "order_counterfactual_js",
        "rows": rows,
    }


def _find_final_table_row(rows: list[object], seed: str, claim_gate: str) -> dict[str, object]:
    matches = [
        _dict(row)
        for row in rows
        if _dict(row).get("training_seed") == seed
        and _dict(row).get("claim_gate") == claim_gate
        and _dict(row).get("table_role") == "main"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one final-table row for seed={seed} claim_gate={claim_gate}, found {len(matches)}")
    return matches[0]


def _oracle_rows(prefix: str, report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_case in _list(report.get("cases")):
        case = _dict(raw_case)
        summary = _dict(case.get("order_counterfactual_summary"))
        diagnosis = str(case["final_diagnostic_label"])
        mode = str(case["mode"])
        row_type = "ordered" if diagnosis == "belief_like_oracle_supported" else "shortcut"
        rows.append(
            {
                "setting": f"{prefix} {row_type}",
                "source": prefix,
                "mode": mode,
                "diagnosis": diagnosis,
                "posterior_js": float(summary["mean_posterior_js"]),
                "representation_js": float(summary["mean_representation_js"]),
            }
        )
    return rows


def _matched_lower_delta(row: dict[str, object]) -> float:
    if "bucket_bootstrap_ci_lower" in row:
        return float(row["bucket_bootstrap_ci_lower"])
    return float(row["bucket_bootstrap_lower_delta"])


def _matched_upper_delta(row: dict[str, object]) -> float:
    if "bucket_bootstrap_ci_upper" in row:
        return float(row["bucket_bootstrap_ci_upper"])
    return float(row["bucket_bootstrap_upper_delta"])


def _write_figure(output_dir: Path, figure_key: str, data: dict[str, object], svg: str) -> dict[str, object]:
    stem = FIGURE_FILENAMES[figure_key]
    json_path = output_dir / f"{stem}.json"
    svg_path = output_dir / f"{stem}.svg"
    data_with_checksum = dict(data)
    data_with_checksum["checksum"] = checksum_payload(data_with_checksum)
    write_json_artifact(json_path, data_with_checksum)
    svg_path.write_text(svg, encoding="utf-8")
    return {
        "figure_key": figure_key,
        "figure_id": data["figure_id"],
        "title": data["title"],
        "json_path": str(json_path),
        "svg_path": str(svg_path),
        "json_sha256": checksum_payload(data_with_checksum),
        "svg_sha256": _text_sha256(svg),
    }


def _render_figure2_svg(data: dict[str, object]) -> str:
    gates = [_dict(gate) for gate in _list(data["gates"])]
    width, height = 1200, 540
    margin = {"left": 90, "right": 60, "top": 82, "bottom": 150}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def x(index: int) -> float:
        return margin["left"] + index * plot_w / max(1, len(gates) - 1)

    def y(level: float) -> float:
        return margin["top"] + (4.3 - level) * plot_h / 3.6

    points = [(x(i), y(float(gate["supported_claim_level"]))) for i, gate in enumerate(gates)]
    parts = _svg_header(width, height, str(data["title"]))
    parts.append(_rect(margin["left"], y(2.5), plot_w, y(0.7) - y(2.5), "#f5e8dd", opacity=0.75))
    parts.append(_rect(margin["left"], y(4.3), plot_w, y(2.5) - y(4.3), "#e8f1fb", opacity=0.75))
    parts.extend(_axis_lines(margin["left"], margin["top"], plot_w, plot_h))
    for level, label in [(4, "belief-like signal"), (3, "nuisance-controlled"), (2, "weak order evidence"), (1, "shortcut-supported")]:
        yy = y(float(level))
        parts.append(_line(margin["left"], yy, margin["left"] + plot_w, yy, "#d6d6d6", width=1))
        parts.append(_text(18, yy + 4, label, size=13, fill="#454545"))
    path = " ".join(f"{'M' if i == 0 else 'L'} {px:.1f} {py:.1f}" for i, (px, py) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="#2f5f9f" stroke-width="4" stroke-linecap="round"/>')
    for index, (gate, (px, py)) in enumerate(zip(gates, points)):
        parts.append(_circle(px, py, 8, "#2f5f9f"))
        metric = gate.get("source_metric")
        value = gate.get("source_value")
        value_text = "n/a" if value is None else f"{float(value):+.3f}"
        parts.append(_text(px, margin["top"] + plot_h + 32, _wrap_label(str(gate["gate"]), 16), size=13, anchor="middle"))
        parts.append(_text(px, margin["top"] + plot_h + 72, str(gate["diagnosis"])[:30], size=11, anchor="middle", fill="#565656"))
        if metric:
            parts.append(_text(px, py - 16 if index < 5 else py + 25, value_text, size=12, anchor="middle", fill="#222222"))
    parts.append(_title_text(str(data["title"])))
    parts.append(_text(90, 504, "Y axis: supported claim level; numeric gate metrics are attached to the JSON artifact.", size=13, fill="#555555"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_figure3_svg(data: dict[str, object]) -> str:
    rows = [_dict(row) for row in _list(data["rows"])]
    width, height = 1180, 580
    margin = {"left": 86, "right": 80, "top": 78, "bottom": 130}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    min_v, max_v = _value_extent(rows, include_zero=True)
    min_v, max_v = min(min_v, -0.075), max(max_v, 0.04)

    def y(value: float) -> float:
        return margin["top"] + (max_v - value) * plot_h / (max_v - min_v)

    group_w = plot_w / len(SEED_ORDER)
    bar_w = 54
    zero_y = y(0.0)
    parts = _svg_header(width, height, str(data["title"]))
    parts.extend(_axis_lines(margin["left"], margin["top"], plot_w, plot_h))
    parts.append(_line(margin["left"], zero_y, margin["left"] + plot_w, zero_y, "#222222", width=1.5))
    for tick in [-0.075, -0.05, -0.025, 0.0, 0.025]:
        yy = y(tick)
        parts.append(_line(margin["left"], yy, margin["left"] + plot_w, yy, "#e0e0e0", width=1))
        parts.append(_text(28, yy + 4, f"{tick:+.3f}", size=12, fill="#555555"))
    for seed_index, seed in enumerate(SEED_ORDER):
        seed_rows = [row for row in rows if row["seed"] == seed]
        center = margin["left"] + group_w * (seed_index + 0.5)
        for row_index, row in enumerate(seed_rows):
            bx = center + (row_index - 1) * 74 - bar_w / 2
            delta = float(row["delta"])
            top_y = y(max(delta, 0.0))
            bottom_y = y(min(delta, 0.0))
            parts.append(_rect(bx, top_y, bar_w, max(1, bottom_y - top_y), str(row["color"])))
            ci_x = bx + bar_w / 2
            parts.append(_error_bar(ci_x, y(float(row["ci_lower"])), y(float(row["ci_upper"]))))
        parts.append(_text(center, height - 72, seed, size=14, anchor="middle", fill="#222222"))
    parts.extend(
        _legend(
            690,
            72,
            [(label, color) for _, label, color in FIGURE3_COMPARISONS],
        )
    )
    parts.append(_title_text(str(data["title"])))
    parts.append(_text(86, 542, "Delta top10 with confidence intervals; main rows use locked encoder_layer_2 where applicable.", size=13, fill="#555555"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_figure4_svg(data: dict[str, object]) -> str:
    rows = [_dict(row) for row in _list(data["rows"])]
    width, height = 980, 460
    margin = {"left": 98, "right": 70, "top": 78, "bottom": 86}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    min_v, max_v = _value_extent(rows, include_zero=True)
    min_v, max_v = min(min_v, -0.1), max(max_v, 0.015)

    def x(value: float) -> float:
        return margin["left"] + (value - min_v) * plot_w / (max_v - min_v)

    parts = _svg_header(width, height, str(data["title"]))
    parts.extend(_axis_lines(margin["left"], margin["top"], plot_w, plot_h))
    zero_x = x(0.0)
    parts.append(_line(zero_x, margin["top"], zero_x, margin["top"] + plot_h, "#222222", width=1.5))
    for tick in [-0.1, -0.075, -0.05, -0.025, 0.0]:
        xx = x(tick)
        parts.append(_line(xx, margin["top"], xx, margin["top"] + plot_h, "#e0e0e0", width=1))
        parts.append(_text(xx, margin["top"] + plot_h + 28, f"{tick:+.3f}", size=12, anchor="middle", fill="#555555"))
    for index, row in enumerate(rows):
        cy = margin["top"] + 58 + index * 82
        parts.append(_text(28, cy + 5, str(row["seed"]), size=14, fill="#222222"))
        parts.append(_line(x(float(row["ci_lower"])), cy, x(float(row["ci_upper"])), cy, "#7a3f98", width=4))
        parts.append(_circle(x(float(row["delta"])), cy, 8, "#7a3f98"))
        parts.append(_text(x(float(row["delta"])) - 6, cy - 18, f"{float(row['delta']):+.3f}", size=12, anchor="end"))
    parts.append(_title_text(str(data["title"])))
    parts.append(_text(98, 428, "Matched-bucket bootstrap delta: residual hidden minus action/value+composition prior.", size=13, fill="#555555"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_figure5_svg(data: dict[str, object]) -> str:
    rows = [_dict(row) for row in _list(data["rows"])]
    width, height = 1180, 560
    margin = {"left": 82, "right": 70, "top": 78, "bottom": 122}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max(max(float(row["posterior_js"]), float(row["representation_js"])) for row in rows) * 1.15

    def y(value: float) -> float:
        return margin["top"] + (max_v - value) * plot_h / max_v

    group_w = plot_w / len(rows)
    bar_w = 34
    parts = _svg_header(width, height, str(data["title"]))
    parts.extend(_axis_lines(margin["left"], margin["top"], plot_w, plot_h))
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        yy = y(tick)
        parts.append(_line(margin["left"], yy, margin["left"] + plot_w, yy, "#e0e0e0", width=1))
        parts.append(_text(32, yy + 4, f"{tick:.1f}", size=12, fill="#555555"))
    for index, row in enumerate(rows):
        center = margin["left"] + group_w * (index + 0.5)
        for offset, field, color in [(-20, "posterior_js", "#2f5f9f"), (20, "representation_js", "#d07a24")]:
            value = float(row[field])
            bx = center + offset - bar_w / 2
            by = y(value)
            parts.append(_rect(bx, by, bar_w, y(0.0) - by, color))
        parts.append(_text(center, height - 82, _wrap_label(str(row["setting"]), 12), size=13, anchor="middle", fill="#222222"))
    parts.extend(_legend(850, 72, [("posterior JS", "#2f5f9f"), ("representation JS", "#d07a24")]))
    parts.append(_title_text(str(data["title"])))
    parts.append(_text(82, 526, "Ordered controls preserve representation divergence; shortcut controls collapse representation divergence.", size=13, fill="#555555"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _figure_pack_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# v1.16 Paper Figure Pack",
        "",
        f"Checksum: `{payload['checksum']}`",
        "",
        "## Figures",
        "",
        "| figure | title | svg | json |",
        "|---|---|---|---|",
    ]
    for figure in _list(payload["figures"]):
        item = _dict(figure)
        lines.append(
            f"| {item['figure_id']} | {item['title']} | `{item['svg_path']}` | `{item['json_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            "| source | path | sha256 |",
            "|---|---|---|",
        ]
    )
    for name, source in _dict(payload["sources"]).items():
        source_record = _dict(source)
        lines.append(f"| {name} | `{source_record['path']}` | `{source_record['sha256']}` |")
    return "\n".join(lines) + "\n"


def _source_record(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required figure source is missing: {path}")
    return {"path": str(path), "sha256": _text_sha256(path.read_text(encoding="utf-8"))}


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _first_case(report: dict[str, object]) -> dict[str, object]:
    cases = _list(report.get("cases"))
    if not cases:
        raise ValueError("report has no cases")
    return _dict(cases[0])


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object, got {type(value).__name__}")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _value_extent(rows: list[dict[str, object]], *, include_zero: bool) -> tuple[float, float]:
    values: list[float] = []
    for row in rows:
        for key in ("delta", "ci_lower", "ci_upper"):
            if key in row:
                values.append(float(row[key]))
    if include_zero:
        values.append(0.0)
    return min(values), max(values)


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;letter-spacing:0}.title{font-size:24px;font-weight:700;fill:#1f1f1f}</style>',
    ]


def _title_text(text: str) -> str:
    return _text(34, 42, text, size=24, css_class="title")


def _axis_lines(left: float, top: float, width: float, height: float) -> list[str]:
    return [
        _line(left, top, left, top + height, "#333333", width=1.5),
        _line(left, top + height, left + width, top + height, "#333333", width=1.5),
    ]


def _rect(x: float, y: float, width: float, height: float, fill: str, *, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{escape(fill)}" opacity="{opacity:.3f}"/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, *, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{escape(stroke)}" stroke-width="{width:.1f}"/>'
    )


def _circle(cx: float, cy: float, r: float, fill: str) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{escape(fill)}"/>'


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    anchor: str = "start",
    fill: str = "#222222",
    css_class: str | None = None,
) -> str:
    lines = str(text).split("\n")
    class_attr = f' class="{escape(css_class)}"' if css_class else ""
    if len(lines) == 1:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{escape(anchor)}" '
            f'fill="{escape(fill)}"{class_attr}>{escape(lines[0])}</text>'
        )
    tspan_lines = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size + 3
        tspan_lines.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{escape(anchor)}" '
        f'fill="{escape(fill)}"{class_attr}>{"".join(tspan_lines)}</text>'
    )


def _error_bar(x: float, y_lower: float, y_upper: float) -> str:
    return "\n".join(
        [
            _line(x, y_lower, x, y_upper, "#222222", width=1.6),
            _line(x - 8, y_lower, x + 8, y_lower, "#222222", width=1.6),
            _line(x - 8, y_upper, x + 8, y_upper, "#222222", width=1.6),
        ]
    )


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> list[str]:
    parts: list[str] = []
    for index, (label, color) in enumerate(items):
        yy = y + index * 22
        parts.append(_rect(x, yy - 11, 14, 14, color))
        parts.append(_text(x + 22, yy, label, size=12, fill="#333333"))
    return parts


def _wrap_label(label: str, max_len: int) -> str:
    words = label.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)
