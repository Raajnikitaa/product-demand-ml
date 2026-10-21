"""Build a portable report and source bundle for the trained DemandWise model."""

from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


SOURCE_FILES = (
    "train_model.py",
    "feature_engineering.py",
    "evaluate_model.py",
    "preprocessing.py",
    "data_loader.py",
)


def _svg_roc_chart(metadata: Dict[str, Any]) -> str:
    curves = metadata.get("curves", {})
    roc = curves.get("roc", {}) if curves.get("available") else {}
    fpr, tpr = roc.get("fpr", []), roc.get("tpr", [])
    if not fpr or not tpr or len(fpr) != len(tpr):
        return "<p>ROC chart is not available for this evaluation.</p>"

    width, height = 640, 360
    left, top, plot_width, plot_height = 54, 22, 550, 280
    points = " ".join(
        f"{left + float(x) * plot_width:.1f},{top + (1 - float(y)) * plot_height:.1f}"
        for x, y in zip(fpr, tpr)
    )
    diagonal_end_x = left + plot_width
    diagonal_end_y = top
    return (
        f'<svg role="img" aria-label="ROC curve" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{diagonal_end_x}" y2="{diagonal_end_y}" '
        'stroke="#CBD5E1" stroke-dasharray="6 5"/>'
        f'<polyline points="{points}" fill="none" stroke="#15803D" stroke-width="3"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748B"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#64748B"/>'
        f'<text x="{left + plot_width / 2}" y="{height - 14}" text-anchor="middle">False positive rate</text>'
        f'<text x="16" y="{top + plot_height / 2}" transform="rotate(-90 16 {top + plot_height / 2})" '
        'text-anchor="middle">True positive rate</text>'
        f'<text x="{left - 8}" y="{top + plot_height + 18}" text-anchor="end">0</text>'
        f'<text x="{left + plot_width}" y="{top + plot_height + 18}" text-anchor="middle">1</text>'
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end">1</text></svg>'
    )


def _svg_confusion_matrix(metadata: Dict[str, Any]) -> str:
    matrix = (
        metadata.get("test_metrics", {})
        .get("confusion_matrix", {})
        .get("matrix", [])
    )
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        return "<p>Confusion matrix is not available.</p>"

    labels = (("Correct normal", "Missed surge"), ("False alert", "Detected surge"))
    cells = []
    for row in range(2):
        for col in range(2):
            x, y = 20 + col * 220, 16 + row * 106
            fill = "#DCFCE7" if row == col else "#FEF3C7"
            cells.append(
                f'<rect x="{x}" y="{y}" width="204" height="90" rx="9" fill="{fill}"/>'
                f'<text x="{x + 102}" y="{y + 37}" text-anchor="middle" class="count">{int(matrix[row][col]):,}</text>'
                f'<text x="{x + 102}" y="{y + 66}" text-anchor="middle">{labels[row][col]}</text>'
            )
    return (
        '<svg role="img" aria-label="Confusion matrix" viewBox="0 0 464 230" '
        'xmlns="http://www.w3.org/2000/svg">'
        + "".join(cells)
        + "</svg>"
    )


def _svg_coefficients(metadata: Dict[str, Any]) -> str:
    coefficients = metadata.get("coefficients", [])[:12]
    if not coefficients:
        return "<p>Feature coefficients are not available.</p>"

    max_value = max((abs(float(item.get("coefficient", 0))) for item in coefficients), default=0) or 1
    width, row_height, label_width, bar_width = 760, 34, 235, 480
    center = label_width + bar_width / 2
    rows = []
    for index, item in enumerate(coefficients):
        y = index * row_height + 8
        value = float(item.get("coefficient", 0))
        length = abs(value) / max_value * (bar_width / 2 - 8)
        x = center if value >= 0 else center - length
        color = "#15803D" if value >= 0 else "#D97706"
        label = html.escape(str(item.get("feature", "")))
        rows.append(
            f'<text x="{label_width - 10}" y="{y + 17}" text-anchor="end">{label}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="20" rx="4" fill="{color}"/>'
            f'<text x="{label_width + bar_width + 8}" y="{y + 16}">{value:.4f}</text>'
        )
    return (
        f'<svg role="img" aria-label="Largest model coefficients" viewBox="0 0 {width} '
        f'{len(coefficients) * row_height + 8}" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{center}" y1="0" x2="{center}" y2="{len(coefficients) * row_height}" stroke="#CBD5E1"/>'
        + "".join(rows)
        + "</svg>"
    )


def build_model_report_html(metadata: Dict[str, Any], pipeline: Any) -> str:
    """Create a self-contained HTML report with metrics, SVG charts, parameters, and code."""
    metrics = metadata.get("test_metrics", {})
    split = metadata.get("split_info", {})
    classifier = pipeline.named_steps["classifier"]
    model_parameters = {
        "Algorithm": metadata.get("algorithm", "Logistic Regression"),
        "Regularization C": classifier.C,
        "Solver": classifier.solver,
        "Class weight": classifier.class_weight,
        "Maximum iterations": classifier.max_iter,
        "Random state": classifier.random_state,
        "Decision threshold": metadata.get("optimal_decision_threshold", 0.5),
        "Surge definition": metadata.get("surge_threshold_pct", "+30%"),
        "Forecast horizon": "Next 7 days",
    }
    parameter_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in model_parameters.items()
    )

    metric_values = (
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1 score", metrics.get("f1_score")),
        ("ROC-AUC", metrics.get("roc_auc")),
        ("PR-AUC", metrics.get("pr_auc")),
    )
    metric_cards = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>'
        f'{float(value) * 100:.2f}%</strong></div>'
        for label, value in metric_values
        if value is not None
    )

    source_path = Path(__file__).with_name("train_model.py")
    source_code = html.escape(source_path.read_text(encoding="utf-8"))
    trained_on = html.escape(str(metadata.get("data_type", "Retail sales data")))
    test_range = html.escape(f"{split.get('test_start', 'N/A')} to {split.get('test_end', 'N/A')}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DemandWise Model Report</title>
<style>
body{{font:15px/1.55 Arial,sans-serif;color:#1f2937;background:#f8fafc;margin:0}}
main{{max-width:1000px;margin:32px auto;padding:0 20px}}
h1,h2{{color:#14532d}}h1{{margin-bottom:4px}}.muted{{color:#64748b}}
section{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:18px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}}
.metric{{border:1px solid #e2e8f0;border-radius:10px;padding:14px}}
.metric span{{display:block;color:#64748b;font-size:12px}}.metric strong{{font-size:22px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}}
th{{width:40%;color:#475569}}svg{{width:100%;max-height:420px}}svg text{{font:12px Arial,sans-serif;fill:#475569}}
svg .count{{font-size:22px;font-weight:bold;fill:#1f2937}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;max-height:650px;overflow:auto}}
@media print{{body{{background:#fff}}main{{margin:0 auto}}section{{break-inside:avoid}}}}
</style></head><body><main>
<h1>DemandWise Model Report</h1>
<div class="muted">Trained: {html.escape(str(metadata.get('training_timestamp', 'N/A')))} · Data: {trained_on}</div>
<section><h2>Test performance</h2><p>Held-out dates: {test_range} · Records: {int(split.get('test_rows', 0)):,}</p><div class="metrics">{metric_cards}</div></section>
<section><h2>Model settings</h2><table>{parameter_rows}</table></section>
<section><h2>Evaluation graphs</h2><h3>ROC curve</h3>{_svg_roc_chart(metadata)}
<h3>Test confusion matrix</h3>{_svg_confusion_matrix(metadata)}
<h3>Largest feature coefficients</h3>{_svg_coefficients(metadata)}</section>
<section><h2>Training code: train_model.py</h2><p class="muted">The saved pipeline and related source files are included in this ZIP bundle.</p>
<details><summary>Show training pipeline source</summary><pre>{source_code}</pre></details></section>
</main></body></html>"""


def create_model_bundle(
    metadata: Dict[str, Any],
    pipeline: Any,
    models_dir: Optional[str] = None,
) -> bytes:
    """Package the trained estimator, metadata, report, and required Python source files."""
    project_dir = Path(__file__).resolve().parent.parent
    artifact_dir = Path(models_dir) if models_dir else project_dir / "models"
    artifact_name = Path(
        metadata.get("pipeline_artifact", "model_pipeline.joblib")
    ).name
    model_path = artifact_dir / artifact_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model artifact was not found: {model_path}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model_path, "model_pipeline.joblib")
        bundle_metadata = dict(metadata)
        bundle_metadata["pipeline_artifact"] = "model_pipeline.joblib"
        bundle.writestr("model_metadata.json", json.dumps(bundle_metadata, indent=2))
        bundle.writestr("Model_Report.html", build_model_report_html(metadata, pipeline))
        for name in SOURCE_FILES:
            source_path = project_dir / "src" / name
            bundle.write(source_path, f"src/{name}")
        requirements_path = project_dir / "requirements.txt"
        if requirements_path.is_file():
            bundle.write(requirements_path, "requirements.txt")
    return buffer.getvalue()
