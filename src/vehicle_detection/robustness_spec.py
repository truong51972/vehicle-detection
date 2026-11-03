"""Metadata-only final-test cost/specification guard; this module opens no image pixels."""

import itertools

import numpy as np

from .common import digest, file_hash
from .protocol import validate_protocol


def final_stress_spec(
    protocol, source_study, matrix, policy_id, confirmation_gate_id, *, support_policy
):
    validate_protocol(protocol)
    # Cover three distinct failure mechanisms without promoting screened operators to routes.
    conditions = [("clean", 0), *itertools.product(("shot_read", "motion", "veil"), (2, 3))]
    treatments = ["Raw", "Gamma-or-Skip"]
    models = ["pretrained YOLO26n", "Raw-trained YOLO26n", "Gamma-or-Skip-trained YOLO26n"]
    reserved = protocol["splits"]["final_test"]
    count = len(reserved)
    dataset_count = len(conditions) * len(treatments)
    entries = [
        {
            "family": family,
            "severity": severity,
            "treatment": treatment,
            "model": model,
            "images": count,
        }
        for (family, severity), treatment, model in itertools.product(
            conditions, treatments, models
        )
    ]
    required_columns = {
        "image_GiB",
        "images",
        "model_inference_ms",
        "corruption_p95_ms",
        "restore_p95_ms",
        "operator",
        "family",
        "severity",
    }
    if not required_columns <= set(matrix.columns) or (matrix.images <= 0).any():
        raise ValueError("Measured calibration costs are required")
    columns = ["image_GiB", "model_inference_ms", "corruption_p95_ms", "restore_p95_ms"]
    if not np.isfinite(matrix[columns].to_numpy()).all() or (matrix[columns] < 0).any().any():
        raise ValueError("Invalid calibration cost measurements")
    largest_mean_gib = float((matrix.image_GiB / matrix.images).max())
    inference_ms = float(matrix.model_inference_ms.max())
    restore_ms = float(matrix.loc[matrix.operator == "adaptive_v2", "restore_p95_ms"].max())
    corruption_seconds = 0.0
    for family, severity in conditions:
        measured = matrix[(matrix.family == family) & (matrix.severity == severity)]
        if measured.empty:
            raise ValueError("Missing calibration condition cost")
        corruption_seconds += float(measured.corruption_p95_ms.max()) * count * 2 / 1000
    if not np.isfinite(restore_ms):
        raise ValueError("Missing Gamma-or-Skip cost")
    spec = {
        "version": "issue11-final-stress-spec-v2",
        "role": "final_test",
        "status": "specification_only_not_executed",
        "protocol_id": protocol["manifest_id"],
        "reserved_images": count,
        "reservation_metadata_id": digest(
            [(r["id"], r["group"], r["labels_sha256"]) for r in reserved]
        ),
        "models": models,
        "treatments": treatments,
        "policy_label": "Gamma-or-Skip",
        "policy_id": policy_id,
        "confirmation_gate_id": confirmation_gate_id,
        "conditions": [{"family": f, "severity": s} for f, s in conditions],
        "corruptions": {f: source_study["corruptions"][f] for f in ("shot_read", "motion", "veil")},
        "seed": 2026,
        "encoding": {"format": "jpg", "quality": 95},
        "settings": source_study["settings"],
        "evaluation_matrix": entries,
        "natural_benchmark": "six clean evaluations; natural prevalence; separate reporting",
        "synthetic_benchmark": "36 evaluations; never blend with natural clean AP",
        "source_id_parity": "identical reserved source IDs/groups/GT in every condition/treatment",
        "primary_metric": "native descriptive mAP50:95; fixed supported-class recovery inference",
        "uncertainty": {
            **{
                k: support_policy[k]
                for k in ("version", "minimum_source_groups", "missing_class_draw", "limit")
            },
            "class_set": "freeze from final-source GT support before comparing models/conditions",
            "excluded_classes": "descriptive AP only; never impute absent-class AP",
            "point_and_gate": "same target as CI; audit invalid draws and per-class support",
        },
        "excluded": "all screened standalone operators; defocus/exposure stay calibration-only",
        "budget": {
            "dataset_variants": dataset_count,
            "evaluations": len(entries),
            "natural_image_passes": count * 6,
            "synthetic_image_passes": count * 36,
            "total_image_passes": count * len(entries),
            "prepared_image_files": count * dataset_count,
            "max_simultaneous_datasets": 2,
            "disk_policy": "one condition pair at a time; all three models before image cleanup",
            "image_size_basis": "largest pilot mean; 1.5x headroom; not a worst-case bound",
            "full_retention_image_GiB": largest_mean_gib * count * dataset_count * 1.5,
            "streaming_image_GiB": largest_mean_gib * count * 2 * 1.5,
            "scratch_GiB": 5,
            "model_stage_basis_ms": inference_ms,
            "model_stage_estimate_seconds": inference_ms * count * len(entries) / 1000 * 2,
            "model_stage_headroom_factor": 2,
            "serial_corruption_estimate_seconds": corruption_seconds,
            "serial_router_estimate_seconds": restore_ms * count * len(conditions) / 1000,
            "cost_limitations": (
                "calibration extrapolation; excludes I/O, codec, AP/plots and startup; "
                "not end-to-end"
            ),
        },
        "release_gate": [
            "freeze exact trained checkpoint hashes and verify semantic class mappings",
            "freeze model/policy/protocol/spec before any final results; no retuning afterward",
            "complete final pixel/geometry/content leakage checks; no silent ID drops",
            "identity-check checkpoint/data/settings/code and retained prediction-statistic caches",
            "enforce this exact matrix; expanding scope requires a new version before release",
            "one final release; unchanged identity cache replay only, never test-driven selection",
        ],
        "code_sha256": file_hash(__file__),
    }
    spec["spec_id"] = digest(spec)
    return spec
