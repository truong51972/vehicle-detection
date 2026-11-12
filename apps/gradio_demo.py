"""Local video demo. Launch from the repository with the separate `demo` extra."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr

from vehicle_detection.common import project_root
from vehicle_detection.experiment_data import load_experiment
from vehicle_detection.inference import (
    DEFAULT_LIMITS,
    EXPLORATORY,
    VERSIONS,
    VideoLimits,
    discover_models,
    infer_video,
)
from vehicle_detection.model_registry import discover_sign_models

MODES = ("Vehicle", "Traffic Sign", "Combined")


def build_app(root, *, device="cpu", limits=DEFAULT_LIMITS):
    root = Path(root).resolve()
    registry = discover_models(root)
    available = [version for version, entry in registry.items() if not entry.error]
    sign_registry = discover_sign_models(root)
    signs = [key for key, entry in sign_registry.items() if not entry.error]
    policies, policy_errors = {}, {}
    for version in VERSIONS:
        try:
            policies[version] = load_experiment(root, version)
        except Exception as exc:
            policy_errors[version] = str(exc)

    def pair_warning(model_version, policy_version, detection_mode="Vehicle"):
        if detection_mode == "Traffic Sign":
            return "Native / No additional filtering. This video demo does not measure AP."
        if model_version and policy_version and model_version != policy_version:
            return f"**{EXPLORATORY}**. Cross-policy results do not establish AP improvements."
        return "Matched model/policy. This video demo does not measure AP."

    with gr.Blocks(title="Road detection demo", delete_cache=(600, 3600)) as app:
        gr.Markdown(
            "# Road detection\nUpload once and detect vehicles, traffic signs, or both. "
            "Counts sum detections across frames, not unique tracked objects."
        )
        gr.Markdown(
            f"Limits: {limits.max_mb} MiB, {limits.max_seconds} seconds, "
            f"{limits.max_frames} frames, {limits.max_pixels:,} pixels/frame. "
            "Output keeps source dimensions and FPS; audio is omitted."
        )
        upload = gr.File(label="Upload MP4 / MOV / AVI", file_types=[".mp4", ".mov", ".avi"])
        mode = gr.Radio(MODES, value="Vehicle", label="Detection mode")
        with gr.Group() as vehicle_controls:
            with gr.Row():
                model = gr.Dropdown(
                    available,
                    value=available[0] if available else None,
                    label="Model version (unavailable versions excluded)",
                    interactive=bool(available),
                )
                policy = gr.Dropdown(
                    [(f"{v} — {config['name']}", v) for v, config in policies.items()],
                    value=next(iter(policies), None),
                    label="Preprocessing version",
                )
        with gr.Group(visible=False) as sign_controls:
            sign = gr.Dropdown(
                [(sign_registry[key].label, key) for key in signs],
                value=signs[0] if signs else None,
                label="Traffic Sign model (unavailable models excluded)",
                interactive=bool(signs),
            )
            gr.Markdown("Preprocessing: **Native / No additional filtering**")
        with gr.Row():
            confidence = gr.Slider(0, 1, value=0.25, step=0.01, label="Detection confidence")
            device_choice = gr.Dropdown(["cpu", "0"], value=device, label="Device (0 = CUDA GPU)")
        warning = gr.Markdown(pair_warning(model.value, policy.value))
        model.change(pair_warning, [model, policy, mode], warning, queue=False)
        policy.change(pair_warning, [model, policy, mode], warning, queue=False)
        with gr.Accordion("Checkpoint identities and unavailable versions", open=not available):
            gr.JSON(
                {v: entry.identity() for v, entry in (registry | sign_registry).items()},
                label="Models",
            )
            if policy_errors:
                gr.JSON(policy_errors, label="Unavailable policies")
        with gr.Row():
            run = gr.Button(
                "Run inference", variant="primary", interactive=bool(available and policies)
            )
            stop = gr.Button("Cancel active job")
        with gr.Row():
            original = gr.Video(label="Original video", interactive=False)
            annotated = gr.Video(label="Annotated video", interactive=False)
        download = gr.DownloadButton("Download annotated MP4")
        report = gr.JSON(label="Timing, detections and preprocessing report")
        cancel_state = gr.State({"requested": False})

        def change_mode(detection_mode, model_version, policy_version):
            needs_vehicle = detection_mode != "Traffic Sign"
            needs_sign = detection_mode != "Vehicle"
            ready = (not needs_vehicle or bool(available and policies)) and (
                not needs_sign or bool(signs)
            )
            return (
                gr.update(visible=needs_vehicle),
                gr.update(visible=needs_sign),
                gr.update(interactive=ready),
                pair_warning(model_version, policy_version, detection_mode),
            )

        mode.change(
            change_mode,
            [mode, model, policy],
            [vehicle_controls, sign_controls, run, warning],
            queue=False,
        )

        def run_job(
            source,
            model_version,
            policy_version,
            conf,
            selected_device,
            state,
            detection_mode="Vehicle",
            sign_version=None,
            progress=gr.Progress(),  # noqa: B008 — Gradio injects progress through the default.
        ):
            state["requested"] = False
            if not source:
                raise gr.Error("Upload an MP4, MOV or AVI video first")
            if detection_mode not in MODES:
                raise gr.Error("Select Vehicle, Traffic Sign or Combined mode")
            needs_vehicle = detection_mode != "Traffic Sign"
            needs_sign = detection_mode != "Vehicle"
            if needs_vehicle and (model_version not in available or policy_version not in policies):
                raise gr.Error("Select an available model and preprocessing version")
            if needs_sign and sign_version not in signs:
                reason = sign_registry.get(sign_version)
                raise gr.Error(
                    reason.error if reason else "Select an available Traffic Sign model; see Models"
                )
            try:
                with TemporaryDirectory(prefix="vehicle-demo-") as temporary:
                    output = Path(temporary) / "annotated.mp4"
                    progress(0, desc="Validating and loading model")
                    summary = infer_video(
                        source,
                        output,
                        registry[model_version] if needs_vehicle else sign_registry[sign_version],
                        policies[policy_version] if needs_vehicle else None,
                        sign_entry=(
                            sign_registry[sign_version] if detection_mode == "Combined" else None
                        ),
                        confidence=conf,
                        device=selected_device,
                        limits=limits,
                        cancelled=lambda: state["requested"],
                        progress=lambda done, total: progress(
                            min(done / total, 1) if total else None,
                            desc=f"Processed {done} frames" + (f" / {total}" if total else ""),
                        ),
                    )
                    cached = annotated.move_resource_to_block_cache(output)
                    return source, cached, cached, summary
            except (ValueError, OSError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        run.click(
            run_job,
            [upload, model, policy, confidence, device_choice, cancel_state, mode, sign],
            [original, annotated, download, report],
            concurrency_limit=1,
            concurrency_id="inference",
        )

        def cancel_job(state):
            state["requested"] = True

        stop.click(cancel_job, [cancel_state], queue=False)
    return app.queue(max_size=4, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "0"], default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-mb", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--max-pixels", type=int, default=3840 * 2160)
    parser.add_argument("--max-frames", type=int, default=7200)
    parser.add_argument(
        "--encoding-timeout", type=int, default=600, help="FFmpeg timeout in seconds"
    )
    args = parser.parse_args()
    limits = VideoLimits(
        args.max_mb, args.max_seconds, args.max_pixels, args.max_frames, args.encoding_timeout
    )
    app = build_app(project_root(), device=args.device, limits=limits)
    app.launch(server_name="127.0.0.1", server_port=args.port, max_file_size=f"{args.max_mb}mb")


if __name__ == "__main__":
    main()
