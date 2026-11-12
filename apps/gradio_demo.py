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


def build_app(root, *, device="cpu", limits=DEFAULT_LIMITS):
    root = Path(root).resolve()
    registry = discover_models(root)
    available = [version for version, entry in registry.items() if not entry.error]
    policies, policy_errors = {}, {}
    for version in VERSIONS:
        try:
            policies[version] = load_experiment(root, version)
        except Exception as exc:
            policy_errors[version] = str(exc)

    def pair_warning(model_version, policy_version):
        if model_version and policy_version and model_version != policy_version:
            return f"**{EXPLORATORY}**. Cross-policy results do not establish AP improvements."
        return "Matched model/policy. This video demo does not measure AP."

    with gr.Blocks(title="Vehicle detection demo", delete_cache=(600, 3600)) as app:
        gr.Markdown(
            "# Vehicle detection\nUpload a video and choose a model and preprocessing policy."
        )
        gr.Markdown(
            f"Limits: {limits.max_mb} MiB, {limits.max_seconds} seconds, "
            f"{limits.max_frames} frames, {limits.max_pixels:,} pixels/frame. "
            "Output keeps source dimensions and FPS; audio is omitted."
        )
        upload = gr.File(label="Upload MP4 / MOV / AVI", file_types=[".mp4", ".mov", ".avi"])
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
            confidence = gr.Slider(0, 1, value=0.25, step=0.01, label="Detection confidence")
            device_choice = gr.Dropdown(["cpu", "0"], value=device, label="Device (0 = CUDA GPU)")
        warning = gr.Markdown(pair_warning(model.value, policy.value))
        model.change(pair_warning, [model, policy], warning, queue=False)
        policy.change(pair_warning, [model, policy], warning, queue=False)
        with gr.Accordion("Checkpoint identities and unavailable versions", open=not available):
            gr.JSON({v: entry.identity() for v, entry in registry.items()}, label="Models")
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

        def run_job(
            source,
            model_version,
            policy_version,
            conf,
            selected_device,
            state,
            progress=gr.Progress(),  # noqa: B008 — Gradio injects progress through the default.
        ):
            state["requested"] = False
            if not source:
                raise gr.Error("Upload an MP4, MOV or AVI video first")
            if model_version not in available or policy_version not in policies:
                raise gr.Error("Select an available model and preprocessing version")
            try:
                with TemporaryDirectory(prefix="vehicle-demo-") as temporary:
                    output = Path(temporary) / f"{model_version}-{policy_version}.mp4"
                    progress(0, desc="Validating and loading model")
                    summary = infer_video(
                        source,
                        output,
                        registry[model_version],
                        policies[policy_version],
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
            [upload, model, policy, confidence, device_choice, cancel_state],
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
