#!/usr/bin/env python3
"""Web UI for the YouTube Japanese dubbing pipeline.

Usage:
    python web.py [--port 8080] [--host 0.0.0.0]

Opens a browser-based interface for the dubbing pipeline.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
import uuid
import webbrowser

from flask import Flask, render_template, request, jsonify, Response, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB upload limit

# Store running jobs: job_id -> {status, result, error, events_queue, ...}
jobs: dict[str, dict] = {}

logger = logging.getLogger("dubbing.web")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_job():
    """Start a new dubbing job."""
    job_id = uuid.uuid4().hex[:12]
    events = queue.Queue()

    # Collect parameters
    audio_mode = request.form.get("audio", "mute")
    subtitle_mode = request.form.get("subtitle", "soft")
    whisper_model = request.form.get("whisper_model", "large-v3")
    keep_tmp = request.form.get("keep_tmp", "false") == "true"

    # Determine input
    url = request.form.get("url", "").strip()
    uploaded = request.files.get("file")

    if url:
        input_path = url
    elif uploaded and uploaded.filename:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, f"{job_id}_{uploaded.filename}")
        uploaded.save(save_path)
        input_path = save_path
    else:
        return jsonify({"error": "URLまたはファイルが指定されていません"}), 400

    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    job = {
        "status": "running",
        "result": None,
        "error": None,
        "events": events,
    }
    jobs[job_id] = job

    # Run pipeline in background thread
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, job_output_dir, audio_mode, subtitle_mode,
              whisper_model, keep_tmp),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id, input_path, output_dir, audio_mode, subtitle_mode,
             whisper_model, keep_tmp):
    """Run the dubbing pipeline in a background thread."""
    job = jobs[job_id]
    eq = job["events"]

    # Custom log handler that pushes to SSE
    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                eq.put({"log": self.format(record)})
            except Exception:
                pass

    handler = QueueHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger("dubbing").addHandler(handler)
    logging.getLogger("pipeline").addHandler(handler)

    def on_progress(step, total, message):
        eq.put({"step": step, "total": total, "message": message})

    try:
        from dubbing import run_pipeline

        result = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            audio_mode=audio_mode,
            subtitle_mode=subtitle_mode,
            whisper_model=whisper_model,
            keep_tmp=keep_tmp,
            on_progress=on_progress,
        )
        job["status"] = "done"
        job["result"] = result
        eq.put({"status": "done"})
    except Exception as e:
        logger.exception("Pipeline error for job %s", job_id)
        job["status"] = "error"
        job["error"] = str(e)
        eq.put({"status": "error", "error": str(e)})
    finally:
        logging.getLogger("dubbing").removeHandler(handler)
        logging.getLogger("pipeline").removeHandler(handler)


@app.route("/api/progress/<job_id>")
def progress(job_id):
    """SSE endpoint for real-time progress updates."""
    if job_id not in jobs:
        return jsonify({"error": "ジョブが見つかりません"}), 404

    def generate():
        eq = jobs[job_id]["events"]
        while True:
            try:
                event = eq.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("done", "error"):
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'keepalive': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/download/<job_id>/<file_type>")
def download(job_id, file_type):
    """Download output files."""
    if job_id not in jobs:
        return jsonify({"error": "ジョブが見つかりません"}), 404

    job = jobs[job_id]
    if job["status"] != "done" or not job["result"]:
        return jsonify({"error": "ジョブが未完了です"}), 400

    if file_type == "video":
        path = job["result"]["video"]
        return send_file(path, as_attachment=True)
    elif file_type == "srt":
        path = job["result"]["srt"]
        return send_file(path, as_attachment=True)
    else:
        return jsonify({"error": "無効なファイル種別です"}), 400


def parse_args():
    parser = argparse.ArgumentParser(description="Dubbing pipeline Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="Port. Default: 8080")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"\n  Dubbing Web UI: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
