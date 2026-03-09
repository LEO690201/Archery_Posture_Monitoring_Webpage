from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pose_service import UPLOAD_DIR, WEB_DATA_DIR, analyze_video, init_web_dirs, revert_current_result

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None, notice=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    init_web_dirs()
    upload = request.files.get("video")
    athlete_name = (request.form.get("athlete_name") or "").strip()

    if upload is None or upload.filename == "":
        return render_template("index.html", result=None, error="请先选择一个视频文件。", notice=None)

    if not allowed_file(upload.filename):
        return render_template("index.html", result=None, error="仅支持 mp4、mov、avi、mkv、webm 格式。", notice=None)

    original_name = secure_filename(upload.filename)
    filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
    save_path = UPLOAD_DIR / filename
    upload.save(save_path)

    try:
        result = analyze_video(save_path, athlete_name or None)
        return render_template("index.html", result=result, error=None, notice=None)
    except Exception as exc:
        return render_template("index.html", result=None, error=f"分析失败：{exc}", notice=None)


@app.route("/result-action", methods=["POST"])
def result_action():
    action = (request.form.get("action") or "").strip()
    upload_path = (request.form.get("current_upload_path") or "").strip() or None
    run_dir = (request.form.get("current_run_dir") or "").strip() or None
    profile_file = (request.form.get("profile_file") or "").strip() or None
    profile_previous_line_count_raw = (request.form.get("profile_previous_line_count") or "").strip()
    profile_previous_line_count = int(profile_previous_line_count_raw) if profile_previous_line_count_raw else None

    if action == "discard":
        absolute_profile_file = None
        if profile_file:
            absolute_profile_file = str((WEB_DATA_DIR / profile_file).resolve())
        revert_current_result(upload_path, run_dir, absolute_profile_file, profile_previous_line_count)
        return render_template(
            "index.html",
            result=None,
            error=None,
            notice="本次结果未保存，当前这一次分析产生的数据已撤销。",
        )

    if action == "save":
        return render_template(
            "index.html",
            result=None,
            error=None,
            notice="本次结果已保留在本机，可继续上传新视频或下次再查看。",
        )

    return render_template("index.html", result=None, error="未识别的操作。", notice=None)


@app.route("/files/<path:filename>", methods=["GET"])
def files(filename: str):
    return send_from_directory(WEB_DATA_DIR, filename)


if __name__ == "__main__":
    init_web_dirs()
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
