from __future__ import annotations

import html
import os
import shutil
import uuid
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

import angle
import yolo
import yolo_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
WEB_DATA_DIR = BASE_DIR / "web_data"
UPLOAD_DIR = WEB_DATA_DIR / "uploads"
RUNS_DIR = WEB_DATA_DIR / "runs"
PROFILE_DIR = WEB_DATA_DIR / "profiles"
REPORT_DIR = WEB_DATA_DIR / "reports"


def init_web_dirs() -> None:
    for path in [WEB_DATA_DIR, UPLOAD_DIR, RUNS_DIR, PROFILE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def secure_stem(value: str, fallback: str = "user") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback



def calculate_archery_score(front_angle: float, behind_angle: float) -> float:
    if front_angle >= 175:
        front_score = 100
    elif front_angle >= 160:
        front_score = 70 + (front_angle - 160) / 15 * 30
    elif front_angle >= 140:
        front_score = 40 + (front_angle - 140) / 20 * 30
    else:
        front_score = max(0, front_angle / 140 * 40)

    if behind_angle <= 15:
        behind_score = 100
    elif behind_angle <= 30:
        behind_score = 70 + (30 - behind_angle) / 15 * 30
    elif behind_angle <= 60:
        behind_score = 40 + (60 - behind_angle) / 30 * 30
    else:
        behind_score = max(0, (90 - behind_angle) / 90 * 40)

    angle_diff = abs(180 - front_angle - behind_angle)
    if angle_diff <= 20:
        coordination_score = 100
    elif angle_diff <= 40:
        coordination_score = 70 + (40 - angle_diff) / 20 * 30
    elif angle_diff <= 70:
        coordination_score = 40 + (70 - angle_diff) / 30 * 30
    else:
        coordination_score = max(0, (100 - angle_diff) / 100 * 40)

    return front_score * 0.40 + behind_score * 0.35 + coordination_score * 0.25


def get_score_grade(score: float) -> str:
    if score >= 90:
        return "大师级"
    if score >= 80:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 60:
        return "合格"
    return "需改进"


def build_evaluation(front_angle: float, behind_angle: float, score: float) -> dict:
    suggestions: list[str] = []
    issues: list[str] = []

    if front_angle >= 175:
        suggestions.append("前臂拉弓角度优秀。")
    elif front_angle >= 160:
        issues.append("前臂未完全伸直，建议加强伸展。")
    else:
        issues.append("前臂弯曲过大，会明显影响发力稳定性。")

    if behind_angle <= 15:
        suggestions.append("后臂拉弦直线度优秀。")
    elif behind_angle <= 30:
        issues.append("后臂略有弯曲，注意保持更稳定的拉弦直线。")
    else:
        issues.append("后臂弯曲过大，容易影响箭矢飞行方向。")

    if not issues and suggestions:
        suggestions.append("姿态接近理想状态，可以继续保持。")

    return {
        "grade": get_score_grade(score),
        "issues": issues,
        "suggestions": suggestions,
        "html": (
            "<div class='analysis-block'>"
            f"<p><strong>前臂角度：</strong>{front_angle}°</p>"
            f"<p><strong>后臂角度：</strong>{behind_angle}°</p>"
            f"<p><strong>综合评分：</strong>{score} 分</p>"
            f"<p><strong>等级：</strong>{html.escape(get_score_grade(score))}</p>"
            "</div>"
        ),
    }


def decide_front_and_behind(left_arm_angle: float, right_arm_angle: float) -> tuple[float, float, str]:
    if left_arm_angle > 90 and right_arm_angle <= 90:
        return left_arm_angle, right_arm_angle, "左手为前手，右手为后手"
    if right_arm_angle > 90 and left_arm_angle <= 90:
        return right_arm_angle, left_arm_angle, "右手为前手，左手为后手"
    if left_arm_angle >= right_arm_angle:
        return left_arm_angle, right_arm_angle, "按角度大小判断左手为前手"
    return right_arm_angle, left_arm_angle, "按角度大小判断右手为前手"


def save_profile_record(name: str, front_angle: float, behind_angle: float, score: float) -> tuple[Path, int]:
    safe_name = secure_stem(name, "athlete")
    profile_path = PROFILE_DIR / f"{safe_name}.txt"
    previous_line_count = 0
    if profile_path.exists():
        with profile_path.open("r", encoding="utf-8") as file:
            previous_line_count = sum(1 for _ in file)
    with profile_path.open("a", encoding="utf-8") as file:
        file.write(f"{front_angle} {behind_angle} {score}\n")
    return profile_path, previous_line_count


def revert_current_result(
    upload_path: str | None,
    run_dir: str | None,
    profile_path: str | None,
    profile_previous_line_count: int | None,
) -> None:
    if upload_path:
        Path(upload_path).unlink(missing_ok=True)

    if run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)

    if not profile_path:
        return

    target = Path(profile_path)
    chart_dir = REPORT_DIR / secure_stem(target.stem, "athlete")
    if not target.exists():
        shutil.rmtree(chart_dir, ignore_errors=True)
        return

    previous_count = max(profile_previous_line_count or 0, 0)
    with target.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    if previous_count <= 0:
        target.unlink(missing_ok=True)
        shutil.rmtree(chart_dir, ignore_errors=True)
        return

    with target.open("w", encoding="utf-8") as file:
        file.writelines(lines[:previous_count])

    if previous_count >= 1:
        generate_profile_charts(target)
    else:
        shutil.rmtree(chart_dir, ignore_errors=True)


def generate_profile_charts(profile_path: Path) -> dict[str, str]:
    raw_numbers: list[float] = []
    with profile_path.open("r", encoding="utf-8") as file:
        for line in file:
            for token in line.split():
                raw_numbers.append(float(token))

    if len(raw_numbers) < 3:
        return {}

    group1 = raw_numbers[0::3]
    group2 = [-value for value in raw_numbers[1::3]]
    group3 = raw_numbers[2::3]

    safe_name = secure_stem(profile_path.stem, "athlete")
    chart_dir = REPORT_DIR / safe_name
    chart_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    charts = {
        "front": (group1, "Front Arm Angle", "#0f766e"),
        "behind": (group2, "Rear Arm Angle", "#b45309"),
        "score": (group3, "Overall Score", "#1d4ed8"),
    }
    result: dict[str, str] = {}
    for key, (data, title, color) in charts.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(data, marker="o", linewidth=1.8, color=color)
        ax.set_title(title)
        ax.set_xlabel("Record Index")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        output_path = chart_dir / f"{key}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result[key] = output_path.relative_to(WEB_DATA_DIR).as_posix()
    return result


def build_loop_gif(image_paths: list[Path], output_path: Path, duration_ms: int = 180) -> str | None:
    frames: list[Image.Image] = []
    for path in image_paths:
        if not path.exists():
            continue
        with Image.open(path) as img:
            frames.append(img.convert("RGB"))

    if len(frames) < 2:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path.relative_to(WEB_DATA_DIR).as_posix()


def frame_relative_path(path: Path) -> str:
    return path.relative_to(WEB_DATA_DIR).as_posix()


def analyze_video(video_path: str | os.PathLike[str], athlete_name: str | None = None) -> dict:
    init_web_dirs()

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    output_dir = run_dir / "output_frames"
    skeleton_dir = run_dir / "black"
    original_dir = run_dir / "original"
    for path in [run_dir, output_dir, skeleton_dir, original_dir]:
        path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("无法打开上传的视频文件。")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    process_count = 0
    left_wrist_y_last = 0
    left_wrist_x_last = 0
    right_wrist_y_last = 0
    right_wrist_x_last = 0
    feature_values: list[float] = [1000.0]
    processed_frame_numbers: list[int] = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if process_count == 10:
            frame_filename = output_dir / f"frame_{frame_count:04}.png"
            skeleton_filename = skeleton_dir / f"frame_{frame_count:04}.png"
            original_filename = original_dir / f"frame_{frame_count:04}.png"

            cv2.imwrite(str(original_filename), frame)
            cv2.imwrite(str(frame_filename), frame)

            (
                frame_processed,
                left_wrist_x,
                left_wrist_y,
                left_elbow_x,
                left_elbow_y,
                right_wrist_x,
                right_wrist_y,
                right_elbow_x,
                right_elbow_y,
                black_img,
            ) = yolo.yolo(str(frame_filename))

            cv2.imwrite(str(frame_filename), frame_processed)
            cv2.imwrite(str(skeleton_filename), black_img)

            left_displacement = (left_wrist_y - left_wrist_y_last) ** 2 + (left_wrist_x - left_wrist_x_last) ** 2
            right_displacement = (right_wrist_y - right_wrist_y_last) ** 2 + (right_wrist_x - right_wrist_x_last) ** 2

            if left_displacement >= right_displacement:
                wrist_x, wrist_y = left_wrist_x, left_wrist_y
                elbow_x, elbow_y = left_elbow_x, left_elbow_y
                wrist_x_last, wrist_y_last = left_wrist_x_last, left_wrist_y_last
            else:
                wrist_x, wrist_y = right_wrist_x, right_wrist_y
                elbow_x, elbow_y = right_elbow_x, right_elbow_y
                wrist_x_last, wrist_y_last = right_wrist_x_last, right_wrist_y_last

            numerator = (elbow_y - wrist_y) ** 2 + (elbow_x - wrist_x) ** 2
            denominator = (wrist_y - wrist_y_last) ** 2 + (wrist_x - wrist_x_last) ** 2 + 0.001
            feature_values.append(numerator / denominator)
            processed_frame_numbers.append(frame_count)

            process_count = 0
            left_wrist_x_last, left_wrist_y_last = left_wrist_x, left_wrist_y
            right_wrist_x_last, right_wrist_y_last = right_wrist_x, right_wrist_y

        frame_count += 1
        process_count += 1

    capture.release()

    if not processed_frame_numbers:
        raise ValueError("视频内容不足，未能提取到可分析的有效帧。")

    min_index = 1 if len(feature_values) > 1 else 0
    min_value = feature_values[min_index]
    for idx in range(2, len(feature_values)):
        if feature_values[idx] < min_value:
            min_value = feature_values[idx]
            min_index = idx

    frame_position = min(min_index - 1, len(processed_frame_numbers) - 1) if min_index > 0 else 0
    start_frame_number = processed_frame_numbers[0]
    key_frame_number = processed_frame_numbers[frame_position]
    start_image_name = f"frame_{start_frame_number:04}.png"
    shot_image_name = f"frame_{key_frame_number:04}.png"
    start_frame_path = output_dir / start_image_name
    start_skeleton_path = skeleton_dir / start_image_name
    key_frame_path = output_dir / shot_image_name
    skeleton_path = skeleton_dir / shot_image_name

    combined_gif_path = build_loop_gif(
        [output_dir / f"frame_{frame_no:04}.png" for frame_no in processed_frame_numbers],
        run_dir / "combined_loop.gif",
    )
    skeleton_gif_path = build_loop_gif(
        [skeleton_dir / f"frame_{frame_no:04}.png" for frame_no in processed_frame_numbers],
        run_dir / "skeleton_loop.gif",
    )

    left_wrist, left_elbow, right_elbow, right_shoulder, right_wrist = yolo_2.yolo(str(key_frame_path))
    left_arm_angle = round(
        angle.calculate_angle(right_shoulder, left_elbow, left_wrist, (left_elbow[0] + 1, left_elbow[1])),
        1,
    )
    right_arm_angle = round(
        angle.calculate_angle(right_shoulder, right_elbow, right_wrist, (right_elbow[0] + 1, right_elbow[1])),
        1,
    )
    front_arm_angle, behind_arm_angle, handedness = decide_front_and_behind(left_arm_angle, right_arm_angle)
    score = round(calculate_archery_score(front_arm_angle, behind_arm_angle), 1)
    evaluation = build_evaluation(front_arm_angle, behind_arm_angle, score)

    result = {
        "run_id": run_id,
        "total_frames": total_frames,
        "processed_frame_count": len(processed_frame_numbers),
        "start_frame_number": start_frame_number,
        "key_frame_number": key_frame_number,
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "front_arm_angle": front_arm_angle,
        "behind_arm_angle": behind_arm_angle,
        "score": score,
        "grade": evaluation["grade"],
        "issues": evaluation["issues"],
        "suggestions": evaluation["suggestions"],
        "handedness": handedness,
        "timeline_cards": [
            {
                "label": "起手阶段",
                "frame_number": start_frame_number,
                "combined_image": frame_relative_path(start_frame_path),
                "skeleton_image": frame_relative_path(start_skeleton_path),
            },
            {
                "label": "射箭阶段",
                "frame_number": key_frame_number,
                "combined_image": frame_relative_path(key_frame_path),
                "skeleton_image": frame_relative_path(skeleton_path),
            },
        ],
        "loop_previews": {
            "combined": combined_gif_path,
            "skeleton": skeleton_gif_path,
        },
        "key_frame_image": frame_relative_path(key_frame_path),
        "skeleton_image": frame_relative_path(skeleton_path),
        "original_video": Path(video_path).relative_to(WEB_DATA_DIR).as_posix(),
    }

    if athlete_name:
        profile_path, previous_line_count = save_profile_record(athlete_name, front_arm_angle, behind_arm_angle, score)
        result["athlete_name"] = athlete_name
        result["profile_file"] = profile_path.relative_to(WEB_DATA_DIR).as_posix()
        result["profile_previous_line_count"] = previous_line_count
        result["charts"] = generate_profile_charts(profile_path)

    result["current_upload_path"] = str(Path(video_path).resolve())
    result["current_run_dir"] = str(run_dir.resolve())

    return result
