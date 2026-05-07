# RMSE, 복귀시간, 최소거리, 흔들림 계산
import argparse
import csv
import math
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def to_optional_float(value):
    try:
        if value is None:
            return None

        value_str = str(value).strip()

        if value_str == "":
            return None

        return float(value_str)

    except Exception:
        return None


def to_bool(value):
    return str(value).strip() in ("1", "True", "true", "YES", "yes")


def angle_diff_deg(a, b):
    """
    yaw처럼 -180도 ~ 180도 또는 0도 ~ 360도 경계가 있는 각도 차이를 안전하게 계산한다.
    """
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def analyze_csv(path: Path, collision_threshold: float):
    rows = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found: {path}")

    # 이륙 안정화 이후 데이터만 사용한다.
    # 해당 컬럼이 없으면 전체 데이터를 사용한다.
    stable_rows = [
        r for r in rows
        if str(r.get("takeoff_stable", "0")) in ("1", "True", "true")
    ]

    data = stable_rows if stable_rows else rows

    # =========================
    # Distance
    # =========================
    # 최소거리는 정면뿐 아니라 회피 중 측면 접근도 반영하기 위해
    # min(front_min, side_min)을 사용한다.
    obstacle_values = []

    for r in data:
        front = to_float(r.get("front_min"), 10.0)
        side = to_float(r.get("side_min"), 10.0)
        obstacle_values.append(min(front, side))

    min_distance = min(obstacle_values)

    # 거리 기반 충돌 위험 여부
    # 실제 물리 충돌 여부와는 구분한다.
    collision_risk = min_distance <= collision_threshold

    # Gazebo contact sensor 기반 실제 물리 충돌 여부
    actual_collision = any(
        to_bool(r.get("actual_collision", "0"))
        for r in data
    )

    # =========================
    # Path error
    # =========================
    abs_errors = [
        abs(to_float(r.get("path_error"), 0.0))
        for r in data
    ]

    avg_path_error = sum(abs_errors) / len(abs_errors)
    max_path_error = max(abs_errors)
    rmse_path_error = math.sqrt(
        sum(e * e for e in abs_errors) / len(abs_errors)
    )

    # =========================
    # Velocity command
    # =========================
    vx = [
        to_float(r.get("vx"), 0.0)
        for r in data
    ]

    vy = [
        to_float(r.get("vy"), 0.0)
        for r in data
    ]

    vz = [
        to_float(r.get("vz"), 0.0)
        for r in data
    ]

    t = [
        to_float(r.get("wall_time"), 0.0)
        for r in data
    ]

    modes = [
        r.get("mode", "")
        for r in data
    ]

    # =========================
    # Attitude / Angular velocity
    # =========================
    roll = [
        to_optional_float(r.get("roll_deg"))
        for r in data
    ]

    pitch = [
        to_optional_float(r.get("pitch_deg"))
        for r in data
    ]

    yaw = [
        to_optional_float(r.get("yaw_deg"))
        for r in data
    ]

    roll_rate = [
        to_optional_float(r.get("roll_rate"))
        for r in data
    ]

    pitch_rate = [
        to_optional_float(r.get("pitch_rate"))
        for r in data
    ]

    yaw_rate = [
        to_optional_float(r.get("yaw_rate"))
        for r in data
    ]

    # =========================
    # Control effort / Velocity command oscillation
    # =========================
    control_effort = 0.0
    oscillation = 0.0

    for i in range(1, len(data)):
        dt = max(0.0, t[i] - t[i - 1])

        control_effort += (
            vx[i] ** 2
            + vy[i] ** 2
            + vz[i] ** 2
        ) * dt

        oscillation += (
            abs(vx[i] - vx[i - 1])
            + abs(vy[i] - vy[i - 1])
            + abs(vz[i] - vz[i - 1])
        )

    # =========================
    # Attitude oscillation / Angular velocity oscillation
    # =========================
    attitude_oscillation = 0.0
    angular_velocity_oscillation = 0.0
    angular_velocity_energy = 0.0

    attitude_valid_count = 0
    angular_velocity_valid_count = 0

    for i in range(1, len(data)):
        dt = max(0.0, t[i] - t[i - 1])

        # 자세 변화량 기반 흔들림
        if (
            roll[i] is not None and roll[i - 1] is not None
            and pitch[i] is not None and pitch[i - 1] is not None
            and yaw[i] is not None and yaw[i - 1] is not None
        ):
            attitude_oscillation += (
                angle_diff_deg(roll[i], roll[i - 1])
                + angle_diff_deg(pitch[i], pitch[i - 1])
                + angle_diff_deg(yaw[i], yaw[i - 1])
            )

            attitude_valid_count += 1

        # 각속도 변화량 기반 흔들림 + 각속도 에너지
        if (
            roll_rate[i] is not None and roll_rate[i - 1] is not None
            and pitch_rate[i] is not None and pitch_rate[i - 1] is not None
            and yaw_rate[i] is not None and yaw_rate[i - 1] is not None
        ):
            angular_velocity_oscillation += (
                abs(roll_rate[i] - roll_rate[i - 1])
                + abs(pitch_rate[i] - pitch_rate[i - 1])
                + abs(yaw_rate[i] - yaw_rate[i - 1])
            )

            angular_velocity_energy += (
                roll_rate[i] ** 2
                + pitch_rate[i] ** 2
                + yaw_rate[i] ** 2
            ) * dt

            angular_velocity_valid_count += 1

    # =========================
    # Recovery time
    # =========================
    # 복귀 시간:
    # 첫 AVOID 진입 시점부터 최종 FOLLOW_PATH 복귀 시점까지.
    #
    # 예:
    # FOLLOW_PATH -> AVOID -> RETURN_PATH -> AVOID -> RETURN_PATH -> FOLLOW_PATH
    # 이 경우 recovery_time은 첫 AVOID 시작부터 마지막 FOLLOW_PATH 복귀까지로 계산한다.
    avoid_start_time = None
    recovery_complete_time = None
    avoid_start_idx = None

    for i, mode in enumerate(modes):
        if mode == "AVOID":
            avoid_start_idx = i
            avoid_start_time = t[i]
            break

    if avoid_start_idx is not None:
        last_recovery_related_idx = avoid_start_idx

        for i in range(avoid_start_idx, len(modes)):
            if modes[i] in ("AVOID", "RETURN_PATH"):
                last_recovery_related_idx = i

        for i in range(last_recovery_related_idx + 1, len(modes)):
            if modes[i] == "FOLLOW_PATH":
                recovery_complete_time = t[i]
                break

    if avoid_start_time is not None and recovery_complete_time is not None:
        recovery_time = recovery_complete_time - avoid_start_time
    else:
        recovery_time = None

    # =========================
    # Re-avoid count
    # =========================
    # RETURN_PATH 도중 다시 AVOID로 들어간 횟수.
    reavoid_count = 0
    avoid_episode_count = 0

    prev_mode = None
    seen_first_avoid = False

    for mode in modes:
        if mode == "AVOID" and prev_mode != "AVOID":
            avoid_episode_count += 1

            if prev_mode == "RETURN_PATH" and seen_first_avoid:
                reavoid_count += 1

            seen_first_avoid = True

        prev_mode = mode

    # AVOID 상태에 실제로 머문 총 시간
    total_avoid_time = 0.0

    for i in range(1, len(data)):
        dt = max(0.0, t[i] - t[i - 1])

        if modes[i - 1] == "AVOID":
            total_avoid_time += dt

    return {
        "file": str(path),
        "rows": len(rows),
        "used_rows": len(data),

        "min_distance": min_distance,
        "collision_risk": collision_risk,
        "actual_collision": actual_collision,

        "avg_path_error": avg_path_error,
        "max_path_error": max_path_error,
        "rmse_path_error": rmse_path_error,

        "recovery_time": recovery_time,
        "reavoid_count": reavoid_count,
        "avoid_episode_count": avoid_episode_count,
        "total_avoid_time": total_avoid_time,

        # 속도 명령 기반 지표
        "oscillation": oscillation,
        "control_effort": control_effort,

        # 자세/각속도 기반 지표
        "attitude_oscillation": attitude_oscillation,
        "angular_velocity_oscillation": angular_velocity_oscillation,
        "angular_velocity_energy": angular_velocity_energy,

        # 디버깅용 유효 샘플 수
        "attitude_valid_count": attitude_valid_count,
        "angular_velocity_valid_count": angular_velocity_valid_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Experiment CSV files",
    )

    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.7,
        help="Distance-based collision risk threshold in meters",
    )

    args = parser.parse_args()

    print(
        "file,rows,used_rows,"
        "min_distance,collision_risk,actual_collision,"
        "avg_path_error,max_path_error,rmse_path_error,"
        "recovery_time,reavoid_count,avoid_episode_count,total_avoid_time,"
        "oscillation,control_effort,"
        "attitude_oscillation,angular_velocity_oscillation,angular_velocity_energy,"
        "attitude_valid_count,angular_velocity_valid_count"
    )

    for file_name in args.csv_files:
        result = analyze_csv(
            Path(file_name),
            args.collision_threshold,
        )

        recovery_time_text = ""

        if result["recovery_time"] is not None:
            recovery_time_text = f"{result['recovery_time']:.4f}"

        print(
            f"{result['file']},"
            f"{result['rows']},"
            f"{result['used_rows']},"

            f"{result['min_distance']:.4f},"
            f"{int(result['collision_risk'])},"
            f"{int(result['actual_collision'])},"

            f"{result['avg_path_error']:.4f},"
            f"{result['max_path_error']:.4f},"
            f"{result['rmse_path_error']:.4f},"

            f"{recovery_time_text},"
            f"{result['reavoid_count']},"
            f"{result['avoid_episode_count']},"
            f"{result['total_avoid_time']:.4f},"

            f"{result['oscillation']:.4f},"
            f"{result['control_effort']:.4f},"

            f"{result['attitude_oscillation']:.4f},"
            f"{result['angular_velocity_oscillation']:.4f},"
            f"{result['angular_velocity_energy']:.4f},"

            f"{result['attitude_valid_count']},"
            f"{result['angular_velocity_valid_count']}"
        )


if __name__ == "__main__":
    main()