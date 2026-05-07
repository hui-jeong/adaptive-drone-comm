import csv
import math
import os


# =========================
# 실험 기본 설정
# =========================

OBSTACLE_X = 10.0
OBSTACLE_Y = 0.0

START_X = 0.0
END_X = 20.0

ALTITUDE_Z = -2.5

DT = 0.1  # 0.1초 간격으로 시뮬레이션
DRONE_SPEED = 1.5  # m/s


# =========================
# 상태 전환 기준 파라미터
# =========================

T_CAUTION = 3.0      # caution 진입을 위한 시간 여유
T_EMERGENCY = 1.5    # emergency 진입을 위한 시간 여유
D_MARGIN = 1.0       # 센서 오차 및 제어 반응 여유 거리


# =========================
# 로그 저장 경로
# =========================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "simulation_test_log.csv")


def calculate_distance(x, y, obstacle_x, obstacle_y):
    """
    드론과 장애물 사이의 거리 계산
    """
    dx = obstacle_x - x
    dy = obstacle_y - y
    return math.sqrt(dx * dx + dy * dy)


def calculate_closing_speed(x, y, vx, vy, obstacle_x, obstacle_y):
    """
    장애물 방향으로 실제 접근하는 속도 계산

    v_close > 0  : 장애물에 가까워지는 중
    v_close = 0  : 장애물과의 거리가 거의 변하지 않음
    v_close < 0  : 장애물에서 멀어지는 중
    """
    dx = obstacle_x - x
    dy = obstacle_y - y

    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return 0.0

    unit_x = dx / distance
    unit_y = dy / distance

    v_close = vx * unit_x + vy * unit_y

    return max(v_close, 0.0)


def calculate_thresholds(v_close):
    """
    속도 기반 거리 임계값 계산
    """
    d_caution = v_close * T_CAUTION + D_MARGIN
    d_emergency = v_close * T_EMERGENCY + D_MARGIN

    return d_caution, d_emergency


def determine_state(distance, d_caution, d_emergency):
    """
    현재 거리와 임계값을 기준으로 상태 판단
    """
    if distance <= d_emergency:
        return "EMERGENCY"
    elif distance <= d_caution:
        return "CAUTION"
    else:
        return "NORMAL"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    x = START_X
    y = 0.0
    z = ALTITUDE_Z

    vx = DRONE_SPEED
    vy = 0.0
    vz = 0.0

    time = 0.0

    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "speed",
            "obstacle_distance",
            "v_close",
            "d_caution",
            "d_emergency",
            "state",
            "path_error"
        ])

        while x <= END_X:
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)

            distance = calculate_distance(
                x, y,
                OBSTACLE_X, OBSTACLE_Y
            )

            v_close = calculate_closing_speed(
                x, y,
                vx, vy,
                OBSTACLE_X, OBSTACLE_Y
            )

            d_caution, d_emergency = calculate_thresholds(v_close)

            state = determine_state(
                distance,
                d_caution,
                d_emergency
            )

            path_error = abs(y)

            writer.writerow([
                round(time, 2),
                round(x, 3),
                round(y, 3),
                round(z, 3),
                round(vx, 3),
                round(vy, 3),
                round(vz, 3),
                round(speed, 3),
                round(distance, 3),
                round(v_close, 3),
                round(d_caution, 3),
                round(d_emergency, 3),
                state,
                round(path_error, 3)
            ])

            print(
                f"time={time:.1f}s, "
                f"x={x:.2f}, "
                f"distance={distance:.2f}, "
                f"v_close={v_close:.2f}, "
                f"d_caution={d_caution:.2f}, "
                f"d_emergency={d_emergency:.2f}, "
                f"state={state}"
            )

            x += vx * DT
            time += DT

    print()
    print(f"로그 저장 완료: {LOG_FILE}")


if __name__ == "__main__":
    main()