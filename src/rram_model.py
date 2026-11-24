# rram_model.py 주요 기능
# 1. update_conductance(): 입력 비트에 따른 컨덕턴스의 시간 감쇠 및 비선형 증가 계산
# 2. process_sequence(): 4비트 펄스를 순서대로 입력하고 최종 컨덕턴스와 변화 과정 반환
# 3. generate_reservoir_states(): 256개 펄스 시퀀스를 256차원 Reservoir State로 변환
# 4. create_device_windows(): 소자별 G_MIN/G_MAX를 한 번 생성해 고정
# 5. 실행 및 검증: 대표 4비트 패턴 4개의 컨덕턴스 변화와 최종 출력 크기 확인
# 논문 참고: RRAM의 컨덕턴스 업데이트는 현재 상태에 따라 비선형적으로 달라질 수 있음
# 아래 식은 논문의 식을 그대로 사용한 것이 아니라, 비선형 증가와 단기 기억을 가정한 간소화 모델임
# 실제 소자의 측정값이나 RESET 동작, 물리적 시간 단위는 반영하지 않음

import matplotlib.pyplot as plt # 대표 4비트 펄스의 컨덕턴스 변화 시각화
import numpy as np # 지수함수 및 Reservoir State 배열 연산


G_MIN = 0.0 # 정규화된 최소 컨덕턴스(실제 단위 S가 아님)
G_MAX = 1.0 # 정규화된 최대 컨덕턴스
TAU = 3.0 # 컨덕턴스 감쇠 시정수(임의 시간 단위)
DELTA_T = 1.0 # 비트 입력 간 시간 간격(임의 시간 단위)
ALPHA = 0.35 # 입력 비트가 1일 때 컨덕턴스 증가 강도
NONLINEARITY = 1.5 # 컨덕턴스 증가의 비선형성 지수
BITS_PER_PULSE = 4 # 하나의 RRAM에 순차적으로 입력할 비트 수


def create_device_windows(num_devices=256, variation=0.0, seed=42): # 소자별 G_MIN/G_MAX 생성
    if num_devices <= 0 or variation < 0:
        raise ValueError("소자 수는 양수, variation은 0 이상이어야 합니다.")

    rng = np.random.default_rng(seed) # 같은 seed는 항상 같은 소자 특성을 생성
    window = G_MAX - G_MIN
    if window <= 0:
        raise ValueError("G_MAX가 G_MIN보다 커야 합니다.")

    # variation은 기준 Window 대비 편차의 표준편차 비율(0.05 = 5%)
    # G_MIN=0에서는 기준값에 대한 상대 오차를 정의하기 어려우므로 Window 기준으로 설정
    g_min = G_MIN + rng.normal(0.0, variation * window, num_devices)
    g_max = G_MAX + rng.normal(0.0, variation * window, num_devices)

    # 음수 컨덕턴스와 G_MIN >= G_MAX인 비물리적 범위 방지
    g_min = np.maximum(g_min, 0.0)
    g_max = np.maximum(g_max, g_min + 1e-6)
    return np.column_stack((g_min, g_max)).astype(np.float32)


def update_conductance(conductance, bit, tau=TAU, delta_t=DELTA_T,
                       alpha=ALPHA, nonlinearity=NONLINEARITY,
                       g_min=G_MIN, g_max=G_MAX): # 비트 하나를 입력했을 때의 컨덕턴스 계산
    if bit not in (0, 1):
        raise ValueError("입력 비트는 0 또는 1이어야 합니다.")
    if tau <= 0 or delta_t < 0 or not 0 <= alpha <= 1 or nonlinearity <= 0:
        raise ValueError("tau, delta_t, alpha, nonlinearity 설정값을 확인하세요.")
    if g_min < 0 or g_max <= g_min:
        raise ValueError("g_min과 g_max의 범위를 확인하세요.")
    if not g_min - 1e-7 <= conductance <= g_max + 1e-7:
        raise ValueError("컨덕턴스가 해당 소자의 범위를 벗어났습니다.")

    # 소자별 Window를 적용해도 동일한 정규화 업데이트 동역학을 사용
    # (u=0: G_MIN, u=1: G_MAX). Variation 효과와 비선형성 변화가 섞이지 않도록 함
    window = g_max - g_min
    u = np.clip((conductance - g_min) / window, 0.0, 1.0)
    decayed = u * np.exp(-delta_t / tau) # 단기 기억: 0 방향으로 감쇠

    if bit == 1:
        decayed = decayed + alpha * (1.0 - decayed) ** nonlinearity

    return float(g_min + window * np.clip(decayed, 0.0, 1.0))


def process_sequence(sequence, g_min=G_MIN, g_max=G_MAX): # 4비트 입력 및 소자별 컨덕턴스 기록
    sequence = np.asarray(sequence)
    if sequence.shape != (BITS_PER_PULSE,) or not np.all(np.isin(sequence, [0, 1])):
        raise ValueError(f"입력은 0과 1로 구성된 {BITS_PER_PULSE}비트 시퀀스여야 합니다.")

    conductance = float(g_min) # 4비트 입력마다 소프트웨어상 초기화
    history = [conductance]
    for bit in sequence:
        conductance = update_conductance(conductance, int(bit), g_min=g_min, g_max=g_max)
        history.append(conductance)
    return conductance, np.array(history, dtype=np.float32)


def generate_reservoir_states(pulses, device_windows=None): # 256개 소자별 최종 상태 추출
    pulses = np.asarray(pulses)
    if pulses.ndim != 2 or pulses.shape[1] != BITS_PER_PULSE:
        raise ValueError(f"펄스 배열은 (시퀀스 수, {BITS_PER_PULSE}) 형태여야 합니다.")
    if not np.all(np.isin(pulses, [0, 1])):
        raise ValueError("펄스 배열에는 0과 1만 포함되어야 합니다.")

    if device_windows is not None:
        device_windows = np.asarray(device_windows, dtype=np.float64)
        if device_windows.shape != (len(pulses), 2):
            raise ValueError(f"device_windows는 ({len(pulses)}, 2) 형태여야 합니다.")
        if np.any(device_windows[:, 0] < 0) or np.any(device_windows[:, 1] <= device_windows[:, 0]):
            raise ValueError("각 소자는 0 <= G_MIN < G_MAX를 만족해야 합니다.")

    states = np.zeros(len(pulses), dtype=np.float32)
    for i, sequence in enumerate(pulses):
        if device_windows is None:
            states[i], _ = process_sequence(sequence)
        else:
            g_min, g_max = device_windows[i]
            states[i], _ = process_sequence(sequence, g_min=float(g_min), g_max=float(g_max))
    return states


# 테스트 코드: 대표 패턴 4개의 응답을 시각화하고 256차원 출력 검증
if __name__ == "__main__":
    # 0000: 감쇠만 발생 / 0110과 1001: 입력 순서 비교 / 1111: 연속 입력
    selected_patterns = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]]

    plt.figure(figsize=(8, 5))
    for pattern in selected_patterns:
        final_g, history = process_sequence(pattern)
        label = "".join(str(bit) for bit in pattern)
        print(f"입력 패턴: {label}, 최종 컨덕턴스: {final_g:.4f}")
        plt.plot(range(BITS_PER_PULSE + 1), history, marker="o", linewidth=2, label=label)

    # 출력 크기 검증은 기존처럼 16개 패턴을 모두 사용
    all_patterns = [[int(bit) for bit in f"{number:04b}"] for number in range(16)]
    test_pulses = np.tile(np.array(all_patterns), (16, 1))
    states = generate_reservoir_states(test_pulses)
    print("입력 펄스 배열 크기:", test_pulses.shape)
    print("Reservoir State 크기:", states.shape)

    plt.title("RRAM Conductance Response (4 Representative Patterns)")
    plt.xlabel("Input step")
    plt.ylabel("Normalized conductance")
    plt.xticks(range(BITS_PER_PULSE + 1))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="4-bit pattern")
    plt.tight_layout()
    plt.show()
