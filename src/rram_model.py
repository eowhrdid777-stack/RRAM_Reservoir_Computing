# rram_model.py 주요 기능

# 1. update_conductance(): 입력 비트에 따른 컨덕턴스의 시간 감쇠 및 비선형 증가 계산
# 2. process_sequence(): 4비트 펄스를 순서대로 입력하고 최종 컨덕턴스와 변화 과정 반환
# 3. generate_reservoir_states(): 256개 펄스 시퀀스를 256차원 Reservoir State로 변환
# 4. 실행 및 검증: 서로 다른 4비트 패턴의 컨덕턴스 변화와 최종 출력 크기 확인
# 논문 참고: RRAM의 컨덕턴스 업데이트는 현재 상태에 따라 비선형적으로 달라질 수 있음
# 아래 식은 논문의 식을 그대로 사용한 것이 아니라, 비선형 증가와 단기 기억을 가정한 간소화 모델임
# 실제 소자의 측정값이나 RESET 동작, 물리적 시간 단위는 반영하지 않음

import matplotlib.pyplot as plt # 4비트 펄스별 컨덕턴스 변화 시각화
import numpy as np # 지수함수 및 Reservoir State 배열 연산


G_MIN = 0.0 # 정규화된 최소 컨덕턴스(실제 단위 S가 아님)
G_MAX = 1.0 # 정규화된 최대 컨덕턴스
TAU = 3.0 # 컨덕턴스 감쇠 시정수(임의 시간 단위)
DELTA_T = 1.0 # 비트 입력 간 시간 간격(임의 시간 단위)
ALPHA = 0.35 # 입력 비트가 1일 때 컨덕턴스 증가 강도
NONLINEARITY = 1.5 # 컨덕턴스 증가의 비선형성 지수
BITS_PER_PULSE = 4 # 하나의 RRAM에 순차적으로 입력할 비트 수


def update_conductance(conductance, bit, tau=TAU, delta_t=DELTA_T,
                       alpha=ALPHA, nonlinearity=NONLINEARITY): # 비트 하나를 입력했을 때의 컨덕턴스 계산
    if bit not in (0, 1):
        raise ValueError("입력 비트는 0 또는 1이어야 합니다.")
    if tau <= 0 or delta_t < 0 or not 0 <= alpha <= 1 or nonlinearity <= 0:
        raise ValueError("tau, delta_t, alpha, nonlinearity 설정값을 확인하세요.")
    if not G_MIN <= conductance <= G_MAX:
        raise ValueError("컨덕턴스가 정규화 범위를 벗어났습니다.")

    # 단기 기억: 펄스 간격 동안 컨덕턴스가 G_MIN 방향으로 지수 감쇠
    decayed = G_MIN + (conductance - G_MIN) * np.exp(-delta_t / tau)

    # bit=1: 현재 상태에 따라 증가량이 달라지는 비선형 Potentiation
    # bit=0: 추가 증가 없이 감쇠한 상태를 그대로 유지
    if bit == 1:
        decayed = decayed + alpha * (G_MAX - decayed) ** nonlinearity

    return float(np.clip(decayed, G_MIN, G_MAX))


def process_sequence(sequence): # 4비트 펄스를 순서대로 적용하고 컨덕턴스 변화 기록
    sequence = np.asarray(sequence)
    if sequence.shape != (BITS_PER_PULSE,) or not np.all(np.isin(sequence, [0, 1])):
        raise ValueError(f"입력은 0과 1로 구성된 {BITS_PER_PULSE}비트 시퀀스여야 합니다.")

    conductance = G_MIN # 각 4비트 시퀀스의 시작 상태를 소프트웨어상 초기화
    history = [conductance]

    for bit in sequence:
        conductance = update_conductance(conductance, int(bit))
        history.append(conductance)

    return conductance, np.array(history, dtype=np.float32)


def generate_reservoir_states(pulses): # (256, 4) 펄스를 (256,) Reservoir State로 변환
    pulses = np.asarray(pulses)
    if pulses.ndim != 2 or pulses.shape[1] != BITS_PER_PULSE:
        raise ValueError(f"펄스 배열은 (시퀀스 수, {BITS_PER_PULSE}) 형태여야 합니다.")
    if not np.all(np.isin(pulses, [0, 1])):
        raise ValueError("펄스 배열에는 0과 1만 포함되어야 합니다.")

    states = np.zeros(len(pulses), dtype=np.float32)
    for i, sequence in enumerate(pulses):
        states[i], _ = process_sequence(sequence)

    return states


# 테스트 코드: 4비트 패턴별 컨덕턴스 변화를 비교하고 256차원 출력 확인
if __name__ == "__main__":
    patterns = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]

    for pattern in patterns:
        final_g, history = process_sequence(pattern)
        label = "".join(str(bit) for bit in pattern)
        print(f"입력 패턴: {label}, 최종 컨덕턴스: {final_g:.4f}")
        plt.plot(range(BITS_PER_PULSE + 1), history, marker="o", label=label)

    # 임의의 256개 입력 시퀀스로 출력 형태 검증 (실제 FSDD 음성은 사용하지 않음)
    test_pulses = np.tile([1, 0, 0, 1], (256, 1))
    states = generate_reservoir_states(test_pulses)
    print("입력 펄스 배열 크기:", test_pulses.shape)
    print("Reservoir State 크기:", states.shape)

    plt.title("RRAM conductance response to 4-bit sequences")
    plt.xlabel("Input step")
    plt.ylabel("Normalized conductance")
    plt.xticks(range(BITS_PER_PULSE + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
