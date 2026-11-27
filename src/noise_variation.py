# S2_noise_variation.py 주요 기능

# 1. Clean WAV 음성으로 최종 RRAM Reservoir + MLP-128 Readout 학습
# 2. 학습된 Readout 가중치를 고정하고 Validation WAV에 AWGN 잡음 추가
# 3. SNR 수준(Clean, 30, 20, 10, 5, 0 dB)별 Validation Accuracy 비교
# 4. 잡음 시드(42, 43, 44)별 반복 평가 후 정확도 평균/표준편차 계산
# 5. Mel 전처리 및 이진화 후 원본 대비 이진 펄스 변화율 측정
# 6. 4비트 펄스 패턴 16개의 컨덕턴스 응답을 미리 계산하여 재사용
# 7. 정확도와 이진 펄스 변화율을 CSV 및 그래프로 저장 (Test 데이터는 사용하지 않음)

# 실행: 프로젝트 최상위 폴더에서 python src/S2_noise_variation.py
from pathlib import Path  # 현재 파일을 기준으로 데이터 및 결과 저장 경로 설정
import csv  # 실험 결과를 CSV 파일로 저장

import matplotlib  # 결과 그래프 저장에 사용하는 라이브러리
matplotlib.use("Agg")  # 그래프 창 없이 파일로 저장하도록 설정
import matplotlib.pyplot as plt  # 정확도 및 펄스 변화율 그래프 작성
import numpy as np  # 음성 배열, 난수, 평균 및 표준편차 계산
import torch  # MLP 학습과 평가를 위한 텐서 연산

from data_loader import load_dataset, load_wav, split_dataset  # F1: WAV 읽기 및 Train/Validation 분할
from preprocessing import signal_to_pulses, THRESHOLD  # F2: Mel 전처리 및 이진 펄스 변환
from rram_model import process_sequence  # F3: 4비트 펄스에 대한 RRAM 컨덕턴스 계산
from readout import Readout, train_readout, INPUT_SIZE  # F5: 최종 MLP-128 모델과 학습 함수
from evaluation import evaluate_model  # Validation Loss 및 Accuracy 계산

# 실험 조건: Clean으로 Readout을 한 번 학습한 뒤 가중치를 고정하여 잡음 평가
TRAIN_SEED = 42  # Clean Readout 학습에 사용하는 난수 시드
NOISE_SEEDS = (42, 43, 44)  # 동일 SNR에서 서로 다른 잡음 3회 생성
SNR_LEVELS = (30, 20, 10, 5, 0)  # 낮을수록 잡음 전력이 큰 SNR 조건 (dB)
EPOCHS = 300  # Clean 데이터에 대한 최대 Readout 학습 Epoch

ROOT = Path(__file__).resolve().parent.parent  # src의 상위 폴더 = 프로젝트 루트
DATA_DIR = ROOT / "data" / "recordings"  # FSDD WAV 파일 경로
OUT_DIR = ROOT / "results" / "S2_noise_variation"  # S2 실험 결과 저장 폴더

# RRAM에 4비트씩 입력되는 경우 가능한 패턴 16개를 미리 계산
PATTERNS = np.array([[int(bit) for bit in f"{i:04b}"] for i in range(16)], dtype=np.int8)  # 0000~1111
PATTERN_WEIGHTS = np.array([8, 4, 2, 1], dtype=np.int8)  # 4비트 패턴을 0~15 인덱스로 변환
LOOKUP = np.array([process_sequence(p)[0] for p in PATTERNS], dtype=np.float32)  # 각 패턴의 최종 컨덕턴스


def get_features(signal, sample_rate):  # WAV를 RRAM State와 이진 펄스로 변환
    """WAV 신호 -> Mel 이진 펄스 -> 576차원 RRAM Reservoir State."""
    _, pulses = signal_to_pulses(signal, sample_rate)  # Mel 추출과 Threshold 이진화
    pulses = np.asarray(pulses)  # (576, 4) 형태의 펄스 배열
    if pulses.ndim != 2 or pulses.shape[1] != 4:  # 4비트 입력 여부 확인
        raise ValueError(f"4비트 펄스를 예상했지만 shape={pulses.shape}")
    states = LOOKUP[pulses @ PATTERN_WEIGHTS]  # 각 4비트를 인덱스로 바꿔 RRAM 값 조회
    if states.shape != (INPUT_SIZE,):  # Readout 입력 차원(576) 확인
        raise ValueError(f"Reservoir State 차원이 다릅니다: {states.shape}")
    return states, pulses  # 모델 입력 및 잡음 전후 변화율 비교에 사용


def add_awgn(signal, snr_db, rng):  # 목표 SNR에 맞는 백색 가우시안 잡음 생성
    """입력 WAV 신호에 신호 전력 기준 AWGN을 더함 (clipping 없음)."""
    signal = np.asarray(signal, dtype=np.float32)  # 원본 WAV 신호를 float32로 변환
    signal_power = np.mean(signal.astype(np.float64) ** 2)  # 신호 평균 전력
    noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))  # SNR(dB)에 대응하는 잡음 표준편차
    noise = rng.standard_normal(len(signal)).astype(np.float32) * np.float32(noise_std)  # 난수 잡음 생성
    return signal + noise  # WAV 전처리 이전에 잡음 추가


def save_csv(path, rows):  # 딕셔너리 형태의 결과를 CSV로 저장
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # 한글 CSV 호환 인코딩
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))  # 첫 결과의 열 이름 사용
        writer.writeheader()  # CSV 헤더 저장
        writer.writerows(rows)  # 시드별 또는 SNR별 결과 저장


def main():  # Clean 학습 → 잡음 평가 → 통계 계산 → 결과 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # 결과 폴더가 없으면 생성

    dataset = load_dataset(DATA_DIR)  # 전체 WAV 파일 목록 로드
    train, val, _ = split_dataset(dataset)  # Train/Validation만 사용, Test 제외
    print(f"Train: {len(train)}, Validation: {len(val)}")

    # 1. 잡음 없는 Train 데이터로 Reservoir State와 정답 라벨 생성
    train_states, train_labels = [], []
    for item in train:  # Train 음성을 하나씩 전처리
        sr, signal = load_wav(item["path"])  # WAV 샘플링 주파수와 신호 읽기
        state, _ = get_features(signal, sr)  # 576차원 RRAM State 계산
        train_states.append(state)  # 음성별 RRAM 특징 저장
        train_labels.append(item["label"])  # 숫자 정답 저장

    # 2. Clean Validation 입력과 펄스를 미리 저장해 잡음 입력과 비교
    val_states, val_labels, val_signals, val_pulses = [], [], [], []
    for item in val:  # Validation 음성을 하나씩 전처리
        sr, signal = load_wav(item["path"])  # WAV 샘플링 주파수와 신호 읽기
        state, pulses = get_features(signal, sr)  # State와 원본 이진 펄스 모두 계산
        val_states.append(state)  # 잡음 없는 Validation 특징
        val_labels.append(item["label"])  # Validation 정답
        val_signals.append((sr, signal))  # 잡음을 넣기 전 WAV 신호 보관
        val_pulses.append(pulses)  # 잡음 적용 전 펄스 비교 기준

    # 3. NumPy 특징/라벨을 Readout 학습용 PyTorch 텐서로 변환
    train_x = torch.tensor(np.stack(train_states), dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)  # CrossEntropyLoss용 정수 라벨
    val_x = torch.tensor(np.stack(val_states), dtype=torch.float32)  # Clean Validation 입력
    val_y = torch.tensor(val_labels, dtype=torch.long)  # Validation 정답
    val_pulses = np.stack(val_pulses)  # 전체 Validation의 이진 펄스 배열

    # Clean 데이터로 단 한 번 학습, Validation Loss가 최저인 가중치 복원
    torch.manual_seed(TRAIN_SEED)  # Readout 초기화/학습 난수 재현
    np.random.seed(TRAIN_SEED)  # NumPy 난수 시드 고정
    model, history = train_readout(Readout(), train_x, train_y, val_x, val_y, epochs=EPOCHS)  # 최저 Validation Loss 가중치 복원
    model.eval()  # Dropout을 비활성화하고 가중치 고정
    torch.save(model.state_dict(), OUT_DIR / "clean_readout_seed42.pth")  # 학습된 Readout 저장
    clean_loss, clean_acc, _, _ = evaluate_model(model, val_x, val_y)  # 잡음 없는 정확도 측정
    print(f"Clean: Accuracy={clean_acc*100:.2f}%, Loss={clean_loss:.4f}")

    # 4. 동일한 Clean Readout으로 SNR별/잡음 시드별 Validation 평가
    raw = []
    for snr in SNR_LEVELS:  # 30, 20, 10, 5, 0 dB 순서로 실험
        for seed in NOISE_SEEDS:  # SNR마다 세 가지 잡음 패턴 생성
            rng = np.random.default_rng(seed)  # 잡음 생성 전용 난수기
            states, changed = [], []  # 새 State 및 원본 대비 펄스 차이
            for (sr, signal), original_pulses in zip(val_signals, val_pulses):  # 같은 음성/펄스끼리 비교
                noisy = add_awgn(signal, snr, rng)  # 원본 WAV에 지정한 SNR의 AWGN 추가
                state, noisy_pulses = get_features(noisy, sr)  # 잡음 WAV를 동일한 과정으로 전처리
                states.append(state)  # 잡음 적용 후 모델 입력 저장
                changed.append(np.mean(original_pulses != noisy_pulses))  # 값이 달라진 이진 펄스의 비율
            noise_x = torch.tensor(np.stack(states), dtype=torch.float32)  # 잡음 조건의 Validation 입력
            loss, acc, _, _ = evaluate_model(model, noise_x, val_y)  # 재학습 없이 고정 Readout 평가
            row = dict(snr_db=snr, seed=seed, accuracy_pct=acc*100,  # 해당 시드의 Accuracy
                       loss=loss, pulse_change_pct=100*np.mean(changed))
            raw.append(row)  # 개별 실험 결과를 누적
            print(f"SNR {snr:>2} dB | Seed {seed} | Accuracy {acc*100:.2f}% | "
                  f"Loss {loss:.4f} | Pulse change {row['pulse_change_pct']:.2f}%")

    # 5. 같은 SNR의 세 실험을 묶어 평균과 표본표준편차 계산
    summary = []
    for snr in SNR_LEVELS:  # 30, 20, 10, 5, 0 dB 순서로 실험  # 동일 SNR 조건만 모아서 통계 계산
        subset = [row for row in raw if row["snr_db"] == snr]
        accuracy = np.array([row["accuracy_pct"] for row in subset])  # 시드별 정확도 배열
        pulse_change = np.array([row["pulse_change_pct"] for row in subset])  # 펄스 변화율 배열
        summary.append(dict(snr_db=snr, mean_accuracy_pct=accuracy.mean(),  # 평균 정확도
                            std_accuracy_pp=accuracy.std(ddof=1),
                            mean_pulse_change_pct=pulse_change.mean(),
                            std_pulse_change_pp=pulse_change.std(ddof=1)))

    # 6. 개별 결과 및 평균/표준편차를 CSV 파일로 저장
    save_csv(OUT_DIR / "noise_raw.csv", raw)
    save_csv(OUT_DIR / "noise_summary.csv", summary)
    levels = sorted(summary, key=lambda row: row["snr_db"])  # 그래프의 SNR을 오름차순 정렬

    # Fig: Validation Accuracy ± standard deviation, Clean baseline
    fig, ax = plt.subplots(figsize=(8, 5))  # 정확도 그래프 영역 생성
    ax.errorbar([r["snr_db"] for r in levels],  # SNR별 평균 정확도와 오차 막대
                [r["mean_accuracy_pct"] for r in levels],
                yerr=[r["std_accuracy_pp"] for r in levels],
                marker="o", capsize=4, label="AWGN (3 noise seeds)")
    ax.axhline(clean_acc*100, linestyle="--", label=f"Clean: {clean_acc*100:.2f}%")  # 잡음 없는 기준선
    ax.set(xlabel="Input SNR (dB)", ylabel="Validation Accuracy (%)",
           title="S2: AWGN Robustness of Clean-trained Readout")
    ax.set_xticks(sorted(SNR_LEVELS))  # 실험한 SNR만 x축 눈금에 표시
    ax.grid(alpha=0.25)  # 그래프 가독성을 위한 격자
    ax.legend()  # 잡음 조건과 Clean 기준선 구분
    fig.tight_layout()  # 축 및 제목이 겹치지 않도록 정렬
    fig.savefig(OUT_DIR / "noise_accuracy.png", dpi=180)  # 정확도 비교 그래프 저장
    plt.close(fig)  # 사용한 그래프 메모리 정리

    # Fig: Percentage of changed binary Mel pulses
    fig, ax = plt.subplots(figsize=(8, 5))  # 정확도 그래프 영역 생성
    ax.plot([r["snr_db"] for r in levels],  # SNR별 이진 펄스 변화율 그래프
            [r["mean_pulse_change_pct"] for r in levels], marker="o")
    ax.set(xlabel="Input SNR (dB)", ylabel="Changed Binary Pulses (%)",
           title="S2: AWGN Impact on Binary Mel Pulses")
    ax.set_xticks(sorted(SNR_LEVELS))  # 실험한 SNR만 x축 눈금에 표시
    ax.grid(alpha=0.25)  # 그래프 가독성을 위한 격자
    fig.tight_layout()  # 축 및 제목이 겹치지 않도록 정렬
    fig.savefig(OUT_DIR / "noise_pulse_changes.png", dpi=180)  # 펄스 변화율 그래프 저장
    plt.close(fig)  # 사용한 그래프 메모리 정리

    # 7. 실험 완료 후 Clean 기준값 및 SNR별 평균 결과 출력
    print("\nClean:", f"{clean_acc*100:.2f}%", "| Best Epoch:", int(np.argmin(history["val_loss"]))+1)
    for row in summary:
        print(f"SNR {row['snr_db']:>2} dB: {row['mean_accuracy_pct']:.2f}% ± "
              f"{row['std_accuracy_pp']:.2f}%p | Pulse change {row['mean_pulse_change_pct']:.2f}%")


if __name__ == "__main__":  # 파일을 직접 실행할 때만 main 호출
    main()  # S2 잡음 강건성 실험 시작
