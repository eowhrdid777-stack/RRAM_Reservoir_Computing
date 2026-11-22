# main.py 주요 기능

# 1. train(): FSDD Train/Validation 데이터를 처리하고 Readout 학습 및 모델 저장
# 2. evaluate(): 저장된 모델로 Validation 또는 Test 정확도와 Confusion Matrix 평가
# 3. predict(): WAV 파일 하나의 숫자를 예측
# 4. main(): 실행 모드(train, evaluate, predict, all)에 따라 F1~F6 기능 연결
# 5. 실행 및 검증: 학습 곡선, 평가 결과, 개별 음성 예측 출력

# Test 데이터는 모델을 최종 확정한 뒤 --split test로 평가함

from pathlib import Path # 프로젝트 및 결과 파일 경로 처리
import argparse # 실행 모드와 WAV 파일 경로 입력

import matplotlib.pyplot as plt # Readout 학습 곡선 시각화
import numpy as np # 그래프용 배열 및 난수 시드
import torch # Readout 가중치 저장 및 난수 시드

from data_loader import load_dataset, load_wav, split_dataset # F1: WAV 로드 및 분할
from preprocessing import signal_to_pulses # F2: 4비트 펄스 생성
from rram_model import generate_reservoir_states # F3-F4: RRAM 상태 벡터 생성
from readout import Readout, prepare_data, train_readout, predict_digit, SEED # F5: Readout 학습 및 예측
from evaluation import (load_model, evaluate_model, make_confusion_matrix,
                        calculate_class_accuracy, plot_confusion_matrix) # F6: 성능 평가


PROJECT_DIR = Path(__file__).resolve().parent.parent # src 폴더의 상위 경로
DATA_DIR = PROJECT_DIR / "data" / "recordings" # FSDD WAV 파일 폴더
RESULT_DIR = PROJECT_DIR / "results" # 모델 및 그래프 저장 폴더
MODEL_PATH = RESULT_DIR / "readout_model.pth" # 학습된 Readout 가중치


def train(): # Train/Validation 데이터로 모델을 학습하고 결과 저장
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = load_dataset(DATA_DIR)
    train_data, val_data, test_data = split_dataset(dataset)
    print("전체 WAV 개수:", len(dataset))
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test (학습에 미사용):", len(test_data))

    print("Train Reservoir State 생성 중...")
    train_x, train_y = prepare_data(train_data)
    print("Validation Reservoir State 생성 중...")
    val_x, val_y = prepare_data(val_data)
    print("Train 입력 크기:", tuple(train_x.shape))
    print("Validation 입력 크기:", tuple(val_x.shape))

    model = Readout()
    model, history = train_readout(model, train_x, train_y, val_x, val_y)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("학습 모델 저장:", MODEL_PATH)

    # F5 학습 결과를 그래프로 저장
    epochs_axis = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs_axis, history["train_loss"], label="Train")
    axes[0].plot(epochs_axis, history["val_loss"], label="Validation")
    axes[0].set_title("Readout Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs_axis, np.array(history["train_accuracy"]) * 100, label="Train")
    axes[1].plot(epochs_axis, np.array(history["val_accuracy"]) * 100, label="Validation")
    axes[1].set_title("Readout Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    image_path = RESULT_DIR / "readout_training.png"
    plt.savefig(image_path, dpi=150)
    plt.close(fig)
    print("학습 그래프 저장:", image_path)


def evaluate(split="val"): # Validation 또는 Test 데이터의 인식 성능 평가
    if split not in ("val", "test"):
        raise ValueError("평가 데이터는 val 또는 test여야 합니다.")

    model = load_model(MODEL_PATH)
    dataset = load_dataset(DATA_DIR)
    _, val_data, test_data = split_dataset(dataset)
    selected_data = val_data if split == "val" else test_data
    split_name = "Validation" if split == "val" else "Test"

    print("평가 데이터:", split_name)
    print("데이터 개수:", len(selected_data))
    print("Reservoir State 생성 중...")
    x, y = prepare_data(selected_data)

    loss, accuracy, labels, predictions = evaluate_model(model, x, y)
    matrix = make_confusion_matrix(labels, predictions)
    class_accuracies = calculate_class_accuracy(matrix)

    print(f"{split_name} Loss: {loss:.4f}")
    print(f"{split_name} Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(matrix)
    print("숫자별 정확도:")
    for digit, value in enumerate(class_accuracies):
        print(f"{digit}: {value * 100:.2f}%")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = RESULT_DIR / f"confusion_matrix_{split}.png"
    plot_confusion_matrix(matrix, image_path)
    print("Confusion Matrix 저장:", image_path)


def predict(file_path): # 선택한 WAV 파일 하나를 숫자 0~9로 분류
    model = load_model(MODEL_PATH)
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"WAV 파일이 없습니다: {file_path}")

    sample_rate, signal = load_wav(file_path)
    _, pulses = signal_to_pulses(signal, sample_rate)
    state = generate_reservoir_states(pulses)
    digit = predict_digit(model, state)

    print("입력 파일:", file_path.name)
    print("예측 숫자:", digit)


def main(): # 사용자 실행 옵션에 맞춰 학습, 평가, 예측 수행
    parser = argparse.ArgumentParser(description="RRAM Reservoir Computing 기반 FSDD 숫자 인식")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict", "all"],
                        default="all", help="기본값 all: 학습 후 Validation 평가")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="평가 데이터 선택 (기본값 val)")
    parser.add_argument("--file", type=str, help="predict 모드에서 사용할 WAV 경로")
    args = parser.parse_args()

    if args.mode == "predict" and not args.file:
        parser.error("predict 모드에서는 --file로 WAV 파일 경로를 지정하세요.")

    if args.mode in ("train", "all"):
        train()
    if args.mode in ("evaluate", "all"):
        evaluate(args.split)
    if args.mode == "predict":
        predict(args.file)


# 테스트 코드: 프로젝트 전체 기능을 실행 모드에 따라 연결
if __name__ == "__main__":
    main()
