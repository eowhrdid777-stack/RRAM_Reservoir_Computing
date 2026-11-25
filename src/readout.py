# readout.py 주요 기능
# 1. prepare_data(): WAV 음성을 256차원 Reservoir State로 변환하고 숫자 라벨과 묶음
# 2. Readout(): 256차원 입력을 10개 숫자 클래스의 점수로 변환하는 선형 계층
# 3. calculate_accuracy(): 모델이 예측한 숫자와 실제 정답을 비교하여 정확도 계산
# 4. train_readout(): Train 데이터로 Readout 가중치를 학습하고 Validation으로 검증
# 5. predict_digit(): 학습된 Readout으로 음성 하나의 숫자 예측
# 6. 실행 및 검증: 학습 로그 및 Loss/Accuracy 그래프 출력, 최적 모델 저장
# Reservoir의 가중치는 학습하지 않고 Readout의 가중치와 편향만 학습함
# Test 데이터는 학습과 모델 선택에 사용하지 않으며, F5에서 최종 평가 예정

from pathlib import Path # 데이터 및 학습 모델 파일의 경로 처리

import matplotlib.pyplot as plt # 학습 과정의 Loss 및 Accuracy 시각화
import numpy as np # 배열과 라벨 처리
import torch # 텐서 연산 및 Readout 학습
import torch.nn as nn # 선형 계층과 Cross-Entropy Loss

from data_loader import load_dataset, load_wav, split_dataset # F1의 음성 파일 로드 및 데이터 분할
from preprocessing import signal_to_pulses # F2의 4비트 펄스 생성
from rram_model import generate_reservoir_states # F3의 RRAM Reservoir State 생성


INPUT_SIZE = 256 # 한 WAV에서 생성되는 Reservoir State의 차원 수
NUM_CLASSES = 10 # 숫자 0~9에 해당하는 출력 클래스 수
EPOCHS = 300 # 전체 Train 데이터를 반복 학습하는 횟수
BATCH_SIZE = 64 # 한 번에 학습할 음성 데이터 수
LEARNING_RATE = 0.001 # Adam Optimizer의 학습률
SEED = 42 # 동일한 조건에서 실험을 재현하기 위한 난수 시드


def prepare_data(dataset, device_windows=None): # WAV 데이터를 Reservoir State와 정답 라벨 배열로 변환
    states = []
    labels = []

    for item in dataset:
        sample_rate, signal = load_wav(item["path"])
        _, pulses = signal_to_pulses(signal, sample_rate)
        state = generate_reservoir_states(pulses, device_windows=device_windows)
        if state.shape != (INPUT_SIZE,):
            raise ValueError(f"Reservoir State 크기가 ({INPUT_SIZE},)가 아닙니다: {item['path']}")

        states.append(state)
        labels.append(item["label"])

    if not states:
        raise ValueError("처리할 음성 데이터가 없습니다.")

    # 한 행에 음성 하나의 256차원 특징을 저장
    x = torch.tensor(np.array(states), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss의 정답 라벨은 정수형
    return x, y


class Readout(nn.Module): # Reservoir State를 숫자 0~9의 점수로 변환하는 선형 분류기
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, NUM_CLASSES)

    def forward(self, x): # y = Wx + b (Softmax 적용 전 점수 반환)
        return self.linear(x)


def calculate_accuracy(model, x, y): # 정답과 예측값이 일치한 비율 계산
    model.eval() # 평가 모드로 전환
    with torch.no_grad(): # 검증 중에는 기울기를 계산하지 않음
        scores = model(x)
        predictions = torch.argmax(scores, dim=1) # 가장 높은 점수의 숫자 선택
        accuracy = (predictions == y).float().mean().item()
    return accuracy


def train_readout(model, train_x, train_y, val_x, val_y,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE): # Train 학습 및 Validation 검증
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, learning_rate는 양수여야 합니다.")
    if len(train_x) == 0 or len(val_x) == 0:
        raise ValueError("Train과 Validation 데이터는 비어 있으면 안 됩니다.")

    criterion = nn.CrossEntropyLoss() # 숫자 분류에서 사용하는 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train() # Readout의 가중치를 업데이트하는 학습 모드
        indices = torch.randperm(len(train_x)) # 매 Epoch마다 Train 데이터 순서를 섞음

        for start in range(0, len(train_x), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]

            optimizer.zero_grad() # 이전 배치에서 계산한 기울기 초기화
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            loss.backward() # Loss를 각 가중치로 미분
            optimizer.step() # 가중치 및 편향 업데이트

        # Epoch가 끝나면 Train과 Validation에 대한 성능 계산
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(train_x), train_y).item()
            val_loss = criterion(model(val_x), val_y).item()
        train_accuracy = calculate_accuracy(model, train_x, train_y)
        val_accuracy = calculate_accuracy(model, val_x, val_y)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Validation Loss가 가장 낮은 Epoch의 가중치만 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = {name: value.detach().clone()
                            for name, value in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

    model.load_state_dict(best_weights) # 최종 Epoch가 아닌 Validation Loss가 가장 낮은 모델 복원
    print(f"최적 Epoch: {best_epoch}, Validation Loss: {best_val_loss:.4f}")
    return model, history


def predict_digit(model, state): # 256차원 Reservoir State 하나를 숫자 0~9로 분류
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (INPUT_SIZE,):
        raise ValueError(f"입력은 ({INPUT_SIZE},) 크기의 Reservoir State여야 합니다.")

    model.eval()
    with torch.no_grad():
        x = torch.tensor(state).unsqueeze(0) # (256,) -> (1, 256)
        scores = model(x)
        digit = torch.argmax(scores, dim=1).item()
    return int(digit)


# 테스트 코드: FSDD Train/Validation으로 학습하고 학습 곡선과 저장 모델 확인
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    project_dir = Path(__file__).resolve().parent.parent
    dataset = load_dataset(project_dir / "data" / "recordings")
    train_data, val_data, test_data = split_dataset(dataset)
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

    # F5 평가에서 같은 모델을 사용할 수 있도록 가중치 저장
    model_dir = project_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "readout_model.pth"
    torch.save(model.state_dict(), model_path)
    print("학습 모델 저장:", model_path)

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
    plt.savefig(model_dir / "readout_training.png", dpi=150)
    plt.show()
