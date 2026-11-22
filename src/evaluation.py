# evaluation.py 주요 기능

# 1. load_model(): F5에서 학습한 Readout 가중치를 불러옴
# 2. evaluate_model(): 정답과 예측값을 비교하여 Loss 및 전체 정확도 계산
# 3. make_confusion_matrix(): 실제 숫자와 예측 숫자의 관계를 10×10 행렬로 계산
# 4. calculate_class_accuracy(): 숫자 0~9 각각의 인식 정확도 계산
# 5. plot_confusion_matrix(): Confusion Matrix를 그래프로 저장
# 6. 실행 및 검증: Validation 또는 Test 데이터를 평가하고 결과 출력 및 저장

# Validation은 개선 실험에, Test는 최종 모델을 확정한 뒤 평가하는 용도로 사용함

from pathlib import Path # 저장된 모델 및 결과 파일 경로 처리
import argparse # 실행할 데이터 분할(Validation/Test)을 선택하기 위한 라이브러리

import matplotlib.pyplot as plt # Confusion Matrix 시각화
import numpy as np # 행렬 및 정확도 계산
import torch # Readout 모델 불러오기와 예측
import torch.nn as nn # Cross-Entropy Loss 계산

from data_loader import load_dataset, split_dataset # F1의 데이터 목록 및 분할
from readout import NUM_CLASSES, Readout, prepare_data # F5의 모델과 Reservoir State 생성


def load_model(model_path): # 저장된 Readout 가중치를 불러와 평가용 모델 생성
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"학습된 모델 파일이 없습니다: {model_path}\n먼저 python src/readout.py를 실행하세요.")

    model = Readout()
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model


def evaluate_model(model, x, y): # Loss, 정확도, 실제 정답과 예측 라벨 계산
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        raise ValueError("평가 데이터가 비어 있거나 입력과 정답 개수가 다릅니다.")

    model.eval()
    with torch.no_grad(): # 평가 과정에서 가중치를 업데이트하지 않음
        scores = model(x)
        loss = nn.CrossEntropyLoss()(scores, y).item()
        predictions = torch.argmax(scores, dim=1)
        accuracy = (predictions == y).float().mean().item()

    return loss, accuracy, y.cpu().numpy(), predictions.cpu().numpy()


def make_confusion_matrix(labels, predictions): # 실제 숫자(행), 예측 숫자(열)의 개수 계산
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def calculate_class_accuracy(matrix): # 숫자별 정답 수를 해당 숫자의 전체 개수로 나눔
    accuracies = np.zeros(NUM_CLASSES, dtype=np.float32)
    for digit in range(NUM_CLASSES):
        total = np.sum(matrix[digit])
        if total > 0:
            accuracies[digit] = matrix[digit, digit] / total
    return accuracies


def plot_confusion_matrix(matrix, save_path): # Confusion Matrix를 이미지로 저장
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))
    plt.colorbar(label="Number of samples")

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            plt.text(col, row, str(matrix[row, col]),
                     ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# 테스트 코드: 저장된 Readout을 불러와 Validation 또는 Test 데이터의 성능 확인
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="val: 개선 실험용 / test: 최종 성능 평가용")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "results" / "readout_model.pth"
    result_dir = project_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(project_dir / "data" / "recordings")
    train_data, val_data, test_data = split_dataset(dataset)
    selected_data = val_data if args.split == "val" else test_data
    split_name = "Validation" if args.split == "val" else "Test"

    print("평가 데이터:", split_name)
    print("데이터 개수:", len(selected_data))
    print("Reservoir State 생성 중...")
    x, y = prepare_data(selected_data)
    model = load_model(model_path)

    loss, accuracy, labels, predictions = evaluate_model(model, x, y)
    matrix = make_confusion_matrix(labels, predictions)
    class_accuracies = calculate_class_accuracy(matrix)

    print(f"{split_name} Loss: {loss:.4f}")
    print(f"{split_name} Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(matrix)
    print("숫자별 정확도:")
    for digit in range(NUM_CLASSES):
        print(f"{digit}: {class_accuracies[digit] * 100:.2f}%")

    image_path = result_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(matrix, image_path)
    print("Confusion Matrix 저장:", image_path)
