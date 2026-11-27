# d2d_variation.py 주요 기능

# 1. RRAM 소자 576개의 G_MIN/G_MAX에 D2D Variation 적용
# 2. Variation 수준(0%, 5%, 10%, 20%)별 Validation Accuracy 비교
# 3. 소자별 특성은 Train/Validation 전체에서 고정하고 난수 시드별 반복
# 4. retrain: 각 Variation 조건에서 Readout 재학습
# 5. fixed: 이상적인 소자로 학습한 Readout을 고정하고 Variation 적용
# 6. 576개 소자의 컨덕턴스 응답을 한 그래프에 표시
# 7. 평균/표준편차를 CSV와 그래프로 저장 (Test 데이터는 사용하지 않음)

from pathlib import Path # 결과 저장 경로 처리
import argparse # Variation, 실험 모드, 반복 횟수 선택
import csv # 실험 결과 CSV 저장

import matplotlib.pyplot as plt # Variation 수준별 정확도 시각화
import numpy as np # 결과 평균 및 표준편차 계산
import torch # Readout 가중치 초기화 및 학습

from data_loader import load_dataset, split_dataset # F1: Train/Validation 분리
from rram_model import create_device_windows, process_sequence, BITS_PER_PULSE # F3-F4: 소자별 Conductance Window 및 응답 계산
from readout import Readout, prepare_data, train_readout, calculate_accuracy # F5: Readout 학습


PROJECT_DIR = Path(__file__).resolve().parent.parent # src 폴더 상위의 프로젝트 경로
DATA_DIR = PROJECT_DIR / "data" / "recordings"
RESULT_DIR = PROJECT_DIR / "results" / "d2d"
NUM_DEVICES = 576 # Reservoir State 벡터 차원



def plot_device_responses(windows, variation, seed, pattern=(1, 1, 1, 1), show=False): # 576개 소자 응답을 한 그래프에 표시
    # 모든 소자에 동일한 펄스를 입력해야 소자 간 차이만 비교할 수 있음
    x = np.arange(BITS_PER_PULSE + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    histories = []

    for g_min, g_max in windows:
        _, history = process_sequence(pattern, g_min=float(g_min), g_max=float(g_max))
        histories.append(history)
        ax.plot(x, history, linewidth=0.65, alpha=0.18, color="tab:blue")

    # 576개 곡선이 겹칠 때 기준이 보이도록 평균 응답도 표시
    mean_history = np.mean(histories, axis=0)
    ax.plot(x, mean_history, color="tab:red", linewidth=2.5, label="Mean of 576 devices")
    pattern_label = "".join(map(str, pattern))
    ax.set(title=f"D2D {variation * 100:g}% | 576 RRAM responses | Pulse {pattern_label}",
           xlabel="Input step", ylabel="Conductance (normalized units)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    variation_label = f"{variation * 100:g}".replace(".", "p")
    save_path = RESULT_DIR / f"d2d_conductance_{variation_label}pct_seed{seed}.png"
    fig.savefig(save_path, dpi=180)
    print("576개 소자 응답 그래프 저장:", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def run_experiment(variations, repeats, mode, epochs, seed, show=False): # D2D 조건에 따른 정확도 측정
    dataset = load_dataset(DATA_DIR)
    train_data, val_data, _ = split_dataset(dataset) # Test 데이터는 실험 선택에 사용하지 않음
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for repeat in range(repeats):
        run_seed = seed + repeat
        torch.manual_seed(run_seed) # 조건별 Readout 초기화 및 배치 순서 비교 조건 고정
        np.random.seed(run_seed)

        baseline_windows = create_device_windows(NUM_DEVICES, variation=0.0, seed=run_seed)
        if mode == "fixed":
            print(f"\n반복 {repeat + 1}/{repeats}: Baseline Readout 학습 중")
            base_train_x, base_train_y = prepare_data(train_data, baseline_windows)
            base_val_x, base_val_y = prepare_data(val_data, baseline_windows)
            fixed_model, _ = train_readout(Readout(), base_train_x, base_train_y,
                                           base_val_x, base_val_y, epochs=epochs)

        for variation in variations:
            windows = create_device_windows(NUM_DEVICES, variation=variation, seed=run_seed)
            print(f"\n반복 {repeat + 1}/{repeats} | D2D: {variation * 100:g}% | mode: {mode}")
            if repeat == 0: # 같은 Variation 수준에서는 첫 번째 시드의 576개 소자만 시각화
                plot_device_responses(windows, variation, run_seed, show=show)

            print("G_MIN 평균:", round(float(windows[:, 0].mean()), 4),
                  "G_MAX 평균:", round(float(windows[:, 1].mean()), 4))

            val_x, val_y = prepare_data(val_data, windows)
            if mode == "retrain":
                train_x, train_y = prepare_data(train_data, windows)
                torch.manual_seed(run_seed)
                model, _ = train_readout(Readout(), train_x, train_y, val_x, val_y,
                                         epochs=epochs)
            else:
                model = fixed_model # Baseline Readout의 가중치를 그대로 유지

            accuracy = calculate_accuracy(model, val_x, val_y)
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")
            records.append({"mode": mode, "repeat": repeat + 1, "seed": run_seed,
                            "variation": variation, "accuracy": accuracy})

    return records


def save_results(records, mode): # 정확도 CSV와 평균·표준편차 그래프 저장
    csv_path = RESULT_DIR / f"d2d_{mode}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["mode", "repeat", "seed", "variation", "accuracy"])
        writer.writeheader()
        writer.writerows(records)

    levels = sorted({item["variation"] for item in records})
    means = []
    stds = []
    for level in levels:
        values = np.array([r["accuracy"] for r in records if r["variation"] == level]) * 100
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        print(f"D2D {level * 100:>5.1f}%: {means[-1]:.2f}% ± {stds[-1]:.2f}%p")

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(np.array(levels) * 100, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel("D2D variation (% of reference conductance window)")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"RRAM D2D Variation - {mode}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figure_path = RESULT_DIR / f"d2d_{mode}.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print("CSV 저장:", csv_path)
    print("그래프 저장:", figure_path)


def main(): # 실행 조건 입력 및 실험 진행
    parser = argparse.ArgumentParser(description="RRAM Conductance Window D2D Variation 실험")
    parser.add_argument("--variations", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2],
                        help="기준 Window에 대한 소자별 표준편차 비율 (0.05 = 5%%)")
    parser.add_argument("--repeats", type=int, default=3, help="각 조건을 반복할 난수 시드 개수")
    parser.add_argument("--mode", choices=["retrain", "fixed"], default="retrain",
                        help="retrain: 조건별 재학습 / fixed: Baseline Readout 고정")
    parser.add_argument("--epochs", type=int, default=100, help="Readout 학습 Epoch")
    parser.add_argument("--seed", type=int, default=42, help="초기 난수 시드")
    parser.add_argument("--show", action="store_true", help="576개 소자 응답 그래프 창도 표시")
    args = parser.parse_args()
    if args.repeats <= 0 or args.epochs <= 0 or not args.variations or any(v < 0 for v in args.variations):
        parser.error("repeats, epochs는 양수, variations는 0 이상이어야 합니다.")

    records = run_experiment(sorted(set(args.variations)), args.repeats,
                             args.mode, args.epochs, args.seed, show=args.show)
    save_results(records, args.mode)


# 테스트 코드: 결과는 results/d2d에 저장하고 F1~F6 기본 실행은 변경하지 않음
if __name__ == "__main__":
    main()
