# preprocessing.py 주요 기능

# 1. normalize_signal(): WAV 음성 신호의 최대 진폭을 기준으로 정규화
# 2. hz_to_mel(), mel_to_hz(): 주파수(Hz)와 Mel 척도 사이의 변환
# 3. make_mel_filterbank(): 주파수 성분을 Mel 채널로 변환하는 삼각 필터 행렬 생성
# 4. extract_features(): FFT 및 행렬곱으로 32×32 로그 Mel 특징 추출
# 5. normalize_features(): Mel 특징값을 0~1 범위로 정규화
# 6. signal_to_pulses(): Mel 특징을 이진화하여 4비트 RRAM 입력 펄스로 변환
# 7. 실행 및 검증: 예시 WAV 파일의 특징 및 펄스 크기 출력, 신호·특징·펄스 시각화

# 강의 연계: 1-1강 Sampling Theory(샘플링 주파수), 4-1강 Matrix(행렬곱)
# 강의 외 추가 구현: FFT(Fast Fourier Transform), Mel-Spectrogram, 이진화 및 펄스 인코딩

# Mel Scale: 인간의 청각 특성을 반영한 주파수 척도. 낮은 주파수는 선형, 높은 주파수는 로그 스케일로 변환하여 음성 특징을 추출하는 데 사용.
# m = 2595 * log10(1 + f / 700), f: 주파수(Hz), m: Mel 척도
# N_MELS x N_FRAMES 크기의 Mel 특징 행렬을 생성하고 이를 BITS_PER_PULSE로 나눈 만큼의 RRAM이 필요

from pathlib import Path # WAV 파일 경로를 설정하기 위한 라이브러리

import matplotlib.pyplot as plt # WAV 신호와 전처리 결과를 그래프로 확인하기 위한 라이브러리
import numpy as np # FFT, 배열 및 행렬 연산을 위한 라이브러리

from data_loader import load_dataset, load_wav # data_loader에 작성한 WAV 로더 함수 사용


SAMPLE_RATE = 8000 # FSDD의 샘플링 주파수(Hz), 상수로 고정
FFT_SIZE = 256 # 한 번의 FFT에서 분석하는 샘플 개수
HOP_LENGTH = 128 # FFT 시간 구간의 이동 간격(샘플), FFT_SIZE/2로 설정하여 50% 오버랩
N_MELS = 32 # Mel 주파수 채널 수, 129개의 주파수 성분을 32개의 Mel 채널로 변환
N_FRAMES = 32 # 음성마다 사용할 시간 구간 수, wav 파일 길이가 다르므로 32개의 시간 구간으로 보간하여 동일한 크기로 변환
BITS_PER_PULSE = 4 # 하나의 RRAM 입력 시퀀스를 구성하는 비트 수
THRESHOLD = 0.5 # 정규화된 Mel 특징을 0과 1로 나누는 기준값
# T = FFT_SIZE / SAMPLE_RATE = 0.032초, H = HOP_LENGTH / SAMPLE_RATE = 0.016초, N_FRAMES * H = 0.512초

def normalize_signal(signal): # 음성의 최대 절댓값을 기준으로 진폭을 정규화
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("음성 신호는 비어 있지 않은 1차원 배열이어야 합니다.")

    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal.copy()
    return signal / peak


def hz_to_mel(frequency): # 일반 주파수(Hz)를 Mel 척도로 변환
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel): # Mel 척도를 일반 주파수(Hz)로 변환
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(sample_rate, fft_size=FFT_SIZE, n_mels=N_MELS): # Mel 삼각 필터 행렬 생성
    if n_mels <= 0:
        raise ValueError("Mel 채널 수는 양수여야 합니다.")

    # FFT 결과에 대응하는 주파수: 0 ~ sample_rate/2 (Nyquist 주파수)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate) # rfftfreq: real FFT에 대응하는 주파수 배열 생성
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2) # Mel 척도에서 n_mels+2개의 균등한 점 생성
    hz_points = mel_to_hz(mel_points) # Mel 척도 점을 다시 Hz로 변환하여 필터의 경계 주파수 결정
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32) # Mel 채널 수 × 주파수 성분 수 크기의 필터 행렬 초기화

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # 각 주파수 구간에 (left, 0), (center, 1), (right, 0)의 삼각 필터 적용
        # 생성된 행렬 크기는 32 × 129, 각 Mel 채널마다 주파수 성분에 대한 가중치가 저장됨
        rise = (frequencies - left) / (center - left)
        fall = (right - frequencies) / (right - center)
        filters[i] = np.maximum(0, np.minimum(rise, fall))

        # 필터별 합이 1이 되도록 정규화
        total = np.sum(filters[i])
        if total > 0:
            filters[i] = filters[i] / total

    return filters


def extract_features(signal, sample_rate, n_mels=N_MELS, n_frames=N_FRAMES): # 음성 신호로부터 로그 Mel 특징 추출
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{SAMPLE_RATE} Hz 음성만 지원합니다.")
    if n_frames <= 0:
        raise ValueError("시간 프레임 수는 양수여야 합니다.")

    signal = normalize_signal(signal)
    window = np.hanning(FFT_SIZE) # FFT 구간의 양끝을 부드럽게 만드는 Hann window -> 불연속에 따른 스펙트럼 누출 감소

    # 음성을 일정 길이의 시간 구간으로 나누어 FFT 수행
    spectra = []
    for start in range(0, len(signal), HOP_LENGTH):
        segment = signal[start:start + FFT_SIZE]
        if len(segment) < FFT_SIZE:
            segment = np.pad(segment, (0, FFT_SIZE - len(segment))) # 부족한 길이는 0으로 채움

        fft_result = np.fft.rfft(segment * window)
        spectra.append(np.abs(fft_result) ** 2) # 주파수 성분의 크기 제곱(파워)

    # 파워 스펙트럼: (주파수 성분 수, 시간 구간 수)
    power_spectrum = np.array(spectra, dtype=np.float32).T

    # 4-1강 Matrix: (Mel 채널, 주파수) @ (주파수, 시간) 행렬곱
    filters = make_mel_filterbank(sample_rate, FFT_SIZE, n_mels)
    mel_spectrum = filters @ power_spectrum
    log_mel = np.log1p(mel_spectrum) # log(1 + x)로 특징값의 범위를 압축

    # 서로 다른 WAV 길이를 동일한 n_frames개 시간 구간으로 변환
    old_time = np.linspace(0, 1, log_mel.shape[1])
    new_time = np.linspace(0, 1, n_frames)
    features = np.zeros((n_mels, n_frames), dtype=np.float32)
    for i in range(n_mels):
        features[i] = np.interp(new_time, old_time, log_mel[i])

    return features


def normalize_features(features): # 한 WAV의 Mel 특징을 0~1 범위로 정규화
    minimum = np.min(features)
    maximum = np.max(features)
    if maximum - minimum < 1e-12:
        return np.zeros_like(features, dtype=np.float32)
    return (features - minimum) / (maximum - minimum)


def signal_to_pulses(signal, sample_rate, threshold=THRESHOLD,
                     bits_per_pulse=BITS_PER_PULSE, n_mels=N_MELS): # 음성 신호를 RRAM 입력 펄스로 변환
    if not 0 <= threshold <= 1:
        raise ValueError("이진화 임계값은 0~1 사이여야 합니다.")
    if bits_per_pulse <= 0 or N_FRAMES % bits_per_pulse != 0:
        raise ValueError("펄스 묶음 길이는 N_FRAMES의 양의 약수여야 합니다.")

    features = extract_features(signal, sample_rate, n_mels=n_mels)
    features = normalize_features(features)
    binary = (features >= threshold).astype(np.int8) # 임계값 이상은 1, 미만은 0

    # Mel 채널별 연속된 시간 특징을 4비트 시퀀스로 묶음 -> 256개의 sequnces 생성
    pulses = binary.reshape(-1, bits_per_pulse)
    return features, pulses


# 테스트 코드: 첫 번째 FSDD WAV 파일의 전처리 결과를 출력하고 그래프로 확인
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data" / "recordings"
    dataset = load_dataset(data_dir)
    sample_rate, signal = load_wav(dataset[0]["path"])
    features, pulses = signal_to_pulses(signal, sample_rate)

    print("예시 WAV 파일:", dataset[0]["path"].name)
    print("음성 샘플 개수:", len(signal))
    print("Mel 특징 크기:", features.shape)
    print("펄스 배열 크기:", pulses.shape)
    print("활성 펄스 비율: {:.2f}%".format(100 * np.mean(pulses)))
    print("첫 8개 펄스:\n", pulses[:8])

    # 음성 신호, 로그 Mel 특징, 이진화된 펄스를 순서대로 표시
    time_axis = np.arange(len(signal)) / sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(time_axis, signal)
    axes[0].set_title("Original WAV signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].imshow(features, origin="lower", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Normalized log-Mel spectrogram")
    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Mel channel")

    axes[2].imshow(pulses.reshape(N_MELS, N_FRAMES), origin="lower",
                   aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("Binary pulses (4 bits per sequence)")
    axes[2].set_xlabel("Time frame")
    axes[2].set_ylabel("Mel channel")

    plt.tight_layout()
    plt.show()
