# data_loader.py 주요 기능
# 1. load_wav(): WAV 헤더를 직접 읽어 파일 형식을 확인하고, 16-bit PCM 음성 데이터를 NumPy 배열로 변환 및 정규화
# 2. load_dataset(): FSDD 파일명에서 정답 라벨, 화자, 녹음 번호 추출
# 3. split_dataset(): 녹음 번호를 기준으로 Train(2400), Validation(300), Test(300) 데이터 분할
# 4. 실행 및 검증: 전체 데이터 개수와 예시 WAV 파일의 정보를 출력
# 강의 예제와는 달리 stereo가 아닌 mono 16-bit PCM WAV 파일만 지원하며, 샘플링 주파수는 8kHz로 고정되어 있음

from pathlib import Path # 파일이나 폴더의 경로를 편리하게 다루기 위한 라이브러리
import struct # 바이트 단위로 저장된 WAV 헤더를 자료형에 맞게 읽기 위한 라이브러리

import numpy as np # 음성 신호를 다루기 위한 라이브러리, 배열 전체를 한 번에 정규화할 수 있음


SAMPLE_RATE = 8000 # FSDD는 8kHz 샘플링 주파수로 녹음됨
HEADER_SIZE = 44 # WAV 헤더의 크기(바이트 단위)


def load_wav(file_path): # WAV 파일을 읽어 샘플링 주파수와 음성 신호를 반환
    with open(file_path, "rb") as file: # rb: read binary
        # 1-2강: WAV = Header(44 bytes) + audio data
        header = file.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"WAV 헤더가 44바이트보다 짧습니다: {file_path}")

        # little-endian: WAV 헤더의 각 항목을 자료형에 맞게 읽음
        (
            riff, file_size, wave, fmt, fmt_size,
            audio_format, channels, sample_rate, byte_rate,
            block_align, bits_per_sample, data_id, data_size,
        ) = struct.unpack("<4sI4s4sIHHIIHH4sI", header)

        # 예외 처리: WAV 헤더의 각 항목이 예상과 다르면 오류 발생
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"올바른 RIFF/WAVE 파일이 아닙니다: {file_path}")
        if fmt != b"fmt " or fmt_size != 16 or data_id != b"data":
            raise ValueError("표준 44바이트 PCM 헤더만 지원합니다.")
        if audio_format != 1 or channels != 1 or bits_per_sample != 16:
            raise ValueError("mono 16-bit PCM WAV 파일만 지원합니다.")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"샘플링 주파수가 {SAMPLE_RATE} Hz가 아닙니다: {sample_rate} Hz")
        if block_align != channels * (bits_per_sample // 8):
            raise ValueError("WAV 헤더의 block_align 값이 올바르지 않습니다.")
        if byte_rate != sample_rate * block_align:
            raise ValueError("WAV 헤더의 byte_rate 값이 올바르지 않습니다.")

        # 강의 Exercise 1: 데이터 바이트 수로 샘플 개수 계산
        number_of_samples = data_size // (bits_per_sample // 8)
        raw_data = file.read(data_size)
        if len(raw_data) != data_size or len(raw_data) % 2 != 0:
            raise ValueError("WAV 샘플 데이터가 불완전합니다.")

    # C++ short(16 bits)와 동일하게 signed 16-bit 정수로 해석
    signal = np.frombuffer(raw_data, dtype="<i2").astype(np.float32)
    if len(signal) != number_of_samples:
        raise ValueError("헤더에 기록된 샘플 개수와 실제 데이터가 일치하지 않습니다.")

    # [-32768, 32767] -> [-1, 1) 정규화
    signal = signal / 32768.0
    return sample_rate, signal


def load_dataset(data_dir="data/recordings"): # FSDD 파일명 digit_speaker_index.wav에서 정답 정보를 추출
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")

    dataset = []
    for file_path in sorted(data_dir.glob("*.wav")):
        parts = file_path.stem.split("_")
        if len(parts) != 3:
            raise ValueError(f"FSDD 파일명 형식이 아닙니다: {file_path.name}")

        digit = int(parts[0])
        speaker = parts[1]
        index = int(parts[2])
        if digit < 0 or digit > 9:
            raise ValueError(f"숫자 라벨이 0~9 범위를 벗어났습니다: {file_path.name}")

        dataset.append({"path": file_path, "label": digit,
                        "speaker": speaker, "index": index})

    if not dataset:
        raise ValueError(f"WAV 파일이 없습니다: {data_dir}")
    return dataset


def split_dataset(dataset):
    # 녹음 번호를 기준으로 Train/Validation/Test를 분리
    # Test: 0~4 / Validation: 5~9 / Train: 10~49 
    
    train_data = []
    val_data = []
    test_data = []

    for item in dataset:
        index = item["index"]
        if 0 <= index <= 4:
            test_data.append(item)
        elif 5 <= index <= 9:
            val_data.append(item)
        elif 10 <= index <= 49:
            train_data.append(item)
        else:
            raise ValueError(f"예상하지 못한 녹음 번호: {index}")

    return train_data, val_data, test_data

# 테스트 코드: 데이터셋을 로드하고 분리한 후, 첫 번째 샘플의 정보를 출력
if __name__ == "__main__":
    dataset = load_dataset()
    train_data, val_data, test_data = split_dataset(dataset)

    print("전체 WAV 개수:", len(dataset))
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    sample = dataset[0]
    sample_rate, signal = load_wav(sample["path"])
    print("예시 파일:", sample["path"].name)
    print("정답 숫자:", sample["label"])
    print("화자:", sample["speaker"])
    print("Sample rate:", sample_rate, "Hz")
    print("Number of samples:", len(signal))
    print("Duration:", len(signal) / sample_rate, "seconds")

""" 

----- 실행결과 -----
전체 WAV 개수: 3000
Train: 2400
Validation: 300
Test: 300
예시 파일: 0_george_0.wav
정답 숫자: 0
화자: george
Sample rate: 8000 Hz
Number of samples: 2384
Duration: 0.298 seconds

"""