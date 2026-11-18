# RRAM 기반 Reservoir Computing(RC)을 이용한 Free Spoken Digit Dataset(FSDD) 음성 숫자 인식 시스템

FSDD 음성 데이터셋을 활용하여 RRAM의 비선형 응답과 단기 기억 특성을 모사하는 Reservoir Computing 기반 음성 숫자 인식 시스템을 구현한다.

## 1. 강의 연계

data_loader.py: 1-2강 wav: wav header file 및 음성 data 읽기
preprocessing.py: 1-1강 Sampling Theory: 샘플링 주파수 및 시간축 계산
                  4-1강 Matrix: Mel 필터 행렬과 주파수 특징의 행렬곱


## 2. 개발 환경

- Python 3.14.3 (Visual Studio Code)
- pathlib, numpy, struct
- Dataset: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)

## 3. 프로그램 구성

data_loader.py: WAV 헤더 분석 및 FSDD 데이터 분할 
preprocessing.py: 음성 신호의 Mel 특징 추출 및 4비트 RRAM 입력 펄스 생성

## 4. 설치 및 실행


## 5. 실험 결과


## 6. AI 활용 내역

**F1. FSDD data load**
- 도구: ChatGPT
- 용도: 1-2강 WAV 실습을 기반으로 PCM 헤더 파싱 및 데이터 분할 코드 작성 보조
- 주요 프롬프트: 강의 Exercise 1을 활용하여 WAV 헤더를 직접 읽고 FSDD 데이터셋을 분할하는 Python 코드 구현 요청
- 검증: WAV 헤더 정보 및 샘플 개수, Train/Validation/Test 분할 결과 확인

F2. Speech preprocessing

- 도구: ChatGPT
- 용도: FFT 및 Mel 필터 행렬을 이용한 음성 특징 추출과 이진 펄스 인코딩 구현 보조
- 주요 프롬프트: 강의의 Sampling Theory와 Matrix 개념을 활용하여 WAV 음성을 Mel-Spectrogram으로 변환하고 4비트 펄스로 구성하도록 요청. 기존 data_loader.py의 코드 및 주석 형식 유지 요청
- 검증: Mel 특징 크기(32×32), 펄스 배열 크기(256×4) 및 시각화 결과 확인

## 7. 참고자료

- Software Lab 강의자료 (1강, 4강, )
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Du et al., *Reservoir computing using dynamic memristors for temporal information processing*, Nature Communications (2017)
