# 주제: RRAM 기반 Reservoir Computing(RC)을 이용한 Free Spoken Digit Dataset(FSDD) 음성 숫자 인식 시스템

FSDD 음성 데이터셋을 활용하여 RRAM의 비선형 응답과 단기 기억 특성을 모사하는 Reservoir Computing 기반 음성 숫자 인식 시스템을 구현한다.

## 1. 강의 연계

data_loader.py: 1-2강 wav: wav header file 및 음성 data 읽기
preprocessing.py: 1-1강 Sampling Theory: 샘플링 주파수 및 시간축 계산
                  4-1강 Matrix: Mel 필터 행렬과 주파수 특징의 행렬곱
readout.py: 8강 Deep Learning – CNN: PyTorch 모델 정의, Linear Layer 및 손실 함수와 Optimizer를 이용한 학습·평가 과정


## 2. 개발 환경

- Python 3.14.3 (Visual Studio Code)
- pathlib, numpy, struct
- Dataset: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)

## 3. 프로그램 구성

data_loader.py (F1): WAV 헤더 분석 및 FSDD 데이터 분할
preprocessing.py (F2): 음성 신호의 Mel 특징 추출 및 4비트 RRAM 입력 펄스 생성
rram_model.py (F3, F4): 4비트 펄스 입력에 따른 RRAM conductance 변화 및 256차원 Reservoir State 생성
readout.py (F5): 256차원 Reservoir State를 입력받아 선형 Readout Layer를 학습하고, Validation Loss를 기준으로 최적 모델 선택 및 숫자 0~9 분류

## 4. 설치 및 실행


## 5. 실험 결과


## 6. AI 활용 내역

**F1. FSDD data load**
- 도구: ChatGPT
- 용도: 1-2강 WAV 실습을 기반으로 PCM 헤더 파싱 및 데이터 분할 코드 작성 보조
- 주요 프롬프트: 강의 Exercise 1을 활용하여 WAV 헤더를 직접 읽고 FSDD 데이터셋을 분할하는 Python 코드 구현 요청
- 검증: WAV 헤더 정보 및 샘플 개수, Train/Validation/Test 분할 결과 확인

**F2. Speech preprocessing**
- 도구: ChatGPT
- 용도: FFT 및 Mel 필터 행렬을 이용한 음성 특징 추출과 이진 펄스 인코딩 구현 보조
- 주요 프롬프트: 강의의 Sampling Theory와 Matrix 개념을 활용하여 WAV 음성을 Mel-Spectrogram으로 변환하고 4비트 펄스로 구성하도록 요청. 기존 data_loader.py의 코드 및 주석 형식 유지 요청
- 검증: Mel 특징 크기(32×32), 펄스 배열 크기(256×4) 및 시각화 결과 확인

**F3, F4. RRAM reservoir model**
- 도구: ChatGPT
- 용도: RRAM의 비선형 컨덕턴스 변화와 단기 기억 특성을 모사하는 코드 작성 보조
- 주요 프롬프트: RRAM 리뷰 논문을 참고하여 4비트 펄스에 따른 컨덕턴스 변화를 구현하고, 256차원 Reservoir State를 생성하는 Python 코드 작성 요청
- 검증: 4비트 입력 패턴에 따른 컨덕턴스 변화 시각화 및 Reservoir State 배열 크기(256,) 확인

**F5. Readout training**
- 도구: ChatGPT
- 용도: 256차원 Reservoir State를 이용한 선형 분류기 구현 및 학습·검증 코드 작성 보조
- 주요 프롬프트: 기존 F1~F3 코드를 연결하여 256차원 Reservoir State를 입력받고, 8강 CNN의 학습 구조를 활용하는 선형 Readout 구현 요청
- 검증: 임의 입력 데이터에서 학습 및 예측 기능과 출력 크기 확인. 실제 FSDD 데이터 학습 결과는 추가 확인 예정

## 7. 참고자료

- Software Lab 강의자료 (1-1강, 1-2강, 4강, 8강)
- 머신러닝개론 강의자료 : 신경망의 학습 과정, 손실 함수 및 최적화 알고리즘 참고
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Du et al., *Reservoir computing using dynamic memristors for temporal information processing*, Nature Communications (2017)
- Ielmini, D., & Pedretti, G. (2025). *Resistive Switching Random-Access Memory (RRAM): Applications and Requirements for Memory and Computing.* Chemical Reviews, 125(12), 5584–5625. https://doi.org/10.1021/acs.chemrev.4c00845 (Section 5.3, Figure 26 및 식 (11)~(15): 상태 의존적 비선형 컨덕턴스 업데이트 특성 참고)