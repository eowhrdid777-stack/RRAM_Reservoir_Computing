# RRAM 기반 Reservoir Computing(RC)을 이용한 Free Spoken Digit Dataset(FSDD) 음성 숫자 인식 시스템

FSDD 음성 데이터셋을 활용하여 RRAM의 비선형 응답과 단기 기억 특성을 모사하는 Reservoir Computing 기반 음성 숫자 인식 시스템을 구현한다.

## 1. 강의 연계

1-2강 wav: wav header file 및 음성 data 읽기


## 2. 개발 환경

- Python 3.14.3 (Visual Studio Code)
- path, numpy, struct
- Dataset: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)

## 3. 프로그램 구성

data_loader.py: WAV 헤더 분석 및 FSDD 데이터 분할 

## 4. 설치 및 실행


## 5. 실험 결과


## 6. AI 활용 내역

**F1. FSDD data load**
- 도구: ChatGPT
- 용도: 1-2강 WAV 실습을 기반으로 PCM 헤더 파싱 및 데이터 분할 코드 작성 보조
- 주요 프롬프트: 강의 Exercise 1을 활용하여 WAV 헤더를 직접 읽고 FSDD 데이터셋을 분할하는 Python 코드 구현 요청
- 검증: WAV 헤더 정보 및 샘플 개수, Train/Validation/Test 분할 결과 확인
- 직접 수정 사항: 해당 사항이 있으면 추가

이후 기능별 AI 활용 및 검증 내용을 추가한다.

## 7. 참고자료

- Software Lab 강의자료 (1-2강, )
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Du et al., *Reservoir computing using dynamic memristors for temporal information processing*, Nature Communications (2017)
