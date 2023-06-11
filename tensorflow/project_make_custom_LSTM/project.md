#project_code_first
1. data preprocessing
--> csv형식의 데이터를 받아서 한자, 이모티콘 등의 데이터들을 제거한다. 
    또한 NaN 값에 해당되는 데이터들을 삭제하였다. 이후 인덱스를 재배치하여 
    데이터 전처리 과정 이후에도 일관성이 깨지지 않도록 조절하였다.
2. data processing for using LSTM
--> LSTM모델에 사용할 수 있도록  Tokenizer 라이브러리를 활용하여  sequence 형식의
    데이터로 재가공하였다. 
3. make LSTM based model
--> 시퀀셜 클래스로 Embedding, LSTM, dense으로 이루어진  3계층 모델을 만들었다.
    영화 리뷰에 대한 평가값을 예측하는 문제이기 때문에 회귀형 문제라고 이해하였다.
    따라서 최종 레이어의 활성화 함수로 linear를 활용하였다.
4. model compile & train
--> 생성한 시퀀셜 모델을 컴파일하였고 이에 맞추어 학습시켰다.
5. model predict
--> 학습시킨 모델을 평가하기 위하여 평가데이터로 예측을 수행하였다.
6. visualize
--> 예측 결과를 matplotlib 라이브러리를 활용하여 시각화하였다. 

#project_code_second
1. data preprocessing
--> csv형식의 데이터를 받아서 한자, 이모티콘 등의 데이터들을 제거한다. 
    또한 NaN 값에 해당되는 데이터들을 삭제하였다. 이후 인덱스를 재배치하여 
    데이터 전처리 과정 이후에도 일관성이 깨지지 않도록 조절하였다.
2. data processing for 감정분석
--> 감정분석을 위하여 countvectorizer 라이브러리를 활용하여 벡터화하였다. 
    (예측모델에 맞게 데이터 정제)
    countvertorizer는  tokenizer에 비해 메모리 사용량이 적다고 알려져 있다.
3. data training
--> logsticregression 모델을 활용하였고 이를 학습시켰다.
4. data prediction
--> 테스트 데이터로 완성된 모델에 대해 예측을 수행하였다. 
    이후 conf 값으로 데이터들을 정렬시켰다. 
5. 감정분석 - 빈도수가 많았던 단어들
--> 예측 결과를 기반으로 빈도수가 많았던 단어들을 추출하였다.
    각 단어 데이터들을 취합하여 누적시켜 가장 빈도수가 많은 단어들을 추출하였다.
6. 감정분석 - 평가에 좋은 영향을 주었던 단어들
--> 예측 결과를 기반으로 평가에 좋은 영향을 주었던 단어들을 추출하였다.
    conf 값이 음수였던 (평가에 긍정적인 도움이 되었던) 상위 10개 데이터들을 추출하였다.
7. 감정분석 - 평가에 안좋은 영향을 주었던 단어들
--> 예측 결과를 기반으로 평가에 안좋은 영향을 주었던 단어들을 추출하였다.
    conf 값이 양수였던 (평가에 부정적인 영향을 주었던) 상위 10개 데이터들을 추출하였다.
8. 시각화
--> 감정분석의 결과들을 matplotlib 라이브러리를 활용하여 시각화하였다.

++ 감정분석의 결과
the, and, to, is , of, was, in, food, it ,for, with, we, good, this, place ,you 등의 단어가
가장 빈도수가 많았던 것으로 분석되었다.
excellent, delicious, tasty, amazing, great, nice, best, loved, must 등의 단어가 
가장 평가에 좋은 영향을 미친 단어로 분석되었다.
worst, terrible, pathetic, horrible, bad, poor, never, tasteless, avoid, rude 등의 단어가
가장 평가에 안좋은 영향을 미친 단어로 분석되었다.
