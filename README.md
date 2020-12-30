# 국립국어원 세종 말뭉치 활용 품사 분석

[국립국어원 세종 말뭉치](https://ithub.korean.go.kr/user/guide/corpus/guide1.do)를 활용하여 품사 분석을 수행하는 프로젝트 레포입니다.

## 데이터셋 구축

1. [국립국어원 세종 말뭉치를 다운로드하는 스크립트](https://github.com/coolengineer/sejong-corpus)를 이용, 말뭉치를 raw data로 다운로드
1. [세종 말뭉치 정제를 위한 utils](https://github.com/lovit/sejong_corpus_cleaner)를 이용, 품사 분석 관련 데이터 전처리
    * `1.`에서 다운 받은 데이터 중 `corpus` 폴더 내의 `?CT*.txt` 형태의 파일은 구어 말뭉치, `BT*.txt` 파일은 문어 말뭉치이다. 각각을 정제 유틸에서 지정한 폴더에 넣고 스크립트를 실행하여 전처리된 데이터를 얻는다. 
    * 전처리된 데이터는 다음과 같은 파이썬 리스트 형태로 저장된다.
    ```
    [
     '프랑스의\t프랑스/Noun + 의/Josa\n',
     '세계적\t세계적/Noun\n',
     '인\t이/Adjective + ㄴ/Eomi\n',
     '의상\t의상/Noun\n',
     '디자이너\t디자이너/Noun\n',
     '엠마누엘\t엠마누엘/Noun\n',
     '웅가로가\t웅가로/Noun + 가/Josa\n',
     '실내\t실내/Noun\n',
     '장식용\t장식용/Noun\n',
     '직물\t직물/Noun\n',
     '디자이너로\t디자이너/Noun + 로/Josa\n',
     '나섰다\t나서/Verb + 었다/Eomi\n',
     ...
     ]    
    ```
    
## 품사 분석 모델 구현

### 1. Hidden Markov Model 기반 품사 판별 모델

다음 포스팅을 참고하여 HMM(Hidden Markov Model) 기반 품사 판별 모델을 구축(`HMM.py`)한다.

* [Hidden Markov Model (HMM) 기반 품사 판별기의 원리와 문제점](https://lovit.github.io/nlp/2018/09/11/hmm_based_tagger/)
* [Hidden Markov Model 기반 품사 판별기의 decode 함수](https://lovit.github.io/nlp/2018/10/23/hmm_based_tagger_tag/)


### 2. Conditional Random Field 기반 품사 판별 모델

다음 포스팅을 참고하여 CRF(Conditional Random Field) 기반 품사 판별 모델을 구축 예정이다.

* [From Softmax Regression to Conditional Random Field for Sequential Labeling](https://lovit.github.io/nlp/machine%20learning/2018/04/24/crf/)
* [Conditional Random Field (CRF) 기반 품사 판별기의 원리와 HMM 기반 품사 판별기와의 차이점](https://lovit.github.io/nlp/2018/09/13/crf_based_tagger/)

