# KR-BERT 활용한 한국어 대화 텍스트 내의 폭력성 유무 분류

이화여자대학교 컴퓨터공학과 2271059 좌연주

## 요약


## 개요
방송통신위원회와 한국지능정보사회진흥원의 '2024년도 사이버폭력 실태조사 결과'에 따르면 우리나라 청소년의 42.7%, 성인 13.5%가 사이버폭력을 경험했다고 응답했다. 사이버폭력 경험은 가해와 피해 경험을 모두 포함한다. 2023년 대비 청소년은 1.9%포인트(p), 성인은 5.5%포인트 증가한 수치다. 방송통신위원회는 성별·장애·종교 등이 다르다는 이유로 특정 개인이나 집단에 편견과 차별을 표현하는 '디지털 혐오'나 불법 영상물이나 몰래카메라 등 '디지털 성범죄'와 같은 부정적 콘텐츠에 노출되는 정도가 증가했기 때문이라고 분석했다. 특히 청소년은 성인에 비해 이유가 없거나, 재미·장난으로도 사이버폭력을 행하고 있어 사이버폭력의 심각성을 인지하지 못하는 것으로 나타났다. 이와 같은 배경으로, 대화 텍스트 내 폭력성 유무를 정확하고 신속하게 판단할 수 있는 자동화된 시스템 개발이 사회적으로 매우 중요해졌다.


## 목표
한국어 대화 텍스트에서 문장 단위의 폭력성 유무를 자동으로 분류하는 NLP 모델을 검증하고 LoRA (Low-Rank Adaptation) 기법을 적용하여 모델의 파인튜닝 효율성을 극대화하고 성능을 분석한다.


## 통계
실제 상황에서 흔히 접하는 7가지 주요 범주로 분류체계를 정의하고 이 연구에서는 트레이닝 세트 365,500개와 테스트 세트 44,998개로 분할했다. 트레이닝 세트와 테스트 세트는 문장, 폭력성 유무, 폭력성 유형으로 구성되어 있다.
<img width="1379" height="750" alt="image" src="https://github.com/user-attachments/assets/0c8168a3-1cf5-4547-b1c6-48225827490a" />


## 사용 데이터
- 텍스트 윤리검증
  - 출처 : [Ai-Hub 텍스트윤리검증](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%ED%85%8D%EC%8A%A4%ED%8A%B8%EC%9C%A4%EB%A6%AC%EA%B2%80%EC%A6%9D&aihubDataSe=data&dataSetSn=558)
  - 데이터 수 : 약 360,000 개
  - 용도 : 모델 학습용

- LLM 학습용 데이터 내 유해표현 검출 AI모델 학습용 데이터
  - 출처 : [Ai-Hub 외부기관데이터 LLM 학습용 데이터 내 유해표현 검출 AI모델 학습용 데이터](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=118&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8F%AD%EB%A0%A5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=extrldata&dataSetSn=71833)
  - 데이터 수 : 약 200,000개
  - 용도 : 모델학습용(fine-tuning)
 
 
## 실험결과
- Train Data Size에 따른 F1 Score 변화
  - 트레이닝 데이터개수를 10,000개, 50,000개, 100,000개, 200,000개, 365,500개로 늘리며 트레이닝 후 테스트를 진행하고 유형별로 분석했을 때 HATE, DISCRIMINATION, VIOLENCE는 더이상 f1-score가 향상되지 않는 것을 확인할 수 있었다. 그 외에 CENSURE, ABUSE, CRIME, SEXUAL은 f1-score가 지속적으로 향상되는 것을 확인할 수 있다. 따라서 HATE, DISCRIMINATION, VIOLENCE는 추가적인 트레이닝이 불필요하고 CENSURE, ABUSE, CRIME, SEXUAL은 1,000,000개 이상의 추가적인 트레이닝이 필요할 것으로 추측된다.
    <img width="1370" height="745" alt="image" src="https://github.com/user-attachments/assets/9ef4e822-4675-427e-9354-f1897c58c314" />

-fine-tuning 후 결과 비교




## 고찰


