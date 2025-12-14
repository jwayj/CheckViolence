# KR-BERT 활용한 한국어 대화 텍스트 내의 폭력성 유무 분류

이화여자대학교 컴퓨터공학과 2271059 좌연주

## 목표: 기존 폭력성 확인 모델을 fine tuning 후 성능 비교

## 사용 데이터
- 텍스트 윤리검증
  - 출처 : [Ai-Hub 텍스트윤리검증](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%ED%85%8D%EC%8A%A4%ED%8A%B8%EC%9C%A4%EB%A6%AC%EA%B2%80%EC%A6%9D&aihubDataSe=data&dataSetSn=558)
  - 데이터 수 : 약 360,000 개
  - 용도 : 모델 학습용

- LLM 학습용 데이터 내 유해표현 검출 AI모델 학습용 데이터
  - 출처 : [Ai-Hub 외부기관데이터 LLM 학습용 데이터 내 유해표현 검출 AI모델 학습용 데이터](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=118&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8F%AD%EB%A0%A5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=extrldata&dataSetSn=71833)
  - 데이터 수 : 약 200,000개
  - 용도 : 모델학습용(fine-tuning)

