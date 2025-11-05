# 목표: 기존 폭력성 확인 모델을 fine tuning 후 성능 비교

# 사용 데이터
- 텍스트 윤리검증
  - 출처 : Ai-Hub 텍스트윤리검증
  - 데이터 수 : 약 360,000 개
  - 용도 : 모델 학습용

- 한국어 혐오표현 데이터셋
  - 출처 : Smilegate(github)
  - 데이터 수 : 약 15,000 개
  - 용도 : 모델 학습용

- 유튜브 댓글 크롤링
  - 용도 : 폭력성 높은 채팅 데이터 생성용(test용)

- LLM 학습용 데이터 내 유해표현 검출 AI모델 학습용 데이터
  - 모델학습용(fine-tuning)
  https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=118&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8F%AD%EB%A0%A5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=extrldata&dataSetSn=71833
