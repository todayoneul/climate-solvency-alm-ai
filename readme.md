# 강화도: 기후 리스크 대응형 재보험 ALM 모델

본 프로젝트는 기후 위기로 인한 비정형적 손실(Tail Risk)에 대응하기 위해 심층 강화학습(PPO)을 활용한 자산-부채 통합 관리(ALM) 시스템을 제안합니다.

## 핵심 모델링
1. **부채 모델**: Compound Poisson-Pareto Process를 통한 거대 재난 구현
2. **자산 모델**: Fed Model 기반 금리-주식 역상관 자산 동학 적용
3. **리스크 관리**: $CVaR_{95\%}$ 페널티 및 Solvency II 규제 준수 로직

## 실행 방법
1. 환경 설치: `pip install -r requirements.txt`
2. 모델 학습: `python src/train.py`
3. 성과 시연: `python src/demo.py`