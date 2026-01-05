import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import warnings
from collections import deque

# Pygame 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

class UltraProReinsuranceEnv(gym.Env):
    """
    [금융 전문가용 기후 리스크-자산 통합 관리 시뮬레이션]
    - 자산(Asset): 금리 연동형 주식 수익률 모델 (Fed Model)
    - 부채(Liability): 복합 포아송-파레토 프로세스 (Tail Risk)
    - 자본(Capital): Solvency II 기반 지급여력 관리
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 1. Action Space: [안전할증률, 주식투자비중, 헤징비중, CatBond비중]
        # 보험료 결정부터 자본 시장 도구 활용까지 4차원 의사결정 수행
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # 2. Observation Space: [자본금, 기후위험, 예보, 시장수익률, 금리]
        # 에이전트가 거시 경제와 물리적 리스크를 동시에 관측
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # 금융 수리 상수 설정
        self.INIT_CAPITAL = 2000.0    # 초기 자본금 2,000억 원
        self.SCR_LIMIT = 1.5          # Solvency II 권고 지급여력비율 (150%)
        self.PARETO_ALPHA = 2.5       # 사고 심도 파레토 지수 (Tail Risk의 두께 결정)
        self.BASE_INTEREST = 0.03     # 기준 금리 3%
        self.FIXED_OPEX = 60.0        # 분기별 고정 운영 비용
        
        # 리스크 지표(CVaR) 계산용 큐
        self.loss_history = deque(maxlen=100)
        
        # 시각화 관련 변수
        self.window = None
        self.clock = None
        self.font = None
        self.mode_text = "초기화 상태"
        
        self.reset()

    def _calculate_cvar(self, alpha=0.95):
        """[금융공학] CVaR (Conditional Value at Risk): 최악의 상황에서의 평균 손실 기댓값"""
        if len(self.loss_history) < 20: return 0
        losses = np.array(self.loss_history)
        var = np.percentile(losses, alpha * 100)
        cvar = losses[losses >= var].mean() if any(losses >= var) else var
        return cvar

    def _get_forecast(self):
        """불완전 정보 하의 베이지안 업데이트용 기상 예보 신호 생성"""
        real_future = 1 if np.random.random() < self.climate_risk else 0
        return float(real_future) if np.random.random() < 0.6 else float(1 - real_future)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.INIT_CAPITAL
        self.climate_risk = 0.05      # 초기 재난 발생 확률 (lambda)
        self.interest_rate = self.BASE_INTEREST
        self.time_step = 0
        self.last_loss = 0
        self.last_market_ret = 0.05
        self.last_disaster = False
        self.last_action = [0.2, 0.4, 0.1, 0.1]
        self.loss_history.clear()
        self.forecast = self._get_forecast()
        
        state = np.array([self.capital, self.climate_risk, self.forecast, self.last_market_ret, self.interest_rate], dtype=np.float32)
        return state, {}

    def step(self, action):
        if self.render_mode == "human":
            self._handle_events()

        self.last_action = action
        premium_rate, stock_ratio, hedge_ratio, cat_bond_ratio = action
        prev_capital = self.capital

        # 1. 거시 경제 동학 (Macro Dynamics)
        # 금리 변동 모델링 및 주식 수익률과의 역상관 관계 (Fed Model 적용)
        self.interest_rate += np.random.normal(0, 0.002)
        self.interest_rate = np.clip(self.interest_rate, 0.01, 0.08)
        interest_impact = -2.0 * (self.interest_rate - self.BASE_INTEREST)
        market_crash = np.random.random() < 0.12 # 블랙 스완 발생 확률
        stock_ret = np.random.normal(-0.25, 0.30) if market_crash else 0.07 + interest_impact + np.random.normal(0, 0.12)
        self.last_market_ret = stock_ret

        # 2. 보험 수리 동학 (Insurance Dynamics)
        # 기후 위기 가속화에 따른 손실 빈도 및 심도 증가 모델링
        self.climate_risk += 0.002
        num_claims = np.random.poisson(self.climate_risk)
        current_loss = np.sum((np.random.pareto(self.PARETO_ALPHA, num_claims) + 1) * 450) if num_claims > 0 else 0
        self.last_disaster = True if current_loss > 0 else False
        self.last_loss = current_loss
        self.loss_history.append(current_loss)

        # 3. 금융 공학 전략 (Financial Engineering)
        # 공매도 효과(Hedge) 및 대재해 채권(Cat Bond)을 통한 리스크 전이
        hedge_cost = hedge_ratio * 0.015
        hedge_profit = hedge_ratio * (-stock_ret) - hedge_cost
        cat_bond_payout = cat_bond_ratio * 800 if current_loss > 1200 else 0
        cat_bond_coupon = cat_bond_ratio * (self.interest_rate + 0.05) * 100

        # 4. ALM 및 유동성 정산
        # 자산 부채 통합 관리 및 자산 강제 매각 페널티(Fire Sale) 반영
        fire_sale_penalty = current_loss * 0.15 if current_loss > self.capital * 0.4 else 0
        premium_income = (self.climate_risk * 900) * (1 + premium_rate) * np.exp(-2.2 * premium_rate)
        
        bond_ratio = max(0, 1 - stock_ratio - hedge_ratio)
        inv_profit = (self.capital * stock_ratio * stock_ret) + \
                     (self.capital * hedge_profit) + \
                     (self.capital * bond_ratio * self.interest_rate)

        # 회계적 자본금 업데이트
        self.capital = self.capital + premium_income + inv_profit + cat_bond_payout - \
                       current_loss - cat_bond_coupon - fire_sale_penalty - self.FIXED_OPEX

        # 5. 리스크 조정 보상 함수 (Reward Function)
        cvar_95 = self._calculate_cvar()
        solvency_ratio = self.capital / (self.INIT_CAPITAL * 0.5 + 1e-6)
        reward = ((self.capital - prev_capital) * 0.1) - (cvar_95 * 0.07)
        
        # 지급여력비율(SCR) 위반 및 파산에 대한 강력한 규제 페널티
        if solvency_ratio < self.SCR_LIMIT: reward -= 700 
        if self.capital <= 0: reward -= 2500

        self.time_step += 1
        self.forecast = self._get_forecast()
        terminated = self.capital <= 0
        truncated = self.time_step >= 360 # 30년 장기 경영 목표
        
        state = np.array([self.capital, self.climate_risk, self.forecast, stock_ret, self.interest_rate], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, truncated, {"loss": self.last_loss, "cvar": cvar_95}

    def _handle_events(self):
        if not pygame.display.get_init(): return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

    def render(self):
        if self.render_mode != "human": return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((850, 650))
            pygame.display.set_caption("지급여력 강화단: AI 리스크 관제 시스템 v4.2")
            try:
                self.font = pygame.font.SysFont("malgungothic", 18)
            except:
                self.font = pygame.font.SysFont("arial", 18)
            self.clock = pygame.time.Clock()

        if not pygame.display.get_init(): return

        canvas = pygame.Surface((850, 650))
        canvas.fill((15, 15, 20))

        # 상단 헤더: 현재 경영 모드 및 거시 지표
        color_main = (0, 255, 180) if "AI" in self.mode_text else (200, 200, 200)
        canvas.blit(self.font.render(f"운영 모드: {self.mode_text}", True, color_main), (50, 10))

        def draw_stat(label, val, x, y, col=(255, 255, 255)):
            lbl = self.font.render(label, True, (160, 160, 160))
            v = self.font.render(val, True, col)
            canvas.blit(lbl, (x, y))
            canvas.blit(v, (x, y + 25))

        draw_stat("현재 자본금", f"{self.capital:,.1f} B", 50, 50, color_main)
        draw_stat("시장 수익률", f"{self.last_market_ret:+.2%}", 300, 50, (255, 200, 0))
        draw_stat("기준 금리", f"{self.interest_rate:.2%}", 500, 50, (0, 150, 255))
        draw_stat("기후 리스크 (λ)", f"{self.climate_risk:.4f}", 700, 50, (255, 80, 80))

        # 중앙: 자산 배분 현황 (Action Visualization)
        def draw_bar(label, val, max_val, x, y, col):
            pygame.draw.rect(canvas, (40, 40, 45), (x, y, 350, 20))
            w = int(350 * min(1.0, val / max_val))
            if w > 0: pygame.draw.rect(canvas, col, (x, y, w, 20))
            canvas.blit(self.font.render(label, True, (200, 200, 200)), (x, y - 22))

        draw_bar("자본 건전성 현황", self.capital, 5000, 50, 140, (0, 180, 120))
        draw_bar(f"주식 투자 비중: {self.last_action[1]:.1%}", self.last_action[1], 1.0, 50, 210, (0, 100, 250))
        draw_bar(f"헤징 및 공매도 수준: {self.last_action[2]:.1%}", self.last_action[2], 1.0, 50, 280, (250, 120, 0))
        draw_bar(f"대재해 채권 활용도: {self.last_action[3]:.1%}", self.last_action[3], 1.0, 50, 350, (180, 50, 255))

        # 우측: 리스크 분석 리포트
        pygame.draw.rect(canvas, (25, 25, 35), (450, 120, 350, 300))
        report_title = self.font.render("경영 리스크 종합 리포트", True, (255, 255, 255))
        canvas.blit(report_title, (470, 140))
        
        cvar_val = self._calculate_cvar()
        reports = [
            f"운영 기간: Year {self.time_step//12 + 1} ({self.time_step % 12 + 1}개월)",
            f"95% CVaR (꼬리 위험): {cvar_val:,.1f} B",
            f"지급여력비율(Solvency): {self.capital / 1000.0:.1%}",
            f"기상 예보 신호: {'재난 경보' if self.forecast == 1 else '안정'}",
            f"직전 사고 손실액: {self.last_loss:,.1f} B"
        ]
        for i, text in enumerate(reports):
            col = (255, 100, 100) if (i==4 and self.last_loss > 500) else (220, 220, 220)
            canvas.blit(self.font.render(text, True, col), (470, 190 + i * 40))

        # 하단: 재난 발생 경고 시스템
        if self.last_disaster:
            pygame.draw.rect(canvas, (150, 0, 0), (0, 0, 850, 650), 10)
            alert_msg = f"!!! 대규모 기후 재난 발생: -{self.last_loss:,.1f} B 원 !!!"
            canvas.blit(self.font.render(alert_msg, True, (255, 255, 255)), (220, 570))

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None