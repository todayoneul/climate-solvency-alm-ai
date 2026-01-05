import time
from stable_baselines3 import PPO
from environment import UltraProReinsuranceEnv

def execute_comparison():
    # 시각화 모드로 환경 초기화
    env = UltraProReinsuranceEnv(render_mode="human")
    
    # 학습된 최적 모델 로드
    try:
        model = PPO.load("models/PPO_ClimateAI/final_climate_ai_model")
        print("[SYSTEM] 지급여력 강화단 AI 모델을 성공적으로 로드하였습니다.")
    except Exception as e:
        print(f"[WARNING] 모델 로드 실패: {e}. 무작위 전략으로 대체 시뮬레이션합니다.")
        model = None

    # 시나리오 1: 학습 전 (무작위 경영 전략)
    env.unwrapped.mode_text = "학습 전: 무작위 경영 (Baseline)"
    obs, _ = env.reset()
    print("[RUN] 시나리오 1: 기준 모델 시뮬레이션 시작")
    for _ in range(360):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print(f"[-] 기준 모델 파산 또는 종료 (Month: {env.unwrapped.time_step})")
            time.sleep(2)
            break

    # 시나리오 2: 학습 후 (AI 리스크 관리 전략)
    if model:
        env.unwrapped.mode_text = "학습 후: 강화도"
        obs, _ = env.reset()
        print("[RUN] 시나리오 2: AI 모델 리스크 관리 시연 시작")
        for _ in range(360):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print(f"[+] AI 모델 시뮬레이션 완료 (Month: {env.unwrapped.time_step})")
                time.sleep(2)
                break
    
    env.close()

if __name__ == "__main__":
    execute_comparison()