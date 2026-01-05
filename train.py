import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import UltraProReinsuranceEnv

def train_agent():
    # 1. 저장 경로 및 로그 경로 설정
    models_dir = "models/PPO_ClimateAI"
    log_dir = "tensorboard_logs"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 학습 전용 환경 생성 (시각화 비활성화를 통한 학습 속도 최적화)
    env = UltraProReinsuranceEnv(render_mode=None)

    # 3. PPO 모델 정의 및 하이퍼파라미터 설정
    # ent_coef: 탐색(Exploration) 가중치를 부여하여 로컬 옵티멈 방지
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        ent_coef=0.03,            # 전략의 다양성 확보
        learning_rate=0.0003,     # 학습률 미세 조정
        n_steps=2048,             # 업데이트당 샘플 수
        batch_size=64,            # 배치 크기
        gamma=0.99,               # 미래 보상 할인율
        device="auto"
    )

    # 4. 중간 저장 콜백 (20,000 스텝마다 모델 스냅샷 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=models_dir,
        name_prefix="climate_expert_model"
    )

    # 5. 대규모 시뮬레이션 학습 실행 (500,000 타임스텝)
    TOTAL_STEPS = 500000 
    print(f"[SYSTEM] 지급여력 강화단 AI의 학습을 개시합니다. 목표: {TOTAL_STEPS} steps")
    
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint_callback,
        tb_log_name="Reinsurance_Expert_V4",
        reset_num_timesteps=False
    )

    # 6. 최종 고도화 모델 저장
    final_path = f"{models_dir}/final_climate_ai_model"
    model.save(final_path)
    print(f"[SUCCESS] 최적화된 경영 모델이 저장되었습니다: {final_path}")

if __name__ == "__main__":
    train_agent()