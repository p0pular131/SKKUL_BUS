cd carla_garage/carla
# 시뮬 실행
./CarlaUE4.sh
# 저사양 실행
# ./CarlaUE4.sh -quality-level=Low

# 가상환경 활성화 
conda activate garage_2
# 경로설정
export CARLA_ROOT=/home/heven/carla_garage/carla
export WORK_DIR=/home/heven/SKKUL_BUS
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:/home/heven/SKKUL_BUS/scenario_runner
export SAVE_PATH=/home/heven/SKKUL_BUS/results
export DEBUG_CHALLENGE=1
cd SKKUL_BUS/leaderboard/leaderboard
# 모델 검증 실행
python3 leaderboard_evaluator_local.py --agent-config /home/heven/carla_garage/pretrained_models/all_towns --agent /home/heven/SKKUL_BUS/team_code/sensor_agent_skku.py --routes /home/heven/SKKUL_BUS/leaderboard/data/debug.xml --checkpoint $SAVE_PATH/result.json


# No graphic 
# System : Game = 47.0 : 10.0

