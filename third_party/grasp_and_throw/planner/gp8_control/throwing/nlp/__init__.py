"""[throwing/nlp] 던지기 NLP 플래너 (Ryu et al., "Robotic Throwing with Unmeasured Grasp Geometry and
Uncertain Release Timing") — CasADi/IPOPT B-spline NLP, 릴리즈 윈도우 목적함수, 스핀 페널티, warm-start DB.

throw_nlp.py      NLP 본체 (solve_throw_nlp; 종점 자세 p_f 경계조건 포함)
throwing.py       GP8 FK/자코비안·발사점·한계·게이트 상수 (planner 프레임)
gp8_interp.py     MotoROS 3차 Hermite 보간 모형 (시뮬 정합용; 런타임 필수 아님)
planner.py        로봇용 traj_fn 어댑터: 시작점 규약, warm DB(v2 축적 포맷), nlp_traj_fn, p_f
build_warm_db.py  warm DB 빌더 (cold multistart 144쌍 → warm_db_thr.pkl)
warm_db_thr.pkl   warm DB (config 여러 벌; 런타임은 현재 공식화와 일치하는 config 만 사용)

순수 numpy/casadi — ROS 무관. 스킬은 gp8_control.throwing.nlp_throw_skill.NlpThrowSkill.
"""
