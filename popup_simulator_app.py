import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, FancyArrow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 페이지 설정
st.set_page_config(
    page_title="팝업 시스템 시뮬레이터",
    page_icon="🚀",
    layout="wide"
)

# 상수 정의
g = 9.81  # 중력가속도 (m/s^2)
rho_air = 1.225  # 공기 밀도 (kg/m^3)

def psi_to_pa(psi):
    """PSI를 Pascal로 변환"""
    return psi * 6894.76

def calculate_popup_system(diameter_mm, pressure_psi, stroke_mm, mass_accel_kg, mass_inertial_kg,
                          num_cylinders, energy_loss_percent, mu_friction, projectile_area_m2, Cd):
    """
    팝업 시스템 전체 계산

    Parameters:
    - mass_accel_kg: 가속 구간 질량 (발사체 + 구동부)
    - mass_inertial_kg: 관성 구간 질량 (발사체만)
    - mu_friction: 마찰계수
    - projectile_area_m2: 발사체 단면적 (m²)
    - Cd: 항력계수

    Returns:
    - 딕셔너리: 모든 계산 결과
    """
    # 단위 변환
    diameter = diameter_mm / 1000  # m
    pressure = psi_to_pa(pressure_psi)  # Pa
    stroke = stroke_mm / 1000  # m

    # 시스템 효율 (에너지 손실 반영)
    eta_system = 1.0 - (energy_loss_percent / 100.0)

    # 피스톤 면적
    area = np.pi * (diameter / 2) ** 2  # m^2

    # 총 힘 계산 (병렬 배치)
    F_pressure = num_cylinders * pressure * area * eta_system  # N

    # 마찰력 계산 (중량 기반)
    F_friction = mu_friction * mass_accel_kg * g  # N

    # 중력
    F_gravity = mass_accel_kg * g  # N

    # 순 힘
    F_net = F_pressure - F_friction - F_gravity  # N

    if F_net <= 0:
        return None  # 힘이 충분하지 않음

    # 가속도
    a = F_net / mass_accel_kg  # m/s^2

    # 가속 구간 종료 시 속도
    v_exit = np.sqrt(2 * a * stroke)  # m/s

    # 가속 구간 높이
    accel_height = stroke  # m

    # 시간에 따른 가속 구간 데이터
    t_accel = np.sqrt(2 * stroke / a)  # 가속 시간
    time_accel = np.linspace(0, t_accel, 100)
    height_accel = 0.5 * a * time_accel ** 2
    velocity_accel = a * time_accel
    force_accel = np.full_like(time_accel, F_net)

    # 관성 구간 시뮬레이션
    dt = 0.001
    v = v_exit
    h = 0
    t = 0

    time_inertial = [0]
    height_inertial = [0]
    velocity_inertial = [v_exit]

    # 공기 저항 면적 (발사체 단면적 사용)
    A_drag = projectile_area_m2

    while v > 0:
        # 공기 저항력
        F_drag = 0.5 * rho_air * Cd * A_drag * v ** 2

        # 가속도 (음수)
        a_inertial = -(g + F_drag / mass_inertial_kg) if mass_inertial_kg > 0 else -g

        # 속도 및 높이 업데이트
        v += a_inertial * dt
        if v > 0:
            h += v * dt
            t += dt

            time_inertial.append(t)
            height_inertial.append(h)
            velocity_inertial.append(v)

    inertial_height = h
    total_height = accel_height + inertial_height

    # 관성 구간에서의 힘 (공기 저항 + 중력)
    force_inertial = [-mass_inertial_kg * g - 0.5 * rho_air * Cd * A_drag * v**2
                      for v in velocity_inertial] if mass_inertial_kg > 0 else [-0.5 * rho_air * Cd * A_drag * v**2 for v in velocity_inertial]

    return {
        'total_height': total_height,
        'accel_height': accel_height,
        'inertial_height': inertial_height,
        'exit_velocity': v_exit,
        'max_force': F_net,
        'time_accel': time_accel,
        'height_accel': height_accel,
        'velocity_accel': velocity_accel,
        'force_accel': force_accel,
        'time_inertial': np.array(time_inertial) + t_accel,
        'height_inertial': np.array(height_inertial) + accel_height,
        'velocity_inertial': velocity_inertial,
        'force_inertial': force_inertial,
        'total_time': t_accel + t,
        'acceleration': a
    }

def simulate_projectile(v0, angle_deg, mass_kg, projectile_area_m2, Cd):
    """포물선 운동 시뮬레이션"""
    angle_rad = np.radians(angle_deg)

    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x = 0
    y = 0

    dt = 0.001

    trajectory = {
        't': [0],
        'x': [0],
        'y': [0],
        'vx': [vx],
        'vy': [vy]
    }

    # 발사체 단면적 사용
    A_drag = projectile_area_m2

    # 수평 발사의 경우 최대 반복 제한
    max_iterations = 100000
    iteration = 0

    while y >= -0.01 and iteration < max_iterations:
        v = np.sqrt(vx**2 + vy**2)

        if v > 0:
            F_drag_x = -0.5 * rho_air * Cd * A_drag * v * vx
            F_drag_y = -0.5 * rho_air * Cd * A_drag * v * vy
        else:
            F_drag_x = 0
            F_drag_y = 0

        ax = F_drag_x / mass_kg
        ay = -g + F_drag_y / mass_kg

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        trajectory['t'].append(trajectory['t'][-1] + dt)
        trajectory['x'].append(x)
        trajectory['y'].append(y)
        trajectory['vx'].append(vx)
        trajectory['vy'].append(vy)

        iteration += 1

    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    return trajectory

# ============================================================================
# Streamlit UI
# ============================================================================

st.title("🚀 팝업 시스템 시뮬레이터")
st.markdown("---")

# 사이드바에 입력 변수 배치
st.sidebar.header("⚙️ 시스템 파라미터")

st.sidebar.subheader("1️⃣ 공압 시스템")
num_cylinders = st.sidebar.slider(
    "실린더 개수",
    min_value=1,
    max_value=6,
    value=2,
    step=1,
    help="병렬로 배치된 실린더 개수"
)

diameter_mm = st.sidebar.number_input(
    "직경 (mm)",
    min_value=50.0,
    max_value=500.0,
    value=188.0,
    step=10.0,
    help="피스톤 직경"
)

pressure_psi = st.sidebar.number_input(
    "압력 (psi)",
    min_value=10.0,
    max_value=300.0,
    value=150.0,
    step=10.0,
    help="작동 압력"
)

stroke_mm = st.sidebar.number_input(
    "스트로크 (mm)",
    min_value=10.0,
    max_value=500.0,
    value=129.6,
    step=10.0,
    help="피스톤 이동 거리"
)

st.sidebar.subheader("2️⃣ 물체 특성")
mass_accel_kg = st.sidebar.number_input(
    "가속 구간 질량 (kg)",
    min_value=0.0,
    max_value=500.0,
    value=22.0,
    step=1.0,
    help="발사체 + 구동부 전체 질량"
)

mass_inertial_kg = st.sidebar.number_input(
    "관성 구간 질량 (kg)",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    step=1.0,
    help="발사체만의 질량 (구동부 제외)"
)

projectile_area_m2 = st.sidebar.number_input(
    "발사체 단면적 (m²)",
    min_value=0.001,
    max_value=2.0,
    value=0.1,
    step=0.01,
    format="%.3f",
    help="공기 저항을 받는 발사체의 단면적 (예: 0.4m × 0.25m = 0.1m²)"
)

st.sidebar.subheader("3️⃣ 시스템 손실")
energy_loss = st.sidebar.slider(
    "에너지 손실 (%)",
    min_value=0,
    max_value=100,
    value=10,
    step=5,
    help="마찰, 누설 등으로 인한 에너지 손실"
)

mu_friction = st.sidebar.number_input(
    "마찰계수 (μ)",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01,
    help="실린더 및 가이드 레일의 마찰계수"
)

Cd = st.sidebar.number_input(
    "항력계수 (Cd)",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05,
    help="공기 저항 계수 (평평한 끝 원통: 0.8~1.0, 둥근 끝: 0.5, 유선형: 0.2)"
)

st.sidebar.subheader("4️⃣ 발사 각도")
launch_angle = st.sidebar.slider(
    "발사 각도 (°)",
    min_value=0,
    max_value=90,
    value=90,
    step=5,
    help="+Y축(연직)에서 +X축 방향으로의 각도. 0°=수평, 90°=수직"
)

st.sidebar.markdown("---")
calculate_button = st.sidebar.button("🚀 시뮬레이션 실행", type="primary")

# ============================================================================
# 계산 및 시각화
# ============================================================================

if calculate_button:
    with st.spinner('계산 중...'):
        # 팝업 시스템 계산
        result = calculate_popup_system(
            diameter_mm, pressure_psi, stroke_mm,
            mass_accel_kg, mass_inertial_kg, num_cylinders, energy_loss, mu_friction, projectile_area_m2, Cd
        )

        if result is None:
            st.error("⚠️ 힘이 충분하지 않습니다! 압력을 높이거나 질량을 줄여주세요.")
        else:
            # 포물선 운동 계산 (관성 구간 질량 사용)
            trajectory = simulate_projectile(
                result['exit_velocity'],
                launch_angle,
                mass_inertial_kg if mass_inertial_kg > 0 else mass_accel_kg,
                projectile_area_m2,
                Cd
            )

            # 결과 저장 (애니메이션용)
            st.session_state['result'] = result
            st.session_state['trajectory'] = trajectory
            st.session_state['params'] = {
                'num_cylinders': num_cylinders,
                'diameter_mm': diameter_mm,
                'pressure_psi': pressure_psi,
                'stroke_mm': stroke_mm,
                'mass_accel_kg': mass_accel_kg,
                'mass_inertial_kg': mass_inertial_kg,
                'projectile_area_m2': projectile_area_m2,
                'energy_loss': energy_loss,
                'mu_friction': mu_friction,
                'Cd': Cd,
                'launch_angle': launch_angle
            }

            # ================================================================
            # 포물선 운동 애니메이션
            # ================================================================
            st.header("🎬 팝업 시스템 시뮬레이터")

            col1, col2 = st.columns([2, 1])

            with col1:
                # 애니메이션 설정 (고정값)
                animation_speed = 1.0
                show_velocity = True
                show_trail = True

                # 플로틀리로 애니메이션 생성
                max_x = max(trajectory['x']) * 1.1 if max(trajectory['x']) > 0 else 1
                max_y = max(trajectory['y']) * 1.1

                # 프레임 생성
                num_frames = min(200, len(trajectory['x']))
                frame_indices = np.linspace(0, len(trajectory['x'])-1, num_frames, dtype=int)

                frames = []
                for idx in frame_indices:
                    frame_data = []

                    # 궤적 (선)
                    if show_trail:
                        frame_data.append(
                            go.Scatter(
                                x=trajectory['x'][:idx+1],
                                y=trajectory['y'][:idx+1],
                                mode='lines',
                                line=dict(color='blue', width=2),
                                name='궤적',
                                showlegend=False
                            )
                        )

                    # 현재 위치 (점)
                    frame_data.append(
                        go.Scatter(
                            x=[trajectory['x'][idx]],
                            y=[trajectory['y'][idx]],
                            mode='markers',
                            marker=dict(size=20, color='red', symbol='circle'),
                            name='발사체',
                            showlegend=False
                        )
                    )

                    # 속도 벡터
                    if show_velocity and idx > 0:
                        scale = 0.1
                        vx = trajectory['vx'][idx] * scale
                        vy = trajectory['vy'][idx] * scale

                        frame_data.append(
                            go.Scatter(
                                x=[trajectory['x'][idx], trajectory['x'][idx] + vx],
                                y=[trajectory['y'][idx], trajectory['y'][idx] + vy],
                                mode='lines',
                                line=dict(color='green', width=3),
                                name='속도',
                                showlegend=False
                            )
                        )

                    frames.append(go.Frame(data=frame_data, name=str(idx)))

                # 초기 프레임
                fig_anim = go.Figure(
                    data=[
                        go.Scatter(
                            x=[0],
                            y=[0],
                            mode='markers',
                            marker=dict(size=20, color='red'),
                            showlegend=False
                        )
                    ],
                    frames=frames
                )

                # 레이아웃
                fig_anim.update_layout(
                    xaxis=dict(range=[-0.5, max_x], title="수평 거리 (m)"),
                    yaxis=dict(range=[-0.2, max_y], title="수직 높이 (m)"),
                    title=f"발사 각도: {launch_angle}° (+Y축에서 +X축 방향, 0°=수평/90°=수직)",
                    height=500,
                    updatemenus=[
                        dict(
                            type="buttons",
                            buttons=[
                                dict(label="▶️ 재생",
                                     method="animate",
                                     args=[None, {"frame": {"duration": 50/animation_speed, "redraw": True},
                                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                                dict(label="⏸️ 정지",
                                     method="animate",
                                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                    "mode": "immediate",
                                                    "transition": {"duration": 0}}])
                            ],
                            direction="left",
                            pad={"r": 10, "t": 87},
                            showactive=False,
                            x=0.1,
                            xanchor="left",
                            y=0,
                            yanchor="top"
                        )
                    ]
                )

                # 지면 표시
                fig_anim.add_shape(
                    type="rect",
                    x0=-0.5, y0=-0.2, x1=max_x, y1=0,
                    fillcolor="brown",
                    opacity=0.3,
                    line=dict(width=0)
                )

                st.plotly_chart(fig_anim, use_container_width=True)

            with col2:
                st.subheader("입력 파라미터")
                st.json({
                    "실린더 개수": num_cylinders,
                    "직경 (mm)": diameter_mm,
                    "압력 (psi)": pressure_psi,
                    "스트로크 (mm)": stroke_mm,
                    "가속 구간 질량 (kg)": mass_accel_kg,
                    "관성 구간 질량 (kg)": mass_inertial_kg,
                    "발사체 단면적 (m²)": projectile_area_m2,
                    "에너지 손실 (%)": energy_loss,
                    "마찰계수 (μ)": mu_friction,
                    "항력계수 (Cd)": Cd,
                    "발사 각도 (°)": launch_angle
                })

                st.subheader("계산 결과")
                st.json({
                    "총 상승 높이 (m)": round(result['total_height'], 3),
                    "가속 구간 높이 (m)": round(result['accel_height'], 3),
                    "관성 구간 높이 (m)": round(result['inertial_height'], 3),
                    "발사 속도 (m/s)": round(result['exit_velocity'], 2),
                    "최대 하중 (kN)": round(result['max_force']/1000, 1),
                    "G-Force": round(result['max_force']/(mass_accel_kg*g), 1),
                    "총 비행 시간 (s)": round(trajectory['t'][-1], 3),
                    "착지 거리 (m)": round(trajectory['x'][-1], 2),
                    "최대 수평 거리 (m)": round(max(trajectory['x']), 2)
                })

                st.subheader("📐 계산식")
                st.markdown(f"""
                **1. 압력 변환**
                ```
                P(Pa) = P(psi) × 6894.76
                ```

                **2. 피스톤 면적**
                ```
                A = π × (D/2)²
                ```

                **3. 시스템 효율**
                ```
                η = 1 - (에너지손실% / 100)
                ```

                **4. 압력에 의한 힘**
                ```
                F_압력 = n × P × A × η
                (n: 실린더 개수)
                ```

                **5. 마찰력 (중량 기반)**
                ```
                F_마찰 = μ × m_가속 × g
                (μ = {mu_friction}, m_가속: 가속구간 질량)
                ```

                **6. 가속 구간 중력**
                ```
                F_중력 = m_가속 × g
                ```

                **7. 순 힘 (가속 구간)**
                ```
                F_순 = F_압력 - F_마찰 - F_중력
                ```

                **8. 가속도**
                ```
                a = F_순 / m_가속
                ```

                **9. 발사 속도**
                ```
                v = √(2 × a × s)
                (s: 스트로크)
                ```

                **10. 가속 구간 높이**
                ```
                h_가속 = s
                ```

                **11. 관성 구간 (공기저항 포함)**
                ```
                F_항력 = 0.5 × ρ × Cd × A × v²
                a_관성 = -(g + F_항력/m_관성)
                (m_관성: 관성구간 질량)
                ```

                **12. 총 높이**
                ```
                h_총 = h_가속 + h_관성
                ```

                **상수 및 입력 값:**
                - g = 9.81 m/s²
                - ρ = 1.225 kg/m³ (공기 밀도)
                - Cd = {Cd} (항력계수)
                - μ = {mu_friction} (마찰계수)

                **질량 구분:**
                - m_가속: 가속 구간 질량 (발사체 + 구동부)
                - m_관성: 관성 구간 질량 (발사체만)
                """, unsafe_allow_html=True)

else:
    # 초기 화면
    st.info("👈 왼쪽 사이드바에서 파라미터를 설정하고 '시뮬레이션 실행' 버튼을 눌러주세요.")

    # 사용 가이드
    st.header("📖 사용 가이드")

    st.markdown("""
    ### 1️⃣ 공압 시스템
    - **실린더 개수**: 병렬로 배치된 실린더의 개수
    - **직경**: 피스톤의 직경 (mm)
    - **압력**: 작동 압력 (psi)
    - **스트로크**: 피스톤의 이동 거리 (mm)

    ### 2️⃣ 물체 특성
    - **가속 구간 질량**: 발사체 + 구동부 전체 질량 (kg)
    - **관성 구간 질량**: 발사체만의 질량, 구동부 제외 (kg)
    - **발사체 단면적**: 공기 저항을 받는 발사체의 단면적 (m²)

    ### 3️⃣ 시스템 손실
    - **에너지 손실**: 마찰, 공기 누설 등으로 인한 전체 에너지 손실률 (%)
    - **마찰계수**: 실린더 및 가이드 레일의 마찰계수 (μ)
    - **항력계수**: 발사체의 공기 저항 계수 (Cd)
      - 평평한 끝 원통: 0.8~1.0
      - 둥근 끝: 0.5
      - 유선형: 0.2

    ### 4️⃣ 발사 각도
    - **발사 각도**: +Y축(연직)에서 +X축 방향으로의 각도
      - 0° = 완전 수평 발사
      - 45° = 최대 사거리 (이론값)
      - 90° = 완전 수직 발사

    ### 📊 결과
    시뮬레이션을 실행하면 다음을 확인할 수 있습니다:
    - ✅ 총 상승 높이 (가속 + 관성 구간)
    - ✅ 발사 속도 및 가속도
    - ✅ 최대 하중 (G-Force)
    - ✅ 시간에 따른 높이/속도/하중 변화 그래프
    - ✅ 포물선 운동 애니메이션
    - ✅ 착지 거리 예측
    """)

    st.markdown("---")
    st.success("💡 **Tip**: 다양한 파라미터 조합을 시도하여 최적의 설계를 찾아보세요!")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🚀 팝업 시스템 시뮬레이터 v1.0 | Made with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
