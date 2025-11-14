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
Cd = 0.5  # 항력계수
mu_friction = 0.15  # 마찰계수

def psi_to_pa(psi):
    """PSI를 Pascal로 변환"""
    return psi * 6894.76

def calculate_popup_system(diameter_mm, pressure_psi, stroke_mm, mass_kg, 
                          num_cylinders, energy_loss_percent):
    """
    팝업 시스템 전체 계산
    
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
    
    # 마찰력 계산
    F_friction = mu_friction * F_pressure  # N
    
    # 중력
    F_gravity = mass_kg * g  # N
    
    # 순 힘
    F_net = F_pressure - F_friction - F_gravity  # N
    
    if F_net <= 0:
        return None  # 힘이 충분하지 않음
    
    # 가속도
    a = F_net / mass_kg  # m/s^2
    
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
    
    # 공기 저항 면적
    A_drag = area
    
    while v > 0:
        # 공기 저항력
        F_drag = 0.5 * rho_air * Cd * A_drag * v ** 2
        
        # 가속도 (음수)
        a_inertial = -(g + F_drag / mass_kg)
        
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
    force_inertial = [-mass_kg * g - 0.5 * rho_air * Cd * A_drag * v**2 
                      for v in velocity_inertial]
    
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

def simulate_projectile(v0, angle_deg, mass_kg):
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
    
    # 피스톤 직경 기반 공기 저항 (간단히 처리)
    A_drag = 0.028  # m^2 (대략적)
    
    while y >= -0.01:
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
mass_kg = st.sidebar.number_input(
    "질량 (kg)",
    min_value=10.0,
    max_value=2000.0,
    value=500.0,
    step=10.0,
    help="발사체 질량"
)

st.sidebar.subheader("3️⃣ 시스템 손실")
energy_loss = st.sidebar.slider(
    "에너지 손실 (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="마찰, 누설 등으로 인한 에너지 손실"
)

st.sidebar.subheader("4️⃣ 발사 각도")
launch_angle = st.sidebar.slider(
    "발사 각도 (°)",
    min_value=30,
    max_value=90,
    value=90,
    step=5,
    help="Y축(연직) 기준, X축 방향으로의 각도. 90°는 완전 수직"
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
            mass_kg, num_cylinders, energy_loss
        )
        
        if result is None:
            st.error("⚠️ 힘이 충분하지 않습니다! 압력을 높이거나 질량을 줄여주세요.")
        else:
            # 포물선 운동 계산
            trajectory = simulate_projectile(
                result['exit_velocity'], 
                launch_angle, 
                mass_kg
            )
            
            # 결과 저장 (애니메이션용)
            st.session_state['result'] = result
            st.session_state['trajectory'] = trajectory
            st.session_state['params'] = {
                'num_cylinders': num_cylinders,
                'diameter_mm': diameter_mm,
                'pressure_psi': pressure_psi,
                'stroke_mm': stroke_mm,
                'mass_kg': mass_kg,
                'energy_loss': energy_loss,
                'launch_angle': launch_angle
            }
            
            # ================================================================
            # 주요 결과 표시
            # ================================================================
            st.header("📊 시뮬레이션 결과")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="총 상승 높이",
                    value=f"{result['total_height']:.2f} m",
                    delta=f"목표 대비"
                )
            
            with col2:
                st.metric(
                    label="발사 속도",
                    value=f"{result['exit_velocity']:.2f} m/s",
                    delta=f"{result['exit_velocity'] * 3.6:.1f} km/h"
                )
            
            with col3:
                st.metric(
                    label="최대 하중",
                    value=f"{result['max_force']/1000:.1f} kN",
                    delta=f"{result['max_force']/mass_kg/g:.1f}G"
                )
            
            with col4:
                landing_dist = trajectory['x'][-1]
                st.metric(
                    label="착지 거리",
                    value=f"{landing_dist:.2f} m",
                    delta=f"각도 {launch_angle}°"
                )
            
            st.markdown("---")
            
            # ================================================================
            # 상세 분석 그래프
            # ================================================================
            st.header("📈 상세 분석")
            
            tab1, tab2, tab3 = st.tabs(["높이 분석", "속도 분석", "하중 분석"])
            
            with tab1:
                # 높이-시간 그래프
                fig1 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("높이 vs 시간", "높이 구성"),
                    specs=[[{"type": "xy"}, {"type": "domain"}]]
                )
                
                # 가속 구간
                fig1.add_trace(
                    go.Scatter(
                        x=result['time_accel'],
                        y=result['height_accel'],
                        mode='lines',
                        name='가속 구간',
                        line=dict(color='blue', width=3)
                    ),
                    row=1, col=1
                )
                
                # 관성 구간
                fig1.add_trace(
                    go.Scatter(
                        x=result['time_inertial'],
                        y=result['height_inertial'],
                        mode='lines',
                        name='관성 구간',
                        line=dict(color='orange', width=3)
                    ),
                    row=1, col=1
                )
                
                # 높이 구성 파이 차트
                fig1.add_trace(
                    go.Pie(
                        labels=['가속 구간', '관성 구간'],
                        values=[result['accel_height'], result['inertial_height']],
                        marker=dict(colors=['blue', 'orange'])
                    ),
                    row=1, col=2
                )
                
                fig1.update_xaxes(title_text="시간 (s)", row=1, col=1)
                fig1.update_yaxes(title_text="높이 (m)", row=1, col=1)
                
                fig1.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig1, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**가속 구간**: {result['accel_height']:.3f} m")
                with col2:
                    st.info(f"**관성 구간**: {result['inertial_height']:.3f} m")
                with col3:
                    st.info(f"**총 높이**: {result['total_height']:.3f} m")
            
            with tab2:
                # 속도-시간 그래프
                fig2 = go.Figure()
                
                # 가속 구간
                fig2.add_trace(
                    go.Scatter(
                        x=result['time_accel'],
                        y=result['velocity_accel'],
                        mode='lines',
                        name='가속 구간',
                        line=dict(color='green', width=3),
                        fill='tozeroy'
                    )
                )
                
                # 관성 구간
                fig2.add_trace(
                    go.Scatter(
                        x=result['time_inertial'],
                        y=result['velocity_inertial'],
                        mode='lines',
                        name='관성 구간',
                        line=dict(color='red', width=3),
                        fill='tozeroy'
                    )
                )
                
                fig2.update_layout(
                    title="속도 변화",
                    xaxis_title="시간 (s)",
                    yaxis_title="속도 (m/s)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**최대 속도**: {result['exit_velocity']:.2f} m/s")
                with col2:
                    st.success(f"**가속도**: {result['acceleration']:.2f} m/s²")
            
            with tab3:
                # 하중-시간 그래프
                fig3 = go.Figure()
                
                # 가속 구간
                fig3.add_trace(
                    go.Scatter(
                        x=result['time_accel'],
                        y=result['force_accel'] / 1000,  # kN 단위
                        mode='lines',
                        name='가속 구간',
                        line=dict(color='purple', width=3),
                        fill='tozeroy'
                    )
                )
                
                # 관성 구간
                fig3.add_trace(
                    go.Scatter(
                        x=result['time_inertial'],
                        y=np.array(result['force_inertial']) / 1000,  # kN 단위
                        mode='lines',
                        name='관성 구간 (공기저항+중력)',
                        line=dict(color='brown', width=3),
                        fill='tozeroy'
                    )
                )
                
                fig3.update_layout(
                    title="하중 변화",
                    xaxis_title="시간 (s)",
                    yaxis_title="하중 (kN)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.warning(f"**최대 하중**: {result['max_force']/1000:.1f} kN")
                with col2:
                    g_force = result['max_force'] / (mass_kg * g)
                    st.warning(f"**G-Force**: {g_force:.1f} G")
            
            st.markdown("---")
            
            # ================================================================
            # 포물선 운동 애니메이션
            # ================================================================
            st.header("🎬 포물선 운동 시뮬레이션")
            
            # 애니메이션 컨트롤
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("애니메이션 설정")
                animation_speed = st.slider(
                    "재생 속도",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.5
                )
                
                show_velocity = st.checkbox("속도 벡터 표시", value=True)
                show_trail = st.checkbox("궤적 표시", value=True)
            
            with col1:
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
                    title=f"발사 각도: {launch_angle}° (연직에서 {90-launch_angle}° 틀어짐)",
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
            
            # ================================================================
            # 추가 정보
            # ================================================================
            st.markdown("---")
            st.header("📋 상세 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("입력 파라미터")
                st.json({
                    "실린더 개수": num_cylinders,
                    "직경 (mm)": diameter_mm,
                    "압력 (psi)": pressure_psi,
                    "스트로크 (mm)": stroke_mm,
                    "질량 (kg)": mass_kg,
                    "에너지 손실 (%)": energy_loss,
                    "발사 각도 (°)": launch_angle
                })
            
            with col2:
                st.subheader("계산 결과")
                st.json({
                    "총 상승 높이 (m)": round(result['total_height'], 3),
                    "가속 구간 높이 (m)": round(result['accel_height'], 3),
                    "관성 구간 높이 (m)": round(result['inertial_height'], 3),
                    "발사 속도 (m/s)": round(result['exit_velocity'], 2),
                    "최대 하중 (kN)": round(result['max_force']/1000, 1),
                    "G-Force": round(result['max_force']/(mass_kg*g), 1),
                    "총 비행 시간 (s)": round(trajectory['t'][-1], 3),
                    "착지 거리 (m)": round(trajectory['x'][-1], 2),
                    "최대 수평 거리 (m)": round(max(trajectory['x']), 2)
                })

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
    - **질량**: 발사체의 질량 (kg)
    
    ### 3️⃣ 시스템 손실
    - **에너지 손실**: 마찰, 공기 누설 등으로 인한 전체 에너지 손실률 (%)
    
    ### 4️⃣ 발사 각도
    - **발사 각도**: Y축(연직) 기준으로 X+ 방향으로의 각도
      - 90° = 완전 수직 발사
      - 60° = 30도 경사 발사
      - 45° = 최대 사거리
    
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
