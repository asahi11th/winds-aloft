from datetime import datetime, timedelta, timezone
import glob
import io
import math
import time

import matplotlib

matplotlib.use('Agg')  # マルチスレッド/複数アクセス時の描画スレッド安全化
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# サーバーにインストールされたフォントをMatplotlibへ直接読み込ませる
for font_path in glob.glob('/usr/share/fonts/**/*.[o|t]tf', recursive=True):
  try:
    fm.fontManager.addfont(font_path)
  except Exception:
    pass

plt.rcParams['font.family'] = [
    'Meiryo',
    'Yu Gothic',
    'IPAGothic',
    'IPAexGothic',
    'sans-serif',
]
plt.rcParams['axes.unicode_minus'] = False

# 8地点の定義
LOCATIONS_CONFIG = [
    {
        'name': '姫路',
        'lat': 34.815,
        'lon': 134.685,
        'rect': [0.015, 0.66, 0.285, 0.28],
        'conn': (1.0, 0.2),
    },
    {
        'name': '八尾 (RJOY)',
        'lat': 34.596,
        'lon': 135.594,
        'rect': [0.357, 0.66, 0.285, 0.28],
        'conn': (0.5, 0.0),
    },
    {
        'name': '名古屋 (RJNA)',
        'lat': 35.255,
        'lon': 136.924,
        'rect': [0.700, 0.66, 0.285, 0.28],
        'conn': (0.0, 0.2),
    },
    {
        'name': '岡山 (RJOB)',
        'lat': 34.757,
        'lon': 133.918,
        'rect': [0.015, 0.33, 0.285, 0.28],
        'conn': (1.0, 0.5),
    },
    {
        'name': '和歌山',
        'lat': 34.226,
        'lon': 135.168,
        'rect': [0.700, 0.33, 0.285, 0.28],
        'conn': (0.0, 0.5),
    },
    {
        'name': '高松 (RJOT)',
        'lat': 34.214,
        'lon': 134.015,
        'rect': [0.015, 0.025, 0.285, 0.28],
        'conn': (1.0, 0.8),
    },
    {
        'name': '淡路',
        'lat': 34.341,
        'lon': 134.856,
        'rect': [0.357, 0.025, 0.285, 0.28],
        'conn': (0.5, 0.88),
    },
    {
        'name': '南紀白浜 (RJBD)',
        'lat': 33.662,
        'lon': 135.364,
        'rect': [0.700, 0.025, 0.285, 0.28],
        'conn': (0.0, 0.8),
    },
]

PRESSURE_LEVELS = [
    {'hpa': '1000', 'ft': 364},
    {'hpa': '950', 'ft': 1940},
    {'hpa': '925', 'ft': 2500},
    {'hpa': '900', 'ft': 3210},
    {'hpa': '850', 'ft': 4780},
    {'hpa': '800', 'ft': 6390},
]

TARGET_ALTITUDES = [0, 2000, 2500, 3000, 5000, 6400]
MAG_VARIATION = 8.0


@st.cache_data(ttl=86400)
def get_japan_geojson():
  url = 'https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson'
  try:
    resp = requests.get(url, timeout=5)
    if resp.status_code == 200:
      return resp.json()
  except Exception:
    pass
  return None


def calculate_isa_diff(alt_ft, temp_c):
  isa_temp = 15.0 - (alt_ft / 1000.0) * 1.98
  return round(temp_c - isa_temp)


def fetch_weather_data(lat, lon, model_code):
  if model_code == 'JMA':
    urls = [
        'https://api.open-meteo.com/v1/jma',
        'https://api.open-meteo.com/v1/forecast',
    ]
  elif model_code == 'GFS':
    urls = [
        'https://api.open-meteo.com/v1/gfs',
        'https://api.open-meteo.com/v1/forecast',
    ]
  elif model_code == 'ICON':
    urls = [
        'https://api.open-meteo.com/v1/dwd-icon',
        'https://api.open-meteo.com/v1/forecast',
    ]
  else:
    urls = [
        'https://api.open-meteo.com/v1/ecmwf',
        'https://api.open-meteo.com/v1/forecast',
    ]

  hourly_vars = ['temperature_2m', 'wind_speed_10m', 'wind_direction_10m']
  for p in PRESSURE_LEVELS:
    h = p['hpa']
    hourly_vars.extend([
        f'temperature_{h}hPa',
        f'wind_speed_{h}hPa',
        f'wind_direction_{h}hPa',
    ])

  last_error = None
  for url in urls:
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join(hourly_vars),
        'wind_speed_unit': 'kn',
        'timezone': 'Asia/Tokyo',
    }
    try:
      resp = requests.get(url, params=params, timeout=10)
      resp.raise_for_status()
      return resp.json()
    except Exception as e:
      last_error = e
      continue
  raise last_error


def get_raw_var_value(hourly_dict, var_prefix, hpa_str, idx):
  target_key = f'{var_prefix}_{hpa_str}hPa'.lower()
  for k, v in hourly_dict.items():
    if k.lower() == target_key:
      if idx < len(v) and v[idx] is not None:
        return float(v[idx])
  return None


def process_location_data(data_json, target_datetime):
  hourly = data_json.get('hourly', {})
  time_list = hourly.get('time', [])
  if not time_list:
    raise ValueError('データなし')

  times = pd.to_datetime(time_list)
  target_dt = pd.to_datetime(target_datetime)
  idx = abs(times - target_dt).argmin()

  sfc_temp = hourly.get('temperature_2m', [15.0])[idx] or 15.0
  sfc_ws = hourly.get('wind_speed_10m', [0.0])[idx] or 0.0
  sfc_wd = hourly.get('wind_direction_10m', [0.0])[idx] or 0.0

  press_alts = [0.0]
  temps = [sfc_temp]

  rad_sfc = math.radians(sfc_wd)
  u_comp = [-sfc_ws * math.sin(rad_sfc)]
  v_comp = [-sfc_ws * math.cos(rad_sfc)]

  for p in PRESSURE_LEVELS:
    h = p['hpa']
    alt_ft = float(p['ft'])

    t_val = get_raw_var_value(hourly, 'temperature', h, idx)
    ws_val = get_raw_var_value(hourly, 'wind_speed', h, idx)
    wd_val = get_raw_var_value(hourly, 'wind_direction', h, idx)

    if t_val is None:
      t_val = sfc_temp - (alt_ft / 1000.0) * 1.98
    if ws_val is None:
      ws_val = sfc_ws
    if wd_val is None:
      wd_val = sfc_wd

    press_alts.append(alt_ft)
    temps.append(t_val)

    rad = math.radians(wd_val)
    u_comp.append(-ws_val * math.sin(rad))
    v_comp.append(-ws_val * math.cos(rad))

  rows = []
  for target_alt in TARGET_ALTITUDES:
    if target_alt == 0:
      t_interp = sfc_temp
      u_interp = u_comp[0]
      v_interp = v_comp[0]
    else:
      t_interp = float(np.interp(target_alt, press_alts, temps))
      u_interp = float(np.interp(target_alt, press_alts, u_comp))
      v_interp = float(np.interp(target_alt, press_alts, v_comp))

      expected_isa_temp = sfc_temp - (target_alt / 1000.0) * 1.98
      t_interp = np.clip(t_interp, expected_isa_temp - 8.0, sfc_temp + 5.0)

    ws_interp = math.hypot(u_interp, v_interp)
    wd_true = (math.degrees(math.atan2(-u_interp, -v_interp)) + 360) % 360

    wd_mag = (wd_true + MAG_VARIATION) % 360
    wd_mag_rounded = int(round(wd_mag))

    wd_mag_str = (
        '360'
        if wd_mag_rounded in (0, 360)
        else f'{wd_mag_rounded:03d}'
    )
    isa_diff = calculate_isa_diff(target_alt, t_interp)
    temp_rounded = int(round(t_interp))

    rows.append([
        f'{target_alt}',
        f'{temp_rounded}',
        f'{wd_mag_str}M / {int(round(ws_interp)):02d}',
        f'{isa_diff:+d}',
    ])

  return rows


def generate_map_figure(
    all_location_data, geojson_data, target_datetime, model_code
):
  fig = plt.figure(figsize=(16, 11), dpi=150, facecolor='white')

  model_display_names = {
      'ECMWF': 'ECMWF (IFS)',
      'JMA': 'JMA (GSM)',
      'GFS': 'GFS (NOAA)',
      'ICON': 'ICON (DWD)',
  }
  model_name = model_display_names.get(model_code, model_code)

  dt_str = target_datetime.strftime('%Y年%m月%d日 %H:00 JST')
  fig.suptitle(
      f'出発時刻の予想風 （ {dt_str} / モデル: {model_name}）',
      fontsize=16,
      fontweight='bold',
      color='#1A365D',
      y=0.97,
  )

  ax_map = fig.add_axes([0.27, 0.20, 0.46, 0.54])
  ax_map.set_aspect('equal')
  ax_map.axis('off')
  ax_map.set_xlim(133.0, 137.5)
  ax_map.set_ylim(33.2, 35.8)
  ax_map.set_facecolor('#D4E6F1')

  if geojson_data:
    for feature in geojson_data.get('features', []):
      geom = feature.get('geometry', {})
      gtype = geom.get('type')
      coords = geom.get('coordinates', [])

      if gtype == 'Polygon':
        for poly in coords:
          pts = np.array(poly)
          ax_map.fill(
              pts[:, 0],
              pts[:, 1],
              facecolor='#E2EFDA',
              edgecolor='#548235',
              linewidth=0.6,
              zorder=2,
          )
      elif gtype == 'MultiPolygon':
        for mpoly in coords:
          for poly in mpoly:
            pts = np.array(poly)
            ax_map.fill(
                pts[:, 0],
                pts[:, 1],
                facecolor='#E2EFDA',
                edgecolor='#548235',
                linewidth=0.6,
                zorder=2,
            )

  for loc in LOCATIONS_CONFIG:
    name = loc['name']
    lat, lon = loc['lat'], loc['lon']
    table_data = all_location_data.get(name, [])

    if not table_data:
      table_data = [
          [f'{alt}', '---', '---', '---'] for alt in TARGET_ALTITUDES
      ]

    is_airport = 'RJ' in name or '空港' in name
    if is_airport:
      ax_map.scatter(
          lon, lat, marker='$✈$', s=130, color='#C0392B', zorder=5
      )
    else:
      ax_map.scatter(
          lon,
          lat,
          color='#555555',
          s=55,
          zorder=5,
          edgecolors='white',
          linewidths=1.0,
      )

    rect = loc['rect']
    ax_table = fig.add_axes(rect, facecolor='white')
    ax_table.axis('off')

    ax_table.text(
        0.0,
        0.91,
        name,
        fontsize=15,
        fontweight='bold',
        color='#1A365D',
        transform=ax_table.transAxes,
        va='bottom',
        fontfamily=['Meiryo', 'Yu Gothic', 'IPAGothic', 'sans-serif'],
    )

    col_labels = ['高度(ft)', '気温(℃)', '風', 'ISA差']
    col_widths = [0.22, 0.20, 0.38, 0.20]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc='center',
        loc='lower center',
        bbox=[0.0, 0.0, 1.0, 0.88],
    )

    for (r, c), cell in table.get_celld().items():
      cell.set_linewidth(0.8)
      cell.set_edgecolor('#1A5276')
      txt = cell.get_text()
      txt.set_clip_on(False)

      if r == 0:
        cell.set_facecolor('#2B6CB0')
        txt.set_color('white')
        txt.set_fontsize(12.5)
        txt.set_weight('bold')
        txt.set_fontfamily(['Meiryo', 'Yu Gothic', 'IPAGothic', 'sans-serif'])
      else:
        cell.set_facecolor('#FFFFFF' if r % 2 == 1 else '#EDF2F7')
        txt.set_color('#000000')
        txt.set_fontsize(13.0)
        txt.set_weight('bold')
        txt.set_fontfamily(['DejaVu Sans', 'Arial', 'sans-serif'])

    conn_x, conn_y = loc['conn']
    con = mpatches.ConnectionPatch(
        xyA=(conn_x, conn_y),
        coordsA=ax_table.transAxes,
        xyB=(lon, lat),
        coordsB=ax_map.transData,
        arrowstyle='-',
        color='#2B6CB0',
        linewidth=1.2,
        zorder=4,
    )
    fig.add_artist(con)

  source_dict = {
      'ECMWF': 'ECMWF (IFS Model)',
      'JMA': 'JMA (GSM Model)',
      'GFS': 'NOAA (GFS Model)',
      'ICON': 'DWD (ICON Model)',
  }
  source_label = source_dict.get(model_code, model_code)

  fig.text(
      0.985,
      0.005,
      f'Data Source: {source_label} via Open-Meteo',
      fontsize=8.0,
      color='#4A5568',
      ha='right',
      va='bottom',
      style='italic',
  )

  return fig


def render_runway_html(progress_pct, is_takeoff=False):
  if is_takeoff:
    plane_x, plane_y, plane_rotate = 265, -18, -20
  else:
    plane_x = int((progress_pct / 100) * 220)
    plane_y, plane_rotate = 0, 0

  return f"""
    <style>
    .runway-container {{ position: relative; width: 100%; max-width: 320px; height: 48px; background-color: #f0f4f8; border-radius: 8px; overflow: hidden; margin: 8px 0; border: 1px solid #cbd5e1; }}
    .runway-line {{ position: absolute; bottom: 12px; left: 0; width: 100%; height: 2px; border-top: 2px dashed #94a3b8; }}
    .plane-icon {{ position: absolute; font-size: 20px; line-height: 1; left: 12px; bottom: 10px; transform: translate3d({plane_x}px, {plane_y}px, 0) rotate({plane_rotate}deg); transition: transform 0.8s cubic-bezier(0.2, 0.8, 0.2, 1); will-change: transform; z-index: 10; }}
    </style>
    <div class="runway-container">
        <div class="runway-line"></div>
        <div class="plane-icon">🛫</div>
    </div>
    """


# --- Streamlit 画面構成 ---
st.set_page_config(page_title='🛫Winds Aloft 予想風作成🛫', layout='wide')

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Meiryo', 'Meiryo UI', 'Hiragino Kaku Gothic ProN', sans-serif !important; }
    div[data-baseweb="select"], div[data-baseweb="input"] { cursor: pointer !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('🛫Winds Aloft 予想風 出力🛫')

# セッション状態の初期化
if 'pdf_bytes' not in st.session_state:
  st.session_state.pdf_bytes = None
if 'pdf_filename' not in st.session_state:
  st.session_state.pdf_filename = ''
if 'info_text' not in st.session_state:
  st.session_state.info_text = ''

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)

MODEL_OPTIONS = {
    'ECMWF (欧州中期予報センター)': 'ECMWF',
    '気象庁 JMA (日本)': 'JMA',
    'GFS (アメリカ海洋大気庁)': 'GFS',
    'ICON (ドイツ気象庁)': 'ICON',
}

col1, col2, col3, col4, _ = st.columns([1.6, 1.0, 0.9, 1.2, 1.3])

with col1:
  selected_label = st.selectbox('気象モデル', list(MODEL_OPTIONS.keys()), index=0)
  model_code = MODEL_OPTIONS[selected_label]

with col2:
  selected_date = st.date_input('日付', now_jst.date())

with col3:
  current_hour = now_jst.hour
  all_hours = [(current_hour + i) % 24 for i in range(24)]
  hours_list = [f'{h:02d}:00' for h in all_hours]
  selected_hour_str = st.selectbox('時刻 (JST)', hours_list, index=0)
  selected_hour = int(selected_hour_str.split(':')[0])

with col4:
  st.write(' ')
  st.write(' ')
  if st.session_state.pdf_bytes:
    st.download_button(
        label='📄 PDFをダウンロード',
        data=st.session_state.pdf_bytes,
        file_name=st.session_state.pdf_filename,
        mime='application/pdf',
        type='secondary',
        key='download_top',
    )
  else:
    st.button('📄 PDFをダウンロード', disabled=True)

target_datetime = datetime.combine(
    selected_date, datetime.min.time()
) + timedelta(hours=selected_hour)

if st.button('データ取得＆予想風を作成', type='primary'):
  loading_container = st.container()

  with loading_container:
    plane_box = st.empty()
    status_text = st.empty()

    plane_box.markdown(render_runway_html(0), unsafe_allow_html=True)
    status_text.markdown('**[1/3]** 日本地図データを準備中...')
    geojson_data = get_japan_geojson()

    all_location_data = {}
    total_locs = len(LOCATIONS_CONFIG)
    has_error = False

    for i, loc in enumerate(LOCATIONS_CONFIG, 1):
      pct = int((i / total_locs) * 80)
      plane_box.markdown(render_runway_html(pct), unsafe_allow_html=True)
      status_text.markdown(
          f'**[2/3]** 気象データ（{model_code}）を取得中: **{loc["name"]}**'
      )

      try:
        data_json = fetch_weather_data(loc['lat'], loc['lon'], model_code)
        rows = process_location_data(data_json, target_datetime)
        all_location_data[loc['name']] = rows
      except Exception as e:
        st.error(
            f"{loc['name']} のデータ取得に失敗しました。"
            ' 時間をおいて再試行してください。'
        )
        has_error = True
        break

    if not has_error:
      status_text.markdown(
          '**[3/3]** 高度別予想風の表とPDFを出力中...'
      )
      fig = generate_map_figure(
          all_location_data, geojson_data, target_datetime, model_code
      )

      # PDFデータ化
      pdf_buffer = io.BytesIO()
      fig.savefig(
          pdf_buffer,
          format='pdf',
          bbox_inches='tight',
          facecolor=fig.get_facecolor(),
      )
      pdf_buffer.seek(0)

      # セッションに保管
      filename = f"WindsAloft_{model_code}_{selected_date.strftime('%Y%m%d')}_{selected_hour:02d}00.pdf"
      executed_at = datetime.now(jst).strftime('%Y年%m月%d日 %H:%M JST')

      st.session_state.pdf_bytes = pdf_buffer.getvalue()
      st.session_state.pdf_filename = filename
      st.session_state.info_text = (
          f'取得日時: <b>{executed_at}</b> ／ 対象日時:'
          f' <b>{selected_date.strftime("%Y年%m月%d日")}'
          f' {selected_hour:02d}:00 JST</b> （モデル: <b>{model_code}</b>）'
      )

      # メモリ解放（超重要）
      plt.close(fig)

      plane_box.markdown(
          render_runway_html(100, is_takeoff=True), unsafe_allow_html=True
      )
      status_text.markdown('✨ **テイクオフ！完成しました。**')
      time.sleep(0.5)

  loading_container.empty()
  st.rerun()

# 結果表示
if st.session_state.pdf_bytes:
  st.markdown(
      f"""
    <div style="
        display: inline-block; background-color: #E6F4EA; color: #137333;
        padding: 8px 16px; border-radius: 8px; font-size: 14px; margin-bottom: 12px; border: 1px solid #CEEAD6;
    ">
        {st.session_state.info_text}
    </div>
    """,
      unsafe_allow_html=True,
  )

  # PDFバッファから画面表示用画像を再描画して軽量に表示
  # （Streamlitのセッションを肥大化させないための措置）
  st.info('※上の「PDFをダウンロード」ボタンからPDFファイルを取得できます。')
