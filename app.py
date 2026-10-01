from datetime import datetime, timedelta, timezone
import glob
import io
import math

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# サーバーにインストールされたフォントをMatplotlibへ直接読み込ませる
for font_path in glob.glob('/usr/share/fonts/**/*.[o|t]tf', recursive=True):
  try:
    fm.fontManager.addfont(font_path)
  except Exception:
    pass

# 日本語フォントを優先設定
plt.rcParams['font.family'] = ['IPAGothic', 'IPAexGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 8地点の定義（名称・緯度・経度・表の配置位置・引出線接続位置）
LOCATIONS_CONFIG = [
    # 姫路 (左上)
    {
        'name': '姫路',
        'lat': 34.815,
        'lon': 134.685,
        'rect': [0.015, 0.66, 0.285, 0.28],
        'conn': (1.0, 0.2),
    },
    # 八尾 (中央上)
    {
        'name': '八尾 (RJOY)',
        'lat': 34.596,
        'lon': 135.594,
        'rect': [0.357, 0.66, 0.285, 0.28],
        'conn': (0.5, 0.0),
    },
    # 名古屋 (右上)
    {
        'name': '名古屋 (RJNA)',
        'lat': 35.255,
        'lon': 136.924,
        'rect': [0.700, 0.66, 0.285, 0.28],
        'conn': (0.0, 0.2),
    },
    # 岡山 (左中)
    {
        'name': '岡山 (RJOB)',
        'lat': 34.757,
        'lon': 133.918,
        'rect': [0.015, 0.33, 0.285, 0.28],
        'conn': (1.0, 0.5),
    },
    # 和歌山 (右中)
    {
        'name': '和歌山',
        'lat': 34.226,
        'lon': 135.168,
        'rect': [0.700, 0.33, 0.285, 0.28],
        'conn': (0.0, 0.5),
    },
    # 高松 (左下)
    {
        'name': '高松 (RJOT)',
        'lat': 34.214,
        'lon': 134.015,
        'rect': [0.015, 0.025, 0.285, 0.28],
        'conn': (1.0, 0.8),
    },
    # 淡路 (中央下)
    {
        'name': '淡路',
        'lat': 34.341,
        'lon': 134.856,
        'rect': [0.357, 0.025, 0.285, 0.28],
        'conn': (0.5, 0.88),
    },
    # 南紀白浜 (右下)
    {
        'name': '南紀白浜 (RJBD)',
        'lat': 33.662,
        'lon': 135.364,
        'rect': [0.700, 0.025, 0.285, 0.28],
        'conn': (0.0, 0.8),
    },
]

# 気圧面 (hPa) と標準高度 (ft)
PRESSURE_LEVELS = [
    {'hpa': '1000', 'ft': 364},
    {'hpa': '950', 'ft': 1940},
    {'hpa': '925', 'ft': 2500},
    {'hpa': '900', 'ft': 3210},
    {'hpa': '850', 'ft': 4780},
    {'hpa': '800', 'ft': 6390},
]

# 出力対象高度
TARGET_ALTITUDES = [0, 2000, 2500, 3000, 5000, 6400]
MAG_VARIATION = 8.0  # 磁気偏角補正(+8度)


@st.cache_data(ttl=86400)
def get_japan_geojson():
  """日本地図のGeoJSONを取得"""
  url = 'https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson'
  try:
    resp = requests.get(url, timeout=5)
    if resp.status_code == 200:
      return resp.json()
  except Exception:
    pass
  return None


def calculate_isa_diff(alt_ft, temp_c):
  """標準大気(ISA)との差分計算"""
  isa_temp = 15.0 - (alt_ft / 1000.0) * 1.98
  return round(temp_c - isa_temp)


def fetch_weather_data(lat, lon, model_choice):
  """選択されたモデルに応じた気象データを取得"""
  if model_choice == '気象庁 (JMA)':
    urls = [
        'https://api.open-meteo.com/v1/jma',
        'https://api.open-meteo.com/v1/forecast',
    ]
  elif model_choice == 'GFS (NOAA)':
    urls = [
        'https://api.open-meteo.com/v1/gfs',
        'https://api.open-meteo.com/v1/forecast',
    ]
  elif model_choice == 'ICON (DWD)':
    urls = [
        'https://api.open-meteo.com/v1/dwd-icon',
        'https://api.open-meteo.com/v1/forecast',
    ]
  else:  # ECMWF (デフォルト)
    urls = [
        'https://api.open-meteo.com/v1/ecmwf',
        'https://api.open-meteo.com/v1/forecast',
    ]

  hourly_vars = [
      'temperature_2m',
      'wind_speed_10m',
      'wind_direction_10m',
  ]
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
  """気圧面変数の取得 (未存在・Noneの場合は None を返す)"""
  target_key = f'{var_prefix}_{hpa_str}hPa'.lower()
  for k, v in hourly_dict.items():
    if k.lower() == target_key:
      if idx < len(v) and v[idx] is not None:
        return float(v[idx])
  return None


def process_location_data(data_json, target_datetime):
  """指定地点のデータを解析・欠損補正・減率チェックを行って表用データを作成"""
  hourly = data_json.get('hourly', {})
  time_list = hourly.get('time', [])
  if not time_list:
    raise ValueError('データなし')

  times = pd.to_datetime(time_list)
  target_dt = pd.to_datetime(target_datetime)
  idx = abs(times - target_dt).argmin()

  # 地上(2m / 10m)データ
  sfc_temp = hourly.get('temperature_2m', [15.0])[idx]
  if sfc_temp is None:
    sfc_temp = 15.0

  sfc_ws = hourly.get('wind_speed_10m', [0.0])[idx]
  if sfc_ws is None:
    sfc_ws = 0.0

  sfc_wd = hourly.get('wind_direction_10m', [0.0])[idx]
  if sfc_wd is None:
    sfc_wd = 0.0

  # 0ft(地上)の基準点を追加
  press_alts = [0.0]
  temps = [sfc_temp]

  rad_sfc = math.radians(sfc_wd)
  u_comp = [-sfc_ws * math.sin(rad_sfc)]
  v_comp = [-sfc_ws * math.cos(rad_sfc)]

  # 各気圧面のデータ取得（欠損時は地上からの標準減率で補算）
  for p in PRESSURE_LEVELS:
    h = p['hpa']
    alt_ft = float(p['ft'])

    t_val = get_raw_var_value(hourly, 'temperature', h, idx)
    ws_val = get_raw_var_value(hourly, 'wind_speed', h, idx)
    wd_val = get_raw_var_value(hourly, 'wind_direction', h, idx)

    # 気温欠損時の補正：地上気温から標準減率（-1.98℃/1,000ft）で自動算出
    if t_val is None:
      t_val = sfc_temp - (alt_ft / 1000.0) * 1.98

    # 風速・風向欠損時の補正：直近（地上）の値をフォールバック
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
      # 各高度へ補間
      t_interp = float(np.interp(target_alt, press_alts, temps))
      u_interp = float(np.interp(target_alt, press_alts, u_comp))
      v_interp = float(np.interp(target_alt, press_alts, v_comp))

      # 【標準減率・整合性チェックガード】
      expected_isa_temp = sfc_temp - (target_alt / 1000.0) * 1.98
      t_interp = np.clip(t_interp, expected_isa_temp - 8.0, sfc_temp + 5.0)

    ws_interp = math.hypot(u_interp, v_interp)
    wd_true = (math.degrees(math.atan2(-u_interp, -v_interp)) + 360) % 360

    # 磁方位計算（North 360°表記対応：000Mは存在せず360Mとする）
    wd_mag = (wd_true + MAG_VARIATION) % 360
    wd_mag_rounded = int(round(wd_mag))

    if wd_mag_rounded == 0 or wd_mag_rounded == 360:
      wd_mag_str = '360'
    else:
      wd_mag_str = f'{wd_mag_rounded:03d}'

    isa_diff = calculate_isa_diff(target_alt, t_interp)

    # 気温の表記：0℃以上はプラス記号なし、マイナス時のみ '-' を付与
    temp_rounded = int(round(t_interp))

    rows.append([
        f'{target_alt}',
        f'{temp_rounded}',
        f'{wd_mag_str}M / {int(round(ws_interp)):02d}',
        f'{isa_diff:+d}',
    ])

  return rows


def generate_map_figure(
    all_location_data, geojson_data, target_datetime, model_choice
):
  """カラー着色された地図・表・引出線を描画したMatplotlib Figureを生成"""
  fig = plt.figure(figsize=(16, 11), dpi=200, facecolor='white')

  # 最上部タイトル（対象の予想日時）
  dt_str = target_datetime.strftime('%Y年%m月%d日 %H:00 JST')
  fig.suptitle(
      f'出発時刻の予想風 （ {dt_str} / モデル: {model_choice}）',
      fontsize=16,
      fontweight='bold',
      color='#1A365D',
      y=0.97,
  )

  # 中央の地図用Axes
  ax_map = fig.add_axes([0.27, 0.20, 0.46, 0.54])
  ax_map.set_aspect('equal')
  ax_map.axis('off')
  ax_map.set_xlim(133.0, 137.5)
  ax_map.set_ylim(33.2, 35.8)

  # 海の色
  ax_map.set_facecolor('#D4E6F1')

  # 陸地描画
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

  # 地点表と引出線
  for loc in LOCATIONS_CONFIG:
    name = loc['name']
    lat, lon = loc['lat'], loc['lon']
    table_data = all_location_data.get(name, [])

    if not table_data:
      table_data = [
          [f'{alt}', '---', '---', '---'] for alt in TARGET_ALTITUDES
      ]

    # --- 地点プロット（空港判定で飛行機マークに変更） ---
    is_airport = 'RJ' in name or '空港' in name

    if is_airport:
      # 視認性の高い赤色飛行機マーク
      ax_map.scatter(
          lon,
          lat,
          marker='$✈$',
          s=130,
          color='#C0392B',
          zorder=5,
      )
    else:
      # 非空港地点は落ち着いたグレーピン
      ax_map.scatter(
          lon,
          lat,
          color='#555555',
          s=55,
          zorder=5,
          edgecolors='white',
          linewidths=1.0,
      )

    # 表用サブAxes
    rect = loc['rect']
    ax_table = fig.add_axes(rect, facecolor='white')
    ax_table.axis('off')

    # 地点タイトル
    ax_table.text(
        0.0,
        0.91,
        name,
        fontsize=15,
        fontweight='bold',
        color='#1A365D',
        transform=ax_table.transAxes,
        va='bottom',
    )

    # 表の作成
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
      cell.set_edgecolor('#2B6CB0')

      txt = cell.get_text()
      txt.set_clip_on(False)

      # 数字・英字の見やすさを最優先したフォント設定（等幅＋自動フォールバック）
      txt.set_fontfamily(
          ['Consolas', 'DejaVu Sans', 'IPAGothic', 'sans-serif']
      )

      if r == 0:
        cell.set_facecolor('#2B6CB0')
        txt.set_color('white')
        txt.set_fontsize(12.5)
        txt.set_weight('bold')
      else:
        if r % 2 == 1:
          cell.set_facecolor('#FFFFFF')
        else:
          cell.set_facecolor('#EDF2F7')

        txt.set_color('#000000')
        txt.set_fontsize(12.0)
        txt.set_weight('bold')  # 数値の視認性を高めるため太字化

    # 引出線
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

  # 右下のデータ参照元注記表記の対応付け
  source_dict = {
      'ECMWF (IFS)': 'ECMWF (IFS Model)',
      '気象庁 (JMA)': 'JMA (GSM Model)',
      'GFS (NOAA)': 'NOAA (GFS Model)',
      'ICON (DWD)': 'DWD (ICON Model)',
  }
  source_label = source_dict.get(model_choice, model_choice)

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


# --- Streamlit 画面構成 ---
st.set_page_config(page_title='🛫Winds Aloft 予想風作成🛫', layout='wide')

# 全体フォント設定およびプルダウンタップ判定拡張CSS
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Meiryo', 'Meiryo UI', 'Hiragino Kaku Gothic ProN', sans-serif !important;
    }
    /* プルダウンの枠内どこをタップしても反応するように設定 */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('🛫Winds Aloft 予想風 出力🛫')
st.caption('Data Source: Open-Meteo API | Wind: °M / kt | Temp: °C')

# 現在の日本時間（JST）を自動取得
jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)

# セッション状態の初期化（PDFデータの保持用）
if 'pdf_data' not in st.session_state:
  st.session_state.pdf_data = None
  st.session_state.pdf_filename = ''
  st.session_state.fig = None
  st.session_state.info_text = ''

# --- 1行目：操作パネル ---
col1, col2, col3, col4, _ = st.columns([1.5, 1.1, 0.9, 1.2, 1.4])

with col1:
  model_choice = st.selectbox(
      '気象モデル',
      ['ECMWF (IFS)', '気象庁 (JMA)', 'GFS (NOAA)', 'ICON (DWD)'],
      index=0,
  )

with col2:
  selected_date = st.date_input('日付', now_jst.date())

with col3:
  # 現在時刻の時間（13:13 なら 13）を基準にリストを作成
  current_hour = now_jst.hour

  # 現在の時間を先頭にして24時間分を順番に並べる
  all_hours = [(current_hour + i) % 24 for i in range(24)]
  hours_list = [f'{h:02d}:00' for h in all_hours]

  selected_hour_str = st.selectbox(
      '時刻 (JST)',
      hours_list,
      index=0,  # 先頭（＝現在の時間）を初期選択
  )
  selected_hour = int(selected_hour_str.split(':')[0])

with col4:
  st.write(' ')
  st.write(' ')
  if st.session_state.pdf_data is not None:
    st.download_button(
        label='📄 PDFをダウンロード',
        data=st.session_state.pdf_data,
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

# --- 2行目：データ取得＆予想風を作成 ボタン ---
st.write('')
if st.button('データ取得＆予想風を作成', type='primary'):
  with st.spinner(f'{model_choice} からデータを取得&表を作成中...'):
    geojson_data = get_japan_geojson()
    all_location_data = {}

    for loc in LOCATIONS_CONFIG:
      try:
        data_json = fetch_weather_data(loc['lat'], loc['lon'], model_choice)
        rows = process_location_data(data_json, target_datetime)
        all_location_data[loc['name']] = rows
      except Exception as e:
        st.error(f"{loc['name']} の取得失敗: {e}")

    # カラー図の描画
    fig = generate_map_figure(
        all_location_data, geojson_data, target_datetime, model_choice
    )

    # PDF用バッファ生成
    pdf_buffer = io.BytesIO()
    fig.savefig(
        pdf_buffer,
        format='pdf',
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    pdf_buffer.seek(0)

    # ファイル名用モデルタグの対応付け
    model_tag_dict = {
        'ECMWF (IFS)': 'ECMWF',
        '気象庁 (JMA)': 'JMA',
        'GFS (NOAA)': 'GFS',
        'ICON (DWD)': 'ICON',
    }
    model_tag = model_tag_dict.get(model_choice, 'MODEL')
    filename = f"WindsAloft_{model_tag}_{selected_date.strftime('%Y%m%d')}_{selected_hour:02d}00.pdf"

    # ボタンを押した現在日時（JST）を取得
    executed_at = datetime.now(jst).strftime('%Y年%m月%d日 %H:%M JST')

    # セッションに保存してボタンをアクティブ化
    st.session_state.pdf_data = pdf_buffer.getvalue()
    st.session_state.pdf_filename = filename
    st.session_state.fig = fig
    st.session_state.info_text = (
        f'取得日時: <b>{executed_at}</b> ／ 対象日時:'
        f' <b>{selected_date.strftime("%Y年%m月%d日")}'
        f' {selected_hour:02d}:00 JST</b> （モデル: <b>{model_choice}</b>）'
    )

    st.rerun()

# --- 3行目：生成結果の表示 ---
if st.session_state.fig is not None:
  st.markdown(
      f"""
    <div style="
        display: inline-block;
        background-color: #E6F4EA;
        color: #137333;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 14px;
        margin-bottom: 12px;
        border: 1px solid #CEEAD6;
    ">
        {st.session_state.info_text}
    </div>
    """,
      unsafe_allow_html=True,
  )
  st.pyplot(st.session_state.fig, use_container_width=True)
