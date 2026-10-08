from datetime import datetime, timedelta, timezone
import glob
import io
import math
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# サーバーフォントの読み込み
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
    {'hpa': '1000', 'ft': 360},
    {'hpa': '950', 'ft': 1800},
    {'hpa': '900', 'ft': 3100},
    {'hpa': '850', 'ft': 4780},
    {'hpa': '700', 'ft': 9880},
]

TARGET_ALTITUDES = [0, 2000, 3000, 5000, 6400]
MAG_VARIATION = 8.0


# 地図データのキャッシュ（24時間有効）
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


def interpolate_angle(x_target, x_pts, deg_pts):
  rad_pts = np.radians(deg_pts)
  sin_pts = np.sin(rad_pts)
  cos_pts = np.cos(rad_pts)

  sin_interp = float(np.interp(x_target, x_pts, sin_pts))
  cos_interp = float(np.interp(x_target, x_pts, cos_pts))

  deg_interp = math.degrees(math.atan2(sin_interp, cos_interp)) % 360
  return deg_interp


# APIデータ取得を全ユーザー間で共有（1時間キャッシュ）
@st.cache_data(ttl=3600, show_spinner=False)
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

  last_error
