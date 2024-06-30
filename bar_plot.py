import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import unicodedata
import numpy as np

# 한글 문자 너비 계산 함수
def get_display_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)

# 한글 폰트 설정 (예: 맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 예시 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

# 엑셀 파일 경로
f_special = r'경로\special.xlsx'
f_origin = r'경로\original.xlsx'
sp_df = pd.read_excel(f_special)
or_df = pd.read_excel(f_origin)

# 열 선택
sp_gu_df = sp_df.groupby('행정구').sum()
sp_sum_df = sp_gu_df[['특수_학급수','특수_학생수']].reset_index()
or_gu_df = or_df.groupby('행정구').sum()
or_sum_df = or_gu_df[['학생수_총계_계']].reset_index()
sp_sum_df['특수_학생수/특수_학급_수'] = sp_sum_df['특수_학생수'] / sp_sum_df['특수_학급수']
sp_sum_df['총_학생수'] = or_sum_df['학생수_총계_계']
sp_sum_df['총_학생수/특수_학급_수'] = sp_sum_df['총_학생수'] / sp_sum_df['특수_학급수']
sp_sum_df['총_학생수/특수_학급_수'] = sp_sum_df['총_학생수/특수_학급_수'].fillna(0)  # 0으로 나누는 경우 0으로 대체

# 데이터프레임 정렬 (총_학생수/특수_학급_수 기준, 내림차순)
sp_sum_df = sp_sum_df.sort_values(by='총_학생수/특수_학급_수', ascending=False)

# 각 열 이름과 값의 최대 길이(실제 너비) 계산
max_lengths = {col: max(sp_sum_df[col].astype(str).map(get_display_width).max(), get_display_width(col)) for col in sp_sum_df.columns}

# 각 열의 데이터를 가운데 정렬된 문자열로 변환
for col in sp_sum_df.columns:
    sp_sum_df[col] = sp_sum_df[col].astype(str).map(lambda x: x.center(max_lengths[col]))

# 열 이름을 가운데 정렬된 이름으로 변경
centered_headers = [col.center(max_lengths[col]) for col in sp_sum_df.columns]

# tabulate를 사용하여 포맷팅된 데이터프레임 출력 (tablefmt='psal' 사용)
#print(tabulate(sp_sum_df.values.tolist(), headers=centered_headers, tablefmt='psql'))

# 그래프 그리기
plt.figure(figsize=(12, 8))
sp_sum_df_plot = sp_sum_df.copy()
sp_sum_df_plot[['특수_학급수', '특수_학생수', '총_학생수', '총_학생수/특수_학급_수']] = sp_sum_df_plot[['특수_학급수', '특수_학생수', '총_학생수', '총_학생수/특수_학급_수']].apply(pd.to_numeric)

# melt를 사용해 데이터프레임을 길게 변환
sp_sum_df_melted = sp_sum_df_plot.melt(id_vars=['행정구'], value_vars=['특수_학급수', '특수_학생수', '총_학생수', '총_학생수/특수_학급_수'])

# 로그 스케일 적용
sp_sum_df_melted['value'] = sp_sum_df_melted['value'].apply(lambda x: np.log1p(x) if x > 0 else 0)

# 막대그래프 그리기
sns.barplot(data=sp_sum_df_melted, x='행정구', y='value', hue='variable')
plt.xticks(rotation=90)
plt.title('각 행정구별 데이터')
plt.ylabel('값')
plt.xlabel('행정구')
plt.legend(title='데이터 종류')

plt.tight_layout()
plt.show()