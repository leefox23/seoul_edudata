# 모델링 : K - Clustering + KDE

###########################################
# 지도색상 : 기본, 회색
# 학급 : 전 학교, 특수학교
# 마커 : 전 학교, 특수학교
# 모델링 : K - Clustering(3, 4, 5), KDE, 마커
###########################################

import requests
import json
import folium
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString
import geopandas as gpd
from pyproj import Transformer
import math

# 서울시 구별 경계 데이터 로드
seoul_geo = gpd.read_file('경로/서울시_시군구.geojson')

# CSV 파일 데이터 읽기
#csv_file = '경로/special_lalo2.csv' # 특수학교만 고려함
csv_file = '경로/special_lalo.csv' # 특수학급이 있는 학교 전체를 고려함

df = pd.read_csv(csv_file)

# 서울시 중심 좌표
seoul_center = [37.5665, 126.9780]

# 행정구 별 좌표
data_gu = {
    '행정구': ['강남구', '서초구', '송파구', '강동구', '마포구', '영등포구', '강서구', '양천구', '구로구', '금천구',
            '관악구', '동작구', '동대문구', '중랑구', '성북구', '강북구', '도봉구', '노원구', '은평구',
            '서대문구', '종로구', '중구', '용산구', '성동구', '광진구'],
    'Latitude': [37.5172, 37.4837, 37.5145, 37.5303, 37.5635, 37.5264, 37.5658, 37.5271, 37.4959, 37.4715,
                37.4784, 37.5121, 37.6066, 37.5953, 37.6065, 37.6415, 37.6688, 37.6263, 37.6176,
                37.5791, 37.5723, 37.5639, 37.532, 37.5515, 37.5425],
    'Longitude': [127.0473, 127.0322, 127.1072, 127.1238, 126.9038, 126.8962, 126.8495, 126.8567, 126.886, 126.8515,
                  126.9516, 126.9395, 127.0927, 127.093, 127.0927, 127.0167, 127.0471, 127.056, 126.9278,
                  126.9367, 126.9794, 126.9975, 126.9646, 127.0403, 127.0832]
}
df_gu = pd.DataFrame(data_gu)

# 지도 생성
seoul_map = folium.Map(location=seoul_center, zoom_start=12)
#seoul_map = folium.Map(location=seoul_center, zoom_start=12, tiles='cartodbpositron')

# 서울시 구별 경계 표시
def style_function(feature):
    return {
        'fillColor': '#ffffff',
        'color': '#000000',
        'weight': 1,
        'opacity': 1
    }

folium.GeoJson(
    seoul_geo.to_json(),
    name='지역구',
    style_function=style_function
).add_to(seoul_map)

# 로그 스케일 함수 정의
def log_scale(value):
    if value > 0:
        return math.log(value + 1)  # 0일 때 오류를 방지하기 위해 1을 더해줍니다.
    else:
        return 0

# 특수 학급 수를 로그 스케일로 변환하여 저장
df['로그_특수_학급수'] = df['특수_학급수'].apply(log_scale)

# 25개의 선형적인 색상 생성
cmap = plt.get_cmap('hsv', len(df_gu))
colors = [cmap(i) for i in range(cmap.N)]

# RGB to hex 변환 함수
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# 구별로 색상 매핑
color_dict = {df_gu['행정구'][i]: rgb_to_hex(colors[i]) for i in range(len(df_gu))}


##############################↓↓K-clustering 코드↓↓##############################
# 최적의 클러스터링 결과 선택
best_kmeans = None
best_score = -1
num_clusters = 5

for _ in range(100):  # 100번 반복
    kmeans = KMeans(n_clusters=num_clusters, random_state=None)
    kmeans.fit(df[['Latitude', 'Longitude']].values)
    score = silhouette_score(df[['Latitude', 'Longitude']].values, kmeans.labels_)
    if score > best_score:
        best_score = score
        best_kmeans = kmeans

# 최적의 클러스터링 결과 사용
labels = best_kmeans.labels_
centroids = best_kmeans.cluster_centers_

# 클러스터 중심을 마커로 표시
for idx, centroid in enumerate(centroids):
    folium.Marker(location=[centroid[0], centroid[1]], popup=f'Cluster {idx}', icon=folium.Icon(color='red')).add_to(seoul_map)

# Voronoi 경계선을 지도에 추가
def add_voronoi_polygons(vor, boundary, m):
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            points = vor.vertices[simplex]
            # 투영된 좌표를 위경도로 다시 변환
            points_latlon = [transformer.transform(x, y, direction="INVERSE") for x, y in points]
            line = LineString(points_latlon)
            
            # 서울시 경계와 교차하는 부분만 추출
            clipped_line = line.intersection(boundary)
            
            if not clipped_line.is_empty:
                if isinstance(clipped_line, MultiLineString):
                    for l in clipped_line.geoms:
                        folium.PolyLine(locations=[(y, x) for x, y in l.coords], color='blue', weight=2).add_to(m)
                elif isinstance(clipped_line, LineString):
                    folium.PolyLine(locations=[(y, x) for x, y in clipped_line.coords], color='blue', weight=2).add_to(m)


# 좌표계 변환을 위한 Transformer 생성
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# 중심점 좌표 변환
centroids_proj = np.array([transformer.transform(lon, lat) for lat, lon in centroids])

# 서울 경계 확장
seoul_bounds = seoul_geo.total_bounds
x_min, y_min, x_max, y_max = seoul_bounds
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1
extended_bounds = (x_min - x_padding, y_min - y_padding, x_max + x_padding, y_max + y_padding)

# 확장된 경계를 투영 좌표계로 변환
extended_bounds_proj = np.array([
    transformer.transform(extended_bounds[0], extended_bounds[1]),
    transformer.transform(extended_bounds[2], extended_bounds[3])
]).flatten()

# Voronoi 다이어그램 생성 (확장된 범위 사용)
vor = Voronoi(np.vstack([centroids_proj, [
    [extended_bounds_proj[0], extended_bounds_proj[1]],
    [extended_bounds_proj[0], extended_bounds_proj[3]],
    [extended_bounds_proj[2], extended_bounds_proj[1]],
    [extended_bounds_proj[2], extended_bounds_proj[3]]
]]))

# Voronoi 경계선 추가
try:
    print("Starting to add Voronoi polygons")
    add_voronoi_polygons(vor, seoul_geo.geometry.union_all(), seoul_map)
    print("Finished adding Voronoi polygons")
except Exception as e:
    print(f"Error occurred while adding Voronoi polygons: {e}")
    import traceback
    traceback.print_exc()

##############################↑↑K-clustering 코드↑↑##############################


##################################↓↓KDE 코드↓↓##################################
# KDE 적용 (전체 데이터에 대해)
kde = KernelDensity(bandwidth=0.01)
kde.fit(df[['Latitude', 'Longitude']].values)

# 서울 경계 내에서 밀도 추정할 좌표 생성
seoul_bbox = [126.683040, 37.425853, 127.183505, 37.701749]  # [min_lon, min_lat, max_lon, max_lat]
x_min, x_max = seoul_bbox[0], seoul_bbox[2]  # Longitude
y_min, y_max = seoul_bbox[1], seoul_bbox[3]  # Latitude
x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([y.ravel(), x.ravel()]).T
density = np.exp(kde.score_samples(positions))
density = density.reshape(100, 100)

# 밀도 시각화
heatmap_layer = folium.raster_layers.ImageOverlay(
    np.rot90(density),
    bounds=[[y_min, x_min], [y_max, x_max]],
    opacity=0.4,
    colormap=lambda x: (1, 0, 0, x),
)
seoul_map.add_child(heatmap_layer)
##################################↑↑KDE 코드↑↑##################################


CircleMarkerCheck = False # 마커 생성 코드 여부 확인 변수
################################↓↓마커 생성 코드↓↓################################
# 구별로 마커 생성

for idx, row in df.iterrows():
    address = row['학교명']
    administrative_district = row['행정구']
    coords = (row['Latitude'], row['Longitude'])
    log_special_classrooms = row['로그_특수_학급수']*6
    if pd.notna(coords[0]) and pd.notna(coords[1]) and administrative_district in color_dict:  # 유효한 좌표와 색상이 매핑된 경우
        icon_color = color_dict[administrative_district]
        folium.CircleMarker(location=coords, radius=log_special_classrooms, popup=address, color=icon_color, fill=True, fill_color=icon_color).add_to(seoul_map)
        CircleMarkerCheck = True
################################↑↑마커 생성 코드↑↑################################


###############################↓↓파일명 생성 코드↓↓###############################
# 파일명 생성에 필요한 정보들
map_color = '회색' if 'cartodbpositron' in str(seoul_map) else '기본'
school_type = '특수' if csv_file.endswith('special_lalo2.csv') else '전체'
kde_applied = False  # KDE와 마커 모두 적용 안되는 경우

# 파일명 생성 함수
def create_filename(map_color, school_type, num_clusters, kde_applied):
    filename = ""
    
    if map_color == '회색':
        filename += "회색_"
    else:
        filename += "기본_"
    
    if school_type == '특수':
        filename += "특수_"
    else:
        filename += "전체_"
    
    filename += f"KC{num_clusters}_"
    
    if 'kde' in globals() and kde is not None:
        filename += "KDE.html"
    elif CircleMarkerCheck == True:
        filename += "마커.html"
    else:
        filename += ".html"
    
    return filename

# 파일명 생성
output_filename = create_filename(map_color, school_type, num_clusters, kde_applied)
###############################↑↑파일명 생성 코드↑↑###############################


# 현재 작업 디렉토리 출력
current_directory = os.getcwd()
print(f"현재 작업 디렉토리: {current_directory}")

# 지도 저장
output_map = os.path.join(current_directory, output_filename)
seoul_map.save(output_map)
print(f"경계선이 표시된 지도를 생성했습니다. 파일 '{output_map}'을 확인하세요.")
