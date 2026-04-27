import math
import pandas as pd
import pymysql

# =========================
# 1. 사용자 설정
# =========================
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(os.path.join(project_root, "SSP", ".env"))
load_dotenv(dotenv_path=os.path.join(project_root,  ".env"))

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': os.getenv('DB_NAME'),
    'charset': os.getenv('DB_CHARSET')
}

ROUTE_ID = 3

INDOOR_CSV = os.path.join(project_root, "SSP", "data", "processed", "indoor", "indoor_train_features.csv")
OUTDOOR_CSV = os.path.join(project_root, "SSP", "data", "processed", "pothole", "test_v3_s2.csv")

INDOOR_OUTPUT_CSV = os.path.join(project_root, "SSP", "data", "processed", "indoor", "indoor_route_features_ready.csv")
OUTDOOR_OUTPUT_CSV = os.path.join(project_root, "SSP", "data", "processed", "pothole", "outdoor_route_features_ready.csv")

# 실내 CSV의 표면 컬럼명
INDOOR_SURFACE_COL = "surface"

# 실외 CSV의 표면 컬럼명
# 실외 CSV에 표면 컬럼이 없으면 None으로 두세요.
OUTDOOR_SURFACE_COL = None
# 예: OUTDOOR_SURFACE_COL = "surface_type"

# DB 테이블에 넣을 때 원본 라벨 컬럼명을 바꾸고 싶으면 사용
RENAME_OUTDOOR_LABEL_TO_SOURCE_LABEL = True

# surface 이름 매핑이 필요하면 여기에 추가
# 왼쪽: route_points.surface_type
# 오른쪽: feature csv 안의 surface 값
SURFACE_MAP = {
    "soft_tiles": "soft_tiles",
    "soft_pvc": "soft_pvc",
    "fine_concrete": "fine_concrete",
    "concrete": "concrete",
    "carpet": "carpet",
    "wood": "wood",
    "tiled": "tiled",
}

# CSV 저장만 할지, DB insert까지 할지
WRITE_CSV_ONLY = False

# DB insert 대상 테이블명
INDOOR_TABLE = "indoor_route_features"
OUTDOOR_TABLE = "outdoor_route_features"


# =========================
# 2. 유틸
# =========================
def normalize_surface(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def mapped_surface(surface):
    s = normalize_surface(surface)
    return normalize_surface(SURFACE_MAP.get(s, s))

def evenly_pick_indices(total_count, target_count):
    if target_count <= 0:
        return []
    if total_count == target_count:
        return list(range(total_count))
    if target_count == 1:
        return [0]
    step = (total_count - 1) / (target_count - 1)
    return [round(i * step) for i in range(target_count)]

def assign_features_to_points(points_df, features_df, feature_surface_col=None, default_surface=None):
    """
    points_df: route_points subset (same area_type)
    features_df: feature csv subset (same area_type)
    feature_surface_col:
        - 실내: 보통 'surface'
        - 실외: 없으면 None
    default_surface:
        - feature_surface_col이 없을 때 강제로 붙일 값
    """
    assignments = []

    work_points = points_df.copy().sort_values("sequence_no").reset_index(drop=True)
    work_points["surface_norm"] = work_points["surface_type"].map(mapped_surface)

    work_features = features_df.copy().reset_index(drop=True)
    work_features["source_row_no"] = work_features.index + 1

    if feature_surface_col and feature_surface_col in work_features.columns:
        work_features["surface_norm"] = work_features[feature_surface_col].map(mapped_surface)
    else:
        work_features["surface_norm"] = normalize_surface(default_surface)

    route_surfaces = work_points["surface_norm"].dropna().tolist()
    unique_route_surfaces_in_order = []
    seen = set()
    for s in route_surfaces:
        if s not in seen:
            unique_route_surfaces_in_order.append(s)
            seen.add(s)

    for route_surface in unique_route_surfaces_in_order:
        point_subset = work_points[work_points["surface_norm"] == route_surface].copy()
        feature_subset = work_features[work_features["surface_norm"] == route_surface].copy()

        if feature_subset.empty:
            raise ValueError(
                f"surface 매칭 실패: route_points에는 '{route_surface}'가 있는데 "
                f"feature CSV에는 해당 surface 데이터가 없습니다."
            )

        point_count = len(point_subset)
        feature_count = len(feature_subset)

        if feature_count >= point_count:
            selected_idx = evenly_pick_indices(feature_count, point_count)
            chosen_features = feature_subset.iloc[selected_idx].reset_index(drop=True)
        else:
            repeats = math.ceil(point_count / feature_count)
            chosen_features = pd.concat([feature_subset] * repeats, ignore_index=True).iloc[:point_count].copy()

        point_subset = point_subset.reset_index(drop=True)

        merged = pd.concat(
            [
                point_subset[["point_id", "sequence_no", "area_type", "surface_type"]],
                chosen_features.reset_index(drop=True)
            ],
            axis=1
        )
        assignments.append(merged)

    if not assignments:
        return pd.DataFrame()

    result = pd.concat(assignments, ignore_index=True)
    result = result.sort_values("sequence_no").reset_index(drop=True)
    return result

def get_route_points(route_id):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        sql = """
            SELECT point_id, route_id, sequence_no, pos_x, pos_y, area_type, surface_type
            FROM route_points
            WHERE route_id = %s
            ORDER BY sequence_no
        """
        return pd.read_sql(sql, conn, params=[route_id])
    finally:
        conn.close()

def write_df_to_mysql(df, table_name):
    if df.empty:
        print(f"[SKIP] {table_name}: empty dataframe")
        return

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cols = list(df.columns)
            placeholders = ", ".join(["%s"] * len(cols))
            col_sql = ", ".join([f"`{c}`" for c in cols])

            sql = f"INSERT INTO `{table_name}` ({col_sql}) VALUES ({placeholders})"
            rows = [tuple(None if pd.isna(v) else v for v in row) for row in df[cols].itertuples(index=False, name=None)]
            cur.executemany(sql, rows)

        conn.commit()
        print(f"[OK] inserted {len(df)} rows into {table_name}")
    finally:
        conn.close()


# =========================
# 3. 실행
# =========================
route_points = get_route_points(ROUTE_ID)

if route_points.empty:
    raise ValueError(f"route_id={ROUTE_ID} 에 해당하는 route_points가 없습니다.")

indoor_points = route_points[route_points["area_type"].str.lower() == "indoor"].copy()
outdoor_points = route_points[route_points["area_type"].str.lower() == "outdoor"].copy()
indoor_features = pd.read_csv(INDOOR_CSV)
outdoor_features = pd.read_csv(OUTDOOR_CSV)

# 실내 매칭
if INDOOR_SURFACE_COL not in indoor_features.columns:
    raise ValueError(f"실내 CSV에 '{INDOOR_SURFACE_COL}' 컬럼이 없습니다.")

indoor_ready = assign_features_to_points(
    points_df=indoor_points,
    features_df=indoor_features,
    feature_surface_col=INDOOR_SURFACE_COL
)
print(indoor_ready[:3])
# 실외 매칭
if OUTDOOR_SURFACE_COL is not None and OUTDOOR_SURFACE_COL not in outdoor_features.columns:
    raise ValueError(f"실외 CSV에 '{OUTDOOR_SURFACE_COL}' 컬럼이 없습니다.")

default_outdoor_surface = None
if OUTDOOR_SURFACE_COL is None:
    outdoor_route_surfaces = outdoor_points["surface_type"].map(mapped_surface).dropna().unique().tolist()
    if len(outdoor_route_surfaces) != 1:
        raise ValueError(
            "실외 CSV에 surface 컬럼이 없는데, route_points의 Outdoor surface_type이 여러 종류입니다. "
            "OUTDOOR_SURFACE_COL을 지정하거나 실외 CSV에 surface_type 컬럼을 추가하세요."
        )
    default_outdoor_surface = outdoor_route_surfaces[0]

outdoor_ready = assign_features_to_points(
    points_df=outdoor_points,
    features_df=outdoor_features,
    feature_surface_col=OUTDOOR_SURFACE_COL,
    default_surface=default_outdoor_surface
)

# DB insert용 정리
drop_meta_cols = ["sequence_no", "area_type", "surface_type", "surface_norm"]

for col in drop_meta_cols:
    if col in indoor_ready.columns:
        indoor_ready = indoor_ready.drop(columns=[col])
    if col in outdoor_ready.columns:
        outdoor_ready = outdoor_ready.drop(columns=[col])

# outdoor label 컬럼 rename 옵션
if RENAME_OUTDOOR_LABEL_TO_SOURCE_LABEL and "label" in outdoor_ready.columns:
    outdoor_ready = outdoor_ready.rename(columns={"label": "source_label"})

# 중복 point_id 체크
if indoor_ready["point_id"].duplicated().any():
    raise ValueError("indoor_ready에 중복 point_id가 있습니다.")

if outdoor_ready["point_id"].duplicated().any():
    raise ValueError("outdoor_ready에 중복 point_id가 있습니다.")

print("\n[Indoor mapping preview]")
print(indoor_ready.head())

print("\n[Outdoor mapping preview]")
print(outdoor_ready.head())

print("\n[Count summary]")
print(f"Indoor route points  : {len(indoor_points)}")
print(f"Indoor feature rows  : {len(indoor_features)}")
print(f"Indoor mapped rows   : {len(indoor_ready)}")
print(f"Outdoor route points : {len(outdoor_points)}")
print(f"Outdoor feature rows : {len(outdoor_features)}")
print(f"Outdoor mapped rows  : {len(outdoor_ready)}")

# CSV 저장
indoor_ready.to_csv(INDOOR_OUTPUT_CSV, index=False, encoding="utf-8-sig")
outdoor_ready.to_csv(OUTDOOR_OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n[OK] indoor csv saved -> {INDOOR_OUTPUT_CSV}")
print(f"[OK] outdoor csv saved -> {OUTDOOR_OUTPUT_CSV}")

# DB에 직접 넣고 싶으면 WRITE_CSV_ONLY=False
if not WRITE_CSV_ONLY:
    write_df_to_mysql(indoor_ready, INDOOR_TABLE)
    write_df_to_mysql(outdoor_ready, OUTDOOR_TABLE)
