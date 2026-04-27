import os
from math import sqrt

import pymysql
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME"),
    "charset": os.getenv("DB_CHARSET"),
}

ROUTE_NAME = "main_js_expanded_route"
ROUTE_DESCRIPTION = "Expanded route imported from main.js"

# Waypoint format is [y, x], matching the existing route/map coordinate usage.
WAYPOINTS = [
    [182, 1160],
    [182, 1390],
    [240, 1160],
    [276, 1206],
    [276, 1141],
    [400, 1141],
    [400, 1293],
    [469, 1293],
    [469, 1330],
    [327, 1330],
    [327, 1357],
    [327, 1293],
    [400, 1293],
    [400, 1141],
    [704, 1141],
    [704, 1233],
    [639, 1233],
    [639, 1330],
    [777, 1330],
    [777, 1321],
    [777, 1233],
    [704, 1233],
    [704, 1141],
    [538, 1141],
    [538, 837],
    [184, 837],
    [184, 833],
    [184, 612],
    [193, 612],
    [524, 612],
    [524, 607],
    [538, 382],
    [179, 382],
    [179, 156],
    [184, 156],
    [796, 156],
    [796, 193],
    [796, 483],
    [759, 483],
    [538, 483],
    [538, 479],
    [538, 787],
    [630, 805],
    [768, 805],
    [768, 672],
    [768, 948],
    [768, 805],
    [630, 805],
    [538, 805],
    [538, 1141],
    [207, 1141],
    [207, 1353],
    [212, 1353],
    [175, 1353],
    [175, 1348],
    [175, 1141],
    [207, 1141],
]

# Rectangles are (left, top, right, bottom), using normal map x/y coordinates.
# More specific zones must come before broad parent zones.
AREA_RULES = [
    ((55, 90, 265, 460), "Indoor", "concrete"),
    ((280, 90, 490, 460), "Indoor", "tiled"),
    ((505, 90, 715, 460), "Indoor", "carpet"),
    ((730, 90, 955, 260), "Indoor", "wood"),
    ((730, 280, 955, 460), "Indoor", "soft_tiles"),
    
    # 세로 통로 및 중간 연결 구간 (빈틈 메움)
    ((40, 480, 970, 580), "Indoor", "tiled"), 
    
    ((55, 630, 325, 930), "Indoor", "concrete"),
    ((345, 630, 615, 930), "Indoor", "soft_pvc"),
    ((635, 630, 955, 930), "Indoor", "fine_concrete"),
    
    # 실외 경계 판정 (X=1000 이상부터 Outdoor)
    ((1000, 20, 1480, 980), "Outdoor", "asphalt"),
    
    # 나머지 전체 구역 커버
    ((40, 40, 970, 480), "Indoor", "concrete"),
    ((40, 580, 970, 950), "Indoor", "concrete"),
]

# Pothole areas kept from the existing route's pothole anchor points.
POTHOLE_ZONES = [
    (1162, 276, 1164, 276),
]

CRACK_ZONES = [
    (600, 175, 625, 200),
    (370, 525, 395, 545),
    (795, 620, 815, 640),
]


def in_rect(x, y, rect):
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def classify_route_point(y, x):
    for rect, area_type, surface_type in AREA_RULES:
        if in_rect(x, y, rect):
            road_condition = "normal_road"
            if area_type == "Outdoor" and any(in_rect(x, y, zone) for zone in POTHOLE_ZONES):
                road_condition = "pothole"
            elif area_type == "Indoor" and any(in_rect(x, y, zone) for zone in CRACK_ZONES):
                road_condition = "crack"
            return [y, x, area_type, surface_type, road_condition]

    return [y, x, "Outdoor", "asphalt", "normal_road"]


routePoints = [classify_route_point(y, x) for y, x in WAYPOINTS]


def interpolate(p1, p2, steps):
    points = []
    for i in range(1, steps + 1):
        y = p1[0] + (p2[0] - p1[0]) * (i / steps)
        x = p1[1] + (p2[1] - p1[1]) * (i / steps)
        _, _, area_type, surface_type, road_condition = classify_route_point(y, x)
        points.append(
            {
                "pos_x": x,
                "pos_y": y,
                "area_type": area_type,
                "surface_type": surface_type,
                "road_condition": road_condition,
            }
        )
    return points


def build_expanded_route(route_points):
    expanded = [
        {
            "pos_y": route_points[0][0],
            "pos_x": route_points[0][1],
            "area_type": route_points[0][2],
            "surface_type": route_points[0][3],
            "road_condition": route_points[0][4],
        }
    ]

    for i in range(len(route_points) - 1):
        p1 = route_points[i]
        p2 = route_points[i + 1]
        dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        steps = max(1, round(dist / 20))
        expanded.extend(interpolate(p1, p2, steps))

    return expanded


def ensure_route_points_schema(cursor):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'route_points'
          AND COLUMN_NAME = 'road_condition'
        """
    )
    if cursor.fetchone()[0] == 0:
        cursor.execute("ALTER TABLE route_points ADD COLUMN road_condition VARCHAR(30) NULL")


def insert_route():
    expanded_route = build_expanded_route(routePoints)

    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cur:
            ensure_route_points_schema(cur)
            cur.execute(
                """
                INSERT INTO routes (route_name, description, is_active, loop_enabled)
                VALUES (%s, %s, 1, 1)
                """,
                (ROUTE_NAME, ROUTE_DESCRIPTION),
            )
            route_id = cur.lastrowid

            sql = """
                INSERT INTO route_points
                (route_id, sequence_no, pos_x, pos_y, area_type, surface_type, road_condition, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
            """

            for idx, point in enumerate(expanded_route, start=1):
                cur.execute(
                    sql,
                    (
                        route_id,
                        idx,
                        point["pos_x"],
                        point["pos_y"],
                        point["area_type"],
                        point["surface_type"],
                        point["road_condition"],
                    ),
                )

        conn.commit()
        print(f"route_id={route_id} inserted")
        print(f"{len(expanded_route)} points inserted into route_points")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    insert_route()
