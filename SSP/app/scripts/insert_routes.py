import pymysql
from math import sqrt
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(os.path.join(project_root, "SSP", ".env"))
load_dotenv(dotenv_path=os.path.join(project_root,  ".env"))

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': os.getenv('DB_NAME'),
    'charset': os.getenv('DB_CHARSET')
}

ROUTE_NAME = "main_js_expanded_route"
ROUTE_DESCRIPTION = "Expanded route imported from main.js"

routePoints = [
    [207, 1141, 'Outdoor', 'asphalt'],  # 출발점
    [207, 1206, 'Outdoor', 'asphalt'],
    [276, 1206, 'Outdoor', 'asphalt'],  # 크랙
    [276, 1141, 'Outdoor', 'asphalt'],  # 주차장 앞 도로
    [400, 1141, 'Outdoor', 'asphalt'],
    [400, 1293, 'Outdoor', 'asphalt'],  # 주차장
    [469, 1293, 'Outdoor', 'asphalt'],
    [469, 1330, 'Outdoor', 'asphalt'],  # 주차장 안쪽
    [327, 1330, 'Outdoor', 'asphalt'],
    [327, 1357, 'Outdoor', 'asphalt'],  # 주차장 안쪽2
    [327, 1293, 'Outdoor', 'asphalt'],
    [400, 1293, 'Outdoor', 'asphalt'],
    [400, 1141, 'Outdoor', 'asphalt'],  # 주차장 앞 도로
    [704, 1141, 'Outdoor', 'asphalt'],  # 외부창고 방향 직진
    [704, 1233, 'Outdoor', 'asphalt'],  # 외부창고 문
    [639, 1233, 'Outdoor', 'asphalt'],
    [639, 1330, 'Outdoor', 'asphalt'],  # 외부창고 안1
    [777, 1330, 'Outdoor', 'asphalt'],
    [777, 1321, 'Outdoor', 'asphalt'],  # 외부창고 안2
    [777, 1233, 'Outdoor', 'asphalt'],
    [704, 1233, 'Outdoor', 'asphalt'],  # 외부창고 문
    [704, 1141, 'Outdoor', 'asphalt'],  # 외부창고 앞 도로
    [538, 1141, 'Outdoor', 'asphalt'],  # 출입구 앞 도로
    [538, 837,  'Indoor',  'soft_tiles'], # A-5
    [184, 837,  'Indoor',  'wood'],
    [184, 833,  'Indoor',  'wood'],      # A-4
    [184, 612,  'Indoor',  'carpet'],
    [193, 612,  'Indoor',  'carpet'],    # A-3 뒷문
    [524, 612,  'Indoor',  'carpet'],
    [524, 607,  'Indoor',  'carpet'],    # A-3 앞문
    [538, 382,  'Indoor',  'tiled'],     # A-2 앞
    [179, 382,  'Indoor',  'tiled'],     # A-2 안쪽
    [179, 156,  'Indoor',  'concrete'],
    [184, 156,  'Indoor',  'concrete'],  # A-1 안쪽
    [796, 156,  'Indoor',  'concrete'],
    [796, 193,  'Indoor',  'concrete'],  # B-1
    [796, 483,  'Indoor',  'soft_pvc'],
    [759, 483,  'Indoor',  'soft_pvc'],  # B-2 안쪽
    [538, 483,  'Indoor',  'soft_pvc'],
    [538, 479,  'Indoor',  'soft_pvc'],  # B-2 앞
    [538, 787,  'Indoor',  'fine_concrete'], # B-3 앞 복도
    [630, 805,  'Indoor',  'fine_concrete'], # B-3 출입구
    [768, 805,  'Indoor',  'fine_concrete'],
    [768, 672,  'Indoor',  'fine_concrete'], # B-3 왼쪽 안쪽
    [768, 948,  'Indoor',  'fine_concrete'], # B-3 오른쪽 안쪽
    [768, 805,  'Indoor',  'fine_concrete'],
    [630, 805,  'Indoor',  'fine_concrete'], # B-3 출입구
    [538, 805,  'Indoor',  'fine_concrete'],
    [538, 1141, 'Outdoor', 'asphalt'],  # 출입구
    [207, 1141, 'Outdoor', 'asphalt'],  # 출발점 방향
    [207, 1353, 'Outdoor', 'asphalt'],
    [212, 1353, 'Outdoor', 'asphalt'],  # 크랙2 앞
    [175, 1353, 'Outdoor', 'asphalt'],
    [175, 1348, 'Outdoor', 'asphalt'],  # 크랙2
    [175, 1141, 'Outdoor', 'asphalt'],
    [207, 1141, 'Outdoor', 'asphalt'],  # 출발점 복귀
]

def interpolate(p1, p2, steps):
    points = []
    for i in range(1, steps + 1):
        y = p1[0] + (p2[0] - p1[0]) * (i / steps)
        x = p1[1] + (p2[1] - p1[1]) * (i / steps)
        points.append({
            "pos_x": x,
            "pos_y": y,
            "area_type": p2[2],
            "surface_type": p2[3],
        })
    return points

def build_expanded_route(route_points):
    expanded = [{
        "pos_y": route_points[0][0],
        "pos_x": route_points[0][1],
        "area_type": route_points[0][2],
        "surface_type": route_points[0][3],
    }]

    for i in range(len(route_points) - 1):
        p1 = route_points[i]
        p2 = route_points[i + 1]
        dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        steps = max(1, round(dist / 20))
        expanded.extend(interpolate(p1, p2, steps))

    return expanded

def insert_route():
    expanded_route = build_expanded_route(routePoints)

    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO routes (route_name, description, is_active, loop_enabled)
                VALUES (%s, %s, 1, 1)
                """,
                (ROUTE_NAME, ROUTE_DESCRIPTION)
            )
            route_id = cur.lastrowid

            sql = """
                INSERT INTO route_points
                (route_id, sequence_no, pos_x, pos_y, area_type, surface_type, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, 1)
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
                    )
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
