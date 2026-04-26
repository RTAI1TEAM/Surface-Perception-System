from flask import Flask, render_template, jsonify, request
import pymysql
import os

app = Flask(__name__)

def get_db():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        charset=os.getenv('DB_CHARSET')
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sensor_logs')
def sensor_logs():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM sensor_logs ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    db.close()
    return jsonify(rows)

@app.route('/api/robot_path')
def robot_path():
    db = get_db()
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute(
        """
        SELECT
            rp.point_id,
            rp.sequence_no,
            rp.pos_x,
            rp.pos_y,
            rp.area_type,
            rp.surface_type,
            irf.surface AS indoor_feature_label,
            CASE
                WHEN orf.source_label = 1 THEN 'pothole'
                WHEN orf.source_label = 0 THEN 'normal_road'
                ELSE NULL
            END AS outdoor_feature_label
        FROM route_points rp
        INNER JOIN routes r ON r.route_id = rp.route_id
        LEFT JOIN indoor_route_features irf ON irf.point_id = rp.point_id
        LEFT JOIN outdoor_route_features orf ON orf.point_id = rp.point_id
        WHERE r.is_active = 1 AND rp.is_active = 1
        ORDER BY rp.route_id, rp.sequence_no
        """
    )
    rows = cursor.fetchall()
    db.close()
    return jsonify([
        {
            'point_id': row['point_id'],
            'sequence_no': row['sequence_no'],
            'x': float(row['pos_x']),
            'y': float(row['pos_y']),
            'area_type': row['area_type'],
            'surface_type': row['surface_type'],
            'feature_label': row['indoor_feature_label'] if row['area_type'] == 'Indoor' else row['outdoor_feature_label']
        }
        for row in rows
    ])

@app.route('/api/update_position', methods=['POST'])
def update_position():
    data = request.json
    db = get_db()
    cursor = db.cursor()
    # cursor.execute(
    #     "INSERT INTO sensor_logs (pos_x, pos_y) VALUES (%s, %s)",
    #     (data['x'], data['y'])
    # )
    # log_id = cursor.lastrowid
    # cursor.execute(
    #     "INSERT INTO detection_results (log_id, area_type, surface_type, confidence) VALUES (%s, %s, %s, %s)",
    #     (log_id, data.get('area_type', 'Outdoor'), data.get('surface_type', 'asphalt'), 0.95)
    # )
    db.commit()
    db.close()
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
