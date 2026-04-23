from flask import Flask, render_template, jsonify
import pymysql

app = Flask(__name__)

def get_db():
    return pymysql.connect(
        host='localhost',
        user='factory',
        password='1234',
        database='factory_twin',
        charset='utf8'
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
    cursor = db.cursor()
    cursor.execute("SELECT pos_x, pos_y FROM sensor_logs ORDER BY log_id")
    rows = cursor.fetchall()
    db.close()
    return jsonify([{'x': row[0], 'y': row[1]} for row in rows])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)