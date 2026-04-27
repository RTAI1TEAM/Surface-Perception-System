from flask import Flask, jsonify, render_template, request

from services.db import get_db
from services.prediction_service import get_robot_path_points, process_point_prediction, _select_prediction_log


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sensor_logs")
def sensor_logs():
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM sensor_logs ORDER BY timestamp DESC LIMIT 10")
            rows = cursor.fetchall()
    finally:
        db.close()
    return jsonify(rows)


@app.route("/api/robot_path")
def robot_path():
    return jsonify(get_robot_path_points())


@app.route("/api/update_position", methods=["POST"])
def update_position():
    return process_point_prediction(request.json or {})


@app.route("/api/fetch_pred")
def fetch_pred():
    return _select_prediction_log()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
