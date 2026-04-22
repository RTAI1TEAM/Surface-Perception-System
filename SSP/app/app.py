from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # 메인 대시보드 페이지 렌더링
    return render_template('index.html')

if __name__ == '__main__':
    # 디버그 모드로 실행 (포트 5000)
    app.run(debug=True, host='0.0.0.0', port=5000)