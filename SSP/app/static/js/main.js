document.addEventListener("DOMContentLoaded", function() {
    
    // 1. 하단 센서 파형 그래프 (X, Y, Z 가속도) - 빈 데이터로 초기화
    const sensorCtx = document.getElementById('sensorChart').getContext('2d');
    
    // 이 chart 객체를 변수(sensorChart)에 담아두면, 
    // 나중에 실제 데이터가 들어올 때 sensorChart.update() 로 화면을 갱신할 수 있습니다.
    const sensorChart = new Chart(sensorCtx, {
        type: 'line',
        data: {
            labels: [], // 실제 시간이 들어갈 빈 배열
            datasets: [
                { label: 'X Axis', data: [], borderColor: '#003f5c', tension: 0.4 },
                { label: 'Y Axis', data: [], borderColor: '#58508d', tension: 0.4 },
                { label: 'Z Axis', data: [], borderColor: '#bc5090', tension: 0.4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', align: 'end' } },
            scales: {
                y: { min: -60, max: 60 } // 나중에 실제 데이터의 범위에 맞춰 수정하세요
            }
        }
    });
});