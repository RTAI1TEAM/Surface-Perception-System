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
// 1. 지도 초기화
const bounds = [[0, 0], [1000, 1500]]; 
const map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -1,
    zoomControl: false,
    attributionControl: false
});

// 배경 조감도 로드
const image = L.imageOverlay('/static/images/plan_map.png', bounds).addTo(map);
map.fitBounds(bounds);

// 2. 재질별 구역 지정 (기획안 스타일 반영)
const zones = {
    // 실외 아스팔트 구간 (보라색 계열)
    outdoor: L.polygon([
        [200, 1000], [200, 1400], [800, 1400], [800, 1000]
    ], {
        color: '#9b59b6',
        weight: 2,
        dashArray: '5, 5',
        fillColor: '#9b59b6',
        fillOpacity: 0.1
    }).addTo(map).bindTooltip("Outdoor: Asphalt Area"),

    // 실내 콘크리트 구간 (파란색 계열)
    indoor_concrete: L.polygon([
        [300, 200], [300, 600], [700, 600], [700, 200]
    ], {
        color: '#3498db',
        weight: 2,
        dashArray: '5, 5',
        fillColor: '#3498db',
        fillOpacity: 0.1
    }).addTo(map).bindTooltip("Indoor: Concrete Zone"),

    // 위험/관리 구간 (노란색/오렌지 계열)
    caution_zone: L.polygon([
        [400, 700], [400, 900], [600, 900], [600, 700]
    ], {
        color: '#f1c40f',
        weight: 2,
        dashArray: '5, 5',
        fillColor: '#f1c40f',
        fillOpacity: 0.1
    }).addTo(map).bindTooltip("Caution: Speed Limit Zone")
};

// 3. 로봇 주행 예상 경로 (기획안의 하늘색 라인)
const travelPath = L.polyline([
    [500, 1300], [500, 800], [400, 800], [400, 400], [600, 400]
], {
    color: '#00f2ff',
    weight: 3,
    opacity: 0.6,
    dashArray: '1, 10' // 점선으로 표현하여 세련미 추가
}).addTo(map);

// 4. 로봇 마커 (초기 위치)
const robotIcon = L.divIcon({
    className: 'custom-robot-icon',
    html: '<div class="robot-glow"></div>',
    iconSize: [20, 20]
});
const marker = L.marker([500, 1300], { icon: robotIcon }).addTo(map);