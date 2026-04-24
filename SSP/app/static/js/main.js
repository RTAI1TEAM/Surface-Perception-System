document.addEventListener("DOMContentLoaded", function() {
    const sensorCtx = document.getElementById('sensorChart').getContext('2d');
    const sensorChart = new Chart(sensorCtx, {
        type: 'line',
        data: {
            labels: [],
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
            scales: { y: { min: -60, max: 60 } }
        }
    });
});

// 1. 지도 초기화
const bounds = [[0, 0], [1000, 1500]];
const map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -2,
    maxZoom: 1,
    zoomControl: true,
    attributionControl: false,
    scrollWheelZoom: false
});

const image = L.imageOverlay('/static/images/plan_map.svg', bounds, {opacity: 1.0}).addTo(map);
map.fitBounds(bounds);
setTimeout(function() {
    map.invalidateSize();
    map.fitBounds(bounds);
}, 100);

// 2. 재질별 구역
const zones = {
    outdoor: L.polygon([
        [200, 1000], [200, 1400], [800, 1400], [800, 1000]
    ], { color: '#9b59b6', weight: 2, dashArray: '5, 5', fillColor: '#9b59b6', fillOpacity: 0.1 })
    .addTo(map).bindTooltip("Outdoor: Asphalt Area"),

    indoor_concrete: L.polygon([
        [300, 200], [300, 600], [700, 600], [700, 200]
    ], { color: '#3498db', weight: 2, dashArray: '5, 5', fillColor: '#3498db', fillOpacity: 0.1 })
    .addTo(map).bindTooltip("Indoor: Concrete Zone"),

    caution_zone: L.polygon([
        [400, 700], [400, 900], [600, 900], [600, 700]
    ], { color: '#f1c40f', weight: 2, dashArray: '5, 5', fillColor: '#f1c40f', fillOpacity: 0.1 })
    .addTo(map).bindTooltip("Caution: Speed Limit Zone")
};

// 3. 경로 포인트
const routePoints = [
    [801, 1148],  // 출발점
    [801, 1360],  // 크랙 2 방향
    [773, 1448],  // 크랙 2 끝
    [801, 1360],  // 돌아오기
    [801, 1148],  // 출발점
    [605, 1148],  // 주차장 앞 도로
    [605, 1440],  // 주차장 진입
    [545, 1440],  // 주차장 안쪽
    [545, 1256],  // 주차장 안쪽 2
    [605, 1256],  // 주차장 나오기
    [605, 1148],  // 주차장 앞 도로
    [280, 1148],  // 외부창고 앞 도로
    [280, 1396],  // 외부창고 진입
    [352, 1396],  // 외부창고 안 1
    [240, 1396],  // 외부창고 안 2
    [280, 1396],  // 외부창고 나오기
    [280, 1148],  // 외부창고 앞 도로
    [456, 1148],  // 출입구 앞 도로
    [456, 980],   // 출입구
    [472, 852],   // A-5
    [801, 852],   // A-4
    [805, 604],   // A-3 뒷문
    [452, 604],   // A-3 앞문
    [480, 380],   // A-2 앞
    [825, 380],   // A-2 안쪽
    [837, 156],   // A-1
    [276, 156],   // B-1
    [276, 476],   // B-2
    [452, 476],   // B-2 나오기
    [452, 792],   // B-3 앞 복도
    [124, 792],   // B-3 왼쪽
    [172, 928],   // B-3 오른쪽
    [328, 792],   // B-3 출입구
    [456, 980],   // 출입구
    [456, 1148],  // 출입구 앞 도로
    [801, 1148],  // 출발점
];

// 경로 라인 표시
const travelPath = L.polyline(routePoints, {
    color: '#00f2ff',
    weight: 3,
    opacity: 0.6,
    dashArray: '1, 10'
}).addTo(map);

// 4. 로봇 마커
const robotIcon = L.divIcon({
    className: 'custom-robot-icon',
    html: '<div class="robot-glow"></div>',
    iconSize: [20, 20]
});
const marker = L.marker(routePoints[0], { icon: robotIcon, draggable: true }).addTo(map);
marker.dragging.enable();

// 5. 경로 보간 (직각 이동)
function interpolate(p1, p2, steps) {
    const points = [];
    for (let i = 1; i <= steps; i++) {
        points.push([
            p1[0] + (p2[0] - p1[0]) * (i / steps),
            p1[1] + (p2[1] - p1[1]) * (i / steps)
        ]);
    }
    return points;
}

let expandedRoute = [];
for (let i = 0; i < routePoints.length - 1; i++) {
    const midPoints = interpolate(routePoints[i], routePoints[i + 1], 10);
    expandedRoute = expandedRoute.concat(midPoints);
}
expandedRoute.push(routePoints[routePoints.length - 1]);

// 6. 로봇 자동 이동
let stepIndex = 0;

const robotInterval = setInterval(function() {
    stepIndex = (stepIndex + 1) % expandedRoute.length;
    const nextPos = expandedRoute[stepIndex];
    marker.setLatLng(nextPos);

    fetch('/api/update_position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: nextPos[1], y: nextPos[0] })
    });
}, 150);

// 7. 좌표 확인용
map.on('click', function(e) {
    console.log(e.latlng);
});