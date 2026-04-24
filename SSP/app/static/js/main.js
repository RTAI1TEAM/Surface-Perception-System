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

    const SVG_W = 1500;
    const SVG_H = 1000;
const routePoints = [
    [207, 1141, 'Outdoor', 'asphalt'],  // 출발점
    [207, 1206, 'Outdoor', 'asphalt'],
    [276, 1206, 'Outdoor', 'asphalt'],  // 크랙
    [276, 1141, 'Outdoor', 'asphalt'],  // 주차장 앞 도로
    [400, 1141, 'Outdoor', 'asphalt'],
    [400, 1293, 'Outdoor', 'asphalt'],  // 주차장
    [469, 1293, 'Outdoor', 'asphalt'],
    [469, 1330, 'Outdoor', 'asphalt'],  // 주차장 안쪽
    [327, 1330, 'Outdoor', 'asphalt'],
    [327, 1357, 'Outdoor', 'asphalt'],  // 주차장 안쪽2
    [327, 1293, 'Outdoor', 'asphalt'],
    [400, 1293, 'Outdoor', 'asphalt'],
    [400, 1141, 'Outdoor', 'asphalt'],  // 주차장 앞 도로
    [704, 1141, 'Outdoor', 'asphalt'],  // 외부창고 방향 직진
    [704, 1233, 'Outdoor', 'asphalt'],  // 외부창고 문
    [639, 1233, 'Outdoor', 'asphalt'],
    [639, 1330, 'Outdoor', 'asphalt'],  // 외부창고 안1
    [777, 1330, 'Outdoor', 'asphalt'],
    [777, 1321, 'Outdoor', 'asphalt'],  // 외부창고 안2
    [777, 1233, 'Outdoor', 'asphalt'],
    [704, 1233, 'Outdoor', 'asphalt'],  // 외부창고 문
    [704, 1141, 'Outdoor', 'asphalt'],  // 외부창고 앞 도로
    [538, 1141, 'Outdoor', 'asphalt'],  // 출입구 앞 도로
    [538, 837,  'Indoor',  'soft_tiles'], // A-5
    [184, 837,  'Indoor',  'wood'],
    [184, 833,  'Indoor',  'wood'],      // A-4
    [184, 612,  'Indoor',  'carpet'],
    [193, 612,  'Indoor',  'carpet'],    // A-3 뒷문
    [524, 612,  'Indoor',  'carpet'],
    [524, 607,  'Indoor',  'carpet'],    // A-3 앞문
    [538, 382,  'Indoor',  'tiled'],     // A-2 앞
    [179, 382,  'Indoor',  'tiled'],     // A-2 안쪽
    [179, 156,  'Indoor',  'concrete'],
    [184, 156,  'Indoor',  'concrete'],  // A-1 안쪽
    [796, 156,  'Indoor',  'concrete'],
    [796, 193,  'Indoor',  'concrete'],  // B-1
    [796, 483,  'Indoor',  'soft_pvc'],
    [759, 483,  'Indoor',  'soft_pvc'],  // B-2 안쪽
    [538, 483,  'Indoor',  'soft_pvc'],
    [538, 479,  'Indoor',  'soft_pvc'],  // B-2 앞
    [538, 787,  'Indoor',  'fine_concrete'], // B-3 앞 복도
    [630, 805,  'Indoor',  'fine_concrete'], // B-3 출입구
    [768, 805,  'Indoor',  'fine_concrete'],
    [768, 672,  'Indoor',  'fine_concrete'], // B-3 왼쪽 안쪽
    [768, 948,  'Indoor',  'fine_concrete'], // B-3 오른쪽 안쪽
    [768, 805,  'Indoor',  'fine_concrete'],
    [630, 805,  'Indoor',  'fine_concrete'], // B-3 출입구
    [538, 805,  'Indoor',  'fine_concrete'],
    [538, 1141, 'Outdoor', 'asphalt'],  // 출입구
    [207, 1141, 'Outdoor', 'asphalt'],  // 출발점 방향
    [207, 1353, 'Outdoor', 'asphalt'],
    [212, 1353, 'Outdoor', 'asphalt'],  // 크랙2 앞
    [175, 1353, 'Outdoor', 'asphalt'],
    [175, 1348, 'Outdoor', 'asphalt'],  // 크랙2
    [175, 1141, 'Outdoor', 'asphalt'],
    [207, 1141, 'Outdoor', 'asphalt'],  // 출발점 복귀
];
    function interpolate(p1, p2, steps) {
        const points = [];
        for (let i = 1; i <= steps; i++) {
            points.push({
                y: p1[0] + (p2[0] - p1[0]) * (i / steps),
                x: p1[1] + (p2[1] - p1[1]) * (i / steps),
                area_type: p2[2],
                surface_type: p2[3]
            });
        }
        return points;
    }

    let expandedRoute = [];
    for (let i = 0; i < routePoints.length - 1; i++) {
        const p1 = routePoints[i];
        const p2 = routePoints[i + 1];
        const dist = Math.sqrt(Math.pow(p2[0]-p1[0],2) + Math.pow(p2[1]-p1[1],2));
        const steps = Math.max(1, Math.round(dist / 20));
        expandedRoute = expandedRoute.concat(interpolate(p1, p2, steps));
    }

    const marker = document.getElementById('robotMarker');
    const mapImg = document.getElementById('mapImg');

    function updateMarker(svgY, svgX) {
        const rect = mapImg.getBoundingClientRect();
        const scale = Math.min(rect.width / SVG_W, rect.height / SVG_H);
        const renderedW = SVG_W * scale;
        const renderedH = SVG_H * scale;
        const offsetX = (rect.width - renderedW) / 2;
        const offsetY = (rect.height - renderedH) / 2;

        const px = offsetX + (svgX / SVG_W) * renderedW;
        const py = offsetY + (svgY / SVG_H) * renderedH;

        marker.style.left = px + 'px';
        marker.style.top = py + 'px';
    }

    let stepIndex = 0;

    setInterval(function() {
        stepIndex = (stepIndex + 1) % expandedRoute.length;
        const pos = expandedRoute[stepIndex];
        updateMarker(pos.y, pos.x);

        fetch('/api/update_position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: pos.x,
                y: pos.y,
                area_type: pos.area_type || 'Outdoor',
                surface_type: pos.surface_type || 'asphalt'
            })
        });
    }, 150);

    updateMarker(routePoints[0][0], routePoints[0][1]);

    mapImg.addEventListener('click', function(e) {
        const rect = mapImg.getBoundingClientRect();
        const scale = Math.min(rect.width / SVG_W, rect.height / SVG_H);
        const renderedW = SVG_W * scale;
        const renderedH = SVG_H * scale;
        const offsetX = (rect.width - renderedW) / 2;
        const offsetY = (rect.height - renderedH) / 2;

        const svgX = Math.round((e.clientX - rect.left - offsetX) / scale);
        const svgY = Math.round((e.clientY - rect.top - offsetY) / scale);

        console.log(`[${svgY}, ${svgX}],`);
    });
});