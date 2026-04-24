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

// 경로 포인트 (lat, lng, area_type, surface_type)
const routePoints = [
    [801, 1148, 'Outdoor', 'asphalt'],
    [801, 1360, 'Outdoor', 'asphalt'],
    [773, 1448, 'Outdoor', 'asphalt'],
    [801, 1360, 'Outdoor', 'asphalt'],
    [801, 1148, 'Outdoor', 'asphalt'],
    [605, 1148, 'Outdoor', 'asphalt'],
    [605, 1440, 'Outdoor', 'asphalt'],
    [545, 1440, 'Outdoor', 'asphalt'],
    [545, 1256, 'Outdoor', 'asphalt'],
    [605, 1256, 'Outdoor', 'asphalt'],
    [605, 1148, 'Outdoor', 'asphalt'],
    [280, 1148, 'Outdoor', 'asphalt'],
    [280, 1396, 'Outdoor', 'asphalt'],
    [352, 1396, 'Outdoor', 'asphalt'],
    [240, 1396, 'Outdoor', 'asphalt'],
    [280, 1396, 'Outdoor', 'asphalt'],
    [280, 1148, 'Outdoor', 'asphalt'],
    [456, 1148, 'Outdoor', 'asphalt'],
    [456, 980,  'Indoor',  'tiled'],
    [472, 852,  'Indoor',  'soft_tiles'],
    [801, 852,  'Indoor',  'wood'],
    [805, 604,  'Indoor',  'carpet'],
    [452, 604,  'Indoor',  'carpet'],
    [480, 380,  'Indoor',  'tiled'],
    [825, 380,  'Indoor',  'tiled'],
    [837, 156,  'Indoor',  'concrete'],
    [276, 156,  'Indoor',  'concrete'],
    [276, 476,  'Indoor',  'soft_pvc'],
    [452, 476,  'Indoor',  'soft_pvc'],
    [452, 792,  'Indoor',  'tiled'],
    [124, 792,  'Indoor',  'fine_concrete'],
    [172, 928,  'Indoor',  'fine_concrete'],
    [328, 792,  'Indoor',  'fine_concrete'],
    [456, 980,  'Indoor',  'tiled'],
    [456, 1148, 'Outdoor', 'asphalt'],
    [801, 1148, 'Outdoor', 'asphalt'],
];

// 로봇 마커
const robotIcon = L.divIcon({
    className: 'custom-robot-icon',
    html: '<div class="robot-glow"></div>',
    iconSize: [20, 20]
});
const marker = L.marker([routePoints[0][0], routePoints[0][1]], { icon: robotIcon, draggable: true }).addTo(map);
marker.dragging.enable();

// 경로 보간
function interpolate(p1, p2, steps) {
    const points = [];
    for (let i = 1; i <= steps; i++) {
        points.push({
            lat: p1[0] + (p2[0] - p1[0]) * (i / steps),
            lng: p1[1] + (p2[1] - p1[1]) * (i / steps),
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
    const dist = Math.sqrt(Math.pow(p2[0]-p1[0], 2) + Math.pow(p2[1]-p1[1], 2));
    const steps = Math.max(1, Math.round(dist / 20));
    expandedRoute = expandedRoute.concat(interpolate(p1, p2, steps));
}

// 로봇 자동 이동
let stepIndex = 0;

const robotInterval = setInterval(function() {
    stepIndex = (stepIndex + 1) % expandedRoute.length;
    const nextPos = expandedRoute[stepIndex];
    marker.setLatLng([nextPos.lat, nextPos.lng]);

    fetch('/api/update_position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            x: nextPos.lng,
            y: nextPos.lat,
            area_type: nextPos.area_type || 'Outdoor',
            surface_type: nextPos.surface_type || 'asphalt'
        })
    });
}, 150);

map.on('click', function(e) {
    console.log(e.latlng);
});