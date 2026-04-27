document.addEventListener("DOMContentLoaded", function() {
    const sensorCtx = document.getElementById("sensorChart").getContext("2d");
    const sensorChart = new Chart(sensorCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "X Axis", data: [], borderColor: "#003f5c", tension: 0.4 },
                { label: "Y Axis", data: [], borderColor: "#58508d", tension: 0.4 },
                { label: "Z Axis", data: [], borderColor: "#bc5090", tension: 0.4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "top", align: "end" } },
            scales: { y: { min: -20, max: 20 } }
        }
    });

    const SVG_W = 1500;
    const SVG_H = 1000;
    const HAZARD_PAUSE_MS = 0;
    const marker = document.getElementById("robotMarker");
    const mapImg = document.getElementById("mapImg");
    const MAX_CHART_POINTS = 20;

    function updateSensorChart(sequenceNo, chartData) {
        if (!chartData) {
            return;
        }

        if (Array.isArray(chartData.labels) && chartData.labels.length === 3) {
            sensorChart.data.datasets[0].label = chartData.labels[0];
            sensorChart.data.datasets[1].label = chartData.labels[1];
            sensorChart.data.datasets[2].label = chartData.labels[2];
        }

        sensorChart.data.labels.push(`P${sequenceNo}`);
        sensorChart.data.datasets[0].data.push(chartData.x);
        sensorChart.data.datasets[1].data.push(chartData.y);
        sensorChart.data.datasets[2].data.push(chartData.z);

        if (sensorChart.data.labels.length > MAX_CHART_POINTS) {
            sensorChart.data.labels.shift();
            sensorChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        sensorChart.update("none");
    }

    function interpolate(p1, p2, steps) {
        const points = [];
        for (let i = 1; i <= steps; i++) {
            points.push({
                y: p1.y + (p2.y - p1.y) * (i / steps),
                x: p1.x + (p2.x - p1.x) * (i / steps),
                point_id: p2.point_id,
                sequence_no: p2.sequence_no,
                area_type: p2.area_type,
                surface_type: p2.surface_type,
                feature_label: p2.feature_label
            });
        }
        return points;
    }

    function buildExpandedRoute(routePoints) {
        if (!routePoints.length) {
            return [];
        }

        let expandedRoute = [{
            y: routePoints[0].y,
            x: routePoints[0].x,
            point_id: routePoints[0].point_id,
            sequence_no: routePoints[0].sequence_no,
            area_type: routePoints[0].area_type,
            surface_type: routePoints[0].surface_type,
            feature_label: routePoints[0].feature_label
        }];

        for (let i = 0; i < routePoints.length - 1; i++) {
            const p1 = routePoints[i];
            const p2 = routePoints[i + 1];
            const dist = Math.sqrt(Math.pow(p2.y - p1.y, 2) + Math.pow(p2.x - p1.x, 2));
            const steps = Math.max(1, Math.round(dist / 20));
            expandedRoute = expandedRoute.concat(interpolate(p1, p2, steps));
        }

        return expandedRoute;
    }

    function updateMarker(svgY, svgX) {
        const rect = mapImg.getBoundingClientRect();
        const scale = Math.min(rect.width / SVG_W, rect.height / SVG_H);
        const renderedW = SVG_W * scale;
        const renderedH = SVG_H * scale;
        const offsetX = (rect.width - renderedW) / 2;
        const offsetY = (rect.height - renderedH) / 2;

        const px = offsetX + (svgX / SVG_W) * renderedW;
        const py = offsetY + (svgY / SVG_H) * renderedH;

        marker.style.left = px + "px";
        marker.style.top = py + "px";
    }

function addHazardPin(svgY, svgX, color = "red") {
    const rect = mapImg.getBoundingClientRect();
    const scale = Math.min(rect.width / SVG_W, rect.height / SVG_H);
    const renderedW = SVG_W * scale;
    const renderedH = SVG_H * scale;
    const offsetX = (rect.width - renderedW) / 2;
    const offsetY = (rect.height - renderedH) / 2;

    const px = offsetX + (svgX / SVG_W) * renderedW;
    const py = offsetY + (svgY / SVG_H) * renderedH;

    const pin = document.createElement("div");
    pin.className = "hazard-pin";
    pin.textContent = "📍";
    
    // 스타일 설정: 색상을 파라미터로 받아 동적으로 변경
    pin.style.left = px + "px";
    pin.style.top = py + "px";
    pin.style.position = "absolute";
    pin.style.fontSize = "24px";
    
    // 📍 이모지는 텍스트 색상 대신 drop-shadow로 색감을 강조하거나, 
    // 필터를 사용해 색상을 변경할 수 있습니다. 
    // 가장 확실한 방법은 CSS 필터(blue계열)를 거는 것입니다.
    if (color === "blue") {
        pin.style.filter = "hue-rotate(200deg) brightness(1.2)"; // 파란색 느낌으로 변색
    }

    document.getElementById("map").appendChild(pin);
}

    function startRobot(routePoints) {
        const expandedRoute = buildExpandedRoute(routePoints);
        if (!expandedRoute.length) {
            console.error("No route points available from database.");
            return;
        }

        let stepIndex = 0;
        let lastProcessedPointId = null;
        let pausedUntil = 0;
        updateMarker(expandedRoute[0].y, expandedRoute[0].x);

        setInterval(function() {
            if (Date.now() < pausedUntil) {
                return;
            }

            const pos = expandedRoute[stepIndex];
            updateMarker(pos.y, pos.x);

            if (pos.point_id !== lastProcessedPointId) {
                fetch("/api/update_position", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        point_id: pos.point_id
                    })
                })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Prediction request failed: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(result => {
                        updateSensorChart(result.sequence_no, result.chart);

                        // 1. 실외 위험 (pothole) -> 기본 빨간색 핀
                        if (result.pred_label === "pothole") {
                            addHazardPin(result.y, result.x, "red");
                        } 
                        
                        // 2. 실내 이상치 (outlier) -> 파란색 핀 📍
                        else if (result.pred_label === "outlier") {
                            addHazardPin(result.y, result.x, "blue");
                            
                            // 로그 출력 (파란색으로 강조)
                            console.log(`%c[Indoor Outlier] 감지: (${result.x}, ${result.y})`, "color: blue; font-weight: bold;");
                        }

                        // 로그 업데이트 이벤트 (공통)
                        if (result.logged) {
                            document.dispatchEvent(new CustomEvent("prediction-log-updated", { detail: result }));
                        }
                    })
                    .catch(error => {
                        console.error("Failed to process point prediction.", error);
                    });

                lastProcessedPointId = pos.point_id;
            }

            stepIndex = (stepIndex + 1) % expandedRoute.length;
        }, 150);
    }

    fetch("/api/robot_path")
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load route: ${response.status}`);
            }
            return response.json();
        })
        .then(routePoints => {
            startRobot(routePoints);
        })
        .catch(error => {
            console.error("Failed to initialize robot route.", error);
        });

    mapImg.addEventListener("click", function(e) {
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
