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
            scales: { y: { min: -60, max: 60 } }
        }
    });

    const SVG_W = 1500;
    const SVG_H = 1000;
    const marker = document.getElementById("robotMarker");
    const mapImg = document.getElementById("mapImg");

    function interpolate(p1, p2, steps) {
        const points = [];
        for (let i = 1; i <= steps; i++) {
            points.push({
                y: p1.y + (p2.y - p1.y) * (i / steps),
                x: p1.x + (p2.x - p1.x) * (i / steps),
                area_type: p2.area_type,
                surface_type: p2.surface_type
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
            area_type: routePoints[0].area_type,
            surface_type: routePoints[0].surface_type
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

    function startRobot(routePoints) {
        const expandedRoute = buildExpandedRoute(routePoints);
        if (!expandedRoute.length) {
            console.error("No route points available from database.");
            return;
        }

        let stepIndex = 0;
        updateMarker(expandedRoute[0].y, expandedRoute[0].x);

        setInterval(function() {
            const pos = expandedRoute[stepIndex];
            updateMarker(pos.y, pos.x);

            console.log(`Current position x: ${pos.x}, y: ${pos.y}`);

            fetch("/api/update_position", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    x: pos.x,
                    y: pos.y,
                    area_type: pos.area_type || "Outdoor",
                    surface_type: pos.surface_type || "asphalt"
                })
            }).catch(error => {
                console.error("Failed to update robot position.", error);
            });

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
