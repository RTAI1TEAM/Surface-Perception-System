document.addEventListener("DOMContentLoaded", function() {
    const logList = document.getElementById("logList");
    let lastPredictionId = null;

    function buildLogMessage(data) {
        const playedAt = data.played_at || "unknown_time";
        const areaType = data.area_type || "Unknown";
        const surfaceType = data.surface_type || "unknown_surface";
        const predLabel = data.pred_label || "unknown";
        const predProb = data.pred_prob || "0.0";
        const x = data.x !== undefined ? Number(data.x).toFixed(3) : "0.000";
        const y = data.y !== undefined ? Number(data.y).toFixed(3) : "0.000";

        return `[${playedAt}] ${areaType} / ${surfaceType} / ${predLabel} ${predProb}% @ (${x}, ${y})`;
    }

    function appendLatestPredictionLog() {
        fetch("/api/fetch_pred")
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch latest prediction log: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (!data || data.status !== "ok" || !data.prediction_id) {
                    return;
                }

                if (data.prediction_id === lastPredictionId) {
                    return;
                }

                lastPredictionId = data.prediction_id;

                const li = document.createElement("li");
                li.textContent = buildLogMessage(data);
                logList.prepend(li);

                while (logList.children.length > 20) {
                    logList.removeChild(logList.lastElementChild);
                }
            })
            .catch(error => {
                console.error("Failed to append alert log.", error);
            });
    }

    appendLatestPredictionLog();

    document.addEventListener("prediction-log-updated", function() {
        appendLatestPredictionLog();
    });
});
