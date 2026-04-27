document.addEventListener("DOMContentLoaded", function() {
    const myGauge = new JustGage({
        id: "myGauge",
        value: 45,
        min: 0,
        max: 100,
        title: "Confidence",
        label: "%",
        pointer: true,
        gaugeWidthScale: 0.6,
        customSectors: [
            { color: "#00ff00", lo: 0, hi: 33 },
            { color: "#ffff00", lo: 34, hi: 66 },
            { color: "#ff0000", lo: 67, hi: 100 }
        ],
        counter: true
    });

    function refreshGauge() {
        fetch("/api/fetch_pred")
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch prediction: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Latest prediction log:", data);
                if (data && data.pred_prob !== undefined) {
                    myGauge.refresh(parseFloat(data.pred_prob));
                    document.getElementById("pred_label").innerText = data.pred_label;
                    document.getElementById("pos_x").innerText = data.x;
                    document.getElementById("pos_y").innerText = data.y;
                }
            })
            .catch(error => {
                console.error("Failed to refresh gauge.", error);
            });
    }

    refreshGauge();

    document.addEventListener("prediction-log-updated", function() {
        refreshGauge();
    });
});
