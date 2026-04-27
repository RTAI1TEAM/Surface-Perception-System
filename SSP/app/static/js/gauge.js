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

    document.addEventListener("prediction-updated", function(e) {
        const data = e.detail;
        if (data && data.pred_prob !== undefined) {
            myGauge.refresh(parseFloat(data.pred_prob) * 100);
            document.getElementById("pred_label").innerText = data.pred_label;
            document.getElementById("pos_x").innerText = parseFloat(data.x).toFixed(3);
            document.getElementById("pos_y").innerText = parseFloat(data.y).toFixed(3);
        }
    });
});