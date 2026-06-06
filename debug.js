function fmtFixed(value, decimals) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(decimals) : "-";
}

async function updateDebugPanel() {
    try {
        const response = await fetch("/emotion_state", {
            cache: "no-store"
        });

        const data = await response.json();

        document.getElementById("emotion").textContent = data.emotion ?? "-";
        document.getElementById("raw-emotion").textContent = data.raw_emotion ?? "-";
        const topN = Array.isArray(data.top_n) ? data.top_n : [];
        document.getElementById("top-n").textContent = topN.length
            ? topN
                  .map((row) => {
                      const c = fmtFixed(row.confidence, 3);
                      return `${row.emotion ?? "?"} ${c}`;
                  })
                  .join(" · ")
            : "-";
        document.getElementById("confidence").textContent = fmtFixed(data.confidence, 3);
        document.getElementById("crop-used").textContent = data.used_face_crop ? "yes" : "no";
        document.getElementById("fps").textContent = fmtFixed(data.fps, 1);
        document.getElementById("camera-running").textContent = data.camera_running ? "yes" : "no";
        document.getElementById("stream-ready").textContent = data.stream_ready ? "yes" : "no";
        document.getElementById("camera-error").textContent = data.camera_start_error || "-";
        document.getElementById("model-loaded").textContent = data.model_loaded ? "yes" : "no";
    } catch (error) {
        console.error("Failed to update debug panel:", error);
    }
}

setInterval(updateDebugPanel, 300);
updateDebugPanel();
