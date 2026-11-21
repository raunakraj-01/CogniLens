// Real-time live dashboard logic subscribing to SSE stream
document.addEventListener("DOMContentLoaded", () => {
  const loadBadge = document.getElementById("load-badge");
  const confidenceVal = document.getElementById("confidence-val");
  const scoreVal = document.getElementById("score-val");
  const gaugeFill = document.getElementById("gauge-fill");

  const chipFace = document.getElementById("chip-face");
  const chipCalib = document.getElementById("chip-calib");
  const fpsBadge = document.getElementById("fps-badge");

  const indBlink = document.getElementById("ind-blink");
  const deltaBlink = document.getElementById("delta-blink");
  const indEar = document.getElementById("ind-ear");
  const indGaze = document.getElementById("ind-gaze");
  const deltaGaze = document.getElementById("delta-gaze");
  const indHead = document.getElementById("ind-head");
  const deltaHead = document.getElementById("delta-head");
  const indTension = document.getElementById("ind-tension");
  const deltaTension = document.getElementById("delta-tension");

  const warningsBox = document.getElementById("warnings-box");
  const warningText = document.getElementById("warning-text");

  const toggleOverlay = document.getElementById("toggle-overlay");
  const liveCanvas = document.getElementById("live-canvas");
  const canvasCtx = liveCanvas ? liveCanvas.getContext("2d") : null;

  // Universal high-performance video renderer (compatible with Safari, Chrome, Firefox)
  if (liveCanvas && canvasCtx) {
    let frameActive = true;
    function fetchAndRenderFrame() {
      if (!frameActive) return;
      const overlayOn = toggleOverlay ? toggleOverlay.checked : true;
      const img = new Image();
      img.onload = () => {
        canvasCtx.drawImage(img, 0, 0, liveCanvas.width, liveCanvas.height);
        requestAnimationFrame(fetchAndRenderFrame);
      };
      img.onerror = () => {
        setTimeout(fetchAndRenderFrame, 80);
      };
      img.src = `/api/frame?overlay=${overlayOn}&_t=${Date.now()}`;
    }
    fetchAndRenderFrame();
  }

  // Chart setup
  const ctx = document.getElementById("timelineChart").getContext("2d");
  const maxDataPoints = 60; // 60 seconds
  const chartData = {
    labels: Array(maxDataPoints).fill(""),
    datasets: [{
      label: "Cognitive Load Score",
      data: Array(maxDataPoints).fill(0),
      borderColor: "#38bdf8",
      backgroundColor: "rgba(56, 189, 248, 0.15)",
      borderWidth: 2,
      fill: true,
      tension: 0.3,
    }]
  };

  const timelineChart = new Chart(ctx, {
    type: "line",
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: {
          min: 0,
          max: 1.0,
          grid: { color: "#334155" },
          ticks: { color: "#94a3b8" }
        },
        x: {
          grid: { display: false },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // Connect to SSE stream
  const eventSource = new EventSource("/api/stream");

  eventSource.onmessage = (event) => {
    try {
      const sanitized = event.data.replace(/:\s*NaN\b/g, ': 0.0').replace(/:\s*Infinity\b/g, ': 999.0');
      const data = JSON.parse(sanitized);

      // Status chips
      if (data.face_detected) {
        chipFace.textContent = "Face Tracked";
        chipFace.className = "chip chip-success";
      } else {
        chipFace.textContent = "No Face Detected";
        chipFace.className = "chip chip-warning";
      }

      if (data.calibrated) {
        chipCalib.textContent = "Calibrated";
        chipCalib.className = "chip chip-success";
      } else {
        chipCalib.textContent = "Uncalibrated Baseline";
        chipCalib.className = "chip chip-muted";
      }

      fpsBadge.textContent = `${data.fps} FPS`;

      // Load Hero
      const level = data.load.level;
      loadBadge.textContent = level;
      loadBadge.className = "load-badge " + (
        level === "HIGH" ? "load-high" : level === "MEDIUM" ? "load-med" : "load-low"
      );

      const confPct = Math.round((data.load.confidence || 0) * 100);
      confidenceVal.textContent = `${confPct}%`;
      const score = data.load.score || 0.0;
      scoreVal.textContent = score.toFixed(2);
      gaugeFill.style.width = Math.round(score * 100) + "%";

      // Indicators
      if (data.indicators) {
        indBlink.textContent = data.indicators.blink_rate;
        indEar.textContent = data.indicators.ear;
        indGaze.textContent = data.indicators.gaze_dispersion;
        indHead.textContent = data.indicators.head_motion;
        if (indTension) indTension.textContent = data.indicators.brow_furrow || "--";

        if (data.calibrated && data.indicators.z) {
          const zb = data.indicators.z.blink_rate;
          const zg = data.indicators.z.gaze_dispersion;
          const zh = data.indicators.z.head_motion;
          const zt = data.indicators.z.brow_furrow;
          deltaBlink.textContent = `Δ baseline: ${zb > 0 ? "+" : ""}${zb} σ`;
          deltaGaze.textContent = `Δ baseline: ${zg > 0 ? "+" : ""}${zg} σ`;
          deltaHead.textContent = `Δ baseline: ${zh > 0 ? "+" : ""}${zh} σ`;
          if (deltaTension) deltaTension.textContent = `Δ baseline: ${zt > 0 ? "+" : ""}${zt} σ`;
        } else {
          deltaBlink.textContent = "Δ: (Calibrate to compare)";
          deltaGaze.textContent = "Δ: (Calibrate to compare)";
          deltaHead.textContent = "Δ: (Calibrate to compare)";
          if (deltaTension) deltaTension.textContent = "Δ: (Calibrate to compare)";
        }
      }

      // Warnings
      if (data.warnings && data.warnings.length > 0) {
        const readable = data.warnings.map(w => {
          if (w === "not_calibrated") return "⚠️ Run calibration for personalized baseline deviation";
          if (w === "no_face") return "⚠️ No face detected in frame";
          if (w === "high_head_motion") return "ℹ️ Excessive head motion detected";
          if (w === "oral_activity") return "ℹ️ Oral activity (eating / speaking) — jaw tension artifacts filtered";
          if (w === "model_missing") return "⚠️ Running in indicators-only mode (model missing)";
          return w;
        });
        warningText.textContent = readable.join(" | ");
        warningsBox.classList.remove("hidden");
      } else {
        warningsBox.classList.add("hidden");
      }

      // Update Chart
      chartData.datasets[0].data.push(score);
      chartData.datasets[0].data.shift();
      timelineChart.update();

    } catch (err) {
      console.error("Error parsing SSE event:", err);
    }
  };

  eventSource.onerror = (err) => {
    console.warn("SSE connection interrupted. Reconnecting...", err);
  };
});
