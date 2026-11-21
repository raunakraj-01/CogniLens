// In-browser n-back task presentation and event logger (PRD Section 12.4)

document.addEventListener("DOMContentLoaded", () => {
  const stepConsent = document.getElementById("step-consent");
  const stepTasks = document.getElementById("step-tasks");
  const stepTlx = document.getElementById("step-tlx");

  const btnConsent = document.getElementById("btn-consent-next");
  const btnStartBlock = document.getElementById("btn-start-block");
  const btnSubmitTlx = document.getElementById("btn-submit-tlx");

  const stimLetter = document.getElementById("stimulus-letter");
  const taskTitle = document.getElementById("task-title");
  const blockIndicator = document.getElementById("block-indicator");
  const instructionText = document.getElementById("instruction-text");

  let currentBlockIndex = 0;
  // Latin square 3-level sequence: Level 1 (Low), Level 2 (Med), Level 3 (High)
  const blocks = [
    { level: 1, name: "1-Back (Low Load)" },
    { level: 2, name: "2-Back (Medium Load)" },
    { level: 3, name: "3-Back (High Load)" }
  ];

  let participantId = "P001";
  let activeBlockId = "";
  let stimulusSequence = [];
  let stimIndex = 0;
  let taskInterval = null;

  // Step 1: Consent
  btnConsent.addEventListener("click", async () => {
    participantId = document.getElementById("part-id").value.trim() || "P001";
    const lighting = document.getElementById("part-light").value;
    const glasses = document.getElementById("part-glasses").value === "true";

    // Start session on server
    await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        participant_id: participantId,
        lighting: lighting,
        glasses: glasses,
      })
    });

    stepConsent.classList.add("hidden");
    stepTasks.classList.remove("hidden");
    prepareNextBlock();
  });

  function prepareNextBlock() {
    if (currentBlockIndex >= blocks.length) {
      // All blocks complete
      fetch("/api/session/stop", { method: "POST" });
      alert("All task blocks complete! Session data successfully recorded.");
      window.location.href = "/";
      return;
    }

    const b = blocks[currentBlockIndex];
    activeBlockId = `B${currentBlockIndex + 1}_L${b.level}`;
    taskTitle.textContent = b.name;
    blockIndicator.textContent = `Block ${currentBlockIndex + 1} of ${blocks.length}`;
    instructionText.innerHTML = `Press <strong>Spacebar</strong> when the current letter matches the letter from <strong>${b.level} step${b.level > 1 ? "s" : ""} ago</strong>.`;
    stimLetter.textContent = "+";
    btnStartBlock.classList.remove("hidden");
  }

  // Generate sequence of letters (consonants)
  function generateNBackSequence(level, length = 30) {
    const consonants = "BCDFGHJKLMNPQRSTVWXYZ";
    const seq = [];
    for (let i = 0; i < length; i++) {
      if (i >= level && Math.random() < 0.30) {
        // 30% target match rate
        seq.push(seq[i - level]);
      } else {
        seq.push(consonants[Math.floor(Math.random() * consonants.length)]);
      }
    }
    return seq;
  }

  // Start block
  btnStartBlock.addEventListener("click", async () => {
    btnStartBlock.classList.add("hidden");
    const b = blocks[currentBlockIndex];

    await fetch("/api/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_type: "block_start",
        block_id: activeBlockId,
        task: "n-back",
        level: b.level
      })
    });

    stimulusSequence = generateNBackSequence(b.level, 20); // 20 stimuli for demonstration
    stimIndex = 0;

    taskInterval = setInterval(async () => {
      if (stimIndex >= stimulusSequence.length) {
        clearInterval(taskInterval);
        // End block
        await fetch("/api/events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            event_type: "block_stop",
            block_id: activeBlockId,
            task: "n-back",
            level: b.level
          })
        });

        // Show TLX
        stepTasks.classList.add("hidden");
        stepTlx.classList.remove("hidden");
        return;
      }

      const letter = stimulusSequence[stimIndex];
      stimLetter.textContent = letter;

      // Log stimulus onset
      fetch("/api/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_type: "stimulus",
          block_id: activeBlockId,
          level: b.level,
          payload: { letter: letter, index: stimIndex }
        })
      });

      stimIndex++;
    }, 2000); // 2000ms ISI
  });

  // Keypress listener for Spacebar response
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !stepTasks.classList.contains("hidden")) {
      e.preventDefault();
      fetch("/api/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_type: "response",
          block_id: activeBlockId,
          payload: { key: "Space" }
        })
      });
    }
  });

  // Submit TLX
  btnSubmitTlx.addEventListener("click", async () => {
    const mental = parseFloat(document.getElementById("tlx-mental").value);
    const physical = parseFloat(document.getElementById("tlx-physical").value);
    const temporal = parseFloat(document.getElementById("tlx-temporal").value);
    const perf = parseFloat(document.getElementById("tlx-perf").value);
    const effort = parseFloat(document.getElementById("tlx-effort").value);
    const frust = parseFloat(document.getElementById("tlx-frust").value);

    const rawTlx = (mental + physical + temporal + perf + effort + frust) / 6.0;

    await fetch("/api/nasa-tlx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        block_id: activeBlockId,
        mental, physical, temporal, performance: perf, effort, frustration: frust,
        raw_tlx: rawTlx
      })
    });

    currentBlockIndex++;
    stepTlx.classList.add("hidden");
    stepTasks.classList.remove("hidden");
    prepareNextBlock();
  });
});
