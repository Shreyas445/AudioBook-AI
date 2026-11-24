(() => {
  "use strict";

  const state = {
    sys: null,
    sfxEnabled: true,
    modelChoices: [],
    uploadedAutoFile: null,
    uploadedPromptFile: null,
    promptParts: [],
    activePromptIdx: 0,
    activeEventSource: null,
    isRunning: false,
    audioCtx: null,
    analyser: null,
    sourceNode: null,
    rafId: null,
  };

  const $ = (id) => document.getElementById(id);

  // =========================================================================
  // CUSTOM TRANSLUCENT FROST-BLUR DROPDOWN & COMBOBOX ENGINE
  // =========================================================================
  const dropdownMgr = {
    activeAnchor: null,
    activeOwner: null,
    items: [],
    focusedIdx: -1,
    onPick: null,

    close() {
      const portal = $("dropdown-portal");
      portal.classList.add("hidden");
      portal.classList.remove("drop-up");
      portal.innerHTML = "";
      if (this.activeOwner) {
        this.activeOwner.classList.remove("open");
      }
      this.activeAnchor = null;
      this.activeOwner = null;
      this.items = [];
      this.focusedIdx = -1;
      this.onPick = null;
    },

    open({ anchorEl, ownerEl, options, selectedValue, mono = false, onSelect }) {
      const portal = $("dropdown-portal");
      if (this.activeOwner === ownerEl && !portal.classList.contains("hidden")) {
        this.close();
        return;
      }
      this.close();

      this.activeAnchor = anchorEl;
      this.activeOwner = ownerEl;
      this.items = options;
      this.onPick = onSelect;
      ownerEl.classList.add("open");

      portal.innerHTML = "";
      if (!options.length) {
        const empty = document.createElement("div");
        empty.className = "dd-empty";
        empty.textContent = "Custom model — press Enter to use";
        portal.appendChild(empty);
      } else {
        options.forEach((opt, idx) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.role = "option";
          const isSel = opt.value === selectedValue;
          if (isSel) this.focusedIdx = idx;
          btn.className = `dd-item${mono ? " mono-opt" : ""}${isSel ? " selected" : ""}`;

          const main = document.createElement("span");
          main.className = "dd-main";
          main.textContent = opt.label;
          btn.appendChild(main);

          if (opt.badge) {
            const sub = document.createElement("span");
            sub.className = "dd-sub";
            sub.textContent = opt.badge;
            btn.appendChild(sub);
          } else {
            const chk = document.createElement("span");
            chk.className = "dd-check";
            chk.textContent = "✓";
            btn.appendChild(chk);
          }

          btn.addEventListener("mousedown", (e) => {
            e.preventDefault(); // Prevent input blur before selection fires
            onSelect(opt.value, opt.label);
            this.close();
          });
          portal.appendChild(btn);
        });
      }

      portal.classList.remove("hidden");
      this.reposition();

      const selEl = portal.querySelector(".dd-item.selected");
      if (selEl) selEl.scrollIntoView({ block: "nearest" });
    },

    reposition() {
      if (!this.activeAnchor) return;
      const portal = $("dropdown-portal");
      const rect = this.activeAnchor.getBoundingClientRect();
      const menuHeight = Math.min(245, portal.scrollHeight || 200);
      const spaceBelow = window.innerHeight - rect.bottom - 12;
      const openUpward = spaceBelow < menuHeight && rect.top > spaceBelow;

      portal.style.width = `${Math.max(rect.width, 190)}px`;
      portal.style.left = `${Math.min(rect.left, window.innerWidth - Math.max(rect.width, 190) - 10)}px`;

      if (openUpward) {
        portal.classList.add("drop-up");
        portal.style.top = "auto";
        portal.style.bottom = `${window.innerHeight - rect.top + 6}px`;
      } else {
        portal.classList.remove("drop-up");
        portal.style.bottom = "auto";
        portal.style.top = `${rect.bottom + 6}px`;
      }
    },

    moveFocus(delta) {
      const portal = $("dropdown-portal");
      if (portal.classList.contains("hidden") || !this.items.length) return;
      const nodes = portal.querySelectorAll(".dd-item");
      if (!nodes.length) return;
      this.focusedIdx = (this.focusedIdx + delta + nodes.length) % nodes.length;
      nodes.forEach((n, i) => n.classList.toggle("focused", i === this.focusedIdx));
      nodes[this.focusedIdx]?.scrollIntoView({ block: "nearest" });
    },

    commitFocused() {
      if (this.focusedIdx >= 0 && this.focusedIdx < this.items.length && this.onPick) {
        const picked = this.items[this.focusedIdx];
        this.onPick(picked.value, picked.label);
        this.close();
        return true;
      }
      return false;
    },
  };

  // Upgrade a <select> element into a Custom Frosted Glass Dropdown
  const customSelectRegistry = {};

  function enhanceFrostedSelect(selectId) {
    const sel = $(selectId);
    if (!sel) return;
    sel.classList.add("native-hidden");

    let trig = sel.parentElement.querySelector(".frosted-trigger");
    if (!trig) {
      trig = document.createElement("button");
      trig.type = "button";
      trig.className = "frosted-trigger";
      trig.innerHTML = `<span class="trig-label"></span><span class="trig-caret">▾</span>`;
      sel.parentElement.appendChild(trig);

      trig.addEventListener("click", (e) => {
        e.stopPropagation();
        const opts = Array.from(sel.options).map((o) => ({
          value: o.value,
          label: o.textContent,
          badge: o.dataset.badge || "",
        }));
        dropdownMgr.open({
          anchorEl: trig,
          ownerEl: trig,
          options: opts,
          selectedValue: sel.value,
          onSelect: (val) => {
            sel.value = val;
            syncLabel();
            sel.dispatchEvent(new Event("change", { bubbles: true }));
          },
        });
      });
    }

    function syncLabel() {
      const chosen = sel.options[sel.selectedIndex] || sel.options[0];
      trig.querySelector(".trig-label").textContent = chosen ? chosen.textContent : "Select...";
    }

    customSelectRegistry[selectId] = { refresh: syncLabel };
    syncLabel();
  }

  function refreshFrostedSelect(selectId) {
    if (customSelectRegistry[selectId]) {
      customSelectRegistry[selectId].refresh();
    } else {
      enhanceFrostedSelect(selectId);
    }
  }

  function setupModelFrostedCombobox() {
    const combo = $("model-combo");
    const inp = $("model-input");
    const chev = $("model-combo-btn");

    function openModelMenu(filterQuery = "") {
      const q = filterQuery.trim().toLowerCase();
      const filtered = state.modelChoices
        .filter((m) => !q || m.toLowerCase().includes(q))
        .map((m) => {
          let badge = "";
          if (m.includes("bolt")) badge = "FAST";
          else if (m.includes("lightning") || m.includes("flash")) badge = "BALANCED";
          else if (m.includes("ultra") || m.includes("pro") || m.startsWith("o")) badge = "REASONING";
          return { value: m, label: m, badge };
        });

      dropdownMgr.activeOwner = null; // Allow live typing re-render without toggling shut
      dropdownMgr.open({
        anchorEl: combo,
        ownerEl: combo,
        options: filtered,
        selectedValue: inp.value.trim(),
        mono: true,
        onSelect: (val) => {
          inp.value = val;
        },
      });
    }

    chev.addEventListener("click", (e) => {
      e.stopPropagation();
      const portal = $("dropdown-portal");
      if (dropdownMgr.activeOwner === combo && !portal.classList.contains("hidden")) {
        dropdownMgr.close();
      } else {
        openModelMenu("");
        inp.focus();
      }
    });

    inp.addEventListener("focus", () => {
      openModelMenu("");
    });

    inp.addEventListener("input", () => {
      openModelMenu(inp.value);
    });

    inp.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        dropdownMgr.moveFocus(1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        dropdownMgr.moveFocus(-1);
      } else if (e.key === "Enter") {
        if (dropdownMgr.commitFocused()) e.preventDefault();
        else dropdownMgr.close();
      } else if (e.key === "Escape") {
        dropdownMgr.close();
      }
    });
  }

  // Close portal on outside click / resize / scroll
  document.addEventListener("mousedown", (e) => {
    const portal = $("dropdown-portal");
    if (portal.classList.contains("hidden")) return;
    if (portal.contains(e.target)) return;
    if (dropdownMgr.activeOwner && dropdownMgr.activeOwner.contains(e.target)) return;
    dropdownMgr.close();
  });
  window.addEventListener("resize", () => dropdownMgr.reposition());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") dropdownMgr.close();
  });

  // =========================================================================
  // HELPERS & INITIALIZATION
  // =========================================================================
  function showToast(msg, isError = false) {
    const container = $("toast-container");
    const el = document.createElement("div");
    el.className = `toast${isError ? " error" : ""}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.2s ease";
      setTimeout(() => el.remove(), 220);
    }, 3000);
  }

  function appendLog(line) {
    if (!line) return;
    const clean = line.replace(/\r/g, "").trim();
    if (!clean) return;
    const pre = $("console-log");
    pre.textContent += `\n${clean}`;
    pre.scrollTop = pre.scrollHeight;
    $("ticker-text").textContent = clean.split("\n").pop();
  }

  function formatTime(sec) {
    if (!isFinite(sec) || sec < 0) return "00:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function shortenGpuLabel(raw) {
    if (!raw) return "CPU";
    return raw
      .replace("CUDA GPU Active:", "")
      .replace("NVIDIA GeForce ", "")
      .replace("Laptop GPU", "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function setupGlassSpotlight() {
    document.querySelectorAll(".spotlight-card").forEach((card) => {
      card.addEventListener("mousemove", (e) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${e.clientX - rect.left}px`);
        card.style.setProperty("--my", `${e.clientY - rect.top}px`);
      });
    });
  }

  async function initSystem() {
    enhanceFrostedSelect("soundscape-mode");

    try {
      const res = await fetch("/api/system");
      const data = await res.json();
      state.sys = data;

      $("gpu-text").textContent = shortenGpuLabel(data.gpu);
      $("gpu-pill").title = data.gpu;
      $("disk-text").textContent = `${data.free_gb} GB free`;

      // Populate Narrator Voices with clean badge metadata
      const voiceSel = $("narrator-voice");
      voiceSel.innerHTML = "";
      const accentShort = { american: "US", british: "UK", indian: "IN" };
      (data.voices || []).forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.id;
        const acc = accentShort[v.accent] || v.accent.toUpperCase();
        const gen = v.gender === "female" ? "F" : "M";
        const firstTrait = (v.traits || "").split(",")[0].trim();
        opt.textContent = `${v.id} (${firstTrait})`;
        opt.dataset.badge = `${acc} · ${gen}`;
        if (v.id === data.default_voice) opt.selected = true;
        voiceSel.appendChild(opt);
      });
      refreshFrostedSelect("narrator-voice");

      // Populate Providers
      const provSel = $("provider-select");
      provSel.innerHTML = "";
      (data.providers || []).forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p.replace(" - Optional", "");
        if (p.includes("NVIDIA")) opt.dataset.badge = "NIM";
        else if (p.includes("Gemini")) opt.dataset.badge = "GOOGLE";
        else if (p.includes("OpenAI")) opt.dataset.badge = "CLOUD";
        else opt.dataset.badge = "LOCAL";
        provSel.appendChild(opt);
      });
      refreshFrostedSelect("provider-select");

      if (data.standalone_prompt) {
        state.promptParts = [{ part: 1, total: 1, prompt: data.standalone_prompt }];
        state.activePromptIdx = 0;
        renderPromptParts();
      }

      applyProviderDefaults(provSel.value);
      drawStandbyWaveform();
    } catch (err) {
      appendLog(`⚠️ Connection error: ${err.message}`);
    }
  }

  function applyProviderDefaults(provider) {
    if (!state.sys) return;
    state.modelChoices = [...((state.sys.provider_models || {})[provider] || [])];
    $("model-input").value = state.modelChoices[0] || "";
    $("base-url-input").value = (state.sys.base_urls || {})[provider] || "";

    const isOllama = provider.toLowerCase().includes("ollama");
    const keyInput = $("api-key-input");
    const keyField = $("api-key-field");
    if (isOllama) {
      keyInput.value = "";
      keyInput.disabled = true;
      keyInput.placeholder = "Local Ollama (no key required)";
      keyField.style.opacity = "0.5";
    } else {
      keyInput.disabled = false;
      keyField.style.opacity = "1";
      keyInput.value = (state.sys.saved_keys || {})[provider] || "";
      const placeholders = {
        "NVIDIA NIM API": "nvapi-...",
        "Google Gemini API": "AIza...",
        "OpenAI API": "sk-...",
      };
      keyInput.placeholder = placeholders[provider] || "API Key";
    }
  }

  // 2. Controls, Switchers & Inspector Tabs
  function setupUIControls() {
    setupModelFrostedCombobox();

    document.querySelectorAll(".mode-tab").forEach((btn, idx) => {
      btn.addEventListener("click", () => {
        dropdownMgr.close();
        document.querySelectorAll(".mode-tab").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        $("mode-glider").style.transform = `translateX(${idx * 100}%)`;

        const tab = btn.dataset.tab;
        $("panel-auto").classList.toggle("active", tab === "auto");
        $("panel-byoai").classList.toggle("active", tab === "byoai");
      });
    });

    document.querySelectorAll(".insp-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".insp-tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.insp;
        $("insp-cast").classList.toggle("active", target === "cast");
        $("insp-console").classList.toggle("active", target === "console");
        $("insp-json").classList.toggle("active", target === "json");
        $("btn-clear-log").classList.toggle("hidden", target !== "console");
        $("btn-copy-active-json").classList.toggle("hidden", target !== "json");
      });
    });

    $("toggle-adv-btn").addEventListener("click", () => {
      $("adv-drawer").classList.toggle("open");
    });

    $("toggle-embed-story-btn").addEventListener("click", () => {
      const dr = $("embed-story-drawer");
      dr.classList.toggle("open");
      $("toggle-embed-story-btn").textContent = dr.classList.contains("open")
        ? "− Hide Story Embed"
        : "+ Embed Story (Optional)";
    });

    $("bgm-volume").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      $("bgm-volume-val").textContent = `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB`;
    });
    $("chunk-size-auto").addEventListener("input", (e) => {
      $("chunk-size-auto-val").textContent = `${e.target.value} chars`;
    });
    $("prompt-chunk-size").addEventListener("input", (e) => {
      $("prompt-chunk-val").textContent = `${e.target.value}c`;
    });

    $("sfx-toggle").addEventListener("click", () => {
      state.sfxEnabled = !state.sfxEnabled;
      $("sfx-toggle").classList.toggle("active", state.sfxEnabled);
      $("sfx-toggle").setAttribute("aria-checked", String(state.sfxEnabled));
    });

    $("provider-select").addEventListener("change", (e) => applyProviderDefaults(e.target.value));
    $("toggle-key-vis").addEventListener("click", () => {
      const inp = $("api-key-input");
      const show = inp.type === "password";
      inp.type = show ? "text" : "password";
      $("toggle-key-vis").textContent = show ? "Hide" : "Show";
    });

    $("fetch-models-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      const btn = $("fetch-models-btn");
      btn.classList.add("spinning");
      try {
        const res = await fetch("/api/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: $("provider-select").value,
            api_key: $("api-key-input").value,
            base_url: $("base-url-input").value,
          }),
        });
        const data = await res.json();
        if (data.models?.length) {
          state.modelChoices = data.models;
          $("model-input").value = data.models[0];
          showToast(`${data.models.length} models loaded`);
          $("model-combo-btn").click();
        }
      } catch (_) {
        showToast("Could not fetch models", true);
      } finally {
        btn.classList.remove("spinning");
      }
    });

    $("free-vram-btn").addEventListener("click", async () => {
      const res = await fetch("/api/vram/free", { method: "POST" });
      const data = await res.json();
      if (data.gpu) $("gpu-text").textContent = shortenGpuLabel(data.gpu);
      appendLog("🧹 Cleared cached models from GPU VRAM.");
      showToast("VRAM Cleared");
    });

    $("btn-clear-log").addEventListener("click", () => {
      $("console-log").textContent = "● Cleared.";
    });
  }

  // 3. Manuscript Drag & Drop
  function wireManuscriptBox(boxId, browseBtnId, inputId, idleId, loadedId, nameId, statsId, clearId, slotKey) {
    const box = $(boxId);
    const input = $(inputId);

    $(browseBtnId).addEventListener("click", () => input.click());

    ["dragenter", "dragover"].forEach((ev) => {
      box.addEventListener(ev, (e) => {
        e.preventDefault();
        box.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      box.addEventListener(ev, (e) => {
        e.preventDefault();
        box.classList.remove("dragover");
      });
    });

    box.addEventListener("drop", (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) uploadBookFile(file, idleId, loadedId, nameId, statsId, slotKey);
    });

    input.addEventListener("change", () => {
      const file = input.files?.[0];
      if (file) uploadBookFile(file, idleId, loadedId, nameId, statsId, slotKey);
    });

    $(clearId).addEventListener("click", () => {
      state[slotKey] = null;
      input.value = "";
      $(loadedId).classList.add("hidden");
      $(idleId).classList.remove("hidden");
    });
  }

  async function uploadBookFile(file, idleId, loadedId, nameId, statsId, slotKey) {
    try {
      $(nameId).textContent = `Uploading ${file.name}...`;
      $(statsId).textContent = "";
      $(idleId).classList.add("hidden");
      $(loadedId).classList.remove("hidden");

      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "X-Filename": encodeURIComponent(file.name) },
        body: file,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");

      state[slotKey] = data.file_path;
      $(nameId).textContent = data.filename;
      $(statsId).textContent = `${data.word_count.toLocaleString()} words`;
      if (slotKey === "uploadedPromptFile" && data.title) {
        $("project-title-input").value = data.title;
      }
      appendLog(`📄 Attached '${data.filename}' (${data.char_count.toLocaleString()} chars).`);
      showToast(`Attached ${data.filename}`);
    } catch (err) {
      state[slotKey] = null;
      $(loadedId).classList.add("hidden");
      $(idleId).classList.remove("hidden");
      showToast(err.message, true);
    }
  }

  // 4. Story-Free Standalone Prompt + Paste JSON
  async function fetchStandaloneOrSplitPrompt(forceStandalone = false) {
    const res = await fetch("/api/prompts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_path: forceStandalone ? null : state.uploadedPromptFile,
        raw_text: forceStandalone ? "" : $("raw-text-prompt").value,
        custom_direction: $("custom-direction-input").value,
        chunk_chars: parseInt($("prompt-chunk-size").value, 10),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prompt generation failed");
    state.promptParts = data.parts || [];
    state.activePromptIdx = 0;
    renderPromptParts();
    return state.promptParts[0]?.prompt || "";
  }

  function setupBYOAIWorkflow() {
    $("btn-copy-standalone").addEventListener("click", async () => {
      try {
        const promptText = await fetchStandaloneOrSplitPrompt(true);
        await navigator.clipboard.writeText(promptText);
        const btn = $("btn-copy-standalone");
        btn.textContent = "✓ Copied!";
        showToast("Prompt copied! Attach your story in your AI & paste this prompt.");
        setTimeout(() => (btn.textContent = "Copy Prompt"), 1800);
      } catch (err) {
        showToast(err.message, true);
      }
    });

    $("btn-preview-prompt").addEventListener("click", async () => {
      const box = $("prompt-result-box");
      if (box.classList.contains("hidden")) {
        await fetchStandaloneOrSplitPrompt(true);
        box.classList.remove("hidden");
        $("btn-preview-prompt").textContent = "Hide";
      } else {
        box.classList.add("hidden");
        $("btn-preview-prompt").textContent = "Preview";
      }
    });

    $("btn-gen-prompts").addEventListener("click", async () => {
      try {
        await fetchStandaloneOrSplitPrompt(false);
        $("prompt-result-box").classList.remove("hidden");
        $("btn-preview-prompt").textContent = "Hide";
        showToast(`Built ${state.promptParts.length} prompt part(s)`);
      } catch (err) {
        showToast(err.message, true);
      }
    });

    $("btn-copy-prompt").addEventListener("click", async () => {
      const txt = $("generated-prompt-text").value;
      if (!txt) return;
      await navigator.clipboard.writeText(txt);
      $("btn-copy-prompt").textContent = "✓ Copied";
      setTimeout(() => ($("btn-copy-prompt").textContent = "Copy Shown Prompt"), 1600);
    });

    $("btn-copy-active-json").addEventListener("click", async () => {
      const txt = $("active-script-json").value;
      if (!txt) return;
      await navigator.clipboard.writeText(txt);
      showToast("JSON copied");
    });

    $("btn-load-sample-json").addEventListener("click", () => {
      const demo = {
        characters: [
          { name: "Narrator", voice_profile: "Warm cinematic storyteller", gender: "female", age_group: "adult", accent: "american" },
          { name: "Aarav", voice_profile: "Indian 30y man, calm & resolute", gender: "male", age_group: "adult", accent: "indian" },
          { name: "Lily", voice_profile: "10y old curious girl, bright", gender: "female", age_group: "child", accent: "american" },
          { name: "Inspector Croft", voice_profile: "68y grumpy British man", gender: "male", age_group: "elderly", accent: "british" },
        ],
        blocks: [
          {
            speaker: "Narrator",
            text: "Midnight rain whispered against the study windows when three sharp knocks echoed down the hall.",
            emotion: "tense",
            bg_music: "suspense_tense",
            music_intensity: 0.6,
            ambience: "gentle_rain",
            sfx: "door_knock",
            sfx_timing: "after",
            pause_after_ms: 420,
          },
          {
            speaker: "Lily",
            text: "Aarav, did you hear that? Someone is outside the library door!",
            emotion: "whisper",
            bg_music: "suspense_tense",
            music_intensity: 0.72,
            ambience: "gentle_rain",
            sfx: "door_creak",
            sfx_timing: "before",
            pause_after_ms: 280,
          },
          {
            speaker: "Inspector Croft",
            text: "Put that lantern down! Nobody leaves this manor until I have my answers.",
            emotion: "grumpy",
            bg_music: "intense",
            music_intensity: 0.85,
            ambience: "gentle_rain",
            sfx: "thunder_clap",
            sfx_timing: "before",
            pause_after_ms: 400,
          },
          {
            speaker: "Aarav",
            text: "Relax, Inspector. Look at the seal on the letter—the mystery is solved, and everyone here is safe.",
            emotion: "happy",
            bg_music: "warm_heartfelt",
            music_intensity: 0.55,
            ambience: "crackling_fireplace",
            sfx: "magic_chime",
            sfx_timing: "after",
            pause_after_ms: 480,
          },
        ],
      };
      $("pasted-json-input").value = JSON.stringify(demo, null, 2);
      $("btn-validate-json").click();
    });

    $("btn-validate-json").addEventListener("click", async () => {
      try {
        const res = await fetch("/api/validate-json", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pasted_json: $("pasted-json-input").value,
            narrator_voice: $("narrator-voice").value,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Invalid JSON");
        renderCastOverview(data.overview);
        $("active-script-json").value = data.normalized_json || "";
        document.querySelector('.insp-tab[data-insp="cast"]').click();
        showToast(`${data.overview.cast.length} voices & ${data.overview.total_blocks} blocks ready`);
      } catch (err) {
        showToast(err.message, true);
        appendLog(`❌ JSON Error: ${err.message}`);
      }
    });
  }

  function renderPromptParts() {
    const wrap = $("prompt-part-pills");
    wrap.innerHTML = "";
    if (state.promptParts.length <= 1) {
      const span = document.createElement("span");
      span.className = "part-pill active";
      span.textContent = "Universal Story-Free Prompt";
      wrap.appendChild(span);
    } else {
      state.promptParts.forEach((p, idx) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `part-pill${idx === state.activePromptIdx ? " active" : ""}`;
        b.textContent = `Part ${p.part}/${p.total}`;
        b.addEventListener("click", () => {
          state.activePromptIdx = idx;
          renderPromptParts();
        });
        wrap.appendChild(b);
      });
    }
    const active = state.promptParts[state.activePromptIdx];
    $("generated-prompt-text").value = active ? active.prompt : "";
  }

  // 5. Cast & Cues Inspector View
  function renderCastOverview(overview) {
    if (!overview) return;
    $("cast-empty").classList.add("hidden");
    $("cast-content").classList.remove("hidden");
    $("cast-count-badge").textContent = String((overview.cast || []).length);

    const clusters = $("cue-clusters");
    clusters.innerHTML = "";

    (overview.bgm_themes || []).forEach((bgm) => {
      const el = document.createElement("span");
      el.className = "cue-chip";
      el.innerHTML = `<strong>BGM</strong>${bgm}`;
      clusters.appendChild(el);
    });
    (overview.ambiences || []).forEach((amb) => {
      const el = document.createElement("span");
      el.className = "cue-chip";
      el.innerHTML = `<strong>AMB</strong>${amb}`;
      clusters.appendChild(el);
    });
    (overview.sfx_cues || []).slice(0, 6).forEach((s) => {
      const el = document.createElement("span");
      el.className = "cue-chip";
      el.innerHTML = `<strong>SFX·${s.timing}</strong>${s.cue}`;
      clusters.appendChild(el);
    });

    const list = $("cast-list");
    list.innerHTML = "";
    (overview.cast || []).forEach((c) => {
      const row = document.createElement("div");
      row.className = "cast-row";

      const left = document.createElement("div");
      left.className = "cast-left";
      const av = document.createElement("div");
      av.className = "avatar";
      av.textContent = c.initials;

      const meta = document.createElement("div");
      meta.className = "cast-meta";
      const st = document.createElement("strong");
      st.textContent = c.speaker;
      const sp = document.createElement("span");
      sp.textContent = c.profile;
      meta.appendChild(st);
      meta.appendChild(sp);
      left.appendChild(av);
      left.appendChild(meta);

      const right = document.createElement("div");
      right.className = "cast-right";
      const vp = document.createElement("span");
      vp.className = "voice-pill";
      vp.textContent = c.summary.split("|")[0].trim();
      const lc = document.createElement("span");
      lc.className = "line-count";
      lc.textContent = `${c.lines} lines`;
      right.appendChild(vp);
      right.appendChild(lc);

      row.appendChild(left);
      row.appendChild(right);
      list.appendChild(row);
    });
  }

  // 6. Pipeline Execution & SSE
  function setRunningState(running) {
    state.isRunning = running;
    $("btn-run-auto").disabled = running;
    $("btn-run-json").disabled = running;
    $("btn-stop-auto").classList.toggle("hidden", !running);
    $("btn-stop-json").classList.toggle("hidden", !running);
    $("progress-fill").classList.toggle("running", running);
    document.querySelector(".live-ticker").classList.toggle("busy", running);
  }

  function updateStepperUI(stageNum, progressFrac, labelOverride = null) {
    const pct = Math.round((progressFrac || 0) * 100);
    $("progress-fill").style.width = `${pct}%`;
    $("progress-pct").textContent = labelOverride || `${pct}%`;

    document.querySelectorAll(".st-item").forEach((node) => {
      const s = parseInt(node.dataset.stage, 10);
      node.classList.toggle("done", s < stageNum || pct >= 100);
      node.classList.toggle("active", s === stageNum && pct < 100);
    });
  }

  function subscribeJobStream(jobId) {
    if (state.activeEventSource) state.activeEventSource.close();
    const es = new EventSource(`/api/jobs/${jobId}/stream`);
    state.activeEventSource = es;
    let finishedCleanly = false;

    es.addEventListener("progress", (ev) => {
      const d = JSON.parse(ev.data);
      updateStepperUI(d.stage, d.progress);
      if (d.log) appendLog(d.log);
      if (d.overview) renderCastOverview(d.overview);
      if (d.script_json) $("active-script-json").value = d.script_json;
    });

    es.addEventListener("complete", (ev) => {
      finishedCleanly = true;
      const d = JSON.parse(ev.data);
      updateStepperUI(4, 1.0, "100%");
      setRunningState(false);
      es.close();
      if (d.audio_url) {
        loadMasterAudio(d.audio_url, d.audio_filename || "audiobook.mp3");
        showToast("Audiobook Master Ready");
      }
    });

    es.addEventListener("job_error", (ev) => {
      finishedCleanly = true;
      let errMsg = "Pipeline failed.";
      try {
        if (ev.data) errMsg = JSON.parse(ev.data).message || errMsg;
      } catch (_) {}
      appendLog(`❌ ${errMsg}`);
      updateStepperUI(0, 0, "ERROR");
      setRunningState(false);
      es.close();
      showToast(errMsg, true);
    });

    es.onerror = () => {
      if (finishedCleanly) return;
      if (es.readyState === EventSource.CLOSED) setRunningState(false);
    };
  }

  function setupConversionActions() {
    $("btn-run-auto").addEventListener("click", async () => {
      if (state.isRunning) return;
      const rawText = $("raw-text-auto").value.trim();
      if (!state.uploadedAutoFile && !rawText) {
        showToast("Attach a PDF/TXT file or paste story text first", true);
        return;
      }

      setRunningState(true);
      updateStepperUI(1, 0.02);
      try {
        const res = await fetch("/api/convert/auto", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_path: state.uploadedAutoFile,
            raw_text: rawText,
            provider: $("provider-select").value,
            model_name: $("model-input").value,
            api_key: $("api-key-input").value,
            save_key: $("save-key-chk").checked,
            base_url: $("base-url-input").value,
            chunk_chars: parseInt($("chunk-size-auto").value, 10),
            soundscape_mode: $("soundscape-mode").value,
            narrator_voice: $("narrator-voice").value,
            bgm_volume_db: parseFloat($("bgm-volume").value),
            enable_sfx: state.sfxEnabled,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to start");
        subscribeJobStream(data.job_id);
      } catch (err) {
        setRunningState(false);
        showToast(err.message, true);
      }
    });

    $("btn-run-json").addEventListener("click", async () => {
      if (state.isRunning) return;
      const pasted = $("pasted-json-input").value.trim();
      if (!pasted) {
        showToast("Paste Production JSON first", true);
        return;
      }

      setRunningState(true);
      updateStepperUI(2, 0.05);
      try {
        const res = await fetch("/api/convert/json", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pasted_json: pasted,
            project_title: $("project-title-input").value,
            soundscape_mode: $("soundscape-mode").value,
            narrator_voice: $("narrator-voice").value,
            bgm_volume_db: parseFloat($("bgm-volume").value),
            enable_sfx: state.sfxEnabled,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to start");
        subscribeJobStream(data.job_id);
      } catch (err) {
        setRunningState(false);
        showToast(err.message, true);
      }
    });

    const requestStop = async () => {
      await fetch("/api/jobs/stop", { method: "POST" });
      appendLog("🛑 Stopping after current block...");
    };
    $("btn-stop-auto").addEventListener("click", requestStop);
    $("btn-stop-json").addEventListener("click", requestStop);
  }

  // 7. Waveform Monitor & Studio Scrubber
  function drawStandbyWaveform() {
    const canvas = $("waveform-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const bars = 88;
    const gap = 4;
    const bw = (w - gap * (bars - 1)) / bars;
    ctx.fillStyle = "rgba(255, 255, 255, 0.14)";
    for (let i = 0; i < bars; i++) {
      const env = Math.sin((i / bars) * Math.PI);
      const wave = Math.abs(Math.sin(i * 0.28) * Math.cos(i * 0.11));
      const bh = Math.max(2, (wave * 26 + 3) * env + 2);
      ctx.fillRect(i * (bw + gap), (h - bh) / 2, bw, bh);
    }
  }

  function ensureAudioGraph() {
    if (state.audioCtx) return;
    const AudioContextCls = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCls) return;
    state.audioCtx = new AudioContextCls();
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 256;
    state.analyser.smoothingTimeConstant = 0.82;
    state.sourceNode = state.audioCtx.createMediaElementSource($("studio-audio"));
    state.sourceNode.connect(state.analyser);
    state.analyser.connect(state.audioCtx.destination);
  }

  function renderLiveSpectrum() {
    const audio = $("studio-audio");
    const canvas = $("waveform-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    if (!state.analyser || audio.paused) return;

    const freqData = new Uint8Array(state.analyser.frequencyBinCount);
    state.analyser.getByteFrequencyData(freqData);
    ctx.clearRect(0, 0, w, h);

    const bars = 84;
    const gap = 4;
    const bw = (w - gap * (bars - 1)) / bars;
    for (let i = 0; i < bars; i++) {
      const bin = Math.floor((i / bars) * (freqData.length * 0.72));
      const val = freqData[bin] / 255.0;
      const bh = Math.max(2, val * (h - 12));
      ctx.fillStyle = `rgba(255, 255, 255, ${(0.28 + val * 0.72).toFixed(2)})`;
      ctx.fillRect(i * (bw + gap), (h - bh) / 2, bw, bh);
    }
    state.rafId = requestAnimationFrame(renderLiveSpectrum);
  }

  function loadMasterAudio(audioUrl, filename) {
    const audio = $("studio-audio");
    audio.src = audioUrl;
    audio.load();

    $("track-title").textContent = filename;
    $("track-title").style.color = "#ffffff";
    const tag = $("player-state-tag");
    tag.textContent = "READY";
    tag.classList.add("ready");

    $("seek-slider").disabled = false;
    $("btn-play-pause").disabled = false;
    $("btn-skip-back").disabled = false;
    $("btn-skip-fwd").disabled = false;

    const ext = filename.toLowerCase().endsWith(".wav") ? "WAV" : "MP3";
    const dlBtn = $("download-audio-btn");
    dlBtn.textContent = `↓ ${ext}`;
    dlBtn.href = `${audioUrl.split("?")[0]}?download=1`;
    dlBtn.setAttribute("download", filename);
    dlBtn.classList.remove("disabled");
  }

  function setupStudioPlayer() {
    const audio = $("studio-audio");
    const seek = $("seek-slider");
    const fill = $("scrub-fill");
    const playIcon = $("play-icon");
    const rateButtons = document.querySelectorAll("#speed-pills button");
    rateButtons[0].classList.add("active");

    $("btn-play-pause").addEventListener("click", async () => {
      ensureAudioGraph();
      if (state.audioCtx?.state === "suspended") await state.audioCtx.resume();
      if (audio.paused) await audio.play();
      else audio.pause();
    });

    audio.addEventListener("play", () => {
      playIcon.textContent = "❚❚";
      cancelAnimationFrame(state.rafId);
      renderLiveSpectrum();
    });
    audio.addEventListener("pause", () => {
      playIcon.textContent = "▶";
      cancelAnimationFrame(state.rafId);
    });
    audio.addEventListener("ended", () => {
      playIcon.textContent = "▶";
      cancelAnimationFrame(state.rafId);
    });
    audio.addEventListener("loadedmetadata", () => {
      $("time-duration").textContent = formatTime(audio.duration);
    });
    audio.addEventListener("timeupdate", () => {
      $("time-current").textContent = formatTime(audio.currentTime);
      if (audio.duration > 0) {
        const pct = (audio.currentTime / audio.duration) * 100;
        fill.style.width = `${pct}%`;
        seek.value = String(Math.round(pct * 10));
      }
    });

    seek.addEventListener("input", () => {
      if (audio.duration > 0) {
        const pct = parseFloat(seek.value) / 10;
        fill.style.width = `${pct}%`;
        audio.currentTime = (pct / 100) * audio.duration;
      }
    });

    $("btn-skip-back").addEventListener("click", () => {
      audio.currentTime = Math.max(0, audio.currentTime - 10);
    });
    $("btn-skip-fwd").addEventListener("click", () => {
      if (audio.duration) audio.currentTime = Math.min(audio.duration, audio.currentTime + 10);
    });

    rateButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        rateButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        audio.playbackRate = parseFloat(btn.dataset.rate);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupGlassSpotlight();
    setupUIControls();
    wireManuscriptBox(
      "dropzone-auto",
      "btn-browse-auto",
      "file-input-auto",
      "dz-auto-idle",
      "dz-auto-loaded",
      "dz-auto-name",
      "dz-auto-stats",
      "dz-auto-clear",
      "uploadedAutoFile"
    );
    wireManuscriptBox(
      "dropzone-prompt",
      "btn-browse-prompt",
      "file-input-prompt",
      "dz-prompt-idle",
      "dz-prompt-loaded",
      "dz-prompt-name",
      "dz-prompt-stats",
      "dz-prompt-clear",
      "uploadedPromptFile"
    );
    setupBYOAIWorkflow();
    setupConversionActions();
    setupStudioPlayer();
    initSystem();
  });
})();
