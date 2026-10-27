(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const page = document.body.dataset.page;

  // --- Création ---
  if (page === "create") {
    const form = $("#create-form");
    const input = $("#image-input");
    const zone = $("#dropzone");
    const range = $("#colors");
    const out = $("#colors-value");
    const btn = $("#submit-btn");
    const err = $("#form-error");
    let file = null;

    const syncRange = () => {
      out.textContent = range.value;
      const p = ((range.value - range.min) / (range.max - range.min)) * 100;
      range.style.setProperty("--p", `${p}%`);
      renderPalette();
    };

    // Pastilles des couleurs retenues pour la valeur courante de la jauge ; clic = menu "Supprimer cette couleur"
    const paletteEl = $("#palette");
    const removedEl = $("#palette-removed");
    const menu = $("#swatch-menu");
    let palettes = null;
    let excluded = [];
    let menuColor = null;

    const swatch = (c, title) => {
      const b = document.createElement("button");
      b.type = "button";
      b.style.setProperty("--c", c);
      b.title = title;
      b.setAttribute("aria-label", title);
      return b;
    };
    const currentColors = () => palettes?.[range.value] || [];

    function renderPalette() {
      const colors = currentColors();
      paletteEl.hidden = !colors.length;
      paletteEl.replaceChildren(...colors.map((c) => {
        const b = swatch(c, `${c} – cliquer pour supprimer cette couleur`);
        b.addEventListener("click", (e) => { e.stopPropagation(); openMenu(b, c); });
        return b;
      }));
      removedEl.hidden = !excluded.length;
      removedEl.replaceChildren("Supprimées :", ...excluded.map((ex, i) => {
        const b = swatch(ex.color, `${ex.color} – cliquer pour rétablir cette couleur`);
        b.addEventListener("click", () => restoreColor(i));
        return b;
      }));
    }

    function openMenu(anchor, color) {
      menuColor = color;
      const field = $("#colors-field").getBoundingClientRect();
      const r = anchor.getBoundingClientRect();
      menu.style.left = `${r.left - field.left}px`;
      menu.style.top = `${r.bottom - field.top + 6}px`;
      $("button", menu).disabled = currentColors().length <= 1;
      menu.hidden = false;
      $("button", menu).focus();
    }
    const closeMenu = () => { menu.hidden = true; menuColor = null; };
    document.addEventListener("click", (e) => { if (!menu.contains(e.target)) closeMenu(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

    $("button", menu).addEventListener("click", () => {
      const colors = currentColors();
      if (!menuColor || colors.length <= 1) return;
      const others = colors.filter((c) => c !== menuColor);
      excluded.push({ color: menuColor, others });
      closeMenu();
      const n = Math.max(1, Math.min(+range.value, colors.length) - 1);
      palettes = { ...palettes, [n]: others };  // retour immédiat en attendant la nouvelle analyse
      range.value = n;
      userTouched = true;
      syncRange();
      analyse();
    });

    function restoreColor(i) {
      excluded.splice(i, 1);
      range.value = Math.min(+range.max, +range.value + 1);
      userTouched = true;
      syncRange();
      analyse();
    }
    range.addEventListener("input", syncRange);
    syncRange();

    const showError = (msg) => { err.textContent = msg; err.hidden = !msg; };
    const bgValue = () => $("input[name=background]:checked", form).value;

    // Pré-positionne la jauge selon les couleurs détectées, sauf si l'utilisateur l'a déjà réglée
    const hint = $("#colors-hint");
    const defaultHint = hint.textContent;
    let analyseId = 0;
    let userTouched = false;
    let sizeTouched = false;
    const sizeInputs = [$("#width-mm"), $("#height-mm")];
    range.addEventListener("input", () => { userTouched = true; });
    sizeInputs.forEach((i) => i.addEventListener("input", () => { sizeTouched = true; }));

    const busy = (on) => {
      $("#colors-field").classList.toggle("is-busy", on && !userTouched);
      $("#size-field").classList.toggle("is-busy", on && !sizeTouched);
    };

    const analyse = async () => {
      if (!file) return;
      const id = ++analyseId;
      hint.textContent = "Analyse des couleurs de l'image…";
      busy(true);
      const data = new FormData();
      data.append("image", file);
      data.append("background", bgValue());
      data.append("exclude", JSON.stringify(excluded));
      try {
        const res = await fetch(form.dataset.analyse, { method: "POST", body: data });
        const json = await res.json();
        if (id !== analyseId) return;
        if (!res.ok) throw new Error(json.error);
        palettes = json.palettes;
        if (!userTouched) range.value = json.colors;
        syncRange();
        if (!sizeTouched) { sizeInputs[0].value = json.width_mm; sizeInputs[1].value = json.height_mm; }
        const s = json.colors > 1 ? "s" : "";
        const bgNote = bgValue() === "keep" ? "" : " (fond exclu)";
        hint.textContent = `${json.colors} couleur${s} distincte${s} détectée${s}${bgNote}. Chaque couleur correspond à un fil.`;
      } catch {
        if (id === analyseId) hint.textContent = defaultHint;
      } finally {
        if (id === analyseId) busy(false);
      }
    };
    $$("input[name=background]", form).forEach((r) => r.addEventListener("change", analyse));

    const setFile = (f) => {
      if (!f) return;
      if (!/^image\/(png|jpeg|webp|avif)$/.test(f.type) && !/\.(avif|webp)$/i.test(f.name)) {
        showError("Format non pris en charge : choisissez une image PNG, JPG, WebP ou AVIF.");
        return;
      }
      showError("");
      file = f;
      const img = $("#image-preview");
      if (img.src) URL.revokeObjectURL(img.src);
      img.src = URL.createObjectURL(f);
      $("#image-name").textContent = f.name;
      $(".dz-empty", zone).hidden = true;
      $(".dz-preview", zone).hidden = false;
      zone.classList.add("filled");
      btn.disabled = false;
      userTouched = false;
      sizeTouched = false;
      palettes = null;
      excluded = [];
      renderPalette();
      analyse();
    };

    input.addEventListener("change", () => setFile(input.files[0]));
    $("#image-clear").addEventListener("click", (e) => { e.preventDefault(); input.click(); });

    ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
    zone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
    document.addEventListener("paste", (e) => {
      const f = [...(e.clipboardData?.files || [])][0];
      if (f) setFile(f);
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!file) return;
      if (!sizeInputs.every((i) => i.reportValidity())) return;
      showError("");
      btn.disabled = true;
      btn.classList.add("loading");
      $(".btn-label", btn).textContent = "Envoi…";
      const data = new FormData();
      data.append("image", file);
      data.append("colors", range.value);
      data.append("background", bgValue());
      data.append("width_mm", sizeInputs[0].value);
      data.append("height_mm", sizeInputs[1].value);
      data.append("exclude", JSON.stringify(excluded));
      data.append("thicken_lines", $("#thicken-lines").checked ? "true" : "false");
      try {
        const res = await fetch(form.action, { method: "POST", body: data });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "Une erreur est survenue.");
        window.location.href = json.urls.page;
      } catch (ex) {
        showError(ex.message);
        btn.disabled = false;
        btn.classList.remove("loading");
        $(".btn-label", btn).textContent = "Générer le motif";
      }
    });
  }

  // --- Détail d'un motif ---
  if (page === "design") {
    // Renommer le motif (crayon à côté du titre)
    const titleBox = $("#title-edit");
    if (titleBox) {
      const h1 = $("#design-title"), form = $("#title-form"), input = $("#title-input");
      const editBtn = $("#title-edit-btn"), err = $("#title-error");
      const toggle = (on) => {
        form.hidden = !on;
        h1.hidden = editBtn.hidden = on;
        err.hidden = true;
        if (on) { input.value = h1.textContent; input.focus(); input.select(); }
      };
      editBtn.addEventListener("click", () => toggle(true));
      $("#title-cancel").addEventListener("click", () => toggle(false));
      input.addEventListener("keydown", (e) => { if (e.key === "Escape") toggle(false); });
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        try {
          const res = await fetch(titleBox.dataset.url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: input.value }),
          });
          const json = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(json.error || "Le nom n'a pas pu être enregistré.");
          h1.textContent = json.name;
          document.title = `${json.name} · img2stitch`;
          toggle(false);
        } catch (ex) {
          err.textContent = ex.message;
          err.hidden = false;
        }
      });
    }
    const root = $("#design");
    if (!root) return;
    const tabs = $$(".segmented button", root);
    const imgs = $$(".stage img", root);

    const show = (view) => {
      tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
      imgs.forEach((i) => i.classList.toggle("shown", i.dataset.view === view));
    };
    tabs.forEach((t) => t.addEventListener("click", () => !t.disabled && show(t.dataset.view)));
    show(root.dataset.hasPhoto === "1" ? "photo" : root.dataset.hasRaw === "1" ? "raw" : "source");

    if (root.dataset.status === "processing") {
      const poll = async () => {
        try {
          const res = await fetch(root.dataset.api, { cache: "no-store" });
          const d = await res.json();
          const changed = d.status !== root.dataset.status
            || (d.urls.raw ? "1" : "0") !== root.dataset.hasRaw
            || (d.urls.photo ? "1" : "0") !== root.dataset.hasPhoto;
          if (changed) return window.location.reload();
          const step = $("#step");
          if (step && d.step) step.textContent = `${d.step}…`;
        } catch { /* réseau indisponible : on réessaie */ }
        setTimeout(poll, 1500);
      };
      setTimeout(poll, 1500);
    }

    // Choix du tissu : regénère l'aperçu réaliste côté serveur puis remplace l'image
    const fabrics = $("#fabrics");
    if (fabrics) {
      const buttons = $$(".fabric", fabrics);
      const progress = $("#photo-progress");
      const photo = $('.stage img[data-view="photo"]', root);
      let current = buttons.find((b) => b.getAttribute("aria-checked") === "true")?.dataset.fabric;

      const select = (name) => buttons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.fabric === name)));
      const setBusy = (on) => {
        progress.hidden = !on;
        fabrics.classList.toggle("is-busy", on);
      };
      const waitPhoto = async () => {
        try {
          const d = await (await fetch(root.dataset.api, { cache: "no-store" })).json();
          if (d.photo_rendering) return setTimeout(waitPhoto, 1000);
          select(d.fabric);
          current = d.fabric;
          if (d.photo_error || !d.urls.photo) return setBusy(false);
          photo.addEventListener("load", () => setBusy(false), { once: true });
          photo.src = d.urls.photo;
        } catch {
          setTimeout(waitPhoto, 2000);
        }
      };

      buttons.forEach((b) => b.addEventListener("click", async () => {
        if (b.dataset.fabric === current && progress.hidden) return;
        select(b.dataset.fabric);
        show("photo");
        setBusy(true);
        try {
          const res = await fetch(fabrics.dataset.url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fabric: b.dataset.fabric }),
          });
          if (!res.ok) throw new Error();
          current = b.dataset.fabric;
          waitPhoto();
        } catch {
          select(current);
          setBusy(false);
        }
      }));
      if (root.dataset.photoRendering === "1") {
        setBusy(true);
        waitPhoto();
      }
    }

    // Fils : pastille -> sélecteur de couleur (toute la broderie), fusion des couleurs identiques
    const threads = $("#threads");
    if (threads) {
      const err = $("#threads-error");
      const send = async (url, body) => {
        err.hidden = true;
        try {
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          const json = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(json.error || "La modification n'a pas pu être appliquée.");
          window.location.reload();
        } catch (ex) {
          err.textContent = ex.message;
          err.hidden = false;
        }
      };
      $$('input[type="color"]', threads).forEach((input) => {
        input.addEventListener("input", () => { input.previousElementSibling.style.setProperty("--c", input.value); });
        input.addEventListener("change", () => {
          if (input.value.toLowerCase() === input.dataset.from) return;
          send(threads.dataset.recolor, { from: input.dataset.from, to: input.value.toLowerCase() });
        });
      });
      $("#merge-btn")?.addEventListener("click", (e) => {
        e.currentTarget.disabled = true;
        send(threads.dataset.merge, { enabled: true });
      });
    }
  }

  // --- Galerie : rafraîchit tant que des motifs sont en cours ---
  if (page === "gallery" && $(".tile .badge.busy")) {
    setTimeout(() => window.location.reload(), 4000);
  }
  if (page === "gallery") {
    $$(".tile-delete").forEach((form) => form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!window.confirm(`Supprimer définitivement « ${form.dataset.name} » ?`)) return;
      const button = $("button", form);
      button.disabled = true;
      try {
        const res = await fetch(form.action, { method: "POST" });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "La suppression n'a pas pu être effectuée.");
        window.location.reload();
      } catch (ex) {
        window.alert(ex.message);
        button.disabled = false;
      }
    }));
  }

  // --- Édition d'un motif : clic sur une zone -> couleur / motif de remplissage ---
  if (page === "edit") {
    const root = $("#editor");
    const { zones, fills: fillList } = JSON.parse($("#editor-data").textContent);
    const fills = Object.fromEntries(fillList);
    const edits = zones.map((z) => ({ color: z.color, fill: z.fill }));
    const img = $("#editor-img");
    const overlay = $("#editor-overlay");
    const list = $("#zone-list");
    const menu = $("#zone-menu");
    const applyBtn = $("#edit-apply");
    const resetBtn = $("#edit-reset");
    const errEl = $("#edit-error");
    const loading = $("#editor-loading");
    const loadingText = $("#editor-loading-text");
    const loadingStep = $("#editor-loading-step");
    let zmap = null, zw = 0, zh = 0;
    // Sélection : {type: "block", i} (toute une couleur, depuis la liste) ou {type: "region", j} (zone fermée cliquée)
    let target = null, selMask = null;
    const pending = [];  // nouvelles zones fermées : {block, fx, fy, color, fill, mask}

    // Aperçus des motifs de remplissage (SVG statiques)
    const FILL_ICONS = {
      tatami: '<path d="M2 14 14 2M2 22 22 2M10 22l12-12M2 6l4-4M18 22l4-4"/>',
      tatami_h: '<path d="M3 5h18M3 9h18M3 13h18M3 17h18M3 21h18"/>',
      tatami_v: '<path d="M5 3v18M9 3v18M13 3v18M17 3v18M21 3v18"/>',
      cross: '<path d="M2 14 14 2M2 22 22 2M10 22l12-12M2 10 14 22M2 2l20 20M10 2l12 12"/>',
      satin: '<path d="M4 3v18M6 3v18M8 3v18M10 3v18M12 3v18M14 3v18M16 3v18M18 3v18M20 3v18"/>',
      satin_flow: '<path d="M4 20q2-14 16-16" stroke-width="5" stroke-opacity=".25"/><path d="m3 18 3 3M4 14l5 4M6 10l5 5M9 7l5 4M13 4l4 5M17 3l3 4"/>',
      contour: '<rect x="3" y="3" width="18" height="18" rx="3"/><rect x="7" y="7" width="10" height="10" rx="2"/><rect x="11" y="11" width="2" height="2"/>',
      wave: '<path d="M2 6q2.5-3 5 0t5 0 5 0 5 0M2 12q2.5-3 5 0t5 0 5 0 5 0M2 18q2.5-3 5 0t5 0 5 0 5 0"/>',
      zigzag: '<path d="m2 7 3-3 3 3 3-3 3 3 3-3 3 3 3-3M2 13l3-3 3 3 3-3 3 3 3-3 3 3 3-3M2 19l3-3 3 3 3-3 3 3 3-3 3 3 3-3"/>',
      none: '<rect x="3" y="3" width="18" height="18" rx="3" stroke-dasharray="3 2.5"/>',
      relief_stripes: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 3v18M12 3v18M16 3v18" stroke-dasharray="1 1.6"/>',
      relief_chevron: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m6 3 3 4.5L6 12l3 4.5L6 21M12 3l3 4.5-3 4.5 3 4.5-3 4.5" stroke-dasharray="1 1.6"/>',
      relief_waves: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 3q3 4.5 0 9t0 9M14 3q3 4.5 0 9t0 9" stroke-dasharray="1 1.6"/>',
      relief_bricks: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v6M15 3v6M6 9v6M12 9v6M18 9v6M9 15v6M15 15v6" stroke-dasharray="1 1.6"/>',
      relief_diamonds: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m3 9 6-6 12 12M3 15 15 3l6 6M9 21l12-12M3 9l12 12" stroke-dasharray="1 1.6"/>',
      relief_circles: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="3" stroke-dasharray="1 1.6"/><circle cx="16" cy="8" r="3" stroke-dasharray="1 1.6"/><circle cx="12" cy="15" r="3" stroke-dasharray="1 1.6"/>',
    };
    const icon = (k) => `<svg viewBox="0 0 24 24" aria-hidden="true">${FILL_ICONS[k] || ""}</svg>`;
    const changed = (i) => edits[i].color !== zones[i].color || edits[i].fill !== zones[i].fill;
    const regionChanged = (r) => r.color !== edits[r.block].color || r.fill !== edits[r.block].fill;
    const rgb = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
    const cur = () => (target?.type === "block" ? edits[target.i] : target ? pending[target.j] : null);
    const isSel = (t) => target && t.type === target.type && (t.i ?? t.j) === (target.i ?? target.j);
    const blockLabel = (i) => (zones[i].region !== null ? `Zone isolée ${i + 1}` : `Couleur ${i + 1}`);

    function row(t, label, base, now, isChanged) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "zone-row" + (isSel(t) ? " selected" : "") + (isChanged ? " changed" : "");
      const was = now.color !== base.color ? `<span class="swatch was" style="--c:${base.color}"></span><span class="arrow">→</span>` : "";
      b.innerHTML = `${was}<span class="swatch" style="--c:${now.color}"></span>
        <span class="zone-text"><strong>${label}</strong><small>${fills[now.fill]}</small></span>${icon(now.fill)}`;
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        const r = b.getBoundingClientRect();
        select(t, r.left + window.scrollX - 8, r.bottom + window.scrollY + 6);
      });
      li.append(b);
      return li;
    }

    // -- liste : couleurs entières, puis zones fermées nouvellement retouchées -------------
    function renderList() {
      const items = zones.map((z, i) => row({ type: "block", i }, blockLabel(i), z, edits[i], changed(i)));
      pending.forEach((r, j) => {
        if (regionChanged(r) || isSel({ type: "region", j })) {
          items.push(row({ type: "region", j }, `Zone fermée (${blockLabel(r.block).toLowerCase()})`, edits[r.block], r, true));
        }
      });
      list.replaceChildren(...items);
      const any = zones.some((_, i) => changed(i)) || pending.some(regionChanged);
      applyBtn.disabled = !any;
      resetBtn.disabled = !any;
    }

    // -- carte des zones et calque d'aperçu -------------------------------------------------
    const zimg = new Image();
    zimg.onload = () => {
      zw = zimg.naturalWidth; zh = zimg.naturalHeight;
      const c = document.createElement("canvas");
      c.width = zw; c.height = zh;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(zimg, 0, 0);
      const px = ctx.getImageData(0, 0, zw, zh).data;
      zmap = new Uint8Array(zw * zh);
      for (let i = 0; i < zmap.length; i++) zmap[i] = px[i * 4];
      $("#editor-loading").hidden = true;
      draw();
    };
    zimg.onerror = () => { $("#editor-loading").hidden = true; showError("Impossible de charger les zones du motif."); };
    zimg.src = root.dataset.zonemap;

    function draw() {
      if (!zmap || !img.complete) return;
      const r = img.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      overlay.width = Math.round(r.width * dpr);
      overlay.height = Math.round(r.height * dpr);
      const off = document.createElement("canvas");
      off.width = zw; off.height = zh;
      const octx = off.getContext("2d");
      const data = octx.createImageData(zw, zh);
      const cols = edits.map((e, i) => (e.color !== zones[i].color ? rgb(e.color) : null));
      const regs = pending.filter((r) => r.color !== zones[r.block].color).map((r) => [r.mask, rgb(r.color)]);
      for (let p = 0; p < zmap.length; p++) {
        const z = zmap[p] - 1;
        let a = 0, R = 0, G = 0, B = 0;
        if (z >= 0 && cols[z]) [R, G, B, a] = [...cols[z], 210];
        for (const [m, c] of regs) if (m[p]) [R, G, B, a] = [...c, 210];
        if (selMask && !selMask[p]) {
          if (a) { R *= 0.6; G *= 0.6; B *= 0.6; } else { a = 110; }
        }
        const o = p * 4;
        data.data[o] = R; data.data[o + 1] = G; data.data[o + 2] = B; data.data[o + 3] = a;
      }
      octx.putImageData(data, 0, 0);
      const ctx = overlay.getContext("2d");
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, overlay.width, overlay.height);
    }
    img.addEventListener("load", draw);
    window.addEventListener("resize", draw);

    // Pixel de zone sous le pointeur (petite tolérance : les rangées de points laissent des interstices)
    function zoneAt(clientX, clientY) {
      if (!zmap) return null;
      const r = img.getBoundingClientRect();
      const x = Math.floor(((clientX - r.left) / r.width) * zw);
      const y = Math.floor(((clientY - r.top) / r.height) * zh);
      let best = null, bd = Infinity;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= zw || yy >= zh || !zmap[yy * zw + xx]) continue;
          const d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = { z: zmap[yy * zw + xx] - 1, p: yy * zw + xx, x: xx, y: yy }; }
        }
      }
      return best;
    }

    // Zone fermée : pixels contigus de la même couleur (remplissage par diffusion)
    function floodRegion(start) {
      const v = zmap[start];
      const mask = new Uint8Array(zmap.length);
      const stack = new Int32Array(zmap.length);
      let n = 0;
      stack[n++] = start;
      mask[start] = 1;
      while (n) {
        const p = stack[--n], x = p % zw;
        for (const q of [p - zw, p + zw, x > 0 ? p - 1 : -1, x < zw - 1 ? p + 1 : -1]) {
          if (q >= 0 && q < zmap.length && !mask[q] && zmap[q] === v) { mask[q] = 1; stack[n++] = q; }
        }
      }
      return mask;
    }
    const blockMask = (i) => zmap.map((v) => (v === i + 1 ? 1 : 0));

    const hover = (e) => { e.target.style.cursor = zoneAt(e.clientX, e.clientY) ? "pointer" : "default"; };
    overlay.addEventListener("mousemove", hover);
    overlay.addEventListener("click", (e) => {
      e.stopPropagation();
      const hit = zoneAt(e.clientX, e.clientY);
      if (!hit) return closeMenu();
      let j = pending.findIndex((r) => r.mask[hit.p]);
      if (j < 0 && zones[hit.z].region !== null) {  // déjà isolée lors d'une retouche précédente : c'est son propre bloc
        return select({ type: "block", i: hit.z }, e.pageX + 10, e.pageY + 10, floodRegion(hit.p));
      }
      if (j < 0) {
        const e0 = edits[hit.z];
        pending.push({ block: hit.z, fx: (hit.x + 0.5) / zw, fy: (hit.y + 0.5) / zh, color: e0.color, fill: e0.fill, mask: floodRegion(hit.p) });
        j = pending.length - 1;
      }
      select({ type: "region", j }, e.pageX + 10, e.pageY + 10);
    });

    // -- menu contextuel --------------------------------------------------------------------
    function select(t, x, y, mask) {
      target = t;
      selMask = mask || (t.type === "region" ? pending[t.j].mask : zmap ? blockMask(t.i) : null);
      renderList();
      draw();
      openMenu(x, y);
    }
    function showView(name) {
      $$(".zm-view", menu).forEach((v) => { v.hidden = v.dataset.view !== name; });
      if (name === "color") renderPalette();
      if (name === "fill") renderFills();
      const first = $(`.zm-view[data-view="${name}"] button`, menu);
      if (first) first.focus();
    }
    function openMenu(x, y) {
      const c = cur();
      $("#zm-swatch").style.setProperty("--c", c.color);
      $("#zm-title").textContent = target.type === "region" ? "Zone fermée"
        : zones[target.i].region !== null ? "Zone isolée" : "Toute la couleur";
      $("#zm-fill").textContent = fills[c.fill];
      $("#zm-picker").value = c.color;
      menu.hidden = false;
      showView("main");
      const w = menu.offsetWidth, h = menu.offsetHeight;
      const maxX = window.scrollX + document.documentElement.clientWidth - w - 8;
      const maxY = window.scrollY + window.innerHeight - h - 8;
      menu.style.left = `${Math.max(window.scrollX + 8, Math.min(x, maxX))}px`;
      menu.style.top = `${Math.max(window.scrollY + 8, Math.min(y, maxY))}px`;
    }
    function closeMenu() {
      menu.hidden = true;
      // une zone cliquée puis laissée telle quelle n'est pas conservée
      for (let j = pending.length - 1; j >= 0; j--) if (!regionChanged(pending[j])) pending.splice(j, 1);
      if (target) { target = null; selMask = null; renderList(); draw(); }
    }
    $$("[data-goto]", menu).forEach((b) => b.addEventListener("click", () => showView(b.dataset.goto)));
    menu.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", () => { if (!menu.hidden) closeMenu(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) closeMenu(); });

    function setColor(c) {
      cur().color = c.toLowerCase();
      $("#zm-swatch").style.setProperty("--c", cur().color);
      $("#zm-picker").value = cur().color;
      renderList();
      draw();
    }
    function renderPalette() {
      const colors = [...new Set([...zones.map((z) => z.color), ...edits.map((e) => e.color), ...pending.map((r) => r.color)])];
      $("#zm-palette").replaceChildren(...colors.map((c) => {
        const b = document.createElement("button");
        b.type = "button";
        b.style.setProperty("--c", c);
        b.title = c;
        b.setAttribute("aria-label", `Couleur ${c}`);
        b.classList.toggle("current", c === cur().color);
        b.addEventListener("click", () => { setColor(c); renderPalette(); });
        return b;
      }));
    }
    $("#zm-picker").addEventListener("input", (e) => setColor(e.target.value));
    $("#zm-picker").addEventListener("change", () => renderPalette());

    function renderFills() {
      const items = [];
      fillList.forEach(([k, label]) => {
        const relief = k.startsWith("relief_");
        if (relief && !items.some((el) => el.classList.contains("zm-group"))) {
          const h = document.createElement("p");
          h.className = "zm-label zm-group";
          h.textContent = "En relief";
          items.push(h);
        }
        const b = document.createElement("button");
        b.type = "button";
        b.className = "zm-fill";
        b.setAttribute("role", "radio");
        b.setAttribute("aria-checked", String(k === cur().fill));
        const shown = relief ? label.replace(/^Relief : (.)/, (_, c) => c.toUpperCase()) : label;
        b.innerHTML = `${icon(k)}<span>${shown}</span>${k === "tatami" ? "<em>par défaut</em>" : ""}`;
        b.addEventListener("click", () => {
          cur().fill = k;
          $("#zm-fill").textContent = label;
          renderFills();
          renderList();
        });
        items.push(b);
      });
      $("#zm-fills").replaceChildren(...items);
    }

    // -- appliquer / annuler ----------------------------------------------------------------
    const showError = (msg) => { errEl.textContent = msg; errEl.hidden = !msg; };
    resetBtn.addEventListener("click", () => {
      zones.forEach((z, i) => { edits[i].color = z.color; edits[i].fill = z.fill; });
      pending.length = 0;
      closeMenu();
      renderList();
      draw();
    });
    applyBtn.addEventListener("click", async () => {
      showError("");
      applyBtn.disabled = resetBtn.disabled = true;
      applyBtn.classList.add("loading");
      const regions = pending.filter(regionChanged).map(({ block, fx, fy, color, fill }) => ({ block, fx, fy, color, fill }));
      try {
        const res = await fetch(root.dataset.api, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ zones: edits, regions }),
        });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "Une erreur est survenue.");
        // On reste dans l'éditeur : attente de la régénération puis rechargement (nouvelles zones, nouvel aperçu)
        closeMenu();
        if (loadingText) loadingText.textContent = "Application des modifications…";
        if (loading) loading.hidden = false;
        const wait = async () => {
          try {
            const d = await (await fetch(root.dataset.status, { cache: "no-store" })).json();
            if (d.status === "processing") {
              if (loadingStep) loadingStep.textContent = d.step ? `${d.step}…` : "";
              return setTimeout(wait, 1200);
            }
            window.location.reload();
          } catch {
            setTimeout(wait, 2000);
          }
        };
        setTimeout(wait, 800);
      } catch (ex) {
        showError(ex.message);
        applyBtn.classList.remove("loading");
        renderList();
      }
    });

    renderList();
  }
})();
