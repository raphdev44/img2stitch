(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const page = document.body.dataset.page;
  const setTaskProgress = (bar, value, step = "") => {
    if (!bar) return;
    if (/Analyse|Préparation des points|Nettoyage des points|Réorganisation du parcours|Application des modifications/.test(step)) {
      bar.removeAttribute("value");
    } else if (Number.isFinite(value)) {
      bar.value = value;
    }
  };

  function setupStageNavigation(stage, getContent, fitButton) {
    let scale = 1, fitScale = 1, offsetX = 0, offsetY = 0;
    let pan = null, pinch = null, moved = false, suppressClickUntil = 0;
    const pointers = new Map();
    const clampScale = (value) => Math.max(Math.min(0.2, fitScale), Math.min(Math.max(8, fitScale * 8), value));
    const apply = () => {
      const content = getContent();
      if (content) content.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0) scale(${scale})`;
      stage.classList.toggle("is-zoomed", scale > fitScale * 1.02 || Math.abs(offsetX) > 1 || Math.abs(offsetY) > 1);
    };
    const fit = () => {
      const content = getContent();
      if (!content || !content.offsetWidth || !content.offsetHeight) return;
      fitScale = Math.min(8, Math.max(0.1, Math.min(
        (stage.clientWidth - 32) / content.offsetWidth,
        (stage.clientHeight - 32) / content.offsetHeight,
      )));
      scale = fitScale;
      offsetX = offsetY = 0;
      apply();
    };
    const point = (event) => {
      const rect = stage.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    const zoomAt = (next, clientX, clientY) => {
      const rect = stage.getBoundingClientRect();
      const px = clientX - rect.left - rect.width / 2;
      const py = clientY - rect.top - rect.height / 2;
      const ratio = next / scale;
      offsetX = px - (px - offsetX) * ratio;
      offsetY = py - (py - offsetY) * ratio;
      scale = next;
      apply();
    };
    const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const center = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoomAt(clampScale(scale * Math.exp(-event.deltaY * 0.0015)), event.clientX, event.clientY);
    }, { passive: false });
    stage.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      if (!pointers.size) moved = false;
      const p = point(event);
      pointers.set(event.pointerId, p);
      if (event.target.setPointerCapture) event.target.setPointerCapture(event.pointerId);
      if (pointers.size === 1) {
        pan = { id: event.pointerId, start: p, originX: offsetX, originY: offsetY, moved: false };
      } else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { distance: distance(a, b), center: center(a, b), scale, offsetX, offsetY };
        pan = null;
      }
      stage.classList.add("is-panning");
    });
    stage.addEventListener("pointermove", (event) => {
      if (!pointers.has(event.pointerId)) return;
      const p = point(event);
      pointers.set(event.pointerId, p);
      if (pointers.size === 2 && pinch) {
        const [a, b] = [...pointers.values()];
        const nextCenter = center(a, b);
        const ratio = clampScale(pinch.scale * distance(a, b) / Math.max(1, pinch.distance)) / pinch.scale;
        const originX = pinch.center.x - stage.clientWidth / 2;
        const originY = pinch.center.y - stage.clientHeight / 2;
        offsetX = originX - (originX - pinch.offsetX) * ratio + nextCenter.x - pinch.center.x;
        offsetY = originY - (originY - pinch.offsetY) * ratio + nextCenter.y - pinch.center.y;
        scale = pinch.scale * ratio;
        moved = true;
        apply();
      } else if (pan?.id === event.pointerId) {
        const dx = p.x - pan.start.x, dy = p.y - pan.start.y;
        if (Math.hypot(dx, dy) > 4) pan.moved = moved = true;
        if (pan.moved) {
          offsetX = pan.originX + dx;
          offsetY = pan.originY + dy;
          apply();
        }
      }
    });
    const endPointer = (event) => {
      if (!pointers.has(event.pointerId)) return;
      pointers.delete(event.pointerId);
      if (!pointers.size) {
        if (moved) suppressClickUntil = performance.now() + 350;
        pan = pinch = null;
        stage.classList.remove("is-panning");
      } else if (pointers.size === 1) {
        const [id, p] = [...pointers.entries()][0];
        pan = { id, start: p, originX: offsetX, originY: offsetY, moved: false };
        pinch = null;
      }
    };
    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);
    stage.addEventListener("dragstart", (event) => event.preventDefault());
    fitButton.addEventListener("click", fit);
    window.addEventListener("resize", fit);
    requestAnimationFrame(fit);
    return {
      fit,
      consumeDragClick() {
        if (performance.now() >= suppressClickUntil) return false;
        suppressClickUntil = 0;
        return true;
      },
    };
  }

  // --- Création ---
  if (page === "create" || page === "cross-stitch" || page === "hand-embroidery") {
    const form = $("#create-form") || $("#cross-stitch-form") || $("#hand-form");
    const input = $("#image-input");
    const zone = $("#dropzone");
    const range = $("#colors");
    const out = $("#colors-value");
    const btn = $("#submit-btn");
    const err = $("#form-error");
    const submitLabel = $(".btn-label", btn).textContent;
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

    // Jauge de créativité : libellé du cran courant ; elle ne sert à rien si une famille est imposée.
    const creativity = $("#creativity");
    const creativityOut = $("#creativity-value");
    const creativityLabels = creativity ? [...creativity.list.options].map((o) => o.label) : [];
    const syncCreativity = () => {
      if (!creativity) return;
      creativityOut.textContent = creativityLabels[+creativity.value - 1] || "";
      const p = ((creativity.value - creativity.min) / (creativity.max - creativity.min)) * 100;
      creativity.style.setProperty("--p", `${p}%`);
    };
    const styleSelect = $("#hand-style");
    const syncFamily = () => {
      if (!creativity || !styleSelect) return;
      const auto = styleSelect.value === "auto";
      creativity.disabled = !auto;
      $("#creativity-field").classList.toggle("is-muted", !auto);
    };
    if (creativity) {
      creativity.addEventListener("input", syncCreativity);
      syncCreativity();
      if (styleSelect) {
        styleSelect.addEventListener("change", syncFamily);
        syncFamily();
      }
    }

    const showError = (msg) => { err.textContent = msg; err.hidden = !msg; };
    const bgValue = () => $("input[name=background]:checked", form).value;

    // Pré-positionne la jauge selon les couleurs détectées, sauf si l'utilisateur l'a déjà réglée
    const hint = $("#colors-hint");
    const defaultHint = hint.textContent;
    let analyseId = 0;
    let userTouched = false;
    let sizeTouched = false;
    const sizeInputs = [$("#width-mm"), $("#height-mm")].filter(Boolean);
    range.addEventListener("input", () => { userTouched = true; });
    sizeInputs.forEach((i) => i.addEventListener("input", () => { sizeTouched = true; }));

    const busy = (on) => {
      $("#colors-field").classList.toggle("is-busy", on && !userTouched);
      $("#size-field")?.classList.toggle("is-busy", on && !sizeTouched);
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
        if (!res.ok) throw new Error(json.error || "Impossible d'analyser cette image.");
        showError("");
        palettes = json.palettes;
        if (!userTouched) range.value = json.colors;
        syncRange();
        if (!sizeTouched && sizeInputs.length) { sizeInputs[0].value = json.width_mm; sizeInputs[1].value = json.height_mm; }
        const s = json.colors > 1 ? "s" : "";
        const bgNote = bgValue() === "keep" ? "" : " (fond exclu)";
        hint.textContent = `${json.colors} couleur${s} distincte${s} détectée${s}${bgNote}. Chaque couleur correspond à un fil.`;
      } catch (ex) {
        if (id === analyseId) {
          hint.textContent = defaultHint;
          showError(ex.message || "Impossible d'analyser cette image.");
        }
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
      if (f.size > 20 * 1024 * 1024) {
        showError("Fichier trop volumineux (20 Mo maximum).");
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
    zone.addEventListener("keydown", (event) => {
      if (event.target === zone && ["Enter", " "].includes(event.key)) {
        event.preventDefault();
        input.click();
      }
    });
    $("#image-clear").addEventListener("click", (e) => { e.preventDefault(); input.click(); });

    ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
    zone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

    // Image du presse-papiers : certains navigateurs ne la mettent que dans `items` (captures
    // d'écran, images copiées d'une page), et le fichier peut arriver sans nom.
    const pastedFile = (e) => {
      const data = e.clipboardData;
      if (!data) return null;
      const item = [...(data.items || [])].find((it) => it.kind === "file" && it.type.startsWith("image/"));
      let f = item?.getAsFile() || [...(data.files || [])].find((f) => f.type.startsWith("image/"));
      if (!f) return null;
      const type = f.type || item?.type || "image/png";
      if (!f.name) {  // un fichier sans nom serait refusé par le serveur
        const ext = { "image/jpeg": "jpg", "image/webp": "webp", "image/avif": "avif" }[type] || "png";
        f = new File([f], `collage.${ext}`, { type });
      } else if (!f.type) {
        f = new File([f], f.name, { type });
      }
      return f;
    };
    document.addEventListener("paste", (e) => {
      const f = pastedFile(e);
      if (!f) return;
      e.preventDefault();
      setFile(f);
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!file) return;
      if (!sizeInputs.every((i) => i.reportValidity())) return;
      showError("");
      btn.disabled = true;
      btn.classList.add("loading");
      $(".btn-label", btn).textContent = "Envoi…";
      $("#create-progress").hidden = false;
      const data = new FormData();
      data.append("image", file);
      data.append("colors", range.value);
      data.append("background", bgValue());
      data.append("exclude", JSON.stringify(excluded));
      const stitch = form.querySelector('input[name="stitch"]:checked');
      if (stitch) data.append("stitch", stitch.value);
      if ($("#hand-style")) data.append("hand_style", $("#hand-style").value);
      if ($("#creativity") && !$("#creativity").disabled) data.append("creativity", $("#creativity").value);
      if (sizeInputs.length) {
        data.append("width_mm", sizeInputs[0].value);
        data.append("height_mm", sizeInputs[1].value);
      }
      // Options propres à la numérisation classique : absentes des formulaires pixel art et main.
      if ($("#thicken-lines")) data.append("thicken_lines", $("#thicken-lines").checked ? "true" : "false");
      if ($("#smart-fill")) data.append("smart_fill", $("#smart-fill").checked ? "true" : "false");
      if ($("#scan-engine")) data.append("engine", $("#scan-engine").value);
      try {
        const res = await fetch(form.action, { method: "POST", body: data });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "Une erreur est survenue.");
        window.location.href = json.urls.page;
      } catch (ex) {
        showError(ex.message);
        btn.disabled = false;
        btn.classList.remove("loading");
        $("#create-progress").hidden = true;
        $(".btn-label", btn).textContent = submitLabel;
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
    let zoom;

    const show = (view) => {
      tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
      imgs.forEach((i) => i.classList.toggle("shown", i.dataset.view === view));
      if (zoom) requestAnimationFrame(zoom.fit);
    };
    tabs.forEach((t) => t.addEventListener("click", () => !t.disabled && show(t.dataset.view)));
    show(root.dataset.hasPhoto === "1" ? "photo" : root.dataset.hasRaw === "1" ? "raw" : "source");
    zoom = setupStageNavigation($("#design-stage"), () => imgs.find((i) => i.classList.contains("shown")), $("#stage-fit"));
    imgs.forEach((img) => img.addEventListener("load", () => zoom.fit()));

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
          setTaskProgress($("#processing-progress"), d.progress, d.step);
        } catch { /* réseau indisponible : on réessaie */ }
        setTimeout(poll, 1500);
      };
      setTimeout(poll, 1500);
    }

    // Optimisation : un seul bouton, menu de choix simple / radical. Chaque choix duplique le
    // motif ; on bascule sur la copie optimisée (l'original reste dans la galerie).
    const optimizeBox = $("#optimize-picker");
    if (optimizeBox) {
      const toggle = $("#optimize-toggle");
      const menu = $("#optimize-menu");
      const options = $$(".optimize-option", menu);
      const progress = $("#optimize-progress");
      const bar = $("#optimize-progress-bar");
      const step = $("#optimize-step");
      const error = $("#optimize-error");

      const setOpen = (open) => {
        menu.hidden = !open;
        toggle.setAttribute("aria-expanded", String(open));
        if (open) options[0]?.focus();
      };
      toggle.addEventListener("click", () => setOpen(menu.hidden));
      toggle.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); }
      });
      document.addEventListener("click", (e) => {
        if (!menu.hidden && !optimizeBox.contains(e.target)) setOpen(false);
      });
      optimizeBox.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !menu.hidden) { setOpen(false); toggle.focus(); }
      });

      const setBusy = (busy) => {
        options.forEach((button) => { button.disabled = busy; });
        toggle.disabled = busy;
        toggle.classList.toggle("loading", busy);
        progress.hidden = !busy;
        if (busy) bar.removeAttribute("value");
      };
      options.forEach((button) => button.addEventListener("click", async () => {
        const mode = button.dataset.mode || "simple";
        setOpen(false);
        error.hidden = true;
        setBusy(true);
        step.textContent = "Création de la copie optimisée…";
        try {
          const res = await fetch(root.dataset.optimize, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode }),
          });
          const json = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(json.error || "L'optimisation n'a pas pu démarrer.");
          window.location.assign(json.urls?.page || root.dataset.api);
        } catch (ex) {
          setBusy(false);
          error.textContent = ex.message;
          error.hidden = false;
        }
      }));
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
        if (on) $("#photo-progress-bar").removeAttribute("value");
      };
      const waitPhoto = async () => {
        try {
          const d = await (await fetch(root.dataset.api, { cache: "no-store" })).json();
          const bar = $("#photo-progress-bar");
          if (d.photo_rendering) {
            if (bar) bar.removeAttribute("value");
            return setTimeout(waitPhoto, 1000);
          }
          if (bar && Number.isFinite(d.photo_progress)) bar.value = d.photo_progress;
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
    zones.forEach((z) => { z.outline = z.outline !== false; });  // projets antérieurs : contour actif
    const fills = Object.fromEntries(fillList);
    const edits = zones.map((z) => ({ color: z.color, fill: z.fill, outline: z.outline }));
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
    const zoom = setupStageNavigation($("#editor-stage"), () => $("#editor-canvas"), $("#stage-fit"));
    let zmap = null, zw = 0, zh = 0;
    // Sélection : {type: "block", i} (toute une couleur, depuis la liste) ou {type: "region", j} (zone fermée cliquée)
    let target = null, selMask = null;
    const pending = [];  // nouvelles zones fermées : {block, fx, fy, color, fill, mask}

    // Aperçus des motifs de remplissage (SVG statiques)
    const FILL_ICONS = {
      tatami: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M-30 24 L-6 0 M-24 24 L0 0 M-18 24 L6 0 M-12 24 L12 0 M-6 24 L18 0 M0 24 L24 0 M6 24 L30 0 M12 24 L36 0 M18 24 L42 0 M24 24 L48 0" stroke-dasharray="2.6 2.2" stroke-linejoin="round"/></svg>',
      tatami_h: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M1.5 3 L22.5 3 M1.5 8.5 L22.5 8.5 M1.5 14 L22.5 14 M1.5 19.5 L22.5 19.5" stroke-dasharray="2.6 2.2" stroke-linejoin="round"/></svg>',
      tatami_v: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 1.5 L3 22.5 M8.5 1.5 L8.5 22.5 M14 1.5 L14 22.5 M19.5 1.5 L19.5 22.5" stroke-dasharray="2.6 2.2" stroke-linejoin="round"/></svg>',
      cross: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M-32 24 L-8 0 M-24 24 L0 0 M-16 24 L8 0 M-8 24 L16 0 M0 24 L24 0 M8 24 L32 0 M16 24 L40 0 M24 24 L48 0 M-8 0 L16 24 M0 0 L24 24 M8 0 L32 24 M16 0 L40 24 M24 0 L48 24" stroke-dasharray="2.4 2" stroke-linejoin="round"/></svg>',
      satin: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M1.6 1.5 L1.6 22.5 M3.7 1.5 L3.7 22.5 M5.8 1.5 L5.8 22.5 M7.9 1.5 L7.9 22.5 M10 1.5 L10 22.5 M12.1 1.5 L12.1 22.5 M14.2 1.5 L14.2 22.5 M16.3 1.5 L16.3 22.5 M18.4 1.5 L18.4 22.5 M20.5 1.5 L20.5 22.5 M22.6 1.5 L22.6 22.5" stroke-linejoin="round"/></svg>',
      satin_flow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M-2 23.5C-2 16 7 13 5 0.5 M2.6 23.5C2.6 16 11.6 13 9.6 0.5 M7.2 23.5C7.2 16 16.2 13 14.2 0.5 M11.8 23.5C11.8 16 20.8 13 18.8 0.5 M16.4 23.5C16.4 16 25.4 13 23.4 0.5 M21 23.5C21 16 30 13 28 0.5" stroke-linejoin="round"/></svg>',
      contour: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 2.5L17 2.5Q21.5 2.5 21.5 7L21.5 17Q21.5 21.5 17 21.5L7 21.5Q2.5 21.5 2.5 17L2.5 7Q2.5 2.5 7 2.5Z M9.6 6.8L14.4 6.8Q17.2 6.8 17.2 9.6L17.2 14.4Q17.2 17.2 14.4 17.2L9.6 17.2Q6.8 17.2 6.8 14.4L6.8 9.6Q6.8 6.8 9.6 6.8Z M10.5 12A1.5 1.5 0 1 0 13.5 12A1.5 1.5 0 1 0 10.5 12Z" stroke-linejoin="round"/></svg>',
      wave: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.2 5.6q2.8 -3.1 5.6 0q2.8 3.1 5.6 0q2.8 -3.1 5.6 0 M2.2 12q2.8 -3.1 5.6 0q2.8 3.1 5.6 0q2.8 -3.1 5.6 0 M2.2 18.4q2.8 -3.1 5.6 0q2.8 3.1 5.6 0q2.8 -3.1 5.6 0" stroke-linejoin="round"/></svg>',
      zigzag: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M1 6.2l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6 M1 12.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6 M1 19l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6l1.85 -3.6 l1.85 3.6l1.85 3.6 l1.85 -3.6" stroke-linejoin="round"/></svg>',
      satin_relief: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M-2 23.5C-2 16 7 13 5 0.5 M2.6 23.5C2.6 16 11.6 13 9.6 0.5 M7.2 23.5C7.2 16 16.2 13 14.2 0.5 M11.8 23.5C11.8 16 20.8 13 18.8 0.5 M16.4 23.5C16.4 16 25.4 13 23.4 0.5 M21 23.5C21 16 30 13 28 0.5" stroke-linejoin="round"/><path d="M1.5 23.5C1.5 16 10.5 13 8.5 0.5" stroke-width="2.6" opacity=".45" stroke-linejoin="round"/></svg>',
      none: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 2.5L17 2.5Q21.5 2.5 21.5 7L21.5 17Q21.5 21.5 17 21.5L7 21.5Q2.5 21.5 2.5 17L2.5 7Q2.5 2.5 7 2.5Z" stroke-dasharray="3 2.6" stroke-linejoin="round"/></svg>',
      relief_cotes: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5.7 1.2 L5.7 22.8 M11.7 1.2 L11.7 22.8 M17.7 1.2 L17.7 22.8" stroke-linejoin="round"/><path d="M2.7 1.2 L2.7 22.8 M8.7 1.2 L8.7 22.8 M14.7 1.2 L14.7 22.8 M20.7 1.2 L20.7 22.8" opacity=".45" stroke-width="2.9" stroke-linejoin="round"/></svg>',
      relief_chevrons: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M0.6 14l2.85 -5 l2.85 5l2.85 5 l2.85 -5l2.85 -5 l2.85 5l2.85 5 l2.85 -5 M0.6 17.9l2.85 -5 l2.85 5l2.85 5 l2.85 -5l2.85 -5 l2.85 5l2.85 5 l2.85 -5 M0.6 4.6l2.85 -5 l2.85 5l2.85 5 l2.85 -5l2.85 -5 l2.85 5l2.85 5 l2.85 -5 M0.6 8.5l2.85 -5 l2.85 5l2.85 5 l2.85 -5l2.85 -5 l2.85 5l2.85 5 l2.85 -5" stroke-linejoin="round"/></svg>',
      relief_vagues: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 8.6q2.7 -3.4 5.4 0q2.7 3.4 5.4 0q2.7 -3.4 5.4 0 M2 15.6q2.7 -3.4 5.4 0q2.7 3.4 5.4 0q2.7 -3.4 5.4 0" stroke-linejoin="round"/><path d="M2 5q2.7 -3.4 5.4 0q2.7 3.4 5.4 0q2.7 -3.4 5.4 0 M2 12q2.7 -3.4 5.4 0q2.7 3.4 5.4 0q2.7 -3.4 5.4 0 M2 19q2.7 -3.4 5.4 0q2.7 3.4 5.4 0q2.7 -3.4 5.4 0" opacity=".45" stroke-width="2.7" stroke-linejoin="round"/></svg>',
      relief_damier: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.4 1 L8.4 23M15.6 1 L15.6 23M1 8.4 L23 8.4M1 15.6 L23 15.6 M1 1 L7.9 7.9M7.9 1 L1 7.9 M15.6 1 L22.5 7.9M22.5 1 L15.6 7.9 M8.3 8.3 L15.2 15.2M15.2 8.3 L8.3 15.2 M1 15.6 L7.9 22.5M7.9 15.6 L1 22.5 M15.6 15.6 L22.5 22.5M22.5 15.6 L15.6 22.5" stroke-linejoin="round"/></svg>',
      relief_vannerie: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M1.2 3 L10.8 3 M1.2 5.8 L10.8 5.8 M1.2 8.6 L10.8 8.6M13.2 15.4 L22.8 15.4 M13.2 18.2 L22.8 18.2 M13.2 21 L22.8 21 M15.4 1.2 L15.4 10.8 M18.2 1.2 L18.2 10.8 M21 1.2 L21 10.8M3 13.2 L3 22.8 M5.8 13.2 L5.8 22.8 M8.6 13.2 L8.6 22.8" stroke-linejoin="round"/></svg>',
      relief_losanges: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M0 -6 L6 0 L0 6 L-6 0Z M24 -6 L30 0 L24 6 L18 0Z M0 18 L6 24 L0 30 L-6 24Z M24 18 L30 24 L24 30 L18 24Z M12 6 L18 12 L12 18 L6 12Z M12 -18 L18 -12 L12 -6 L6 -12Z M12 30 L18 36 L12 42 L6 36Z" stroke-linejoin="round"/></svg>',
      relief_nid_abeille: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.8 -4.84 L3 -1.72 L-0.6 -1.72 L-2.4 -4.84 L-0.6 -7.96 L3 -7.96Z M4.8 1.4 L3 4.52 L-0.6 4.52 L-2.4 1.4 L-0.6 -1.72 L3 -1.72Z M4.8 7.64 L3 10.76 L-0.6 10.76 L-2.4 7.64 L-0.6 4.52 L3 4.52Z M4.8 13.88 L3 17 L-0.6 17 L-2.4 13.88 L-0.6 10.76 L3 10.76Z M4.8 20.12 L3 23.24 L-0.6 23.24 L-2.4 20.12 L-0.6 17 L3 17Z M10.2 -1.72 L8.4 1.4 L4.8 1.4 L3 -1.72 L4.8 -4.84 L8.4 -4.84Z M10.2 4.52 L8.4 7.64 L4.8 7.64 L3 4.52 L4.8 1.4 L8.4 1.4Z M10.2 10.76 L8.4 13.88 L4.8 13.88 L3 10.76 L4.8 7.64 L8.4 7.64Z M10.2 17 L8.4 20.12 L4.8 20.12 L3 17 L4.8 13.88 L8.4 13.88Z M10.2 23.24 L8.4 26.36 L4.8 26.36 L3 23.24 L4.8 20.12 L8.4 20.12Z M15.6 -4.84 L13.8 -1.72 L10.2 -1.72 L8.4 -4.84 L10.2 -7.96 L13.8 -7.96Z M15.6 1.4 L13.8 4.52 L10.2 4.52 L8.4 1.4 L10.2 -1.72 L13.8 -1.72Z M15.6 7.64 L13.8 10.76 L10.2 10.76 L8.4 7.64 L10.2 4.52 L13.8 4.52Z M15.6 13.88 L13.8 17 L10.2 17 L8.4 13.88 L10.2 10.76 L13.8 10.76Z M15.6 20.12 L13.8 23.24 L10.2 23.24 L8.4 20.12 L10.2 17 L13.8 17Z M21 -1.72 L19.2 1.4 L15.6 1.4 L13.8 -1.72 L15.6 -4.84 L19.2 -4.84Z M21 4.52 L19.2 7.64 L15.6 7.64 L13.8 4.52 L15.6 1.4 L19.2 1.4Z M21 10.76 L19.2 13.88 L15.6 13.88 L13.8 10.76 L15.6 7.64 L19.2 7.64Z M21 17 L19.2 20.12 L15.6 20.12 L13.8 17 L15.6 13.88 L19.2 13.88Z M21 23.24 L19.2 26.36 L15.6 26.36 L13.8 23.24 L15.6 20.12 L19.2 20.12Z M26.4 -4.84 L24.6 -1.72 L21 -1.72 L19.2 -4.84 L21 -7.96 L24.6 -7.96Z M26.4 1.4 L24.6 4.52 L21 4.52 L19.2 1.4 L21 -1.72 L24.6 -1.72Z M26.4 7.64 L24.6 10.76 L21 10.76 L19.2 7.64 L21 4.52 L24.6 4.52Z M26.4 13.88 L24.6 17 L21 17 L19.2 13.88 L21 10.76 L24.6 10.76Z M26.4 20.12 L24.6 23.24 L21 23.24 L19.2 20.12 L21 17 L24.6 17Z" stroke-linejoin="round"/></svg>',
      relief_ecailles: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M-2.15 5Q2.15 11.4 6.45 5 M6.45 5Q10.75 11.4 15.05 5 M15.05 5Q19.35 11.4 23.65 5 M23.65 5Q27.95 11.4 32.25 5 M-6.45 11Q-2.15 17.4 2.15 11 M2.15 11Q6.45 17.4 10.75 11 M10.75 11Q15.05 17.4 19.35 11 M19.35 11Q23.65 17.4 27.95 11 M-2.15 17Q2.15 23.4 6.45 17 M6.45 17Q10.75 23.4 15.05 17 M15.05 17Q19.35 23.4 23.65 17 M23.65 17Q27.95 23.4 32.25 17 M-6.45 23Q-2.15 29.4 2.15 23 M2.15 23Q6.45 29.4 10.75 23 M10.75 23Q15.05 29.4 19.35 23 M19.35 23Q23.65 29.4 27.95 23" stroke-linejoin="round"/></svg>',
      relief_coquilles: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.6 12.5A8.4 8.4 0 0 1 20.4 12.5 M3.6 24.5A8.4 8.4 0 0 1 20.4 24.5 M12 12.5 L4.77 9.13 M12 12.5 L7.14 6.17 M12 12.5 L12 4.52 M12 12.5 L16.86 6.17 M12 12.5 L19.23 9.13 M12 24.5 L4.77 21.13 M12 24.5 L7.14 18.17 M12 24.5 L12 16.52 M12 24.5 L16.86 18.17 M12 24.5 L19.23 21.13" stroke-linejoin="round"/></svg>',
      relief_pastilles: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.3 6A2.7 2.7 0 1 0 8.7 6A2.7 2.7 0 1 0 3.3 6Z M15.3 6A2.7 2.7 0 1 0 20.7 6A2.7 2.7 0 1 0 15.3 6Z M9.3 12A2.7 2.7 0 1 0 14.7 12A2.7 2.7 0 1 0 9.3 12Z M3.3 18A2.7 2.7 0 1 0 8.7 18A2.7 2.7 0 1 0 3.3 18Z M15.3 18A2.7 2.7 0 1 0 20.7 18A2.7 2.7 0 1 0 15.3 18Z" stroke-linejoin="round"/></svg>',
      relief_trefle: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.2 8.3A3.8 3.8 0 1 0 15.8 8.3A3.8 3.8 0 1 0 8.2 8.3Z M8.2 15.7A3.8 3.8 0 1 0 15.8 15.7A3.8 3.8 0 1 0 8.2 15.7Z M4.5 12A3.8 3.8 0 1 0 12.1 12A3.8 3.8 0 1 0 4.5 12Z M11.9 12A3.8 3.8 0 1 0 19.5 12A3.8 3.8 0 1 0 11.9 12Z M11 12A1 1 0 1 0 13 12A1 1 0 1 0 11 12Z" stroke-linejoin="round"/></svg>',
      relief_etoiles: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.2 13 L6 9.68 L2.52 10.3 L4.8 7.6 L2.52 4.9 L6 5.52 L7.2 2.2 L8.4 5.52 L11.88 4.9 L9.6 7.6 L11.88 10.3 L8.4 9.68Z M16.8 21.8 L15.6 18.48 L12.12 19.1 L14.4 16.4 L12.12 13.7 L15.6 14.32 L16.8 11 L18 14.32 L21.48 13.7 L19.2 16.4 L21.48 19.1 L18 18.48Z" stroke-linejoin="round"/></svg>',
      relief_feuilles: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.08 2.87Q9.17 4.87 9.12 10.33Q4.03 8.33 4.08 2.87Z M19.92 2.87Q19.97 8.33 14.88 10.33Q14.83 4.87 19.92 2.87Z M9.12 13.67Q9.17 19.13 4.08 21.13Q4.03 15.67 9.12 13.67Z M14.88 13.67Q19.97 15.67 19.92 21.13Q14.83 19.13 14.88 13.67Z M4.64 3.69 L8.56 9.51 M19.36 3.69 L15.44 9.51 M8.56 14.49 L4.64 20.31 M15.44 14.49 L19.36 20.31" stroke-linejoin="round"/></svg>',
      relief_spirales: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.05 7.8 L8.15 8.02 L8.12 8.31 L7.95 8.59 L7.62 8.8 L7.19 8.87 L6.73 8.74 L6.31 8.4 L6.04 7.88 L6 7.24 L6.23 6.58 L6.73 6.03 L7.45 5.69 L8.3 5.66 L9.14 5.97 L9.84 6.63 L10.25 7.54 L10.29 8.6 L9.89 9.64 L9.1 10.49 L7.99 11 L6.72 11.04 L5.48 10.58 L4.47 9.64 L3.87 8.35 L3.8 6.86 L4.32 5.42 M15.95 16.4 L15.85 16.62 L15.88 16.91 L16.05 17.19 L16.38 17.4 L16.81 17.47 L17.27 17.34 L17.69 17 L17.96 16.48 L18 15.84 L17.77 15.18 L17.27 14.63 L16.55 14.29 L15.7 14.26 L14.86 14.57 L14.16 15.23 L13.75 16.14 L13.71 17.2 L14.11 18.24 L14.9 19.09 L16.01 19.6 L17.28 19.64 L18.52 19.18 L19.53 18.24 L20.13 16.95 L20.2 15.46 L19.68 14.02" stroke-linejoin="round"/></svg>',
      relief_pelage: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.4 5.2 L5.6 8.4 M8.2 4.4 L10.4 7.6 M13 5.2 L15.2 8.4 M17.8 4.4 L20 7.6 M2 11 L4.2 14.2 M6.8 12.2 L9 15.4 M11.6 11 L13.8 14.2 M16.4 12.2 L18.6 15.4 M21.2 11 L23.4 14.2 M3.4 18.8 L5.6 22 M8.2 18 L10.4 21.2 M13 18.8 L15.2 22 M17.8 18 L20 21.2"/></svg>',
      relief_pelage_long: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.6 21.4 C5.4 15.4 8.6 9.8 12.2 4.2 M8.2 21.8 C11 15.8 14.2 10.2 17.8 4.6 M13.8 22.2 C16.6 16.2 19.8 10.6 23.4 5"/></svg>',
      relief_plumes: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.2C7.4 6.4 6.6 12.6 12 21.8C17.4 12.6 16.6 6.4 12 2.2Z M12 5.4 L12 19.4 M12 8.6 L8.9 7.1 M12 8.6 L15.1 7.1 M12 11.6 L8.4 10.1 M12 11.6 L15.6 10.1 M12 14.6 L8.9 13.1 M12 14.6 L15.1 13.1" stroke-linejoin="round"/></svg>',
      relief_plumes_couchees: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.6 21.4C2.8 16.6 4.6 10.8 10.4 6.2C10.8 12.6 8.6 18.2 4.6 21.4Z M5.2 20.2 L9.4 7.4 M12.4 22C10.6 17.2 12.4 11.4 18.2 6.8C18.6 13.2 16.4 18.8 12.4 22Z M13 20.8 L17.2 8" stroke-linejoin="round"/></svg>',
      relief_triskele: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 9.3 L12.63 8.83 L13.43 8.52 L14.37 8.42 L15.39 8.57 L16.43 9 L17.41 9.71 L18.27 10.69 L18.94 11.93 L19.34 13.37 L19.43 14.96 M14.34 13.35 L14.43 14.13 L14.3 14.98 L13.91 15.84 L13.27 16.65 L12.39 17.34 L11.28 17.84 L10 18.09 L8.6 18.05 L7.14 17.68 L5.72 16.96 M9.66 13.35 L8.94 13.04 L8.27 12.5 L7.72 11.74 L7.34 10.78 L7.19 9.67 L7.31 8.46 L7.73 7.22 L8.47 6.03 L9.51 4.96 L10.85 4.08 M12 12 m-1.15 0a1.15 1.15 0 1 0 2.3 0a1.15 1.15 0 1 0-2.3 0"/></svg>',
      relief_entrelacs: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.2 7.8 L13.9 8.7 L14.5 9.7 L14.9 10.8 L15 12 L14.9 13.2 L14.5 14.3 L13.9 15.3 L13.2 16.2 L12.2 16.9 L11.1 17.4 L10 17.7 L8.8 17.7 L7.7 17.5 L6.6 17 L5.6 16.3 L4.8 15.5 L4.2 14.5 L3.8 13.4 L3.6 12.2 L3.7 11 L4 9.9 L4.5 8.9 L5.3 8 L6.2 7.2 L7.3 6.7 L8.4 6.4 L9.6 6.3 L10.7 6.5 M10.8 16.2 L10.1 15.3 L9.5 14.3 L9.1 13.2 L9 12 L9.1 10.8 L9.5 9.7 L10.1 8.7 L10.8 7.8 L11.8 7.1 L12.9 6.6 L14 6.3 L15.2 6.3 L16.3 6.5 L17.4 7 L18.4 7.7 L19.2 8.5 L19.8 9.5 L20.2 10.6 L20.4 11.8 L20.3 13 L20 14.1 L19.5 15.1 L18.7 16 L17.8 16.8 L16.7 17.3 L15.6 17.6 L14.4 17.7 L13.3 17.5" stroke-linejoin="round"/></svg>',
      relief_vermicelle: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M0.8 6.2q2.6 4.2 5.2 0t5.2 0 5.2 0 5.2 0M2.4 13.4q2.8-4.6 5.6 0t5.6 0 5.6 0 5.6 0M0.8 20.4q2.6 4.2 5.2 0t5.2 0 5.2 0 5.2 0 M4.4 9.8q2.4 3.6 4.8 0t4.8 0M13.6 17q2.4-3.6 4.8 0t4.8 0" stroke-linejoin="round"/></svg>',
    };
    const icon = (k) => `<svg viewBox="0 0 24 24" aria-hidden="true">${FILL_ICONS[k] || ""}</svg>`;
    const changed = (i) => edits[i].color !== zones[i].color || edits[i].fill !== zones[i].fill
      || edits[i].outline !== zones[i].outline;
    const regionChanged = (r) => r.color !== edits[r.block].color || r.fill !== edits[r.block].fill
      || r.outline !== edits[r.block].outline;
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
      const fillLabel = now.fill === "none" ? "Sans remplissage" : fills[now.fill];
      const noOutline = now.outline === false ? " · sans contour" : "";
      b.innerHTML = `${was}<span class="swatch" style="--c:${now.color}"></span>
        <span class="zone-text"><strong>${label}</strong><small>${fillLabel}${noOutline}</small></span>${icon(now.fill)}`;
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
    img.addEventListener("load", () => { draw(); zoom.fit(); });
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
      if (zoom.consumeDragClick()) return;
      const hit = zoneAt(e.clientX, e.clientY);
      if (!hit) return closeMenu();
      let j = pending.findIndex((r) => r.mask[hit.p]);
      if (j < 0 && zones[hit.z].region !== null) {  // déjà isolée lors d'une retouche précédente : c'est son propre bloc
        return select({ type: "block", i: hit.z }, e.pageX + 10, e.pageY + 10, floodRegion(hit.p));
      }
      if (j < 0) {
        const e0 = edits[hit.z];
        pending.push({ block: hit.z, fx: (hit.x + 0.5) / zw, fy: (hit.y + 0.5) / zh, color: e0.color,
                       fill: e0.fill, outline: e0.outline, mask: floodRegion(hit.p) });
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
      $("#zm-title").textContent = target.type === "region" ? "Zone fermée"
        : zones[target.i].region !== null ? "Zone isolée" : "Toute la couleur";
      $("#zm-picker").value = c.color;
      refreshMenu();
      menu.hidden = false;
      showView("main");
      const w = menu.offsetWidth, h = menu.offsetHeight;
      const maxX = window.scrollX + document.documentElement.clientWidth - w - 8;
      const maxY = window.scrollY + window.innerHeight - h - 8;
      menu.style.left = `${Math.max(window.scrollX + 8, Math.min(x, maxX))}px`;
      menu.style.top = `${Math.max(window.scrollY + 8, Math.min(y, maxY))}px`;
    }
    // -- interrupteurs remplissage / point de contour ----------------------------------------
    const lastFill = new Map();  // dernier motif par zone : rallumer le remplissage le rétablit
    const targetKey = () => (target.type === "region" ? `r${target.j}` : `b${target.i}`);

    function refreshMenu() {
      const c = cur();
      if (!c) return;
      $("#zm-swatch").style.setProperty("--c", c.color);
      $("#zm-fill").textContent = c.fill === "none" ? "Sans remplissage" : fills[c.fill];
      $("#zm-toggle-fill").setAttribute("aria-checked", String(c.fill !== "none"));
      $("#zm-toggle-outline").setAttribute("aria-checked", String(c.outline !== false));
    }

    $("#zm-toggle-fill").addEventListener("click", () => {
      const c = cur();
      if (!c) return;
      if (c.fill === "none") {
        c.fill = lastFill.get(targetKey()) || "tatami";
      } else {
        lastFill.set(targetKey(), c.fill);
        c.fill = "none";
      }
      refreshMenu();
      renderList();
    });
    $("#zm-toggle-outline").addEventListener("click", () => {
      const c = cur();
      if (!c) return;
      c.outline = c.outline === false;
      refreshMenu();
      renderList();
    });

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
          refreshMenu();
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
      zones.forEach((z, i) => { edits[i].color = z.color; edits[i].fill = z.fill; edits[i].outline = z.outline; });
      pending.length = 0;
      closeMenu();
      renderList();
      draw();
    });
    applyBtn.addEventListener("click", async () => {
      showError("");
      applyBtn.disabled = resetBtn.disabled = true;
      applyBtn.classList.add("loading");
      const regions = pending.filter(regionChanged)
        .map(({ block, fx, fy, color, fill, outline }) => ({ block, fx, fy, color, fill, outline }));
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
              setTaskProgress($("#editor-progress"), d.progress, d.step);
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
