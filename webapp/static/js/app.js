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
    };
    range.addEventListener("input", syncRange);
    syncRange();

    const showError = (msg) => { err.textContent = msg; err.hidden = !msg; };

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
      showError("");
      btn.disabled = true;
      btn.classList.add("loading");
      $(".btn-label", btn).textContent = "Envoi…";
      const data = new FormData();
      data.append("image", file);
      data.append("colors", range.value);
      data.append("background", $("input[name=background]:checked", form).value);
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
  }

  // --- Galerie : rafraîchit tant que des motifs sont en cours ---
  if (page === "gallery" && $(".tile .badge.busy")) {
    setTimeout(() => window.location.reload(), 4000);
  }
})();
