/* PixelOdyssey - interface locale. Aucun framework : Leaflet (local) + fetch vers le serveur Python. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var nf = function (v, d) { d = d || 0; return new Intl.NumberFormat("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d }).format(v); };
  var kg = function (v) { return v < 1 ? nf(v * 1000) + " g" : nf(v, v < 10 ? 1 : 0) + " kg"; };
  var area = function (m2) { return m2 >= 10000 ? nf(m2 / 10000, 2) + " ha" : nf(m2) + " m²"; };
  var dur = function (s) { if (s == null) return "—"; s = Math.round(s); return s < 60 ? s + " s" : Math.floor(s / 60) + " min " + String(s % 60).padStart(2, "0") + " s"; };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };

  var CLS = {
    Dechet_Plastique: { label: "Déchet plastique", color: "#22d3ee" },
    Bouteille: { label: "Bouteilles", color: "#2a78d6" },
    Debris_Divers: { label: "Débris divers", color: "#eb6834" },
    Bidon: { label: "Bidons et seaux", color: "#1baf7a" },
    Cordage_Filet: { label: "Cordages et filets", color: "#eda100" },
    Flipflops: { label: "Tongs", color: "#e87ba4" },
    Cagette: { label: "Cagettes", color: "#008300" },
    Bouee: { label: "Bouées et flotteurs", color: "#4a3aa7" }
  };
  var clsInfo = function (c) { return CLS[c] || { label: c, color: "#22d3ee" }; };

  var S = { inventory: null, models: [], model: null, job: null, poll: null, res: null, dets: null, thr: 0.25, cls: null, mosaicAdded: false };

  // ------------------------------------------------------------------ carte
  var map = L.map("map", { zoomControl: true, maxZoom: 25, preferCanvas: true, attributionControl: true }).setView([16.76, -24.73], 4);
  var sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxNativeZoom: 17, maxZoom: 25, attribution: "Imagerie Esri, Maxar, Earthstar Geographics"
  }).addTo(map);
  var renderer = L.canvas({ padding: 0.3, tolerance: 4 });
  var footLayer = L.layerGroup().addTo(map);
  var mosaicLayer = null;
  var detLayer = L.layerGroup().addTo(map);
  var alignedFoot = null;
  window.PO = { map: map };  // accès pour les tests et le débogage (console du navigateur)

  function hint(text) { var h = $("map-hint"); h.textContent = text || ""; h.hidden = !text; }

  // ------------------------------------------------------------------ étapes
  function setStep(id, state) {
    var el = $(id);
    el.dataset.state = state;
    el.classList.remove("open");
    if (state === "active") setTimeout(function () { el.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, 50);
  }
  function setSummary(id, text) { var e = $(id + "-sum"); if (e) e.textContent = text; }
  document.querySelectorAll(".step h2").forEach(function (h) {
    h.addEventListener("click", function () { var s = h.parentNode; if (s.dataset.state === "done") s.classList.toggle("open"); });
  });
  function api(path, body, method) {
    var opt = body !== undefined ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
    return fetch(path, opt).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.erreur || ("Erreur " + r.status)); return j; });
    });
  }

  // ------------------------------------------------------------------ 1. photos
  function loadRecents() {
    api("/api/recents").then(function (j) {
      $("recents").innerHTML = j.recents.map(function (p) {
        var name = p.split(/[\\/]/).filter(Boolean).pop();
        return '<button data-p="' + esc(p) + '" title="' + esc(p) + '">' + esc(name) + "</button>";
      }).join("");
      $("recents").querySelectorAll("button").forEach(function (b) { b.onclick = function () { $("folder").value = b.dataset.p; scan(); }; });
    }).catch(function () {});
  }

  function scan() {
    var folder = $("folder").value.trim().replace(/^"|"$/g, "");
    if (!folder) return;
    $("scan-error").hidden = true;
    $("btn-scan").disabled = true;
    $("btn-scan").textContent = "Analyse…";
    hint("Lecture des métadonnées des photos…");
    api("/api/inventaire", { dossier: folder }).then(function (inv) {
      S.inventory = inv;
      showInventory(inv);
      setSummary("step-photos", nf(inv.resume.n_utilisables) + " photos · " + area(inv.resume.surface_m2) + " · " + inv.dossier.split(/[\\/]/).filter(Boolean).pop());
      setStep("step-photos", "done");
      $("step-photos").classList.add("open");
      setStep("step-model", "active");
      $("outdir").value = inv.sortie_defaut;
      renderModels();
      loadRecents();
    }).catch(function (e) {
      $("scan-error").textContent = e.message;
      $("scan-error").hidden = false;
      hint("");
    }).finally(function () { $("btn-scan").disabled = false; $("btn-scan").textContent = "Analyser"; });
  }

  var STATE_COLOR = { ok: "#22d3ee", alerte: "#eda100", isolee: "#eb6834", exclue: "#9aa6aa" };

  function showInventory(inv) {
    var r = inv.resume;
    $("scan-result").hidden = false;
    $("scan-tiles").innerHTML =
      tile("Photos utilisables", nf(r.n_utilisables) + "<small> / " + nf(r.n_total) + "</small>", r.capteurs.join(", ")) +
      tile("Surface couverte", area(r.surface_m2), r.date_debut ? r.date_debut : "") +
      tile("Altitude · résolution", nf(r.altitude_median) + "<small> m</small>", nf(r.gsd_cm_median, 2) + " cm par pixel") +
      tile("Recouvrement", nf(r.recouvrement_median) + "<small> %</small>", "durée estimée ≈ " + nf(r.duree_estimee_min) + " min");
    var groups = {};
    inv.photos.forEach(function (p) {
      p.alertes.forEach(function (a) {
        var key = a.replace(/\(.*\)/, "").trim();
        (groups[key] = groups[key] || []).push(p);
      });
    });
    var li = Object.keys(groups).map(function (k) {
      var ps = groups[k];
      var col = ps.some(function (p) { return !p.utilisable; }) ? STATE_COLOR.exclue : (k.indexOf("isolée") >= 0 ? STATE_COLOR.isolee : STATE_COLOR.alerte);
      return '<li><span class="dot" style="background:' + col + '"></span><span>' + esc(k) + " : " + ps.length + " photo" + (ps.length > 1 ? "s" : "") +
        (ps[0].emprise ? ' · <button data-n="' + esc(ps[0].nom) + '">voir</button>' : "") + "</span></li>";
    });
    if (!li.length) li = ['<li><span class="dot" style="background:var(--ok)"></span><span>Aucune alerte : toutes les photos sont exploitables.</span></li>'];
    $("scan-alerts").innerHTML = li.join("");
    $("scan-alerts").querySelectorAll("button").forEach(function (b) { b.onclick = function () { zoomPhoto(b.dataset.n); }; });
    if (inv.deja_traite && inv.deja_traite.length) {
      $("already").innerHTML = "Ce vol a déjà été traité (" + inv.deja_traite.map(esc).join(", ") + "). " +
        '<br><button class="btn" id="btn-open-prev">Afficher ces résultats</button>';
      $("already").hidden = false;
      $("btn-open-prev").onclick = function () { openExisting(inv.sortie_defaut); };
    } else {
      $("already").hidden = true;
    }
    drawFootprints(inv);
  }

  function tile(lab, val, sub) { return '<div class="tile"><div class="lab">' + lab + '</div><div class="val">' + val + '</div><div class="sub">' + (sub || "") + "</div></div>"; }

  var footByName = {};
  function drawFootprints(inv) {
    footLayer.clearLayers();
    footByName = {};
    var bounds = [];
    inv.photos.forEach(function (p) {
      if (!p.emprise) return;
      var ll = p.emprise.map(function (c) { return [c[1], c[0]]; });
      bounds = bounds.concat(ll);
      var col = STATE_COLOR[p.etat] || STATE_COLOR.ok;
      var poly = L.polygon(ll, { renderer: renderer, color: col, weight: 1.2, fillOpacity: p.etat === "ok" ? 0.04 : 0.18, opacity: 0.9 });
      poly.bindTooltip("<b>" + esc(p.nom) + "</b>" + (p.alertes.length ? "<br>" + p.alertes.map(esc).join("<br>") : "") +
        "<br>altitude " + nf(p.alt, 1) + " m · " + nf(p.gsd_cm, 2) + " cm/px", { sticky: true });
      poly.addTo(footLayer);
      L.circleMarker([p.lat, p.lon], { renderer: renderer, radius: 2.5, color: col, weight: 1, fillOpacity: 1 }).addTo(footLayer);
      footByName[p.nom] = poly;
    });
    if (bounds.length) map.fitBounds(L.latLngBounds(bounds).pad(0.08));
    hint("");
  }
  function zoomPhoto(name) { var p = footByName[name]; if (p) { map.fitBounds(p.getBounds().pad(0.6)); p.openTooltip(); } }

  // dépôt de fichiers
  var drop = $("drop");
  ["dragenter", "dragover"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) {
    var items = e.dataTransfer.items;
    var entries = [];
    for (var i = 0; i < items.length; i++) { var en = items[i].webkitGetAsEntry && items[i].webkitGetAsEntry(); if (en) entries.push(en); }
    collect(entries).then(function (files) {
      files = files.filter(function (f) { return /\.jpe?g$/i.test(f.name); });
      if (!files.length) { $("scan-error").textContent = "Aucun fichier JPG dans ce dépôt (les PNG n'ont pas les métadonnées GPS)."; $("scan-error").hidden = false; return; }
      uploadFiles(files);
    });
  });
  function collect(entries) {
    var out = [];
    function walk(en) {
      return new Promise(function (res) {
        if (en.isFile) { en.file(function (f) { out.push(f); res(); }, function () { res(); }); }
        else if (en.isDirectory) {
          var rd = en.createReader(), all = [];
          (function readAll() {
            rd.readEntries(function (batch) {
              if (!batch.length) { Promise.all(all.map(walk)).then(res); return; }
              all = all.concat(Array.prototype.slice.call(batch)); readAll();
            }, function () { res(); });
          })();
        } else res();
      });
    }
    return Promise.all(entries.map(walk)).then(function () { return out; });
  }
  function uploadFiles(files) {
    $("upload").hidden = false;
    api("/api/import/debut", {}).then(function (j) {
      var i = 0;
      function next() {
        if (i >= files.length) { $("upload-txt").textContent = files.length + " photos copiées"; $("folder").value = j.dossier; scan(); return; }
        var f = files[i];
        $("upload-txt").textContent = "Copie des photos : " + (i + 1) + " / " + files.length;
        $("upload-bar").style.width = (100 * i / files.length) + "%";
        fetch("/api/import/" + j.session + "/" + encodeURIComponent(f.name), { method: "PUT", body: f }).then(function () { i++; next(); })
          .catch(function (e) { $("scan-error").textContent = "Copie interrompue : " + e.message; $("scan-error").hidden = false; });
      }
      next();
    });
  }

  $("btn-pick").onclick = function () {
    $("btn-pick").disabled = true;
    hint("Une fenêtre de choix de dossier s'est ouverte (elle peut être derrière le navigateur).");
    api("/api/choisir-dossier", {}).then(function (j) {
      if (j.chemin) { $("folder").value = j.chemin; scan(); } else hint("");
    }).catch(function (e) { $("scan-error").textContent = e.message; $("scan-error").hidden = false; })
      .finally(function () { $("btn-pick").disabled = false; });
  };
  $("btn-scan").onclick = scan;
  $("folder").addEventListener("keydown", function (e) { if (e.key === "Enter") scan(); });
  $("btn-open-existing").onclick = function () {
    hint("Choisissez le dossier de résultats d'un traitement (…_pixelodyssey).");
    api("/api/choisir-dossier", {}).then(function (j) { if (j.chemin) openExisting(j.chemin); else hint(""); });
  };

  // ------------------------------------------------------------------ 2. modèle
  function renderModels() {
    var r = S.inventory ? S.inventory.resume : null;
    $("models").innerHTML = S.models.map(function (m, i) {
      var warn = "";
      if (!m.disponible) warn = "Fichier de poids introuvable : " + esc(m.chemin);
      else if (r && (r.gsd_cm_median < m.gsd_min_cm * 0.85 || r.gsd_cm_median > m.gsd_max_cm * 1.15))
        warn = "Résolution de ce vol (" + nf(r.gsd_cm_median, 2) + " cm/px) hors de celle vue à l'entraînement (" + nf(m.gsd_min_cm, 2) + " à " + nf(m.gsd_max_cm, 2) + ") : résultats moins fiables.";
      var sel = S.model ? S.model === m.id : i === 0;
      return '<label class="model' + (sel ? " selected" : "") + '"><input type="radio" name="model" value="' + esc(m.id) + '"' + (sel ? " checked" : "") + (m.disponible ? "" : " disabled") + ">" +
        "<b>" + esc(m.nom) + "</b><p>" + esc(m.description) + "</p>" + (warn ? '<div class="warn">' + warn + "</div>" : "") + "</label>";
    }).join("");
    if (!S.model && S.models.length) S.model = S.models[0].id;
    $("models").querySelectorAll("input").forEach(function (inp) {
      inp.onchange = function () { S.model = inp.value; renderModels(); };
    });
    $("btn-run").disabled = !S.inventory || !S.model;
  }

  $("btn-run").onclick = function () {
    $("btn-run").disabled = true;
    api("/api/lancer", { dossier: S.inventory.dossier, modele: S.model, sortie: $("outdir").value.trim() }).then(function (j) {
      startJob(j.job);
    }).catch(function (e) { alertError(e.message); $("btn-run").disabled = false; });
  };
  function alertError(msg) { $("scan-error").textContent = msg; $("scan-error").hidden = false; }

  // ------------------------------------------------------------------ 3. traitement
  function startJob(id) {
    S.job = id;
    var mm = S.models.filter(function (m) { return m.id === S.model; })[0];
    setSummary("step-model", mm ? mm.nom : S.model);
    S.mosaicAdded = false;
    $("step-photos").classList.remove("open");
    setStep("step-model", "done");
    setStep("step-run", "active");
    setStep("step-results", "locked");
    $("run-error").hidden = true;
    clearInterval(S.poll);
    S.poll = setInterval(pollStatus, 1000);
    pollStatus();
  }
  function pollStatus() {
    api("/api/etat/" + S.job).then(function (st) {
      $("prog-bar").style.width = (st.progression * 100) + "%";
      $("prog-pct").textContent = nf(st.progression * 100) + " %";
      $("prog-msg").textContent = st.message || "";
      $("prog-time").textContent = "Écoulé " + dur(st.ecoule_s) + (st.restant_s != null ? " · reste environ " + dur(st.restant_s) : "");
      $("stages").innerHTML = st.etapes.map(function (e) {
        var cls = st.etapes_faites.indexOf(e.cle) >= 0 ? "done" : (st.etape === e.cle && st.etat === "en_cours" ? "current" : "");
        return '<li class="' + cls + '">' + esc(e.libelle) + "</li>";
      }).join("");
      $("log").textContent = st.journal.join("\n");
      $("log").scrollTop = 1e9;
      if (st.mosaique_prete && !S.mosaicAdded) { S.mosaicAdded = true; loadResults(false); }
      if (st.etat === "termine") { clearInterval(S.poll); setSummary("step-run", "Terminé en " + dur(st.ecoule_s)); loadResults(true); }
      if (st.etat === "erreur") {
        clearInterval(S.poll);
        $("run-error").textContent = st.erreur || "Erreur";
        $("run-error").hidden = false;
        setStep("step-model", "active");
        $("btn-run").disabled = false;
      }
    }).catch(function () {});
  }
  $("btn-cancel").onclick = function () { if (S.job) api("/api/annuler", { job: S.job }); };

  // ------------------------------------------------------------------ 4. résultats
  window.PO.openExisting = function (dir) { openExisting(dir); };
  function openExisting(dir) {
    api("/api/ouvrir", { sortie: dir }).then(function (j) {
      S.job = j.job;
      setStep("step-photos", "done");
      setStep("step-model", "done");
      setStep("step-run", "done");
      setSummary("step-run", "Traitement repris depuis le disque");
      loadResults(true);
    }).catch(function (e) { alertError(e.message); hint(""); });
  }

  function loadResults(final) {
    api("/api/resultats/" + S.job).then(function (res) {
      S.res = res;
      if (res.mosaique) addMosaic(res.mosaique);
      if (!final) return;
      setStep("step-run", "done");
      setStep("step-results", "active");
      if (res.modele) { S.thr = res.modele.seuil; $("conf").value = S.thr; setSummary("step-model", res.modele.nom); }
      var syn = res.synthese || {};
      if (!$("step-photos-sum").textContent && syn.photos) {
        setSummary("step-photos", syn.photos + " photos" + (syn.date_traitement ? " · traitées le " + syn.date_traitement : ""));
      }
      return fetch(res.detections_url).then(function (r) { return r.json(); }).then(function (gj) {
        S.dets = gj.features;
        S.cls = null;
        drawDetections();
        renderStats();
        renderDownloads(res);
        renderAlignment(res.alignement);
        loadAlignedFootprints(res);
        hint("");
        $("foot").textContent = "Résultats : " + res.sortie;
      });
    }).catch(function (e) { $("run-error").textContent = e.message; $("run-error").hidden = false; });
  }

  function addMosaic(m) {
    if (mosaicLayer) map.removeLayer(mosaicLayer);
    mosaicLayer = L.tileLayer(m.url, { maxNativeZoom: Math.min(m.maxzoom, 24), maxZoom: 25, bounds: m.bounds, tileSize: 256 });
    if ($("lay-mosaic").checked) mosaicLayer.addTo(map);
    footLayer.eachLayer(function (l) { if (l.setStyle) l.setStyle({ fillOpacity: 0, opacity: 0.35 }); });
    if (!S.res || !S.dets) map.fitBounds(m.bounds);
    if (mosaicLayer.bringToBack) { mosaicLayer.bringToFront(); }
    detLayer.eachLayer(function (l) { if (l.bringToFront) l.bringToFront(); });
  }

  function loadAlignedFootprints(res) {
    fetch(res.emprises_url).then(function (r) { return r.ok ? r.json() : null; }).then(function (gj) {
      if (!gj) return;
      if (alignedFoot) map.removeLayer(alignedFoot);
      alignedFoot = L.geoJSON(gj, { renderer: renderer, style: { color: "#ffffff", weight: 1, opacity: 0.6, fillOpacity: 0, dashArray: "4 4" },
        onEachFeature: function (f, l) { l.bindTooltip(esc(f.properties.photo) + " · zone attribuée " + nf(f.properties.surface_attribuee_m2) + " m²", { sticky: true }); } });
      footLayer.clearLayers();
      if ($("lay-foot").checked) alignedFoot.addTo(map);
    });
  }

  function visible(f) { return f.properties.confiance >= S.thr && (S.cls === null || f.properties.classe === S.cls); }

  function drawDetections() {
    detLayer.clearLayers();
    if (!S.dets || !$("lay-dets").checked) return;
    // contours créés une seule fois puis réutilisés : le curseur de confiance reste fluide même avec
    // des dizaines de milliers de détections
    S.dets.forEach(function (f) {
      if (!visible(f)) return;
      if (!f._poly) {
        var info = clsInfo(f.properties.classe);
        var ll = f.geometry.coordinates[0].map(function (c) { return [c[1], c[0]]; });
        var poly = f._poly = L.polygon(ll, { renderer: renderer, color: info.color, weight: 1.6, fillColor: info.color, fillOpacity: 0.25 });
        poly.on("click", function () { popup(f, poly); });
      }
      detLayer.addLayer(f._poly);
    });
    var classes = {};
    S.dets.forEach(function (f) { classes[f.properties.classe] = 1; });
    $("legend").innerHTML = Object.keys(classes).map(function (c) {
      var i = clsInfo(c); return '<div class="row"><span class="sw" style="color:' + i.color + ";background:" + i.color + '55"></span>' + esc(i.label) + "</div>";
    }).join("");
    $("legend").hidden = !Object.keys(classes).length;
  }

  function popup(f, layer) {
    var p = f.properties, info = clsInfo(p.classe);
    var w = p.poids_kg != null ? "≈ " + kg(p.poids_kg) + " <small>(" + kg(p.poids_bas_kg) + " à " + kg(p.poids_haut_kg) + ")</small>" : "—";
    var html = '<div class="pop"><img src="/imagette/' + S.job + "/" + p.id + '.jpg" alt="Vue de la photo d\'origine">' +
      '<div class="t" style="color:' + info.color + '">' + esc(info.label) + "</div><dl>" +
      "<dt>Confiance</dt><dd>" + nf(p.confiance * 100) + " %</dd>" +
      "<dt>Longueur</dt><dd>≈ " + (p.longueur_m < 1 ? nf(p.longueur_m * 100) + " cm" : nf(p.longueur_m, 2) + " m") + "</dd>" +
      "<dt>Surface</dt><dd>" + nf(p.surface_m2 * 1e4) + " cm²</dd>" +
      "<dt>Poids estimé</dt><dd>" + w + "</dd>" +
      "<dt>Photo</dt><dd>" + esc(p.photo) + (p.vues > 1 ? " <small>(vu sur " + p.vues + " photos)</small>" : "") + "</dd>" +
      "<dt>Position</dt><dd>" + nf(p.lat, 6) + ", " + nf(p.lon, 6) + "</dd></dl></div>";
    layer.bindPopup(html, { maxWidth: 280 }).openPopup();
  }

  function renderStats() {
    var res = S.res, syn = res.synthese || {};
    var sel = S.dets.filter(function (f) { return f.properties.confiance >= S.thr; });
    var n = sel.length, surf = 0, w = 0, wl = 0, wh = 0, per = {};
    sel.forEach(function (f) {
      var p = f.properties;
      surf += p.surface_m2;
      if (p.poids_kg != null) { w += p.poids_kg; wl += p.poids_bas_kg; wh += p.poids_haut_kg; }
      var e = per[p.classe] = per[p.classe] || { n: 0, kg: 0 };
      e.n++; e.kg += p.poids_kg || 0;
    });
    var cov = syn.surface_couverte_m2 || 1;
    $("res-tiles").innerHTML =
      tile("Déchets détectés", nf(n), "sur " + area(cov) + " couverts") +
      tile("Densité", nf(n / cov, 2) + "<small> /m²</small>", nf(100 * n / cov) + " pour 100 m²") +
      tile("Surface de déchets", nf(surf, surf < 10 ? 2 : 1) + "<small> m²</small>", nf(100 * surf / cov, 2) + " % du sol couvert") +
      tile('Poids estimé <span class="pill">ordre de grandeur</span>', "≈ " + kg(w), "fourchette " + kg(wl) + " à " + kg(wh));
    $("conf-val").textContent = nf(S.thr * 100) + " %";
    var model = res.modele || {};
    $("conf-hint").textContent = "Réglage par défaut du modèle : " + nf((model.seuil || 0.25) * 100) + " %. Plus haut : moins de détections, mais plus sûres.";
    var keys = Object.keys(per).sort(function (a, b) { return per[b].n - per[a].n; });
    var max = keys.length ? per[keys[0]].n : 1;
    $("classes").innerHTML = keys.length > 1 ? keys.map(function (c) {
      var i = clsInfo(c), e = per[c];
      return '<button class="cls-row' + (S.cls && S.cls !== c ? " dim" : "") + '" data-c="' + esc(c) + '" aria-pressed="' + (S.cls === c) + '">' +
        '<span class="nm"><span class="sq" style="background:' + i.color + '"></span>' + esc(i.label) + "</span>" +
        '<span class="v"><b>' + nf(e.n) + "</b> · " + kg(e.kg) + "</span>" +
        '<span class="track"><span class="fill" style="width:' + (100 * e.n / max) + "%;background:" + i.color + '"></span></span></button>';
    }).join("") : "";
    $("classes").querySelectorAll(".cls-row").forEach(function (b) {
      b.onclick = function () { S.cls = S.cls === b.dataset.c ? null : b.dataset.c; drawDetections(); renderStats(); };
    });
  }

  function renderAlignment(a) {
    var w = $("align-warn"), info = $("align-info");
    if (!a) { w.hidden = true; info.textContent = ""; return; }
    info.textContent = "Qualité du recalage de ce vol : " + a.photos_recalees + " photos recalées sur " + a.photos_total +
      ", écart résiduel médian entre photos voisines " + nf(a.residu_median_cm, 1) + " cm, écart au GPS médian " +
      nf(a.ecart_gps_median_m, 1) + " m.";
    var msg = "";
    if (a.mode === "gps") msg = "Photos placées sur leur seul GPS : la mosaïque présente des décalages et des déchets situés dans les recouvrements peuvent être comptés deux fois.";
    else if (a.mode === "rigide") msg = "Recalage simplifié pour ce vol : coutures de la mosaïque moins nettes, comptage peu affecté.";
    else if (a.photos_total > 1 && a.photos_recalees < a.photos_total) msg = (a.photos_total - a.photos_recalees) + " photo(s) sans recouvrement exploitable avec leurs voisines : placées sur leur seul GPS.";
    w.textContent = msg; w.hidden = !msg;
  }

  function renderDownloads(res) {
    $("downloads").innerHTML = res.telechargements.map(function (f) {
      var size = f.taille > 1e6 ? nf(f.taille / 1e6, 1) + " Mo" : nf(f.taille / 1e3) + " Ko";
      return '<a href="' + f.url + '">' + esc(f.libelle) + "<span>" + size + "</span></a>";
    }).join("");
  }

  var raf = 0;
  $("conf").oninput = function () {
    S.thr = Math.round(parseFloat($("conf").value) * 100) / 100;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(function () { drawDetections(); renderStats(); });
  };
  $("lay-mosaic").onchange = function (e) { if (!mosaicLayer) return; e.target.checked ? mosaicLayer.addTo(map) : map.removeLayer(mosaicLayer); };
  $("lay-dets").onchange = drawDetections;
  $("lay-foot").onchange = function (e) { if (!alignedFoot) return; e.target.checked ? alignedFoot.addTo(map) : map.removeLayer(alignedFoot); };
  $("lay-sat").onchange = function (e) { e.target.checked ? sat.addTo(map) : map.removeLayer(sat); };
  $("btn-folder").onclick = function () { if (S.res) api("/api/ouvrir-dossier", { chemin: S.res.sortie }); };
  $("btn-new").onclick = function () { location.reload(); };

  // ------------------------------------------------------------------ démarrage
  api("/api/modeles").then(function (j) {
    S.models = j.modeles;
    $("device").textContent = "Détection sur : " + j.appareil;
    $("foot").textContent = "PixelOdyssey " + j.version;
    setTimeout(function refreshDevice() {
      api("/api/modeles").then(function (k) { $("device").textContent = "Détection sur : " + k.appareil; if (/en cours/.test(k.appareil)) setTimeout(refreshDevice, 2000); });
    }, 2500);
  });
  loadRecents();
})();
