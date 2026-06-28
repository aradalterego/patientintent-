import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ============================================================
   Concilium — Patient Intent capture (Agent 6 front-end)
   Frontal_neutral photo + MediaPipe Face Landmarker (468 pts).
   Each perceptual region is an ORDERED polygon of canonical
   indices (calibrated on a real face). Neck has no landmarks of
   its own — projected below the jaw line. Lips split into two
   vertical sides (upper / lower). region->zones is translated in
   the backend; the perceived-state layer is isolated and joins
   only at JOIN Final.

   Click model (250ms window, counts taps on one side):
     1 click  -> select the whole area (all sides)
     2 clicks -> remove the side you clicked
     3 clicks -> remove the whole area (both sides)
   ============================================================ */

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const POLY = {
  forehead: [103, 67, 109, 10, 338, 297, 332, 334, 296, 9, 107, 66, 105],
  glabella: [9, 107, 55, 8, 285, 336],
  nose: [122, 48, 64, 98, 2, 326, 294, 278, 351],
  lip_upper: [97, 2, 326, 426, 291, 308, 13, 78, 61, 206],
  lip_lower: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 14, 78],
  chin: [182, 83, 18, 313, 406, 377, 152, 148, 176],
  brow_R: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  brow_L: [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
  upperlid_R: [33, 246, 161, 160, 159, 158, 157, 173, 133, 56, 28, 27, 29, 30, 247],
  upperlid_L: [263, 466, 388, 387, 386, 385, 384, 398, 362, 286, 258, 257, 259, 260, 467],
  undereye_R: [33, 7, 163, 144, 145, 153, 154, 155, 133, 244, 233, 232, 231, 230, 229, 228, 31, 226],
  undereye_L: [263, 249, 390, 373, 374, 380, 381, 382, 362, 464, 453, 452, 451, 450, 449, 448, 261, 446],
  nasolabial_R: [129, 165, 92, 57, 206, 209],
  nasolabial_L: [358, 391, 322, 287, 426, 429],
  mouth_R: [61, 91, 181, 84, 83, 182, 106, 43, 57],
  mouth_L: [291, 321, 405, 314, 313, 406, 335, 273, 287],
};
const BAND = {
  jaw_R: [58, 172, 136, 150, 149, 176, 148, 152],
  jaw_L: [152, 377, 400, 378, 379, 365, 397, 288],
};
const HULL = {
  cheek_R: [116, 117, 118, 119, 120, 205, 50, 123, 187, 142],
  cheek_L: [345, 346, 347, 348, 349, 425, 280, 352, 411, 371],
};
const NECK_TOP = [132, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 361];

const ZONE_STATES = {
  forehead: ["stressed"],
  glabella: ["angry", "stressed"],
  brow_complex: ["tired", "sad", "angry", "stressed"],
  upper_eyelid: ["tired"],
  tear_trough: ["tired", "sad", "aged", "gaunt"],
  lower_eyelid: ["tired", "gaunt", "dull"],
  temples: ["gaunt"],
  malar_zygomatic: ["tired", "sad", "gaunt", "dull"],
  buccal_submalar: ["gaunt", "heavy"],
  nasolabial: ["aged"],
  lips: ["aged"],
  oral_commissures: ["sad", "angry", "aged", "heavy"],
  jawline: ["aged", "heavy"],
  mandibular_border: ["aged", "heavy"],
  nose: [], lateral_canthal: [], chin: [], neck: [], platysma: [],
};
const STATE_LABELS = {
  tired: "Tired", sad: "Sad", angry: "Angry / stern", aged: "Older",
  heavy: "Heavy / sagging", gaunt: "Gaunt / hollow", stressed: "Tense / worried", dull: "Dull skin",
};

const REGIONS = [
  { key: "forehead", label: "Forehead", axis: "single", zones: ["forehead"], poly: ["forehead"] },
  { key: "glabella", label: "Between brows", axis: "single", zones: ["glabella"], poly: ["glabella"] },
  { key: "brows", label: "Brows", axis: "lr", zones: ["brow_complex"], poly: ["brow_R", "brow_L"] },
  { key: "upper_lids", label: "Upper eyelids", axis: "lr", zones: ["upper_eyelid"], poly: ["upperlid_R", "upperlid_L"] },
  { key: "under_eye", label: "Under-eye", axis: "lr", zones: ["tear_trough", "lower_eyelid"], poly: ["undereye_R", "undereye_L"] },
  { key: "cheeks", label: "Cheeks", axis: "lr", zones: ["malar_zygomatic", "buccal_submalar"], hull: ["cheek_R", "cheek_L"] },
  { key: "nose", label: "Nose", axis: "single", zones: ["nose"], poly: ["nose"] },
  { key: "nasolabial", label: "Nasolabial folds", axis: "lr", zones: ["nasolabial"], poly: ["nasolabial_R", "nasolabial_L"] },
  { key: "lips", label: "Lips", axis: "ud", zones: ["lips"], poly: ["lip_upper", "lip_lower"] },
  { key: "mouth_corners", label: "Mouth corners", axis: "lr", zones: ["oral_commissures"], poly: ["mouth_R", "mouth_L"] },
  { key: "jawline", label: "Jawline", axis: "lr", zones: ["jawline", "mandibular_border"], band: ["jaw_R", "jaw_L"] },
  { key: "chin", label: "Chin", axis: "single", zones: ["chin"], poly: ["chin"] },
  { key: "neck", label: "Neck", axis: "single", zones: ["neck", "platysma"], neck: true },
];
const EXTERNAL = [
  { key: "temples", label: "Temples", axis: "lr", zones: ["temples"], anchor: { right: 71, left: 301 } },
  { key: "crows_feet", label: "Crow's feet", axis: "lr", zones: ["lateral_canthal"], anchor: { right: 143, left: 372 } },
];
const SIDE_LABELS = { both: "", right: "Right", left: "Left", upper: "Upper", lower: "Lower" };
const ALL = [...REGIONS, ...EXTERNAL];
const regionByKey = Object.fromEntries(ALL.map((r) => [r.key, r]));
const sidesOf = (r) => (r.axis === "ud" ? ["upper", "lower"] : r.axis === "lr" ? ["right", "left"] : ["both"]);

const ALL_STATES = ["tired", "sad", "angry", "aged", "heavy", "gaunt", "stressed", "dull"];
const STATE_TO_REGIONS = Object.fromEntries(
  ALL_STATES.map((st) => [st, ALL.filter((r) => (r.zones || []).some((z) => (ZONE_STATES[z] || []).includes(st))).map((r) => r.key)])
);

const toPath = (pts) => pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
function smoothPath(pts, t = 0.9) {
  const n = pts.length;
  if (n < 3) return "M " + toPath(pts) + " Z";
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)} `;
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
    const c1x = p1.x + ((p2.x - p0.x) / 6) * t, c1y = p1.y + ((p2.y - p0.y) / 6) * t;
    const c2x = p2.x - ((p3.x - p1.x) / 6) * t, c2y = p2.y - ((p3.y - p1.y) / 6) * t;
    d += `C ${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)} `;
  }
  return d + "Z";
}
const polyPts = (idx, lm, w, h) => idx.map((i) => lm[i]).filter(Boolean).map((p) => ({ x: p.x * w, y: p.y * h }));
function convexHull(points) {
  if (points.length < 3) return points;
  const pts = points.slice().sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lo = [], hi = [];
  for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (hi.length >= 2 && cross(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); }
  return lo.slice(0, -1).concat(hi.slice(0, -1));
}
const hullPts = (idx, lm, w, h) => convexHull(polyPts(idx, lm, w, h));
function bandPts(idx, lm, w, h, cx, cy, inset = 0.14) {
  const line = polyPts(idx, lm, w, h);
  const inner = [...line].reverse().map((p) => ({ x: p.x + (cx - p.x) * inset, y: p.y + (cy - p.y) * inset }));
  return line.concat(inner);
}
function neckPts(lm, w, h, faceH) {
  const line = polyPts(NECK_TOP, lm, w, h);
  if (!line.length) return [];
  const midX = (lm[152]?.x || 0.5) * w;
  const dy = faceH * 0.20;
  const bottom = [...line].reverse().map((p) => ({ x: p.x + (midX - p.x) * 0.22, y: p.y + dy }));
  return line.concat(bottom);
}
function centroid(pts) {
  return { cx: pts.reduce((s, p) => s + p.x, 0) / pts.length, cy: pts.reduce((s, p) => s + p.y, 0) / pts.length };
}

export default function FaceAreaSelector() {
  const [landmarker, setLandmarker] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [modelError, setModelError] = useState(null);

  const [imgSrc, setImgSrc] = useState(null);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  const [landmarks, setLandmarks] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState(null);

  const [selections, setSelections] = useState({}); // { key: { sides: [...] } }
  const [debug, setDebug] = useState(false);
  const [view, setView] = useState({ z: 1, x: 0, y: 0 });
  const [labelChip, setLabelChip] = useState(null); // {region, side} — shown after a click, fades after 2s
  const [selectedStates, setSelectedStates] = useState([]); // which state chips the clinician explicitly activated

  const imgRef = useRef(null);
  const fileRef = useRef(null);
  const stageRef = useRef(null);
  const pointers = useRef(new Map());
  const gesture = useRef({ moved: false, sx: 0, sy: 0, ox: 0, oy: 0, startDist: 0, startView: null, startCenter: null });
  const labelTimer = useRef(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
        const lm = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "IMAGE", numFaces: 1,
        });
        if (!cancelled) { setLandmarker(lm); setLoadingModel(false); }
      } catch (e) {
        if (!cancelled) { setModelError(e?.message || "model load failed"); setLoadingModel(false); }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e) => {
      if (!landmarks) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.12 : 1 / 1.12);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [landmarks]);

  const resetAll = () => {
    setSelections({}); setSelectedStates([]); setLabelChip(null); setLandmarks(null); setDetectError(null); setView({ z: 1, x: 0, y: 0 });
  };
  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    resetAll();
    setImgSrc(URL.createObjectURL(file));
  };
  const onImgLoad = useCallback(() => {
    const el = imgRef.current;
    if (!el || !landmarker) return;
    setImgDims({ w: el.naturalWidth, h: el.naturalHeight });
    setDetecting(true); setDetectError(null);
    try {
      const res = landmarker.detect(el);
      if (res?.faceLandmarks?.length) setLandmarks(res.faceLandmarks[0]);
      else setDetectError("No face detected. Use a clear, well-lit, front-facing photo.");
    } catch (e) { setDetectError("Detection failed: " + (e?.message || "")); }
    finally { setDetecting(false); }
  }, [landmarker]);

  const faceGeom = useMemo(() => {
    if (!landmarks) return { cx: 0, cy: 0, h: 0 };
    const { w, h } = imgDims;
    const cx = landmarks.reduce((s, p) => s + p.x, 0) / landmarks.length * w;
    const cy = landmarks.reduce((s, p) => s + p.y, 0) / landmarks.length * h;
    const ys = landmarks.map((p) => p.y * h);
    return { cx, cy, h: Math.max(...ys) - Math.min(...ys) };
  }, [landmarks, imgDims]);

  const polygons = useMemo(() => {
    if (!landmarks) return [];
    const { w, h } = imgDims;
    const out = [];
    for (const r of REGIONS) {
      if (r.neck) {
        const pts = neckPts(landmarks, w, h, faceGeom.h);
        if (pts.length >= 3) out.push({ region: r.key, side: "both", pts });
        continue;
      }
      const keys = r.poly || r.band || r.hull;
      const isBand = !!r.band, isHull = !!r.hull;
      const sides = sidesOf(r);
      if (r.axis === "single") {
        const pts = polyPts(POLY[keys[0]], landmarks, w, h);
        if (pts.length >= 3) out.push({ region: r.key, side: "both", pts });
      } else {
        keys.forEach((k, i) => {
          const pts = isBand ? bandPts(BAND[k], landmarks, w, h, faceGeom.cx, faceGeom.cy)
            : isHull ? hullPts(HULL[k], landmarks, w, h)
            : polyPts(POLY[k], landmarks, w, h);
          if (pts.length >= 3) out.push({ region: r.key, side: sides[i], pts });
        });
      }
    }
    return out;
  }, [landmarks, imgDims, faceGeom]);

  const cropBox = useMemo(() => {
    if (!landmarks) return null;
    const { w, h } = imgDims;
    const xs = landmarks.map((p) => p.x * w), ys = landmarks.map((p) => p.y * h);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const fw = maxX - minX, fh = maxY - minY;
    const padX = fw * 0.18, padTop = fh * 0.16, padBot = fh * 0.24; // headroom + jaw/neck, room for edge labels
    let x = minX - padX, y = minY - padTop, bw = fw + padX * 2, bh = fh + padTop + padBot;
    if (x < 0) { bw += x; x = 0; }
    if (y < 0) { bh += y; y = 0; }
    if (x + bw > w) bw = w - x;
    if (y + bh > h) bh = h - y;
    return { x, y, w: bw, h: bh };
  }, [landmarks, imgDims]);

  const externals = useMemo(() => {
    if (!landmarks) return [];
    const { w, h } = imgDims;
    const cb = cropBox || { x: 0, w };
    const margin = cb.w * 0.05;
    const out = [];
    for (const r of EXTERNAL) {
      for (const side of ["right", "left"]) {
        const a = landmarks[r.anchor[side]];
        if (!a) continue;
        const ap = { x: a.x * w, y: a.y * h };
        const onLeft = side === "right";
        out.push({ region: r.key, side, ap, chip: { x: onLeft ? cb.x + margin : cb.x + cb.w - margin, y: ap.y }, onLeft });
      }
    }
    return out;
  }, [landmarks, imgDims, cropBox]);

  const labelGeom = useMemo(() => {
    if (!labelChip || !landmarks) return null;
    const pg = polygons.find((p) => p.region === labelChip.region && p.side === labelChip.side);
    if (!pg) return null;
    const { cx, cy } = centroid(pg.pts);
    const cb = cropBox || { x: 0, w: imgDims.w };
    const margin = cb.w * 0.05;
    const onLeft = cx < faceGeom.cx;
    return { region: labelChip.region, side: labelChip.side, ap: { x: cx, y: cy }, onLeft, chipX: onLeft ? cb.x + margin : cb.x + cb.w - margin, chipY: cy };
  }, [labelChip, polygons, faceGeom, imgDims, cropBox]);

  /* ---------- selection logic ---------- */
  const isSel = (key, side) => !!selections[key]?.sides.includes(side);
  const showLabel = (region, side) => {
    setLabelChip({ region, side });
    if (labelTimer.current) clearTimeout(labelTimer.current);
    labelTimer.current = setTimeout(() => setLabelChip(null), 2000);
  };

  const handleAreaClick = (region, side, fromPoly = true) => {
    if (gesture.current.moved) return;
    const r = regionByKey[region];
    const all = sidesOf(r);
    setSelections((prev) => {
      const cur = new Set(prev[region]?.sides || []);
      if (cur.size === 0) all.forEach((s) => cur.add(s));   // first click: select whole area
      else if (cur.has(side)) cur.delete(side);             // click a selected side: remove just it
      else cur.add(side);                                    // click an unselected side: add it back
      const next = { ...prev };
      const kept = all.filter((s) => cur.has(s));
      if (kept.length === 0) delete next[region];
      else next[region] = { sides: kept };
      return next;
    });
    if (fromPoly) showLabel(region, side);
  };
  const clearRegion = (key) =>
    setSelections((prev) => { const n = { ...prev }; delete n[key]; return n; });

  /* ---------- emotional-state chips ---------- */
  const chipActive = (st) => selectedStates.includes(st);
  const toggleStateChip = (st) => {
    const regions = STATE_TO_REGIONS[st];
    const active = selectedStates.includes(st);
    setSelectedStates((prev) => (active ? prev.filter((x) => x !== st) : [...prev, st]));
    setSelections((prev) => {
      const next = { ...prev };
      regions.forEach((k) => {
        if (active) delete next[k];
        else next[k] = { sides: sidesOf(regionByKey[k]) };
      });
      return next;
    });
  };

  const intent = useMemo(() => ({
    schema: "patient_intent_v0.3",
    areas: Object.entries(selections).map(([k, v]) => ({ region_key: k, zones: regionByKey[k].zones, sides: v.sides })),
    states: selectedStates,
  }), [selections, selectedStates]);
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(intent, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "patient_intent.json"; a.click();
  };

  /* ---------- zoom / pan ---------- */
  const clampZ = (z) => Math.min(5, Math.max(1, z));
  function zoomAt(px, py, factor) {
    setView((v) => {
      const z = clampZ(v.z * factor);
      const f = z / v.z;
      if (z === 1) return { z: 1, x: 0, y: 0 };
      return { z, x: px - (px - v.x) * f, y: py - (py - v.y) * f };
    });
  }
  const localXY = (cx, cy) => { const r = stageRef.current.getBoundingClientRect(); return { x: cx - r.left, y: cy - r.top }; };
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const onPointerDown = (e) => {
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pl = localXY(e.clientX, e.clientY);
    if (pointers.current.size === 1) {
      gesture.current = { ...gesture.current, moved: false, sx: pl.x, sy: pl.y, ox: view.x, oy: view.y };
    } else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()].map((p) => localXY(p.x, p.y));
      gesture.current.startDist = dist(a, b);
      gesture.current.startView = { ...view };
      gesture.current.startCenter = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      gesture.current.moved = true;
    }
  };
  const onPointerMove = (e) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 2 && gesture.current.startView) {
      const [a, b] = [...pointers.current.values()].map((p) => localXY(p.x, p.y));
      const nd = dist(a, b), nc = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const sv = gesture.current.startView, sc = gesture.current.startCenter;
      const z = clampZ(sv.z * (nd / (gesture.current.startDist || nd)));
      const contentX = (sc.x - sv.x) / sv.z, contentY = (sc.y - sv.y) / sv.z;
      setView(z === 1 ? { z: 1, x: 0, y: 0 } : { z, x: nc.x - contentX * z, y: nc.y - contentY * z });
      try { stageRef.current.setPointerCapture(e.pointerId); } catch {}
    } else if (pointers.current.size === 1) {
      const pl = localXY(e.clientX, e.clientY);
      const dx = pl.x - gesture.current.sx, dy = pl.y - gesture.current.sy;
      if (Math.abs(dx) + Math.abs(dy) > 5) gesture.current.moved = true;
      if (view.z > 1 && gesture.current.moved) {
        setView({ z: view.z, x: gesture.current.ox + dx, y: gesture.current.oy + dy });
        try { stageRef.current.setPointerCapture(e.pointerId); } catch {}
      }
    }
  };
  const onPointerUp = (e) => {
    pointers.current.delete(e.pointerId);
    try { stageRef.current.releasePointerCapture(e.pointerId); } catch {}
    if (pointers.current.size < 2) gesture.current.startView = null;
  };
  const resetView = () => setView({ z: 1, x: 0, y: 0 });
  const zoomBtn = (factor) => {
    const r = stageRef.current.getBoundingClientRect();
    zoomAt(r.width / 2, r.height / 2, factor);
  };

  const selectedCount = Object.keys(selections).length;
  const fs = (cropBox ? Math.max(cropBox.w, cropBox.h) : Math.max(imgDims.w, imgDims.h)) || 1000;
  const vb = cropBox ? `${cropBox.x.toFixed(1)} ${cropBox.y.toFixed(1)} ${cropBox.w.toFixed(1)} ${cropBox.h.toFixed(1)}` : `0 0 ${imgDims.w} ${imgDims.h}`;
  const vbAspect = cropBox ? `${cropBox.w} / ${cropBox.h}` : (imgDims.w ? `${imgDims.w} / ${imgDims.h}` : undefined);
  const sidesLabel = (key) => {
    const v = selections[key]; if (!v) return "";
    if (v.sides.length === 1) return SIDE_LABELS[v.sides[0]] || "";
    if (v.sides.includes("upper") || v.sides.includes("lower")) return "Upper + lower";
    if (v.sides.length === 2) return "Both sides";
    return "";
  };

  return (
    <div className="app">
      <header className="app-head">
        <span className="brand-tag">Concilium · Patient Intent</span>
        <h1 className="app-title">Which areas bother you most?</h1>
      </header>

      {!imgSrc && (
        <div className="upload">
          <h2>Upload a front photo</h2>
          <p>A front-facing, well-lit photo, looking straight at the camera (frontal_neutral).</p>
          <button className="btn" disabled={loadingModel && !modelError} onClick={() => fileRef.current?.click()}>
            {loadingModel ? "Loading detector…" : "Choose photo"}
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
          {modelError && (
            <p className="status err" style={{ marginTop: 14 }}>
              Detector failed to load ({modelError}). On a host with a strict CSP, allow
              storage.googleapis.com and cdn.jsdelivr.net.
            </p>
          )}
        </div>
      )}

      {imgSrc && (
        <>
          <div className="states-panel">
            <div className="state-cards">
              {ALL_STATES.map((st) => (
                <button key={st} className={"state-card" + (chipActive(st) ? " on" : "")} onClick={() => toggleStateChip(st)}>
                  {STATE_LABELS[st]}
                </button>
              ))}
            </div>
          </div>

          <div className="toolbar">
            <button className="btn ghost sm" onClick={() => { resetAll(); setImgSrc(null); }}>Change photo</button>
            <label className="chk"><input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} /> Calibration points</label>
            <span className="spacer" />
            <button className="btn ghost sm" onClick={() => zoomBtn(1 / 1.3)}>–</button>
            <button className="btn ghost sm" onClick={() => zoomBtn(1.3)}>+</button>
            {view.z > 1 && <button className="btn ghost sm" onClick={resetView}>Reset</button>}
          </div>

          <div
            ref={stageRef}
            className="stage"
            style={{ touchAction: "none" }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
          >
            <div className="stage-inner" style={{ aspectRatio: landmarks ? vbAspect : undefined, transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})`, transformOrigin: "0 0" }}>
              {!landmarks && <img ref={imgRef} src={imgSrc} alt="patient" className="stage-img" onLoad={onImgLoad} draggable={false} />}
              {landmarks && imgDims.w > 0 && (
                <svg className="overlay-svg" viewBox={vb} preserveAspectRatio="xMidYMid meet">
                  <image href={imgSrc} x="0" y="0" width={imgDims.w} height={imgDims.h} />
                  {polygons.map((pg) => {
                    const sel = isSel(pg.region, pg.side);
                    return (
                      <path
                        key={pg.region + pg.side}
                        className={"poly" + (sel ? " selected" : "")}
                        d={smoothPath(pg.pts)}
                        onClick={() => handleAreaClick(pg.region, pg.side)}
                      />
                    );
                  })}

                  {externals.map((ex) => {
                    const sel = isSel(ex.region, ex.side);
                    const label = regionByKey[ex.region].label;
                    const cw = fs * 0.15, chh = fs * 0.045;
                    const rx = ex.onLeft ? ex.chip.x : ex.chip.x - cw;
                    return (
                      <g key={ex.region + ex.side} onClick={() => handleAreaClick(ex.region, ex.side, false)} style={{ cursor: "pointer" }}>
                        <line className={"lead-line" + (sel ? " on" : "")} x1={ex.ap.x} y1={ex.ap.y} x2={ex.onLeft ? rx + cw : rx} y2={ex.chip.y} />
                        <circle className={"lead-dot" + (sel ? " on" : "")} cx={ex.ap.x} cy={ex.ap.y} r={fs * 0.006} />
                        <rect className={"chip-rect" + (sel ? " on" : "")} x={rx} y={ex.chip.y - chh / 2} width={cw} height={chh} rx={chh / 2} />
                        <text className={"chip-text" + (sel ? " on" : "")} x={rx + cw / 2} y={ex.chip.y} dominantBaseline="central" textAnchor="middle" fontSize={fs * 0.02}>
                          {label}{SIDE_LABELS[ex.side] ? " · " + SIDE_LABELS[ex.side] : ""}
                        </text>
                      </g>
                    );
                  })}

                  {labelGeom && (() => {
                    const cw = fs * 0.16, chh = fs * 0.045;
                    const rx = labelGeom.onLeft ? labelGeom.chipX : labelGeom.chipX - cw;
                    const txt = regionByKey[labelGeom.region].label + (SIDE_LABELS[labelGeom.side] ? " · " + SIDE_LABELS[labelGeom.side] : "");
                    return (
                      <g onClick={() => handleAreaClick(labelGeom.region, labelGeom.side)} style={{ cursor: "pointer" }}>
                        <line className="lead-line on" x1={labelGeom.ap.x} y1={labelGeom.ap.y} x2={labelGeom.onLeft ? rx + cw : rx} y2={labelGeom.chipY} />
                        <circle className="lead-dot on" cx={labelGeom.ap.x} cy={labelGeom.ap.y} r={fs * 0.006} />
                        <rect className="chip-rect on" x={rx} y={labelGeom.chipY - chh / 2} width={cw} height={chh} rx={chh / 2} />
                        <text className="chip-text on" x={rx + cw / 2} y={labelGeom.chipY} dominantBaseline="central" textAnchor="middle" fontSize={fs * 0.02}>{txt}</text>
                      </g>
                    );
                  })()}

                  {debug && landmarks && landmarks.map((p, i) => (
                    <g key={i}>
                      <circle cx={p.x * imgDims.w} cy={p.y * imgDims.h} r={fs * 0.0035} fill="#e23" opacity="0.8" />
                      <text x={p.x * imgDims.w} y={p.y * imgDims.h} fontSize={fs * 0.009} fill="#114" opacity="0.7">{i}</text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
            {detecting && <div className="stage-status">Detecting face…</div>}
            {detectError && <div className="stage-status err">{detectError}</div>}
          </div>

          {landmarks && !detectError && (
            <p className="hint">
              Tap an area to mark it · tap a side again to remove just that side · pinch or scroll to zoom
            </p>
          )}

          <div className="summary">
            <h2>Selected areas ({selectedCount})</h2>
            {selectedCount === 0 && <p className="empty">No areas selected yet. Tap a state above or an area on the photo.</p>}
            {Object.keys(selections).map((key) => (
              <div className="sum-item" key={key} onClick={() => clearRegion(key)}>
                <div>
                  <div className="sum-name">{regionByKey[key].label}</div>
                  <div className="sum-meta">{sidesLabel(key) || "Selected"}</div>
                </div>
                <div className="sum-x">remove</div>
              </div>
            ))}
          </div>

          {selectedCount > 0 && (
            <div className="btn-row" style={{ marginTop: 18 }}>
              <button className="btn" onClick={exportJson}>Done</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
