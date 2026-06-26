import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ============================================================
   Concilium — Patient Intent capture (Agent 6 front-end)
   Frontal_neutral photo + MediaPipe Face Landmarker (468 pts).
   Each perceptual region is an ORDERED polygon of canonical
   indices (calibrated on a real face). Neck has no landmarks of
   its own — it is projected below the jaw line. region->zones is
   translated in the backend; the perceived-state layer (zone->
   state) is isolated and only joins at JOIN Final.

   Calibration: all index lists live in POLY / BAND below. Toggle
   "Calibration points" to see the 468 numbered landmarks.
   ============================================================ */

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const POLY = {
  forehead: [103, 67, 109, 10, 338, 297, 332, 334, 296, 9, 107, 66, 105],
  glabella: [9, 107, 55, 8, 285, 336],
  nose: [122, 48, 64, 98, 2, 326, 294, 278, 351],
  upper_lip: [97, 167, 165, 40, 0, 270, 391, 393, 326, 2],
  lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
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
/* convex regions (clean blob) */
const HULL = {
  cheek_R: [116, 117, 118, 119, 120, 205, 50, 123, 187, 142],
  cheek_L: [345, 346, 347, 348, 349, 425, 280, 352, 411, 371],
};
/* jaw-bottom line used to project the neck region downward */
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
  nose: [],
  lateral_canthal: [],
  chin: [],
  neck: [],
  platysma: [],
};
const STATE_LABELS = {
  tired: "Tired", sad: "Sad", angry: "Angry / stern", aged: "Older",
  heavy: "Heavy / sagging", gaunt: "Gaunt / hollow", stressed: "Tense / worried", dull: "Dull skin",
};

const REGIONS = [
  { key: "forehead", label: "Forehead", bilateral: false, zones: ["forehead"], poly: ["forehead"] },
  { key: "glabella", label: "Between brows", bilateral: false, zones: ["glabella"], poly: ["glabella"] },
  { key: "brows", label: "Brows", bilateral: true, zones: ["brow_complex"], poly: ["brow_R", "brow_L"] },
  { key: "upper_lids", label: "Upper eyelids", bilateral: true, zones: ["upper_eyelid"], poly: ["upperlid_R", "upperlid_L"] },
  { key: "under_eye", label: "Under-eye", bilateral: true, zones: ["tear_trough", "lower_eyelid"], poly: ["undereye_R", "undereye_L"] },
  { key: "cheeks", label: "Cheeks", bilateral: true, zones: ["malar_zygomatic", "buccal_submalar"], hull: ["cheek_R", "cheek_L"] },
  { key: "nose", label: "Nose", bilateral: false, zones: ["nose"], poly: ["nose"] },
  { key: "nasolabial", label: "Nasolabial folds", bilateral: true, zones: ["nasolabial"], poly: ["nasolabial_R", "nasolabial_L"] },
  { key: "upper_lip", label: "Upper lip", bilateral: false, zones: ["lips"], poly: ["upper_lip"] },
  { key: "lips", label: "Lips", bilateral: false, zones: ["lips"], poly: ["lips"] },
  { key: "mouth_corners", label: "Mouth corners", bilateral: true, zones: ["oral_commissures"], poly: ["mouth_R", "mouth_L"] },
  { key: "jawline", label: "Jawline", bilateral: true, zones: ["jawline", "mandibular_border"], band: ["jaw_R", "jaw_L"] },
  { key: "chin", label: "Chin", bilateral: false, zones: ["chin"], poly: ["chin"] },
  { key: "neck", label: "Neck", bilateral: false, zones: ["neck", "platysma"], neck: true },
];
const EXTERNAL = [
  { key: "temples", label: "Temples", bilateral: true, zones: ["temples"], anchor: { right: 54, left: 284 } },
  { key: "crows_feet", label: "Crow's feet", bilateral: true, zones: ["lateral_canthal"], anchor: { right: 130, left: 359 } },
];
const SIDE_LABELS = { both: "Both sides", right: "Right", left: "Left" };
const ALL = [...REGIONS, ...EXTERNAL];
const regionByKey = Object.fromEntries(ALL.map((r) => [r.key, r]));

/* emotional states, ordered for display */
const ALL_STATES = ["tired", "sad", "angry", "aged", "heavy", "gaunt", "stressed", "dull"];
/* state -> perceptual regions, derived from ZONE_STATES (single source of truth) */
const STATE_TO_REGIONS = Object.fromEntries(
  ALL_STATES.map((st) => [st, ALL.filter((r) => (r.zones || []).some((z) => (ZONE_STATES[z] || []).includes(st))).map((r) => r.key)])
);

const toPath = (pts) => pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
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

  const [selections, setSelections] = useState({});
  const [activeKey, setActiveKey] = useState(null);
  const [activeSide, setActiveSide] = useState("both");
  const [debug, setDebug] = useState(false);
  const [view, setView] = useState({ z: 1, x: 0, y: 0 });
  const [mode, setMode] = useState("area"); // "area" | "state"
  const [selectedStates, setSelectedStates] = useState([]);

  const imgRef = useRef(null);
  const fileRef = useRef(null);
  const stageRef = useRef(null);
  const pointers = useRef(new Map());
  const gesture = useRef({ moved: false, sx: 0, sy: 0, ox: 0, oy: 0, startDist: 0, startView: null, startCenter: null });

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

  /* non-passive wheel zoom */
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

  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelections({}); setActiveKey(null); setLandmarks(null); setDetectError(null); setView({ z: 1, x: 0, y: 0 }); setSelectedStates([]);
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
      if (r.bilateral) {
        keys.forEach((k, i) => {
          const pts = isBand ? bandPts(BAND[k], landmarks, w, h, faceGeom.cx, faceGeom.cy)
            : isHull ? hullPts(HULL[k], landmarks, w, h)
            : polyPts(POLY[k], landmarks, w, h);
          if (pts.length >= 3) out.push({ region: r.key, side: i === 0 ? "right" : "left", pts });
        });
      } else {
        const pts = polyPts(POLY[keys[0]], landmarks, w, h);
        if (pts.length >= 3) out.push({ region: r.key, side: "both", pts });
      }
    }
    return out;
  }, [landmarks, imgDims, faceGeom]);

  const externals = useMemo(() => {
    if (!landmarks) return [];
    const { w, h } = imgDims;
    const margin = w * 0.04;
    const out = [];
    for (const r of EXTERNAL) {
      for (const side of ["right", "left"]) {
        const a = landmarks[r.anchor[side]];
        if (!a) continue;
        const ap = { x: a.x * w, y: a.y * h };
        const onLeft = side === "right";
        out.push({ region: r.key, side, ap, chip: { x: onLeft ? margin : w - margin, y: ap.y }, onLeft });
      }
    }
    return out;
  }, [landmarks, imgDims]);

  const stateRegions = useMemo(() => {
    const set = new Set();
    selectedStates.forEach((st) => (STATE_TO_REGIONS[st] || []).forEach((k) => set.add(k)));
    return set;
  }, [selectedStates]);
  const toggleSelState = (st) => setSelectedStates((prev) => (prev.includes(st) ? prev.filter((x) => x !== st) : [...prev, st]));

    const statesForRegion = useCallback((regionKey) => {
    const r = regionByKey[regionKey];
    if (!r) return [];
    const set = new Set();
    r.zones.forEach((z) => (ZONE_STATES[z] || []).forEach((s) => set.add(s)));
    return [...set];
  }, []);

  const blankSide = () => ({ rating: 0, states: [], note: "" });
  function ensureEntry(prev, key) {
    if (prev[key]) return prev[key];
    return { split: false, both: blankSide(), left: blankSide(), right: blankSide(), bilateral: regionByKey[key].bilateral };
  }
  const selectRegion = (key, side) => {
    setSelections((prev) => ({ ...prev, [key]: { ...ensureEntry(prev, key) } }));
    setActiveKey(key);
    setActiveSide(regionByKey[key].bilateral && selections[key]?.split ? side : "both");
  };
  const removeRegion = (key) => {
    setSelections((prev) => { const n = { ...prev }; delete n[key]; return n; });
    if (activeKey === key) setActiveKey(null);
  };
  /* click toggles the whole area (both sides): select on first tap, remove on second */
  const onAreaClick = (key, side) => {
    if (gesture.current.moved) return;
    if (selections[key]) removeRegion(key);
    else selectRegion(key, side);
  };
  const focusFromSummary = (key, side) => { setActiveKey(key); setActiveSide(selections[key]?.split ? side : "both"); };

  const activeRegion = activeKey ? regionByKey[activeKey] : null;
  const activeEntry = activeKey ? selections[activeKey] : null;
  const sideKey = activeEntry?.split ? activeSide : "both";
  const sideData = activeEntry ? activeEntry[sideKey] : null;

  const patch = (changes) => {
    setSelections((prev) => {
      const entry = { ...ensureEntry(prev, activeKey) };
      entry[sideKey] = { ...entry[sideKey], ...changes };
      return { ...prev, [activeKey]: entry };
    });
  };
  const toggleSplit = () => {
    setSelections((prev) => {
      const entry = { ...ensureEntry(prev, activeKey) };
      if (!entry.split) { entry.split = true; entry.left = { ...entry.both }; entry.right = { ...entry.both }; }
      else entry.split = false;
      return { ...prev, [activeKey]: entry };
    });
    setActiveSide("right");
  };
  const toggleState = (s) => {
    const cur = sideData.states;
    patch({ states: cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s] });
  };

  const isPolySelected = (key, side) => {
    const e = selections[key];
    if (!e) return false;
    if (!e.bilateral || !e.split) return true;
    return e[side]?.rating > 0 || e[side]?.states.length || e[side]?.note;
  };
  const ratingOf = (key, side) => {
    const e = selections[key];
    if (!e) return 0;
    return e.split ? e[side]?.rating : e.both?.rating;
  };

  function clean(side) { return { rating: side.rating || null, states: side.states, note: side.note.trim() }; }
  const intent = useMemo(() => {
    const sel = [];
    for (const [key, e] of Object.entries(selections)) {
      const r = regionByKey[key];
      const base = { region_key: key, zones: r.zones };
      if (e.bilateral && e.split) sel.push({ ...base, split: true, sides: { right: clean(e.right), left: clean(e.left) } });
      else sel.push({ ...base, split: false, ...clean(e.both) });
    }
    const byState = selectedStates.map((st) => ({
      state: st,
      regions: (STATE_TO_REGIONS[st] || []).map((k) => ({ region_key: k, zones: regionByKey[k].zones })),
    }));
    return { schema: "patient_intent_v0.2", primary_mode: mode === "area" ? "by_area" : "by_state", by_area: sel, by_state: byState };
  }, [selections, selectedStates, mode]);
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
  const fs = Math.max(imgDims.w, imgDims.h) || 1000;

  return (
    <div className="app">
      <header className="app-head">
        <span className="brand-tag">Concilium · Patient Intent</span>
        <h1 className="app-title">Which areas bother you most?</h1>
        <p className="app-sub">
          Tap the areas on your photo you'd like to treat. For each one you can say how much it bothers you and
          what concerns you. This is your input — it's shown to your physician alongside the system's assessment.
        </p>
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
              Detector failed to load ({modelError}). Check your connection; on a host with a strict CSP, allow
              storage.googleapis.com and cdn.jsdelivr.net.
            </p>
          )}
        </div>
      )}

      {imgSrc && (
        <>
          <div className="mode-toggle">
            <button className={mode === "area" ? "on" : ""} onClick={() => setMode("area")}>Facial areas</button>
            <button className={mode === "state" ? "on" : ""} onClick={() => setMode("state")}>Emotional states</button>
          </div>

          {mode === "state" && (
            <div className="states-panel">
              <h2>What would you like to change about how you look?</h2>
              <p className="sub">Pick any that apply — the related areas light up on the photo below.</p>
              <div className="state-cards">
                {ALL_STATES.map((st) => (
                  <button key={st} className={"state-card" + (selectedStates.includes(st) ? " on" : "")} onClick={() => toggleSelState(st)}>
                    {STATE_LABELS[st]}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="toolbar">
            <button className="btn btn-ghost btn-sm" onClick={() => { setImgSrc(null); setLandmarks(null); setSelections({}); setActiveKey(null); resetView(); }}>
              Change photo
            </button>
            <label className="dbg">
              <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
              Calibration points
            </label>
          </div>

          <div className="stage-wrap">
            <div
              className="stage"
              ref={stageRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            >
              <div className="stage-inner" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})` }}>
                <img ref={imgRef} src={imgSrc} alt="patient" onLoad={onImgLoad} crossOrigin="anonymous" draggable={false} />
                {landmarks && imgDims.w > 0 && (
                  <svg viewBox={`0 0 ${imgDims.w} ${imgDims.h}`} preserveAspectRatio="xMidYMid meet">
                    {polygons.map((pg) => {
                      const sel = mode === "area" ? isPolySelected(pg.region, pg.side) : stateRegions.has(pg.region);
                      const { cx, cy } = centroid(pg.pts);
                      const rating = mode === "area" && sel ? ratingOf(pg.region, pg.side) : 0;
                      return (
                        <g key={pg.region + pg.side}>
                          <polygon className={"poly" + (sel ? " selected" : "") + (mode === "area" ? "" : " ro")} points={toPath(pg.pts)} onClick={mode === "area" ? () => onAreaClick(pg.region, pg.side) : undefined} />
                          {mode === "area" && sel && rating > 0 && (
                            <>
                              <circle className="poly-badge" cx={cx} cy={cy} r={fs * 0.018} />
                              <text className="poly-badge-txt" x={cx} y={cy} style={{ fontSize: fs * 0.024 }}>{rating}</text>
                            </>
                          )}
                        </g>
                      );
                    })}
                    {externals.map((ex) => {
                      const sel = mode === "area" ? isPolySelected(ex.region, ex.side) : stateRegions.has(ex.region);
                      const rating = mode === "area" && sel ? ratingOf(ex.region, ex.side) : 0;
                      const label = regionByKey[ex.region].label;
                      const cw = fs * 0.15, chh = fs * 0.045;
                      const rx = ex.onLeft ? ex.chip.x : ex.chip.x - cw;
                      return (
                        <g key={ex.region + ex.side} onClick={mode === "area" ? () => onAreaClick(ex.region, ex.side) : undefined} style={{ cursor: mode === "area" ? "pointer" : "default" }}>
                          <line className={"lead-line" + (sel ? " on" : "")} x1={ex.ap.x} y1={ex.ap.y} x2={ex.onLeft ? rx + cw : rx} y2={ex.chip.y} />
                          <circle className={"lead-dot" + (sel ? " on" : "")} cx={ex.ap.x} cy={ex.ap.y} r={fs * (sel ? 0.009 : 0.006)} />
                          <rect className={"chip-rect" + (sel ? " on" : "")} x={rx} y={ex.chip.y - chh / 2} width={cw} height={chh} rx={chh / 2} />
                          <text className={"chip-txt" + (sel ? " on" : "")} x={rx + cw / 2} y={ex.chip.y} style={{ fontSize: fs * 0.02 }}>
                            {label}{rating ? ` · ${rating}` : ""}
                          </text>
                        </g>
                      );
                    })}
                    {debug && landmarks.map((p, i) => (
                      <g key={i}>
                        <circle className="dbg-dot" cx={p.x * imgDims.w} cy={p.y * imgDims.h} r={fs * 0.0025} />
                        <text className="dbg-num" x={p.x * imgDims.w} y={p.y * imgDims.h - fs * 0.004} style={{ fontSize: fs * 0.008 }}>{i}</text>
                      </g>
                    ))}
                  </svg>
                )}
              </div>
              {landmarks && (
                <div className="zoom-ctl">
                  <button onClick={() => zoomBtn(1.3)} aria-label="Zoom in">+</button>
                  <button onClick={() => zoomBtn(1 / 1.3)} aria-label="Zoom out">−</button>
                  <button className="reset" onClick={resetView} aria-label="Reset zoom">1:1</button>
                </div>
              )}
            </div>
            {detecting && <p className="status">Detecting facial points…</p>}
            {detectError && <p className="status err">{detectError}</p>}
            {landmarks && !detectError && <p className="hint">{mode === "area" ? "Tap an area to select it · pinch or scroll to zoom · tap again to remove" : "Pick concerns above · related areas light up here · pinch or scroll to zoom"}</p>}
          </div>

          {mode === "area" && activeRegion && activeEntry && (
            <div className="panel">
              <div className="panel-head">
                <h3 className="panel-title">{activeRegion.label}</h3>
                {activeEntry.split && (
                  <div className="panel-side-tabs">
                    {["right", "left"].map((s) => (
                      <button key={s} className={"side-tab" + (activeSide === s ? " active" : "")} onClick={() => setActiveSide(s)}>
                        {SIDE_LABELS[s]}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="field">
                <div className="field-label">How much does it bother you?</div>
                <div className="rating">
                  {[1, 2, 3].map((n) => (
                    <button key={n} className={sideData.rating === n ? "on" : ""} onClick={() => patch({ rating: n })}>
                      {n === 1 ? "A little" : n === 2 ? "Moderate" : "A lot"}
                    </button>
                  ))}
                </div>
              </div>

              {statesForRegion(activeKey).length > 0 && (
                <div className="field">
                  <div className="field-label">What about this area concerns you? <small>(choose any)</small></div>
                  <div className="chips">
                    {statesForRegion(activeKey).map((s) => (
                      <button key={s} className={"chip" + (sideData.states.includes(s) ? " on" : "")} onClick={() => toggleState(s)}>
                        {STATE_LABELS[s]}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="field">
                <div className="field-label">Anything else you'd like to add? <small>(optional)</small></div>
                <textarea placeholder="e.g. I'd like to look more refreshed" value={sideData.note} onChange={(e) => patch({ note: e.target.value })} />
              </div>

              <div className="split-row">
                {activeRegion.bilateral ? (
                  <>
                    <span>{activeEntry.split ? "Each side separately" : "Both sides together"}</span>
                    <button className="btn btn-ghost btn-sm" onClick={toggleSplit}>
                      {activeEntry.split ? "Merge sides" : "Separate sides"}
                    </button>
                  </>
                ) : <span />}
                <button className="btn btn-ghost btn-sm" onClick={() => removeRegion(activeKey)} style={{ color: "var(--danger)", borderColor: "#e0b3aa" }}>
                  Remove
                </button>
              </div>
            </div>
          )}

          {mode === "area" && (
            <div className="summary">
              <h2>Your selections ({selectedCount})</h2>
              {selectedCount === 0 && <p className="empty">No areas selected yet.</p>}
              {Object.entries(selections).map(([key, e]) => {
                const r = regionByKey[key];
                const rows = e.split ? ["right", "left"] : ["both"];
                return rows.map((s) => {
                  const d = e[s];
                  if (e.split && !d.rating && !d.states.length && !d.note) return null;
                  return (
                    <div className="sum-item" key={key + s} onClick={() => focusFromSummary(key, s)}>
                      <div>
                        <div className="sum-name">{r.label}{e.split ? ` · ${SIDE_LABELS[s]}` : ""}</div>
                        <div className="sum-meta">
                          {d.states.map((x) => STATE_LABELS[x]).join(" · ")}
                          {d.note ? (d.states.length ? " — " : "") + d.note : ""}
                          {!d.states.length && !d.note ? "—" : ""}
                        </div>
                      </div>
                      <div className="sum-rating">{d.rating ? `${d.rating}/3` : ""}</div>
                    </div>
                  );
                });
              })}
            </div>
          )}

          {mode === "state" && (
            <div className="summary">
              <h2>Selected concerns ({selectedStates.length})</h2>
              {selectedStates.length === 0 && <p className="empty">No concerns selected yet.</p>}
              {selectedStates.map((st) => (
                <div className="state-sum-item" key={st}>
                  <div className="state-sum-name">{STATE_LABELS[st]}</div>
                  <div className="state-sum-regions">{(STATE_TO_REGIONS[st] || []).map((k) => regionByKey[k].label).join(" · ") || "—"}</div>
                </div>
              ))}
            </div>
          )}

          {(mode === "area" ? selectedCount > 0 : selectedStates.length > 0) && (
            <div className="btn-row" style={{ marginTop: 18 }}>
              <button className="btn" onClick={exportJson}>Done</button>
              <button className="btn btn-ghost" onClick={() => console.log(JSON.stringify(intent, null, 2))}>Log patient_intent</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
