import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ============================================================
   Concilium — Patient Intent capture (Agent 6 front-end)
   ------------------------------------------------------------
   רץ על תמונת frontal_neutral. MediaPipe Face Landmarker מחזיר
   468 נקודות; כל אזור תפיסתי מורכב מ-convex hull של קבוצת אינדקסים.
   המיפוי region -> zones מתורגם בבקאנד. שכבת המצבים (zone -> state)
   מבודדת מהפייפליין האבחוני ומצטרפת רק ב-JOIN Final.

   ⚙️ כיול: כל קבוצות האינדקסים מרוכזות ב-HULLS למטה. אם פוליגון
   יושב לא מדויק על פנים אמיתיות — הפעילו "נקודות עזר" (debug)
   כדי לראות מספרי landmarks, והחליפו אינדקס בקבוצה הרלוונטית.
   ============================================================ */

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

/* convex-hull index groups (MediaPipe canonical 468 map) */
const HULLS = {
  forehead: [103, 67, 109, 10, 338, 297, 332, 334, 296, 9, 66, 105],
  glabella: [107, 9, 336, 285, 8, 55],
  brow_left: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  brow_right: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  upperlid_left: [33, 246, 161, 160, 159, 158, 157, 173, 133],
  upperlid_right: [263, 466, 388, 387, 386, 385, 384, 398, 362],
  undereye_left: [226, 31, 228, 229, 230, 231, 232, 233, 244, 133, 33],
  undereye_right: [446, 261, 448, 449, 450, 451, 452, 453, 464, 362, 263],
  temple_left: [21, 54, 103, 104, 68, 71],
  temple_right: [251, 284, 332, 333, 298, 301],
  cheek_left: [116, 117, 118, 119, 120, 100, 142, 36, 205, 187, 123, 50],
  cheek_right: [345, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352, 280],
  nasolabial_left: [129, 209, 49, 48, 64, 98, 60, 165, 92, 186, 57, 206],
  nasolabial_right: [358, 429, 279, 278, 294, 327, 290, 391, 322, 410, 287, 426],
  nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 48, 278],
  lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
  mouth_left: [61, 91, 146, 57, 43, 106, 182, 83, 84, 181],
  mouth_right: [291, 321, 375, 287, 273, 335, 406, 313, 314, 405],
  jaw_left: [132, 58, 172, 136, 150, 149, 176, 148, 152, 175],
  jaw_right: [361, 288, 397, 365, 379, 378, 400, 377, 152, 175],
  chin: [152, 175, 199, 200, 18, 83, 313, 406, 182, 421, 201],
  crowsfeet_left: [33, 130, 226, 247, 30, 29, 27, 28, 56, 190],
  crowsfeet_right: [263, 359, 446, 467, 260, 259, 257, 258, 286, 414],
};

/* zone -> perceived states (reverse of the approved state→zone table) */
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
  // אזורים שנוספו מחוץ לטבלת המצבים — אין תיוג מצב:
  nose: [],
  lateral_canthal: [],
  chin: [],
};

const STATE_LABELS = {
  tired: "עייפה",
  sad: "עצובה",
  angry: "כועסת / קשוחה",
  aged: "מבוגרת",
  heavy: "כבדה / שמוטה",
  gaunt: "חלולה / רזה",
  stressed: "מתוחה / מודאגת",
  dull: "עור עמום",
};

/* perceptual regions shown to the patient */
const REGIONS = [
  { key: "forehead", label: "מצח", bilateral: false, zones: ["forehead"], hulls: ["forehead"] },
  { key: "glabella", label: "בין הגבות", bilateral: false, zones: ["glabella"], hulls: ["glabella"] },
  { key: "brows", label: "גבות", bilateral: true, zones: ["brow_complex"], hulls: ["brow_left", "brow_right"] },
  { key: "upper_lids", label: "עפעפיים עליונים", bilateral: true, zones: ["upper_eyelid"], hulls: ["upperlid_left", "upperlid_right"] },
  { key: "under_eye", label: "מתחת לעיניים", bilateral: true, zones: ["tear_trough", "lower_eyelid"], hulls: ["undereye_left", "undereye_right"] },
  { key: "temples", label: "רקות", bilateral: true, zones: ["temples"], hulls: ["temple_left", "temple_right"] },
  { key: "cheeks", label: "לחיים", bilateral: true, zones: ["malar_zygomatic", "buccal_submalar"], hulls: ["cheek_left", "cheek_right"] },
  { key: "nasolabial", label: "קפלי אף-שפה", bilateral: true, zones: ["nasolabial"], hulls: ["nasolabial_left", "nasolabial_right"] },
  { key: "crows_feet", label: "קמטי צחוק", bilateral: true, zones: ["lateral_canthal"], hulls: ["crowsfeet_left", "crowsfeet_right"] },
  { key: "nose", label: "אף", bilateral: false, zones: ["nose"], hulls: ["nose"] },
  { key: "lips", label: "שפתיים", bilateral: false, zones: ["lips"], hulls: ["lips"] },
  { key: "mouth_corners", label: "זוויות הפה", bilateral: true, zones: ["oral_commissures"], hulls: ["mouth_left", "mouth_right"] },
  { key: "jawline", label: "קו הלסת", bilateral: true, zones: ["jawline", "mandibular_border"], hulls: ["jaw_left", "jaw_right"] },
  { key: "chin", label: "סנטר", bilateral: false, zones: ["chin"], hulls: ["chin"] },
];

// ב-frontal, צד שמאל של התמונה = צד ימין של המטופלת. hull index 0 = image-left
// = patient right. התוויות אנטומיות (של המטופלת); אם מתהפך על פנים אמיתיות — החליפו כאן.
const SIDE_LABELS = { both: "שני הצדדים", right: "ימין", left: "שמאל" };

/* ---------- geometry helpers ---------- */
function convexHull(points) {
  if (points.length < 3) return points;
  const pts = points.slice().sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function padHull(hull, factor = 1.08) {
  if (!hull.length) return hull;
  const cx = hull.reduce((s, p) => s + p.x, 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p.y, 0) / hull.length;
  return hull.map((p) => ({ x: cx + (p.x - cx) * factor, y: cy + (p.y - cy) * factor }));
}

function hullToPoints(indices, landmarks, w, h) {
  const pts = indices.map((i) => landmarks[i]).filter(Boolean).map((p) => ({ x: p.x * w, y: p.y * h }));
  return padHull(convexHull(pts));
}

function centroid(pts) {
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  return { cx, cy };
}

const toPath = (pts) => pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

/* ---------- component ---------- */
export default function FaceAreaSelector() {
  const [landmarker, setLandmarker] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [modelError, setModelError] = useState(null);

  const [imgSrc, setImgSrc] = useState(null);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  const [landmarks, setLandmarks] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState(null);

  const [selections, setSelections] = useState({}); // regionKey -> data
  const [activeKey, setActiveKey] = useState(null);
  const [activeSide, setActiveSide] = useState("both");
  const [debug, setDebug] = useState(false);

  const imgRef = useRef(null);
  const fileRef = useRef(null);

  /* load MediaPipe model once */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
        const lm = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "IMAGE",
          numFaces: 1,
        });
        if (!cancelled) { setLandmarker(lm); setLoadingModel(false); }
      } catch (e) {
        if (!cancelled) { setModelError(e?.message || "model load failed"); setLoadingModel(false); }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelections({}); setActiveKey(null); setLandmarks(null); setDetectError(null);
    const url = URL.createObjectURL(file);
    setImgSrc(url);
  };

  const onImgLoad = useCallback(() => {
    const el = imgRef.current;
    if (!el || !landmarker) return;
    setImgDims({ w: el.naturalWidth, h: el.naturalHeight });
    setDetecting(true);
    setDetectError(null);
    try {
      const res = landmarker.detect(el);
      if (res?.faceLandmarks?.length) setLandmarks(res.faceLandmarks[0]);
      else setDetectError("לא זוהו פנים בתמונה. נסו תמונת חזית ברורה ומוארת.");
    } catch (e) {
      setDetectError("הזיהוי נכשל: " + (e?.message || ""));
    } finally {
      setDetecting(false);
    }
  }, [landmarker]);

  /* build polygons for every region/side */
  const polygons = useMemo(() => {
    if (!landmarks) return [];
    const { w, h } = imgDims;
    const out = [];
    for (const r of REGIONS) {
      if (r.bilateral) {
        // hulls order: [image-left hull, image-right hull].
        // image-left = patient right; image-right = patient left.
        r.hulls.forEach((hullKey, i) => {
          const pts = hullToPoints(HULLS[hullKey], landmarks, w, h);
          if (pts.length >= 3) out.push({ region: r.key, side: i === 0 ? "right" : "left", pts });
        });
      } else {
        const pts = hullToPoints(HULLS[r.hulls[0]], landmarks, w, h);
        if (pts.length >= 3) out.push({ region: r.key, side: "both", pts });
      }
    }
    return out;
  }, [landmarks, imgDims]);

  const regionByKey = useMemo(() => Object.fromEntries(REGIONS.map((r) => [r.key, r])), []);

  const statesForRegion = useCallback((regionKey) => {
    const r = regionByKey[regionKey];
    if (!r) return [];
    const set = new Set();
    r.zones.forEach((z) => (ZONE_STATES[z] || []).forEach((s) => set.add(s)));
    return [...set];
  }, [regionByKey]);

  /* ensure a selection entry exists */
  function ensureEntry(prev, regionKey) {
    if (prev[regionKey]) return prev[regionKey];
    const r = regionByKey[regionKey];
    return { split: false, both: blankSide(), left: blankSide(), right: blankSide(), bilateral: r.bilateral };
  }
  const blankSide = () => ({ rating: 0, states: [], note: "" });

  const selectRegion = (regionKey, side) => {
    setSelections((prev) => {
      const entry = { ...ensureEntry(prev, regionKey) };
      return { ...prev, [regionKey]: entry };
    });
    setActiveKey(regionKey);
    const r = regionByKey[regionKey];
    setActiveSide(r.bilateral && selections[regionKey]?.split ? side : "both");
  };

  const removeRegion = (regionKey) => {
    setSelections((prev) => { const n = { ...prev }; delete n[regionKey]; return n; });
    if (activeKey === regionKey) setActiveKey(null);
  };

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
      if (!entry.split) {
        // copy "both" into each side as starting point
        entry.split = true;
        entry.left = { ...entry.both };
        entry.right = { ...entry.both };
      } else {
        entry.split = false;
      }
      return { ...prev, [activeKey]: entry };
    });
    setActiveSide("right");
  };

  const toggleState = (s) => {
    const cur = sideData.states;
    patch({ states: cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s] });
  };

  /* which polygons are "selected" for highlight */
  const isPolySelected = (regionKey, side) => {
    const e = selections[regionKey];
    if (!e) return false;
    if (!e.bilateral || !e.split) return true; // whole region selected
    return e[side]?.rating > 0 || e[side]?.states.length || e[side]?.note;
  };

  /* assemble export object */
  const intent = useMemo(() => {
    const sel = [];
    for (const [key, e] of Object.entries(selections)) {
      const r = regionByKey[key];
      const base = { region_key: key, zones: r.zones };
      if (e.bilateral && e.split) {
        sel.push({ ...base, split: true, sides: { left: clean(e.left), right: clean(e.right) } });
      } else {
        sel.push({ ...base, split: false, ...clean(e.both) });
      }
    }
    return { schema: "patient_intent_v0.1", selections: sel };
  }, [selections, regionByKey]);

  function clean(side) {
    return { rating: side.rating || null, states: side.states, note: side.note.trim() };
  }

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(intent, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "patient_intent.json";
    a.click();
  };

  const selectedCount = Object.keys(selections).length;

  return (
    <div className="app">
      <header className="app-head">
        <span className="brand-tag">Concilium · Patient Intent</span>
        <h1 className="app-title">איזה אזורים הכי מפריעים לך?</h1>
        <p className="app-sub">
          סמני על התמונה את האזורים שהיית רוצה לטפל בהם. לכל אזור אפשר לציין כמה הוא מפריע ומה מטריד בו.
          זו הבחירה שלך — היא תוצג לרופא לצד חוות הדעת של המערכת.
        </p>
      </header>

      {!imgSrc && (
        <div className="upload">
          <h2 style={{ margin: "0 0 4px", fontSize: 17 }}>העלאת תמונת חזית</h2>
          <p>תמונת פנים חזיתית, מוארת, מבט ישר למצלמה (frontal_neutral).</p>
          <button className="btn" disabled={loadingModel && !modelError} onClick={() => fileRef.current?.click()}>
            {loadingModel ? "טוען מנוע זיהוי…" : "בחירת תמונה"}
          </button>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
          {modelError && (
            <p className="status err" style={{ marginTop: 14 }}>
              טעינת מנוע הזיהוי נכשלה ({modelError}). ודאו חיבור רשת; בסביבת הרצה עם CSP מגביל יש להתיר את
              storage.googleapis.com ו-cdn.jsdelivr.net.
            </p>
          )}
        </div>
      )}

      {imgSrc && (
        <>
          <div className="toolbar">
            <button className="btn-ghost btn btn-sm" onClick={() => { setImgSrc(null); setLandmarks(null); setSelections({}); setActiveKey(null); }}>
              תמונה אחרת
            </button>
            <label className="dbg">
              <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
              נקודות עזר (כיול)
            </label>
          </div>

          <div className="stage-wrap">
            <div className="stage">
              <img ref={imgRef} src={imgSrc} alt="תמונת המטופלת" onLoad={onImgLoad} crossOrigin="anonymous" />
              {landmarks && imgDims.w > 0 && (
                <svg viewBox={`0 0 ${imgDims.w} ${imgDims.h}`} preserveAspectRatio="xMidYMid meet">
                  {polygons.map((pg) => {
                    const sel = isPolySelected(pg.region, pg.side);
                    const { cx, cy } = centroid(pg.pts);
                    const e = selections[pg.region];
                    let rating = 0;
                    if (e) rating = e.split ? e[pg.side]?.rating : e.both?.rating;
                    return (
                      <g key={pg.region + pg.side}>
                        <polygon
                          className={"poly" + (sel ? " selected" : "")}
                          points={toPath(pg.pts)}
                          onClick={() => selectRegion(pg.region, pg.side)}
                        />
                        {sel && rating > 0 && (
                          <>
                            <circle className="poly-badge" cx={cx} cy={cy} r={Math.max(imgDims.w, imgDims.h) * 0.018} />
                            <text className="poly-badge-txt" x={cx} y={cy} style={{ fontSize: Math.max(imgDims.w, imgDims.h) * 0.022 }}>
                              {rating}
                            </text>
                          </>
                        )}
                      </g>
                    );
                  })}
                  {debug && landmarks.map((p, i) => (
                    <g key={i}>
                      <circle className="dbg-dot" cx={p.x * imgDims.w} cy={p.y * imgDims.h} r={imgDims.w * 0.003} />
                      <text className="dbg-num" x={p.x * imgDims.w} y={p.y * imgDims.h - imgDims.w * 0.004} style={{ fontSize: imgDims.w * 0.009 }}>{i}</text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
            {detecting && <p className="status">מזהה נקודות פנים…</p>}
            {detectError && <p className="status err">{detectError}</p>}
            {landmarks && !detectError && <p className="hint">לחצי על אזור כדי לבחור אותו</p>}
          </div>

          {/* selection panel */}
          {activeRegion && activeEntry && (
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
                <div className="field-label">כמה זה מפריע לך?</div>
                <div className="rating">
                  {[1, 2, 3].map((n) => (
                    <button key={n} className={sideData.rating === n ? "on" : ""} onClick={() => patch({ rating: n })}>
                      {n === 1 ? "קצת" : n === 2 ? "בינוני" : "מאוד"}
                    </button>
                  ))}
                </div>
              </div>

              {statesForRegion(activeKey).length > 0 && (
                <div className="field">
                  <div className="field-label">
                    מה מטריד אותך באזור הזה? <small>(אפשר לבחור כמה)</small>
                  </div>
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
                <div className="field-label">משהו נוסף שתרצי להגיד? <small>(לא חובה)</small></div>
                <textarea
                  placeholder="לדוגמה: אני רוצה להיראות רעננה יותר"
                  value={sideData.note}
                  onChange={(e) => patch({ note: e.target.value })}
                />
              </div>

              <div className="split-row">
                {activeRegion.bilateral ? (
                  <>
                    <span>{activeEntry.split ? "כל צד בנפרד" : "שני הצדדים יחד"}</span>
                    <button className="btn-ghost btn btn-sm" onClick={toggleSplit}>
                      {activeEntry.split ? "אחד את הצדדים" : "הפרד צדדים"}
                    </button>
                  </>
                ) : <span />}
                <button className="btn-ghost btn btn-sm" onClick={() => removeRegion(activeKey)} style={{ color: "#b3402d", borderColor: "#e0b3aa" }}>
                  הסר אזור
                </button>
              </div>
            </div>
          )}

          {/* summary */}
          <div className="summary">
            <h2>הבחירות שלך ({selectedCount})</h2>
            {selectedCount === 0 && <p className="empty">עדיין לא נבחרו אזורים.</p>}
            {Object.entries(selections).map(([key, e]) => {
              const r = regionByKey[key];
              const rows = e.split ? ["right", "left"] : ["both"];
              return rows.map((s) => {
                const d = e[s];
                if (e.split && !d.rating && !d.states.length && !d.note) return null;
                return (
                  <div className="sum-item" key={key + s}>
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

          {selectedCount > 0 && (
            <div className="btn-row" style={{ marginTop: 18 }}>
              <button className="btn" onClick={exportJson}>סיום ושמירה</button>
              <button className="btn-ghost btn" onClick={() => console.log(JSON.stringify(intent, null, 2))}>
                הצג patient_intent (קונסול)
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
