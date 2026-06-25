# Concilium — Patient Intent (Agent 6 front-end)

ממשק לכידת כוונות מטופלת. רץ על תמונת `frontal_neutral`, מזהה 468 נקודות פנים עם
MediaPipe Face Landmarker, ומציג פוליגונים לחיצים לבחירת אזורים. לכל אזור: דירוג 1–3,
הצעות תיוג מצב (zone→state), הערה חופשית, ואפשרות הפרדת צדדים.

הפלט הוא אובייקט `patient_intent_v0.1` — מבודד מהפייפליין האבחוני (Agents 1–5),
מצטרף רק ב-JOIN Final.

## הרצה מקומית

```bash
npm install
npm run dev
```

## פריסה ל-Vercel

הפרויקט הוא Vite סטנדרטי. ב-Vercel:
- Framework Preset: **Vite**
- Build Command: `npm run build`
- Output Directory: `dist`

או דרך CLI:

```bash
npm i -g vercel
vercel
```

אין צורך במשתני סביבה. מודל MediaPipe נטען מ-CDN רשמי בזמן ריצה
(`storage.googleapis.com`) וה-WASM מ-`cdn.jsdelivr.net` — שניהם פתוחים ב-Vercel.

## כיול הפוליגונים

אינדקסי ה-landmarks מרוכזים ב-`HULLS` בראש `src/FaceAreaSelector.jsx`.
אם פוליגון יושב לא מדויק על פנים אמיתיות — סמנו "נקודות עזר (כיול)" בממשק כדי
לראות את מספרי ה-468 נקודות, והחליפו את האינדקס הרלוונטי בקבוצה.

## מיפוי שמשתקף בקוד

- `REGIONS` — 14 אזורים תפיסתיים (כולל nose, crows_feet, chin שאין להם תיוג מצב).
- `ZONE_STATES` — מיפוי zone→state, היפוך של טבלת המצבים המאושרת.
- `HULLS` — קבוצות אינדקסים לכל אזור (convex hull).

## אינטגרציה עתידית

- `submental_view` — נדרש כדי לכסות `neck`/`platysma`/`mandibular_border` שלא נראים על frontal.
- שכבת `state→zone` (כיוון הפוך) — בחירת מצב מדגישה אזורים. כרגע ממומש רק `zone→state`.
- ה-`patient_intent` JSON מוכן לצריכה ב-backend ל-JOIN Final.
