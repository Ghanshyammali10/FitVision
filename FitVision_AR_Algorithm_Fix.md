# FitVision — AR Overlay Algorithm Fix & Depth Tracking
### Build Prompt for Antigravity Agent

---

## 🎯 Objective

Rewrite the `drawAROverlay` function in `client/js/ai.js` to fix three problems visible in the current demo:

1. **Garment sits too low** — collar appears at chest level instead of shoulder/neck junction
2. **Garment is too wide** — especially on narrow-framed subjects
3. **No depth tracking** — garment does not grow/shrink as subject moves closer/farther from camera

The goal is a garment overlay that looks like the subject is **actually wearing** the item — locked to their body at all distances and positions.

---

## 📁 File to Edit

```
client/js/ai.js  →  function drawAROverlay(...)
```

---

## 🧠 Core Concept: Why Shoulder Width = Depth

You do **not** need a depth camera or Z-coordinate to simulate perspective scaling. MediaPipe gives you 2D landmarks in the camera frame. When a person steps backward:

- Their body occupies fewer pixels
- Their shoulder-to-shoulder distance in pixels **shrinks**

So `shoulderWidth` (Euclidean pixel distance between landmark 11 and 12) is your depth proxy. Every garment dimension must be derived **proportionally** from `shoulderWidth`. If you do this correctly, depth scaling is automatic — no extra logic needed.

---

## 🔴 What Is Wrong in the Current Algorithm

### Problem 1 — Pivot offset is 40% of garment height (too much)

```javascript
// CURRENT CODE (broken)
const centerY = ((ls.y + rs.y) / 2) + (garmentHeight * 0.40);
```

`drawImage` is called at `-height/2` from the pivot, so the actual **top of the drawn garment** lands at:

```
pivotY - garmentHeight/2
= (midShoulderY + 0.40 × H) - 0.50 × H
= midShoulderY - 0.10 × H
```

Since `garmentHeight` is typically 300–450px on screen, this means the top of the garment is drawn **30–45px above** the shoulder landmark. But the shoulder landmark itself sits below the actual neck, so the collar ends up at chest level. The fix is to use a small collar offset (~12% of garment height) so the top of the garment aligns with the real shoulder line.

### Problem 2 — Width multiplier 1.75 is too aggressive

```javascript
// CURRENT CODE (broken)
garmentWidth = shoulderWidth * 1.75;
```

This makes the garment far too wide, especially for subjects with narrower frames or those standing farther away. Fix: use `1.50`.

### Problem 3 — X coordinate is NOT mirrored

MediaPipe returns landmarks for the **raw, unmirrored** video. But the canvas draws a **mirrored** video feed (standard webcam behaviour). If you use `landmark.x * canvasWidth` directly without mirroring, the garment drifts to the opposite side when the subject moves left or right.

```javascript
// CURRENT CODE (broken) — landmark X not mirrored
const lsx = ls.x * canvasWidth;

// CORRECT — mirror to match the flipped video
const lsx = canvasWidth - (ls.x * canvasWidth);
```

---

## ✅ Complete Replacement Function

Replace the entire `drawAROverlay` function in `ai.js` with the following:

```javascript
function drawAROverlay(canvasEl, videoEl, poseLandmarks, garmentImg) {
    const ctx = canvasEl.getContext('2d');
    const cW  = canvasEl.width;
    const cH  = canvasEl.height;
    const vW  = videoEl.videoWidth  || cW;
    const vH  = videoEl.videoHeight || cH;

    ctx.clearRect(0, 0, cW, cH);

    // ── Step 1: Draw the mirrored video feed ─────────────────────────────────
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, -cW, 0, cW, cH);
    ctx.restore();

    if (!poseLandmarks || !garmentImg) return;

    // ── Step 2: Helper — convert landmark to canvas pixel (with X mirror) ────
    // MediaPipe gives raw-video coordinates. Since we mirror the canvas,
    // we must mirror the X axis of every landmark too.
    const px = (lm) => ({
        x: cW - (lm.x * cW),   // ← mirrored X
        y: lm.y * cH
    });

    const ls = px(poseLandmarks[11]);  // left shoulder
    const rs = px(poseLandmarks[12]);  // right shoulder
    const lhRaw = poseLandmarks[23];   // left hip  (raw, check visibility first)
    const rhRaw = poseLandmarks[24];   // right hip

    // ── Step 3: Bail out if shoulders not visible ─────────────────────────────
    const shoulderVis = (poseLandmarks[11].visibility + poseLandmarks[12].visibility) / 2;
    if (shoulderVis < 0.5) return;

    // ── Step 4: Shoulder width — this is the DEPTH PROXY ─────────────────────
    // When subject moves backward → shoulder width in pixels shrinks
    // When subject moves forward → shoulder width grows
    // ALL garment dimensions derive from this single measurement.
    const shoulderWidth = Math.hypot(rs.x - ls.x, rs.y - ls.y);

    // ── Step 5: Garment dimensions — 100% proportional, no hardcoded values ──
    const garmentWidth = shoulderWidth * 1.50;

    let garmentHeight;
    const hipsVisible = lhRaw && rhRaw
        && poseLandmarks[23].visibility > 0.4
        && poseLandmarks[24].visibility > 0.4;

    if (hipsVisible) {
        const lh = px(lhRaw);
        const rh = px(rhRaw);
        const avgShoulderY = (ls.y + rs.y) / 2;
        const avgHipY      = (lh.y  + rh.y)  / 2;
        const torsoHeight  = avgHipY - avgShoulderY;

        // Sanity check: torso must be positive and at least 20px
        // (guards against sitting pose or occluded hips giving garbage values)
        garmentHeight = torsoHeight > 20
            ? torsoHeight * 1.15
            : garmentWidth * 1.35;    // fallback
    } else {
        // Hips not visible (e.g. laptop webcam, close-up shot)
        // Use aspect ratio fallback — still scales with depth automatically
        garmentHeight = garmentWidth * 1.35;
    }

    // ── Step 6: Rotation from shoulder tilt ──────────────────────────────────
    // Garment tilts when subject leans side-to-side
    const angle = Math.atan2(rs.y - ls.y, rs.x - ls.x);

    // ── Step 7: Pivot point ───────────────────────────────────────────────────
    // We draw from center. The garment's collar (top edge) should sit at the
    // shoulder/neck junction, not the chest.
    //
    // drawImage draws the image centered at pivot:
    //   top of image = pivotY - garmentHeight/2
    //
    // We want:  top of image ≈ midShoulderY
    // So:       pivotY = midShoulderY + garmentHeight/2  ... but that's 50%
    //
    // In practice the garment image has a small transparent margin at the top
    // before the actual collar. collarOffset (12%) accounts for this gap.
    // Tune between 0.08 and 0.18 based on your garment cutout proportions.
    const midX = (ls.x + rs.x) / 2;
    const midY = (ls.y + rs.y) / 2;
    const collarOffset = garmentHeight * 0.12;

    const targetX = midX;
    const targetY = midY + collarOffset;

    // ── Step 8: Adaptive EMA smoothing (anti-jitter) ─────────────────────────
    // Raw MediaPipe landmarks jump frame-to-frame. We smooth with EMA.
    // When subject moves FAST → higher factor (garment catches up quickly)
    // When subject is STILL  → lower factor  (garment stays rock-steady)
    if (!drawAROverlay._state) {
        // First frame — initialise state to current values (no lerp)
        drawAROverlay._state = {
            x: targetX, y: targetY,
            w: garmentWidth, h: garmentHeight,
            a: angle
        };
    }

    const prev = drawAROverlay._state;

    // Motion magnitude based on pivot point delta
    const motionDelta = Math.hypot(targetX - prev.x, targetY - prev.y);
    const factor = motionDelta > 15 ? 0.55 : 0.12;

    const sX = prev.x + (targetX      - prev.x) * factor;
    const sY = prev.y + (targetY      - prev.y) * factor;
    const sW = prev.w + (garmentWidth  - prev.w) * factor;
    const sH = prev.h + (garmentHeight - prev.h) * factor;
    const sA = prev.a + (angle         - prev.a) * factor;

    drawAROverlay._state = { x: sX, y: sY, w: sW, h: sH, a: sA };

    // ── Step 9: Render ────────────────────────────────────────────────────────
    ctx.save();
    ctx.translate(sX, sY);
    ctx.rotate(sA);
    ctx.drawImage(garmentImg, -sW / 2, -sH / 2, sW, sH);
    ctx.restore();
}
```

---

## 🔧 Tuning Reference

After implementation, if alignment is still slightly off, adjust only these two constants:

| Constant | Location in code | Effect | Tuning range |
|---|---|---|---|
| `collarOffset` multiplier | Step 7 | Moves garment up/down relative to shoulders | `0.08` – `0.20` |
| `garmentWidth` multiplier | Step 5 | Makes garment wider/narrower | `1.40` – `1.60` |
| `garmentHeight` ratio (fallback) | Step 5 | Changes shirt length when hips not visible | `1.25` – `1.50` |
| `torsoHeight` multiplier | Step 5 | Shirt length when hips visible | `1.10` – `1.25` |
| EMA fast factor | Step 8 | How quickly garment follows fast movement | `0.45` – `0.65` |
| EMA slow factor | Step 8 | How smooth garment is when still | `0.08` – `0.18` |

**Start with `collarOffset = 0.12`.** If the collar is still below the neck, reduce to `0.08`. If it rides above the shoulders, increase to `0.18`.

---

## 🔄 How Depth Scaling Works (No Extra Code Needed)

This is the most important concept to understand:

```
Subject 1m away:   shoulderWidth = ~80px  →  garmentWidth = 120px  garmentHeight = 162px
Subject 2m away:   shoulderWidth = ~40px  →  garmentWidth = 60px   garmentHeight = 81px
Subject 3m away:   shoulderWidth = ~22px  →  garmentWidth = 33px   garmentHeight = 45px
```

Because every dimension is a **ratio of shoulderWidth**, and shoulderWidth naturally encodes depth (perspective projection), the garment automatically scales as the subject moves. You get free perspective scaling with zero extra computation.

---

## ⚠️ Common Mistakes to Avoid

### Do NOT hardcode any pixel sizes
```javascript
// ❌ WRONG — breaks at all depths except the one you tested at
garmentWidth  = 200;
garmentHeight = 280;

// ✅ CORRECT — scales automatically
garmentWidth  = shoulderWidth * 1.50;
garmentHeight = garmentWidth  * 1.35;
```

### Do NOT forget to mirror landmark X coordinates
```javascript
// ❌ WRONG — garment drifts opposite direction to subject movement
const x = landmark.x * canvasWidth;

// ✅ CORRECT
const x = canvasWidth - (landmark.x * canvasWidth);
```

### Do NOT reset `_state` every frame
The EMA smoothing state (`drawAROverlay._state`) must persist across frames. Only initialise it once on the first call. If you reset it every frame, there is no smoothing and the garment will jitter.

### Do NOT use `lm[23].x` directly for hips without a visibility check
MediaPipe will still return hip coordinates even when they are off-screen or occluded, but with low visibility scores. Always check `visibility > 0.4` before using hip landmarks, or you'll get garbage `torsoHeight` values.

---

## ✅ Self-Check Checklist

Before marking this as complete, verify each item:

- [ ] `drawAROverlay._state` is only initialised on the first call (not reset every frame)
- [ ] Landmark X coordinates are mirrored: `cW - (lm.x * cW)`
- [ ] `garmentWidth` uses multiplier `1.50` (not `1.75`)
- [ ] `collarOffset` uses `garmentHeight * 0.12` (not `garmentHeight * 0.40`)
- [ ] Hip-based height includes the `torsoHeight > 20` sanity check
- [ ] `ctx.scale(-1, 1)` and `ctx.drawImage(videoEl, -cW, 0, ...)` are used to mirror the video
- [ ] `ctx.save()` and `ctx.restore()` wrap all canvas transform operations
- [ ] The old `drawAROverlay` function is completely replaced (not duplicated)

---

## 📐 Algorithm Summary (Visual)

```
MediaPipe PoseLandmarker
        │
        ▼
Landmark 11 (L shoulder) + Landmark 12 (R shoulder)
        │
        ├─── shoulderWidth = Euclidean distance (L ↔ R) in pixels
        │            │
        │            │   ← THIS IS YOUR DEPTH SIGNAL
        │            │
        ├─── garmentWidth  = shoulderWidth × 1.50
        │
        ├─── garmentHeight = torsoHeight × 1.15   (if hips visible)
        │                  = garmentWidth × 1.35  (fallback)
        │
        ├─── angle         = atan2(rs.y - ls.y, rs.x - ls.x)
        │
        ├─── pivotX        = (ls.x + rs.x) / 2    [mirrored]
        │
        └─── pivotY        = (ls.y + rs.y) / 2 + garmentHeight × 0.12
                                                        │
                                              small collar offset
                                              aligns top of garment
                                              to shoulder/neck line

All 5 values → EMA smoothing (adaptive factor) → ctx.translate + ctx.rotate + ctx.drawImage
```

---

*Generated for FitVision v6 — Pure MediaPipe client-side AR pipeline*
