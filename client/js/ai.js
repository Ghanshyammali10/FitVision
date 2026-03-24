// ==========================================
// FitVision — AI Module (v5 — Front/Back AR)
// TensorFlow.js: Body Segmentation, Pose Detection, Face Detection
// ==========================================

let segmenter = null;
let poseDetector = null;
let faceDetector = null;
let modelsLoaded = false;

// ═══════════════════════════════════════════
// SMOOTHING — Adaptive Exponential Moving Average
// ═══════════════════════════════════════════
const smoothing = {
  x: null, y: null, w: null, h: null, angle: null,
  baseFactor: 0.3,
  minFactor: 0.15,
  maxFactor: 0.6,
  velocityThreshold: 15,
  confidence: 1.0
};

function smooth(key, rawValue) {
  if (smoothing[key] === null || isNaN(smoothing[key])) {
    smoothing[key] = rawValue;
    return rawValue;
  }
  let factor = smoothing.baseFactor;
  if (key === 'x' || key === 'y') {
    const delta = Math.abs(rawValue - smoothing[key]);
    const t = Math.min(delta / smoothing.velocityThreshold, 1.0);
    factor = smoothing.minFactor + t * (smoothing.maxFactor - smoothing.minFactor);
  }
  smoothing[key] = smoothing[key] + factor * (rawValue - smoothing[key]);
  return smoothing[key];
}

function resetSmoothing() {
  smoothing.x = null;
  smoothing.y = null;
  smoothing.w = null;
  smoothing.h = null;
  smoothing.angle = null;
  smoothing.confidence = 1.0;
}

// ═══════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════

async function loadModels(onProgress) {
  const report = (model, status, percent) => {
    if (onProgress) onProgress({ model, status, percent });
  };

  try {
    report('Body Segmentation', 'Loading...', 10);
    try {
      segmenter = await bodySegmentation.createSegmenter(
        bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
        { runtime: 'tfjs', modelType: 'general' }
      );
      report('Body Segmentation', 'Ready ✓', 35);
    } catch (e) {
      console.warn('Body segmentation failed:', e.message);
      report('Body Segmentation', 'Skipped (using fallback)', 35);
    }

    report('Pose Detection', 'Loading...', 40);
    try {
      poseDetector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true
        }
      );
      report('Pose Detection', 'Ready ✓', 70);
    } catch (e) {
      console.warn('Pose detection failed:', e.message);
      report('Pose Detection', 'Skipped', 70);
    }

    report('Face Detection', 'Loading...', 75);
    try {
      faceDetector = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        { runtime: 'tfjs', maxFaces: 1 }
      );
      report('Face Detection', 'Ready ✓', 100);
    } catch (e) {
      console.warn('Face detection failed:', e.message);
      report('Face Detection', 'Skipped', 100);
    }

    modelsLoaded = true;
    report('All Models', 'Ready!', 100);
  } catch (error) {
    console.error('Model loading error:', error);
    modelsLoaded = true;
    report('Models', 'Loaded with fallbacks', 100);
  }
}

// ═══════════════════════════════════════════
// BACKGROUND REMOVAL
// ═══════════════════════════════════════════

async function removeBackground(videoElement) {
  if (!segmenter) return fallbackCapture(videoElement);

  try {
    const segmentation = await segmenter.segmentPeople(videoElement, {
      flipHorizontal: false,
      multiSegmentation: false,
      segmentBodyParts: false
    });

    if (!segmentation || segmentation.length === 0) return fallbackCapture(videoElement);

    const canvas = document.createElement('canvas');
    const w = videoElement.videoWidth || 640;
    const h = videoElement.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(videoElement, 0, 0, w, h);
    const frame = ctx.getImageData(0, 0, w, h);
    const mask = await segmentation[0].mask.toImageData();

    for (let i = 0; i < frame.data.length; i += 4) {
      if (mask.data[i] < 128) {
        frame.data[i + 3] = 0;
      }
    }

    ctx.putImageData(frame, 0, 0);
    return canvas.toDataURL('image/png');
  } catch (error) {
    console.error('BG removal error:', error);
    return fallbackCapture(videoElement);
  }
}

function fallbackCapture(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.8);
}

// ═══════════════════════════════════════════
// ENHANCED AUTO-CROP — Combination Mask Pipeline
// ═══════════════════════════════════════════
//
// Pipeline:
//  1. segmentation mask (person-vs-bg)
//  2. pose-based torso polygon (shoulders→hips, arm-clipped)
//  3. combination mask: keep pixel ONLY if person AND inside torso
//  4. skin pixel removal (HSV-based)
//  5. edge smoothing (blur + alpha feather)
//  6. tight alpha bounding box trim
//
// Result: clean shirt-only cutout, no arms/face/skin

// ── Skin Detection (HSV-based) ──
// Returns true if the pixel is likely exposed skin
function isSkinPixel(r, g, b) {
  // Rule 1: RGB thresholds (covers most skin tones)
  if (r > 95 && g > 40 && b > 20) {
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    if ((maxC - minC) > 15 && Math.abs(r - g) > 15 && r > g && r > b) {
      return true;
    }
  }
  // Rule 2: HSV-based detection for broader skin tones
  // Convert to normalized
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const cMax = Math.max(rn, gn, bn);
  const cMin = Math.min(rn, gn, bn);
  const delta = cMax - cMin;

  let h = 0;
  if (delta > 0) {
    if (cMax === rn) h = 60 * (((gn - bn) / delta) % 6);
    else if (cMax === gn) h = 60 * (((bn - rn) / delta) + 2);
    else h = 60 * (((rn - gn) / delta) + 4);
    if (h < 0) h += 360;
  }
  const s = cMax > 0 ? delta / cMax : 0;
  const v = cMax;

  // Skin in HSV: H ∈ [0°, 50°], S ∈ [0.1, 0.7], V ∈ [0.2, 1.0]
  if (h >= 0 && h <= 50 && s >= 0.1 && s <= 0.7 && v >= 0.2) {
    return true;
  }
  return false;
}

// ── Point-in-Polygon test (ray casting) ──
function isInsidePolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// ── Build torso polygon from pose keypoints ──
// Returns array of [x, y] points defining the garment region
function buildTorsoPolygon(keypoints, imgW, imgH) {
  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  const ls = kp['left_shoulder'];
  const rs = kp['right_shoulder'];
  const lh = kp['left_hip'];
  const rh = kp['right_hip'];
  const le = kp['left_elbow'];
  const re = kp['right_elbow'];

  if (!ls || !rs || ls.score < 0.3 || rs.score < 0.3) return null;

  const shoulderW = Math.abs(rs.x - ls.x);
  if (shoulderW < 10) return null;

  // ── Top edge (neck line) ── tighter than before
  const neckY = Math.min(ls.y, rs.y) - shoulderW * 0.12;
  const neckCenterX = (ls.x + rs.x) / 2;
  const neckWidth = shoulderW * 0.20; // narrower neckhole

  // ── Bottom edge (waist) ── slightly above hips, not below
  let bottomY;
  if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
    bottomY = Math.max(lh.y, rh.y) - shoulderW * 0.05; // tighter
  } else {
    bottomY = Math.max(ls.y, rs.y) + shoulderW * 1.2;
  }

  // ── Side edges ── tighter, use elbows to clip arms
  let leftX = Math.min(ls.x, rs.x) - shoulderW * 0.25; // tighter (was 0.45)
  let rightX = Math.max(ls.x, rs.x) + shoulderW * 0.25;

  // Clip inward using elbows (removes arm regions)
  if (le && le.score > 0.3) {
    leftX = Math.max(leftX, le.x + shoulderW * 0.05);
  }
  if (re && re.score > 0.3) {
    rightX = Math.min(rightX, re.x - shoulderW * 0.05);
  }

  // ── Hip-level width ── slightly narrower than shoulders
  let leftHipX, rightHipX;
  if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
    leftHipX = Math.min(lh.x, rh.x) - shoulderW * 0.15; // tighter (was 0.2)
    rightHipX = Math.max(lh.x, rh.x) + shoulderW * 0.15;
  } else {
    leftHipX = leftX + shoulderW * 0.05;
    rightHipX = rightX - shoulderW * 0.05;
  }

  // Clamp helper
  const c = (v, min, max) => Math.max(min, Math.min(max, v));

  // Build polygon: neckhole shape at top → shoulders → hips → back
  return [
    [c(leftX, 0, imgW),              c(neckY, 0, imgH)],         // top-left
    [c(neckCenterX - neckWidth, 0, imgW), c(neckY, 0, imgH)],    // neckhole left
    [neckCenterX,                     c(neckY + shoulderW * 0.04, 0, imgH)], // neckhole dip
    [c(neckCenterX + neckWidth, 0, imgW), c(neckY, 0, imgH)],    // neckhole right
    [c(rightX, 0, imgW),             c(neckY, 0, imgH)],         // top-right
    [c(rightX, 0, imgW),             c((neckY + bottomY) * 0.5, 0, imgH)], // mid-right (sleeve line)
    [c(rightHipX, 0, imgW),          c(bottomY, 0, imgH)],       // bottom-right
    [c(leftHipX, 0, imgW),           c(bottomY, 0, imgH)],       // bottom-left
    [c(leftX, 0, imgW),              c((neckY + bottomY) * 0.5, 0, imgH)], // mid-left (sleeve line)
  ];
}

// ── Main enhanced crop function ──
async function autoCropGarment(dataUrl, videoElement) {
  return new Promise(async (resolve) => {
    const img = new Image();
    img.onload = async () => {
      const sw = img.width;
      const sh = img.height;

      // Draw the bg-removed image
      const srcCanvas = document.createElement('canvas');
      srcCanvas.width = sw;
      srcCanvas.height = sh;
      const srcCtx = srcCanvas.getContext('2d');
      srcCtx.drawImage(img, 0, 0);

      // Get pose keypoints from the live video
      let torsoPolygon = null;
      if (poseDetector && videoElement) {
        try {
          const poses = await poseDetector.estimatePoses(videoElement, { flipHorizontal: false });
          if (poses.length > 0 && poses[0].keypoints) {
            torsoPolygon = buildTorsoPolygon(poses[0].keypoints, sw, sh);
          }
        } catch (e) {
          console.warn('Pose crop failed:', e);
        }
      }

      // ── Stage 1: Combination Mask + Skin Removal ──
      // Process pixel-by-pixel: keep only if inside torso polygon AND not skin
      const imageData = srcCtx.getImageData(0, 0, sw, sh);
      const data = imageData.data;

      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const idx = (y * sw + x) * 4;
          const alpha = data[idx + 3];

          // Skip already-transparent pixels (removed by bg segmentation)
          if (alpha < 30) continue;

          // Check 1: Is pixel inside torso polygon?
          if (torsoPolygon && !isInsidePolygon(x, y, torsoPolygon)) {
            data[idx + 3] = 0; // outside torso → remove
            continue;
          }

          // Check 2: Is pixel a skin tone? (remove exposed skin)
          const r = data[idx], g = data[idx + 1], b = data[idx + 2];
          if (isSkinPixel(r, g, b)) {
            // Don't aggressively remove inside the core torso — only near edges
            // (skin near neckline & sleeves is the target, not the whole shirt)
            if (torsoPolygon) {
              // Check if pixel is near the polygon edge (within 15% of dimensions)
              const edgeDist = getEdgeDistance(x, y, torsoPolygon);
              const threshold = Math.min(sw, sh) * 0.15;
              if (edgeDist < threshold) {
                data[idx + 3] = 0; // skin near edge → remove
              }
              // else: skin-colored pixel deep inside → probably shirt color, keep
            }
          }
        }
      }

      srcCtx.putImageData(imageData, 0, 0);

      // ── Stage 2: Edge Smoothing ──
      // Apply slight blur for smoother edges
      const smoothCanvas = document.createElement('canvas');
      smoothCanvas.width = sw;
      smoothCanvas.height = sh;
      const smoothCtx = smoothCanvas.getContext('2d');
      smoothCtx.imageSmoothingEnabled = true;
      smoothCtx.imageSmoothingQuality = 'high';
      smoothCtx.filter = 'blur(1.5px)';
      smoothCtx.drawImage(srcCanvas, 0, 0);
      smoothCtx.filter = 'none';

      // Alpha feathering: sharpen interior, soften border
      const smoothData = smoothCtx.getImageData(0, 0, sw, sh);
      const sd = smoothData.data;
      for (let i = 3; i < sd.length; i += 4) {
        if (sd[i] > 200) sd[i] = 255;        // solid interior → fully opaque
        else if (sd[i] > 30) sd[i] = Math.round(sd[i] * 0.8); // edge → feathered
        else sd[i] = 0;                        // near-transparent → remove
      }
      smoothCtx.putImageData(smoothData, 0, 0);

      // ── Stage 3: Tight Alpha Bounding Box Trim ──
      const bounds = getAlphaBounds(smoothCtx, sw, sh);
      if (!bounds || bounds.w < 10 || bounds.h < 10) {
        resolve(dataUrl); // fallback to original if nothing visible
        return;
      }

      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = bounds.w;
      finalCanvas.height = bounds.h;
      const finalCtx = finalCanvas.getContext('2d');
      finalCtx.imageSmoothingEnabled = true;
      finalCtx.imageSmoothingQuality = 'high';
      finalCtx.drawImage(smoothCanvas, bounds.x, bounds.y, bounds.w, bounds.h, 0, 0, bounds.w, bounds.h);
      resolve(finalCanvas.toDataURL('image/png'));
    };
    img.src = dataUrl;
  });
}

// ── Minimum distance from point to any polygon edge ──
function getEdgeDistance(px, py, polygon) {
  let minDist = Infinity;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [x1, y1] = polygon[j];
    const [x2, y2] = polygon[i];
    // Point-to-line-segment distance
    const dx = x2 - x1, dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    let t = lenSq > 0 ? ((px - x1) * dx + (py - y1) * dy) / lenSq : 0;
    t = Math.max(0, Math.min(1, t));
    const cx = x1 + t * dx, cy = y1 + t * dy;
    const dist = Math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
    if (dist < minDist) minDist = dist;
  }
  return minDist;
}

// ── Tight alpha bounding box ──
function getAlphaBounds(ctx, w, h) {
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  let minX = w, minY = h, maxX = 0, maxY = 0;
  let found = false;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      if (data[idx + 3] > 30) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        found = true;
      }
    }
  }
  if (!found) return null;
  const pad = 2; // tighter padding (was 4)
  return {
    x: Math.max(0, minX - pad),
    y: Math.max(0, minY - pad),
    w: Math.min(w - 1, maxX + pad) - Math.max(0, minX - pad) + 1,
    h: Math.min(h - 1, maxY + pad) - Math.max(0, minY - pad) + 1
  };
}

// ═══════════════════════════════════════════
// POSE DETECTION
// ═══════════════════════════════════════════

async function getPose(videoElement) {
  if (!poseDetector) return null;
  try {
    const poses = await poseDetector.estimatePoses(videoElement, { flipHorizontal: false });
    if (poses.length > 0 && poses[0].keypoints) {
      return poses[0].keypoints;
    }
    return null;
  } catch (error) {
    return null;
  }
}

// ═══════════════════════════════════════════
// ROBUST FRONT / BACK ORIENTATION DETECTION
// ═══════════════════════════════════════════
//
// Multi-signal scoring system (0 = definitely back, 1 = definitely front):
//
//   Signal 1 — Face Visibility (weight 40%)
//       nose + eyes + ears confidence
//
//   Signal 2 — Nose Geometry (weight 25%)
//       Is the nose vertically ABOVE the shoulder midpoint
//       and horizontally CENTERED between shoulders?
//       (Only possible when facing camera)
//
//   Signal 3 — Shoulder Width Ratio (weight 20%)
//       When facing front, both shoulders are equally visible
//       and the apparent width is consistent.
//       When turning, one shoulder occludes → width shrinks.
//
//   Signal 4 — Torso Coherence (weight 15%)
//       Do hips align below shoulders in a natural front-facing
//       geometry? Check hip midpoint is below shoulder midpoint.
//
// Temporal Stability:
//   - Raw score smoothed with EMA (α = 0.15)
//   - State switch requires score to cross hysteresis thresholds
//     for at least DEBOUNCE_FRAMES consecutive frames (~300ms)
//   - Prevents flickering during rotation transitions
//
// Returns: 'front' | 'back'

const orientationState = {
  current: 'front',          // locked state: 'front' or 'back'
  rawScore: 0.8,             // smoothed raw score (0→back, 1→front)
  switchCounter: 0,          // frames the score has been past the threshold
  recentShoulderWidths: [],  // last N shoulder widths for ratio tracking
  lastNosePos: null,         // previous nose position for stability
};

// Tuning constants
const ORI_EMA_ALPHA = 0.15;        // smoothing for raw score (lower = more stable)
const ORI_FRONT_THRESHOLD = 0.55;  // score must exceed this to switch TO front
const ORI_BACK_THRESHOLD = 0.40;   // score must drop below this to switch TO back
const ORI_DEBOUNCE_FRAMES = 10;    // ~300ms at 30fps — must be stable this long
const ORI_SHOULDER_HISTORY = 15;   // frames of shoulder width history

function detectOrientation(keypoints) {
  if (!keypoints) return orientationState.current;

  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  const nose     = kp['nose'];
  const leftEye  = kp['left_eye'];
  const rightEye = kp['right_eye'];
  const leftEar  = kp['left_ear'];
  const rightEar = kp['right_ear'];
  const ls       = kp['left_shoulder'];
  const rs       = kp['right_shoulder'];
  const lh       = kp['left_hip'];
  const rh       = kp['right_hip'];

  // Need at least shoulders to compute anything
  if (!ls || !rs || ls.score < 0.2 || rs.score < 0.2) {
    return orientationState.current;
  }

  // ── Signal 1: Face Visibility (0 → 1) ──
  // Weighted combination of nose, eyes, ears confidence
  const noseConf = nose ? nose.score : 0;
  const lEyeConf = leftEye ? leftEye.score : 0;
  const rEyeConf = rightEye ? rightEye.score : 0;
  const lEarConf = leftEar ? leftEar.score : 0;
  const rEarConf = rightEar ? rightEar.score : 0;

  const eyeAvg = (lEyeConf + rEyeConf) / 2;
  const earAvg = (lEarConf + rEarConf) / 2;

  // Nose is the strongest indicator, eyes second, ears third
  const faceSignal = Math.min(1, noseConf * 0.50 + eyeAvg * 0.35 + earAvg * 0.15);

  // ── Signal 2: Nose Geometry (0 → 1) ──
  // If nose exists and is above + centered between shoulders → front
  let noseGeoSignal = 0;
  if (nose && nose.score > 0.25) {
    const shoulderMidX = (ls.x + rs.x) / 2;
    const shoulderMidY = (ls.y + rs.y) / 2;
    const shoulderW = Math.abs(rs.x - ls.x);

    if (shoulderW > 10) {
      // How centered is nose relative to shoulder midpoint? (0=far, 1=centered)
      const xOffset = Math.abs(nose.x - shoulderMidX) / shoulderW;
      const centered = Math.max(0, 1 - xOffset * 2); // 1 if x matches, 0 if far

      // Is nose above shoulders? (front-facing people have nose above shoulders)
      const aboveness = shoulderMidY - nose.y; // positive = nose is above
      const aboveScore = aboveness > 0 ? Math.min(1, aboveness / (shoulderW * 0.5)) : 0;

      noseGeoSignal = centered * 0.6 + aboveScore * 0.4;
    }
  }

  // ── Signal 3: Shoulder Width Consistency (0 → 1) ──
  // When facing front/back directly, shoulder width is at maximum.
  // When turning sideways, apparent shoulder width shrinks.
  // We compare current width to recent average.
  const currentShoulderW = Math.abs(rs.x - ls.x);
  let shoulderRatioSignal = 0.5; // neutral default

  orientationState.recentShoulderWidths.push(currentShoulderW);
  if (orientationState.recentShoulderWidths.length > ORI_SHOULDER_HISTORY) {
    orientationState.recentShoulderWidths.shift();
  }

  if (orientationState.recentShoulderWidths.length >= 5) {
    // Max width seen recently ≈ the "full frontal/back" width
    const maxW = Math.max(...orientationState.recentShoulderWidths);
    if (maxW > 20) {
      const ratio = currentShoulderW / maxW; // 1.0 = full width, <0.7 = turning
      // Full width (>0.85) suggests front OR back (can't distinguish alone)
      // But combined with face signals, narrows it down
      // Very narrow (<0.5) suggests profile/turning = unstable, don't switch
      shoulderRatioSignal = ratio > 0.7 ? 0.5 + (ratio - 0.7) * 1.67 : ratio;
      shoulderRatioSignal = Math.max(0, Math.min(1, shoulderRatioSignal));
    }
  }

  // ── Signal 4: Torso Coherence (0 → 1) ──
  // When facing front, hips should be below shoulders in a natural V-shape.
  // High confidence in all 4 torso points = coherent pose = likely front.
  let torsoSignal = 0.5; // neutral
  if (lh && rh && lh.score > 0.2 && rh.score > 0.2) {
    const hipMidY = (lh.y + rh.y) / 2;
    const shoulderMidY = (ls.y + rs.y) / 2;
    const torsoHeight = hipMidY - shoulderMidY;

    // Natural standing pose: torso has positive height, all 4 points confident
    if (torsoHeight > 0) {
      const avgConf = (ls.score + rs.score + lh.score + rh.score) / 4;
      torsoSignal = Math.min(1, avgConf * 1.2);
    }
  }

  // ── Combine signals with weights ──
  const rawFrontScore =
    faceSignal         * 0.40 +
    noseGeoSignal      * 0.25 +
    shoulderRatioSignal * 0.20 +
    torsoSignal        * 0.15;

  // ── EMA Smoothing ──
  orientationState.rawScore =
    orientationState.rawScore * (1 - ORI_EMA_ALPHA) +
    rawFrontScore * ORI_EMA_ALPHA;

  const smoothedScore = orientationState.rawScore;

  // ── Hysteresis + Debounce State Machine ──
  // Different thresholds for switching TO front vs TO back prevent
  // rapid toggling when the score hovers near a single threshold.
  const wantsFront = smoothedScore > ORI_FRONT_THRESHOLD;
  const wantsBack  = smoothedScore < ORI_BACK_THRESHOLD;

  if (orientationState.current === 'front' && wantsBack) {
    orientationState.switchCounter++;
    if (orientationState.switchCounter >= ORI_DEBOUNCE_FRAMES) {
      orientationState.current = 'back';
      orientationState.switchCounter = 0;
    }
  } else if (orientationState.current === 'back' && wantsFront) {
    orientationState.switchCounter++;
    if (orientationState.switchCounter >= ORI_DEBOUNCE_FRAMES) {
      orientationState.current = 'front';
      orientationState.switchCounter = 0;
    }
  } else {
    // Score is in the dead zone or agrees with current state → reset counter
    orientationState.switchCounter = Math.max(0, orientationState.switchCounter - 2);
  }

  return orientationState.current;
}

// Expose the smooth score for crossfade transitions (0=back, 1=front)
function getOrientationConfidence() {
  return orientationState.rawScore;
}

// ═══════════════════════════════════════════
// AR OVERLAY — Fixed Angle + Front/Back Support
// ═══════════════════════════════════════════
//
// THE UPSIDE-DOWN BUG:
//
// MoveNet labels keypoints by the PERSON's anatomy:
//   left_shoulder = person's physical left
//   right_shoulder = person's physical right
//
// In a FRONT-facing camera (selfie):
//   Person's right shoulder → LEFT side of frame (small x)
//   Person's left shoulder → RIGHT side of frame (large x)
//   So rsx < lsx → Math.atan2(~0, negative) = π (180°)
//   ctx.rotate(π) → UPSIDE DOWN!
//
// FIX: Always compute tilt angle from frame-left to frame-right
// shoulder, regardless of which is anatomically left/right.
// This produces a small tilt angle (near 0°), never 180°.

let framesWithoutPose = 0;
const MAX_FADE_FRAMES = 15;

function drawAROverlay(ctx, canvasWidth, canvasHeight, garmentImg, keypoints, videoElement, backGarmentImg) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  if (!garmentImg) return;

  // No keypoints: graceful fade
  if (!keypoints) {
    framesWithoutPose++;
    if (framesWithoutPose > MAX_FADE_FRAMES || smoothing.x === null) return;
    const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
    drawGarmentAtSmoothedPosition(ctx, garmentImg, fadeAlpha);
    return;
  }

  framesWithoutPose = 0;

  // Extract keypoints
  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  const ls = kp['left_shoulder'];
  const rs = kp['right_shoulder'];
  const lh = kp['left_hip'];
  const rh = kp['right_hip'];

  if (!ls || !rs) return;

  // Confidence check
  const shoulderConf = Math.min(ls.score || 0, rs.score || 0);
  if (shoulderConf < 0.15) {
    framesWithoutPose++;
    if (smoothing.x !== null) {
      const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
      drawGarmentAtSmoothedPosition(ctx, garmentImg, fadeAlpha);
    }
    return;
  }
  const confAlpha = 0.3 + Math.min(1, (shoulderConf - 0.15) / 0.5) * 0.58;

  // ── Detect orientation (front or back) ──
  const orientation = detectOrientation(keypoints);

  // Choose which garment image to display
  let activeGarment = garmentImg; // default: front
  if (orientation === 'back' && backGarmentImg) {
    activeGarment = backGarmentImg;
  }

  // Scale keypoints: video → canvas
  const videoW = videoElement ? (videoElement.videoWidth || canvasWidth) : canvasWidth;
  const videoH = videoElement ? (videoElement.videoHeight || canvasHeight) : canvasHeight;
  const scaleX = canvasWidth / videoW;
  const scaleY = canvasHeight / videoH;

  const lsx = ls.x * scaleX, lsy = ls.y * scaleY;
  const rsx = rs.x * scaleX, rsy = rs.y * scaleY;

  // Shoulder measurements
  const shoulderWidth = Math.hypot(rsx - lsx, rsy - lsy);
  if (shoulderWidth < 20) return;

  // Garment dimensions
  const rawGarmentWidth = shoulderWidth * 1.8;
  let rawGarmentHeight;
  if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
    const lhy = lh.y * scaleY;
    const rhy = rh.y * scaleY;
    rawGarmentHeight = ((lhy + rhy) / 2 - (lsy + rsy) / 2) * 1.35;
  } else {
    rawGarmentHeight = rawGarmentWidth * 1.3;
  }
  rawGarmentHeight = Math.max(rawGarmentHeight, 60);

  // ════════════════════════════════════════
  // ANGLE FIX: Always compute from frame-left to frame-right
  //
  // This prevents the 180° bug. We don't care which shoulder
  // is anatomically "left" or "right" — we only care about
  // the tilt direction in screen space.
  // ════════════════════════════════════════
  let leftPt, rightPt;
  if (lsx < rsx) {
    leftPt = { x: lsx, y: lsy };
    rightPt = { x: rsx, y: rsy };
  } else {
    leftPt = { x: rsx, y: rsy };
    rightPt = { x: lsx, y: lsy };
  }
  const rawAngle = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x);

  // Anchor: shoulder midpoint. Garment hangs DOWN from shoulders.
  const shoulderMidX = (lsx + rsx) / 2;
  const shoulderMidY = (lsy + rsy) / 2;
  const rawCenterX = shoulderMidX;
  const rawCenterY = shoulderMidY + rawGarmentHeight * 0.45;

  // Smoothing
  const cx = smooth('x', rawCenterX);
  const cy = smooth('y', rawCenterY);
  const gw = smooth('w', rawGarmentWidth);
  const gh = smooth('h', rawGarmentHeight);
  const angle = smooth('angle', rawAngle);

  // Draw shadow
  ctx.save();
  ctx.translate(cx + 3, cy + 6);
  ctx.rotate(angle);
  ctx.globalAlpha = 0.1;
  ctx.filter = 'blur(10px)';
  ctx.drawImage(activeGarment, -gw / 2, -gh / 2, gw, gh);
  ctx.filter = 'none';
  ctx.restore();

  // Draw garment
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.globalAlpha = confAlpha;
  ctx.drawImage(activeGarment, -gw / 2, -gh / 2, gw, gh);
  ctx.restore();

  // Draw orientation indicator (small badge)
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.fillStyle = orientation === 'front' ? '#4CAF50' : '#FF9800';
  ctx.font = '10px DM Sans, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(orientation === 'front' ? '👤 Front' : '🔙 Back', 8, canvasHeight - 8);
  ctx.restore();
}

function drawGarmentAtSmoothedPosition(ctx, garmentImg, alpha) {
  if (smoothing.x === null || alpha <= 0) return;
  ctx.save();
  ctx.translate(smoothing.x, smoothing.y);
  if (smoothing.angle !== null) ctx.rotate(smoothing.angle);
  ctx.globalAlpha = alpha;
  const gw = smoothing.w || 150;
  const gh = smoothing.h || 200;
  ctx.drawImage(garmentImg, -gw / 2, -gh / 2, gw, gh);
  ctx.restore();
}

// ═══════════════════════════════════════════
// FACE DETECTION
// ═══════════════════════════════════════════

async function getFace(videoElement) {
  if (!faceDetector) return null;
  try {
    const faces = await faceDetector.estimateFaces(videoElement, { flipHorizontal: false });
    if (faces.length > 0) return faces[0].box;
    return null;
  } catch (error) {
    return null;
  }
}

// ═══════════════════════════════════════════
// MANNEQUIN FACE SWAP
// ═══════════════════════════════════════════

function drawMannequinFaceSwap(canvas, garmentImg, videoElement, faceBox) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);

  ctx.save();
  ctx.fillStyle = '#2a2a3a';
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.12, 35, 42, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(w / 2 - 12, h * 0.16, 24, 25);
  ctx.beginPath();
  ctx.moveTo(w / 2 - 65, h * 0.21);
  ctx.lineTo(w / 2 + 65, h * 0.21);
  ctx.lineTo(w / 2 + 55, h * 0.6);
  ctx.lineTo(w / 2 - 55, h * 0.6);
  ctx.closePath();
  ctx.fill();
  ctx.fillRect(w / 2 - 45, h * 0.6, 35, h * 0.38);
  ctx.fillRect(w / 2 + 10, h * 0.6, 35, h * 0.38);
  ctx.restore();

  if (garmentImg) {
    const gw = 140;
    const gh = gw * (garmentImg.height / garmentImg.width);
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.drawImage(garmentImg, w / 2 - gw / 2, h * 0.2, gw, Math.min(gh, h * 0.42));
    ctx.restore();
  }

  if (faceBox && videoElement) {
    try {
      const faceCanvas = document.createElement('canvas');
      const faceSize = Math.max(faceBox.width, faceBox.height) * 1.2;
      faceCanvas.width = faceSize;
      faceCanvas.height = faceSize;
      const fCtx = faceCanvas.getContext('2d');
      fCtx.drawImage(
        videoElement,
        faceBox.xMin - faceBox.width * 0.1,
        faceBox.yMin - faceBox.height * 0.1,
        faceSize, faceSize,
        0, 0, faceSize, faceSize
      );
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(w / 2, h * 0.12, 33, 40, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(faceCanvas, w / 2 - 40, h * 0.12 - 45, 80, 90);
      ctx.restore();
    } catch (e) { /* skip */ }
  }

  ctx.fillStyle = '#666680';
  ctx.font = '11px DM Sans, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Mannequin Preview', w / 2, h - 10);
}

function isModelLoaded(modelName) {
  switch (modelName) {
    case 'segmenter': return !!segmenter;
    case 'pose': return !!poseDetector;
    case 'face': return !!faceDetector;
    default: return modelsLoaded;
  }
}

// Expose globally
window.FVAI = {
  loadModels, removeBackground, autoCropGarment, getPose,
  drawAROverlay, detectOrientation, getOrientationConfidence,
  getFace, drawMannequinFaceSwap,
  isModelLoaded, resetSmoothing
};
