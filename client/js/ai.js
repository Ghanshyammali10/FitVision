// ==========================================
// FitVision — AI Module (v6 — Pure MediaPipe)
// MediaPipe Vision: Image Segmenter (multiclass) + Pose Landmarker
// NO TensorFlow.js dependency!
// ==========================================

let imageSegmenter = null;
let poseLandmarker = null;
let modelsLoaded = false;
let lastVideoTime = -1;
let lastPoseVideoTime = -1;

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
  // Also reset the AR overlay EMA state
  if (typeof drawAROverlay !== 'undefined') {
    drawAROverlay._state = null;
  }
}

// ═══════════════════════════════════════════
// MODEL LOADING — Pure MediaPipe Vision
// ═══════════════════════════════════════════

async function loadModels(onProgress) {
  const report = (model, status, percent) => {
    if (onProgress) onProgress({ model, status, percent });
  };

  try {
    report('MediaPipe Vision', 'Initializing WASM...', 5);

    // Load the vision WASM module
    const { FilesetResolver, ImageSegmenter, PoseLandmarker } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
    );

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
    );

    report('Image Segmenter', 'Loading multiclass model...', 15);

    // ── Image Segmenter (Multiclass Selfie) ──
    // Categories: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
    try {
      imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
          delegate: 'GPU'
        },
        outputCategoryMask: true,
        outputConfidenceMasks: false,
        runningMode: 'VIDEO'
      });
      report('Image Segmenter', 'Ready ✓ (clothes detection)', 45);
    } catch (e) {
      console.warn('Image Segmenter failed:', e.message);
      report('Image Segmenter', 'Failed — using fallback', 45);
    }

    report('Pose Landmarker', 'Loading model...', 50);

    // ── Pose Landmarker ──
    // 33 landmarks with normalized coordinates (0-1)
    try {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      report('Pose Landmarker', 'Ready ✓ (33 landmarks)', 90);
    } catch (e) {
      console.warn('Pose Landmarker failed:', e.message);
      report('Pose Landmarker', 'Failed', 90);
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
// BACKGROUND REMOVAL — using Multiclass Segmenter
// ═══════════════════════════════════════════
// Keeps ONLY category 4 (clothes) + a person silhouette.
// This is used for the initial capture — it removes the background
// but keeps the person visible for the auto-crop step.

async function removeBackground(videoElement) {
  if (!imageSegmenter) return fallbackCapture(videoElement);

  try {
    const canvas = document.createElement('canvas');
    const w = videoElement.videoWidth || 640;
    const h = videoElement.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Draw the original frame
    ctx.drawImage(videoElement, 0, 0, w, h);
    const frame = ctx.getImageData(0, 0, w, h);

    // Get category mask from the segmenter
    const nowMs = performance.now();
    let categoryMask = null;

    imageSegmenter.segmentForVideo(videoElement, nowMs, (result) => {
      if (result.categoryMask) {
        categoryMask = result.categoryMask.getAsUint8Array();
      }
    });

    if (!categoryMask) return fallbackCapture(videoElement);

    // Remove background (category 0) — keep person (categories 1-5)
    for (let i = 0; i < categoryMask.length; i++) {
      const category = categoryMask[i];
      if (category === 0) { // background
        frame.data[i * 4 + 3] = 0; // make transparent
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
// GARMENT EXTRACTION — Clothes-Only Cutout
// ═══════════════════════════════════════════
//
// This is the KEY improvement: MediaPipe multiclass segmenter
// outputs category 4 = clothes for every pixel.
// We simply keep ONLY those pixels. Perfect garment cutout!
//
// No more:
//  - Skin detection heuristics (RGB/HSV/YCbCr)
//  - Polygon clipping
//  - Island removal
//  - Morphological erosion
//
// Just: "Is this pixel clothes? Yes → keep. No → transparent."

async function autoCropGarment(dataUrl, videoElement) {
  return new Promise(async (resolve) => {
    // If segmenter is available, extract clothes directly from video
    if (imageSegmenter && videoElement) {
      try {
        const w = videoElement.videoWidth || 640;
        const h = videoElement.videoHeight || 480;

        // ── Step 1: Get the main subject's bounding box via Pose ──
        // This ensures we only extract the CLOSEST person's garment
        let subjectBox = null;
        if (poseLandmarker) {
          try {
            const poseNow = performance.now();
            const poseResult = poseLandmarker.detectForVideo(videoElement, poseNow);
            if (poseResult.landmarks && poseResult.landmarks.length > 0) {
              const lm = poseResult.landmarks[0]; // first (largest) pose
              // Build bounding box from shoulders + hips with generous margin
              const ls = lm[11]; // left_shoulder
              const rs = lm[12]; // right_shoulder
              const lh = lm[23]; // left_hip
              const rh = lm[24]; // right_hip

              if (ls && rs) {
                const shoulderW = Math.abs(rs.x - ls.x) * w;
                const margin = shoulderW * 0.6; // generous margin for sleeves

                const allX = [ls.x * w, rs.x * w];
                const allY = [ls.y * h, rs.y * h];
                if (lh && rh) {
                  allX.push(lh.x * w, rh.x * w);
                  allY.push(lh.y * h, rh.y * h);
                }

                subjectBox = {
                  left:   Math.max(0, Math.min(...allX) - margin),
                  right:  Math.min(w, Math.max(...allX) + margin),
                  top:    Math.max(0, Math.min(...allY) - margin * 0.5),
                  bottom: Math.min(h, Math.max(...allY) + margin * 0.4)
                };
              }
            }
          } catch (e) {
            console.warn('Pose for subject isolation failed:', e);
          }
        }

        // ── Step 2: Get the category mask ──
        const nowMs = performance.now();
        let categoryMask = null;

        imageSegmenter.segmentForVideo(videoElement, nowMs, (result) => {
          if (result.categoryMask) {
            categoryMask = result.categoryMask.getAsUint8Array();
          }
        });

        if (categoryMask) {
          // Draw the video frame onto a canvas
          const srcCanvas = document.createElement('canvas');
          srcCanvas.width = w;
          srcCanvas.height = h;
          const srcCtx = srcCanvas.getContext('2d');
          srcCtx.drawImage(videoElement, 0, 0, w, h);

          const imageData = srcCtx.getImageData(0, 0, w, h);
          const data = imageData.data;

          // ── Keep ONLY clothes (category 4) within the main subject's region ──
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const i = y * w + x;
              const category = categoryMask[i];

              if (category !== 4) {
                // Not clothes → transparent
                data[i * 4 + 3] = 0;
              } else if (subjectBox) {
                // Clothes but outside main subject's bounding box → transparent
                if (x < subjectBox.left || x > subjectBox.right ||
                    y < subjectBox.top || y > subjectBox.bottom) {
                  data[i * 4 + 3] = 0;
                }
              }
            }
          }

          srcCtx.putImageData(imageData, 0, 0);

          // ── Step 3: Morphological hole-filling ──
          // Dark/white fabric often gets misclassified. If a transparent pixel
          // is surrounded by mostly opaque neighbors → fill it back in.
          const holeData = srcCtx.getImageData(0, 0, w, h);
          const hd = holeData.data;
          const filled = new Uint8Array(w * h); // track what we fill

          for (let pass = 0; pass < 3; pass++) { // 3 passes for larger holes
            for (let y = 1; y < h - 1; y++) {
              for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                if (hd[idx * 4 + 3] > 30) continue; // already opaque

                // Count opaque neighbors in 3x3 window
                let opaqueCount = 0;
                for (let dy = -1; dy <= 1; dy++) {
                  for (let dx = -1; dx <= 1; dx++) {
                    if (dx === 0 && dy === 0) continue;
                    const ni = (y + dy) * w + (x + dx);
                    if (hd[ni * 4 + 3] > 30) opaqueCount++;
                  }
                }

                // If ≥5 of 8 neighbors are opaque, this is a hole → fill
                if (opaqueCount >= 5) {
                  hd[idx * 4 + 3] = 255;
                  filled[idx] = 1;
                }
              }
            }
          }

          // Also include person-category pixels (skin/hair) that are INSIDE
          // the clothes region — they're likely misclassified fabric
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const i = y * w + x;
              const cat = categoryMask[i];
              // body-skin (2) or hair (1) within subject box = could be dark/light fabric
              if ((cat === 1 || cat === 2) && hd[i * 4 + 3] === 0) {
                if (subjectBox && x >= subjectBox.left && x <= subjectBox.right &&
                    y >= subjectBox.top && y <= subjectBox.bottom) {
                  // Check if this pixel is surrounded by clothes pixels
                  let clothesNeighbors = 0;
                  for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                      if (dx === 0 && dy === 0) continue;
                      const ny = y + dy, nx = x + dx;
                      if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                        const ni = ny * w + nx;
                        if (hd[ni * 4 + 3] > 30) clothesNeighbors++;
                      }
                    }
                  }
                  // If mostly surrounded by clothes → this is misclassified fabric
                  if (clothesNeighbors >= 14) {
                    hd[i * 4 + 3] = 255;
                  }
                }
              }
            }
          }

          srcCtx.putImageData(holeData, 0, 0);

          // ── Step 4: Keep only the largest connected clothes region ──
          // This removes any small stray patches from background people
          keepLargestRegion(srcCtx, w, h);

          // ── Anti-alias edges (gentle) ──
          const smoothCanvas = document.createElement('canvas');
          smoothCanvas.width = w;
          smoothCanvas.height = h;
          const smoothCtx = smoothCanvas.getContext('2d');
          smoothCtx.imageSmoothingEnabled = true;
          smoothCtx.imageSmoothingQuality = 'high';
          smoothCtx.filter = 'blur(0.5px)';
          smoothCtx.drawImage(srcCanvas, 0, 0);
          smoothCtx.filter = 'none';

          // Alpha cleanup: softer thresholds to preserve fabric
          const smoothData = smoothCtx.getImageData(0, 0, w, h);
          const sd = smoothData.data;
          for (let i = 3; i < sd.length; i += 4) {
            if (sd[i] > 100) sd[i] = 255;        // solid → fully opaque
            else if (sd[i] > 15) sd[i] = sd[i];  // edge → keep feather
            else sd[i] = 0;                       // noise → remove
          }
          smoothCtx.putImageData(smoothData, 0, 0);

          // ── Tight bounding box crop ──
          const bounds = getAlphaBounds(smoothCtx, w, h);
          if (bounds && bounds.w > 10 && bounds.h > 10) {
            const finalCanvas = document.createElement('canvas');
            finalCanvas.width = bounds.w;
            finalCanvas.height = bounds.h;
            const finalCtx = finalCanvas.getContext('2d');
            finalCtx.imageSmoothingEnabled = true;
            finalCtx.imageSmoothingQuality = 'high';
            finalCtx.drawImage(smoothCanvas, bounds.x, bounds.y, bounds.w, bounds.h, 0, 0, bounds.w, bounds.h);
            resolve(finalCanvas.toDataURL('image/png'));
            return;
          }
        }
      } catch (e) {
        console.warn('MediaPipe garment extraction failed:', e);
      }
    }

    // Fallback: return the input as-is
    resolve(dataUrl);
  });
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
  const pad = 4;
  return {
    x: Math.max(0, minX - pad),
    y: Math.max(0, minY - pad),
    w: Math.min(w - 1, maxX + pad) - Math.max(0, minX - pad) + 1,
    h: Math.min(h - 1, maxY + pad) - Math.max(0, minY - pad) + 1
  };
}

// ── Keep only the largest connected opaque region ──
// BFS flood-fill to label connected components, then remove all except the biggest
function keepLargestRegion(ctx, w, h) {
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const visited = new Uint8Array(w * h);
  const labels = new Int32Array(w * h).fill(-1);
  let labelId = 0;
  const regionSizes = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const pos = y * w + x;
      if (visited[pos] || data[pos * 4 + 3] < 30) continue;

      const queue = [pos];
      visited[pos] = 1;
      labels[pos] = labelId;
      let count = 0;

      while (queue.length > 0) {
        const cur = queue.shift();
        count++;
        const cx = cur % w, cy = (cur - cx) / w;

        const neighbors = [
          cy > 0 ? cur - w : -1,
          cy < h - 1 ? cur + w : -1,
          cx > 0 ? cur - 1 : -1,
          cx < w - 1 ? cur + 1 : -1
        ];
        for (const n of neighbors) {
          if (n >= 0 && !visited[n] && data[n * 4 + 3] >= 30) {
            visited[n] = 1;
            labels[n] = labelId;
            queue.push(n);
          }
        }
      }
      regionSizes.push(count);
      labelId++;
    }
  }

  if (regionSizes.length === 0) return;

  // Find the largest region
  let maxSize = 0, maxLabel = 0;
  for (let i = 0; i < regionSizes.length; i++) {
    if (regionSizes[i] > maxSize) {
      maxSize = regionSizes[i];
      maxLabel = i;
    }
  }

  // Remove all regions except the largest
  for (let i = 0; i < w * h; i++) {
    if (data[i * 4 + 3] >= 30 && labels[i] !== maxLabel) {
      data[i * 4 + 3] = 0;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// ═══════════════════════════════════════════
// POSE DETECTION — MediaPipe Pose Landmarker
// ═══════════════════════════════════════════
//
// Returns 33 landmarks in NORMALIZED coordinates (0-1).
// We convert them to match the old MoveNet keypoint format
// so the AR overlay code doesn't need to change.
//
// MediaPipe Pose landmark indices:
//  0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer,
//  4: right_eye_inner, 5: right_eye, 6: right_eye_outer,
//  7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right,
//  11: left_shoulder, 12: right_shoulder,
//  13: left_elbow, 14: right_elbow,
//  15: left_wrist, 16: right_wrist,
//  23: left_hip, 24: right_hip,
//  25: left_knee, 26: right_knee,
//  27: left_ankle, 28: right_ankle

const LANDMARK_NAMES = [
  'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
  'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee',
  'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index'
];

async function getPose(videoElement) {
  if (!poseLandmarker) return null;
  try {
    const nowMs = performance.now();
    const result = poseLandmarker.detectForVideo(videoElement, nowMs);

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      const w = videoElement.videoWidth || 640;
      const h = videoElement.videoHeight || 480;

      // Convert to MoveNet-compatible format: { x, y, score, name }
      // MediaPipe returns normalized [0,1], we convert to pixel coords
      return landmarks.map((lm, idx) => ({
        x: lm.x * w,
        y: lm.y * h,
        score: lm.visibility !== undefined ? lm.visibility : 0.9,
        name: LANDMARK_NAMES[idx] || `landmark_${idx}`
      }));
    }
    return null;
  } catch (error) {
    return null;
  }
}

// ═══════════════════════════════════════════
// ROBUST FRONT / BACK ORIENTATION DETECTION
// ═══════════════════════════════════════════

const orientationState = {
  current: 'front',
  rawScore: 0.8,
  switchCounter: 0,
  recentShoulderWidths: [],
  lastNosePos: null,
};

const ORI_EMA_ALPHA = 0.15;
const ORI_FRONT_THRESHOLD = 0.55;
const ORI_BACK_THRESHOLD = 0.40;
const ORI_DEBOUNCE_FRAMES = 10;
const ORI_SHOULDER_HISTORY = 15;

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

  if (!ls || !rs || ls.score < 0.2 || rs.score < 0.2) {
    return orientationState.current;
  }

  // Signal 1: Face Visibility
  const noseConf = nose ? nose.score : 0;
  const lEyeConf = leftEye ? leftEye.score : 0;
  const rEyeConf = rightEye ? rightEye.score : 0;
  const lEarConf = leftEar ? leftEar.score : 0;
  const rEarConf = rightEar ? rightEar.score : 0;
  const eyeAvg = (lEyeConf + rEyeConf) / 2;
  const earAvg = (lEarConf + rEarConf) / 2;
  const faceSignal = Math.min(1, noseConf * 0.50 + eyeAvg * 0.35 + earAvg * 0.15);

  // Signal 2: Nose Geometry
  let noseGeoSignal = 0;
  if (nose && nose.score > 0.25) {
    const shoulderMidX = (ls.x + rs.x) / 2;
    const shoulderMidY = (ls.y + rs.y) / 2;
    const shoulderW = Math.abs(rs.x - ls.x);
    if (shoulderW > 10) {
      const xOffset = Math.abs(nose.x - shoulderMidX) / shoulderW;
      const centered = Math.max(0, 1 - xOffset * 2);
      const aboveness = shoulderMidY - nose.y;
      const aboveScore = aboveness > 0 ? Math.min(1, aboveness / (shoulderW * 0.5)) : 0;
      noseGeoSignal = centered * 0.6 + aboveScore * 0.4;
    }
  }

  // Signal 3: Shoulder Width Ratio
  const currentShoulderW = Math.abs(rs.x - ls.x);
  let shoulderRatioSignal = 0.5;
  orientationState.recentShoulderWidths.push(currentShoulderW);
  if (orientationState.recentShoulderWidths.length > ORI_SHOULDER_HISTORY) {
    orientationState.recentShoulderWidths.shift();
  }
  if (orientationState.recentShoulderWidths.length >= 5) {
    const maxW = Math.max(...orientationState.recentShoulderWidths);
    if (maxW > 20) {
      const ratio = currentShoulderW / maxW;
      shoulderRatioSignal = ratio > 0.7 ? 0.5 + (ratio - 0.7) * 1.67 : ratio;
      shoulderRatioSignal = Math.max(0, Math.min(1, shoulderRatioSignal));
    }
  }

  // Signal 4: Torso Coherence
  let torsoSignal = 0.5;
  if (lh && rh && lh.score > 0.2 && rh.score > 0.2) {
    const hipMidY = (lh.y + rh.y) / 2;
    const shoulderMidY = (ls.y + rs.y) / 2;
    const torsoHeight = hipMidY - shoulderMidY;
    if (torsoHeight > 0) {
      const avgConf = (ls.score + rs.score + lh.score + rh.score) / 4;
      torsoSignal = Math.min(1, avgConf * 1.2);
    }
  }

  // Combine
  const rawFrontScore =
    faceSignal * 0.40 +
    noseGeoSignal * 0.25 +
    shoulderRatioSignal * 0.20 +
    torsoSignal * 0.15;

  // EMA Smoothing
  orientationState.rawScore =
    orientationState.rawScore * (1 - ORI_EMA_ALPHA) +
    rawFrontScore * ORI_EMA_ALPHA;

  const smoothedScore = orientationState.rawScore;

  // Hysteresis + Debounce
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
    orientationState.switchCounter = Math.max(0, orientationState.switchCounter - 2);
  }

  return orientationState.current;
}

function getOrientationConfidence() {
  return orientationState.rawScore;
}

// ═══════════════════════════════════════════
// AR OVERLAY — Multi-Landmark Body-Fitted (v6.2)
// ═══════════════════════════════════════════
//
// Uses 7 MediaPipe landmarks to precisely fit garment to each subject:
//   - Nose (0)            → collar/neckline positioning
//   - Left Shoulder (11)  → top-left anchor + width
//   - Right Shoulder (12) → top-right anchor + width
//   - Left Elbow (13)     → sleeve width constraint
//   - Right Elbow (14)    → sleeve width constraint
//   - Left Hip (23)       → torso left-side length + bottom width
//   - Right Hip (24)      → torso right-side length + bottom width
//
// Each person's unique landmark distances = unique garment dimensions.
// Depth scaling is automatic: all measurements are in pixels, which
// shrink/grow as the subject moves away from / toward the camera.

let framesWithoutPose = 0;
const MAX_FADE_FRAMES = 15;

function drawAROverlay(ctx, canvasWidth, canvasHeight, garmentImg, keypoints, videoElement, backGarmentImg) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  if (!garmentImg) return;

  // ── Fade-out when pose is lost ──
  if (!keypoints) {
    framesWithoutPose++;
    if (framesWithoutPose > MAX_FADE_FRAMES || !drawAROverlay._state) return;
    const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
    drawGarmentAtSmoothedPosition(ctx, garmentImg, fadeAlpha);
    return;
  }

  framesWithoutPose = 0;

  // ── Build keypoint map ──
  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  // Required landmarks
  const ls = kp['left_shoulder'];
  const rs = kp['right_shoulder'];
  if (!ls || !rs) return;

  // Optional landmarks (used when visible for better precision)
  const lh   = kp['left_hip'];
  const rh   = kp['right_hip'];
  const le   = kp['left_elbow'];
  const re   = kp['right_elbow'];
  const nose = kp['nose'];

  // ── Bail if shoulders not visible enough ──
  const shoulderVis = ((ls.score || 0) + (rs.score || 0)) / 2;
  if (shoulderVis < 0.5) {
    framesWithoutPose++;
    if (drawAROverlay._state) {
      const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
      drawGarmentAtSmoothedPosition(ctx, garmentImg, fadeAlpha);
    }
    return;
  }

  // Detect orientation (front / back)
  const orientation = detectOrientation(keypoints);
  let activeGarment = garmentImg;
  if (orientation === 'back' && backGarmentImg) {
    activeGarment = backGarmentImg;
  }

  // ── Convert landmarks to canvas pixel coordinates ──
  const videoW = videoElement ? (videoElement.videoWidth || canvasWidth) : canvasWidth;
  const videoH = videoElement ? (videoElement.videoHeight || canvasHeight) : canvasHeight;
  const scaleX = canvasWidth / videoW;
  const scaleY = canvasHeight / videoH;

  // Helper: scale a keypoint to canvas coords
  const S = (k) => ({ x: k.x * scaleX, y: k.y * scaleY, vis: k.score || 0 });

  const lsC = S(ls);     // left shoulder (canvas)
  const rsC = S(rs);     // right shoulder (canvas)
  const lhC = lh ? S(lh) : null;   // left hip
  const rhC = rh ? S(rh) : null;   // right hip
  const leC = le ? S(le) : null;   // left elbow
  const reC = re ? S(re) : null;   // right elbow
  const noseC = nose ? S(nose) : null; // nose

  // ════════════════════════════════════════════
  // BODY MEASUREMENTS — derived from landmarks
  // ════════════════════════════════════════════

  // 1. SHOULDER WIDTH — primary scale reference (depth proxy)
  const shoulderWidth = Math.hypot(rsC.x - lsC.x, rsC.y - lsC.y);
  if (shoulderWidth < 20) return;

  // 2. SHOULDER MIDPOINT
  const shoulderMidX = (lsC.x + rsC.x) / 2;
  const shoulderMidY = (lsC.y + rsC.y) / 2;

  // 3. HIP MEASUREMENTS (when visible)
  const hipsVis = lhC && rhC && lhC.vis > 0.4 && rhC.vis > 0.4;
  let hipMidX, hipMidY, hipWidth;
  let leftTorsoLen = 0, rightTorsoLen = 0, avgTorsoLen = 0;

  if (hipsVis) {
    hipMidX = (lhC.x + rhC.x) / 2;
    hipMidY = (lhC.y + rhC.y) / 2;
    hipWidth = Math.hypot(rhC.x - lhC.x, rhC.y - lhC.y);

    // Measure BOTH sides of the torso independently
    // This captures the actual torso length even when the body is angled
    leftTorsoLen  = Math.hypot(lhC.x - lsC.x, lhC.y - lsC.y);
    rightTorsoLen = Math.hypot(rhC.x - rsC.x, rhC.y - rsC.y);
    avgTorsoLen   = (leftTorsoLen + rightTorsoLen) / 2;
  }

  // 4. NECK/COLLAR REFERENCE (from nose landmark)
  // The nose position tells us exactly where the neckline should be.
  // Collar sits between nose and shoulders (~75% toward shoulders).
  let collarY = shoulderMidY;
  const noseVis = noseC && noseC.vis > 0.5;
  if (noseVis) {
    const noseToShoulderDist = shoulderMidY - noseC.y;
    if (noseToShoulderDist > 5) {
      // Place collar 75% of the way from nose to shoulders
      collarY = noseC.y + noseToShoulderDist * 0.75;
    }
  }

  // 5. ELBOW WIDTH (for sleeve constraint)
  const elbowsVis = leC && reC && leC.vis > 0.4 && reC.vis > 0.4;
  let elbowSpan = 0;
  if (elbowsVis) {
    elbowSpan = Math.hypot(reC.x - leC.x, reC.y - leC.y);
  }

  // ════════════════════════════════════════════
  // GARMENT SIZE — directly from body landmarks
  // ════════════════════════════════════════════
  // The garment stretches to fit the person's body, like real clothing.
  // No aspect-ratio blending. Body = truth.

  // WIDTH: shoulder width × 2.0 — real shirts extend well beyond skeleton shoulder points
  // MediaPipe "shoulder" landmarks = bony joint tips, NOT the outer edge of fabric.
  // A real t-shirt adds ~50% sleeve overhang on EACH side.
  let garmentWidth;
  if (elbowsVis && elbowSpan > shoulderWidth * 1.1) {
    // Elbows visible — blend shoulder and elbow span for maximum accuracy
    garmentWidth = Math.max(shoulderWidth * 2.0, shoulderWidth * 0.5 + elbowSpan * 0.75);
  } else {
    garmentWidth = shoulderWidth * 2.0;
  }

  // HEIGHT: from actual torso length, or estimated from shoulders
  let garmentHeight;
  if (hipsVis && avgTorsoLen > 20) {
    // Real torso length (shoulder→hip averaged) + 50% for hem below hips
    garmentHeight = avgTorsoLen * 1.50;
  } else {
    // Hips not visible: estimate height from shoulder width
    garmentHeight = shoulderWidth * 2.0;
  }

  // ════════════════════════════════════════════
  // POSITION — garment anchored to body landmarks
  // ════════════════════════════════════════════

  // Rotation: screen-left → screen-right shoulder
  let leftPt, rightPt;
  if (lsC.x < rsC.x) { leftPt = lsC; rightPt = rsC; }
  else                { leftPt = rsC; rightPt = lsC; }
  const angle = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x);

  // The garment is drawn centred on (targetX, targetY) via drawImage(-w/2, -h/2, w, h).
  //
  // ANCHOR: Use collarY (derived from nose position) as the garment top.
  // The captured garment PNG includes the neckline at the top, so we
  // need the garment's top edge to sit at the collar position, NOT at shoulders.
  //
  // top = targetY - garmentHeight/2  =  collarY
  // ∴ targetY = collarY + garmentHeight/2
  //
  // X center: midpoint between shoulders (or between shoulders+hips if both visible)
  let targetX, targetY;

  if (hipsVis) {
    // Both shoulders and hips visible — centre X between both pairs
    targetX = (shoulderMidX + hipMidX) / 2;
    // Top of garment at collar, extends downward from there
    targetY = collarY + garmentHeight / 2;
  } else {
    targetX = shoulderMidX;
    targetY = collarY + garmentHeight / 2;
  }

  // ════════════════════════════════════════════
  // ADAPTIVE EMA SMOOTHING
  // ════════════════════════════════════════════

  if (!drawAROverlay._state) {
    drawAROverlay._state = {
      x: targetX, y: targetY,
      w: garmentWidth, h: garmentHeight,
      a: angle
    };
  }

  const prev = drawAROverlay._state;
  const motionDelta = Math.hypot(targetX - prev.x, targetY - prev.y);
  const posFactor = motionDelta > 15 ? 0.55 : 0.20;

  // SIZE: use fast convergence when size changes are large (>20%)
  // This prevents the garment from being "stuck small" when first placed
  const sizeChangeRatio = Math.max(
    Math.abs(garmentWidth - prev.w) / (prev.w || 1),
    Math.abs(garmentHeight - prev.h) / (prev.h || 1)
  );
  const sizeFactor = sizeChangeRatio > 0.2 ? 0.50 : 0.20;

  const sX = prev.x + (targetX      - prev.x) * posFactor;
  const sY = prev.y + (targetY      - prev.y) * posFactor;
  const sW = prev.w + (garmentWidth  - prev.w) * sizeFactor;
  const sH = prev.h + (garmentHeight - prev.h) * sizeFactor;
  const sA = prev.a + (angle         - prev.a) * posFactor;

  drawAROverlay._state = { x: sX, y: sY, w: sW, h: sH, a: sA };

  // ════════════════════════════════════════════
  // RENDER
  // ════════════════════════════════════════════

  const confAlpha = 0.55 + Math.min(1, (shoulderVis - 0.5) / 0.3) * 0.40;

  // Shadow
  ctx.save();
  ctx.translate(sX + 2, sY + 4);
  ctx.rotate(sA);
  ctx.globalAlpha = 0.08;
  ctx.filter = 'blur(8px)';
  ctx.drawImage(activeGarment, -sW / 2, -sH / 2, sW, sH);
  ctx.filter = 'none';
  ctx.restore();

  // Garment
  ctx.save();
  ctx.translate(sX, sY);
  ctx.rotate(sA);
  ctx.globalAlpha = confAlpha;
  ctx.drawImage(activeGarment, -sW / 2, -sH / 2, sW, sH);
  ctx.restore();

  // Orientation indicator
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.fillStyle = orientation === 'front' ? '#4CAF50' : '#FF9800';
  ctx.font = '10px DM Sans, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(orientation === 'front' ? '👤 Front' : '🔙 Back', 8, canvasHeight - 8);
  ctx.restore();


}

function drawGarmentAtSmoothedPosition(ctx, garmentImg, alpha) {
  const s = drawAROverlay._state;
  if (!s || alpha <= 0) return;
  ctx.save();
  ctx.translate(s.x, s.y);
  ctx.rotate(s.a);
  ctx.globalAlpha = alpha;
  ctx.drawImage(garmentImg, -s.w / 2, -s.h / 2, s.w, s.h);
  ctx.restore();
}

// ═══════════════════════════════════════════
// FACE DETECTION (stub — not needed for demo)
// ═══════════════════════════════════════════

async function getFace(videoElement) {
  // MediaPipe Face Landmarker could be added here if needed
  return null;
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
    case 'segmenter': return !!imageSegmenter;
    case 'pose': return !!poseLandmarker;
    case 'face': return false; // not loaded in v6
    default: return modelsLoaded;
  }
}

// ================================================================
// SMART SIZE RECOMMENDATION — SizeRec module
// MediaPipe Pose → pixel measurements → cm via calibration → S–XXL
// ================================================================

const SizeRec = (() => {

  const SIZE_CHART = [
    { size: "S",   chest: 88.9,  frontLength: 64.8, shoulder: 41.9 },
    { size: "M",   chest: 94.0,  frontLength: 67.3, shoulder: 43.2 },
    { size: "L",   chest: 99.1,  frontLength: 69.9, shoulder: 44.5 },
    { size: "XL",  chest: 104.1, frontLength: 72.4, shoulder: 45.7 },
    { size: "XXL", chest: 109.2, frontLength: 74.9, shoulder: 47.0 },
  ];

  const WEIGHTS = { shoulder: 0.45, chest: 0.35, frontLength: 0.20 };

  // State
  let calFrames = [], mBuffer = [], lockedResult = null, stableCount = 0;

  // ── Helpers ──────────────────────────────────────────────────
  const dist2D = (a, b) =>
    Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

  const median = arr => {
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
  };

  const stdDev = arr => {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v-mean)**2, 0) / arr.length);
  };

  const vis = (kp, idx) =>
    kp[idx] && (kp[idx].visibility ?? kp[idx].score ?? 1) >= 0.70;

  const allVis = (kp, ids) => ids.every(i => vis(kp, i));

  // ── Phase 1: Calibration ─────────────────────────────────────
  function feedCalibrationFrame(kp, userHeightCm) {
    if (!allVis(kp, [0, 27, 28])) return { ready: false };
    const noseY   = kp[0].y;
    const ankleY  = Math.max(kp[27].y, kp[28].y);
    const heightPx = Math.abs(ankleY - noseY);
    if (heightPx < 50) return { ready: false };
    calFrames.push(userHeightCm / heightPx);
    if (calFrames.length > 10) calFrames.shift();
    if (calFrames.length >= 10 && stdDev(calFrames) < 0.008)
      return { ready: true, cmPerPixel: median(calFrames) };
    return { ready: false, collected: calFrames.length };
  }

  function resetCalibration() { calFrames = []; }

  // ── Phase 2: Measurement ─────────────────────────────────────
  function feedMeasurementFrame(kp, cmPerPixel) {
    if (!allVis(kp, [11, 12, 23, 24])) return null;
    const [lS, rS, lH, rH] = [kp[11], kp[12], kp[23], kp[24]];

    // Frontality gate: hip width must be in human range
    const hipCm = dist2D(lH, rH) * cmPerPixel;
    if (hipCm < 18 || hipCm > 60) return null;

    const shoulderCm    = dist2D(lS, rS) * cmPerPixel;
    const shoulderMidY  = (lS.y + rS.y) / 2;
    const hipMidY       = (lH.y + rH.y) / 2;
    const frontLengthCm = Math.abs(hipMidY - shoulderMidY) * cmPerPixel;
    const chestCm       = Math.min(Math.max(shoulderCm * 2.25, 75), 130);

    mBuffer.push({ shoulderCm, chestCm, frontLengthCm });
    if (mBuffer.length > 30) mBuffer.shift();
    if (mBuffer.length < 10) return null;

    return {
      shoulderCm:    median(mBuffer.map(f => f.shoulderCm)),
      chestCm:       median(mBuffer.map(f => f.chestCm)),
      frontLengthCm: median(mBuffer.map(f => f.frontLengthCm)),
      stdDevShoulder: stdDev(mBuffer.map(f => f.shoulderCm)),
      frames: mBuffer.length,
    };
  }

  function resetMeasurements() {
    mBuffer = []; lockedResult = null; stableCount = 0;
  }

  // ── Phase 3: Size matching ───────────────────────────────────
  function recommendSize(m) {
    const scores = SIZE_CHART.map(row => {
      const d = WEIGHTS.shoulder * Math.abs(m.shoulderCm - row.shoulder) / row.shoulder
              + WEIGHTS.chest    * Math.abs(m.chestCm    - row.chest)    / row.chest
              + WEIGHTS.frontLength * Math.abs(m.frontLengthCm - row.frontLength) / row.frontLength;
      return { ...row, confidence: Math.max(0, 100 * Math.exp(-d * 12)), dist: d };
    });
    return scores.sort((a, b) => a.dist - b.dist)[0];
  }

  // ── Phase 4: Lock ────────────────────────────────────────────
  function tryLock(measurements) {
    if (!measurements) { stableCount = Math.max(0, stableCount - 1); return lockedResult; }
    const stable = measurements.stdDevShoulder < 2.0 && measurements.frames >= 15;
    stableCount = stable ? stableCount + 1 : Math.max(0, stableCount - 1);
    if (stableCount >= 20 && !lockedResult) {
      const rec = recommendSize(measurements);
      if (rec.confidence >= 60) {
        lockedResult = { ...rec, measurements, lockedAt: Date.now() };
      }
    }
    return lockedResult;
  }

  function clearLock() { lockedResult = null; stableCount = 0; }

  return { feedCalibrationFrame, resetCalibration,
           feedMeasurementFrame, resetMeasurements,
           recommendSize, tryLock, clearLock,
           getSizeChart: () => SIZE_CHART };
})();


// Expose globally
window.FVAI = {
  loadModels, removeBackground, autoCropGarment, getPose,
  drawAROverlay, detectOrientation, getOrientationConfidence,
  getFace, drawMannequinFaceSwap,
  isModelLoaded, resetSmoothing,
  SizeRec
};
