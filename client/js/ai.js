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
// AUTO-CROP GARMENT — Polygon Clipping
// Clips to torso polygon (shoulders→hips), removes head/hands/legs
// ═══════════════════════════════════════════

async function autoCropGarment(dataUrl, videoElement) {
  return new Promise(async (resolve) => {
    const img = new Image();
    img.onload = async () => {
      const sw = img.width;
      const sh = img.height;

      const srcCanvas = document.createElement('canvas');
      srcCanvas.width = sw;
      srcCanvas.height = sh;
      const srcCtx = srcCanvas.getContext('2d');
      srcCtx.drawImage(img, 0, 0);

      // Pose-based polygon clip
      if (poseDetector && videoElement) {
        try {
          const poses = await poseDetector.estimatePoses(videoElement, { flipHorizontal: false });
          if (poses.length > 0 && poses[0].keypoints) {
            const kp = {};
            poses[0].keypoints.forEach(k => { kp[k.name] = k; });

            const ls = kp['left_shoulder'];
            const rs = kp['right_shoulder'];
            const lh = kp['left_hip'];
            const rh = kp['right_hip'];
            const le = kp['left_elbow'];
            const re = kp['right_elbow'];

            if (ls && rs && ls.score > 0.3 && rs.score > 0.3) {
              const shoulderW = Math.abs(rs.x - ls.x);
              const neckY = Math.min(ls.y, rs.y) - shoulderW * 0.15;

              let bottomY;
              if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
                bottomY = Math.max(lh.y, rh.y) + shoulderW * 0.1;
              } else {
                bottomY = Math.max(ls.y, rs.y) + shoulderW * 1.3;
              }

              let leftX = Math.min(ls.x, rs.x) - shoulderW * 0.45;
              let rightX = Math.max(ls.x, rs.x) + shoulderW * 0.45;

              if (le && le.score > 0.3) leftX = Math.min(leftX, le.x - shoulderW * 0.1);
              if (re && re.score > 0.3) rightX = Math.max(rightX, re.x + shoulderW * 0.1);

              let leftHipX, rightHipX;
              if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
                leftHipX = Math.min(lh.x, rh.x) - shoulderW * 0.2;
                rightHipX = Math.max(lh.x, rh.x) + shoulderW * 0.2;
              } else {
                leftHipX = leftX + shoulderW * 0.1;
                rightHipX = rightX - shoulderW * 0.1;
              }

              const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
              const neckCenterX = (ls.x + rs.x) / 2;
              const neckWidth = shoulderW * 0.25;

              const clipCanvas = document.createElement('canvas');
              clipCanvas.width = sw;
              clipCanvas.height = sh;
              const clipCtx = clipCanvas.getContext('2d');

              clipCtx.beginPath();
              clipCtx.moveTo(clamp(leftX, 0, sw), clamp(neckY, 0, sh));
              clipCtx.lineTo(clamp(neckCenterX - neckWidth, 0, sw), clamp(neckY, 0, sh));
              clipCtx.lineTo(neckCenterX, clamp(neckY + shoulderW * 0.05, 0, sh));
              clipCtx.lineTo(clamp(neckCenterX + neckWidth, 0, sw), clamp(neckY, 0, sh));
              clipCtx.lineTo(clamp(rightX, 0, sw), clamp(neckY, 0, sh));
              clipCtx.lineTo(clamp(rightHipX, 0, sw), clamp(bottomY, 0, sh));
              clipCtx.lineTo(clamp(leftHipX, 0, sw), clamp(bottomY, 0, sh));
              clipCtx.closePath();
              clipCtx.clip();

              clipCtx.drawImage(srcCanvas, 0, 0);
              srcCtx.clearRect(0, 0, sw, sh);
              srcCtx.drawImage(clipCanvas, 0, 0);
            }
          }
        } catch (e) {
          console.warn('Pose-based crop failed:', e);
        }
      }

      // Alpha-trim
      const bounds = getAlphaBounds(srcCtx, sw, sh);
      if (!bounds || bounds.w < 10 || bounds.h < 10) {
        resolve(dataUrl);
        return;
      }

      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = bounds.w;
      finalCanvas.height = bounds.h;
      const finalCtx = finalCanvas.getContext('2d');
      finalCtx.drawImage(srcCanvas, bounds.x, bounds.y, bounds.w, bounds.h, 0, 0, bounds.w, bounds.h);
      resolve(finalCanvas.toDataURL('image/png'));
    };
    img.src = dataUrl;
  });
}

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
// FRONT / BACK ORIENTATION DETECTION
// ═══════════════════════════════════════════
//
// Uses MoveNet heuristics:
//   - FRONT: nose keypoint has high confidence (face visible)
//   - BACK: nose confidence is low but shoulders are detected
//
// Returns: 'front' | 'back'

// Smoothed orientation to avoid flickering
let orientationHistory = [];
const ORIENTATION_WINDOW = 8; // frames to average

function detectOrientation(keypoints) {
  if (!keypoints) return 'front'; // default

  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  const nose = kp['nose'];
  const leftEye = kp['left_eye'];
  const rightEye = kp['right_eye'];
  const leftEar = kp['left_ear'];
  const rightEar = kp['right_ear'];
  const ls = kp['left_shoulder'];
  const rs = kp['right_shoulder'];

  // Score face visibility (nose + eyes)
  const noseConf = nose ? nose.score : 0;
  const eyeConf = ((leftEye ? leftEye.score : 0) + (rightEye ? rightEye.score : 0)) / 2;
  const earConf = ((leftEar ? leftEar.score : 0) + (rightEar ? rightEar.score : 0)) / 2;

  // Face score: high when face is visible
  const faceScore = noseConf * 0.5 + eyeConf * 0.35 + earConf * 0.15;

  // If face is clearly visible → front, otherwise → back
  const isFront = faceScore > 0.35;

  // Add to history for smoothing
  orientationHistory.push(isFront ? 1 : 0);
  if (orientationHistory.length > ORIENTATION_WINDOW) {
    orientationHistory.shift();
  }

  // Average: if > 50% of recent frames say "front", it's front
  const avg = orientationHistory.reduce((a, b) => a + b, 0) / orientationHistory.length;
  return avg > 0.5 ? 'front' : 'back';
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
  drawAROverlay, detectOrientation, getFace, drawMannequinFaceSwap,
  isModelLoaded, resetSmoothing
};
