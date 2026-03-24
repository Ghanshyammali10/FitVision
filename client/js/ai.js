// ==========================================
// FitVision — AI Module (v3 — Production AR)
// TensorFlow.js: Body Segmentation, Pose Detection, Face Detection
// ==========================================

let segmenter = null;
let poseDetector = null;
let faceDetector = null;
let modelsLoaded = false;

// ═══════════════════════════════════════════
// SMOOTHING — Adaptive Exponential Moving Average
// Prevents garment jitter by blending frames
// Uses velocity-aware adaptive smoothing:
//   - Slow movements → heavy smoothing (less jitter)
//   - Fast movements → light smoothing (less lag)
// ═══════════════════════════════════════════
const smoothing = {
  x: null, y: null, w: null, h: null, angle: null,
  baseFactor: 0.3,   // Base EMA factor
  minFactor: 0.15,   // Heavy smoothing for slow motion
  maxFactor: 0.6,    // Light smoothing for fast motion
  velocityThreshold: 15, // px/frame to consider "fast"
  prevRawX: null,
  prevRawY: null,
  confidence: 1.0    // Tracks overall detection confidence for fade
};

function smooth(key, rawValue) {
  if (smoothing[key] === null || isNaN(smoothing[key])) {
    smoothing[key] = rawValue;
    return rawValue;
  }

  // Adaptive factor based on movement velocity
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
  smoothing.prevRawX = null;
  smoothing.prevRawY = null;
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
    // 1. Body Segmentation (for background removal)
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

    // 2. Pose Detection (for AR overlay — THE KEY MODEL)
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

    // 3. Face Detection (for mannequin face swap)
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
        frame.data[i + 3] = 0; // transparent
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
// AUTO-CROP GARMENT
// Trims transparent pixels + crops to torso region
// using pose detection for precise body-part isolation
// ═══════════════════════════════════════════

async function autoCropGarment(dataUrl, videoElement) {
  return new Promise(async (resolve) => {
    const img = new Image();
    img.onload = async () => {
      const sw = img.width;
      const sh = img.height;

      // Step 1: Try pose-based cropping (shoulders to hips)
      let cropBounds = null;
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
              const shoulderWidth = Math.abs(rs.x - ls.x);
              const padding = shoulderWidth * 0.5; // Extra space for sleeves

              // Top: above shoulders
              let top = Math.min(ls.y, rs.y) - shoulderWidth * 0.25;

              // Bottom: at hips or estimate
              let bottom;
              if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
                bottom = Math.max(lh.y, rh.y) + shoulderWidth * 0.15;
              } else {
                bottom = Math.max(ls.y, rs.y) + shoulderWidth * 1.4;
              }

              // Left/Right: beyond shoulders (include sleeves/arms)
              let left = Math.min(ls.x, rs.x) - padding;
              let right = Math.max(ls.x, rs.x) + padding;

              // Extend to elbows if detected (for sleeve coverage)
              if (le && le.score > 0.3) left = Math.min(left, le.x - shoulderWidth * 0.15);
              if (re && re.score > 0.3) right = Math.max(right, re.x + shoulderWidth * 0.15);

              // Clamp to image bounds
              cropBounds = {
                x: Math.max(0, Math.floor(left)),
                y: Math.max(0, Math.floor(top)),
                w: Math.min(sw, Math.ceil(right)) - Math.max(0, Math.floor(left)),
                h: Math.min(sh, Math.ceil(bottom)) - Math.max(0, Math.floor(top))
              };
            }
          }
        } catch (e) {
          console.warn('Pose crop failed, falling back to alpha-trim:', e);
        }
      }

      // Step 2: Apply crop bounds (pose-based or alpha-trim)
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = sw;
      tempCanvas.height = sh;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(img, 0, 0);

      if (!cropBounds) {
        // Fallback: trim transparent pixels
        cropBounds = getAlphaBounds(tempCtx, sw, sh);
      }

      if (!cropBounds || cropBounds.w < 10 || cropBounds.h < 10) {
        resolve(dataUrl); // Too small or no valid bounds
        return;
      }

      // Step 3: Extract cropped region
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = cropBounds.w;
      cropCanvas.height = cropBounds.h;
      const cropCtx = cropCanvas.getContext('2d');
      cropCtx.drawImage(
        tempCanvas,
        cropBounds.x, cropBounds.y, cropBounds.w, cropBounds.h,
        0, 0, cropBounds.w, cropBounds.h
      );

      // Step 4: Clean up — remove pixels outside the body mask
      // Re-alpha any remaining non-garment pixels at edges
      const cropData = cropCtx.getImageData(0, 0, cropBounds.w, cropBounds.h);
      let hasTransparent = false;
      for (let i = 0; i < cropData.data.length; i += 4) {
        if (cropData.data[i + 3] < 30) {
          cropData.data[i + 3] = 0;
          hasTransparent = true;
        }
      }
      if (hasTransparent) cropCtx.putImageData(cropData, 0, 0);

      // Step 5: Final alpha-trim on the cropped result
      const finalBounds = getAlphaBounds(cropCtx, cropBounds.w, cropBounds.h);
      if (finalBounds && finalBounds.w > 10 && finalBounds.h > 10) {
        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = finalBounds.w;
        finalCanvas.height = finalBounds.h;
        const finalCtx = finalCanvas.getContext('2d');
        finalCtx.drawImage(
          cropCanvas,
          finalBounds.x, finalBounds.y, finalBounds.w, finalBounds.h,
          0, 0, finalBounds.w, finalBounds.h
        );
        resolve(finalCanvas.toDataURL('image/png'));
      } else {
        resolve(cropCanvas.toDataURL('image/png'));
      }
    };
    img.src = dataUrl;
  });
}

/**
 * Find the bounding box of non-transparent pixels
 */
function getAlphaBounds(ctx, w, h) {
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  let minX = w, minY = h, maxX = 0, maxY = 0;
  let found = false;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      if (data[idx + 3] > 30) { // Non-transparent
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        found = true;
      }
    }
  }

  if (!found) return null;

  // Add small padding
  const pad = 4;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(w - 1, maxX + pad);
  maxY = Math.min(h - 1, maxY + pad);

  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
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
// AR OVERLAY — PRODUCTION QUALITY
// ═══════════════════════════════════════════
//
// WHY THE PREVIOUS VERSION LOOKED "FLOATING":
//
// 1. TRANSPARENCY DESTROYED: Seller compressed garment to JPEG,
//    which flattens the alpha channel → garment becomes a rectangle.
//    FIX: Use PNG throughout the pipeline.
//
// 2. NO COORDINATE SCALING: MoveNet returns keypoints in the video's
//    native pixel space (e.g. 1280×720). But the <canvas> overlay may
//    be sized differently. If you draw at raw keypoint coordinates
//    without scaling, the garment lands in the wrong spot.
//    FIX: Scale using videoW/videoH → canvasW/canvasH ratio.
//
// 3. NO MIRROR: Front-facing camera is mirrored in <video> but the
//    canvas was not mirrored, so the garment moved opposite.
//    FIX: CSS transform: scaleX(-1) on both video and canvas.
//
// 4. HARD RESET: On low confidence, resetSmoothing() caused jarring
//    jumps. FIX: Fade opacity down instead, keep last known position.
//
// 5. GARMENT NOT CROPPED: The image included background and body parts,
//    causing a rectangular "sticker" look.
//    FIX: Auto-crop function trims to garment-only bounds.
//
// NEW FEATURES:
//   - Adaptive velocity-aware smoothing
//   - Shoulder tilt rotation
//   - Depth shadow for 3D feel
//   - Confidence-based opacity fade
//   - Proper translate+rotate rendering

// Track frames without detection for graceful fade
let framesWithoutPose = 0;
const MAX_FADE_FRAMES = 15;

function drawAROverlay(ctx, canvasWidth, canvasHeight, garmentImg, keypoints, videoElement) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  if (!garmentImg) return;

  // ── No keypoints: graceful fade out ──
  if (!keypoints) {
    framesWithoutPose++;
    if (framesWithoutPose > MAX_FADE_FRAMES || smoothing.x === null) return;

    // Draw at last known position with fading opacity
    const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
    drawGarmentAtSmoothedPosition(ctx, canvasWidth, canvasHeight, garmentImg, fadeAlpha);
    return;
  }

  framesWithoutPose = 0;

  // ── Step 1: Extract keypoints by name ──
  const kp = {};
  keypoints.forEach(k => { kp[k.name] = k; });

  const ls = kp['left_shoulder'];
  const rs = kp['right_shoulder'];
  const lh = kp['left_hip'];
  const rh = kp['right_hip'];

  // Need at least both shoulders
  if (!ls || !rs) return;

  // ── Confidence-based opacity ──
  const shoulderConf = Math.min(ls.score || 0, rs.score || 0);
  if (shoulderConf < 0.15) {
    // Too low — use fade behavior
    framesWithoutPose++;
    if (smoothing.x !== null) {
      const fadeAlpha = Math.max(0, 0.88 * (1 - framesWithoutPose / MAX_FADE_FRAMES));
      drawGarmentAtSmoothedPosition(ctx, canvasWidth, canvasHeight, garmentImg, fadeAlpha);
    }
    return;
  }

  // Confidence maps to alpha: 0.15→0.4 confidence = 0.3→0.88 alpha
  const confAlpha = 0.3 + Math.min(1, (shoulderConf - 0.15) / 0.5) * 0.58;

  // ── Step 2: Scale keypoints from video space → canvas space ──
  const videoW = videoElement ? (videoElement.videoWidth || canvasWidth) : canvasWidth;
  const videoH = videoElement ? (videoElement.videoHeight || canvasHeight) : canvasHeight;
  const scaleX = canvasWidth / videoW;
  const scaleY = canvasHeight / videoH;

  const lsx = ls.x * scaleX, lsy = ls.y * scaleY;
  const rsx = rs.x * scaleX, rsy = rs.y * scaleY;

  // ── Step 3: Calculate garment dimensions ──
  const shoulderWidth = Math.hypot(rsx - lsx, rsy - lsy);

  // Guard: if shoulders are too close, detection is probably wrong
  if (shoulderWidth < 20) return;

  // Width: 1.8× shoulder distance for natural drape (covers sleeves)
  const rawGarmentWidth = shoulderWidth * 1.8;

  let rawGarmentHeight;
  if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
    // Use actual shoulder-to-hip distance
    const lhx = lh.x * scaleX, lhy = lh.y * scaleY;
    const rhx = rh.x * scaleX, rhy = rh.y * scaleY;
    const hipMidY = (lhy + rhy) / 2;
    const shoulderMidY = (lsy + rsy) / 2;
    rawGarmentHeight = (hipMidY - shoulderMidY) * 1.35;
  } else {
    // Estimate: garment height ≈ 1.3× width (standard shirt ratio)
    rawGarmentHeight = rawGarmentWidth * 1.3;
  }

  // Minimum dimensions
  rawGarmentHeight = Math.max(rawGarmentHeight, 60);

  // ── Step 4: Calculate center position ──
  const rawCenterX = (lsx + rsx) / 2;
  // Shift down by 35% of garment height to center on torso (not on shoulder line)
  const rawCenterY = (lsy + rsy) / 2 + rawGarmentHeight * 0.35;

  // ── Step 5: Calculate shoulder tilt angle ──
  const rawAngle = Math.atan2(rsy - lsy, rsx - lsx);

  // ── Step 6: Apply adaptive EMA smoothing ──
  const cx = smooth('x', rawCenterX);
  const cy = smooth('y', rawCenterY);
  const gw = smooth('w', rawGarmentWidth);
  const gh = smooth('h', rawGarmentHeight);
  const angle = smooth('angle', rawAngle);

  // ── Step 7: Draw shadow for depth ──
  ctx.save();
  ctx.translate(cx + 3, cy + 8);
  ctx.rotate(angle);
  ctx.globalAlpha = 0.12;
  ctx.filter = 'blur(12px)';
  ctx.drawImage(garmentImg, -gw / 2, -gh / 2, gw, gh);
  ctx.filter = 'none';
  ctx.restore();

  // ── Step 8: Draw garment with rotation ──
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.globalAlpha = confAlpha;
  ctx.drawImage(garmentImg, -gw / 2, -gh / 2, gw, gh);
  ctx.restore();
}

/**
 * Draw garment at the last smoothed position (for fade-out frames)
 */
function drawGarmentAtSmoothedPosition(ctx, canvasWidth, canvasHeight, garmentImg, alpha) {
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

  // Background
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);

  // Mannequin silhouette
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

  // Garment on torso
  if (garmentImg) {
    const gw = 140;
    const gh = gw * (garmentImg.height / garmentImg.width);
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.drawImage(garmentImg, w / 2 - gw / 2, h * 0.2, gw, Math.min(gh, h * 0.42));
    ctx.restore();
  }

  // Face swap onto mannequin head
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
  drawAROverlay, getFace, drawMannequinFaceSwap,
  isModelLoaded, resetSmoothing
};
