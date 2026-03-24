// ==========================================
// FitVision — AI Module (v4 — Fixed Crop + Alignment)
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

    // 2. Pose Detection (for AR overlay + garment cropping)
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
// AUTO-CROP GARMENT — Polygon Clipping
//
// The key problem: body segmentation removes background but
// keeps the ENTIRE body (head, arms, legs). We need ONLY the
// garment (shirt/t-shirt area: shoulders to hips).
//
// Solution:
// 1. Detect pose to find shoulder, hip, elbow keypoints
// 2. Build a torso polygon (trapezoid shape)
// 3. Clip the image to ONLY that polygon
// 4. Everything outside (head, hands, legs) becomes transparent
// 5. Alpha-trim the result for a tight crop
// ═══════════════════════════════════════════

async function autoCropGarment(dataUrl, videoElement) {
  return new Promise(async (resolve) => {
    const img = new Image();
    img.onload = async () => {
      const sw = img.width;
      const sh = img.height;

      // Draw original image
      const srcCanvas = document.createElement('canvas');
      srcCanvas.width = sw;
      srcCanvas.height = sh;
      const srcCtx = srcCanvas.getContext('2d');
      srcCtx.drawImage(img, 0, 0);

      // Step 1: Get pose keypoints from the video
      let clipped = false;
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

              // Build torso polygon points
              // Top: neckline (slightly above shoulders, no head)
              const neckY = Math.min(ls.y, rs.y) - shoulderW * 0.15;

              // Bottom: below hips
              let bottomY;
              if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
                bottomY = Math.max(lh.y, rh.y) + shoulderW * 0.1;
              } else {
                bottomY = Math.max(ls.y, rs.y) + shoulderW * 1.3;
              }

              // Sleeve width: extend to elbows or fixed padding
              let leftX = Math.min(ls.x, rs.x) - shoulderW * 0.45;
              let rightX = Math.max(ls.x, rs.x) + shoulderW * 0.45;

              if (le && le.score > 0.3) leftX = Math.min(leftX, le.x - shoulderW * 0.1);
              if (re && re.score > 0.3) rightX = Math.max(rightX, re.x + shoulderW * 0.1);

              // Hip width (slightly narrower for shirt taper)
              let leftHipX, rightHipX;
              if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
                leftHipX = Math.min(lh.x, rh.x) - shoulderW * 0.2;
                rightHipX = Math.max(lh.x, rh.x) + shoulderW * 0.2;
              } else {
                leftHipX = leftX + shoulderW * 0.1;
                rightHipX = rightX - shoulderW * 0.1;
              }

              // Clamp all points to image bounds
              const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
              const topLeft = { x: clamp(leftX, 0, sw), y: clamp(neckY, 0, sh) };
              const topRight = { x: clamp(rightX, 0, sw), y: clamp(neckY, 0, sh) };
              const bottomRight = { x: clamp(rightHipX, 0, sw), y: clamp(bottomY, 0, sh) };
              const bottomLeft = { x: clamp(leftHipX, 0, sw), y: clamp(bottomY, 0, sh) };

              // Neck cutout (V-shape at top center to remove chin/neck area)
              const neckCenterX = (ls.x + rs.x) / 2;
              const neckWidth = shoulderW * 0.25;
              const neckDepth = shoulderW * 0.05; // Very shallow — just remove neck skin

              // Step 2: Create clipped version — only torso polygon is visible
              const clipCanvas = document.createElement('canvas');
              clipCanvas.width = sw;
              clipCanvas.height = sh;
              const clipCtx = clipCanvas.getContext('2d');

              // Draw polygon clip path (torso shape)
              clipCtx.beginPath();
              // Start from top-left shoulder area
              clipCtx.moveTo(topLeft.x, topLeft.y);
              // Go to left side of neck
              clipCtx.lineTo(clamp(neckCenterX - neckWidth, 0, sw), clamp(neckY, 0, sh));
              // Neck V cutout
              clipCtx.lineTo(neckCenterX, clamp(neckY + neckDepth, 0, sh));
              // Right side of neck
              clipCtx.lineTo(clamp(neckCenterX + neckWidth, 0, sw), clamp(neckY, 0, sh));
              // Top-right shoulder area
              clipCtx.lineTo(topRight.x, topRight.y);
              // Down to bottom-right hip
              clipCtx.lineTo(bottomRight.x, bottomRight.y);
              // Across to bottom-left hip
              clipCtx.lineTo(bottomLeft.x, bottomLeft.y);
              clipCtx.closePath();
              clipCtx.clip();

              // Draw the original (bg-removed) image inside the clip
              clipCtx.drawImage(srcCanvas, 0, 0);
              clipped = true;

              // Copy clipped result back
              srcCtx.clearRect(0, 0, sw, sh);
              srcCtx.drawImage(clipCanvas, 0, 0);
            }
          }
        } catch (e) {
          console.warn('Pose-based crop failed:', e);
        }
      }

      // Step 3: Alpha-trim (remove all transparent padding)
      const bounds = getAlphaBounds(srcCtx, sw, sh);
      if (!bounds || bounds.w < 10 || bounds.h < 10) {
        resolve(dataUrl);
        return;
      }

      // Step 4: Extract the tight crop
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = bounds.w;
      finalCanvas.height = bounds.h;
      const finalCtx = finalCanvas.getContext('2d');
      finalCtx.drawImage(
        srcCanvas,
        bounds.x, bounds.y, bounds.w, bounds.h,
        0, 0, bounds.w, bounds.h
      );

      resolve(finalCanvas.toDataURL('image/png'));
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
// AR OVERLAY — Fixed Alignment
// ═══════════════════════════════════════════
//
// ALIGNMENT FIX:
// Previously the garment was centered on a point BELOW the shoulders,
// which caused the top of the garment image to appear at chest level
// (looking "upside down" when the captured image still had head/neck).
//
// Now that the garment image is properly cropped (just the shirt),
// we draw it so its TOP EDGE aligns with the buyer's shoulder line
// and it extends DOWN to the hips. The anchor point is the shoulder
// midpoint, and the garment hangs downward from there.

let framesWithoutPose = 0;
const MAX_FADE_FRAMES = 15;

function drawAROverlay(ctx, canvasWidth, canvasHeight, garmentImg, keypoints, videoElement) {
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

  // Confidence-based opacity
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

  // Scale keypoints: video space → canvas space
  const videoW = videoElement ? (videoElement.videoWidth || canvasWidth) : canvasWidth;
  const videoH = videoElement ? (videoElement.videoHeight || canvasHeight) : canvasHeight;
  const scaleX = canvasWidth / videoW;
  const scaleY = canvasHeight / videoH;

  const lsx = ls.x * scaleX, lsy = ls.y * scaleY;
  const rsx = rs.x * scaleX, rsy = rs.y * scaleY;

  // Shoulder measurements
  const shoulderWidth = Math.hypot(rsx - lsx, rsy - lsy);
  if (shoulderWidth < 20) return;

  // Garment width: 1.8× shoulder distance
  const rawGarmentWidth = shoulderWidth * 1.8;

  // Garment height: based on shoulder-to-hip distance
  let rawGarmentHeight;
  if (lh && rh && lh.score > 0.25 && rh.score > 0.25) {
    const lhy = lh.y * scaleY;
    const rhy = rh.y * scaleY;
    const hipMidY = (lhy + rhy) / 2;
    const shoulderMidY = (lsy + rsy) / 2;
    rawGarmentHeight = (hipMidY - shoulderMidY) * 1.35;
  } else {
    rawGarmentHeight = rawGarmentWidth * 1.3;
  }
  rawGarmentHeight = Math.max(rawGarmentHeight, 60);

  // ════════════════════════════════════════
  // ANCHOR POINT: Shoulder midpoint
  // The garment TOP edge sits at shoulder level,
  // and the garment HANGS DOWNWARD from there
  // ════════════════════════════════════════
  const shoulderMidX = (lsx + rsx) / 2;
  const shoulderMidY = (lsy + rsy) / 2;

  // The center of the garment image is at:
  //   Y = shoulderMidY + garmentHeight/2
  // (garment starts at shoulders, extends down to hips)
  // Small upward offset (-0.05) so neckline sits slightly above shoulder line
  const rawCenterX = shoulderMidX;
  const rawCenterY = shoulderMidY + rawGarmentHeight * 0.45;

  // Shoulder tilt angle
  const rawAngle = Math.atan2(rsy - lsy, rsx - lsx);

  // Apply smoothing
  const cx = smooth('x', rawCenterX);
  const cy = smooth('y', rawCenterY);
  const gw = smooth('w', rawGarmentWidth);
  const gh = smooth('h', rawGarmentHeight);
  const angle = smooth('angle', rawAngle);

  // Draw shadow for depth
  ctx.save();
  ctx.translate(cx + 3, cy + 6);
  ctx.rotate(angle);
  ctx.globalAlpha = 0.1;
  ctx.filter = 'blur(10px)';
  ctx.drawImage(garmentImg, -gw / 2, -gh / 2, gw, gh);
  ctx.filter = 'none';
  ctx.restore();

  // Draw garment
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.globalAlpha = confAlpha;
  ctx.drawImage(garmentImg, -gw / 2, -gh / 2, gw, gh);
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

  // Face swap
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
