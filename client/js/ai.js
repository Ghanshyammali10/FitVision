// ==========================================
// FitVision — AI Features Module
// TensorFlow.js models loaded from CDN
// Body segmentation, pose detection, face detection
// ==========================================

// Model instances (loaded once, reused)
let segmenter = null;
let poseDetector = null;
let faceDetector = null;
let modelsLoaded = false;
let cachedPose = null;
let lastFaceData = null;
let lastFaceTimestamp = 0;

/**
 * Load all AI models with progress callbacks
 * @param {function} onProgress - callback({ model: string, status: string, percent: number })
 */
async function loadModels(onProgress) {
  const progress = onProgress || (() => {});

  try {
    // Wait for TF.js to be ready
    await tf.ready();
    progress({ model: 'TensorFlow.js', status: 'Runtime ready', percent: 10 });

    // 1. Body Segmentation (for background removal)
    progress({ model: 'Body Segmentation', status: 'Loading...', percent: 15 });
    try {
      segmenter = await bodySegmentation.createSegmenter(
        bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
        {
          runtime: 'tfjs',
          modelType: 'general'
        }
      );
      progress({ model: 'Body Segmentation', status: 'Ready ✓', percent: 40 });
    } catch (e) {
      console.warn('[AI] Body segmentation failed to load:', e);
      progress({ model: 'Body Segmentation', status: 'Fallback mode', percent: 40 });
    }

    // 2. Pose Detection (MoveNet)
    progress({ model: 'Pose Detection', status: 'Loading...', percent: 45 });
    try {
      poseDetector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
        }
      );
      progress({ model: 'Pose Detection', status: 'Ready ✓', percent: 70 });
    } catch (e) {
      console.warn('[AI] Pose detection failed to load:', e);
      progress({ model: 'Pose Detection', status: 'Fallback mode', percent: 70 });
    }

    // 3. Face Detection
    progress({ model: 'Face Detection', status: 'Loading...', percent: 75 });
    try {
      faceDetector = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        {
          runtime: 'tfjs',
          maxFaces: 1
        }
      );
      progress({ model: 'Face Detection', status: 'Ready ✓', percent: 95 });
    } catch (e) {
      console.warn('[AI] Face detection failed to load:', e);
      progress({ model: 'Face Detection', status: 'Fallback mode', percent: 95 });
    }

    modelsLoaded = true;
    progress({ model: 'All Models', status: 'Complete ✓', percent: 100 });
    return true;
  } catch (error) {
    console.error('[AI] Model loading failed:', error);
    progress({ model: 'Error', status: error.message, percent: 0 });
    return false;
  }
}

/**
 * Remove background from video frame using body segmentation
 * @param {HTMLVideoElement} videoElement
 * @returns {string} PNG data URL with transparent background
 */
async function removeBackground(videoElement) {
  if (!segmenter) {
    // Fallback: return the video frame as-is
    return captureFrame(videoElement);
  }

  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext('2d');

  // Draw current video frame
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  try {
    // Run segmentation
    const people = await segmenter.segmentPeople(canvas, {
      flipHorizontal: false,
      multiSegmentation: false,
      segmentBodyParts: false
    });

    if (people && people.length > 0) {
      // Get binary mask
      const mask = await bodySegmentation.toBinaryMask(
        people,
        { r: 0, g: 0, b: 0, a: 0 },       // background => transparent
        { r: 0, g: 0, b: 0, a: 255 },       // foreground => opaque
        false,
        0.5                                   // threshold
      );

      // Apply mask to image
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const maskData = mask.data;

      for (let i = 0; i < imageData.data.length; i += 4) {
        // If mask says background, set pixel alpha to 0
        if (maskData[i + 3] < 128) {
          imageData.data[i + 3] = 0;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    }

    return canvas.toDataURL('image/png');
  } catch (error) {
    console.error('[AI] Background removal error:', error);
    return captureFrame(videoElement);
  }
}

/**
 * Capture a video frame as JPEG data URL (compressed)
 */
function captureFrame(videoElement, quality) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality || 0.8);
}

/**
 * Detect pose keypoints from video
 * @param {HTMLVideoElement} videoElement
 * @returns {Array} keypoints [{name, x, y, score}, ...]
 */
async function getPose(videoElement) {
  if (!poseDetector) return cachedPose;

  try {
    const poses = await poseDetector.estimatePoses(videoElement, {
      flipHorizontal: false
    });

    if (poses && poses.length > 0 && poses[0].keypoints) {
      cachedPose = poses[0].keypoints;
      return cachedPose;
    }
  } catch (error) {
    console.error('[AI] Pose detection error:', error);
  }

  // Return cached pose if current detection failed
  return cachedPose;
}

/**
 * Detect face in video
 * @param {HTMLVideoElement} videoElement
 * @returns {Object|null} { box: { xMin, yMin, width, height } }
 */
async function getFace(videoElement) {
  if (!faceDetector) return null;

  try {
    const faces = await faceDetector.estimateFaces(videoElement, {
      flipHorizontal: false
    });

    if (faces && faces.length > 0) {
      const face = faces[0];
      lastFaceData = {
        box: face.box || {
          xMin: face.boundingBox ? face.boundingBox.topLeft[0] : 0,
          yMin: face.boundingBox ? face.boundingBox.topLeft[1] : 0,
          width: face.boundingBox ? (face.boundingBox.bottomRight[0] - face.boundingBox.topLeft[0]) : 100,
          height: face.boundingBox ? (face.boundingBox.bottomRight[1] - face.boundingBox.topLeft[1]) : 100
        }
      };
      lastFaceTimestamp = Date.now();
      return lastFaceData;
    }
  } catch (error) {
    console.error('[AI] Face detection error:', error);
  }

  // Return cached face data if fresh (< 2 seconds)
  if (lastFaceData && (Date.now() - lastFaceTimestamp) < 2000) {
    return lastFaceData;
  }
  return null;
}

/**
 * Draw AR garment overlay on canvas using pose keypoints
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} canvasW
 * @param {number} canvasH
 * @param {HTMLImageElement} garmentImg
 * @param {Array} keypoints - pose keypoints array
 */
function drawAROverlay(ctx, canvasW, canvasH, garmentImg, keypoints) {
  if (!keypoints || !garmentImg) return;

  // Extract keypoints
  const leftShoulder = keypoints.find(k => k.name === 'left_shoulder') || keypoints[5];
  const rightShoulder = keypoints.find(k => k.name === 'right_shoulder') || keypoints[6];
  const leftHip = keypoints.find(k => k.name === 'left_hip') || keypoints[11];
  const rightHip = keypoints.find(k => k.name === 'right_hip') || keypoints[12];

  if (!leftShoulder || !rightShoulder) return;

  // Validate confidence
  if ((leftShoulder.score || 0) < 0.25 || (rightShoulder.score || 0) < 0.25) return;

  // Calculate shoulder width
  const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
  if (shoulderWidth < 10) return; // Too small, skip

  // Calculate garment placement
  const x = Math.min(leftShoulder.x, rightShoulder.x) - (shoulderWidth * 0.15);
  const y = Math.min(leftShoulder.y, rightShoulder.y) - (shoulderWidth * 0.08);
  const width = shoulderWidth * 1.4;

  let height;
  if (leftHip && rightHip && (leftHip.score || 0) > 0.2 && (rightHip.score || 0) > 0.2) {
    const hipY = (leftHip.y + rightHip.y) / 2;
    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    height = (hipY - shoulderY) * 1.2;
  } else {
    height = width * 1.3;
  }

  // Draw garment with alpha blending
  ctx.clearRect(0, 0, canvasW, canvasH);
  ctx.globalAlpha = 0.85;
  ctx.drawImage(garmentImg, x, y, width, height);
  ctx.globalAlpha = 1.0;
}

/**
 * Draw mannequin silhouette with face swap and garment overlay
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLImageElement} garmentImg
 * @param {HTMLVideoElement} buyerVideoElement
 * @param {Object|null} faceData - { box: { xMin, yMin, width, height } }
 */
function drawMannequinFaceSwap(canvas, garmentImg, buyerVideoElement, faceData) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;

  // Clear canvas
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);

  const bodyColor = '#2a2a3a';
  const skinColor = '#3a3a4a';

  // Head (circle)
  const headRadius = 35;
  const headCy = 70;
  ctx.fillStyle = skinColor;
  ctx.beginPath();
  ctx.arc(cx, headCy, headRadius, 0, Math.PI * 2);
  ctx.fill();

  // Neck
  ctx.fillStyle = skinColor;
  ctx.fillRect(cx - 10, headCy + headRadius, 20, 15);

  const neckBottom = headCy + headRadius + 15;

  // Torso (trapezoid — wider at shoulders, narrower at hips)
  const shoulderW = 80;
  const hipW = 60;
  const torsoH = 170;
  ctx.fillStyle = bodyColor;
  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, neckBottom);
  ctx.lineTo(cx + shoulderW, neckBottom);
  ctx.lineTo(cx + hipW, neckBottom + torsoH);
  ctx.lineTo(cx - hipW, neckBottom + torsoH);
  ctx.closePath();
  ctx.fill();

  // Arms
  ctx.fillStyle = skinColor;
  // Left arm
  ctx.beginPath();
  ctx.moveTo(cx - shoulderW, neckBottom);
  ctx.lineTo(cx - shoulderW - 25, neckBottom + 10);
  ctx.lineTo(cx - shoulderW - 30, neckBottom + 130);
  ctx.lineTo(cx - shoulderW - 10, neckBottom + 130);
  ctx.lineTo(cx - shoulderW, neckBottom + 30);
  ctx.closePath();
  ctx.fill();

  // Right arm
  ctx.beginPath();
  ctx.moveTo(cx + shoulderW, neckBottom);
  ctx.lineTo(cx + shoulderW + 25, neckBottom + 10);
  ctx.lineTo(cx + shoulderW + 30, neckBottom + 130);
  ctx.lineTo(cx + shoulderW + 10, neckBottom + 130);
  ctx.lineTo(cx + shoulderW, neckBottom + 30);
  ctx.closePath();
  ctx.fill();

  // Legs
  const legTop = neckBottom + torsoH;
  ctx.fillStyle = bodyColor;
  // Left leg
  ctx.fillRect(cx - hipW + 5, legTop, 30, 160);
  // Right leg
  ctx.fillRect(cx + hipW - 35, legTop, 30, 160);

  // Face swap — draw buyer's face on the mannequin head
  if (faceData && buyerVideoElement && buyerVideoElement.videoWidth > 0) {
    const box = faceData.box;

    // Create offscreen canvas for face crop
    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = buyerVideoElement.videoWidth;
    faceCanvas.height = buyerVideoElement.videoHeight;
    const faceCtx = faceCanvas.getContext('2d');
    faceCtx.drawImage(buyerVideoElement, 0, 0);

    // Crop face region
    const cropX = box.xMin || 0;
    const cropY = box.yMin || 0;
    const cropW = box.width || 100;
    const cropH = box.height || 100;

    // Clip to circle and draw face
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, headCy, headRadius - 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(
      faceCanvas,
      cropX, cropY, cropW, cropH,
      cx - headRadius, headCy - headRadius,
      headRadius * 2, headRadius * 2
    );
    ctx.restore();
  }

  // Overlay garment on torso
  if (garmentImg) {
    const garmentW = 140;
    const garmentH = 170;
    ctx.globalAlpha = 0.9;
    ctx.drawImage(garmentImg, cx - garmentW / 2, neckBottom, garmentW, garmentH);
    ctx.globalAlpha = 1.0;
  }
}

/**
 * Draw pose skeleton overlay on canvas
 */
function drawPoseSkeleton(ctx, keypoints, canvasW, canvasH) {
  if (!keypoints) return;

  const connections = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15],
    [12, 14], [14, 16]
  ];

  // Draw connections
  ctx.strokeStyle = 'rgba(124, 111, 247, 0.6)';
  ctx.lineWidth = 2;
  connections.forEach(([i, j]) => {
    const kpA = keypoints[i];
    const kpB = keypoints[j];
    if (kpA && kpB && (kpA.score || 0) > 0.2 && (kpB.score || 0) > 0.2) {
      ctx.beginPath();
      ctx.moveTo(kpA.x, kpA.y);
      ctx.lineTo(kpB.x, kpB.y);
      ctx.stroke();
    }
  });

  // Draw keypoints
  keypoints.forEach(kp => {
    if ((kp.score || 0) > 0.2) {
      ctx.fillStyle = 'rgba(124, 111, 247, 0.9)';
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/**
 * Check if all models are loaded
 */
function areModelsLoaded() {
  return modelsLoaded;
}

// Make available globally
window.FVAI = {
  loadModels,
  removeBackground,
  captureFrame,
  getPose,
  getFace,
  drawAROverlay,
  drawMannequinFaceSwap,
  drawPoseSkeleton,
  areModelsLoaded
};
