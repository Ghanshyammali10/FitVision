// ==========================================
// FitVision — AI Module
// TensorFlow.js: Body Segmentation, Pose Detection, Face Detection
// ==========================================

let segmenter = null;
let poseDetector = null;
let faceDetector = null;
let modelsLoaded = false;

/**
 * Load all AI models with progress callback
 */
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
        {
          runtime: 'tfjs',
          modelType: 'general'
        }
      );
      report('Body Segmentation', 'Ready ✓', 35);
    } catch (e) {
      console.warn('Body segmentation failed to load:', e.message);
      report('Body Segmentation', 'Skipped (will use fallback)', 35);
    }

    // 2. Pose Detection (for AR overlay)
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
      console.warn('Pose detection failed to load:', e.message);
      report('Pose Detection', 'Skipped', 70);
    }

    // 3. Face Detection (for mannequin face swap)
    report('Face Detection', 'Loading...', 75);
    try {
      faceDetector = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        {
          runtime: 'tfjs',
          maxFaces: 1
        }
      );
      report('Face Detection', 'Ready ✓', 100);
    } catch (e) {
      console.warn('Face detection failed to load:', e.message);
      report('Face Detection', 'Skipped', 100);
    }

    modelsLoaded = true;
    report('All Models', 'Ready!', 100);
  } catch (error) {
    console.error('Model loading error:', error);
    modelsLoaded = true; // Continue anyway with fallbacks
    report('Models', 'Loaded with fallbacks', 100);
  }
}

/**
 * Remove background from a video frame
 * Returns a data URL of the person with transparent background
 */
async function removeBackground(videoElement) {
  if (!segmenter) return fallbackCapture(videoElement);

  try {
    const segmentation = await segmenter.segmentPeople(videoElement, {
      flipHorizontal: false,
      multiSegmentation: false,
      segmentBodyParts: false
    });

    if (!segmentation || segmentation.length === 0) {
      return fallbackCapture(videoElement);
    }

    const canvas = document.createElement('canvas');
    const w = videoElement.videoWidth || 640;
    const h = videoElement.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Draw video frame
    ctx.drawImage(videoElement, 0, 0, w, h);
    const frame = ctx.getImageData(0, 0, w, h);

    // Get mask
    const mask = await segmentation[0].mask.toImageData();

    // Apply mask — keep only person pixels
    for (let i = 0; i < frame.data.length; i += 4) {
      const maskIdx = i;
      // MediaPipe mask: 255 = person, 0 = background
      if (mask.data[maskIdx] < 128) {
        frame.data[i + 3] = 0; // Set alpha to 0 (transparent)
      }
    }

    ctx.putImageData(frame, 0, 0);
    return canvas.toDataURL('image/png');
  } catch (error) {
    console.error('Background removal error:', error);
    return fallbackCapture(videoElement);
  }
}

/**
 * Fallback: capture frame without background removal
 */
function fallbackCapture(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.8);
}

/**
 * Detect pose keypoints from video
 */
async function getPose(videoElement) {
  if (!poseDetector) return null;
  try {
    const poses = await poseDetector.estimatePoses(videoElement, {
      flipHorizontal: false
    });
    if (poses.length > 0 && poses[0].keypoints) {
      return poses[0].keypoints;
    }
    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Draw AR garment overlay based on pose keypoints
 */
function drawAROverlay(ctx, canvasWidth, canvasHeight, garmentImg, keypoints) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  if (!keypoints || !garmentImg) return;

  // MoveNet keypoint indices
  const LEFT_SHOULDER = keypoints.find(k => k.name === 'left_shoulder');
  const RIGHT_SHOULDER = keypoints.find(k => k.name === 'right_shoulder');
  const LEFT_HIP = keypoints.find(k => k.name === 'left_hip');
  const RIGHT_HIP = keypoints.find(k => k.name === 'right_hip');

  if (!LEFT_SHOULDER || !RIGHT_SHOULDER) return;
  if (LEFT_SHOULDER.score < 0.3 || RIGHT_SHOULDER.score < 0.3) return;

  // Calculate garment dimensions from body proportions
  const shoulderWidth = Math.abs(RIGHT_SHOULDER.x - LEFT_SHOULDER.x);
  const garmentWidth = shoulderWidth * 1.6; // Slightly wider than shoulders

  // Height: use shoulder-to-hip distance if available, else estimate
  let garmentHeight;
  if (LEFT_HIP && RIGHT_HIP && LEFT_HIP.score > 0.3) {
    const hipY = (LEFT_HIP.y + RIGHT_HIP.y) / 2;
    const shoulderY = (LEFT_SHOULDER.y + RIGHT_SHOULDER.y) / 2;
    garmentHeight = (hipY - shoulderY) * 1.3;
  } else {
    garmentHeight = garmentWidth * 1.2;
  }

  // Center position
  const centerX = (LEFT_SHOULDER.x + RIGHT_SHOULDER.x) / 2;
  const centerY = (LEFT_SHOULDER.y + RIGHT_SHOULDER.y) / 2;

  const x = centerX - garmentWidth / 2;
  const y = centerY - garmentHeight * 0.1; // Slightly above shoulders

  // Draw with transparency
  ctx.save();
  ctx.globalAlpha = 0.85;
  ctx.drawImage(garmentImg, x, y, garmentWidth, garmentHeight);
  ctx.restore();
}

/**
 * Detect face bounding box
 */
async function getFace(videoElement) {
  if (!faceDetector) return null;
  try {
    const faces = await faceDetector.estimateFaces(videoElement, { flipHorizontal: false });
    if (faces.length > 0) {
      return faces[0].box; // { xMin, yMin, width, height }
    }
    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Draw mannequin with face swap and garment overlay
 */
function drawMannequinFaceSwap(canvas, garmentImg, videoElement, faceBox) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Clear + draw mannequin body silhouette
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);

  // Draw mannequin silhouette
  ctx.save();
  ctx.fillStyle = '#2a2a3a';
  // Head circle
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.12, 35, 42, 0, 0, Math.PI * 2);
  ctx.fill();
  // Neck
  ctx.fillRect(w / 2 - 12, h * 0.16, 24, 25);
  // Torso
  ctx.beginPath();
  ctx.moveTo(w / 2 - 65, h * 0.21);
  ctx.lineTo(w / 2 + 65, h * 0.21);
  ctx.lineTo(w / 2 + 55, h * 0.6);
  ctx.lineTo(w / 2 - 55, h * 0.6);
  ctx.closePath();
  ctx.fill();
  // Legs
  ctx.fillRect(w / 2 - 45, h * 0.6, 35, h * 0.38);
  ctx.fillRect(w / 2 + 10, h * 0.6, 35, h * 0.38);
  ctx.restore();

  // Draw garment on torso
  if (garmentImg) {
    const gw = 140;
    const gh = gw * (garmentImg.height / garmentImg.width);
    const gx = w / 2 - gw / 2;
    const gy = h * 0.2;
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.drawImage(garmentImg, gx, gy, gw, Math.min(gh, h * 0.42));
    ctx.restore();
  }

  // Draw buyer's face on mannequin head
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

      // Draw face as circle on mannequin head
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(w / 2, h * 0.12, 33, 40, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(faceCanvas, w / 2 - 40, h * 0.12 - 45, 80, 90);
      ctx.restore();
    } catch (e) {
      // Face draw failed, skip
    }
  }

  // Draw "Mannequin Preview" label
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
  loadModels, removeBackground, getPose,
  drawAROverlay, getFace, drawMannequinFaceSwap,
  isModelLoaded
};
