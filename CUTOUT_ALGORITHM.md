# 🧠 Clothing Cutout Algorithm — Technical Deep Dive

> **Precision AR Neural Shift v4.0**
> Real-time clothing extraction, segmentation, and spatial transfer between users.

---

## Tech Stack Overview

| Feature / Stage | Technology | Why This Tech |
|---|---|---|
| **Body Pose Detection** | [MediaPipe Pose](https://cdn.jsdelivr.net/npm/@mediapipe/pose) | 33-landmark full-body skeletal tracking at 30 fps; runs client-side via WASM+GPU |
| **Camera Stream** | [MediaPipe Camera Utils](https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils) | Handles `getUserMedia` lifecycle, frame-by-frame async dispatch to the pose model |
| **Segmentation Mask** | MediaPipe Pose (`enableSegmentation: true`) | Per-pixel body vs. background mask generated alongside landmarks — no extra model needed |
| **Clothing Texture Capture** | HTML5 Canvas 2D (`CanvasRenderingContext2D`) | Pixel-level compositing: `destination-in` for masking, `destination-out` for head removal |
| **Skin Detection & Removal** | Custom YCbCr Chromaticity Sampler (JS) | Luminance-invariant skin-tone matching with anti-aliased feathered edges |
| **3D Mesh Overlay** | [Three.js r128](https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js) | Orthographic camera + 32×32 subdivided `PlaneGeometry` for per-vertex skeletal warping |
| **Live Vertex Warping** | Three.js Geometry Attributes (position buffer) | Direct manipulation of vertex positions every frame for real-time pose tracking |
| **UV Mapping** | Custom bilinear UV bake (JS) | Maps captured clothing texture onto the deformable quad using shoulder/hip landmarks |
| **Head/Neck Erasure** | Radial gradient brush via Canvas 2D | Soft oval gradient centered on nose landmark, vertically stretched to cover neck |
| **Bounding Box Extraction** | Manual pixel scan of segmentation mask | Scans mask at 4px stride to find min/max body bounds, adds 15px padding |
| **Rendering Pipeline** | Three.js `WebGLRenderer` (alpha: true) | Transparent overlay composited on top of live `<video>` feed via CSS stacking |
| **Webcam Feed** | HTML5 `<video>` element | Full-viewport mirrored feed (`scaleX(-1)`) as the visual background layer |

---

## Algorithm Pipeline — Stage by Stage

### Stage 0: Initialization & Configuration

```
MediaPipe Pose Options:
  modelComplexity   : 2        ← Highest accuracy (heavy model)
  smoothLandmarks   : true     ← Temporal jitter reduction
  enableSegmentation: true     ← Generates body silhouette mask
  smoothSegmentation: true     ← Removes mask flicker
  minDetectionConf  : 0.7      ← Only accept high-confidence detections
  minTrackingConf   : 0.7      ← Only accept high-confidence tracking
  
Camera Resolution: 1280 × 720
Three.js Camera  : OrthographicCamera (world units: 10 × 10)
Mesh Subdivision : 32 × 32 (1089 vertices)
```

---

### Stage 1: Segmentation Mask Bounding Box Scan

**Purpose:** Find the tight bounding rectangle around the user's body silhouette.

**Input:**
- `results.segmentationMask` — a canvas/image where body pixels are non-transparent

**Algorithm:**

```
for y = 0 → height (step 4):
  for x = 0 → width (step 4):
    pixel = maskPixels[(y * width + x) * 4]
    if pixel.r > 50 OR pixel.a > 50:
      update minX, maxX, minY, maxY
```

**Coordinate Tracking:**

| Variable | Description | Unit |
|---|---|---|
| `minX` | Left edge of body silhouette | px (video space) |
| `maxX` | Right edge of body silhouette | px (video space) |
| `minY` | Top edge of body silhouette | px (video space) |
| `maxY` | Bottom edge of body silhouette | px (video space) |
| `pad` | Bounding box padding | 15 px |
| `boxW` | `maxX - minX` (after padding) | px |
| `boxH` | `maxY - minY` (after padding) | px |

> **Stride = 4px** for performance. Pixel check uses RGBA channel `R > 50 || A > 50`.

---

### Stage 2: Clothing Texture Extraction (Masked Crop)

**Purpose:** Cut the clothing region from the live video frame, removing the background.

**Steps:**
1. Draw the video frame cropped to `[minX, minY, boxW, boxH]` onto a capture canvas
2. Apply CSS-like filter: `contrast(1.1) saturate(1.1) brightness(1.05)` for enhanced fabric colors
3. Use `globalCompositeOperation = 'destination-in'` with the segmentation mask to erase all non-body pixels

**Coordinate Mapping:**

```
Source (video):  drawImage(video, minX, minY, boxW, boxH, ...)
Mask composite:  drawImage(segMask, minX, minY, boxW, boxH, ...)
Result canvas:   [0, 0, boxW, boxH]  (local pixel space)
```

---

### Stage 3: Head & Neck Removal

**Purpose:** Erase the head and neck from the clothing cutout so only torso clothing remains.

**Method:** Soft radial gradient brush using `destination-out` compositing.

**Coordinate Tracking:**

| Variable | Formula | Description |
|---|---|---|
| `noseX` | `lm[0].x × videoWidth − minX` | Nose X in local cutout space |
| `noseY` | `lm[0].y × videoHeight − minY` | Nose Y in local cutout space |
| `faceRadius` | `hypot(lm[11].x − lm[12].x, lm[11].y − lm[12].y) × videoWidth × 0.55` | Scaled from shoulder width |

**Gradient Configuration:**

```
Center:       (noseX, noseY + faceRadius × 0.2)   ← shifted below nose
Vertical Scale: 1.35                                ← stretched into oval for neck
Inner Stop:   r = faceRadius × 0.2  → alpha = 1.0  (fully erased)
Outer Stop:   r = faceRadius         → alpha = 0.0  (no erasure)
```

**Landmarks Used:**

| Index | Landmark | Role |
|---|---|---|
| 0 | Nose | Center point for gradient brush |
| 11 | Left Shoulder | Shoulder-width measurement (radius calc) |
| 12 | Right Shoulder | Shoulder-width measurement (radius calc) |

---

### Stage 4: Skin Tone Detection & Removal (YCbCr Chromaticity)

**Purpose:** Remove exposed skin (arms, hands, décolletage) from the clothing cutout using chrominance-based matching.

#### 4a. Skin Sampling

Sample actual skin color from the user's body at known skin landmarks:

| Index | Landmark | Condition |
|---|---|---|
| 0 | Nose | `visibility > 0.5`, `r > 40`, `r > b` |
| 15 | Left Wrist | `visibility > 0.5`, `r > 40`, `r > b` |
| 16 | Right Wrist | `visibility > 0.5`, `r > 40`, `r > b` |

#### 4b. RGB → YCbCr Conversion

```
Y  =  0.299 × R + 0.587 × G + 0.114 × B
Cb = 128 − 0.168736 × R − 0.331264 × G + 0.5 × B
Cr = 128 + 0.5 × R − 0.418688 × G − 0.081312 × B
```

> YCbCr separates luminance (Y) from chrominance (Cb, Cr), making skin matching robust under varying lighting conditions.

#### 4c. Pixel-by-Pixel Skin Matching

For every pixel in the clothing cutout:

```
chromaDist = hypot(pixel.Cb − sample.Cb, pixel.Cr − sample.Cr)
luminanceRatio = pixel.Y / sample.Y

if luminanceRatio ∈ [0.30, 1.40]:    ← generous luminance tolerance
  if chromaDist < 10:
    alpha = 0                          ← pure skin → fully erased
  else if chromaDist < 25:
    fade = (chromaDist − 10) / 15
    alpha = alpha × fade²             ← exponential anti-aliased feather
  else:
    keep pixel as-is                   ← clothing → preserved
```

**Skip conditions:**
- `alpha === 0` → already transparent
- `R < 25 && G < 25 && B < 25` → very dark pixels (likely clothing, not skin)

---

### Stage 5: Skeletal Quad — Mesh Geometry Construction

**Purpose:** Define a deformable quadrilateral anchored to the user's torso skeleton for organic clothing overlay.

#### Quad Vertices (Normalized Coordinates, 0.0–1.0)

| Corner | Derived From | Margin Expansion |
|---|---|---|
| **Top-Right** | `rShoulder` − shoulderVec × 1.0 − spineVecR × 0.5 | Side: 1.0×, Up: 0.5× |
| **Top-Left** | `lShoulder` + shoulderVec × 1.0 − spineVecL × 0.5 | Side: 1.0×, Up: 0.5× |
| **Bottom-Right** | `rHip` − shoulderVec × 1.0 + spineVecR × 0.5 | Side: 1.0×, Down: 0.5× |
| **Bottom-Left** | `lHip` + shoulderVec × 1.0 + spineVecL × 0.5 | Side: 1.0×, Down: 0.5× |

#### Key Vectors

| Vector | Formula | Purpose |
|---|---|---|
| `shoulderVec` | `lShoulder − rShoulder` | Horizontal span of torso |
| `spineVecL` | `lHip − lShoulder` | Left side of torso length |
| `spineVecR` | `rHip − rShoulder` | Right side of torso length |

#### Landmarks Used for Quad

| Index | Landmark | Role |
|---|---|---|
| 11 | Left Shoulder | Top-left anchor + shoulder vector |
| 12 | Right Shoulder | Top-right anchor + shoulder vector |
| 23 | Left Hip | Bottom-left anchor + spine vector |
| 24 | Right Hip | Bottom-right anchor + spine vector |

---

### Stage 6: UV Mapping — Texture Bake onto 32×32 Mesh

**Purpose:** Map the captured clothing image (from Stage 2–4) onto the deformable mesh using bilinear interpolation.

**Algorithm (per vertex):**

```
for i = 0 → 32 (rows):
  facY = i / 32.0
  for j = 0 → 32 (cols):
    u = j / 32.0

    // Bilinear interpolation across quad corners
    topX = quad[0].x × (1−u) + quad[1].x × u
    topY = quad[0].y × (1−u) + quad[1].y × u
    botX = quad[2].x × (1−u) + quad[3].x × u
    botY = quad[2].y × (1−u) + quad[3].y × u

    px = topX × (1−facY) + botX × facY
    py = topY × (1−facY) + botY × facY

    // Convert to texture coordinates
    textureU = (px × videoWidth − minX) / boxW
    textureV = 1.0 − ((py × videoHeight − minY) / boxH)
```

> This maps each mesh vertex to the exact pixel location in the clothing cutout texture.

---

### Stage 7: Live Spatial Alignment (Real-Time Warping)

**Purpose:** Every frame, warp the mesh vertices to follow the target user's body pose.

#### When Clothing is Locked (`isLocked = true`):

Per-vertex world-space positioning using the same quad construction:

```
for each vertex (i, j):
    px, py = bilinear interpolation of quad corners
    worldX = (px − 0.5) × 10 × aspectRatio
    worldY = (0.5 − py) × 10

    vertex.position = (worldX, worldY, 0)
```

**Depth adjustment:** `suitMesh.position.z = −avgDepth × 2` where `avgDepth = (lm[11].z + lm[12].z) / 2`

#### Before Lock (Preview Mode):

| Property | Formula | Description |
|---|---|---|
| `position.x` | `(centerX − 0.5) × 10 × aspect` | Horizontal center between shoulders and hips |
| `position.y` | `(0.5 − centerY) × 10` | Vertical center between shoulders and hips |
| `scale.x` | `|lm[11].x − lm[12].x| × 16 × aspect` | Shoulder width |
| `scale.y` | `|midShoulderY − midHipY| × 20` | Torso height |
| `rotation.z` | `−(atan2(dy, dx) + π/2)` | Spine tilt angle |

---

## Coordinate Space Reference

| Space | Origin | Range | Used By |
|---|---|---|---|
| **Normalized Landmark** | Top-Left | `[0.0, 1.0]` for X, Y; Z is depth in meters | MediaPipe Pose |
| **Video Pixel** | Top-Left | `[0, 1280] × [0, 720]` | Canvas 2D drawing |
| **Cutout Local** | Top-Left of bounding box | `[0, boxW] × [0, boxH]` | Texture extraction |
| **Three.js World** | Center | `[-5×aspect, 5×aspect] × [-5, 5]` | Orthographic camera |
| **UV Texture** | Bottom-Left | `[0.0, 1.0]` (V flipped) | Mesh UV attributes |

---

## Full MediaPipe Landmark Index Reference (Used in Algorithm)

| Index | Landmark | Algorithm Usage |
|---|---|---|
| **0** | Nose | Head erasure center, skin color sampling |
| **11** | Left Shoulder | Quad anchor, shoulder vector, depth calc |
| **12** | Right Shoulder | Quad anchor, shoulder vector, depth calc |
| **15** | Left Wrist | Skin color sampling |
| **16** | Right Wrist | Skin color sampling |
| **23** | Left Hip | Quad anchor, spine vector |
| **24** | Right Hip | Quad anchor, spine vector |

---

## Performance Notes

| Optimization | Detail |
|---|---|
| Mask scan stride | 4px skip → ~16× fewer pixels checked |
| Mesh resolution | 32×32 = 1089 vertices — balances detail vs. GPU cost |
| Skin skip (dark pixels) | `R<25 && G<25 && B<25` bypassed — avoids false skin matches on dark clothing |
| Model complexity 2 | Highest accuracy but heaviest — suitable for desktop/modern mobile |
| Canvas filter pipeline | Single-pass `contrast + saturate + brightness` applied before compositing |
