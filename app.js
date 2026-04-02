const state = { images: [] };

const dom = {
  imageInput: document.getElementById("imageInput"),
  thumbStrip: document.getElementById("thumbStrip"),
  statusText: document.getElementById("statusText"),
  runButton: document.getElementById("runButton"),
  metricGrid: document.getElementById("metricGrid"),
  gaussianSize: document.getElementById("gaussianSize"),
  gaussianSigma: document.getElementById("gaussianSigma"),
  lowThreshold: document.getElementById("lowThreshold"),
  highThreshold: document.getElementById("highThreshold"),
  harrisPercentile: document.getElementById("harrisPercentile"),
  siftThreshold: document.getElementById("siftThreshold"),
  maxKeypoints: document.getElementById("maxKeypoints"),
  ransacThreshold: document.getElementById("ransacThreshold"),
  ransacIterations: document.getElementById("ransacIterations"),
  blendingMode: document.getElementById("blendingMode"),
};

const canvasIds = [
  "edgeOriginalCanvas",
  "gradientCanvas",
  "nmsCanvas",
  "cannyCanvas",
  "harrisCanvas",
  "siftCanvas",
  "pairFeaturesCanvas",
  "initialMatchesCanvas",
  "ransacCanvas",
  "alignedCanvas",
  "panoramaCanvas",
  "blendCompareCanvas",
];

const canvases = Object.fromEntries(canvasIds.map((id) => [id, document.getElementById(id)]));

dom.imageInput.addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) {
    return;
  }
  state.images = await Promise.all(files.map(loadImageData));
  renderThumbnails();
  setStatus(`Loaded ${state.images.length} image(s). Click Run to start.`);
});

dom.runButton.addEventListener("click", async () => {
  if (!state.images.length) {
    setStatus("Please upload at least one image.");
    return;
  }

  setStatus("Running image analysis...");
  await new Promise((resolve) => setTimeout(resolve, 30));

  try {
    const params = readParams();
    const single = preprocessImage(state.images[0], params);
    renderSingleImageResults(single);

    let pairResult = null;
    if (state.images.length >= 2) {
      pairResult = runPairPipeline(state.images[0], state.images[1], params);
      renderPairResult(pairResult);
    } else {
      clearPairCanvases();
    }

    if (state.images.length >= 3) {
      const pano = runPanorama(state.images, params);
      renderPanorama(pano);
    } else {
      clearCanvas(canvases.panoramaCanvas);
      clearCanvas(canvases.blendCompareCanvas);
    }

    renderMetrics(single, pairResult);
  } catch (error) {
    console.error(error);
    setStatus(`Run failed: ${error.message}`);
  }
});

window.addEventListener("load", async () => {
  try {
    const sample = await loadImageFromUrl("./A1.jpg", "A1.jpg");
    state.images = [sample];
    renderThumbnails();
    setStatus("Auto-loaded A1.jpg. Running analysis...");
    await new Promise((resolve) => setTimeout(resolve, 30));
    dom.runButton.click();
  } catch (error) {
    setStatus("Waiting for images. Upload files manually if A1.jpg is not auto-loaded.");
  }
});

function setStatus(text) {
  dom.statusText.textContent = text;
}

function readParams() {
  return {
    gaussianSize: Number(dom.gaussianSize.value),
    gaussianSigma: Number(dom.gaussianSigma.value),
    lowThreshold: Number(dom.lowThreshold.value),
    highThreshold: Number(dom.highThreshold.value),
    harrisPercentile: Number(dom.harrisPercentile.value),
    siftThreshold: Number(dom.siftThreshold.value),
    maxKeypoints: Number(dom.maxKeypoints.value),
    ransacThreshold: Number(dom.ransacThreshold.value),
    ransacIterations: Number(dom.ransacIterations.value),
    blendingMode: dom.blendingMode.value,
  };
}

function renderThumbnails() {
  dom.thumbStrip.innerHTML = "";
  state.images.forEach((item, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "thumb-item";
    const img = document.createElement("img");
    img.src = item.url;
    img.alt = item.name;
    const info = document.createElement("div");
    info.innerHTML = `<strong>Image ${index + 1}</strong><div>${item.name}</div><div>${item.width} x ${item.height}</div>`;
    wrapper.append(img, info);
    dom.thumbStrip.appendChild(wrapper);
  });
}

async function loadImageData(file) {
  const url = URL.createObjectURL(file);
  const bitmap = await createImageBitmap(file);
  const maxSide = 900;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(32, Math.round(bitmap.width * scale));
  const height = Math.max(32, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, width, height);
  return {
    name: file.name,
    url,
    width,
    height,
    imageData: ctx.getImageData(0, 0, width, height),
  };
}

async function loadImageFromUrl(url, name) {
  const image = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Cannot load ${name}`));
    img.src = url;
  });
  const maxSide = 900;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(32, Math.round(image.naturalWidth * scale));
  const height = Math.max(32, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, width, height);
  return {
    name,
    url,
    width,
    height,
    imageData: ctx.getImageData(0, 0, width, height),
  };
}

function preprocessImage(item, params) {
  const gray = rgbaToGray(item.imageData);
  const blurred = gaussianBlur(gray, item.width, item.height, params.gaussianSize, params.gaussianSigma);
  const gradients = sobelGradients(blurred, item.width, item.height);
  const nms = nonMaximumSuppression(gradients.magnitude, gradients.direction, item.width, item.height);
  const canny = hysteresisThreshold(nms, item.width, item.height, params.lowThreshold, params.highThreshold);
  const harris = detectHarris(blurred, item.width, item.height, params);
  const sift = detectSiftLike(blurred, item.width, item.height, params);
  return { item, gray, blurred, gradients, nms, canny, harris, sift };
}

function renderSingleImageResults(result) {
  drawImageData(canvases.edgeOriginalCanvas, result.item.imageData);
  drawGrayMap(canvases.gradientCanvas, result.gradients.magnitude, result.item.width, result.item.height, true);
  drawGrayMap(canvases.nmsCanvas, result.nms, result.item.width, result.item.height, true);
  drawBinaryMap(canvases.cannyCanvas, result.canny, result.item.width, result.item.height);
  drawKeypoints(canvases.harrisCanvas, result.item.imageData, result.harris.keypoints, "#d1495b");
  drawKeypoints(canvases.siftCanvas, result.item.imageData, result.sift.keypoints, "#00798c");
}

function renderMetrics(single, pairResult) {
  const edgeCount = single.canny.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0);
  const cards = [
    { label: "Images", value: state.images.length },
    { label: "Edge Pixels", value: edgeCount },
    { label: "Harris Points", value: single.harris.keypoints.length },
    { label: "SIFT-like Points", value: single.sift.keypoints.length },
  ];
  if (pairResult) {
    cards.push({ label: "Initial Matches", value: pairResult.matches.length });
    cards.push({ label: "RANSAC Inliers", value: pairResult.inliers.length });
  }
  dom.metricGrid.innerHTML = cards
    .map((card) => `<div class="metric-card"><div>${card.label}</div><strong>${card.value}</strong></div>`)
    .join("");
  setStatus("Analysis finished. You can tune parameters and rerun.");
}

function clearPairCanvases() {
  [
    canvases.pairFeaturesCanvas,
    canvases.initialMatchesCanvas,
    canvases.ransacCanvas,
    canvases.alignedCanvas,
  ].forEach(clearCanvas);
}

function runPairPipeline(itemA, itemB, params) {
  const prepA = preprocessImage(itemA, params);
  const prepB = preprocessImage(itemB, params);
  const matches = matchDescriptors(prepA.sift.keypoints, prepB.sift.keypoints);
  const ransac = estimateHomographyRansac(matches, params.ransacIterations, params.ransacThreshold);
  const aligned = ransac.homography ? warpTwoImages(itemA.imageData, itemB.imageData, ransac.homography, "average") : null;
  return { prepA, prepB, matches, inliers: ransac.inliers, homography: ransac.homography, aligned };
}

function renderPairResult(result) {
  drawSideBySideFeatures(
    canvases.pairFeaturesCanvas,
    result.prepA.item.imageData,
    result.prepB.item.imageData,
    result.prepA.sift.keypoints,
    result.prepB.sift.keypoints
  );
  drawMatches(canvases.initialMatchesCanvas, result.prepA.item.imageData, result.prepB.item.imageData, result.matches.slice(0, 60), "#ef476f");
  drawMatches(canvases.ransacCanvas, result.prepA.item.imageData, result.prepB.item.imageData, result.inliers.slice(0, 80), "#06d6a0");
  if (result.aligned) {
    drawImageData(canvases.alignedCanvas, result.aligned.imageData);
  } else {
    clearCanvas(canvases.alignedCanvas);
  }
}

function runPanorama(items, params) {
  const preps = items.map((item) => preprocessImage(item, params));
  const centerIndex = Math.floor(items.length / 2);
  const transforms = items.map(() => identityMatrix3());

  for (let i = centerIndex - 1; i >= 0; i--) {
    const h = buildPairTransform(preps[i], preps[i + 1], params);
    transforms[i] = multiplyMatrix3(transforms[i + 1], h || identityMatrix3());
  }

  for (let i = centerIndex + 1; i < items.length; i++) {
    const h = buildPairTransform(preps[i], preps[i - 1], params);
    transforms[i] = multiplyMatrix3(transforms[i - 1], h || identityMatrix3());
  }

  const overlay = composePanorama(items, transforms, "overlay");
  const average = composePanorama(items, transforms, "average");
  const linear = composePanorama(items, transforms, "linear");
  const finalMap = params.blendingMode === "overlay" ? overlay : params.blendingMode === "linear" ? linear : average;
  return { overlay, average, linear, finalMap };
}

function renderPanorama(result) {
  drawImageData(canvases.panoramaCanvas, result.finalMap.imageData);
  const canvas = canvases.blendCompareCanvas;
  const topWidth = result.overlay.imageData.width + result.average.imageData.width;
  const bottomWidth = result.linear.imageData.width;
  const width = Math.max(topWidth, bottomWidth);
  const height = result.overlay.imageData.height + result.linear.imageData.height;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f5eee5";
  ctx.fillRect(0, 0, width, height);
  ctx.putImageData(result.overlay.imageData, 0, 0);
  ctx.putImageData(result.average.imageData, result.overlay.imageData.width, 0);
  ctx.putImageData(result.linear.imageData, 0, result.overlay.imageData.height);
  ctx.fillStyle = "#2f241b";
  ctx.font = "16px Segoe UI";
  ctx.fillText("Overlay", 16, 22);
  ctx.fillText("Average", result.overlay.imageData.width + 16, 22);
  ctx.fillText("Linear Feather", 16, result.overlay.imageData.height + 22);
}

function buildPairTransform(prepA, prepB, params) {
  const matches = matchDescriptors(prepA.sift.keypoints, prepB.sift.keypoints);
  const ransac = estimateHomographyRansac(matches, params.ransacIterations, params.ransacThreshold);
  return ransac.homography;
}

function rgbaToGray(imageData) {
  const gray = new Float32Array(imageData.width * imageData.height);
  for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 1) {
    gray[j] = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2];
  }
  return gray;
}

function gaussianBlur(src, width, height, size, sigma) {
  const radius = Math.floor(size / 2);
  const kernel = [];
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const value = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(value);
    sum += value;
  }
  for (let i = 0; i < kernel.length; i++) {
    kernel[i] /= sum;
  }
  const temp = new Float32Array(width * height);
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        acc += src[y * width + clamp(x + k, 0, width - 1)] * kernel[k + radius];
      }
      temp[y * width + x] = acc;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        acc += temp[clamp(y + k, 0, height - 1) * width + x] * kernel[k + radius];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}

function sobelGradients(src, width, height) {
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  const magnitude = new Float32Array(width * height);
  const direction = new Float32Array(width * height);
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sx = 0;
      let sy = 0;
      let idx = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const pixel = src[(y + dy) * width + (x + dx)];
          sx += pixel * kx[idx];
          sy += pixel * ky[idx];
          idx += 1;
        }
      }
      const pos = y * width + x;
      gx[pos] = sx;
      gy[pos] = sy;
      magnitude[pos] = Math.hypot(sx, sy);
      direction[pos] = Math.atan2(sy, sx);
    }
  }
  return { gx, gy, magnitude, direction };
}

function nonMaximumSuppression(magnitude, direction, width, height) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const pos = y * width + x;
      let angle = (direction[pos] * 180) / Math.PI;
      if (angle < 0) {
        angle += 180;
      }
      let q = 0;
      let r = 0;
      if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        q = magnitude[pos + 1];
        r = magnitude[pos - 1];
      } else if (angle >= 22.5 && angle < 67.5) {
        q = magnitude[(y + 1) * width + (x - 1)];
        r = magnitude[(y - 1) * width + (x + 1)];
      } else if (angle >= 67.5 && angle < 112.5) {
        q = magnitude[(y + 1) * width + x];
        r = magnitude[(y - 1) * width + x];
      } else {
        q = magnitude[(y - 1) * width + (x - 1)];
        r = magnitude[(y + 1) * width + (x + 1)];
      }
      out[pos] = magnitude[pos] >= q && magnitude[pos] >= r ? magnitude[pos] : 0;
    }
  }
  return out;
}

function hysteresisThreshold(nms, width, height, low, high) {
  const out = new Uint8ClampedArray(width * height);
  const stack = [];
  for (let i = 0; i < nms.length; i++) {
    if (nms[i] >= high) {
      out[i] = 255;
      stack.push(i);
    } else if (nms[i] >= low) {
      out[i] = 100;
    }
  }
  while (stack.length) {
    const current = stack.pop();
    const x = current % width;
    const y = Math.floor(current / width);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) {
          continue;
        }
        const idx = yy * width + xx;
        if (out[idx] === 100) {
          out[idx] = 255;
          stack.push(idx);
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) {
    out[i] = out[i] === 255 ? 255 : 0;
  }
  return out;
}

function detectHarris(gray, width, height, params) {
  const grads = sobelGradients(gray, width, height);
  const ix2 = new Float32Array(width * height);
  const iy2 = new Float32Array(width * height);
  const ixy = new Float32Array(width * height);
  for (let i = 0; i < ix2.length; i++) {
    ix2[i] = grads.gx[i] * grads.gx[i];
    iy2[i] = grads.gy[i] * grads.gy[i];
    ixy[i] = grads.gx[i] * grads.gy[i];
  }
  const sx2 = gaussianBlur(ix2, width, height, 5, 1.2);
  const sy2 = gaussianBlur(iy2, width, height, 5, 1.2);
  const sxy = gaussianBlur(ixy, width, height, 5, 1.2);
  const response = new Float32Array(width * height);
  for (let i = 0; i < response.length; i++) {
    const det = sx2[i] * sy2[i] - sxy[i] * sxy[i];
    const trace = sx2[i] + sy2[i];
    response[i] = det - 0.04 * trace * trace;
  }
  const threshold = percentile(response, params.harrisPercentile);
  const keypoints = [];
  for (let y = 3; y < height - 3; y++) {
    for (let x = 3; x < width - 3; x++) {
      const idx = y * width + x;
      const value = response[idx];
      if (value < threshold) {
        continue;
      }
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) {
            continue;
          }
          if (response[(y + dy) * width + (x + dx)] >= value) {
            isPeak = false;
            break;
          }
        }
      }
      if (isPeak) {
        keypoints.push({ x, y, score: value, size: 5 });
      }
    }
  }
  keypoints.sort((a, b) => b.score - a.score);
  return { response, keypoints: keypoints.slice(0, params.maxKeypoints) };
}

function detectSiftLike(gray, width, height, params) {
  const sigmas = [1.0, 1.6, 2.3, 3.2];
  const pyramid = sigmas.map((sigma) => gaussianBlur(gray, width, height, 7, sigma));
  const dogs = [];
  for (let i = 0; i < pyramid.length - 1; i++) {
    const diff = new Float32Array(width * height);
    for (let j = 0; j < diff.length; j++) {
      diff[j] = pyramid[i + 1][j] - pyramid[i][j];
    }
    dogs.push(diff);
  }

  const grads = sobelGradients(gray, width, height);
  const keypoints = [];
  for (let level = 1; level < dogs.length - 1; level++) {
    const prev = dogs[level - 1];
    const current = dogs[level];
    const next = dogs[level + 1];
    for (let y = 8; y < height - 8; y++) {
      for (let x = 8; x < width - 8; x++) {
        const idx = y * width + x;
        const value = current[idx];
        if (Math.abs(value) < params.siftThreshold * 255) {
          continue;
        }
        let isExtrema = true;
        for (let dy = -1; dy <= 1 && isExtrema; dy++) {
          for (let dx = -1; dx <= 1 && isExtrema; dx++) {
            for (const layer of [prev, current, next]) {
              if (layer === current && dx === 0 && dy === 0) {
                continue;
              }
              const neighbor = layer[(y + dy) * width + (x + dx)];
              if ((value > 0 && neighbor >= value) || (value < 0 && neighbor <= value)) {
                isExtrema = false;
                break;
              }
            }
          }
        }
        if (!isExtrema) {
          continue;
        }
        const descriptor = buildSiftDescriptor(grads, x, y, width, height);
        if (!descriptor) {
          continue;
        }
        keypoints.push({ x, y, score: Math.abs(value), size: 6 + level * 2, descriptor });
      }
    }
  }
  keypoints.sort((a, b) => b.score - a.score);
  return { keypoints: keypoints.slice(0, params.maxKeypoints) };
}

function buildSiftDescriptor(grads, x, y, width, height) {
  const descriptor = new Float32Array(128);
  const cellSize = 4;
  const binCount = 8;
  for (let cy = 0; cy < 4; cy++) {
    for (let cx = 0; cx < 4; cx++) {
      for (let yy = 0; yy < cellSize; yy++) {
        for (let xx = 0; xx < cellSize; xx++) {
          const px = x + cx * cellSize + xx - 8;
          const py = y + cy * cellSize + yy - 8;
          if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) {
            return null;
          }
          const idx = py * width + px;
          let angle = grads.direction[idx];
          if (angle < 0) {
            angle += Math.PI * 2;
          }
          const bin = Math.min(binCount - 1, Math.floor((angle / (Math.PI * 2)) * binCount));
          const offset = (cy * 4 + cx) * binCount + bin;
          descriptor[offset] += grads.magnitude[idx];
        }
      }
    }
  }
  let norm = 0;
  for (const value of descriptor) {
    norm += value * value;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < descriptor.length; i++) {
    descriptor[i] = Math.min(0.2, descriptor[i] / norm);
  }
  let renorm = 0;
  for (const value of descriptor) {
    renorm += value * value;
  }
  renorm = Math.sqrt(renorm) || 1;
  for (let i = 0; i < descriptor.length; i++) {
    descriptor[i] /= renorm;
  }
  return descriptor;
}

function matchDescriptors(pointsA, pointsB) {
  const matches = [];
  for (const pointA of pointsA) {
    if (!pointA.descriptor) {
      continue;
    }
    let best = { distance: Infinity, point: null };
    let second = { distance: Infinity };
    for (const pointB of pointsB) {
      if (!pointB.descriptor) {
        continue;
      }
      const distance = euclideanDistance(pointA.descriptor, pointB.descriptor);
      if (distance < best.distance) {
        second = best;
        best = { distance, point: pointB };
      } else if (distance < second.distance) {
        second = { distance, point: pointB };
      }
    }
    if (best.point && second.distance < Infinity && best.distance < 0.82 * second.distance) {
      matches.push({ a: pointA, b: best.point, distance: best.distance });
    }
  }
  matches.sort((lhs, rhs) => lhs.distance - rhs.distance);
  return matches.slice(0, 220);
}

function estimateHomographyRansac(matches, iterations, threshold) {
  if (matches.length < 4) {
    return { homography: null, inliers: [] };
  }
  let bestInliers = [];
  let bestH = null;
  for (let i = 0; i < iterations; i++) {
    const sample = randomSample(matches, 4);
    const h = computeHomography(
      sample.map((match) => [match.a.x, match.a.y]),
      sample.map((match) => [match.b.x, match.b.y])
    );
    if (!h) {
      continue;
    }
    const inliers = [];
    for (const match of matches) {
      const projected = projectPoint(h, match.a.x, match.a.y);
      const error = Math.hypot(projected.x - match.b.x, projected.y - match.b.y);
      if (error < threshold) {
        inliers.push(match);
      }
    }
    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestH = h;
    }
  }
  if (bestInliers.length >= 4) {
    return {
      homography: computeHomography(
        bestInliers.map((match) => [match.a.x, match.a.y]),
        bestInliers.map((match) => [match.b.x, match.b.y])
      ) || bestH,
      inliers: bestInliers,
    };
  }
  return { homography: bestH, inliers: bestInliers };
}

function computeHomography(srcPoints, dstPoints) {
  if (srcPoints.length < 4) {
    return null;
  }
  const pairs = srcPoints.slice(0, 4).map((point, index) => ({ src: point, dst: dstPoints[index] }));
  const a = [];
  const b = [];
  pairs.forEach(({ src, dst }) => {
    const [x, y] = src;
    const [u, v] = dst;
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(u);
    b.push(v);
  });
  const solution = solveLinearSystem(a, b);
  if (!solution) {
    return null;
  }
  return [
    [solution[0], solution[1], solution[2]],
    [solution[3], solution[4], solution[5]],
    [solution[6], solution[7], 1],
  ];
}

function solveLinearSystem(matrix, vector) {
  const n = matrix.length;
  const m = matrix[0].length;
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let col = 0, row = 0; col < m && row < n; col++, row++) {
    let pivot = row;
    for (let r = row + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) {
        pivot = r;
      }
    }
    if (Math.abs(a[pivot][col]) < 1e-8) {
      return null;
    }
    [a[row], a[pivot]] = [a[pivot], a[row]];
    const div = a[row][col];
    for (let c = col; c <= m; c++) {
      a[row][c] /= div;
    }
    for (let r = 0; r < n; r++) {
      if (r === row) {
        continue;
      }
      const factor = a[r][col];
      for (let c = col; c <= m; c++) {
        a[r][c] -= factor * a[row][c];
      }
    }
  }
  return a.map((row) => row[m]);
}

function warpTwoImages(imageA, imageB, homography, blendingMode) {
  return composePanorama([{ imageData: imageA }, { imageData: imageB }], [identityMatrix3(), homography], blendingMode);
}

function composePanorama(items, transforms, blendingMode) {
  const corners = [];
  items.forEach((item, index) => {
    const h = transforms[index];
    const width = item.imageData.width;
    const height = item.imageData.height;
    [[0, 0], [width, 0], [0, height], [width, height]].forEach(([x, y]) => corners.push(projectPoint(h, x, y)));
  });
  const minX = Math.floor(Math.min(...corners.map((point) => point.x)));
  const minY = Math.floor(Math.min(...corners.map((point) => point.y)));
  const maxX = Math.ceil(Math.max(...corners.map((point) => point.x)));
  const maxY = Math.ceil(Math.max(...corners.map((point) => point.y)));
  const width = clamp(maxX - minX, 64, 2400);
  const height = clamp(maxY - minY, 64, 1800);
  const accum = new Float32Array(width * height * 4);
  const weights = new Float32Array(width * height);

  items.forEach((item, index) => {
    const inverse = invertMatrix3(transforms[index]) || identityMatrix3();
    const src = item.imageData;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const worldX = x + minX;
        const worldY = y + minY;
        const point = projectPoint(inverse, worldX, worldY);
        if (point.x < 0 || point.y < 0 || point.x >= src.width - 1 || point.y >= src.height - 1) {
          continue;
        }
        if (blendingMode === "overlay" && weights[y * width + x] > 0) {
          continue;
        }
        const color = sampleBilinear(src, point.x, point.y);
        let weight = 1;
        if (blendingMode === "linear") {
          const wx = Math.min(point.x, src.width - point.x - 1) / Math.max(1, src.width / 2);
          const wy = Math.min(point.y, src.height - point.y - 1) / Math.max(1, src.height / 2);
          weight = Math.max(0.05, Math.min(wx, wy));
        }
        const idx = (y * width + x) * 4;
        accum[idx] += color[0] * weight;
        accum[idx + 1] += color[1] * weight;
        accum[idx + 2] += color[2] * weight;
        accum[idx + 3] += 255 * weight;
        weights[y * width + x] += weight;
      }
    }
  });

  const out = new ImageData(width, height);
  for (let i = 0; i < weights.length; i++) {
    const idx = i * 4;
    if (weights[i] <= 0) {
      out.data[idx + 3] = 0;
      continue;
    }
    out.data[idx] = clamp(Math.round(accum[idx] / weights[i]), 0, 255);
    out.data[idx + 1] = clamp(Math.round(accum[idx + 1] / weights[i]), 0, 255);
    out.data[idx + 2] = clamp(Math.round(accum[idx + 2] / weights[i]), 0, 255);
    out.data[idx + 3] = 255;
  }
  return { imageData: out };
}

function drawImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function drawGrayMap(canvas, values, width, height, autoScale) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  const max = autoScale ? arrayMax(values) || 1 : 255;
  for (let i = 0; i < values.length; i++) {
    const value = clamp(Math.round((values[i] / max) * 255), 0, 255);
    const idx = i * 4;
    image.data[idx] = value;
    image.data[idx + 1] = value;
    image.data[idx + 2] = value;
    image.data[idx + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function drawBinaryMap(canvas, values, width, height) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < values.length; i++) {
    const idx = i * 4;
    image.data[idx] = values[i];
    image.data[idx + 1] = values[i];
    image.data[idx + 2] = values[i];
    image.data[idx + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function drawKeypoints(canvas, imageData, keypoints, color) {
  drawImageData(canvas, imageData);
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  keypoints.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, point.size || 5, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function drawSideBySideFeatures(canvas, imageA, imageB, pointsA, pointsB) {
  const width = imageA.width + imageB.width;
  const height = Math.max(imageA.height, imageB.height);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f5eee5";
  ctx.fillRect(0, 0, width, height);
  ctx.putImageData(imageA, 0, 0);
  ctx.putImageData(imageB, imageA.width, 0);
  ctx.strokeStyle = "#d1495b";
  pointsA.slice(0, 120).forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, point.size || 5, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.strokeStyle = "#00798c";
  pointsB.slice(0, 120).forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x + imageA.width, point.y, point.size || 5, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function drawMatches(canvas, imageA, imageB, matches, color) {
  const width = imageA.width + imageB.width;
  const height = Math.max(imageA.height, imageB.height);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f5eee5";
  ctx.fillRect(0, 0, width, height);
  ctx.putImageData(imageA, 0, 0);
  ctx.putImageData(imageB, imageA.width, 0);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;
  matches.forEach((match) => {
    ctx.beginPath();
    ctx.moveTo(match.a.x, match.a.y);
    ctx.lineTo(match.b.x + imageA.width, match.b.y);
    ctx.stroke();
  });
}

function clearCanvas(canvas) {
  canvas.width = 320;
  canvas.height = 180;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ede6dc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#715b48";
  ctx.font = "16px Segoe UI";
  ctx.fillText("Waiting for results", 24, 40);
}

function sampleBilinear(imageData, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, imageData.width - 1);
  const y1 = Math.min(y0 + 1, imageData.height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const c00 = getPixel(imageData, x0, y0);
  const c10 = getPixel(imageData, x1, y0);
  const c01 = getPixel(imageData, x0, y1);
  const c11 = getPixel(imageData, x1, y1);
  return [0, 1, 2].map(
    (i) =>
      c00[i] * (1 - fx) * (1 - fy) +
      c10[i] * fx * (1 - fy) +
      c01[i] * (1 - fx) * fy +
      c11[i] * fx * fy
  );
}

function getPixel(imageData, x, y) {
  const idx = (y * imageData.width + x) * 4;
  return [imageData.data[idx], imageData.data[idx + 1], imageData.data[idx + 2]];
}

function percentile(values, percent) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const index = Math.floor((percent / 100) * (sorted.length - 1));
  return sorted[index];
}

function arrayMax(values) {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) {
      max = values[i];
    }
  }
  return max;
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function randomSample(values, count) {
  const pool = [...values];
  const selected = [];
  for (let i = 0; i < count; i++) {
    const index = Math.floor(Math.random() * pool.length);
    selected.push(pool.splice(index, 1)[0]);
  }
  return selected;
}

function projectPoint(matrix, x, y) {
  const denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (Math.abs(denominator) < 1e-8) {
    return { x, y };
  }
  return {
    x: (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
    y: (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
  };
}

function identityMatrix3() {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

function multiplyMatrix3(a, b) {
  const out = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        out[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return out;
}

function invertMatrix3(m) {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-8) {
    return null;
  }
  const invDet = 1 / det;
  return [
    [
      (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invDet,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet,
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
      (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invDet,
    ],
    [
      (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invDet,
      (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invDet,
      (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invDet,
    ],
  ];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

canvasIds.forEach((id) => clearCanvas(canvases[id]));
