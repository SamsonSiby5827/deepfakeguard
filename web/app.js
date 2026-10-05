// DeepfakeGuard in the browser.
// Same pipeline as the Python app (src/ml/deepfake_inference.py), but it runs on the
// visitor's own device, so the photo is never uploaded anywhere:
//   1. OpenCV.js Haar cascade finds the largest face (scaleFactor 1.1, minNeighbors 5, min 40 px)
//   2. Crop with an 18% margin, resize to 224 x 224 (INTER_AREA), RGB, float 0-255
//   3. ONNX Runtime Web runs the converted V2 model -> probability that the face is REAL

const CONFIG = {
  modelUrl: "https://huggingface.co/Samson5827/deepfakeguard-v2/resolve/main/deepfakeguard_v2.onnx",
  labelsUrl: "https://huggingface.co/Samson5827/deepfakeguard-v2/resolve/main/labels_v2.json",
  cascadeUrl: "haarcascade_frontalface_default.xml",
  ortWasmPath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/",
  imgSize: 224,
  marginRatio: 0.18,
  minBlurScore: 35.0,
  minFaceAreaRatio: 0.02,
  maxBytes: 10 * 1024 * 1024,
};

const els = {
  status: document.getElementById("status"),
  file: document.getElementById("file"),
  drop: document.getElementById("drop"),
  canvas: document.getElementById("preview"),
  check: document.getElementById("check"),
  result: document.getElementById("result"),
};

const state = { session: null, inputName: null, labels: null, classifier: null, image: null, file: null };

function setStatus(text, ready = false) {
  els.status.textContent = text;
  els.status.classList.toggle("ready", ready);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function waitForOpenCv() {
  // opencv.js finishes compiling its WebAssembly after the script loads.
  for (let i = 0; i < 600; i++) {
    if (window.cv && typeof cv.Mat === "function") return;
    if (window.cv && typeof cv.then === "function") { window.cv = await cv; return; }
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error("OpenCV did not load");
}

async function init() {
  try {
    setStatus("Loading face detector…");
    await waitForOpenCv();
    const xml = new Uint8Array(await (await fetch(CONFIG.cascadeUrl)).arrayBuffer());
    cv.FS_createDataFile("/", "face.xml", xml, true, false, false);
    state.classifier = new cv.CascadeClassifier();
    if (!state.classifier.load("face.xml")) throw new Error("Could not load the face detector");

    setStatus("Downloading the model (about 16 MB, first visit only)…");
    ort.env.wasm.wasmPaths = CONFIG.ortWasmPath;
    ort.env.wasm.numThreads = 1;
    const [modelBytes, labels] = await Promise.all([
      fetch(CONFIG.modelUrl).then(r => { if (!r.ok) throw new Error("Model download failed (" + r.status + ")"); return r.arrayBuffer(); }),
      fetch(CONFIG.labelsUrl).then(r => r.json()),
    ]);
    state.labels = labels.index_to_label;
    state.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    state.inputName = state.session.inputNames[0];

    setStatus("Ready. Your photo stays on this device.", true);
    if (state.file) els.check.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Could not load the demo: " + err.message + ". Please refresh the page.");
  }
}

function drawPreview(box) {
  const img = state.image;
  if (!img) return;
  const c = els.canvas;
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  if (box) {
    ctx.lineWidth = Math.max(2, Math.round(c.width / 250));
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#0b7a57";
    ctx.strokeRect(box[0], box[1], box[2], box[3]);
  }
  c.style.display = "block";
}

function chooseFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    els.result.innerHTML = '<div class="error">Please choose a JPG, PNG or WEBP image.</div>';
    return;
  }
  if (file.size > CONFIG.maxBytes) {
    els.result.innerHTML = '<div class="error">Image is larger than 10 MB.</div>';
    return;
  }
  state.file = file;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { state.image = img; drawPreview(null); URL.revokeObjectURL(url); };
  img.onerror = () => { state.image = null; els.result.innerHTML = '<div class="error">This file could not be read as an image.</div>'; };
  img.src = url;
  els.check.disabled = !state.session;
  els.result.innerHTML = '<p class="empty">' + (state.session ? 'Ready. Click "Check image".' : "Image ready. Waiting for the model to finish loading…") + "</p>";
}

// The same steps as DeepfakeInference._prepare_face_for_model in Python.
function analyse(img) {
  const full = document.createElement("canvas");
  full.width = img.naturalWidth;
  full.height = img.naturalHeight;
  full.getContext("2d").drawImage(img, 0, 0);

  const rgba = cv.imread(full);
  const gray = new cv.Mat();
  const faces = new cv.RectVector();
  const mats = [rgba, gray];
  try {
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    state.classifier.detectMultiScale(gray, faces, 1.1, 5, 0, new cv.Size(40, 40), new cv.Size(0, 0));

    let box = null;
    let crop;
    if (faces.size() > 0) {
      let best = faces.get(0);
      for (let i = 1; i < faces.size(); i++) {
        const f = faces.get(i);
        if (f.width * f.height > best.width * best.height) best = f;
      }
      const mx = Math.trunc(best.width * CONFIG.marginRatio);
      const my = Math.trunc(best.height * CONFIG.marginRatio);
      const x1 = Math.max(0, best.x - mx);
      const y1 = Math.max(0, best.y - my);
      const x2 = Math.min(rgba.cols, best.x + best.width + mx);
      const y2 = Math.min(rgba.rows, best.y + best.height + my);
      box = [x1, y1, x2 - x1, y2 - y1];
      crop = rgba.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
    } else {
      crop = rgba.clone();
    }
    mats.push(crop);

    const rgb = new cv.Mat();
    const resized = new cv.Mat();
    const faceGray = new cv.Mat();
    const lap = new cv.Mat();
    const mean = new cv.Mat();
    const std = new cv.Mat();
    mats.push(rgb, resized, faceGray, lap, mean, std);

    cv.cvtColor(crop, rgb, cv.COLOR_RGBA2RGB);
    cv.resize(rgb, resized, new cv.Size(CONFIG.imgSize, CONFIG.imgSize), 0, 0, cv.INTER_AREA);

    // Blur score: variance of the Laplacian of the resized face (same as Python).
    cv.cvtColor(resized, faceGray, cv.COLOR_RGB2GRAY);
    cv.Laplacian(faceGray, lap, cv.CV_64F);
    cv.meanStdDev(lap, mean, std);
    const blurScore = std.doubleAt(0, 0) ** 2;

    const input = Float32Array.from(resized.data); // 224*224*3 values, 0-255, RGB, row by row
    const faceAreaRatio = box ? (box[2] * box[3]) / (rgba.cols * rgba.rows) : 0;
    return { input, box, blurScore, faceAreaRatio };
  } finally {
    mats.forEach(m => m.delete());
    faces.delete();
  }
}

async function check() {
  if (!state.image || !state.session) return;
  els.check.disabled = true;
  els.check.textContent = "Checking…";
  try {
    const a = analyse(state.image);
    const tensor = new ort.Tensor("float32", a.input, [1, CONFIG.imgSize, CONFIG.imgSize, 3]);
    const out = await state.session.run({ [state.inputName]: tensor });
    const pReal = out[state.session.outputNames[0]].data[0];
    const label = pReal >= 0.5 ? state.labels["1"] : state.labels["0"];
    const confidence = pReal >= 0.5 ? pReal : 1 - pReal;

    const warnings = [];
    if (!a.box) warnings.push("No face found, so the whole image was checked. Result is less reliable.");
    else if (a.faceAreaRatio < CONFIG.minFaceAreaRatio) warnings.push("The face is very small in the image. Result is less reliable.");
    if (a.blurScore < CONFIG.minBlurScore) warnings.push("The image is blurry. Result is less reliable.");

    drawPreview(a.box);
    render({
      label,
      confidence: +confidence.toFixed(4),
      probabilities: { fake: +(1 - pReal).toFixed(4), real: +pReal.toFixed(4) },
      face_detected: !!a.box,
      face_box: a.box,
      blur_score: +a.blurScore.toFixed(2),
      face_area_ratio: +a.faceAreaRatio.toFixed(4),
      warnings,
    });
  } catch (err) {
    console.error(err);
    els.result.innerHTML = '<div class="error">Something went wrong while checking this image: ' + escapeHtml(err.message) + "</div>";
  } finally {
    els.check.disabled = false;
    els.check.textContent = "Check image";
  }
}

function render(data) {
  const isFake = data.label === "fake";
  const pFake = Math.round(data.probabilities.fake * 100);
  const pReal = 100 - pFake;
  let html = '<div class="verdict ' + (isFake ? "fake" : "real") + '">' + (isFake ? "Likely fake" : "Likely real") + "</div>";
  html += '<div class="conf">Model confidence: ' + Math.round(data.confidence * 100) + "%</div>";
  html += '<div class="bar" role="img" aria-label="' + pReal + "% real, " + pFake + '% fake"><span class="b-real" style="width:' + pReal + '%"></span><span class="b-fake" style="width:' + pFake + '%"></span></div>';
  html += '<div class="legend"><span>Real ' + pReal + "%</span><span>Fake " + pFake + "%</span></div>";
  if (data.warnings.length) {
    html += '<div class="warn"><strong>Take this result with caution:</strong><ul>' + data.warnings.map(w => "<li>" + escapeHtml(w) + "</li>").join("") + "</ul></div>";
  }
  html += "<details><summary>Raw result</summary><pre>" + escapeHtml(JSON.stringify(data, null, 2)) + "</pre></details>";
  els.result.innerHTML = html;
  window.lastResult = data; // handy for automated checks
}

els.file.addEventListener("change", e => chooseFile(e.target.files[0]));
["dragenter", "dragover"].forEach(ev => els.drop.addEventListener(ev, e => { e.preventDefault(); els.drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => els.drop.addEventListener(ev, e => { e.preventDefault(); els.drop.classList.remove("over"); }));
els.drop.addEventListener("drop", e => chooseFile(e.dataTransfer.files[0]));
els.check.addEventListener("click", check);
init();
