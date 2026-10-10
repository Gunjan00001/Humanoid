"""Self-contained interactive HTML viewer (CPU).

Writes ``outputs/<scene>/viewer.html`` - one file, no build step:

  - three.js r128 loaded from cdnjs (pinned version; the only external script),
  - orbit / zoom controls implemented inline (no extra scripts),
  - the display cloud embedded and quantized (uint16 positions relative to the
    scene bbox, uint8 colours, uint8 tag), base64-encoded,
  - toggles for video A, video B and the detected changes,
  - a bounding box + text label per confirmed change.

Budget: <= `visualize.viewer_max_mb` MB (default 5). The cloud is subsampled
deterministically if it would exceed the budget.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("viewer")

THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def _quantize_positions(points: np.ndarray, lo: np.ndarray, extent: np.ndarray) -> bytes:
    """uint16 positions relative to the bbox (extent-normalised)."""
    q = np.clip((points - lo) / np.maximum(extent, 1e-9), 0, 1) * 65535.0
    return q.astype("<u2").tobytes()


def write_viewer(cfg: dict, out_dir: Path, bg_pts: np.ndarray, bg_rgb: np.ndarray,
                 bg_tags: np.ndarray, change_pts: list[dict],
                 focus: np.ndarray | None = None,
                 radius: float | None = None) -> Path:
    vcfg = cfg["visualize"]
    max_mb = float(vcfg.get("viewer_max_mb", 5))
    budget_bytes = int(max_mb * 1024 * 1024 * 0.72)   # base64 overhead ~1.37x

    # deterministic subsample to fit the budget (9 bytes/point quantized)
    max_points = max(1000, budget_bytes // 9 - 20000)
    if len(bg_pts) > max_points:
        idx = np.linspace(0, len(bg_pts) - 1, max_points).round().astype(int)
        bg_pts, bg_rgb, bg_tags = bg_pts[idx], bg_rgb[idx], bg_tags[idx]
        log.info("viewer: subsampled cloud to %d points for the size budget", len(bg_pts))

    all_pts = [bg_pts] + [c["points"] for c in change_pts if len(c["points"])]
    all_pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3), np.float32)
    lo = all_pts.min(axis=0)
    extent = np.maximum(all_pts.max(axis=0) - lo, 1e-6)

    def payload(points, colors, tags=None, n_crop=None) -> dict:
        d = {
            "pos": base64.b64encode(_quantize_positions(points, lo, extent)).decode(),
            "col": base64.b64encode(np.asarray(colors, dtype=np.uint8).tobytes()).decode(),
            "n": int(len(points)),
        }
        if n_crop is not None:
            d["n_crop"] = int(n_crop)
        if tags is not None:
            d["tag"] = base64.b64encode(np.asarray(tags, dtype=np.uint8).tobytes()).decode()
        return d

    # sort the cloud by distance to the focus so the viewer's crop toggle is a
    # cheap drawRange over the first n_crop points
    crop_r = float(vcfg.get("viewer_crop_radius", 0.4))
    dist = np.linalg.norm(bg_pts - np.asarray(focus, dtype=float), axis=1)
    order = np.argsort(dist)
    bg_pts, bg_rgb, bg_tags, dist = bg_pts[order], bg_rgb[order], bg_tags[order], dist[order]
    m_a, m_b = bg_tags == "a", bg_tags == "b"
    data = {
        "bbox": {"lo": lo.tolist(), "extent": extent.tolist()},
        "cloud_a": payload(bg_pts[m_a], bg_rgb[m_a],
                           n_crop=int((dist[m_a] <= crop_r).sum())),
        "cloud_b": payload(bg_pts[m_b], bg_rgb[m_b],
                           n_crop=int((dist[m_b] <= crop_r).sum())),
        "changes": [],
    }
    for c in change_pts:
        ch = c["change"]
        if not len(c["points"]):
            continue
        entry = payload(c["points"], (c["colors"] * 255).astype(np.uint8))
        entry.update({
            "id": ch["id"], "type": ch["type"], "verified": bool(c["verified"]),
            "label": ch.get("label"), "description": ch.get("description"),
            "bbox_min": ch["bbox_min"], "bbox_max": ch["bbox_max"],
        })
        data["changes"].append(entry)

    # initial camera framing: the change region when available (same framing
    # rule as the static renders), else the whole cloud
    if focus is None:
        focus = all_pts.mean(axis=0) if len(all_pts) else np.zeros(3)
    if radius is None:
        radius = float(np.linalg.norm(extent)) / 2 if len(all_pts) else 1.0
    focus = np.asarray(focus, dtype=float).tolist()

    html = _TEMPLATE.replace("__THREE_CDN__", THREE_CDN) \
                    .replace("__DATA__", json.dumps(data, separators=(",", ":"))) \
                    .replace("__FOCUS__", json.dumps(focus)) \
                    .replace("__RADIUS__", f"{radius:.4f}")
    path = out_dir / "viewer.html"
    path.write_text(html)
    log.info("viewer: wrote %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)
    return path


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3D change viewer</title>
<style>
  html,body{margin:0;height:100%;background:#101418;color:#eee;font:13px sans-serif}
  #ui{position:absolute;top:10px;left:10px;background:#0008;padding:10px;border-radius:6px}
  #ui label{display:block;margin:2px 0}
  #labels{position:absolute;inset:0;pointer-events:none}
  .lab{position:absolute;padding:2px 6px;border-radius:4px;font-size:11px;color:#fff;
       transform:translate(-50%,-100%);white-space:nowrap}
</style></head><body>
<div id="ui">
  <b>3D change viewer</b><br>
  <label><input type="checkbox" id="ta" checked> video A</label>
  <label><input type="checkbox" id="tb" checked> video B</label>
  <label><input type="checkbox" id="tc" checked> changes</label>
  <label><input type="checkbox" id="tl" checked> labels</label>
  <label><input type="checkbox" id="tf"> full scene</label>
  <div style="margin-top:6px;font-size:11px;color:#aaa">drag: orbit &middot; wheel: zoom</div>
</div>
<div id="labels"></div>
<script src="__THREE_CDN__"></script>
<script>
const DATA = __DATA__, FOCUS = __FOCUS__, RADIUS = __RADIUS__;
// render parameters for headless-Chrome shots (scripts/render_gif.py):
//   ?shot=1  hide the UI panel (clean frame for the GIF)
//   ?az=..&elev=..  set the camera angles (degrees; elevation from the up axis)
//   ?crop=1  draw only the points within CROP_R of the focus (default ON;
//            the "full scene" toggle switches to the whole cloud)
const Q = new URLSearchParams(location.search);
const SHOT = Q.get('shot') === '1';
const CROP_R = parseFloat(Q.get('crop_r') || '0.4');
if (SHOT) document.getElementById('ui').style.display = 'none';
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101418);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, RADIUS/500, RADIUS*40);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);

const LO = DATA.bbox.lo, EXT = DATA.bbox.extent;
function geom(p){                       // decode quantized payload
  const posBytes = Uint8Array.from(atob(p.pos), c=>c.charCodeAt(0));
  const colBytes = Uint8Array.from(atob(p.col), c=>c.charCodeAt(0));
  const q = new Uint16Array(posBytes.buffer);       // little-endian u16 pairs
  const n = q.length/3;
  const g = new THREE.BufferGeometry();
  const positions = new Float32Array(n*3), colors = new Float32Array(n*3);
  for(let i=0;i<n*3;i++){
    positions[i] = LO[i%3] + (q[i]/65535)*EXT[i%3];
    colors[i] = colBytes[i]/255;
  }
  g.setAttribute('position', new THREE.BufferAttribute(positions,3));
  g.setAttribute('color', new THREE.BufferAttribute(colors,3));
  return g;
}
function points(p, size){
  const m = new THREE.PointsMaterial({size:size, vertexColors:true, sizeAttenuation:true});
  return new THREE.Points(geom(p), m);
}
const pa = points(DATA.cloud_a, RADIUS/220), pb = points(DATA.cloud_b, RADIUS/220);
// the embedded cloud is sorted by distance to the focus (write_viewer), so a
// "crop" is simply a drawRange over the first N points - the toggle switches
// between the focused view (~0.4 m) and the full scene
const N_A = DATA.cloud_a.n, N_B = DATA.cloud_b.n;
const CROP_A = DATA.cloud_a.n_crop ?? N_A, CROP_B = DATA.cloud_b.n_crop ?? N_B;
function applyCrop(cropped){
  pa.geometry.setDrawRange(0, cropped ? CROP_A : N_A);
  pb.geometry.setDrawRange(0, cropped ? CROP_B : N_B);
}
applyCrop(Q.get('full') !== '1');
scene.add(pa, pb);
const changeGroup = new THREE.Group(); scene.add(changeGroup);
const COL = {removed:0xf23333, added:0x40e659, moved:0xff991a, replaced:0xf23ff2};
for(const c of DATA.changes){
  const pts = points(c, RADIUS/160); changeGroup.add(pts);
  const lo = new THREE.Vector3(...c.bbox_min), hi = new THREE.Vector3(...c.bbox_max);
  const box = new THREE.Box3Helper(new THREE.Box3(lo,hi),
      new THREE.Color(c.verified ? (COL[c.type]||0xffffff) : 0x59bfd9));
  changeGroup.add(box);
}
const labelDivs = DATA.changes.map(c=>{
  const d = document.createElement('div'); d.className='lab';
  d.style.background = '#' + (COL[c.type]||'888888') + 'aa';
  d.textContent = `${c.id}: ${c.type}${c.label? ' - '+c.label : ''}` + (c.verified?'':' (unverified)');
  document.getElementById('labels').appendChild(d); return d;
});

// --- minimal inline orbit controls (no extra scripts) ---
// the reconstruction frames are z-up (gravity-ish / camera-0 frame), so the
// camera's up axis is +z; phi is measured from +z
camera.up.set(0, 0, 1);
let theta = (parseFloat(Q.get('az') ?? '-52')) * Math.PI / 180;
let phi = (90 - parseFloat(Q.get('elev') ?? '40')) * Math.PI / 180;
let dist = RADIUS * 2.6;
const target = new THREE.Vector3(...FOCUS);
let drag=false, px=0, py=0;
renderer.domElement.addEventListener('mousedown', e=>{drag=true;px=e.clientX;py=e.clientY;});
addEventListener('mouseup', ()=>drag=false);
addEventListener('mousemove', e=>{ if(!drag) return;
  theta -= (e.clientX-px)*0.008; phi = Math.max(0.15, Math.min(3.0, phi-(e.clientY-py)*0.008));
  px=e.clientX; py=e.clientY; });
renderer.domElement.addEventListener('wheel', e=>{ dist *= (1 + Math.sign(e.deltaY)*0.1);
  dist = Math.max(RADIUS*0.4, Math.min(RADIUS*12, dist)); e.preventDefault(); }, {passive:false});
addEventListener('resize', ()=>{ camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

document.getElementById('ta').onchange = e => pa.visible = e.target.checked;
document.getElementById('tb').onchange = e => pb.visible = e.target.checked;
document.getElementById('tc').onchange = e => changeGroup.visible = e.target.checked;
document.getElementById('tf').onchange = e => applyCrop(!e.target.checked);
document.getElementById('tl').onchange = e =>
  labelDivs.forEach(d => d.style.display = e.target.checked ? '' : 'none');

function frame(){
  requestAnimationFrame(frame);
  camera.position.set(
    target.x + dist*Math.sin(phi)*Math.cos(theta),
    target.y + dist*Math.sin(phi)*Math.sin(theta),
    target.z + dist*Math.cos(phi));
  camera.lookAt(target);
  renderer.render(scene, camera);
  DATA.changes.forEach((c,i)=>{
    const p = new THREE.Vector3((c.bbox_min[0]+c.bbox_max[0])/2,
                                (c.bbox_min[1]+c.bbox_max[1])/2, c.bbox_max[2]);
    p.project(camera);
    const d = labelDivs[i];
    d.style.left = (p.x*0.5+0.5)*innerWidth + 'px';
    d.style.top  = (-p.y*0.5+0.5)*innerHeight + 'px';
    d.style.display = (p.z < 1) ? '' : 'none';
  });
}
frame();
</script></body></html>
"""
