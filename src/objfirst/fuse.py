"""M5b: fuse object-first candidates with the detector's entries.

Documented rule, written before the run:

  1. match on cluster overlap: an object-first candidate and a detector entry
     belong together when they share at least one cluster id;
  2. when they agree on the type, keep the DETECTOR entry (it carries the
     confidence, the evidence frames and the naming the report already uses) and
     record that object-first agreed;
  3. when they disagree on the type, keep the detector entry and record the
     disagreement, so the fusion can never overwrite a released behaviour;
  4. object-first candidates with no detector counterpart are appended and
     marked `source: objfirst`, so the gate evaluation can count them and the
     default path is unaffected.

The function is deliberately side-effect free; the caller decides what to write.
"""


def _ids(entry: dict) -> set[int]:
    ids = set()
    for key in ("cluster_ids", "from_cluster", "to_cluster"):
        val = entry.get(key)
        if isinstance(val, list):
            ids.update(int(v) for v in val)
        elif isinstance(val, str):
            for part in val.split("+"):
                if "-" in part:
                    try:
                        ids.add(int(part.split("-", 1)[1]))
                    except ValueError:
                        pass
    return ids


def fuse(detector: list[dict], objfirst: list[dict]) -> list[dict]:
    fused: list[dict] = []
    used = set()
    for det in detector:
        d_ids = _ids(det)
        partners = [i for i, o in enumerate(objfirst) if d_ids & _ids(o)]
        entry = dict(det)
        entry["objfirst_agreement"] = [
            {"type": objfirst[i].get("type"), "label": objfirst[i].get("label_final")}
            for i in partners
        ]
        disagreements = [p for p in entry["objfirst_agreement"] if p["type"] != det.get("type")]
        if disagreements:
            entry["objfirst_disagreement"] = disagreements
        fused.append(entry)
        used.update(partners)
    for i, o in enumerate(objfirst):
        if i not in used:
            fused.append({**o, "source": "objfirst"})
    return fused
