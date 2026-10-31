import json
import os
import urllib.request

tree = json.load(open("tree.json"))
paths = [e["path"] for e in tree["tree"]
         if e["path"].lower().endswith((".pes", ".dst", ".jef", ".vp3", ".exp", ".xxx", ".u01", ".pec"))]
os.makedirs("samples", exist_ok=True)
ok = fail = 0
for p in paths:
    out = os.path.join("samples", p.replace("/", "_"))
    if os.path.exists(out):
        ok += 1
        continue
    url = "https://raw.githubusercontent.com/EmbroidePy/samples/master/" + urllib.parse.quote(p)
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(out, "wb") as f:
            f.write(r.read())
        ok += 1
    except Exception as exc:
        fail += 1
        print("FAIL", p, exc)
print("ok", ok, "fail", fail)
