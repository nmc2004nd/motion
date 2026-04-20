import re
with open("src/pipeline.py", "r") as f:
    text = f.read()

old = r"def_markers_tracked, valid = match_markers_robust(ref_proc, def_proc, ref_markers)"
new = r"def_markers_tracked, valid = match_markers_robust(ref_markers, def_markers_naive, img_ref.shape)"
text = text.replace(old, new)

with open("src/pipeline.py", "w") as f:
    f.write(text)
print("done")
