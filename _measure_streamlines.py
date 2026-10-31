import cv2, numpy as np, json, urllib.request, math

motif="3b7967ba8ecc4b3fa5811c23ecc37835"
base=f"http://127.0.0.1:5000/api/motifs/{motif}"
req=urllib.request.urlopen(base, timeout=30)
meta=json.loads(req.read())

raw_url=f"http://127.0.0.1:5000{meta['urls']['raw']}"
raw=urllib.request.urlopen(raw_url, timeout=30).read()
arr=np.frombuffer(raw,dtype=np.uint8)
raw_img=cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
source=cv2.imread('_inspection/source.png')
H,W=source.shape[:2]
ppm=22

# 1) count threads (lines) per region: streamlines_info says 342 lines, 5 patches
print('=== STREAMLINE META ===')
print('patches', meta['streamline_info']['patches'], 'lines', meta['streamline_info']['lines'], 'size_mm', meta['streamline_info']['size_mm'])
print('design size_mm', meta['size_mm'], 'target_mm', meta['target_mm'])
print('thread line_width_mm 0.4 -> spacing ~ 0.41 mm => theoretical coverage ~ line^2/density?')

# 2) expected stitches vs actual
stats=meta['stats']
described_lines=342
print('\n=== STITCH COVERAGE ESTIMATE ===')
print('stitches', stats['stitches'], 'jumps', stats['jumps'], 'color_changes', stats['color_changes'])
# design fabric area ~ size_mm^2
mm_area = meta['size_mm'][0]*meta['size_mm'][1]
print('design area mm^2', round(mm_area,1))
# each stitch covers ~ line_width^2? no, each stitch length ~? Unknown. But use raw coverage instead
# fraction of canvas covered by stitches in raw preview
gray=cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
# Assume thread is darker than fabric -> dark pixels are threads
# But raw shows 97% near white, so threads are 3% of area. Compute thread_area_mm^2
th_pct = float((gray<200).mean())
print('thread pixel fraction (V<200)', round(th_pct,4))
thread_area_mm2 = th_pct * mm_area
print('thread area mm^2', round(thread_area_mm2,3))
# thread length estimate: coverage = (line_width * stitch_length)/mm_area
# If line_width=0.4mm, then stitch_length_total = thread_area_mm2 / line_width_mm
line_w_mm=0.4
total_stitch_len_mm = thread_area_mm2 / line_w_mm
print('implied total stitch length mm (raw)', round(total_stitch_len_mm,1))
avg_len = total_stitch_len_mm / stats['stitches']
print('avg stitch length mm (raw)', round(avg_len,3))
print('avg length per jump?', round((total_stitch_len_mm)/stats['jumps'],2) if stats['jumps'] else 0)

# 3) photo preview: what fraction does photo show as thread?
photo=urllib.request.urlopen(f"http://127.0.0.1:5000{meta['urls']['photo']}", timeout=30).read()
photo_img=cv2.imdecode(np.frombuffer(photo,dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print('\n=== PHOTO COVERAGE ===')
print('photo shape', photo_img.shape)
pgray=cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)
p_white_pct=float((pgray>200).mean())
p_thread_pct=float((pgray<200).mean())
p_mm_area = meta['size_mm'][0]*meta['size_mm'][1] * (photo_img.shape[0]*photo_img.shape[1])/( (meta['size_mm'][0]+6)*ppm*((meta['size_mm'][1]+6)*ppm) )
# recalc properly: photo has margin=6mm, so photo canvas mm = (size_mm+12)
photo_canvas_mm = meta['size_mm'][0]+12, meta['size_mm'][1]+12
photo_area = photo_canvas_mm[0]*photo_canvas_mm[1]
p_thread_area_mm2 = p_thread_pct * photo_area
p_total_len_mm = p_thread_area_mm2 / line_w_mm
print('photo thread pixel fraction', round(p_thread_pct,4))
print('photo implied total stitch length mm', round(p_total_len_mm,1))
print('photo avg stitch length per stitch mm', round(p_total_len_mm/stats['stitches'],3))
print('photo avg per jump mm', round(p_total_len_mm/stats['jumps'],2) if stats['jumps'] else 0)

# 4) photo coverage fraction (thread coverage density)
print('\n=== DENSITY COMPARISON ===')
raw_density = th_pct
photo_density = p_thread_pct
print('raw thread density (fraction of design)', round(raw_density,4), '-> %.1f%%' % (raw_density*100))
print('photo thread density (fraction of design+margin)', round(photo_density,4), '-> %.1f%%' % (photo_density*100))
print('ratio photo/raw', round(photo_density/raw_density,3) if raw_density else 0)
PY