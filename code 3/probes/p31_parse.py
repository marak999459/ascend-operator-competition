import re, collections
rows = collections.defaultdict(lambda: collections.defaultdict(list))
i = 0
for ln in open('/tmp/p31_grid.txt', encoding='utf-8', errors='replace'):
    m = re.match(r'^P(\S+)\s+(\d+)\s+nb=(\d+)\s+kb=(\d+)\s+ks=\S+\s*(.*)$', ln.rstrip('\n'))
    if not m: continue
    i += 1
    cs, nb, kb, ks, rest = m.groups()
    p = 1 if i <= 92 else 2
    t = re.search(r'([\d.]+)\s*ms', rest)
    bad = bool(re.search(r'超差\s+[1-9]\d*/', rest)) or ('FAIL' in rest) or ('ERROR' in rest)
    if t: rows[cs][(int(nb), int(kb), int(ks))].append((p, float(t.group(1)), bad))
print("cells parsed:", i)
for cs in ['p1','p2','q1h','q2h','p4','p6','big1']:
    d = rows.get(cs)
    if not d: continue
    good = {k: [x[1] for x in v if not x[2]] for k, v in d.items()}
    good = {k: v for k, v in good.items() if v}
    if not good: print(f"--- {cs}: 无有效格"); continue
    base = min(min(v) for v in good.values())
    print(f"--- {cs}  (最快 {base:.4f})")
    for k in sorted(good):
        v = good[k]
        flag = '  ⚠️含BAD' if any(x[2] for x in d[k]) else ''
        print(f"    nb={k[0]:<2} k={k[1]:<3} ks={k[2]}  {min(v):.4f}  {100*(min(v)/base-1):+7.2f}%   [{' / '.join(f'{x:.4f}' for x in v)}]{flag}")
