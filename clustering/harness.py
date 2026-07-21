"""M0 benchmark harness (PLAN §4): anchors + metrics, so material-texture
features are ranked by measurement instead of eyeballed PNGs.

Subcommands (run with the py38zarr python; no GPU anywhere):

  make_sheets [--chunks a,b] [--ds_target 3600]
      Per-chunk labeling sheets: raw mid slice + 384-px gridlines + each
      patch's rrrccc stem. View next to Output/regis2/cluster_fable/zw3_coarse/
      and hand-write picks. -> Output/clustering/sheets/{chunk}.png

  anchors [--picks clustering/picks.csv]
      picks.csv lines are `chunk,stem,class` (# comments / blanks ok), e.g.
          z0240-0288,010023,laminar
      Validates stems, converts to canonical clustering/anchors.csv with
      header z,x,y,chunk,stem,class (PLAN §6 frame), prints per-class /
      per-chunk counts and warns when a class spans < 4 chunks (the
      leave-one-chunk-out probe needs spread).

  cluster FEAT.npz [--k 8] [--per_cluster 16] [--crop 160]
      Cluster-then-name (replaces the hand-labeling session): k-means over the
      gated cells of a feature npz, then per-cluster montages of raw crops --
      one combined sheet + per-cluster close-ups. You only *name* the clusters:
      fill the `name` column of clustering/cluster_names_{name}.csv (same name
      on two rows merges them; leave blank or `junk` to drop) and run
      names2anchors. -> Output/clustering/clusters_{name}/ + assignments npz.

  names2anchors --clusters Output/clustering/clusters_{...}.npz
      Turns the filled-in names csv into clustering/anchors.csv: per named
      class, the highest-purity patches (fraction of gated cells in the
      class's clusters), spread over chunks. NOTE the circularity caveat: the
      feature that defined the clusters is favored by the resulting probe —
      compare probes *between* features, and read the confusion matrix.

  eval FEAT.npz [FEAT2.npz ...] [--anchors clustering/anchors.csv]
       [--shuffle-check] [--unit {cell,patch}]
      Scores each standard feature npz (see common.save_features):
        probe   leave-one-chunk-out kNN(5, cosine) + multinomial logistic on
                anchor classes; headline = patch-level majority-vote logistic
                accuracy (anchor labels broadcast to the patch's cells)
        amp     amp-leak: max |pearson| between the feature's top-3 PCA axes
                and raw cell std (> 0.8 disqualifies — the amplitude trap)
        coh     coherence: mean cosine of spatially adjacent cells minus
                random same-chunk pairs (label-free)
        zst     z-stability: same cell across adjacent chunks, same form
      Writes Output/clustering/harness/{featname}.json and prints a table.

  report
      Reloads every stored json, prints the ranked table plus the PLAN §7
      gate lines (F1 beats F4, F2 beats F1, amp-leak disqualifier).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common

ANCHORS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'anchors.csv')
PICKS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'picks.csv')
HARNESS_OUT = os.path.join(common.OUT, 'harness')
RNG = np.random.RandomState(0)
MAX_PAIRS = 200000


def make_sheets(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = os.path.join(common.OUT, 'sheets')
    os.makedirs(out, exist_ok=True)
    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(args.codec)
    for ch in chunks:
        sl, zmid = common.read_mid_slice(args.codec, ch)
        stems = common.list_stems(args.codec, ch)
        ds = max(1, max(sl.shape) // args.ds_target)
        sld = sl[::ds, ::ds]
        p = common.PATCH / ds
        vlo, vhi = np.percentile(sld, 1), np.percentile(sld, 99.5)
        fig, ax = plt.subplots(figsize=(24, 24 * sld.shape[0] / sld.shape[1]))
        ax.imshow(sld, cmap='gray', vmin=vlo, vmax=vhi, interpolation='nearest')
        for x in np.arange(0, sld.shape[1] + 1, p):
            ax.axvline(x, color='w', lw=0.4, alpha=0.5)
        for y in np.arange(0, sld.shape[0] + 1, p):
            ax.axhline(y, color='w', lw=0.4, alpha=0.5)
        for st in stems:
            r, c = common.stem_rc(st)
            ax.text(c * p + 2, r * p + 2, st, color='yellow', fontsize=4.5,
                    va='top', ha='left', alpha=0.9)
        ax.set_title(f'{ch}  zmid={zmid} — picking sheet: add lines '
                     f'"{ch},<stem>,<class>" to clustering/picks.csv')
        ax.set_xlim(0, sld.shape[1]), ax.set_ylim(sld.shape[0], 0)
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        png = os.path.join(out, f'{ch}.png')
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f'wrote {png}', flush=True)


def read_anchors(path):
    """anchors.csv -> list of dict rows (z, x, y, chunk, stem, class)."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.strip().split(','))))
    return rows


def write_anchors(path, rows):
    """rows = (z, x, y, chunk, stem, class); prints the per-class spread."""
    with open(path, 'w') as f:
        f.write('z,x,y,chunk,stem,class\n')
        for z, x, y, ch, st, cls in rows:
            f.write(f'{z:.0f},{x:.1f},{y:.1f},{ch},{st},{cls}\n')
    classes = sorted({r[5] for r in rows})
    print(f'wrote {path}: {len(rows)} anchors, {len(classes)} classes',
          flush=True)
    for cls in classes:
        chs = sorted({r[3] for r in rows if r[5] == cls})
        n = sum(r[5] == cls for r in rows)
        warn = ('   ** < 4 chunks — LOCO probe needs more spread **'
                if len(chs) < 4 else '')
        print(f'  {cls:>16}: {n:3d} patches over {len(chs):2d} chunks{warn}')


def anchors(args):
    if not os.path.exists(args.picks):
        sys.exit(f'no picks file {args.picks} — run make_sheets, label, then '
                 'write lines "chunk,stem,class" there')
    rows, bad = [], []
    with open(args.picks) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) != 3:
                bad.append(ln)
                continue
            ch, st, cls = parts
            if not os.path.exists(os.path.join(args.codec, ch, st + '.npz')):
                bad.append(ln + '   (no such codec patch)')
                continue
            z, x, y = common.patch_xyz(ch, st)
            rows.append((z, x, y, ch, st, cls))
    if bad:
        print('REJECTED lines:\n  ' + '\n  '.join(bad), flush=True)
    if not rows:
        sys.exit('no valid picks')
    write_anchors(args.anchors, rows)


# ------------------------------------------------------------------ versions

VERSIONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'versions.json')


def load_versions():
    if os.path.exists(VERSIONS_JSON):
        with open(VERSIONS_JSON) as f:
            return json.load(f)
    return {}


def save_versions(reg):
    with open(VERSIONS_JSON, 'w') as f:
        json.dump(reg, f, indent=2, sort_keys=True)


def child_name(reg, parent):
    for suffix in 'abcdefghijklmnopqrstuvwxyz':
        cand = parent + suffix
        if cand not in reg:
            return cand
    raise RuntimeError(f'no free child name under {parent}')


def cluster_names_csv(clusters_npz):
    """The names csv belonging to a clusters npz (may not exist yet)."""
    meta = json.loads(str(np.load(clusters_npz, allow_pickle=False)['meta']))
    return meta, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f'{meta["name"]}_names.csv')


def read_names_csv(path, k):
    names = {}
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#') or ln.startswith('cluster,'):
                    continue
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) >= 3 and parts[2]:
                    names[int(parts[0])] = parts[2]
    return {i: names.get(i, f'c{i}') for i in range(k)}


def version_cmd(args):
    """`version save --name v0 --clusters X.npz [--keep 0,7] [--note ...]`
    freezes a cluster round (upserts); `version list` prints the lineage."""
    reg = load_versions()
    if args.action == 'list':
        if not reg:
            sys.exit('no versions yet — harness.py version save --name v0 '
                     '--clusters <npz> [--keep ids]')
        roots = sorted(n for n, v in reg.items() if not v.get('parent'))
        def show(n, depth):
            v = reg[n]
            keep = v.get('keep')
            cls = v.get('classes', {})
            kept = ('all' if keep is None else
                    ', '.join(f'{i}:{cls.get(str(i), "c%d" % i)}'
                              for i in keep))
            print(f'{"  " * depth}{n}: k={v.get("k")} keep=[{kept}] '
                  f'{os.path.basename(v["clusters"])}'
                  + (f'  — {v["note"]}' if v.get('note') else ''))
            for c in sorted(cn for cn, cv in reg.items()
                            if cv.get('parent') == n):
                show(c, depth + 1)
        for r in roots:
            show(r, 0)
        return
    # save / upsert
    if not (args.name and args.clusters):
        sys.exit('version save needs --name and --clusters')
    meta, names_csv = cluster_names_csv(args.clusters)
    names = read_names_csv(names_csv, int(meta['k']))
    keep = ([int(x) for x in args.keep.split(',')] if args.keep
            else reg.get(args.name, {}).get('keep'))
    entry = dict(reg.get(args.name, {}))
    entry.update({
        'parent': args.parent if args.parent else entry.get('parent'),
        'clusters': os.path.abspath(args.clusters),
        'k': int(meta['k']), 'keep': keep,
        'classes': {str(i): n for i, n in names.items()},
        'feat': meta.get('feat')})
    if args.note:
        entry['note'] = args.note
    reg[args.name] = entry
    save_versions(reg)
    kept = ('all' if keep is None else
            ', '.join(f'{i}:{names[i]}' for i in keep))
    print(f'{args.name}: k={entry["k"]} keep=[{kept}]  '
          f'({os.path.basename(args.clusters)}) saved to versions.json',
          flush=True)


# ---------------------------------------------------------------- cluster-then-name

CLUSTER_HUES = np.array(
    [[0.90, 0.10, 0.10], [0.10, 0.45, 0.95], [0.10, 0.75, 0.25],
     [0.95, 0.75, 0.05], [0.75, 0.15, 0.85], [0.05, 0.80, 0.80],
     [0.95, 0.45, 0.05], [0.55, 0.35, 0.20], [0.60, 0.80, 0.10],
     [0.95, 0.40, 0.60], [0.35, 0.35, 0.95], [0.50, 0.50, 0.50]])


def _crop(sl, xc, yc, half):
    y0 = int(np.clip(yc - half, 0, sl.shape[0] - 2 * half))
    x0 = int(np.clip(xc - half, 0, sl.shape[1] - 2 * half))
    return sl[y0:y0 + 2 * half, x0:x0 + 2 * half]


def _norm_crop(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-09), 0, 1)


def cluster(args):
    """K-means over gated cells + per-cluster raw-crop montages, so the
    labeling session shrinks to naming ~k clusters from one figure.

    --within CLUSTERS.npz --keep 0,7 restricts to cells a previous cluster
    run assigned to the listed ids — hierarchical refinement: once a round
    separates material from background, the next round spends all its
    capacity inside the material."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.cluster import MiniBatchKMeans
    version = None
    if args.from_v:
        reg = load_versions()
        if args.from_v not in reg:
            sys.exit(f'unknown version {args.from_v} — see harness.py '
                     'version list')
        parent = reg[args.from_v]
        if parent.get('keep') is None:
            sys.exit(f'{args.from_v} has keep=all — set the kept classes '
                     f'first: version save --name {args.from_v} --clusters '
                     f'{parent["clusters"]} --keep <ids>')
        args.within = parent['clusters']
        args.keep = ','.join(str(i) for i in parent['keep'])
        version = child_name(reg, args.from_v)
        args.tag = args.tag or version
        print(f'refining {args.from_v} (keep {args.keep}) -> {version}',
              flush=True)
    fs = common.load_feature_set(args.feat)
    cell_px = int(fs.meta.get('cell_px', common.CELL))
    base = fs.meta.get('name') or os.path.splitext(os.path.basename(args.feat))[0]
    tag = f'_{args.tag}' if args.tag else ''
    name = f'clusters_{base}_k{args.k}{tag}'
    keep = common.row_gate(fs, args.codec, cell_px, args.gate_k)
    print(f'gate kept {keep.sum()}/{len(keep)} cells (k={args.gate_k})',
          flush=True)
    F = fs.feat[keep]
    ch, st = fs.chunk[keep], fs.stem[keep]
    cr, cc = fs.cell_row[keep], fs.cell_col[keep]
    if args.within:
        if not args.keep:
            sys.exit('--within needs --keep <cluster ids, e.g. 0,7>')
        wz = np.load(args.within, allow_pickle=False)
        keep_ids = {int(x) for x in args.keep.split(',')}
        wanted = {(c, s, int(r), int(q))
                  for c, s, r, q, l in zip(wz['chunk'].astype(str),
                                           wz['stem'].astype(str),
                                           wz['cell_row'], wz['cell_col'],
                                           wz['label'])
                  if int(l) in keep_ids}
        sub = np.array([(c, s, int(r), int(q)) in wanted
                        for c, s, r, q in zip(ch, st, cr, cc)])
        F, ch, st = F[sub], ch[sub], st[sub]
        cr, cc = cr[sub], cc[sub]
        print(f'--within kept {sub.sum()} cells (prev clusters '
              f'{sorted(keep_ids)} of {os.path.basename(args.within)})',
              flush=True)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=0, n_init=10,
                         batch_size=4096, max_iter=300).fit(F)
    lab = km.labels_
    dist = np.linalg.norm(F - km.cluster_centers_[lab], axis=1)
    sizes = np.bincount(lab, minlength=args.k)
    print('cluster sizes: ' + ' '.join(f'c{i}={n}' for i, n in enumerate(sizes)),
          flush=True)

    npz = os.path.join(common.OUT, f'{name}.npz')
    np.savez_compressed(
        npz, label=lab.astype(np.int16), dist=dist.astype(np.float32),
        chunk=ch, stem=st, cell_row=cr, cell_col=cc,
        centers=km.cluster_centers_.astype(np.float32),
        meta=json.dumps({'name': name, 'feat': os.path.abspath(args.feat),
                         'feat_name': base, 'k': args.k,
                         'gate_k': args.gate_k, 'cell_px': cell_px,
                         'within': args.within and os.path.abspath(args.within),
                         'keep': args.keep}))
    print(f'wrote {npz}', flush=True)
    if version:
        reg = load_versions()
        reg[version] = {'parent': args.from_v, 'clusters': os.path.abspath(npz),
                        'k': args.k, 'keep': None,
                        'classes': {str(i): f'c{i}' for i in range(args.k)},
                        'feat': os.path.abspath(args.feat)}
        save_versions(reg)
        print(f'registered version {version} (parent {args.from_v}); after '
              f'review: version save --name {version} --clusters {npz} '
              f'--keep <ids>', flush=True)

    # --- montages: per cluster, nearest-to-centroid members, chunk-diverse ---
    fig_dir = os.path.join(common.OUT, name)
    os.makedirs(fig_dir, exist_ok=True)
    half = args.crop // 2
    slices = {}

    def mid_slice(c):
        if c not in slices:
            slices[c] = common.read_mid_slice(args.codec, c)[0]
        return slices[c]

    def pick_members(ci, count):
        m = np.where(lab == ci)[0]
        m = m[np.argsort(dist[m])]
        per_chunk_cap = max(1, count // 4)
        taken, used = [], {}
        for i in m:
            c = ch[i]
            if used.get(c, 0) >= per_chunk_cap:
                continue
            used[c] = used.get(c, 0) + 1
            taken.append(i)
            if len(taken) == count:
                break
        return taken

    ncols = args.per_cluster
    fig, axes = plt.subplots(args.k, ncols,
                             figsize=(1.4 * ncols, 1.5 * args.k))
    axes = np.atleast_2d(axes)
    for ci in range(args.k):
        members = pick_members(ci, ncols)
        for j in range(ncols):
            ax = axes[ci, j]
            ax.set_xticks([]), ax.set_yticks([])
            if j >= len(members):
                ax.axis('off')
                continue
            i = members[j]
            _, xc, yc = common.cell_xyz(ch[i], st[i], cr[i], cc[i], cell_px)
            ax.imshow(_norm_crop(_crop(mid_slice(ch[i]), xc, yc, half)),
                      cmap='gray', vmin=0, vmax=1)
            for sp in ax.spines.values():
                sp.set_color(CLUSTER_HUES[ci % len(CLUSTER_HUES)])
                sp.set_linewidth(3)
            if j == 0:
                ax.set_ylabel(f'c{ci}\nn={sizes[ci]}', fontsize=9,
                              rotation=0, ha='right', va='center')
    fig.suptitle(f'{name} — {args.crop}px crops around cluster cells '
                 f'(nearest-centroid, chunk-diverse). Name the rows in the '
                 f'csv, then run names2anchors.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(common.OUT, f'{name}_montage.png')
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f'wrote {png}', flush=True)

    for ci in range(args.k):  # close-ups, 32 crops each
        members = pick_members(ci, 32)
        fig, axes = plt.subplots(4, 8, figsize=(14, 7.6))
        for ax, i in zip(axes.ravel(), members):
            _, xc, yc = common.cell_xyz(ch[i], st[i], cr[i], cc[i], cell_px)
            ax.imshow(_norm_crop(_crop(mid_slice(ch[i]), xc, yc, half)),
                      cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{ch[i]} {st[i]}', fontsize=5)
        for ax in axes.ravel():
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f'{name} c{ci} (n={sizes[ci]})')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(fig_dir, f'c{ci}.png'), dpi=110)
        plt.close(fig)
    print(f'wrote {fig_dir}/c*.png close-ups', flush=True)

    names_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'{name}_names.csv')
    if not os.path.exists(names_csv):
        with open(names_csv, 'w') as f:
            f.write('# fill `name` per cluster; same name merges clusters; '
                    'blank or junk drops it\ncluster,count,name\n')
            for ci in range(args.k):
                f.write(f'{ci},{sizes[ci]},\n')
        print(f'\nNEXT: look at {png}\n      name the rows in {names_csv}\n'
              f'      then: python clustering/harness.py names2anchors '
              f'--clusters {npz}', flush=True)


def names2anchors(args):
    z = np.load(args.clusters, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    names_csv = args.names or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f'{meta["name"]}_names.csv')
    if not os.path.exists(names_csv):
        sys.exit(f'no names file {names_csv} — run the cluster subcommand '
                 'first and fill in the name column')
    cls_of = {}
    with open(names_csv) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#') or ln.startswith('cluster,'):
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) >= 3 and parts[2] and parts[2].lower() != 'junk':
                cls_of[int(parts[0])] = parts[2]
    if not cls_of:
        sys.exit(f'{names_csv} has no named clusters yet')
    classes = sorted(set(cls_of.values()))
    print(f'named classes: {classes} (from {len(cls_of)} clusters)', flush=True)

    lab, ch, st = z['label'], z['chunk'].astype(str), z['stem'].astype(str)
    pid = np.array([f'{c}/{s}' for c, s in zip(ch, st)])
    uniq, inv = np.unique(pid, return_inverse=True)
    n_cls = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    votes = np.zeros((len(uniq), n_cls + 1))  # last col = unnamed clusters
    for ci in range(int(lab.max()) + 1):
        col = cls_idx.get(cls_of.get(ci, None), n_cls)
        np.add.at(votes[:, col], inv[lab == ci], 1)
    tot = votes.sum(1)
    frac = votes[:, :n_cls] / tot[:, None]
    best = frac.argmax(1)
    purity = frac[np.arange(len(uniq)), best]

    rows = []
    for k, cls in enumerate(classes):
        cand = np.where((best == k) & (purity >= args.min_frac)
                        & (tot >= args.min_cells))[0]
        cand = cand[np.argsort(-purity[cand])]
        per_chunk_cap = max(2, args.per_class // 6)
        used, taken = {}, []
        for i in cand:
            c = uniq[i].split('/')[0]
            if used.get(c, 0) >= per_chunk_cap:
                continue
            used[c] = used.get(c, 0) + 1
            taken.append(i)
            if len(taken) == args.per_class:
                break
        if len(taken) < args.per_class:
            print(f'  note: {cls} has only {len(taken)} qualifying patches '
                  f'(purity >= {args.min_frac}, cells >= {args.min_cells})',
                  flush=True)
        for i in taken:
            c, s = uniq[i].split('/')
            zz, x, y = common.patch_xyz(c, s)
            rows.append((zz, x, y, c, s, cls))
    if not rows:
        sys.exit('no patches passed the purity/count thresholds')
    write_anchors(args.anchors, rows)
    print(f'\ncircularity note: these anchors come from clusters of '
          f'{meta["feat_name"]} — its own probe is optimistic; compare '
          f'features against each other and check the confusion matrix.',
          flush=True)


# ------------------------------------------------------------------- metrics

def _cosine(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


def probe_loco(F, y, groups, patch_id):
    """Leave-one-chunk-out probe. Returns dict with cell/patch accuracies for
    kNN + logistic and the pooled logistic confusion matrix."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    classes = np.unique(y)
    conf = np.zeros((len(classes), len(classes)), int)
    accs = {'knn_cell': [], 'lin_cell': [], 'knn_patch': [], 'lin_patch': []}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(F[tr])
        Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
        knn = KNeighborsClassifier(5, metric='cosine').fit(Xtr, y[tr])
        lin = LogisticRegression(max_iter=1000, multi_class='multinomial',
                                 C=1.0).fit(Xtr, y[tr])
        for name, mdl in (('knn', knn), ('lin', lin)):
            pred = mdl.predict(Xte)
            accs[f'{name}_cell'].append((pred == y[te]).mean())
            votes = {}
            for pid, pr, yy in zip(patch_id[te], pred, y[te]):
                votes.setdefault(pid, ([], yy))[0].append(pr)
            ok = [max(set(v), key=v.count) == yy for v, yy in votes.values()]
            accs[f'{name}_patch'].append(np.mean(ok))
            if name == 'lin':
                for pr, yy in zip(pred, y[te]):
                    conf[np.searchsorted(classes, yy),
                         np.searchsorted(classes, pr)] += 1
    out = {k: float(np.mean(v)) if v else float('nan') for k, v in accs.items()}
    out.update(classes=classes.tolist(), confusion=conf.tolist())
    return out


def amp_leak(F, raw_std):
    from sklearn.decomposition import PCA
    k = min(3, F.shape[1], len(F) - 1)
    pcs = PCA(k, random_state=0).fit_transform((F - F.mean(0)) / (F.std(0) + 1e-09))
    return float(max(abs(np.corrcoef(pcs[:, j], raw_std)[0, 1])
                     for j in range(k)))


def _pair_delta(F, idx_a, idx_b, n_random):
    if len(idx_a) == 0:
        return float('nan')
    if len(idx_a) > MAX_PAIRS:
        keep = RNG.choice(len(idx_a), MAX_PAIRS, replace=False)
        idx_a, idx_b = idx_a[keep], idx_b[keep]
    adj = _cosine(F[idx_a], F[idx_b]).mean()
    ra = RNG.randint(0, len(F), min(n_random, MAX_PAIRS))
    rb = RNG.randint(0, len(F), min(n_random, MAX_PAIRS))
    rnd = _cosine(F[ra], F[rb]).mean()
    return float(adj - rnd)


def grid_index(fs, cell_px):
    """Row index arrays: per-row global cell coordinates (chunk_i, gy, gx)."""
    n = common.ncell(cell_px)
    chunks = sorted(set(fs.chunk))
    ch_map = {c: i for i, c in enumerate(chunks)}
    ch_i = np.array([ch_map[c] for c in fs.chunk])
    rr = np.array([common.stem_rc(s)[0] for s in fs.stem])
    cc = np.array([common.stem_rc(s)[1] for s in fs.stem])
    cell_r = np.where(fs.cell_row < 0, 0, fs.cell_row)
    cell_c = np.where(fs.cell_col < 0, 0, fs.cell_col)
    return chunks, ch_i, rr * n + cell_r, cc * n + cell_c


def adjacency_pairs(chunks, ch_i, gy, gx, dz=0):
    """Index pairs (a, b) of rows adjacent in +x/+y (dz=0) or same-cell in the
    next chunk (dz=1)."""
    key = {}
    for i, (c, y, x) in enumerate(zip(ch_i, gy, gx)):
        key[c, y, x] = i
    a, b = [], []
    offsets = [(0, 0, 1), (0, 1, 0)] if dz == 0 else [(1, 0, 0)]
    for (c, y, x), i in key.items():
        for dc, dy, dx in offsets:
            j = key.get((c + dc, y + dy, x + dx))
            if j is not None:
                a.append(i), b.append(j)
    return np.array(a, int), np.array(b, int)


def raw_std_for_rows(fs, codec, cell_px):
    """Per-row raw std from the cached mid-slice stats (cell or patch level)."""
    out = np.empty(len(fs.feat), np.float32)
    for ch in sorted(set(fs.chunk)):
        stats = common.raw_cell_stats(codec, ch, cell_px)
        stem_i = {s: i for i, s in enumerate(stats['stems'])}
        m = fs.chunk == ch
        si = np.array([stem_i[s] for s in fs.stem[m]])
        cr, cc = fs.cell_row[m], fs.cell_col[m]
        out[m] = np.where(cr < 0, stats['patch_std'][si],
                          stats['cell_std'][si, np.maximum(cr, 0),
                                            np.maximum(cc, 0)])
    return out


def eval_features(args):
    os.makedirs(HARNESS_OUT, exist_ok=True)
    anchor_rows = read_anchors(args.anchors) if os.path.exists(args.anchors) else None
    if anchor_rows is None:
        print(f'NOTE: no {args.anchors} yet — probe skipped; amp/coh/zst '
              'still computed', flush=True)
    results = []
    for path in args.feats:
        fs = common.load_feature_set(path)
        name = fs.meta.get('name') or os.path.splitext(os.path.basename(path))[0]
        cell_px = int(fs.meta.get('cell_px', common.CELL))
        F = fs.feat
        if args.unit == 'patch' and (fs.cell_row >= 0).any():
            pid = np.array([f'{c}/{s}' for c, s in zip(fs.chunk, fs.stem)])
            uniq, inv = np.unique(pid, return_inverse=True)
            Fp = np.zeros((len(uniq), F.shape[1]), np.float32)
            np.add.at(Fp, inv, F)
            F = Fp / np.bincount(inv)[:, None]
            ch = np.array([u.split('/')[0] for u in uniq])
            st = np.array([u.split('/')[1] for u in uniq])
            fs = common.FeatureSet(F, ch, st,
                                   np.full(len(uniq), -1, np.int16),
                                   np.full(len(uniq), -1, np.int16), fs.meta)
        res = {'name': name, 'family': fs.meta.get('family', '?'),
               'path': os.path.abspath(path), 'n_rows': int(len(F)),
               'dim': int(F.shape[1]), 'cell_px': cell_px,
               'unit': args.unit, 'gate_k': args.gate_k}
        keep = (common.row_gate(fs, args.codec, cell_px, args.gate_k)
                if args.gate_k > 0 else np.ones(len(F), bool))
        res['gate_kept'] = int(keep.sum())
        gfs = common.FeatureSet(F[keep], fs.chunk[keep], fs.stem[keep],
                                fs.cell_row[keep], fs.cell_col[keep], fs.meta)
        rstd = raw_std_for_rows(gfs, args.codec, cell_px)
        res['amp_leak'] = amp_leak(gfs.feat, rstd)
        chunks, ch_i, gy, gx = grid_index(gfs, cell_px)
        a, b = adjacency_pairs(chunks, ch_i, gy, gx, dz=0)
        res['coherence'] = _pair_delta(gfs.feat, a, b, len(a))
        a, b = adjacency_pairs(chunks, ch_i, gy, gx, dz=1)
        res['z_stability'] = _pair_delta(gfs.feat, a, b, max(len(a), 1))
        if anchor_rows is not None:
            lab = {(r['chunk'], r['stem']): r['class'] for r in anchor_rows}
            y = np.array([lab.get((c, s), '')
                          for c, s in zip(fs.chunk, fs.stem)])
            m = y != ''
            if m.sum() == 0:
                print(f'{name}: anchors do not intersect features', flush=True)
            else:
                pid = np.array([f'{c}/{s}'
                                for c, s in zip(fs.chunk[m], fs.stem[m])])
                pr = probe_loco(F[m], y[m], fs.chunk[m], pid)
                res.update({'probe': pr['lin_patch'], **pr})
                if args.shuffle_check:
                    ys = y[m].copy()
                    up, ui = np.unique(pid, return_inverse=True)
                    perm = RNG.permutation(len(up))
                    first = np.zeros(len(up), int)
                    first[ui[::-1]] = np.arange(len(ui))[::-1]
                    ys = ys[first][perm][ui]
                    res['probe_shuffled'] = probe_loco(
                        F[m], ys, fs.chunk[m], pid)['lin_patch']
        with open(os.path.join(HARNESS_OUT, f'{name}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        results.append(res)
    print_table(results)


GATES = [
    ('f1', 'f4', 'M2: F1 metacode hists must beat the F4 pooled baseline'),
    ('f2', 'f1', 'M3: F2 co-occurrence must beat F1'),
    ('f1+f2', 'f1', 'M3b: concat must beat F1 to justify itself')]


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '    -'
    return f'{v:5.3f}'


def print_table(results):
    results = sorted(results, key=lambda r: -(r.get('probe') or -1))
    print(f'\n{"name":<24} {"probe":>6} {"knn":>6} {"amp":>6} {"coh":>6} '
          f'{"zst":>6}  verdict')
    for r in results:
        rej = '  REJECT (amp-leak > 0.8)' if r['amp_leak'] > 0.8 else ''
        print(f'{r["name"]:<24} {fmt(r.get("probe")):>6} '
              f'{fmt(r.get("knn_patch")):>6} {fmt(r["amp_leak"]):>6} '
              f'{fmt(r["coherence"]):>6} {fmt(r["z_stability"]):>6}{rej}')
        if r.get('probe_shuffled') is not None:
            print(f'{"":<24} shuffled-label probe {r["probe_shuffled"]:.3f} '
                  '(chance check)')
    print()


def report(args):
    results = []
    for f in (sorted(os.listdir(HARNESS_OUT)) if os.path.isdir(HARNESS_OUT)
              else []):
        if f.endswith('.json'):
            with open(os.path.join(HARNESS_OUT, f)) as fh:
                results.append(json.load(fh))
    if not results:
        sys.exit(f'no results in {HARNESS_OUT} — run eval first')
    print_table(results)
    by_fam = {}
    for r in results:
        fam = r.get('family', '?')
        if r.get('probe') is not None:
            if r['probe'] > by_fam.get(fam, {}).get('probe', -1):
                by_fam[fam] = r
    for fam, base, label in GATES:
        a, b = by_fam.get(fam), by_fam.get(base)
        if a is None or b is None or a.get('probe') is None:
            print(f'{label}: PENDING (missing {fam if a is None else base})')
        else:
            ok = a['probe'] > b['probe']
            print(f'{label}: {a["name"]} {a["probe"]:.3f} vs {b["name"]} '
                  f'{b["probe"]:.3f} -> {"PASS" if ok else "FAIL"}')
    best = max((r for r in results if r.get('probe') is not None),
               key=lambda r: r['probe'], default=None)
    if best and best.get('confusion'):
        print(f'\nbest probe confusion ({best["name"]}), classes '
              f'{best["classes"]}:')
        for row in best['confusion']:
            print('   ', row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('make_sheets')
    s.add_argument('--chunks', default=None)
    s.add_argument('--ds_target', type=int, default=3600)
    s = sub.add_parser('anchors')
    s.add_argument('--picks', default=PICKS_DEFAULT)
    s.add_argument('--anchors', default=ANCHORS_DEFAULT)
    s = sub.add_parser('cluster')
    s.add_argument('feat')
    s.add_argument('--k', type=int, default=8)
    s.add_argument('--gate_k', type=float, default=2.0)
    s.add_argument('--per_cluster', type=int, default=16,
                   help='crops per cluster row in the combined montage')
    s.add_argument('--crop', type=int, default=160,
                   help='crop size (px) around each cell centre')
    s.add_argument('--within', default=None,
                   help='clusters_{...}.npz of a previous round: only cells '
                        'it assigned to --keep ids are re-clustered')
    s.add_argument('--keep', default=None,
                   help='comma-separated cluster ids kept from --within')
    s.add_argument('--tag', default=None,
                   help='suffix for the output name (e.g. fiber)')
    s.add_argument('--from', dest='from_v', default=None,
                   help='refine a registered version (versions.json): sets '
                        '--within/--keep from it and auto-names the child '
                        '(v0 -> v0a)')
    s = sub.add_parser('version')
    s.add_argument('action', choices=('save', 'list'))
    s.add_argument('--name', default=None)
    s.add_argument('--clusters', default=None)
    s.add_argument('--keep', default=None,
                   help='class ids that carry to the next refinement')
    s.add_argument('--parent', default=None)
    s.add_argument('--note', default=None)
    s = sub.add_parser('names2anchors')
    s.add_argument('--clusters', required=True,
                   help='clusters_{...}.npz written by the cluster subcommand')
    s.add_argument('--names', default=None,
                   help='names csv (default: clustering/{name}_names.csv)')
    s.add_argument('--per_class', type=int, default=25)
    s.add_argument('--min_frac', type=float, default=0.5,
                   help='min fraction of a patch\'s gated cells in the class')
    s.add_argument('--min_cells', type=int, default=8,
                   help='min gated cells for a patch to qualify')
    s.add_argument('--anchors', default=ANCHORS_DEFAULT)
    s = sub.add_parser('eval')
    s.add_argument('feats', nargs='+')
    s.add_argument('--anchors', default=ANCHORS_DEFAULT)
    s.add_argument('--unit', choices=('cell', 'patch'), default='cell')
    s.add_argument('--gate_k', type=float, default=2.0,
                   help='raw-contrast gate for the label-free metrics '
                        '(0 = ungated)')
    s.add_argument('--shuffle-check', action='store_true')
    sub.add_parser('report')
    args = ap.parse_args()
    {'make_sheets': make_sheets, 'anchors': anchors, 'cluster': cluster,
     'names2anchors': names2anchors, 'eval': eval_features,
     'report': report, 'version': version_cmd}[args.cmd](args)


if __name__ == '__main__':
    main()
