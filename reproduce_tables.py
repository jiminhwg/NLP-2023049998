"""Recompute the main tables from the saved result files.

Usage:  python reproduce_tables.py --results_dir results/

Needs main_textfooler.jsonl, main_llm_v3.jsonl and judge_cache.json in the results directory.
Needs only pandas and numpy (no GPU, no model downloads).
Not covered here: the NLI check and the hand-label counts (the hand-check numbers printed at the end are the counts
from the hand-labeled sample, entered by hand; see notebook Sections 6.2-6.4).
"""
import argparse, difflib, json, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
CLF = ['BERT', 'RoBERTa', 'DistilBERT']     # same order as the notebook, which fixes the bootstrap draws
METHODS = ['textfooler', 'llm']
THR = 0.80
B = 2000


def load(results_dir):
    tf = pd.read_json(os.path.join(results_dir, 'main_textfooler.jsonl'), lines=True)
    ll = pd.read_json(os.path.join(results_dir, 'main_llm_v3.jsonl'), lines=True)
    jc = json.load(open(os.path.join(results_dir, 'judge_cache.json')))
    df = pd.concat([tf, ll], ignore_index=True)
    for t in CLF:
        if 'fool_' + t not in df:
            df['fool_' + t] = np.nan
    df['judge_ok'] = [bool(jc[a] == l) if f else False for a, l, f in zip(df.adv, df.label, df.fooled_source)]
    df['sim_ok'] = df.fooled_source & (df.sim.fillna(0) >= THR)
    df['success'] = df.sim_ok & df.judge_ok
    return df


def source_success(df, n):
    print(f'=== Source success (N = {n} sentences) ===')
    g = df.groupby(['method', 'source']).agg(fooled=('fooled_source', 'sum'), sim_ok=('sim_ok', 'sum'), valid=('success', 'sum'))
    for c in ('fooled', 'sim_ok', 'valid'):
        g[c + '_rate'] = (g[c] / n).round(3)
    print(g.to_string())
    t = df[(df.method == 'textfooler') & df.sim_ok]
    print(f'TextFooler similarity-passing successes rejected by the judge: {(~t.judge_ok).sum()} of {len(t)} = {1 - t.judge_ok.mean():.3f}')
    ll = df[df.method == 'llm']
    print(f'LLM: mean judge-rejected candidates per attack = {ll.n_flipped.mean():.3f}, mean LLM calls per attack = {ll.llm_calls.mean():.3f}')


def per_pair(df, sids):
    print('\n=== Per-pair transfer (conditional), 95% bootstrap CIs over sentences ===')

    def arrays(m, s, t):
        d = df[(df.method == m) & (df.source == s)].drop_duplicates('sid').set_index('sid').reindex(sids)
        succ = d.success.eq(True).values
        fool = succ & d['fool_' + t].eq(True).values
        return succ.astype(float), fool.astype(float)

    rng = np.random.RandomState(0)
    rows = []
    for s in CLF:
        for t in CLF:
            if s == t:
                continue
            a = {m: arrays(m, s, t) for m in METHODS}
            idx = rng.randint(0, len(sids), (B, len(sids)))
            full = np.arange(len(sids))

            def rate(m, ix):
                num, den = a[m][1][ix].sum(-1), a[m][0][ix].sum(-1)
                return np.where(den > 0, num / np.maximum(den, 1), np.nan)

            d = rate('llm', idx) - rate('textfooler', idx)
            ci = lambda m: np.nanpercentile(rate(m, idx), [2.5, 97.5]).round(3).tolist()
            rows.append(dict(
                pair=f'{s}->{t}',
                estimable=bool(a['textfooler'][0].sum() >= 10 and a['llm'][0].sum() >= 10),
                n_src_tf=int(a['textfooler'][0].sum()), n_src_llm=int(a['llm'][0].sum()),
                transfer_tf=round(float(rate('textfooler', full)), 3), ci_tf=ci('textfooler'),
                transfer_llm=round(float(rate('llm', full)), 3), ci_llm=ci('llm'),
                diff=round(float(rate('llm', full) - rate('textfooler', full)), 3),
                diff_lo=round(float(np.nanpercentile(d, 2.5)), 3), diff_hi=round(float(np.nanpercentile(d, 97.5)), 3),
                uncond_tf=round(float(a['textfooler'][1].sum() / len(sids)), 3),
                uncond_llm=round(float(a['llm'][1].sum() / len(sids)), 3)))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    e = res[res.estimable]
    print(f'Estimable pairs: {len(e)} of 6. LLM higher with the 95% CI of the difference excluding 0: '
          f'{int(((e["diff"] > 0) & (e.diff_lo > 0)).sum())}')
    return res


def pooled(df, sids, thr):
    d = df.copy()
    d['succ'] = d.fooled_source & (d.sim.fillna(0) >= thr) & d.judge_ok
    d['nfool'] = d[['fool_' + t for t in CLF]].eq(True).sum(axis=1) * d.succ
    g = {m: d[d.method == m].groupby('sid')[['succ', 'nfool']].sum().reindex(sids).fillna(0) for m in METHODS}
    idx = np.random.RandomState(1).randint(0, len(sids), (B, len(sids)))
    full = np.arange(len(sids))

    def cond(m, ix):     # target-fooling outcomes / (2 x source successes)
        s_, f_ = g[m].succ.values[ix].sum(-1), g[m].nfool.values[ix].sum(-1)
        return np.where(s_ > 0, f_ / np.maximum(2 * s_, 1), np.nan)

    def unc(m, ix):      # target-fooling outcomes / (6 x number of sentences)
        return g[m].nfool.values[ix].sum(-1) / (6 * len(sids))

    print(f'\n--- similarity threshold {thr}: valid examples TF/LLM = '
          f'{int(g["textfooler"].succ.sum())}/{int(g["llm"].succ.sum())} ---')
    for name, fn in (('conditional (pooled)', cond), ('unconditional (pooled)', unc)):
        dd = fn('llm', idx) - fn('textfooler', idx)
        tfv, llv = float(fn('textfooler', full)), float(fn('llm', full))
        print(f'{name}: TextFooler {tfv:.3f} {np.nanpercentile(fn("textfooler", idx), [2.5, 97.5]).round(3).tolist()} | '
              f'LLM {llv:.3f} {np.nanpercentile(fn("llm", idx), [2.5, 97.5]).round(3).tolist()} | '
              f'diff {llv - tfv:+.3f} [{np.nanpercentile(dd, 2.5):+.3f}, {np.nanpercentile(dd, 97.5):+.3f}]')


def queries(df):
    print('\n=== Classifier queries ===')
    print('Mean per attack, including failures:')
    print(df.groupby(['method', 'source']).queries.mean().round(1).unstack().to_string())
    print('Mean per valid example:')
    print(df[df.success].groupby(['method', 'source']).queries.mean().round(1).unstack().to_string())
    for m in METHODS:
        q = df[df.method == m].queries
        print(f'{m}: mean {q.mean():.1f}, median {q.median():.0f}, max {q.max()}')


def edit_size(df):
    print('\n=== Edit size of valid examples ===')
    er = lambda a, b: 1 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
    ok = df[df.success].copy()
    ok['edit'] = [er(a, b) for a, b in zip(ok.text, ok.adv)]
    ok['transfer'] = ok[['fool_' + t for t in CLF]].astype(float).mean(axis=1)
    print(ok.groupby('method')[['edit', 'sim']].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(3).T.to_string())
    # terciles by rank; bin sizes can differ by one example from the notebook's pd.qcut, depending on the pandas version
    r = ok.edit.rank(method='first').astype(int)
    ok['tercile'] = pd.cut((r - 1) * 3 // len(ok), [-1, 0, 1, 2], labels=['small', 'mid', 'large'])
    print('\nTransfer within edit-size terciles:')
    print(ok.groupby(['tercile', 'method'], observed=True).transfer.agg(['mean', 'count']).round(3).to_string())


def _rate(d):
    return d.nfool.sum() / (2 * len(d)) if len(d) else float('nan')


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def analysis_extras(df, sids):
    """Exploratory analyses used in the Analysis section."""
    df = df.copy()
    df['nfool'] = df[['fool_' + t for t in CLF]].eq(True).sum(axis=1)

    print('\n=== Exploratory: what the sentiment judge removes (TextFooler) ===')
    tf = df[df.method == 'textfooler']
    acc, rej = tf[tf.success], tf[tf.sim_ok & ~tf.judge_ok]

    def by_sid(d):
        g = d.groupby('sid').agg(n=('nfool', 'size'), f=('nfool', 'sum')).reindex(sids).fillna(0)
        return g.n.values, g.f.values

    (na, fa), (nr, fr) = by_sid(acc), by_sid(rej)
    idx = np.random.RandomState(2).randint(0, len(sids), (B, len(sids)))
    ra = fa[idx].sum(1) / (2 * np.maximum(na[idx].sum(1), 1))
    rr = fr[idx].sum(1) / (2 * np.maximum(nr[idx].sum(1), 1))
    lo, hi = np.percentile(rr - ra, [2.5, 97.5])
    print(f'similarity-passing successes: {len(acc) + len(rej)} = accepted {len(acc)} + rejected {len(rej)}')
    print(f'pooled transfer: accepted {_rate(acc):.3f}, rejected {_rate(rej):.3f}, similarity-only {_rate(pd.concat([acc, rej])):.3f}')
    print(f'gap (rejected - accepted) = {_rate(rej) - _rate(acc):.3f}, 95% CI [{lo:.3f}, {hi:.3f}]')

    print('\n=== Exploratory: concentration of valid examples across sentences ===')
    for m in METHODS:
        v = df[(df.method == m) & df.success]
        per = v.groupby('sid').source.nunique()
        print(f'{m}: valid events {len(v)}, distinct sentences {len(per)}, valid on 1/2/3 sources: '
              f'{(per == 1).sum()}/{(per == 2).sum()}/{(per == 3).sum()}')

    print('\n=== Exploratory: examples whose exact text is valid on another source ===')
    v = df[df.success].copy()
    v['shared'] = v.groupby(['method', 'sid', 'adv']).source.transform('nunique') >= 2
    for m in METHODS:
        for flag in (True, False):
            d = v[(v.method == m) & (v.shared == flag)]
            print(f'{m:10s} shared={flag!s:5s} n={len(d):3d} pooled transfer={_rate(d):.3f}')
    u = v[~v.shared]
    g = {m: u[u.method == m].groupby('sid').agg(n=('nfool', 'size'), f=('nfool', 'sum')).reindex(sids).fillna(0) for m in METHODS}
    idx = np.random.RandomState(3).randint(0, len(sids), (B, len(sids)))
    full = np.arange(len(sids))
    r = lambda m, ix: g[m].f.values[ix].sum(-1) / (2 * np.maximum(g[m].n.values[ix].sum(-1), 1))
    d = r('llm', idx) - r('textfooler', idx)
    print(f'unique-text only: TextFooler {float(r("textfooler", full)):.3f}, LLM {float(r("llm", full)):.3f}, '
          f'diff {float(r("llm", full) - r("textfooler", full)):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')

    print('\n=== Exploratory: LLM valid examples by round of success ===')
    ll = v[v.method == 'llm']
    ll = ll.assign(rnd=pd.cut(ll.llm_calls, [0, 1, 2, 5, 10], labels=['1', '2', '3-5', '6-10']))
    print(ll.groupby('rnd', observed=True).agg(n=('nfool', 'size'), transfer=('nfool', lambda x: x.sum() / (2 * len(x)))).round(3).to_string())

    print('\n=== Hand-check counts (from the hand-labeled sample; Wilson 95% intervals) ===')
    for lab, k, n in (('TextFooler strict', 15, 30), ('LLM strict', 20, 30), ('TextFooler lenient', 15, 20), ('LLM lenient', 20, 26),
                      ('TextFooler reversed', 5, 30), ('LLM reversed', 6, 30), ('TextFooler unclear', 10, 30), ('LLM unclear', 4, 30)):
        lo, hi = _wilson(k, n)
        print(f'{lab:20s} {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]')


def random_examples(df, path):
    """Seeded random sample for the Analysis: 3 valid and 3 failed attacks per method (random_state=0)."""
    def outcome(r):
        if r.success:
            return 'valid'
        if r.method == 'llm':
            return 'failed (no valid rewrite within 10 rounds)'
        if not r.fooled_source:
            return 'failed (source not fooled)'
        if r.sim < THR:
            return 'failed (similarity < 0.80)'
        return 'failed (judge: sentiment changed)'

    allx = df.copy()
    allx['outcome'] = allx.apply(outcome, axis=1)
    rows = [allx[(allx.method == m) & (allx.success == ok)].sample(3, random_state=0)
            for m in METHODS for ok in (True, False)]
    ex = pd.concat(rows)[['method', 'source', 'sid', 'outcome', 'sim', 'queries', 'text', 'adv']]
    ex['sim'] = ex.sim.round(3)
    ex.to_csv(path, index=False)
    print(f'\nwrote {len(ex)} random examples to {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out_csv', default=None, help='optional path for the per-pair table')
    p.add_argument('--examples_csv', default=None, help='optional path for the seeded random examples')
    args = p.parse_args()
    df = load(args.results_dir)
    sids = np.array(sorted(df.sid.unique()))
    source_success(df, len(sids))
    res = per_pair(df, sids)
    print('\n=== Pooled transfer (exploratory) ===')
    for th in (0.80, 0.85):
        pooled(df, sids, th)
    queries(df)
    edit_size(df)
    analysis_extras(df, sids)
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
    if args.examples_csv:
        random_examples(df, args.examples_csv)
