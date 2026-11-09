"""Instantiate candidates for one Bitbrains prefix under explicit deployment budgets."""
import argparse
import tempfile
from pathlib import Path

from .io import DATA, ROOT, read, sha, write
from .prefix import prefix_selection
from .runtime import ARCH, build, np, pred, profile_arch, torch, update
from .selection import cases, choose
from .source_training import cpu_state
from .target import PrefixFence, tensorhash


def feasible_records(bank, candidates, horizon, budget):
    operation_cap, parameter_cap = bank['budgets'][budget]
    records = []
    for token in candidates:
        idx = bank['arch_indices'][token]
        params, operations = bank['costs'][token][str(horizon)]
        if not all(isinstance(cost, int) and cost > 0 for cost in (params, operations)):
            raise ValueError(f'Invalid resource profile for {token}')
        if params <= parameter_cap and operations <= operation_cap:
            records.append(dict(token=token, arch_index=idx, params=params, operations=operations))
    return records


def instantiate(case, bank, records, policy, data, output_model):
    """Adapt only the supplied feasible records. Target labels after selection stay sealed."""
    horizon, seed = case['H'], case['seed']
    xs, ys, xv = (torch.from_numpy(data[k]) for k in ('Xs', 'ys', 'Xv'))
    states = {}
    rows = []
    for record in records:
        token, idx = record['token'], record['arch_index']
        if profile_arch(ARCH[idx], 96, 25, horizon) != (record['params'], record['operations']):
            raise ValueError(f'Resource profile differs from the architecture: {token}')
        path = ROOT / bank['checkpoints'][token][str(horizon)]
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = build(idx, horizon, checkpoint.get('state_dict', checkpoint), seed)
        update(model, xs, ys, 50, seed)
        errors = pred(model, xv, idx).astype(np.float64) - data['yv'].astype(np.float64)
        loss = float(np.square(errors).mean(1).mean())
        if not np.isfinite(loss) or loss <= 0:
            raise ValueError(f'Invalid validation loss for {token}: {loss}')
        adapted = cpu_state(model)
        rows.append(dict(record, validation_mean=loss, n_validation=len(errors),
                         source_sha256=sha(path), tensor_sha256=tensorhash(adapted)))
        if output_model is not None:
            states[token] = adapted
    ref = policy['reference']
    selected = ref if policy['policy'] == 'pure_reference' else choose(rows, ref, policy['tau'])
    if output_model is not None:
        torch.save(dict(token=selected, horizon=horizon, state_dict=states[selected]), output_model)
    return dict(feasible_before_loading=[r['token'] for r in records],
                loaded_and_adapted=[r['token'] for r in rows], updates_per_model=50,
                selected=selected, rows=rows)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True, help='Bank definition JSON with budgets and checkpoints')
    parser.add_argument('--policy', type=Path, required=True, help='Frozen policy JSON')
    parser.add_argument('--policy-key', default='main', help='Named policy in the policies map')
    parser.add_argument('--role', choices=['development', 'calibration', 'evaluation'], required=True)
    parser.add_argument('--case', required=True, help='One case slug in the role manifest')
    parser.add_argument('--budgets', nargs='+', required=True, help='Budget names in the bank definition')
    parser.add_argument('--output', type=Path, required=True, help='New output directory outside the frozen package')
    parser.add_argument('--save-models', action='store_true', help='Also save the selected model for each budget')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        parser.error('--output must be a new directory outside the frozen package')
    bank, document = read(args.bank), read(args.policy)
    policy = document['policies'][args.policy_key] if 'policies' in document else document
    if policy['policy'] not in ('selector', 'pure_reference') or policy.get('steps', 50) != 50:
        parser.error('Unsupported policy mode or adaptation length')
    if policy['policy'] == 'selector' and not 0 <= policy['tau'] < 1:
        parser.error('A selection margin must lie in [0, 1)')
    tokens = policy.get('tokens', bank['qaxis'])
    if len(tokens) != len(set(tokens)) or not set(tokens) <= set(bank['qaxis']):
        parser.error('Policy tokens must form a unique subset of the declared bank')
    if len(args.budgets) != len(set(args.budgets)) or not set(args.budgets) <= set(bank['budgets']):
        parser.error('Each requested budget must occur once and be defined in the bank')
    case = next((c for c in cases(args.role) if c['slug'] == args.case), None)
    if case is None:
        parser.error('Case does not belong to the declared role')
    subsets = {b: feasible_records(bank, tokens, case['H'], b) for b in args.budgets}
    for budget, rows in subsets.items():
        if not rows:
            parser.error(f'No feasible candidate under {budget}')
        if policy['reference'] not in [r['token'] for r in rows]:
            parser.error(f'Reference is infeasible under {budget}')
    prefix = DATA / 'prefix' / case['slug'] / 'prefix.bin'
    norm = read(DATA / 'bitbrains/data/source_normalization.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        # The inner directory and receipt are published together after all budgets have succeeded.
        product = Path(staging) / 'completed'
        product.mkdir()
        results = {}
        with PrefixFence() as fence:
            data = prefix_selection(case, prefix, norm)
            for index, (budget, rows) in enumerate(subsets.items()):
                model_path = product / f'selected_{index}.pt' if args.save_models else None
                results[budget] = instantiate(case, bank, rows, policy, data, model_path)
                if model_path is not None:
                    results[budget]['selected_checkpoint'] = model_path.name
        receipt = dict(case_id=case['case_id'], role=case['role'], prefix_sha256=sha(prefix),
                       bank_sha256=sha(args.bank), policy_sha256=sha(args.policy),
                       reference=policy['reference'], policy_mode=policy['policy'],
                       margin=policy.get('tau'), budgets=results,
                       denied_prefix_reads=fence.denied)
        write(product / 'selection.json', receipt)
        product.rename(output)
    print(output / 'selection.json')
