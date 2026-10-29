"""Exercise the installed v1.2 APIs outside the checkout, without network access."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(sys.argv[1]).resolve()
    import quotientguard as q
    package = Path(q.__file__).resolve()
    assert 'site-packages' in str(package) and not package.is_relative_to(root/'quotientguard'), package
    for filename, command in [('full-objective.json', 'full-objective-audit'),
                              ('ppo-rollout-correction.json', 'full-objective-audit'),
                              ('gspo-rollout-correction.json', 'full-objective-audit'),
                              ('objective-window.json', 'objective-window-audit')]:
        process = subprocess.run([sys.executable, '-m', 'quotientguard', command,
            str(root/'examples'/filename), '--json'], capture_output=True, text=True, check=True)
        assert json.loads(process.stdout)['status'] == 'SUPPORTED', filename
    import torch
    from quotientguard.native_validation import validate_parameter_graph, validate_training_step
    p = torch.nn.Parameter(torch.tensor([[0.]], dtype=torch.float64))
    x = p * 2
    spec = q.FullObjectiveSpec(policy=q.ObjectiveSpec(dtype='float64', reduction='sum'))
    trace = q.FullObjectiveTrace(q.ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x)))
    assert validate_parameter_graph(spec, trace, -x.exp().sum(), {'p': p})['status'] == 'PASS'
    report = validate_training_step(lambda _: p.sum(), {'p': p}, [None],
        optimizer=torch.optim.SGD([p], lr=.1), scaler=torch.amp.GradScaler('cpu', init_scale=16), max_grad_norm=1.)
    assert report['status'] == 'PASS' and report['optimizer_step_executed']
    from quotientguard.survey import Repository, SourceFile, SourceInventory, SurveyEngine, SurveyLimits, SurveyStore
    from quotientguard.survey_classify import classify_python
    source = b'def loss(log_probs, old_log_probs, advantages, mask):\n    r=(log_probs-old_log_probs).exp()\n    return (r*advantages*mask).mean()\n'
    with tempfile.TemporaryDirectory() as directory:
        store = SurveyStore(directory, max_disk_bytes=10_000_000)
        try:
            store.create_run('installed-smoke')
            repo = Repository('1', 'local/synthetic', 'a'*40)
            def fetch(repository, acquisition_plan=None):
                yield SourceInventory(('loss.py',), ())
                yield SourceFile('loss.py', source)
            engine = SurveyEngine(store, SurveyLimits())
            first = engine.run('installed-smoke', [repo], fetch, incremental=True)
            def no_fetch(*args, **kwargs):
                raise AssertionError('Completed run must not acquire source again')
            second = engine.run('installed-smoke', [repo], no_fetch, incremental=True)
            assert first['report_sha256'] == second['report_sha256']
            assert first['counts']['inspected_files'] == 1
            assert first['classification'] == 'STATIC_RISK_CANDIDATE'
            assert classify_python(source.decode(), 'loss.py')['components']
        finally:
            store.close()
    print(json.dumps({'status': 'PASS', 'version': q.__version__, 'installed_package': str(package),
        'cli_examples': 4, 'parameter': 'PASS', 'training': 'PASS', 'survey_and_resume': 'PASS',
        'classification': 'PASS', 'foreign_code_executed': False}))


if __name__ == '__main__':
    main()
