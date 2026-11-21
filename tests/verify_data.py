"""Release integrity: promoted files, input contract and public assets agree with each other."""
from pathlib import Path
import hashlib
import json
import os
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / '준비패키지' / 'release'
SHARE = ROOT / '준비패키지' / 'share'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    info = json.loads((REL / 'release_info.json').read_text(encoding='utf-8'))
    for name, digest in info['files'].items():
        assert sha(REL / name) == digest, f'{name}: release_info 해시 불일치 (promote.py 없이 수정됨)'
    bench = json.loads((REL / 'benchmark_public.json').read_text(encoding='utf-8'))
    check = json.loads((REL / 'release_check.json').read_text(encoding='utf-8'))
    dictionary = json.loads((REL / 'data_dictionary.json').read_text(encoding='utf-8'))
    bundle = joblib.load(REL / 'model.joblib')
    assert bench['model_sha256'] == check['model_sha256'] == info['model_sha256'] == sha(REL / 'model.joblib')
    assert bench['version'] == check['version'] == info['version'] == bundle['version'] == dictionary['version']
    assert check['passed'] or info.get('forced'), '승격 조건 미통과 모델'
    assert bundle['schema_version'] == 2
    assert [f['name'] for f in dictionary['features']] == bundle['features']
    assert sorted(bundle['missing_policy']['required'] + bundle['missing_policy']['optional']) == sorted(bundle['features'])
    assert bundle['caution_threshold'] <= bundle['alert_threshold'] == bench['alert_threshold']
    for f in bundle['features']:
        assert not f.startswith(tuple(dictionary['excluded_input_prefixes'])), f
    for fa in bench['feature_analysis']:
        assert all(g['count'] >= 5 for g in fa['quality_relation']), fa['feature']
    definition = json.loads((SHARE / 'synthetic_definition.json').read_text(encoding='utf-8'))
    assert definition['release_version'] == bundle['version'], '합성 예제를 make_synthetic.py로 다시 만드세요'
    for name, meta in definition['files'].items():
        assert sha(SHARE / name) == meta['sha256'], name
        df = pd.read_csv(SHARE / name)
        assert set(df.data_source) == {dictionary['synthetic_source']}, name
    result = {'status': 'passed', 'version': bundle['version'], 'model': bundle['selected_model'],
              'features': bundle['features'], 'release_check_passed': check['passed'],
              'synthetic_rows': {k: v['rows'] for k, v in definition['files'].items()}}
    evidence = Path(os.environ.get('ICT_EVIDENCE_DIR', str(ROOT / 'tmp' / 'test-evidence')))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'data_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
