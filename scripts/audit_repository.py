"""Inspect exactly the selected Git tree; report patterns without printing secrets."""
from pathlib import Path,PurePosixPath
import argparse
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--staged',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.staged: names=git('ls-files','-z').decode('utf-8').split('\0')
    else:
        tracked=git('ls-files','-z').decode('utf-8').split('\0')
        extra=git('ls-files','--others','--exclude-standard','-z').decode('utf-8').split('\0')
        names=tracked+extra
    names=sorted(set(filter(None,names)));included=set(names);errors=[];files=[];total=0
    patterns={
        'personal_absolute_path':re.compile(r'(?:[A-Za-z]:[\\/]+Users[\\/]+(?!Public(?:[\\/]))|/' + r'Users/(?!Shared/)[^/\s"<>]+/)[^\s"<>]*'),  # 패턴이 자기 자신에 걸리지 않게 분리
        'github_token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
        'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'openai_key':re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{40,}'),
        'korean_mobile':re.compile(r'01[016789]-\d{3,4}-\d{4}'),
        'personal_download_id':re.compile(r'(?:MSS_' + r'hjus36|GB20261026-PU\d+)')}
    row_level={'split.csv','cv_folds.csv','oof_predictions.csv','holdout_predictions.csv','benchmark.json','real_demo.csv'}
    for name in names:
        path=PurePosixPath(name)
        if any(p in {'.venv','venv','node_modules','__pycache__','baseline_snapshot','참고자료','.git'} for p in path.parts) or path.parts[0] in {'tmp','outputs','output'}:
            errors.append(dict(file=name,rule='excluded_directory'));continue
        if path.suffix.lower() in {'.sqlite3','.db','.zip','.docx','.pdf','.pyc'} or '.sqlite3-' in name:
            errors.append(dict(file=name,rule='local_or_binary_document'));continue
        if path.parts[0]=='ml_workflow' and len(path.parts)>1 and path.parts[1] in {'runs','data'} and name!='ml_workflow/data/README.md':
            errors.append(dict(file=name,rule='workflow_data_or_runs'));continue
        if path.suffix.lower() in {'.parquet','.joblib'} and name!='준비패키지/release/model.joblib':
            errors.append(dict(file=name,rule='dataset_or_unpromoted_model'));continue
        if path.name in row_level or name.startswith(('준비패키지/local/','준비패키지/release/previous/')) or path.name.startswith('dataset_V'):
            errors.append(dict(file=name,rule='row_level_data'));continue
        content=git('show',':'+name) if args.staged else (ROOT/name).read_bytes()
        total+=len(content)
        if len(content)>5*1024*1024: errors.append(dict(file=name,rule='over_5MiB_review_limit'))
        files.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        if path.suffix in {'.py','.ps1','.md','.txt','.json','.jsx','.js','.mjs','.yml','.yaml','.toml','.csv'} or path.name in {'.gitignore','.gitattributes'}:
            try: text=content.decode('utf-8-sig')
            except UnicodeDecodeError:
                errors.append(dict(file=name,rule='unexpected_text_encoding'));continue
            for rule,pattern in patterns.items():
                for match in pattern.finditer(text): errors.append(dict(file=name,rule=rule,line=text[:match.start()].count('\n')+1))
            if path.suffix=='.md':
                for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                    target=target.strip().strip('<>')
                    if re.match(r'^[a-zA-Z]+:',target) or target.startswith('#'): continue
                    target=unquote(target.split('#')[0])
                    resolved=(ROOT/path.parent/target).resolve()
                    try: relative=resolved.relative_to(ROOT).as_posix()
                    except ValueError:
                        errors.append(dict(file=name,rule='link_outside_repo'));continue
                    if relative not in included and not any(n.startswith(relative.rstrip('/')+'/') for n in included):
                        errors.append(dict(file=name,rule='link_missing_from_git_tree',target=relative))
    result=dict(status='passed' if not errors else 'failed',scope='index' if args.staged else 'working_candidates',
        files=len(files),bytes=total,findings=errors,manifest=files)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if not errors else 1)
if __name__=='__main__': main()
