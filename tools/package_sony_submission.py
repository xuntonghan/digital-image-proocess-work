"""Extend the verified v2 course package with sealed Sony1000 and the revised PDF.

Refuses partial results. Preserves the v2 package and Canon's eight frozen sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/submission_20261025'
RUN = ROOT / 'outputs/sony1000_20261025'
REPORT = ROOT / 'output/pdf/sied_final_report_20261025'
ASSETS = ROOT / 'reports/assets/enhancement/sony1000_20261025'
PDF = ROOT / 'output/pdf/韩勋同-2025312132086-数字图像处理课程实习综合报告.pdf'
SONY_SOURCES = ['cloud/prepare_sony1000.py', 'cloud/sony1000_pipeline.py', 'cloud/run_sony1000.sh',
                'cloud/verify_sony1000.py', 'cloud/check_sony_parallel_pilot.py',
                'cloud/export_sony_visualizations.py', 'cloud/reproduce_sony_submission.py',
                'cloud/sony_deadline.py', 'cloud/run_sony_deadline.sh',
                'cloud/download_official_sied_assets.py', 'tests/test_sony_submission.py',
                'tests/test_sony_visualizations.py', 'tests/test_sony_reproduction.py',
                'tests/test_sony_deadline.py', 'tools/build_sony_report.py',
                'tools/package_sony_submission.py', 'manifests/source/sony_all.txt',
                'manifests/source/sony_audit.json', 'manifests/enhancement/sony_train.txt',
                'manifests/enhancement/sony_val.txt', 'manifests/enhancement/sony_test.txt']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def readzip(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt package: ' + str(path))
        return {name: z.read(name) for name in z.namelist() if not name.endswith('/')}


def writezip(path, entries):
    temporary = path.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, value in entries.items():
            z.writestr(name, value)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--overleaf-export', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    assert len(commit) == 40
    sys.path.insert(0, str(ROOT / 'cloud'))
    spec = importlib.util.spec_from_file_location('sony_statistics', ROOT / 'cloud/verify_sony1000.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    final = RUN / 'submission1000_native'
    statistics = verifier.verify(final)
    binding = load(final / 'binding.json')
    report_data = load(REPORT / 'sony_report_data.json')
    assert report_data['statistics'] == statistics
    pdf_proof = load(ROOT / 'reports/assets/course_report_20261025/pdf_receipt.json')
    overleaf_proof = load(ROOT / 'reports/assets/course_report_20261025/overleaf_verification.json')
    assert pdf_proof['sony_1000_count'] == 1000 and pdf_proof['all_pages_visual_qa_passed']
    assert overleaf_proof['sony_1000_count'] == 1000
    assert digest(PDF.read_bytes()) == pdf_proof['sha256']
    # Retain the prior local package. Remote v2 release assets are never changed.
    backup = ROOT / 'output/submission_20261025_v2'
    if not backup.exists():
        shutil.copytree(OUT, backup)
    entries = readzip(backup / '代码与指标.zip')
    canonical = json.loads(entries['project/reports/assets/enhancement/submission1000_20261024/binding.json'])
    for name, expected in canonical['code_sha256'].items():
        assert digest(entries['project/'+name]) == expected, name
    added = {}
    for name in SONY_SOURCES:
        added[name] = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
    for name, expected in binding['code_sha256'].items():
        if name in added:
            assert digest(added[name]) == expected
        else:
            assert digest(entries['project/'+name]) == expected, name
    for p in ASSETS.rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.xml','.txt','.py'}:
            added[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for name in ['binding.json','verification.json','execution_receipt.json','metrics.csv','baseline_metrics.csv','summary.json']:
        added['reports/assets/enhancement/sony1000_20261025/submission1000_native/'+name] = (final/name).read_bytes()
    added['reports/assets/enhancement/sony1000_20261025/source_data_sha256.json'] = (RUN/'logs/sony_data_verified.json').read_bytes()
    added['reports/assets/enhancement/sony1000_20261025/reproduction_smoke_verification.json'] = (RUN/'reproduction_smoke/verification_comparison.json').read_bytes()
    original_commands = (backup/'完整复现命令.txt').read_text()
    sony_commands = '''

Sony全部1000张补充复现

Sony源码追加版本 SOURCE_COMMIT
Sony与Canon分别用各自1000张，不能互换相机权重。
先安装前述Python3.12与PyTorch2.8.0等固定依赖，以下命令在project目录执行。
只复算保存的Sony1000张指标不需要GPU。
python3 cloud/verify_sony1000.py

如需要重新下载Sony原始数据和官方模型，准备至少70GiB空闲空间。
SONY_RUN="$(pwd)/outputs/sony_reproduce"
mkdir -p "$SONY_RUN/third_party"
test -e "$SONY_RUN/project" || ln -s "$(pwd)" "$SONY_RUN/project"
python cloud/prepare_sony1000.py --root "$SONY_RUN"
git clone https://github.com/JianghaiSCU/SIED.git "$SONY_RUN/third_party/SIED-official"
git -C "$SONY_RUN/third_party/SIED-official" checkout --detach fb7af94ac831e91c647415bf1ca0b1a75931833a

解压自训增强权重包到project中，然后检查全部附件权重。
python submission_weights/verify_weights.py

Sony原始RAW独立推理，不使用训练缓存，先检查3张训练图。
python cloud/reproduce_sony_submission.py \\
  --data-root "$SONY_RUN/data/SIED_dataset" \\
  --official-repo "$SONY_RUN/third_party/SIED-official" \\
  --official-checkpoint "$SONY_RUN/checkpoints/official/sony_0.001.pt" \\
  --lut-checkpoint submission_weights/sony_color_lut_selected.pt \\
  --output-dir "$SONY_RUN/reproduced" --limit-new 3

完整1000张使用同样参数并去掉--limit-new 3，程序保留已完成的前3张，继续原始822/91/87清单。
新输出写入reproduced，不覆盖报告中的封存指标。每张预测后才读参考图评分。
GT-Mean追加训练未达到预设收益门槛，最终模型保留原1000步Sony色彩表。
两组追加对照的源码、声明、权重和逐项指标均随附件保留。
'''.replace('SOURCE_COMMIT',commit)
    commands = original_commands + sony_commands
    notes = (backup/'提交说明.txt').read_text().replace('先运行其中verify_weights.py核对5个文件。', '运行其中verify_weights.py核对全部附件权重，原Canon的5个文件保留。')
    notes += f"\nSony补充版本\nSony全部1000张已完成，PSNR为{statistics['overall']['psnr_mean']:.9f}，SSIM为{statistics['overall']['ssim_mean']:.12f}。\n"
    notes += '主指标均为原生浮点逐图算术平均。含822张训练和91张官方预训练暴露验证图，87张留出测试单列。\n'
    notes += '代码保留Canon固定版本，另加Sony源码版本'+commit+'。原其他项目未提交改动未进入代码包。\n'
    notes += '自训增强权重增加Sony最终色彩表和两个未采用的损失对照。官方Sony2.6GB权重从作者处另行下载。\n'
    added['完整复现命令.txt'] = commands.encode()
    for name, value in added.items():
        entries['project/'+name] = value
    materials = json.loads(entries['project/REPORT_MATERIALS.json'])
    materials['sony_source_commit'] = commit
    materials['sony_1000_complete'] = True
    for name, value in added.items():
        materials['added_or_sealed_materials'][name] = {'bytes':len(value),'sha256':digest(value)}
    entries['project/REPORT_MATERIALS.json'] = (json.dumps(materials,ensure_ascii=False,indent=2)+'\n').encode()
    entries['project/复现说明.txt'] = notes.encode()
    for name, expected in canonical['code_sha256'].items():
        assert digest(entries['project/'+name]) == expected
    OUT.mkdir(exist_ok=True)
    writezip(OUT/'代码与指标.zip', entries)
    weights = readzip(backup/'自训增强权重.zip')
    manifest = json.loads(weights['submission_weights/WEIGHTS.json'])
    own = {'sony_color_lut_selected.pt':RUN/binding['selection']['selected']['checkpoint'],
           'sony_ablation_control_1250.pt':RUN/'research/gtmean_deadline/control.pt',
           'sony_ablation_gtmean_1250.pt':RUN/'research/gtmean_deadline/gtmean.pt'}
    for name, path in own.items():
        raw = path.read_bytes()
        weights['submission_weights/'+name] = raw
        manifest['files'][name] = {'source':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':digest(raw),'camera':'sony'}
    assert manifest['files']['sony_color_lut_selected.pt']['sha256'] == binding['selection']['selected']['checkpoint_sha256']
    manifest['official_sony_required'] = {'bytes':2610974590,'sha256':binding['official_sha256'],
                                        'obtain_from':'https://github.com/JianghaiSCU/SIED','redistributed':False}
    weights['submission_weights/WEIGHTS.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    weights['submission_weights/verify_weights.py'] = b'''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
files=json.loads((root/'WEIGHTS.json').read_text())['files']
for name,item in files.items():
 data=(root/name).read_bytes()
 if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
  raise ValueError('Weight mismatch: '+name)
 print('OK',name)
print('Verified',len(files),'included project-trained weights; official models are obtained separately.')
'''
    weights['submission_weights/说明.txt'] = notes.encode()
    writezip(OUT/'自训增强权重.zip',weights)
    source = readzip(args.overleaf_export)
    # The browser export is authoritative for TeX and display assets.
    for p in REPORT.glob('*.tex'):
        assert source[p.name] == p.read_bytes(),p.name
    for p in (REPORT/'images').iterdir():
        if p.is_file():
            key=p.name if p.name in source else 'images/'+p.name
            assert source[key] == p.read_bytes(),p.name
    for name in ['report_data.json','course_report_data.json','sony_report_data.json']:
        source[name] = (REPORT/name).read_bytes()
    source['overleaf_verification.json'] = json.dumps(overleaf_proof,ensure_ascii=False,indent=2).encode()
    source['编译说明.txt'] = '使用XeLaTeX编译main.tex。工程包含Canon与Sony各1000张、几何和压缩的综合报告。\n'.encode()
    writezip(OUT/'Overleaf_LaTeX工程.zip',source)
    shutil.copy2(PDF,OUT/PDF.name)
    (OUT/'完整复现命令.txt').write_text(commands)
    (OUT/'提交说明.txt').write_text(notes)
    proof = {'revision':'v3 Sony1000 complete','sony_statistics':statistics,'sony_source_commit':commit,
             'eight_canon_frozen_sources_match':True,'pdf':pdf_proof,'overleaf':overleaf_proof,
             'included_weight_count':len(manifest['files'])}
    for path in OUT.glob('*.zip'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        proof[path.name] = {'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())}
    dump(OUT/'附件独立复核.json',proof)
    dump(OUT/'附件校验.json',{p.name:{'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
         for p in OUT.iterdir() if p.is_file() and p.name not in {'附件校验.json','附件独立复核.json'}})
    deliver=ROOT/'reports/deliverables'
    for name,target in [(PDF.name,PDF.name),('代码与指标.zip','三项目代码与指标_20261025.zip'),
                        ('Overleaf_LaTeX工程.zip','Overleaf_三项目综合报告_20261025.zip'),('完整复现命令.txt','完整复现命令.txt')]:
        shutil.copy2(OUT/name,deliver/target)
    outer=ROOT/'output/韩勋同-2025312132086-数字图像处理课程实习.zip'
    with zipfile.ZipFile(outer,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,outer.stem+'/'+p.name)
    with zipfile.ZipFile(outer) as z:assert z.testzip() is None
    release=ROOT/'tmp/release-v3';release.mkdir(exist_ok=True)
    for p,name in [(outer,'han-xuntong-2025312132086-dip-course-submission-v3.zip'),
                   (PDF,'han-xuntong-2025312132086-dip-course-report-v3.pdf')]:
        shutil.copy2(p,release/name)
    (release/'release_sha256.txt').write_text(''.join(digest(p.read_bytes())+'  '+p.name+'\n'
        for p in sorted(release.iterdir()) if p.suffix in {'.pdf','.zip'}))
    print(json.dumps({'sony_1000_complete':True,'weight_count':len(manifest['files']),
                      'outer_bytes':outer.stat().st_size,'outer_sha256':digest(outer.read_bytes())},indent=2))


if __name__=='__main__':
    main()
