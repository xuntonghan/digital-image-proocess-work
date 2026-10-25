"""Create portable course attachments from the fixed source snapshot and verified materials."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output/pdf/sied_final_report_20261025"
OUT = ROOT / "output/submission_20261025"
COMMIT = "beebb3454e99f039dfd98bc848aa7fafb9772f46"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data=json.loads((REPORT/'report_data.json').read_text())
    weights={name:item for name,item in data['weights'].items() if not name.startswith('official')}
    notes = """韩勋同 2025312132086  数字图像处理课程实习

一 代码
project目录为固定提交beebb3454e99f039dfd98bc848aa7fafb9772f46的源码。
新增reproduce_statistics.py只复算1000张封存CSV，不需要GPU。
新增cloud/reproduce_sied_submission.py用于原始RAW的独立固定权重推理。
新增入口不使用原86GiB缓存，未在本次报告编写期间重跑CUDA全量评价。
先在project中安装依赖，再按PDF逐项执行。

二 权重
将自训增强权重包解压到project中，得到submission_weights目录。
先运行其中verify_weights.py核对5个文件。
最终增强只需要canon_naf_phase4.pt与canon_color_lut_1250.pt，另需官方SIED。
官方SIED权重不在附件，按PDF原作者地址获取并核对完整SHA256。

三 数据和缺失材料
附件不含大型原始数据和全部原生PNG。最终1000张PNG已在本机outputs/sied1000_20261024/submission1000_native/images备份。
GeoCorrection原200对图像和旧PageMaskUNet权重当前未在工作目录找到。报告保留了其固定划分、训练日志、逐图指标和原权重SHA256。
恢复原数据及模型后才能严格重跑旧几何推理。从头重训可以复现方法，不能保证旧模型逐位一致。
UVDoc和OCR使用PDF给出的公开来源、模型修订和已记录哈希。
Cheng-A采用CompressAI 1.2.8官方MSE预训练模型，Q1、Q3、Q6。旧实验未保存各模型完整SHA256，不能补写旧文件哈希。

四 指标
Canon1000张平均PSNR 26.707356438407885，SSIM 0.8435157579481601。
LPIPS 0.3243558964841068。它们是逐图算术均值，包含训练图。
Cheng-A Q6原始RGB负载压缩比29.790094；PNG文件负载压缩比16.198257。
Cheng码率不含共享模型和独立文件容器头。
两项历史几何、压缩开发中未提交的改动没有混入固定源码。

五 PDF和Overleaf
打开LaTeX工程中的main.tex，编译器选择XeLaTeX。
正文使用图像文件名，graphicspath同时支持images子目录和Overleaf根目录。
封存数据的路径、展示图来源和处理方式保存在report_data.json和course_report_data.json。
"""
    weight_manifest={'files':weights,'official_required':data['weights']['official_canon_0.0001.pt'],
                     'missing_geometry_checkpoint_sha256':'250789695f3c92b3acaa2c373fb6a51083cf0dfa54093ed3acd3c2cd084eccc6'}
    verify_code='''"""Verify the five included project-trained weights."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
d=json.loads((p/'WEIGHTS.json').read_text())
for name,item in d['files'].items():
 data=(p/name).read_bytes()
 if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
  raise ValueError('Weight verification failed: '+name)
 print('OK',name,item['sha256'])
print('All five included weights verified. Official SIED is obtained separately.')
'''
    with zipfile.ZipFile(OUT/'自训增强权重.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,item in weights.items():
            raw=(ROOT/item['source']).read_bytes()
            assert digest(raw)==item['sha256'] and len(raw)==item['bytes']
            z.writestr('submission_weights/'+name,raw)
        z.writestr('submission_weights/WEIGHTS.json',json.dumps(weight_manifest,ensure_ascii=False,indent=2)+'\n')
        z.writestr('submission_weights/verify_weights.py',verify_code)
        z.writestr('submission_weights/说明.txt',notes)
    snapshot=Path('/tmp/course_source_beebb345.zip')
    if not snapshot.exists():
        subprocess.run(['git','archive','--format=zip','--output='+str(snapshot),COMMIT,'--',
                        'src','tests','cloud','configs','manifests','docs','pyproject.toml','README.md','DATA_AND_WEIGHTS.md'],
                       cwd=ROOT,check=True,timeout=60)
    materials={}
    # Metrics and logs only. Do not package official weights, native caches or unrelated dirty code.
    for task in ['enhancement','geocorrection','compression']:
        for p in (ROOT/'reports/assets'/task).rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl','.xml','.txt','.log'}:
                materials[p.relative_to(ROOT).as_posix()]=p.read_bytes()
    for name in ['reproduce_statistics.py','cloud/reproduce_sied_submission.py',
                 'tools/prepare_final_pdf_assets.py','tools/build_final_pdf_tables.py',
                 'tools/prepare_course_report_assets.py','tools/package_course_submission.py']:
        materials[name]=(ROOT/name).read_bytes()
    manifest={name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in sorted(materials.items())}
    with zipfile.ZipFile(OUT/'代码与指标.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        with zipfile.ZipFile(snapshot) as old:
            for entry in old.infolist():
                if not entry.is_dir() and entry.filename not in materials:
                    z.writestr('project/'+entry.filename,old.read(entry))
        for name,raw in materials.items():z.writestr('project/'+name,raw)
        z.writestr('project/REPORT_MATERIALS.json',json.dumps({'base_commit':COMMIT,'added_or_sealed_materials':manifest},ensure_ascii=False,indent=2)+'\n')
        z.writestr('project/复现说明.txt',notes)
    with zipfile.ZipFile(OUT/'Overleaf_LaTeX工程.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in REPORT.rglob('*'):
            if p.is_file() and (p.suffix=='.tex' or p.name in {'report_data.json','course_report_data.json'} or p.parent.name=='images'):
                z.write(p,p.relative_to(REPORT))
        z.writestr('编译说明.txt','使用XeLaTeX编译main.tex。请保留images目录。正文为韩勋同2025312132086三项目综合报告。\n')
    (OUT/'提交说明.txt').write_text(notes)
    (OUT/'附件校验.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
                                               for p in OUT.glob('*.zip')},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({p.name:p.stat().st_size for p in OUT.iterdir() if p.is_file()},ensure_ascii=False))


if __name__=='__main__':main()
