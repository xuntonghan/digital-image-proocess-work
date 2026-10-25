"""Refresh only report assets and their page references after camera separation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/submission_20261025'
OLD = ROOT / 'output/submission_20261025_v3'
REPORT = ROOT / 'output/pdf/sied_final_report_20261025'
ASSETS = ROOT / 'reports/assets/course_report_20261025'
PDF_NAME = '韩勋同-2025312132086-数字图像处理课程实习综合报告.pdf'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def readzip(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return {n: z.read(n) for n in z.namelist() if not n.endswith('/')}


def writezip(path, entries):
    temporary = path.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in entries.items():
            z.writestr(name, raw)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overleaf-export', type=Path, required=True)
    parser.add_argument('--render-directory', type=Path, required=True)
    args = parser.parse_args()
    pdf = ROOT / 'output/pdf' / PDF_NAME
    pdf_proof = load(ASSETS / 'pdf_receipt.json')
    assert pdf_proof['all_pages_visual_qa_passed']
    assert sha(pdf.read_bytes()) == pdf_proof['sha256']
    pages = pdf_proof['pages']
    texts = load(args.render_directory / 'text.json')
    assert len(texts) == pages
    assert '补充实验' not in ''.join(texts)
    revision = load(ASSETS / 'camera_section_revision.json')
    for name, expected in revision['protected_sources_sha256'].items():
        assert sha((REPORT / name).read_bytes()) == expected
    assert (OUT / '自训增强权重.zip').read_bytes() == (OLD / '自训增强权重.zip').read_bytes()

    def page_of(needle):
        return next(i + 1 for i, text in enumerate(texts) if i > 1 and needle in ''.join(text.split()))

    locations = {'canon': page_of('Canon实验'), 'sony': page_of('Sony实验'),
                 'enhancement_summary': page_of('图像增强实验总结'),
                 'geometry': page_of('项目二几何校正'), 'compression': page_of('项目三图像压缩'),
                 'submission': page_of('第四部分提交与复核')}
    audit = load(OLD / '课程要求核对.json')
    audit['report_pdf_pages'] = pages
    audit['report_structure'] = revision['new_structure']
    audit['report_physical_pages'] = locations
    (OUT / '课程要求核对.json').write_bytes(encoded(audit))
    (ASSETS / 'course_requirement_audit.json').write_bytes(encoded(audit))
    text = (OLD / '课程要求核对.txt').read_text()
    text = text.replace('报告PDF物理第3页有主表，第8至10页有Canon结果，第20至22页有Sony方法和结果。',
        f"报告PDF物理第3页有主表。Canon实验为第{locations['canon']}至{locations['sony']-1}页，"
        f"Sony实验为第{locations['sony']}至{locations['enhancement_summary']-1}页，第{locations['enhancement_summary']}页为共同总结。")
    text = text.replace('第26至32页', f"第{locations['geometry']}至{locations['compression']-1}页")
    text = text.replace('第33至41页', f"第{locations['compression']}至{locations['submission']-1}页")
    text = text.replace('物理第34页表21', f"物理第{locations['compression']+1}页的压缩质量与时间表")
    text = text.replace('同页表22', '同页压缩比表')
    text = text.replace('综合PDF为43页', f'综合PDF为{pages}页').replace('10个TeX', '11个TeX')
    (OUT / '课程要求核对.txt').write_text(text)
    notes = (OLD / '提交说明.txt').read_text()
    notes += '\n本版只调整报告第一部分，分为Canon实验、Sony实验和图像增强实验总结。其他项目正文和全部图片不变，模型、权重与指标不变。\n'
    (OUT / '提交说明.txt').write_text(notes)

    previous = readzip(OLD / '代码与指标.zip')
    entries = dict(previous)
    for name in ['课程要求核对.txt', '课程要求核对.json']:
        entries['project/' + name] = (OUT / name).read_bytes()
    entries['project/复现说明.txt'] = notes.encode()
    entries['project/reports/assets/course_report_20261025/camera_section_revision.json'] = encoded(revision)
    materials = json.loads(entries['project/REPORT_MATERIALS.json'])
    for name in ['课程要求核对.txt', '课程要求核对.json']:
        raw = (OUT / name).read_bytes()
        materials['added_or_sealed_materials'][name] = {'bytes': len(raw), 'sha256': sha(raw)}
    entries['project/REPORT_MATERIALS.json'] = encoded(materials)
    for camera, folder in [('canon', 'submission1000_20261024'),
                           ('sony', 'sony1000_20261025/submission1000_native')]:
        prefix = 'project/reports/assets/enhancement/' + folder + '/'
        binding = json.loads(entries[prefix + 'binding.json'])
        for name, expected in binding['code_sha256'].items():
            assert sha(entries['project/' + name]) == expected
        for name in ['metrics.csv', 'binding.json', 'summary.json', 'verification.json']:
            assert entries[prefix + name] == previous[prefix + name]
    for name, raw in previous.items():
        if any(part in name for part in ['/compression/', '/geocorrection/']):
            assert entries[name] == raw
    writezip(OUT / '代码与指标.zip', entries)
    source = readzip(args.overleaf_export)
    for p in list(REPORT.glob('*.tex')) + list((REPORT / 'images').iterdir()):
        if p.is_file():
            key = p.name if p.name in source else 'images/' + p.name
            assert source[key] == p.read_bytes(), p.name
    for name in ['report_data.json', 'course_report_data.json', 'sony_report_data.json']:
        source[name] = (REPORT / name).read_bytes()
    source['camera_section_revision.json'] = encoded(revision)
    source['overleaf_verification.json'] = (ASSETS / 'overleaf_verification.json').read_bytes()
    source['编译说明.txt'] = '使用XeLaTeX编译main.tex。第一部分分为Canon实验、Sony实验和共同总结。\n'.encode()
    writezip(OUT / 'Overleaf_LaTeX工程.zip', source)
    shutil.copy2(pdf, OUT / PDF_NAME)
    proof = load(OLD / '附件独立复核.json')
    proof['revision'] = 'v4 separate Canon and Sony experiments'
    proof['pdf'] = pdf_proof
    proof['overleaf'] = load(ASSETS / 'overleaf_verification.json')
    proof['report_revision'] = revision
    proof['course_requirement_audit'] = audit
    proof['all_model_weights_and_sealed_metrics_unchanged_from_v3'] = True
    for path in OUT.glob('*.zip'):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        proof[path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}
    (OUT / '附件独立复核.json').write_bytes(encoded(proof))
    (ASSETS / 'package_verification.json').write_bytes(encoded(proof))
    checks = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
              for p in OUT.iterdir() if p.is_file() and p.name != '附件校验.json'}
    (OUT / '附件校验.json').write_bytes(encoded(checks))
    for name, target in [(PDF_NAME, PDF_NAME), ('代码与指标.zip', '三项目代码与指标_20261025.zip'),
                         ('Overleaf_LaTeX工程.zip', 'Overleaf_三项目综合报告_20261025.zip'),
                         ('课程要求核对.txt', '课程要求核对.txt')]:
        shutil.copy2(OUT / name, ROOT / 'reports/deliverables' / target)
    outer = ROOT / 'output/韩勋同-2025312132086-数字图像处理课程实习.zip'
    with zipfile.ZipFile(outer, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p, outer.stem + '/' + p.name)
    with zipfile.ZipFile(outer) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read(outer.stem + '/附件校验.json'))
        for name, item in manifest.items():
            raw = z.read(outer.stem + '/' + name)
            assert sha(raw) == item['sha256'] and len(raw) == item['bytes']
    release = ROOT / 'tmp/release-v4'
    release.mkdir(exist_ok=True)
    for path, name in [(outer, 'han-xuntong-2025312132086-dip-course-submission-v4.zip'),
                       (pdf, 'han-xuntong-2025312132086-dip-course-report-v4.pdf')]:
        shutil.copy2(path, release / name)
    checksum_text = ''.join(sha(p.read_bytes()) + '  ' + p.name + '\n'
                            for p in sorted(release.iterdir()) if p.suffix in {'.pdf', '.zip'})
    (release / 'release_sha256.txt').write_text(checksum_text)
    (ASSETS / 'release_sha256_v4.txt').write_text(checksum_text)
    final_proof = {'revision': 'v4', 'pdf_pages': pages, 'physical_page_locations': locations,
                   'outer_bytes': outer.stat().st_size, 'outer_sha256': sha(outer.read_bytes()),
                   'all_zip_crc_and_inner_manifest_hashes_verified': True,
                   'all_weights_and_model_metrics_unchanged': True}
    (ASSETS / 'camera_split_package_verification.json').write_bytes(encoded(final_proof))
    print(json.dumps(final_proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
