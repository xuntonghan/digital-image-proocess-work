"""Check the original course requirements and add standalone deployment documents.

Uses sealed report tables and packaged sources. Never changes model predictions,
frozen sources, or the historical geometry/compression results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/submission_20261025'
ASSETS = ROOT / 'reports/assets/course_report_20261025'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment-pdf', type=Path, required=True)
    args = parser.parse_args()
    code = OUT / '代码与指标.zip'
    with zipfile.ZipFile(code) as z:
        assert z.testzip() is None
        entries = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    report_dir = ROOT / 'output/pdf/sied_final_report_20261025'
    report_text = ''.join(json.loads((ROOT / 'tmp/pdfs/sony_overleaf_render/text.json').read_text()))
    tables = (report_dir / 'course_tables.tex').read_text()
    checks = {'assignment_sha256': sha(args.assignment_pdf.read_bytes()),
              'requirement_pages': {'enhancement': [8, 9], 'geometry': [10, 11],
                                    'compression': [12, 13], 'package': [14]},
              'enhancement': {}, 'compression': {}, 'geometry': {},
              'report_pdf_pages': 43, 'comparison_groups': 31}
    for camera, folder, psnr, ssim, lpips in [
        ('canon', 'submission1000_20261024', 26.707356438407885,
         0.8435157579481601, 0.3243558964841068),
        ('sony', 'sony1000_20261025/submission1000_native', 26.721838080611242,
         0.8532759628295898, 0.4139315472096205)]:
        prefix = 'project/reports/assets/enhancement/' + folder + '/'
        rows = list(csv.DictReader(io.StringIO(entries[prefix + 'metrics.csv'].decode())))
        assert len(rows) == len({r['key'] for r in rows}) == 1000
        for name, expected in [('psnr', psnr), ('ssim', ssim), ('lpips', lpips)]:
            actual = statistics.fmean(float(r[name]) for r in rows)
            assert math.isclose(actual, expected, abs_tol=1e-12)
            assert f'{actual:.6f}' in report_text or f'{actual:.9f}' in report_text
        checks['enhancement'][camera] = {'count': 1000, 'psnr': psnr,
                                         'ssim': ssim, 'lpips': lpips,
                                         'arithmetic_means_verified': True,
                                         'all_three_required_metrics_in_pdf': True}
    for name in ['jpeg_baseline', 'dctc_v1', 'dctc_v2_fast', 'cheng2020_anchor']:
        prefix = 'project/reports/assets/compression/' + name + '/'
        summary = json.loads(entries[prefix + 'summary.json'])
        rows = list(csv.DictReader(io.StringIO(entries[prefix + 'metrics.csv'].decode())))
        qualities = []
        for item in summary['qualities']:
            subset = [r for r in rows if int(r['quality']) == item['quality']]
            assert len(subset) == len({r['key'] for r in subset}) == 24
            for field, digits in [('psnr', 6), ('ssim', 8), ('encode_ms', 4), ('decode_ms', 4)]:
                actual = statistics.fmean(float(r[field]) for r in subset)
                assert math.isclose(actual, item[field + '_mean'], rel_tol=1e-10, abs_tol=1e-10)
                assert f'{actual:.{digits}f}' in tables
            size = sum(int(r['compressed_bytes']) for r in subset)
            assert size == item['compressed_bytes_total']
            for ratio in [28311552 / size, 15394305 / size]:
                assert f'{ratio:.6f}' in tables
            qualities.append(item['quality'])
        checks['compression'][name] = {'count_per_quality': 24, 'qualities': qualities,
                                        'compression_ratios_psnr_ssim_encode_decode_times_in_pdf': True,
                                        'all_means_and_byte_ratios_recomputed': True}
    prefix = 'project/reports/assets/geocorrection/'
    geo = json.loads(entries[prefix + 'final_4090/predicted/summary.json'])
    assert geo['input_count'] == 22
    for field in ['input_text_line_skew_degrees_mean', 'corrected_text_line_skew_degrees_mean']:
        assert f"{geo['overall'][field]:.6f}" in tables
    ocr = json.loads(entries[prefix + 'ocr/summary.json'])
    rows = list(csv.DictReader(io.StringIO(entries[prefix + 'ocr/metrics.csv'].decode())))
    assert ocr['subset_count'] == 40 and len(rows) == 160
    for stage in ocr['stages']:
        subset = [r for r in rows if r['stage'] == stage]
        assert len(subset) == 40
        for field in ['line_count', 'confident_line_count', 'confident_character_count']:
            actual = sum(int(r[field]) for r in subset)
            assert actual == ocr['overall'][stage][field + '_total']
            assert f'{actual:,}' in tables
    checks['geometry'] = {'historical_dataset_count': 200, 'held_out_automatic_count': 22,
                          'ocr_fixed_subset_count': 40, 'before_after_images_in_pdf': True,
                          'text_line_skew_in_pdf': True, 'ocr_counts_and_confidence_in_pdf': True,
                          'no_transcripts_no_cer_or_recognition_accuracy_claim': True,
                          'limitations': ['原GeoCorrection数据和旧PageMaskUNet权重未附带，保留历史指标、划分与源码。']}
    audit = '''课程要求逐项核对
韩勋同 2025312132086

核对依据
数字图像处理课程-课间实习-2026-09-30.pdf，共15页。
第9页要求增强的PSNR、SSIM、LPIPS在全部1000张增强后计算。
第11页要求几何校正前后对比、文字行规整程度和OCR效果评价。
第13页要求压缩率、压缩后PSNR、压缩后SSIM、压缩与解压时间。
第14页要求可复现源码、部署文档、含指标和前后对比图的报告，并按姓名和学号命名ZIP。

项目一 图像增强
已分别写入Canon1000张和Sony1000张的PSNR、SSIM、LPIPS。
Canon为26.707356438407885dB、0.8435157579481601、0.3243558964841068。
Sony为26.721838080611242dB、0.8532759628295898、0.4139315472096205。
均为逐图算术平均。总体包含训练图。原划分和留出测试成绩单列。
报告PDF物理第3页有主表，第8至10页有Canon结果，第20至22页有Sony方法和结果。
PDF保留12组Canon和新增6组Sony对比图。逐图CSV、8项指标和种子保存在代码包。

项目二 几何校正
报告PDF物理第26至32页包含方法、文字行倾斜、边界指标、OCR诊断和8组对比图。
区分200张真值掩膜辅助历史实验、22张自动方法隔离测试和40张固定OCR诊断。
OCR给出文本行、高置信行、高置信字符、置信度及加权字符。
没有人工文字转录，因此未把这些诊断写成识别准确率或CER。
原GeoCorrection数据和旧分割权重未附带。现有源码、划分、指标、模型SHA和历史图均保留。
这一资料缺失会限制原自动分割结果的直接复现，报告和部署说明已明确说明。

项目三 图像压缩
报告PDF物理第33至41页包含Kodak全部24张的结果和5组对比图。
物理第34页表21给出bpp、PSNR、SSIM、编码毫秒和解码毫秒。
同页表22给出码流字节数、RGB压缩比、PNG压缩比和RGB节省百分比。
四种方法共有15个质量设置，每个设置均覆盖完整24张。逐图CSV随代码包提供。
时间口径、重复次数、预热、硬件和学习式码流不含共享模型的限制已注明。

提交材料
综合PDF为43页，共31组前后对比。PDF含GitHub链接。
代码与指标.zip为固定源码、划分、逐图CSV、训练和核验资料。
部署说明.txt为独立部署文档，完整复现命令.txt保留全部操作命令。
Overleaf_LaTeX工程.zip包含10个TeX和193个图形文件。
自训增强权重.zip含8份项目权重及SHA检查脚本，未采用的两份Sony对照已注明。
原始数据和公开大模型按取得方式另行获取。未用其他相机分数代替Sony结果。
最终文件名为韩勋同-2025312132086-数字图像处理课程实习.zip。
'''
    deploy = '''三项目部署说明
韩勋同 2025312132086

一 解压和目录
先解压课程提交ZIP，再把代码与指标.zip解压为project目录。
把自训增强权重.zip解压到project，使project/submission_weights目录存在。
PDF为正式报告。Overleaf_LaTeX工程.zip可在Overleaf使用XeLaTeX编译main.tex。
所有详细命令见同目录完整复现命令.txt。命令均在project目录执行。

二 无GPU核验
Python3.12可直接核对保存的统计和权重，不需要下载原始数据。
python3 reproduce_statistics.py
python3 cloud/verify_sony1000.py
python3 submission_weights/verify_weights.py
前两项应各输出count为1000和verified为true。第三项应检查通过8份附件权重。
原始PNG没有重复放入小型代码包，完整预测已保存在作者本机，报告提供31组对比。

三 增强推理环境
完整盲推理使用Linux和NVIDIA CUDA。实测环境为RTX4090 24GB。
Python3.12，PyTorch2.8.0，torchvision0.23.0，CUDA12.8。
NumPy2.3.2，Pillow11.3.0，scikit-image0.26.0，LPIPS0.1.4。
先按完整复现命令.txt创建虚拟环境、安装依赖和本项目，再取原始RAW与参考图。
Canon与Sony各使用自己的官方模型和附件色彩表，不能互换权重。
Sony完整下载与新推理建议准备至少70GiB空闲空间。大规模缓存不是复现入口的必要条件。
官方SIED代码固定fb7af94ac831e91c647415bf1ca0b1a75931833a。
作者仓库为https://github.com/JianghaiSCU/SIED。
Sony官方模型名称sony_0.001.pt，大小2610974590字节。
其SHA256为7379cdec9401b92ea28b26568a5b33f7e72be06a0e7c12d1c76f76e3499f693d。
各模型的完整SHA、划分、精度、20步采样、3次平均和种子保存在相应binding.json。
先用两个相机各自的reproduce入口做3张训练图检查，再去掉--limit-new 3处理完整1000张。
指定全新的输出目录，不覆盖封存结果。按原生浮点预测计算指标，不能用8位PNG替代。
Sony已有独立3张RAW重跑记录，PNG哈希和全部8项指标与封存结果一致。

四 几何校正
依赖安装、传统校正、UVDoc、自动页面分割、重训和OCR命令见完整复现命令.txt。
原GeoCorrection的images与mask目录各需200张。先按原清单和数据SHA核对。
当前附件不含原GeoCorrection数据和旧PageMaskUNet权重，不能直接重现原自动分割预测。
补齐同一权重后可运行固定22张隔离测试。也可按保存的训练设置重新训练并另记结果。
公开UVDoc与OCR模型按命令下载。保留固定版本和模型SHA核验资料。
几何历史结果、200张辅助实验、22张自动测试及40张OCR诊断分开使用。
没有平整参考图和文字转录，不能补造几何PSNR、SSIM或OCR准确率。

五 图像压缩
Kodak24张从命令中固定的数据仓库提交取得，先核对kodak.sha256。
JPEG和DCTC使用CPU，Cheng-A使用CompressAI1.2.8和官方MSE模型。
安装压缩扩展后，依次运行JPEG、DCTC v1、DCTC v2与Cheng-A命令。
JPEG和DCTC均有4个Q设置，Cheng-A有Q1、Q3和Q6。每个设置均评价24张。
输出包括真实码流、解码图、metrics.csv和summary.json。
报告同时写出RGB与PNG压缩比、PSNR、SSIM、压缩和解压时间。
原传统压缩环境与GPU环境不同，时间不能直接当成统一硬件排名。
三个公开Cheng-A权重的原完整SHA未保存。重跑时应记录实际下载文件，不能声称逐字节相同。

六 结果和资料位置
两相机的1000行指标、模型绑定和种子位于代码包reports/assets/enhancement。
几何与压缩逐图指标位于reports/assets/geocorrection和reports/assets/compression。
数据划分位于manifests。必要源码和部署命令均随包提供。
GitHub仓库为https://github.com/xuntonghan/digital-image-proocess-work。
仓库为私有，阅卷者需要访问权限。课程提交ZIP可在本机直接使用。
'''
    checks['required_metrics_present'] = True
    checks['standalone_deployment_document_added'] = True
    (ASSETS / 'course_requirement_audit.json').write_bytes(encoded(checks))
    docs = {'课程要求核对.txt': audit.encode(), '部署说明.txt': deploy.encode(),
            '课程要求核对.json': encoded(checks)}
    for name, raw in docs.items():
        (OUT / name).write_bytes(raw)
        entries['project/' + name] = raw
    entries['project/复现说明.txt'] = (OUT / '提交说明.txt').read_bytes()
    own_source = Path(__file__).read_bytes()
    entries['project/tools/finalize_course_package.py'] = own_source
    material = json.loads(entries['project/REPORT_MATERIALS.json'])
    for name, raw in docs.items():
        material['added_or_sealed_materials'][name] = {'bytes': len(raw), 'sha256': sha(raw)}
    material['added_or_sealed_materials']['tools/finalize_course_package.py'] = {
        'bytes': len(own_source), 'sha256': sha(own_source)}
    entries['project/REPORT_MATERIALS.json'] = encoded(material)
    temporary = code.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in entries.items():
            z.writestr(name, raw)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
    temporary.replace(code)
    for name, target in [('代码与指标.zip', '三项目代码与指标_20261025.zip'),
                         ('部署说明.txt', '部署说明.txt'), ('课程要求核对.txt', '课程要求核对.txt')]:
        shutil.copy2(OUT / name, ROOT / 'reports/deliverables' / target)
    # Package proofs are refreshed after the independent unpack check.
    print(json.dumps({'required_metrics_present': True, 'deployment_document_added': True,
                      'compression_quality_settings_verified': 15}, ensure_ascii=False))


if __name__ == '__main__':
    main()
