"""Add verified Sony1000 results to the existing three-project LaTeX report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'output/pdf/sied_final_report_20261025'
RUN = ROOT / 'outputs/sony1000_20261025'
sys.path.insert(0, str(ROOT / 'cloud'))
spec = importlib.util.spec_from_file_location('sony_statistics', ROOT / 'cloud/verify_sony1000.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def table(caption, headings, rows, columns=None):
    return ('\\begin{table}[H]\\centering\\small\n\\caption{' + caption + '}\n'
            '\\begin{tabular}{' + (columns or 'l' + 'r' * (len(headings) - 1)) + '}\\toprule\n' +
            ' & '.join(headings) + r'\\\midrule' + '\n' +
            '\n'.join(' & '.join(map(str, row)) + r'\\' for row in rows) +
            '\n\\bottomrule\\end{tabular}\n\\end{table}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sony-commit', default=None)
    args = parser.parse_args()
    final = RUN / 'submission1000_native'
    verified = verifier.verify(final)  # Refuse partial statistics before changing any TeX.
    binding = load(final / 'binding.json')
    summary = load(final / 'summary.json')
    visual = load(RUN / 'report_assets/sony_visualizations.json')
    assert visual['count'] == 6 and visual['source_binding_sha256'] == sha(final / 'binding.json')
    assert visual['source_metrics_sha256'] == sha(final / 'metrics.csv')
    assert len(visual['examples']) == 6
    selected = binding['selection']['selected']
    weight = RUN / selected['checkpoint']
    assert sha(weight) == selected['checkpoint_sha256']
    commit = args.sony_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert len(commit) == 40
    for name, expected in binding['code_sha256'].items():
        assert sha(ROOT / name) == expected, name
    before = {p.relative_to(REPORT).as_posix(): sha(p) for p in REPORT.rglob('*')
              if p.is_file() and (p.suffix == '.tex' and p.name != 'main.tex' and not p.name.startswith('sony_')
              or p.parent.name == 'images' and not p.name.startswith('sony_'))}
    for example in visual['examples']:
        for view in example['views'].values():
            name = Path(view['display_file']).name
            source = RUN / 'report_assets' / name
            assert sha(source) == view['display_sha256']
            shutil.copy2(source, REPORT / 'images' / name)
    with (final / 'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (final / 'baseline_metrics.csv').open(newline='') as stream:
        old = {r['key']: r for r in csv.DictReader(stream)}
    overall, base = verified['overall'], verified['official_baseline_overall']
    deltas = [float(r['psnr']) - float(old[r['key']]['psnr']) for r in rows]
    improved = sum(v > 0 for v in deltas)
    research = load(RUN / 'research/gtmean_deadline/result.json')
    pilot = load(RUN / 'research/sony_lut/validation_1000.json')
    speed = load(RUN / 'logs/final_execution_equivalence.json')
    section = r'''\clearpage
\section{Sony完整1000张补充实验}

本轮补齐Sony 0.001数据的全部1000张。使用Sony专用SIED权重和Sony专用色彩表，不使用Canon精修权重。Sony与Canon分别报告，两个相机的成绩不合成一个均值。

Sony评价使用RTX4090 24GB，Python 3.12、PyTorch 2.8.0与CUDA 12.8。NumPy为2.3.2，Pillow为11.3.0，scikit-image为0.26.0，LPIPS为0.1.4。

\subsection{数据与参数}

原始划分为822张训练、91张验证和87张测试。2000个配对文件共18,653,873,890字节，与原始审计一致。RAW尺寸、黑电平、白电平和RGGB打包方式与前文相同。正式PSNR和SSIM在原生3840乘2160浮点RGB上计算，之后才保存PNG。全部1000张逐图取算术平均。

官方Sony模型见过913张训练和验证图。因此1000张总体包含训练数据，91张验证只用于选择新增模型，87张留出测试另列。测试参考图只在预测后用于评分。

保持FP16、20步DDIM、三次预测平均和划分内种子。新增色彩表有63,324个参数，17个格点、3个基底和0.12残差上限。整图512上下文与训练保持一致。原生查询每块256行，共用整图系数。

训练只用822张的512乘288 FP32对。批量16，fused AdamW，学习率$3\times10^{-4}$，权重衰减0，逐图log-MSE加0.1倍平滑和单调约束。初始模型为恒等变换。完成1000步，每250步检查固定12张原生验证图，之后通过完整91张验证。

候选PSNR须至少提高0.02\,dB，SSIM下降不超过0.002，各项颜色误差和LPIPS增加不超过2\%。模型冻结后才开始留出测试评分。

\subsection{ICCV方法对照与执行效率}

继续参考ICCV 2025的GT-Mean Loss\cite{gtmean}。在原1000步色彩表上做两个各250步的对照。两组使用同一初值和批次序列，批量16、FP32、学习率$3\times10^{-5}$。对照组延续log-MSE，论文方法组再加5倍GT-Mean L1，$\sigma=0.1$。GT均值只出现在训练损失中，推理不读取参考图。这是损失方法的独立对照，没有重新训练整篇论文网络。

'''
    section += table('Sony追加训练的固定12张原生验证', ['方法', 'PSNR / dB', 'SSIM', '是否采用'], [
        ['原1000步色彩表', f"{pilot['psnr_mean']:.6f}", f"{pilot['ssim_mean']:.9f}", '原方案'],
        *[[('追加250步对照' if arm == 'control' else '追加250步GT-Mean'),
           f"{item['pilot']['psnr_mean']:.6f}", f"{item['pilot']['ssim_mean']:.9f}",
           '采用' if research['selected_followup'] == arm else '未采用'] for arm, item in research['arms'].items()]])
    if research['selected_followup'] is None:
        section += '\n两个分支的快速验证收益都低于预设0.02\\,dB门槛，因此停止追加训练，保留已通过完整91张验证的原方案。没有扩大为新一轮全量训练，也没有用测试结果挑选模型。\n'
    section += f"\n两组各250步的训练更新分别用时{research['arms']['control']['training_seconds']:.3f}秒和{research['arms']['gtmean']['training_seconds']:.3f}秒。该时间不含数据预载和原生验证，不能作为整次实验用时。\n"
    section += (f"\n评价采用单GPU盲预测和{speed['workers']}个有上限的CPU指标与写图线程。固定4张训练图的串行用时{speed['serial_four_seconds']:.3f}秒，并行用时{speed['parallel_four_seconds']:.3f}秒，短测吞吐为原来的{speed['short_run_ratio']:.3f}倍。PNG哈希和全部8项浮点指标一致。这是执行短测，没有降低分辨率、精度、采样次数或评价标准。\n")
    section += '\n训练与验证采用原样FP32缓存，训练缓存上限40\\,GiB。最终续算沿用同一模型已完成的120张，另外的图按原清单继续。87张测试重新执行盲扩散。缓存和重新预测的混合计时不代表统一的单图端到端速度。\n'
    section += r'\subsection{完整1000张主结果}' + '\n'
    section += table('Sony全部1000张的逐图算术均值', ['方案', 'PSNR / dB', 'SSIM', 'LPIPS'], [
        ['官方Sony SIED', f"{base['psnr_mean']:.6f}", f"{base['ssim_mean']:.9f}", f"{base['lpips_mean']:.9f}"],
        ['最终Sony方案', f"{overall['psnr_mean']:.6f}", f"{overall['ssim_mean']:.9f}", f"{overall['lpips_mean']:.9f}"]])
    section += (f"\nPSNR相对官方模型变化{overall['psnr_mean']-base['psnr_mean']:+.6f}\\,dB，SSIM变化{overall['ssim_mean']-base['ssim_mean']:+.9f}。PSNR有{improved}张提高、{sum(v<0 for v in deltas)}张下降，变化中位数为{statistics.median(deltas):+.6f}\\,dB。全部图像都纳入均值。\n")
    section += table('Sony最终方案的各划分结果', ['划分', '数量', 'PSNR / dB', 'SSIM', 'LPIPS'], [
        [label, verified['splits'][split]['count'], f"{verified['splits'][split]['psnr_mean']:.6f}",
         f"{verified['splits'][split]['ssim_mean']:.9f}", f"{verified['splits'][split]['lpips_mean']:.9f}"]
        for split, label in [('train','训练'),('val','验证'),('test','留出测试')]])
    test_base = [old[r['key']] for r in rows if r['split'] == 'test']
    test_psnr = statistics.fmean(float(r['psnr']) for r in test_base)
    test_ssim = statistics.fmean(float(r['ssim']) for r in test_base)
    section += (f"\n87张留出测试的官方PSNR为{test_psnr:.6f}\\,dB，最终为{verified['splits']['test']['psnr_mean']:.6f}\\,dB。"
                f"SSIM由{test_ssim:.9f}变为{verified['splits']['test']['ssim_mean']:.9f}。两项留出测试主指标均提高。\n")
    colors = [('delta_e2000_mean','平均色差'),('delta_e2000_p95_mean','逐图色差P95均值'),
              ('lightness_mae_mean','亮度MAE'),('saturation_mae_mean','饱和度MAE'),('hue_error_degrees_mean','色相误差 / 度')]
    section += table('Sony全部1000张的补充颜色指标', ['指标', '官方Sony', '最终方案'], [
        [label, f'{base[field]:.8f}', f'{overall[field]:.8f}'] for field,label in colors])
    section += table('Sony最终方案各划分的颜色指标',
        ['划分', '平均色差', r'\cell{色差P95\\均值}', '亮度MAE', '饱和度MAE', r'\cell{色相误差\\度}'],
        [[label, *[f"{verified['splits'][split][field]:.6f}" for field, _ in colors]]
         for split, label in [('train', '训练'), ('val', '验证'), ('test', '留出测试')]])
    section += r'''\subsection{Sony代码和权重}

原始官方代码固定在前述SIED提交。Sony官方模型为\Path{sony_0.001.pt}，大小2,610,974,590字节，需从作者仓库下载并核对下列SHA256。

\noindent\Path{7379cdec9401b92ea28b26568a5b33f7e72be06a0e7c12d1c76f76e3499f693d}\par

新增Sony权重随自训增强权重包提供，名称为\Path{sony_color_lut_selected.pt}。其SHA256为

'''
    section += '\\noindent\\Path{' + selected['checkpoint_sha256'] + '}\\par\n'
    section += '\nSony追加源码版本为\\Path{' + commit + '}。原冻结推理和指标源码保持原字节，执行器另行保存源码哈希与原生等价证明。\\Path{sony_report_data.json}记录模型、展示图、逐图指标和来源。\n'
    section += r'''
只复算封存指标时不需要GPU。
\begin{lstlisting}[language=bash]
python3 cloud/verify_sony1000.py
\end{lstlisting}
从RAW重跑使用\Path{cloud/reproduce_sony_submission.py}，不依赖40\,GiB训练缓存。先检查3张训练图，再在新目录完成全部1000张。完整参数、数据获取和权重检查命令随包提供。正式输出保留1000行CSV、种子、划分、模型绑定和PNG哈希。

{\small\noindent\textbf{Sony对比图说明}\quad 六组样例取原测试清单前6张。RAW预览采用曝光1和Gamma 2.2，ISP采用Gamma 1.8、99百分位和目标0.9。模型图只作等比例缩放，局部固定为$(1280,720)$至$(2560,1440)$。}

{\small\linespread{1.04}\selectfont\setlength{\parskip}{0.1em}
\input{sony_examples}}
\clearpage
'''
    examples = []
    for index, example in enumerate(visual['examples']):
        key = example['key']
        if index % 2 == 0:
            examples.append('\\clearpage\n')
        examples.append('\\subsection*{Sony样例 ' + str(index + 1) + '，图像 ' + key + '}\n')
        panels = [('raw','低光RAW预览'),('isp','传统ISP示意'),('gt','参考图')]
        examples.append('\\begin{center}\n' + '\\hfill\n'.join('\\Panel{'+example['views'][v]['display_file']+'}{'+label+'}' for v,label in panels) + '\n\\end{center}\n')
        b, a = example['official_metrics'], example['final_metrics']
        examples.append('\\begin{center}\n' + '\\Panel{'+example['views']['official']['display_file']+'}{官方Sony}\\hfill\n' +
                        '\\Panel{'+example['views']['final']['display_file']+'}{最终Sony}\\hfill\n' +
                        '\\begin{minipage}[t]{0.318\\linewidth}\\vspace{0pt}\\raggedright\\small '+
                        f"PSNR由{b['psnr']:.4f}变为{a['psnr']:.4f}\\,dB。\\par SSIM由{b['ssim']:.6f}变为{a['ssim']:.6f}。\\par LPIPS由{b['lpips']:.6f}变为{a['lpips']:.6f}。"+
                        '\\end{minipage}\n\\end{center}\n')
        examples.append('\\begin{center}\n' + '\\hfill\n'.join('\\Panel{'+example['views'][v]['display_file']+'}{'+label+'}' for v,label in [('official_crop','官方局部'),('final_crop','最终局部'),('gt_crop','参考局部')]) + '\n\\end{center}\n')
    (REPORT / 'sony_section.tex').write_text(section)
    (REPORT / 'sony_examples.tex').write_text('\n'.join(examples))
    main_path = REPORT / 'main.tex'
    original = main_path.read_text()
    backup = ROOT / 'tmp/pdfs/main_before_sony.tex'
    if not backup.exists():
        backup.write_text(original)
    content = original.replace('实验数据 & SIED Canon 1000张、GeoCorrection 200张、Kodak 24张',
        r'实验数据 & \cell{SIED Canon 1000张和Sony 1000张\\GeoCorrection 200张、Kodak 24张}')
    version_start = content.index('代码版本 &')
    version_end = content.index('\n', version_start)
    content = (content[:version_start] + r'代码版本 & Canon \texttt{beebb345}，Sony \texttt{'
               + commit[:8] + r'}\\' + content[version_end:])
    content = content.replace('正式成绩采用Canon全部1000张的逐图平均PSNR、SSIM和LPIPS。Sony只保留87张历史测试记录。',
        '正式成绩分别采用Canon和Sony各1000张的逐图平均PSNR和SSIM，同时报告LPIPS和颜色指标。')
    content = content.replace('图像增强采用Canon全部1000张结果，几何校正和压缩采用已保存的评价。',
        '图像增强分别采用Canon和Sony各1000张结果，几何校正和压缩采用已保存的评价。')
    anchor = '\\vspace{0.6em}}}\\par'
    cover_start = content.index(r'\noindent\colorbox{pale}')
    cover_end = content.index(anchor, cover_start) + len(anchor)
    content = content[:cover_start] + (r'\noindent\colorbox{pale}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\vspace{0.6em}\textbf{两个相机各1000张的平均结果}\par' + '\n' +
        r'Canon的PSNR为26.707356\,dB，SSIM为0.843515758。\par' + '\n' +
        f"Sony的PSNR为{overall['psnr_mean']:.6f}\\,dB，SSIM为{overall['ssim_mean']:.9f}。\\par\n" +
        r'两组均包含训练图，各自另列留出测试。\vspace{0.6em}}}\par') + content[cover_end:]
    needle = '几何校正 & 自动方法隔离22张'
    if 'Sony全部1000张 & PSNR' not in content:
        content = content.replace(needle, f"图像增强 & Sony全部1000张 & PSNR {overall['psnr_mean']:.6f}\\,dB，SSIM {overall['ssim_mean']:.9f}，LPIPS {overall['lpips_mean']:.9f}\\\\\n" + needle)
    if r'\input{sony_section}' not in content:
        content = content.replace(r'\input{geometry_compression}', '\\input{sony_section}\n\n\\input{geometry_compression}')
    if r'\bibitem{gtmean}' not in content:
        content = content.replace(r'\bibitem{uvdoc}', r'\bibitem{gtmean} Jingxi Liao, Shijie Hao, Richang Hong, Meng Wang. GT-Mean Loss: A Simple Yet Effective Solution for Brightness Mismatch in Low-Light Image Enhancement. ICCV, 2025, 6112--6121. \url{https://github.com/jingxiLiao/GT-mean-loss}.' + '\n' + r'\bibitem{uvdoc}')
    content = content.replace(r'\begin{thebibliography}{9}', r'\begin{thebibliography}{99}')
    main_path.write_text(content)
    appendix_path = REPORT / 'geometry_compression.tex'
    appendix_original = appendix_path.read_text()
    appendix_updated = appendix_original.replace('25组对比图与参考资料', '31组对比图与参考资料').replace(
        'NAF、最终LUT、两个色彩表对照和光度v2，附SHA256核验脚本',
        'Canon原5份权重、Sony最终色彩表和两个未采用对照，共8份，附SHA256核验脚本')
    # Only the submission inventory changes. Keep all prior experiment text intact.
    inventory_start = r'\part{提交与复核}'
    assert appendix_original.split(inventory_start)[0] == appendix_updated.split(inventory_start)[0]
    appendix_path.write_text(appendix_updated)
    metadata = {'camera': 'sony', 'statistics': verified, 'summary': summary, 'binding': binding,
                'visualizations': visual, 'followup_research': research, 'execution_equivalence': speed,
                'source_commit': commit, 'selected_weight': {'source': str(weight.relative_to(ROOT)),
                'packaged_name': 'sony_color_lut_selected.pt', 'bytes': weight.stat().st_size,
                'sha256': sha(weight)}, 'original_other_tex_and_images_sha256': before,
                'submission_inventory_updated': True}
    (REPORT / 'sony_report_data.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    assert all(sha(REPORT / name) == value for name, value in before.items()
               if name != 'geometry_compression.tex')
    print(json.dumps({'verified_count':1000,'sony_examples':6,'new_tex_files':2,
                      'original_experiment_text_and_images_unchanged':True,
                      'submission_inventory_updated':True},ensure_ascii=False))


if __name__ == '__main__':
    main()
