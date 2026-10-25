"""Give Canon and Sony separate experiment sections without changing other tasks."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'output/pdf/sied_final_report_20261025'
BACKUP = ROOT / 'tmp/pdfs/report_before_camera_split_20261025'
ASSETS = ROOT / 'reports/assets/course_report_20261025'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    if (BACKUP / 'main.tex').exists():
        raise RuntimeError('This one-time revision has already been applied')
    for path in REPORT.glob('*.tex'):
        shutil.copy2(path, BACKUP / path.name)
    protected = {p.name: digest(p) for p in REPORT.glob('*.tex')
                 if p.name not in {'main.tex', 'sony_section.tex', 'examples.tex', 'sony_examples.tex'}}
    images = {p.name: digest(p) for p in (REPORT / 'images').iterdir() if p.is_file()}
    original = (REPORT / 'main.tex').read_text()
    first = r'\part{项目一 极暗图像增强}'
    last = r'\input{geometry_compression}'
    a, b = original.index(first) + len(first), original.index(last)
    prefix, suffix = original[:a], original[b:]
    canon = original[a:original.index(r'\input{sony_section}')]
    task_end = canon.index(r'\section{数据读取和评价方法}')
    canon = canon[task_end:]
    old_start = canon.index('\\begin{table}[H]\\centering\\small\n\\caption{Sony的87张历史测试记录}')
    old_end = canon.index(r'\section{可视化比较}', old_start)
    historical_sony = canon[old_start:old_end]
    canon = canon[:old_start] + canon[old_end:]
    canon = canon.replace(r'\subsection{', r'\subsubsection{')
    canon = canon.replace(r'\section{', r'\subsection{')
    canon = canon.replace(r'\subsection{模型选择和实验结果}', r'\subsection{实验结果与分析}')
    canon = canon.replace(r'\subsection{可视化比较}', r'\subsection{效果图与分析}')
    canon = canon.replace(r'\subsection{效果图与分析}', '\\clearpage\n\\subsection{效果图与分析}')
    canon = canon.replace(
        '三个划分互不重复。官方预训练使用过其中的924张，因此92张验证图仅对新增分支独立。1000张总体包含训练图，不能解释为独立测试成绩。最终模型按验证结果冻结，再评价1000张。',
        '三个划分互不重复。官方模型预训练见过其中924张，92张验证图只用于选择新增分支。'
        '1000张总体含训练图，不能视为独立测试。最终模型按验证结果冻结，再评价全部1000张。')
    canon = canon.replace(r'\ctexset{subsection=', r'\ctexset{subsubsection=')
    canon = ('\n\n\\section{Canon实验}\n\n'
             'Canon实验使用SIED的Canon\\_0.0001数据，共1000对低光RAW和正常曝光参考图。'
             '实验先比较传统ISP与早期学习方法，再使用Canon专用SIED、RAW引导NAF精修和自适应色彩表。'
             '最终结果按全部1000张逐图计算PSNR、SSIM和LPIPS，并单列训练、验证和留出测试。\n\n' + canon)
    intro_start = canon.index(r'\subsection{效果图与分析}') + len(r'\subsection{效果图与分析}')
    intro_end = canon.index('RAW和ISP列从封存图截取。', intro_start)
    canon = (canon[:intro_start] + '\n\n'
             '12组样例均来自76张测试图。前6组按历史固定顺序，后6组从其余图像中按PSNR变化排序取6个等距位置。'
             '效果图包含整图和相同区域的细节，保留提高与下降样例。它们不参与模型选择或总体统计。\n\n'
             + canon[intro_end:])
    sony = (REPORT / 'sony_section.tex').read_text()
    sony = sony.replace(r'\section{Sony完整1000张补充实验}', r'\section{Sony实验}')
    sony = sony.replace(
        '本轮补齐Sony 0.001数据的全部1000张。使用Sony专用SIED权重和Sony专用色彩表，不使用Canon精修权重。Sony与Canon分别报告，两个相机的成绩不合成一个均值。',
        'Sony实验使用SIED的Sony\\_0.001数据，共1000对低光RAW和正常曝光参考图。'
        '模型采用Sony专用SIED和Sony专用自适应色彩表。独立完成数据检查、训练、模型选择和全部1000张评价。')
    sony = sony.replace('RAW尺寸、黑电平、白电平和RGGB打包方式与前文相同。',
        'RAW尺寸为3840乘2160，黑电平为512，白电平为16383，按RGGB打包为四通道。')
    sony = sony.replace('保持FP16、20步DDIM、三次预测平均和划分内种子。',
        '使用FP16、20步DDIM和三次独立预测平均。划分内第$i$张图像、第$j$次采样的种子为'
        '$3407+i+1000003j$，$j=0,1,2$。PSNR、SSIM、LPIPS和颜色指标沿用Canon实验中的计算定义。')
    sony = sony.replace(r'\subsection{完整1000张主结果}', r'\subsection{实验结果与分析}')
    sony = sony.replace('Sony全部1000张的补充颜色指标', 'Sony全部1000张的颜色指标')
    sony = sony.replace('Sony追加源码版本为', 'Sony实验源码版本为')
    historical_sony = historical_sony.replace('Sony的87张历史测试记录', 'Sony早期方法的87张测试结果')
    sony = sony.replace(r'\subsection{Sony代码和权重}',
        '\\subsubsection{早期方法对照}\n\n' + historical_sony + '\\subsection{代码和权重}')
    sony = sony.replace('{\\small\\noindent\\textbf{Sony对比图说明}',
        '\\subsection{效果图与分析}\n\n'
        '椅子与门窗样例中，最终方案的亮度更接近参考图。车辆和文字样例可用于比较颜色、边缘和文字区域。'
        '局部图使用相同范围，便于检查色彩变化和细节保留。所有分数仍来自整幅原生浮点预测。\n\n'
        '{\\small\\noindent\\textbf{对比图说明}')
    summary = r'''\clearpage
\section{图像增强实验总结}

Canon与Sony均完成各自全部1000张的增强和评价。两组实验分别使用对应相机的模型和固定划分。下表汇总最终结果，所有主指标均为逐图算术平均。

\begin{table}[H]\centering\small
\caption{Canon与Sony两个实验的最终结果}
\begin{tabular}{lrrrr}\toprule
实验 & 图像数 & PSNR / dB & SSIM & LPIPS\\\midrule
Canon & 1000 & 26.707356 & 0.843515758 & 0.324355896\\
Sony & 1000 & 26.721838 & 0.853275963 & 0.413931547\\
\bottomrule\end{tabular}
\end{table}

Canon最终方案比冻结基线的平均PSNR提高0.152867\,dB，SSIM略降。76张留出测试的PSNR和SSIM也下降。效果图中同时保留了改善和下降样例，因此总体PSNR提高不能代替留出测试中的实际表现。

Sony最终方案比同协议官方模型的平均PSNR提高0.428325\,dB，SSIM提高0.000891891，LPIPS和各项颜色误差均改善。87张留出测试的PSNR和SSIM也提高。GT-Mean损失对照未达到预设门槛，最终采用已通过完整验证的1000步色彩表。

两组结果说明，小型色彩模型可以调整亮度和颜色，但实际收益仍要结合感知指标、颜色误差和留出测试判断。数据缓存、显存常驻和并行评分减少了部分等待，短测吞吐提高不代表完整训练具有相同加速比。

两个1000张总体都包含训练图及官方预训练已见的验证图。它们用于课程规定的全数据评价，不能写成2000张独立测试成绩。两个相机的数据与模型不同，报告保留各自结果，不合并成一个平均分。
'''
    (REPORT / 'main.tex').write_text(prefix + canon + '\\input{sony_section}\n\n\\input{enhancement_summary}\n\n' + suffix)
    (REPORT / 'sony_section.tex').write_text(sony)
    (REPORT / 'enhancement_summary.tex').write_text(summary)
    for name in ['examples.tex', 'sony_examples.tex']:
        p = REPORT / name
        content = p.read_text().replace(r'\subsection*{', r'\subsubsection*{')
        if name == 'examples.tex':
            content = content.replace('\\clearpage\n', '', 1)
        p.write_text(content)
    changed = (REPORT / 'main.tex').read_text()
    assert changed[:changed.index(first) + len(first)] == prefix
    assert changed[changed.index(last):] == suffix
    assert all(digest(REPORT / name) == value for name, value in protected.items())
    assert all(digest(REPORT / 'images' / name) == value for name, value in images.items())
    assert '补充实验' not in changed + sony + summary
    assert 'Sony的87张历史测试记录' not in canon
    receipt = {'updated_at': datetime.now().astimezone().isoformat(),
               'new_structure': ['Canon实验', 'Sony实验', '图像增强实验总结'],
               'non_enhancement_main_prefix_and_suffix_unchanged': True,
               'protected_sources_sha256': protected, 'all_193_images_unchanged': True,
               'metrics_and_models_unchanged': True, 'pdf_skill_operation_marked_once': True}
    (ASSETS / 'camera_section_revision.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
