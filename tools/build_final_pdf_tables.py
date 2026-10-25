"""Generate LaTeX data tables, case figures and one scientific distribution plot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/sied_final_report_20261025"


def fmt(value, digits=6):
    return f"{float(value):.{digits}f}"


def table(headers, rows, spec):
    return ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(headers)
            + " \\\\\n\\midrule\n" + "\n".join(" & ".join(map(str, row)) + " \\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def plot_distribution(data):
    final = ROOT / "reports/assets/enhancement/submission1000_20261024/metrics.csv"
    baseline = ROOT / "reports/assets/enhancement/full1000_20261024/metrics.csv"
    with final.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with baseline.open(newline="") as stream:
        base = {r["key"]: r for r in csv.DictReader(stream)}
    c = canvas.Canvas(str(OUT / "images/delta_distribution.pdf"), pagesize=(520, 220))
    palette = {"train": colors.HexColor("#24547b"), "val": colors.HexColor("#32806a"),
               "test": colors.HexColor("#b45b36")}
    for panel, (metric, xmin, xmax, ticks, title) in enumerate([
        ("psnr", -2, 4.5, [-2, -1, 0, 1, 2, 3, 4], "PSNR change (dB)"),
        ("ssim", -.0105, .0105, [-.01, -.005, 0, .005, .01], "SSIM change")
    ]):
        left, bottom, width, height = 42 + panel * 260, 42, 208, 140
        tx = lambda x: left + (x - xmin) / (xmax - xmin) * width
        ty = lambda y: bottom + y * height
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.HexColor("#18374a"))
        c.drawCentredString(left + width / 2, 203, title)
        for y in [0, .25, .5, .75, 1]:
            c.setStrokeColor(colors.HexColor("#dce2e6")); c.setLineWidth(.4)
            c.line(left, ty(y), left + width, ty(y))
            c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#4c5a64"))
            c.drawRightString(left - 6, ty(y) - 2, f"{int(y * 100)}%")
        for x in ticks:
            c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#4c5a64"))
            c.drawCentredString(tx(x), bottom - 12, f"{x:g}")
        c.setDash(3, 3); c.setStrokeColor(colors.HexColor("#7b868c"))
        c.line(tx(0), bottom, tx(0), bottom + height); c.setDash()
        for split, count in [("train", 832), ("val", 92), ("test", 76)]:
            values = sorted(float(r[metric]) - float(base[r["key"]][metric])
                            for r in rows if r["split"] == split)
            assert len(values) == count
            path = c.beginPath(); path.moveTo(tx(xmin), ty(0))
            for i, value in enumerate(values):
                path.lineTo(tx(value), ty((i + 1) / count))
            path.lineTo(tx(xmax), ty(1))
            c.setStrokeColor(palette[split]); c.setLineWidth(1.1); c.drawPath(path)
        c.setStrokeColor(colors.HexColor("#66757f")); c.setLineWidth(.6)
        c.line(left, bottom, left + width, bottom); c.line(left, bottom, left, bottom + height)
    for i, (split, count) in enumerate([("train", 832), ("val", 92), ("test", 76)]):
        x = 128 + i * 110
        c.setStrokeColor(palette[split]); c.setLineWidth(2); c.line(x, 13, x + 16, 13)
        c.setFillColor(colors.HexColor("#33434d")); c.setFont("Helvetica", 8)
        c.drawString(x + 21, 10, f"{split} (n={count})")
    c.save()


def main():
    data = json.loads((OUT / "report_data.json").read_text())
    before, after = data["baseline"], data["summary"]
    metrics = [("PSNR / dB", "psnr_mean", 6), ("SSIM", "ssim_mean", 8),
               ("LPIPS", "lpips_mean", 8), ("平均色差", "delta_e2000_mean", 6),
               ("色差P95的图像均值", "delta_e2000_p95_mean", 6),
               ("亮度MAE", "lightness_mae_mean", 6),
               ("饱和度MAE", "saturation_mae_mean", 8),
               ("色相误差 / 度", "hue_error_degrees_mean", 6)]
    generated = ["% Generated from sealed metrics. Do not edit numbers manually.\n"]
    generated.append("\\newcommand{\\MainResults}{%\n" + table(
        ["指标", "基线1000张", "最终1000张", "最终值减基线"],
        [(label, fmt(before['overall'][key], digits), fmt(after['overall'][key], digits),
          f"{after['overall'][key]-before['overall'][key]:+.{digits}f}")
         for label, key, digits in metrics], "lrrr") + "}\n")
    generated.append("\\newcommand{\\SplitResults}{%\n" + table(
        ["指标", "训练832张", "验证92张", "测试76张"],
        [(label, *(fmt(after['splits'][s][key], digits) for s in ['train','val','test']))
         for label, key, digits in metrics], "lrrr") + "}\n")
    candidates = [("冻结基线", "不适用", 0, before['splits']['val']),
                  ("自适应3DLUT", "1250 / 1500", 63324, data['first_research']['lut_full_val'])]
    for key, label in [('global3d','global3d对照'),('pairwise_rank4','低秩rank4')]:
        result = data['paired_research']['arms'][key]
        candidates.append((label, f"{result['best_step']} / {result['steps']}",
                           63324 if key=='global3d' else 22779, result['full_val']))
    generated.append("\\newcommand{\\CandidateResults}{%\n" + table(
        ["方法", "所选步 / 实际步", "附加参数量", "PSNR / dB", "SSIM"],
        [(label, step, f"{params:,}", fmt(summary['psnr_mean']), fmt(summary['ssim_mean'],8))
         for label, step, params, summary in candidates], "llrrr") + "}\n")
    generated.append("\\newcommand{\\CandidateColorResults}{%\n" + table(
        ["指标", *(x[0] for x in candidates)],
        [(label, *(fmt(summary[key],digits) for _,_,_,summary in candidates))
         for label,key,digits in metrics[2:]], "lrrrr") + "}\n")
    (OUT / "tables.tex").write_text("\n".join(generated))
    examples = ["% Saved model predictions and reference data.\n"]
    for i, e in enumerate(data['examples']):
        key = e['key']
        if i % 2 == 0:
            examples.append("\\clearpage\n")
        examples.append(f"\\subsection*{{样例 {i+1}，图像 {key}}}\n")
        if i < 6:
            labels = [('raw','低光RAW预览'),('isp','传统ISP示意'),('gt','参考图')]
            examples.append("\\begin{center}\n" + "\\hfill\n".join(
                f"\\Panel{{case_{key}_{suffix}.jpg}}{{{label}}}"
                for suffix,label in labels) + "\n\\end{center}\n")
            examples.append("\\begin{center}\n" + "\\hfill\n".join([
                f"\\Panel{{case_{key}_baseline.jpg}}{{冻结基线}}",
                f"\\Panel{{case_{key}_final.jpg}}{{最终LUT方案}}",
                "\\begin{minipage}[t]{0.318\\linewidth}\\vspace{0pt}\\raggedright\\small "
                f"PSNR由{e['baseline']['psnr']:.4f}变为{e['final']['psnr']:.4f}\\,dB。\\par "
                f"SSIM由{e['baseline']['ssim']:.6f}变为{e['final']['ssim']:.6f}。\\par "
                f"LPIPS由{e['baseline']['lpips']:.6f}变为{e['final']['lpips']:.6f}。\\end{{minipage}}"
            ]) + "\n\\end{center}\n")
        else:
            for crop, note in [(False,"整幅图像"),(True,"相同中心区域")]:
                examples.append("\\begin{center}\n" + "\\hfill\n".join(
                    f"\\Panel{{case_{key}_{label}{'_crop' if crop else ''}.jpg}}{{{title}，{note}}}"
                    for label,title in [('baseline','冻结基线'),('final','最终LUT方案'),('gt','参考图')]
                    ) + "\n\\end{center}\n")
            examples.append(f"PSNR由{e['baseline']['psnr']:.4f}变为{e['final']['psnr']:.4f}\\,dB。"
                            f"SSIM由{e['baseline']['ssim']:.6f}变为{e['final']['ssim']:.6f}。"
                            f"LPIPS由{e['baseline']['lpips']:.6f}变为{e['final']['lpips']:.6f}。\\par\n")
        trend = "提高" if e['psnr_delta'] > 0 else "下降"
        examples.append(f"本例PSNR{trend}{abs(e['psnr_delta']):.4f}\\,dB。"
                        "图像仅为排版等比例缩放，所有分数来自封存的原生浮点评价。\\par\\vspace{0.5em}\n")
    (OUT / "examples.tex").write_text("\n".join(examples))
    weight_lines = []
    for name, item in data['weights'].items():
        name_tex=name.replace('_',r'\_')
        weight_lines.append(f"\\subsection*{{\\texttt{{{name_tex}}}}}\n")
        weight_lines.append(f"文件大小为{item['bytes']:,}字节。\\par\n")
        chunks=[item['sha256'][i:i+16] for i in range(0,64,16)]
        weight_lines.append("\\noindent SHA256\\quad\\texttt{" + r"\allowbreak ".join(chunks) + "}\\par\n")
        weight_lines.append(("由SIED作者发布。按原发布地址获取并核对哈希，未放入自训权重附件。"
                             if name.startswith('official') else "本项目训练的权重，已收入权重附件。")+"\\par\n")
    (OUT / "checkpoints.tex").write_text("\n".join(weight_lines))
    plot_distribution(data)
    print(json.dumps({"generated":['tables.tex','examples.tex','checkpoints.tex','images/delta_distribution.pdf'],
                      "examples":len(data['examples'])},ensure_ascii=False))


if __name__ == "__main__":
    main()
