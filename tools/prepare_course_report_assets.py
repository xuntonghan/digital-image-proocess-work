"""Build tables and faithful saved-image panels for geometry and compression."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/sied_final_report_20261025"
ASSETS = ROOT / "reports/assets"


def load(path):
    return json.loads(path.read_text())


def table(headers, rows, spec):
    return ("\\begin{tabular}{" + spec + "}\\toprule\n" + " & ".join(headers)
            + " \\\\\n\\midrule\n" + "\n".join(" & ".join(map(str, row)) + " \\\\" for row in rows)
            + "\n\\bottomrule\\end{tabular}\n")


def crop(source, target, xy):
    with Image.open(source) as im:
        im.crop(xy).convert("RGB").save(OUT / "images" / target, quality=98, subsampling=0)
    return {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "display": target, "crop_xyxy": xy, "processing": "Exact saved montage panel; no enhancement"}


def main():
    geo = {key: load(ASSETS / f"geocorrection/{key}/summary.json") for key in
           ("classical", "uvdoc", "hybrid", "page_mask_unet", "final_4090/predicted", "final_4090/oracle", "ocr")}
    comp = {key: load(ASSETS / f"compression/{key}/summary.json") for key in
            ("jpeg_baseline", "dctc_v1", "dctc_v2_fast", "cheng2020_anchor")}
    checks = []
    for key, summary in geo.items():
        path = ASSETS / f"geocorrection/{key}/metrics.csv"
        if not path.exists() or key == "ocr":
            continue
        rows = list(csv.DictReader(path.open(newline="")))
        assert len(rows) == summary["input_count"]
        for name, expected in summary["overall"].items():
            col = name.removesuffix("_mean")
            if name.endswith("_mean") and col in rows[0]:
                values = [float(r[col]) for r in rows if r[col] not in ("", "None", "null")]
                assert math.isclose(statistics.fmean(values), expected, rel_tol=1e-10, abs_tol=1e-10), (key, name)
        checks.append({"dataset": key, "rows": len(rows), "means_verified": True})
    rows_quality, rows_rate = [], []
    labels = {"jpeg_baseline": "JPEG", "dctc_v1": "DCTC v1", "dctc_v2_fast": "DCTC v2", "cheng2020_anchor": "Cheng-A"}
    for key, summary in comp.items():
        rows = list(csv.DictReader((ASSETS / f"compression/{key}/metrics.csv").open(newline="")))
        for q in summary['qualities']:
            sub = [r for r in rows if int(r['quality']) == q['quality']]
            assert len(sub) == 24 and len({r['key'] for r in sub}) == 24
            assert sum(int(r['compressed_bytes']) for r in sub) == q['compressed_bytes_total']
            for col in ['psnr', 'ssim', 'encode_ms', 'decode_ms', 'bits_per_pixel']:
                assert math.isclose(statistics.fmean(float(r[col]) for r in sub), q[col+'_mean'], rel_tol=1e-10, abs_tol=1e-10), (key,col)
            size=q['compressed_bytes_total']
            rows_quality.append([labels[key], q['quality'], f"{q['bits_per_pixel_mean']:.6f}", f"{q['psnr_mean']:.6f}",
                                 f"{q['ssim_mean']:.8f}", f"{q['encode_ms_mean']:.4f}", f"{q['decode_ms_mean']:.4f}"])
            rows_rate.append([labels[key], q['quality'], f"{size:,}", f"{28311552/size:.6f}",
                              f"{15394305/size:.6f}", f"{100*(1-size/28311552):.4f}"])
        checks.append({"dataset":key,"rows":len(rows),"means_verified":True})
    generated = ['% Tables independently checked against the archived per-image CSV.\n']
    generated.append('\\newcommand{\\CompressionQuality}{%\n'+table(
        ['方法','Q','bpp','PSNR / dB','SSIM','编码 / ms','解码 / ms'], rows_quality, 'llrrrrr')+'}\n')
    generated.append('\\newcommand{\\CompressionRates}{%\n'+table(
        ['方法','Q','码流字节','RGB压缩比','PNG压缩比','RGB节省 / \\%'],rows_rate,'llrrrr')+'}\n')
    s=geo['page_mask_unet']; seg=[]
    for label,key in [('验证22张','validation'),('测试22张','test')]:
        seg.append([label,*[f"{s[key][name]:.6f}" for name in ['iou','dice','precision','recall','boundary_f1']]])
    generated.append('\\newcommand{\\SegmentationResults}{%\n'+table(
        ['集合','IoU','Dice','Precision','Recall','边界F1'],seg,'lrrrrr')+'}\n')
    fields=[('边界RMSE','boundary_rmse_normalized',8),('边界P95','boundary_p95_normalized',8),
            ('页面填充率','page_fill_ratio',6),('文本倾斜 / 度','text_line_skew_degrees',6)]
    auto=[]
    for label,name,digits in fields:
        auto.append([label,*[f"{geo[setting]['overall'][stage+'_'+name+'_mean']:.{digits}f}"
                            for setting,stage in [('final_4090/predicted','input'),('final_4090/predicted','corrected'),
                                                  ('final_4090/oracle','input'),('final_4090/oracle','corrected')]]])
    generated.append('\\newcommand{\\AutomaticGeometry}{%\n'+table(
        ['指标','预测掩膜输入','预测掩膜输出','真值掩膜输入','真值掩膜输出'],auto,'lrrrr')+'}\n')
    old=[]
    for label,setting,stage in [('传统输入','classical','input'),('传统校正','classical','corrected'),
                                ('UVDoc裁剪输入','uvdoc','input'),('UVDoc输出','uvdoc','uvdoc'),('混合输出','hybrid','corrected')]:
        d=geo[setting]['overall']
        old.append([label,*[f"{d[stage+'_'+name+'_mean']:.{digits}f}" for _,name,digits in fields]])
    generated.append('\\newcommand{\\HistoricalGeometry}{%\n'+table(
        ['设置','边界RMSE','边界P95','填充率','倾斜 / 度'],old,'lrrrr')+'}\n')
    ocr=[]
    for key,label in [('input','输入'),('classical','传统'),('uvdoc','UVDoc'),('hybrid','混合')]:
        d=geo['ocr']['overall'][key]
        ocr.append([label, f"{d['line_count_total']:,}", f"{d['confident_line_count_total']:,}",
                    f"{d['confident_character_count_total']:,}", f"{d['mean_line_confidence']:.6f}",
                    f"{d['confidence_weighted_characters_total']:.2f}"])
    generated.append('\\newcommand{\\OCRResults}{%\n'+table(
        ['方法','文本行','高置信行','高置信字符','平均置信度','加权字符'],ocr,'lrrrrr')+'}\n')
    (OUT/'course_tables.tex').write_text('\n'.join(generated))
    panels=[];figures=[]
    keys=geo['hybrid']['visualization_keys']
    for i,key in enumerate(keys):
        if i%4==0:figures.append('\\clearpage\n')
        figures.append(f'\\subsection*{{几何样例 {i+1}，\\texttt{{{key.replace("_", chr(92)+"_")}}}}}\n')
        figures.append('\\noindent\n'+'\\hfill\n'.join(
            f'\\GeoPanel{{geo_{i}_{j}.jpg}}{{{label}}}' for j,label in enumerate(['页面裁剪','传统校正','UVDoc','UVDoc后精修']))+'\\par\n')
        for j in range(4):
            x=190+j*372;y=34+i*296+24
            panels.append(crop(ASSETS/'geocorrection/hybrid/comparison.png',f'geo_{i}_{j}.jpg',(x,y,x+360,y+260)))
        figures.append('四列来自同一封存样例。此组使用数据集掩膜辅助裁剪。\\par\\vspace{1em}\n')
    figures.append('\\subsection*{自动校正测试样例}原图、预测掩膜、单应性和边界校正依次列出。图像编号为\\texttt{dewarp20260728\\_perspective\\_019}。\\par\n')
    figures.append('\\noindent\n'+'\\hfill\n'.join(f'\\GeoPanel{{geo_auto_{j}.jpg}}{{{label}}}' for j,label in enumerate(['原始图像','预测掩膜','单应性','边界校正']))+'\\par\n')
    for j in range(4):
        x=170+j*330;panels.append(crop(ASSETS/'geocorrection/final_4090/predicted/comparison.png',f'geo_auto_{j}.jpg',(x,38,x+330,273)))
    figures.append('该图来自22张隔离测试。其掩膜由PageMaskUNet预测。页面外边界更规整，页内形变仍有残留。\\par\n')
    (OUT/'geometry_examples.tex').write_text('\n'.join(figures))
    figures=[]
    for i,key in enumerate(['01','04','13','23','24']):
        figures.append(f'\\clearpage\\subsection*{{Kodak样例 {key}}}\n')
        for row, items in enumerate([
            [('cheng2020_anchor',0,'原图'),('cheng2020_anchor',1,'Cheng Q1'),('cheng2020_anchor',2,'Cheng Q3')],
            [('jpeg_baseline',3,'JPEG Q70'),('dctc_v1',3,'DCTC v1 Q70'),('dctc_v2_fast',3,'DCTC v2 Q70')],
            [('cheng2020_anchor',3,'Cheng Q6'),('jpeg_baseline',4,'JPEG Q90'),('dctc_v1',4,'DCTC v1 Q90')]
        ]):
            names=[]
            for j,(kind,col,label) in enumerate(items):
                name=f'compression_{key}_{row}_{j}.jpg';x=58+col*300;y=38+i*220
                panels.append(crop(ASSETS/f'compression/{kind}/comparison.png',name,(x,y,x+300,y+220)))
                names.append(f'\\Panel{{{name}}}{{{label}}}')
            figures.append('\\begin{center}\n'+'\\hfill\n'.join(names)+'\n\\end{center}\n')
        figures.append('各列保留原封存图的相同画面范围。质量档和码率不同，不能只按Q的数值比较算法。\\par\n')
    (OUT/'compression_examples.tex').write_text('\n'.join(figures))
    # A readable vector plot, using the same archived operating points as the tables.
    c=canvas.Canvas(str(OUT/'images/compression_rd.pdf'),pagesize=(520,238))
    palette=['#24547b','#b36539','#779143','#7b5aa6']
    for panel,(metric,title,ymin,ymax) in enumerate([('psnr_mean','PSNR (dB)',27,40),('ssim_mean','SSIM',.74,.98)]):
        left,bottom,w,h=43+260*panel,44,205,155
        tx=lambda v:left+float(v)/3.8*w;ty=lambda v:bottom+(float(v)-ymin)/(ymax-ymin)*h
        c.setFont('Helvetica-Bold',10);c.drawCentredString(left+w/2,220,title)
        for f in [0,.25,.5,.75,1]:
            c.setStrokeColor(colors.HexColor('#dce2e6'));c.setLineWidth(.4);c.line(left,bottom+f*h,left+w,bottom+f*h)
            c.setFont('Helvetica',7);c.setFillColor(colors.HexColor('#33434d'));c.drawRightString(left-5,bottom+f*h-2,f'{ymin+f*(ymax-ymin):.2f}')
        for x in [0,1,2,3]:c.drawCentredString(tx(x),bottom-13,str(x))
        for i,(key,d) in enumerate(comp.items()):
            points=sorted(d['qualities'],key=lambda q:q['bits_per_pixel_mean']);p=c.beginPath()
            for j,q in enumerate(points):
                (p.moveTo if j==0 else p.lineTo)(tx(q['bits_per_pixel_mean']),ty(q[metric]))
            c.setStrokeColor(colors.HexColor(palette[i]));c.setLineWidth(1.2);c.drawPath(p)
            for q in points:c.circle(tx(q['bits_per_pixel_mean']),ty(q[metric]),2,stroke=1,fill=0)
        c.setFillColor(colors.HexColor('#33434d'));c.setFont('Helvetica',8);c.drawCentredString(left+w/2,bottom-27,'bits per pixel')
    for i,key in enumerate(comp):
        x=51+i*122;c.setStrokeColor(colors.HexColor(palette[i]));c.setLineWidth(2);c.line(x,8,x+15,8)
        c.setFillColor(colors.HexColor('#33434d'));c.setFont('Helvetica',8);c.drawString(x+19,5,labels[key])
    c.save()
    (OUT/'course_report_data.json').write_text(json.dumps({'geometry':geo,'compression':comp,'panels':panels,'checks':checks},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'panels':len(panels),'checks':checks},ensure_ascii=False))


if __name__=='__main__':main()
