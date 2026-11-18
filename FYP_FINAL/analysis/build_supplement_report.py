"""Create a Word review report from computed supplementary tables.

Uses bundled Codex Python and python-docx; scientific analysis runs separately.
"""
from pathlib import Path
from copy import deepcopy
import csv
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'analysis/outputs'
FIG=ROOT/'manuscript/figures/supplementary'
BASE=ROOT/'manuscript/drafts/FYP_Word_Review_01.docx'
DEST=ROOT/'manuscript/drafts/FYP_Supplementary_Analysis_20261118.docx'
def read(name):
 with (O/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def find(rows,**kw):return next(r for r in rows if all(r[k]==v for k,v in kw.items()))
def num(row,k):return f'{float(row[k]):.4f}'
scale=read('scale_groups.csv');contrast=read('boundary_contrast_groups.csv')
prov=json.loads((O/'provenance.json').read_text())
doc=Document(BASE);body=doc._element.body;sect=deepcopy(body.sectPr)
for e in list(body):body.remove(e)
body.append(sect)
normal=doc.styles['Normal'];normal.paragraph_format.left_indent=Pt(0);normal.paragraph_format.right_indent=Pt(0)
normal.paragraph_format.first_line_indent=Pt(0)
normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(6)
for n in ['Heading 1','Heading 2','Title']:
 doc.styles[n].paragraph_format.left_indent=Pt(0);doc.styles[n].paragraph_format.right_indent=Pt(0)
pending=False
def p(text='',style=None):
 global pending
 a=doc.add_paragraph(text,style)
 snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');a._p.get_or_add_pPr().append(snap)
 if pending:a.paragraph_format.page_break_before=True;pending=False
 return a
def h(t,l=1):return p(t,'Heading '+str(l))
def page():
 global pending
 pending=True
def fig(name,caption):
 a=p();a.alignment=WD_ALIGN_PARAGRAPH.CENTER
 a.add_run().add_picture(str(FIG/f'{name}.png'),width=Cm(16.4))
 a.paragraph_format.keep_with_next=True
 a=p(caption,'Caption');a.paragraph_format.space_after=Pt(8)
def table(headers,rows,widths):
 t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
 for c,w in zip(t.columns,widths):c.width=Cm(w)
 for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
 for row in rows:
  for c,s in zip(t.add_row().cells,row):c.text=str(s)
 for i,row in enumerate(t.rows):
  for cell,w in zip(row.cells,widths):
   cell.width=Cm(w)
   tc=cell._tc.get_or_add_tcPr()
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
   tc.append(borders)
   if i==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EDF3');tc.append(shade)
   for a in cell.paragraphs:
    a.paragraph_format.space_after=Pt(3);a.paragraph_format.space_before=Pt(3);a.paragraph_format.line_spacing=1
    a.paragraph_format.keep_with_next=False
    for r in a.runs:r.font.size=Pt(10);r.bold=i==0
  if i==0:
   rp=row._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');rp.append(repeat)
 return t

p('Supplementary Polyp Segmentation Analysis','Title')
p('补充实验与提示工具包审阅报告 | 2026年11月18日')
p('本报告汇总本轮新增分析及其适用范围，供后续整合进Word论文。复用现有图像、真实掩膜、正式逐图记录和保存预测，未重新训练模型，也未改写正式结果。所有新增分析均为事后描述性分析，仍需作者和导师审阅。')
h('本轮交付')
table(['项目','已完成内容','解释范围'],[
 ['目标尺度','732张图像、8模型；固定面积分组与训练集三分位敏感性','图像相对面积，不是临床毫米尺寸'],
 ['弱边界探索','颜色对比度分组、梯度敏感性、尺度交叉分组、边界F1与表面HD95','图像代理指标，未作临床弱边界标注验证'],
 ['提示敏感性','整理28个数据集／模型／外扩量组合的原始JSON结果并绘图','固定框外扩；未新增点提示或位移推理'],
 ['提示工具包','独立脚本、使用说明及732条提示记录','支持框与点生成；依赖真实标注']],[2.3,8,6.2])
h('数据与一致性检查',2)
p('Kvasir验证集120张，CVC评估集612张。8模型共5856条逐图结果按数据集和文件名逐一匹配；5856份保存预测重算Dice与正式逐图值的绝对差均不超过1e−6。分组加权均值还原正式Dice／IoU均值，缺失或重复ID均会触发失败。')
h('发现的指标命名问题',2)
p('已检查的原始SAM与PP-SAM评估函数将最大Hausdorff距离写入名为hd95的字段，未计算95百分位。旧字段保留为legacy_distance，不再直接解释为HD95。本轮另算双向表面距离各自95百分位的较大值，单位为原图像素；边界F1容差为图像对角线的0.5%，至少1像素。')
p('有5条空预测：边界F1记0，表面HD95不定义，表中保留有效数量而非填0。新指标与旧距离不同，不能只修改表头后继续使用旧数值。')

page();h('目标尺度分析')
p('面积比为真实前景像素数除以整幅图像像素数。小目标≤5%，中目标>5%且≤20%，大目标>20%。阈值是本研究的操作定义；另用880张训练掩膜的面积三分位数作敏感性分析，不根据模型得分调整分组。')
fig('scale_dice','图S1 8个模型的面积分组Dice。组内样本数在横轴标明；连线仅辅助阅读有序分组。')
rows=[]
for ds,short in [('Kvasir-SEG-val','Kvasir'),('CVC-ClinicDB','CVC')]:
 for model in ['SAM-B','SAMed','MedSAM+LoRA','PP-SAM']:
  vals=[find(scale,dataset=ds,model=model,size_group=g) for g in ['Small','Medium','Large']]
  rows.append([short,model]+[num(r,'dice_mean') for r in vals])
table(['数据集','模型','小目标Dice','中目标Dice','大目标Dice'],rows,[2.2,3.6,3.5,3.5,3.7])
p('Kvasir各组n=29／61／30，CVC各组n=229／321／62。CVC上PP-SAM由小目标0.7970升至大目标0.8986；SAM-B却由0.5573降至0.2295，因此不能概括为所有模型均在小目标上更差。面积分组同时受到形态、外观及固定像素框相对大小的影响。')

page();h('边界对比度探索分析')
p('在真实边界内外分别取窄带，计算两个带状区域的平均CIELAB颜色距离。主分析带宽为图像对角线的0.5%，至少2像素。每个数据集按对比度Q25和Q75划分低、中、高组；分组只使用图像与真实标注，不使用模型预测或得分。')
fig('contrast_boundary_f1','图S2 边界F1按颜色对比度代理值分组。低组代表较小的平均颜色分离，不等同于临床标注的弱边界。')
rows=[]
for ds,short in [('Kvasir-SEG-val','Kvasir'),('CVC-ClinicDB','CVC')]:
 for m in ['SAM-B','SAMed','MedSAM+LoRA','PP-SAM']:
  lo=find(contrast,dataset=ds,model=m,contrast_main_group='Low');hi=find(contrast,dataset=ds,model=m,contrast_main_group='High')
  rows.append([short,m,num(lo,'dice_mean'),num(hi,'dice_mean'),num(lo,'boundary_f1_mean'),num(hi,'boundary_f1_mean')])
table(['数据集','模型','低组Dice','高组Dice','低组BF1','高组BF1'],rows,[2,3.2,2.8,2.8,2.8,2.9])
p('低／中／高组数量为Kvasir30／60／30、CVC153／306／153。CVC的SAMed低组BF1为0.4787，高组0.5617；但SAM-B的高组Dice反而更低。对比度分组对重叠与边界指标的关系不同，不能据此声称低对比度必然导致更差的所有指标。')

page();h('代理指标的检查与限制')
fig('boundary_proxy_examples','图S3 每个数据集在中等面积组内选取对比度最低2张与最高2张；青色为真实轮廓。按图像特征选图，未按预测好坏筛选。ID见boundary_proxy_example_ids.csv。')
p('这些示例用于核查代理指标的含义，不能替代专家标注。均值颜色距离可能忽略局部模糊、纹理和反光，出现颜色相消；也不能据此判断病理或临床难度。')
p('尺度与颜色对比度存在相关：Spearman相关系数在Kvasir为0.2574，在CVC为0.3707。已输出尺度×对比度交叉分组表，但稀疏组不能支持可靠的独立效应结论。')
p('敏感性检查使用0.25%和1%对角线带宽，以及真实边界上的灰度梯度中位数。CVC图像的0.25%与0.5%带宽均因至少2像素规则取2，二者不构成独立的带宽变化。CVC的MSA-L在主对比度定义下高减低组BF1为−0.0231，在较宽带下为+0.0037，说明趋势依赖定义。')
p('边界梯度分组中，两数据集的8模型均表现为高组平均BF1更高；该结果作为敏感性分析报告，不因趋势更整齐而替换已记录的主代理定义。灰度梯度同样可能受纹理、反光和成像条件影响。')

page();h('提示框敏感性与工具包')
fig('prompt_expansion','图S4 仅使用现有原始summary JSON绘制固定框外扩曲线。CVC缺少MedSAM+LoRA的0／10／50原始JSON，因此没有连接该模型的CVC曲线。')
p('PP-SAM在两个数据集均于10像素外扩取得本次四档中的最高Dice；CVC从0.8948降至50像素的0.8025。该结果说明已测试模型对框范围敏感，不代表存在适用于所有模型的全局最佳外扩量。SAMed的Kvasir box30消融仍用本机0.8951，主表保留云端0.8955。')
h('可复用提示接口',2)
p('tools/prompt_toolkit.py提供紧框／固定外扩框、确定性内部正点、随机前景正点、背景负点以及四向平移框。已生成Kvasir120条和CVC612条记录，包含坐标、尺寸、随机种子和掩膜SHA256。')
p('5项几何测试与2项表面指标测试通过；另用100个随机掩膜×4档外扩，共400次对照，确认legacy_v3框与原SAM评估函数一致。提供half_open坐标选项，明确处理空掩膜、全前景、单像素和裁剪后的退化框。')
p('点提示和位移框仅完成生成与几何检查，没有运行新的模型评估；各模型仍需自己的缩放与坐标适配。工具依赖真实掩膜，不能称作自动息肉检测器。命令、依赖和接入示例见tools/README.md。')

page();h('可纳入论文的英文结果草稿')
p('以下段落需与方法部分一同整合，并由作者核验。相应新表与图尚未编入正式论文章节号。')
h('Performance across relative object sizes',2)
p('Object scale was defined as the fraction of image pixels occupied by the ground-truth foreground. Images were grouped into small (at most 5%), medium (more than 5% and at most 20%), and large (more than 20%) foreground regions. The Kvasir validation groups contained 29, 61, and 30 images, respectively, while the corresponding CVC groups contained 229, 321, and 62 images. These categories describe relative image area rather than physical lesion dimensions.')
p('The relationship between scale and overlap accuracy differed across models. On CVC, PP-SAM achieved mean Dice scores of 0.7970, 0.8930, and 0.8986 across the small, medium, and large groups. SAM ViT-B showed a different pattern, with corresponding scores of 0.5573, 0.5460, and 0.2295. The observed results therefore do not support a general statement that smaller polyps were more difficult for every evaluated model. Other image properties and the relative extent of the fixed-pixel box prompt may contribute to the group differences.')
h('Boundary visibility proxy analysis',2)
p('Boundary visibility was explored using the colour separation between narrow foreground and background bands adjacent to the ground-truth contour. This image-derived measure was treated as a proxy rather than an expert annotation of weak boundaries. In the CVC colour-contrast groups, SAMed achieved mean boundary F1 scores of 0.4787 in the lowest quartile and 0.5617 in the highest quartile. However, the relationship was not uniform across methods and outcomes. SAM ViT-B had lower mean Dice in the high-contrast group despite a higher boundary F1 score.')
p('Relative area and the contrast proxy were correlated, with Spearman coefficients of 0.2574 on Kvasir and 0.3707 on CVC. Crossed size and contrast tables and alternative band widths were therefore examined descriptively. The results remain dependent on the operational proxy and do not establish an independent causal effect of boundary visibility. Further validation with expert boundary assessments would be needed before interpreting these groups as clinically defined weak-boundary cases.')
h('Sensitivity to box extent',2)
p('The existing fixed-expansion evaluations showed that prompt extent affected the recorded segmentation results. PP-SAM obtained its highest mean Dice among the four tested settings at an expansion of 10 pixels on both datasets. On CVC, its mean Dice decreased from 0.8948 at 10 pixels to 0.8025 at 50 pixels. This comparison concerns deterministic outward expansion of ground-truth-derived boxes; it does not evaluate displaced user prompts, point prompts, or automatic lesion localisation.')

page();h('复现入口与证据状态')
p('新增分析协议：analysis/PROTOCOL.md。执行脚本：analysis/complete_supplement.py。所有原始图像、模型权重、正式summary和per-image文件均作为只读输入；结果写入本项目analysis/outputs。')
table(['输出','用途'],[
 ['image_features.csv','732条图像面积、对比度、梯度及分组'],
 ['per_image_joined.csv','5856条正式指标与新增表面指标'],
 ['scale_groups.csv / scale_training_tertiles.csv','主要尺度表与训练分位数敏感性'],
 ['boundary_*_sensitivity.csv','带宽及梯度定义敏感性'],
 ['scale_contrast_crossed.csv','尺度与对比度交叉分组'],
 ['prediction_audit.csv / summary_reconciliation.csv','预测来源匹配及正式均值复核'],
 ['prompt_expansion_curves.csv','可追溯的框外扩曲线数据'],
 ['provenance.json','输入路径、SHA256、软件与阈值记录']],[8,8.5])
p('复现命令：使用现有sam_env的Python运行analysis/complete_supplement.py。生成Word报告需使用Codex工作区依赖中的Python运行analysis/build_supplement_report.py。提示工具包的完整命令见tools/README.md。')
h('现在可以如何表述',2)
p('目标尺度：已补充系统的图像相对面积分组描述性分析。弱边界：已补充操作定义明确的图像对比度／梯度探索分析及表面指标，尚无临床或专家弱边界标签。提示质量：已有固定外扩敏感性研究，不宣称覆盖所有提示方式。工具包：已整理为可执行、可复用的标注派生提示脚本与说明，仍需模型特定接入。')
p('这轮补充不改变多种子、显著性检验、独立测试和临床验证尚未完成的状态，也不构成SOTA或部署效果证明。')

doc.save(DEST)
print(str(DEST))
