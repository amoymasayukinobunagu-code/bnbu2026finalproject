"""Build the first Word review draft from the retained university template.

Run with the Codex bundled Python. Preserve all untouched OPC package parts.
"""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
import hashlib
import json
import zipfile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'manuscript/working'
OUT = ROOT / 'manuscript/drafts'
OUT.mkdir(exist_ok=True)
REF = ROOT / 'templet/[GroupID] FYP Thesis.docx'
DEST = OUT / 'FYP_Word_Review_02.docx'
sha = lambda b: hashlib.sha256(b).hexdigest()
ref_hash = sha(REF.read_bytes())

research_questions = [
 ('RQ1 零样本表现', '在相同的框提示协议下，零样本 SAM 和 MedSAM 在 Kvasir-SEG 验证集和 CVC-ClinicDB 上的息肉分割表现有何差异？'),
 ('RQ2 息肉领域适配', '使用完整 Kvasir-SEG 训练集进行领域适配后，LoRA 和 Adapter 模型相对于零样本基线，在 Kvasir-SEG 验证集和 CVC-ClinicDB 上的表现有何变化？'),
 ('RQ3 提示与图像条件（探索性）', '对已评估的模型，改变提示形式、框外扩量或框位置时，分割表现如何变化？按息肉相对面积和边界对比度分组后，又观察到哪些差异？'),
]

study_mapping = [
 ('RQ1', 'SAM ViT-B、ViT-L 与 MedSAM 的零样本比较。'),
 ('RQ2', '全量训练的 LoRA 和 Adapter 与零样本基线比较。PP-SAM 为全参参照，标注预算另作补充。'),
 ('RQ3', '框外扩、点提示和框位移，以及相对面积与边界对比度分组分析。'),
]

chapters = [
('1 Introduction', '从研究动机引出可由现有实验回答的问题，明确本研究的贡献与适用条件。', [
('1.1 Background and motivation', '说明息肉像素级分割的任务及困难，引出可提示基础模型与领域适配的研究动机。临床意义须由原论文支持，避免将本项目结果直接写成临床收益。'),
('1.2 Research problem', '说明零样本模型向息肉图像迁移的问题，以及在给定提示条件下评估息肉领域适配的动机。'),
('1.3 Research questions', '采用本稿首页的三项研究问题。RQ1 和 RQ2 构成主线，RQ3 探索提示条件、相对面积和边界对比度。'),
('1.4 Contributions and scope', '介绍复现与适配流程、统一协议下的模型比较及探索分析。界定由真实标注生成的提示和各项实验的数据范围。'),
('1.5 Thesis organisation', '用一个短段落概述各章内容，保持与最终章节一致。')], '背景与动机需补经核验的文献。第5章按 RQ1、RQ2 和 RQ3 回答。'),
('2 Related Work', '解释选取这些方法的依据，并指出本研究能够补充的经验性比较。', [
('2.1 Endoscopic polyp segmentation', '概述任务、数据集及监督分割研究。引用不同论文的性能时注明划分和提示条件，不构造未经对齐的排名。'),
('2.2 SAM and medical adaptation', '介绍 SAM 的可提示分割机制及 MedSAM 的医学适配背景，区分预训练能力与本项目中的息肉训练。'),
('2.3 Parameter efficient adaptation', '介绍 LoRA、SAMed 和 Medical SAM Adapter，说明参数更新方式。具体插入位置及训练范围在核对实现后写入。'),
('2.4 Limited annotation and prompt sensitivity', '介绍 PP-SAM 及与标注预算、提示信息相关的研究，为少样本和框消融实验提供依据。'),
('2.5 Position of this study', '说明本研究如何在共同的框提示协议下比较二维逐帧息肉分割模型，并考察领域适配、提示形式、目标尺度和边界对比度。')], '计划表2.1：方法、适配方式、训练数据需求及提示输入。现有34条 BibTeX 是待核验来源，不等同于已核验引用。'),
('3 Research Methodology', '解释研究如何设计，以及每组实验能够回答什么问题。', [
('3.1 Study design', '概述两个核心比较与一项探索性分析，并说明各实验如何回答研究问题。'),
('3.2 Datasets and partitioning', 'Kvasir-SEG 的 880 张训练图像用于适配，120 张验证图像用于权重选择及域内评估。CVC-ClinicDB 的 612 张图像用于跨数据集评估。交代 ETIS 和 CVC-ColonDB 在外部评估中的用途。'),
('3.3 Model groups and comparisons', '区分零样本、LoRA、Adapter 和 PP-SAM 全参参照；列明骨干、初始化及可训练模块，避免把不同训练方案视为严格单因素消融。'),
('3.4 Prompt construction and variants', '先说明主比较的默认提示：将真实掩膜的紧框向外扩展 30 像素。再介绍三种提示变化：固定外扩量检验框的大小，框位移检验框的位置，点提示检验另一种输入形式。框位移和点提示仅在 SAM-B 与 MSA-B 上评估。'),
('3.5 Evaluation and subgroup analysis', ''),
('3.5.1 Evaluation metrics', '定义 Dice、IoU、精确率、召回率、表面 HD95 和边界 F1，交代距离单位、容差与空预测处理。'),
('3.5.2 Image subgroup definitions', '定义真实掩膜相对面积及其分组阈值。以边界内外颜色差分组，并用灰度梯度检视趋势。'),
('3.6 Statistical analysis and validity', ''),
('3.6.1 Statistical comparisons', '说明逐图配对比较。CVC 的提示比较以视频序列为单位汇总，报告序列等权的 Dice 变化及 Holm 校正结果。'),
('3.6.2 Validity of comparisons', '说明 Kvasir 验证集参与权重选择、标注生成提示和边界代理指标的解释范围。固定外扩对不同大小的息肉产生不同程度的相对扩张，可能影响面积组之间的差异。')], '拟配整体研究框架图、提示生成示意图，以及数据与模型设置表。'),
('4 Design and Implementation', '说明模型如何从输入图像和提示生成可复现的评估结果。', [
('4.1 Experimental pipeline', '概述两条路径。零样本模型加载已有权重后直接评估。息肉适配模型先训练并选择权重，再进入评估。两条路径采用相同的评估协议，各模型沿用各自的推理设置。'),
('4.2 Input preprocessing and prompt preparation', '说明训练与推理如何准备输入，包括图像缩放、归一化和提示坐标映射。第 3 章定义提示规则，本节说明怎样将这些提示传入各模型。'),
('4.3 Model adaptation and training', '用表 4.1 汇总 SAMed、MedSAM+LoRA、MSA 和 PP-SAM 的初始权重、可训练模块、损失、优化器和训练轮数，并说明权重选择规则。'),
('4.4 Inference and evaluation pipeline', '按生成掩膜、选择输出、恢复到原图尺寸、计算逐图指标和汇总结果的顺序说明实现。举例说明 SAM-B 使用单掩膜输出，MSA-B 依据预测质量从候选掩膜中选取一个。'),
('4.5 Reproducibility', '将正文主要结果对应到代码入口、权重、数据划分、配置和输出文件。注明随机种子用于训练、支持集抽样还是随机点提示。完整命令和路径放入附录。')], '图 4.1 展示零样本与适配两条实现路径怎样汇入评估。表 4.1 汇总实际训练配置。LoRA 和 Adapter 的结构示意图待核对代码后再决定是否加入。'),
('5 Results and Discussion', '开头用一段引入主结果表，说明默认框与两个主数据集。正文依次讨论零样本表现、适配后的变化，以及提示和图像条件下的表现差异。', [
('5.1 Zero-shot performance（RQ1）', '用主表比较 SAM-B、SAM-L 与 MedSAM，说明零样本结果如何随评估数据集变化。'),
('5.2 Polyp-specific adaptation（RQ2）', ''),
('5.2.1 Full-data comparison', '在两个主数据集上比较全量训练的 LoRA 和 Adapter 与零样本基线。PP-SAM 作为已有的全参适配对照。'),
('5.2.2 External evaluation of SAM-B and MSA-B', '比较两种配置在额外数据集上的平均 Dice。MSA-B 在 CVC-ColonDB 上得分较高，SAM-B 在 ETIS 上得分较高。'),
('5.2.3 Annotation budget', '用 SAMed 和 PP-SAM 的 1、5、10 和 880 张实验说明不同标注预算下的表现。短程重复训练作为补充证据。'),
('5.3 Prompt sensitivity（RQ3）', '比较同一张图更换提示前后的表现。依次报告框外扩、框位移和点提示结果，并说明 CVC 序列级检验。'),
('5.4 Object scale and boundary contrast（RQ3）', '比较不同图像组的表现，分别报告相对面积和边界对比度的结果。将定性案例放在对应结果旁。'),
('5.5 Inference speed', '单列 SAM-B、SAM-L 和 MSA-B 的统一测速，说明各配置的推理分辨率和计时范围。'),
('5.6 Interpretation and limitations', '解释验证集参与权重选择、提示依赖真实标注和边界代理的限制。固定外扩会改变不同面积组的框相对松紧。少样本稳定性仅有短程试验支持。')], '主结果表采用重算的表面 HD95。完整的短程重复训练、自动提示和伪标签试验结果放附录。'),
('6 Conclusion and Future Work', '直接回答研究问题，总结实际完成的工作和证据允许的结论。', [
('6.1 Answers to the research questions', '用三个简短段落回答 RQ1–RQ3，分别概括零样本表现、适配后的变化，以及提示和图像条件下的发现。'),
('6.2 Contributions and scope', '总结复现与统一比较的贡献，简述结论适用的数据、模型与提示条件。详细局限见第 5.6 节。'),
('6.3 Future work', '讨论专家边界评级、正式长程复现和自动提示。摘要最后写，控制在 250–400 词。')], '')
]

sample = [
"This study evaluates endoscopic polyp segmentation with SAM-based models. RQ1 asks how zero-shot SAM and MedSAM perform on the Kvasir-SEG validation split and CVC-ClinicDB under a common box-prompt protocol. RQ2 asks how models adapted with LoRA or adapters on the full Kvasir-SEG training split compare with the zero-shot baselines on those datasets. RQ3 explores how performance varies with prompt type, box expansion and box translation, and across groups defined by relative polyp area and boundary contrast.",
"The data partition separates model fitting from validation. Kvasir-SEG provides 880 training images and 120 validation images, and the supplied filename lists have no overlap. Adaptation uses images from the training partition, including the smaller support sets used for the few-shot experiments. The validation partition is also used to select checkpoints during training. Its reported scores therefore describe performance on the model-selection validation set, rather than on an independent test set. CVC-ClinicDB, comprising 612 images, provides the cross-dataset evaluation set. The two evaluation sets are reported separately so that performance on the development dataset can be distinguished from performance on another dataset.",
"The first comparison establishes the zero-shot reference using SAM with ViT-B and ViT-L backbones and MedSAM. The second comparison evaluates polyp-specific adaptation using SAMed LoRA, MedSAM with LoRA, and Medical SAM Adapter with ViT-B and ViT-L backbones. PP-SAM is included as an existing full-parameter adaptation reference. These methods differ in their initial weights, trainable components, and training procedures. A shared evaluation partition and prompt-generation rule improve the interpretability of their results, but do not isolate the causal effect of any single architectural or optimisation choice. Implementation and training details are therefore reported for each method.",
"A supplementary analysis of RQ2 compares recorded SAMed and PP-SAM runs with 1, 5, 10, and 880 labelled training images. A separate short-budget SAMed 5-shot pilot examines variation across training seeds and support draws. The formal few-shot comparisons use the recorded checkpoints for each budget, while the pilot uses its own fixed training schedule.",
"The default prompt is a tight ground-truth box expanded outward by 30 pixels in the original image. Selected trained models were also evaluated with fixed expansions of 0, 10, 30, and 50 pixels. The point and box-translation experiments use SAM ViT-B and MSA-B with the same image-level prompts. Translations preserve box dimensions and are limited by the available space at the image edge. On CVC-ClinicDB, paired changes in Dice are also summarised by video sequence.",
"For the image-condition analysis, relative polyp area is measured as the fraction of image pixels covered by the ground-truth mask. A fixed pixel expansion changes the box size proportionally more for smaller boxes, which may contribute to differences between area groups. Boundary contrast is estimated from the colour difference between narrow regions inside and outside the annotated contour. A grayscale gradient measure provides a sensitivity check. Dice and intersection over union measure mask overlap, while boundary F1 and surface HD95 describe contour agreement and distance. Surface HD95 is reported in pixels at the original image resolution."
]

doc=Document(REF)
body=doc._element.body
sect=deepcopy(body.sectPr)
for child in list(body): body.remove(child)
body.append(sect)
# This review has one body section. The template's declarations and sample pages
# are intentionally omitted; no author or approval statement is fabricated.
section=doc.sections[0]
section.page_width=Cm(21);section.page_height=Cm(29.7)
section.top_margin=Cm(2.5);section.bottom_margin=Cm(2.5)
section.left_margin=Cm(2);section.right_margin=Cm(2)
# Use the written formatting requirements in the supplied template.
for tag in ['w:pgNumType']:
 el=section._sectPr.find(qn(tag))
 if el is not None: el.set(qn('w:fmt'),'decimal');el.set(qn('w:start'),'1')

for name,size in [('Normal',12),('Heading 1',16),('Heading 2',14),('Heading 3',12),('Caption',10)]:
 st=doc.styles[name]
 st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
 st.font.italic=False;st.font.bold=name.startswith('Heading')
 rpr=st.element.get_or_add_rPr()
 rf=rpr.find(qn('w:rFonts'))
 if rf is None:rf=OxmlElement('w:rFonts');rpr.insert(0,rf)
 rf.set(qn('w:eastAsia'),'宋体')
 pf=st.paragraph_format
 pf.space_before=Pt(0 if name=='Normal' else 10);pf.space_after=Pt(6)
 pf.line_spacing=1.15
 pf.first_line_indent=Pt(0)
 pf.keep_with_next=name.startswith('Heading')
 pf.widow_control=True
 # Suppress inherited template list numbering; headings below include numbers.
 ppr=st.element.get_or_add_pPr()
 old=ppr.find(qn('w:numPr'))
 if old is not None:ppr.remove(old)
 if name.startswith('Heading'):
  num=OxmlElement('w:numPr');nid=OxmlElement('w:numId');nid.set(qn('w:val'),'0');num.append(nid);ppr.append(num)
doc.styles['Normal'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
if 'Title' not in doc.styles:
 st=doc.styles.add_style('Title',WD_STYLE_TYPE.PARAGRAPH);st.base_style=doc.styles['Normal']
st=doc.styles['Title'];st.font.name='Times New Roman';st.font.size=Pt(20);st.font.bold=True;st.font.color.rgb=RGBColor(0,0,0)
st.paragraph_format.space_after=Pt(12)

next_page=False
def p(text,style=None):
 global next_page
 para=doc.add_paragraph(text,style=style)
 if next_page:
  para.paragraph_format.page_break_before=True
  next_page=False
 return para
def h(text,level=1):
 return p(text,f'Heading {level}')
def item(label,text):
 para=p('');r=para.add_run(label);r.bold=True
 if text:para.add_run('  '+text)
 else:para.paragraph_format.keep_with_next=True
def mapping_table():
 table=doc.add_table(rows=0, cols=2)
 table.autofit=False
 widths=(Cm(2.3),Cm(14.0))
 for column,width in zip(table.columns,widths):column.width=width
 for i,row in enumerate([('研究问题','对应实验'),*study_mapping]):
  for cell,width,value in zip(table.add_row().cells,widths,row):
   cell.width=width
   para=cell.paragraphs[0]
   para.alignment=WD_ALIGN_PARAGRAPH.LEFT
   para.paragraph_format.space_after=Pt(3)
   para.paragraph_format.space_before=Pt(3)
   para.paragraph_format.keep_with_next=i<len(study_mapping)
   run=para.add_run(value)
   run.bold=i==0
   run.font.size=Pt(11)
   if i==0:
    shade=OxmlElement('w:shd')
    shade.set(qn('w:fill'),'E7EDF3')
    cell._tc.get_or_add_tcPr().append(shade)
 borders=OxmlElement('w:tblBorders')
 for side in ('top','bottom','left','right','insideH','insideV'):
  border=OxmlElement('w:'+side)
  border.set(qn('w:val'),'single')
  border.set(qn('w:sz'),'4')
  border.set(qn('w:color'),'D9D9D9')
  borders.append(border)
 table._tbl.tblPr.append(borders)
def page():
 global next_page
 next_page=True

p('FYP Thesis Writing Plan and Study Design','Title')
p('第二轮审阅稿  |  2026年11月18日')
p('本稿供讨论论文结构和英文表达使用。前半部分是中文写作提纲，后半部分为可纳入论文的英文3.1节草稿。全文尚未完成，作者和导师审阅、引用核验及实现细节审计仍待进行。')
h('研究叙述与范围')
p('课题名称：Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation。')
p('论文主线是比较零样本模型和息肉领域适配模型。实验采用共同的数据划分与默认框提示规则，并保留各模型的训练和推理设置。')
h('三项研究问题',2)
for label,question in research_questions:
 item(label,question)
h('各问题对应的实验',2)
p('RQ1 和 RQ2 使用新版主结果表。SAMed 与 PP-SAM 的 1、5、10 和 880 张标注数据实验作为 RQ2 的补充。PP-SAM 为已有的全参适配对照。')
p('点提示和框位移比较 SAM-B 与 MSA-B。相对面积以真实掩膜占图像的比例表示，边界分析使用颜色对比度和灰度梯度作为探索性指标。')

for k in range(0,6,2):
 page()
 for title,purpose,subs,display in chapters[k:k+2]:
  if title.startswith('4 '):page()
  h(title)
  p('章节目的：'+purpose)
  for label,text in subs:
   item(label,text)
   if label=='3.1 Study design':mapping_table()
  if display:
   para=p(display);para.paragraph_format.space_before=Pt(3)

page()
h('3 Research Methodology')
h('3.1 Study design',2)
for i,text in enumerate(sample):
 if i==4:page()
 p(text)

page()
h('写作核验与后续工作')
p('本页为审阅记录，不纳入最终论文章节。样章依据本地项目材料撰写，尚未登记为作者已核验的最终文本。样章没有引用外部研究结论；正式介绍各方法时将补入经原文核验的引用。')
h('本轮已核对',2)
item('数据划分','训练列表880项、验证列表120项，集合交集为0；正式主线summary文件中的数据集数量与上述评估数量相符。')
item('实验定位','主线模型、少样本设置和框外扩范围依据正式数字索引、消融记录及结果文件整理。Kvasir验证参与权重选择的表述有训练代码和记录支持。')
item('证据来源','官方范围见课题说明。提示、尺度和边界的新增实验以分析总记录为准。详细对应关系保存在本稿的来源清单中。')
h('后续写作前核实',2)
item('训练与推理细节','逐模型核实初始化、可训练参数、损失、支持集清单、训练配置、候选掩膜选择及阈值。当前代码可能含后续改动，需与正式运行记录对应。')
item('指标与比较条件','主表采用重算的原图像素表面 HD95。FPS 按统一测速记录单列，覆盖 SAM-B、SAM-L 和 MSA-B。')
item('图表与引用','从正式结果生成曲线；定性图使用对应权重预测。核验现有34条BibTeX及其正文论点，避免仅凭搜索摘要确认引用。')
item('特殊来源','区分SAMed主表与本机消融的box30结果；MedSAM+LoRA的CVC box0/10/50为用户提供数值，没有本地原始JSON。')
h('建议审阅重点',2)
p('本轮已按确认的三个问题同步修改首页、第 1、3、5、6 章提纲和 3.1 英文样章。后续核实章节中的具体实现和模型来源。')

settings=doc.settings.element
up=settings.find(qn('w:updateFields'))
if up is None:up=OxmlElement('w:updateFields');settings.append(up)
up.set(qn('w:val'),'true')
buffer=BytesIO();doc.save(buffer)
changed={'word/document.xml','word/styles.xml','word/settings.xml'}
with zipfile.ZipFile(REF) as src,zipfile.ZipFile(buffer) as generated,zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED) as dst:
 for info in src.infolist():
  dst.writestr(info,generated.read(info.filename) if info.filename in changed else src.read(info.filename))
with zipfile.ZipFile(REF) as src,zipfile.ZipFile(DEST) as dst:
 assert all(src.read(n)==dst.read(n) for n in src.namelist() if n not in changed)
assert sha(REF.read_bytes())==ref_hash

sources=[
 ('E01','charter/PROJECT_BRIEF.md','Project scope and experiment status; documentary source'),
 ('E02','charter/SCOPE.md','Writing boundaries and incomplete analyses'),
 ('E03','numbers/NUMBERS_INDEX.md','Formal runs and source caveats'),
 ('E04','splits/Kvasir-SEG/Data-split/train.txt','880 training IDs; checked programmatically'),
 ('E05','splits/Kvasir-SEG/Data-split/val.txt','120 validation IDs; checked programmatically'),
 ('E06','numbers/prompt_ablation_box_expand_SUMMARY.md','Fixed outward expansions and source differences'),
 ('E07','numbers/notes/medsam_peft_EVAL_R4_FULL_BEST.md','Best validation checkpoint and evaluation records'),
 ('E08','numbers/notes/msa_polyp_zeroshot_or_trained_v3.md','Trained adapter, best checkpoint and evaluation'),
 ('E09','E:/Project/FYP/code/samed_baseline/train_samed.py','Validation and best-checkpoint selection; current code, run-version reconciliation pending'),
 ('E10','E:/Project/FYP/code/medical_sam_adapter/eval_polyp_v3.py','Evaluation split and prompt generation; current code'),
 ('E11','templet/[GroupID] FYP Thesis.docx','University structure and written formatting requirements'),
 ('E12','analysis/REMAINING_EXPERIMENTS_RESULTS.md','MSA-B prompt comparison, sequence analysis, training pilot and new metric table'),
 ('E13','analysis/PROTOCOL.md','Relative area and boundary proxy definitions and metric protocol'),
 ('E14','manuscript/tables/table_main_boxprompt_surface_hd95.csv','Reconciled primary results with surface metrics'),
]
records=[]
for id,path,purpose in sources:
 f=Path(path) if Path(path).is_absolute() else ROOT/path
 records.append(dict(id=id,path=str(f),sha256=sha(f.read_bytes()),purpose=purpose,human_verification='pending'))
train=(ROOT/sources[3][1]).read_text().splitlines();val=(ROOT/sources[4][1]).read_text().splitlines()
assert len(train)==880 and len(val)==120 and not set(train)&set(val)
result_files=sorted((ROOT/'numbers/official_core').rglob('summary.json'))
for f in result_files:
 d=json.loads(f.read_text(encoding='utf-8-sig'))
 for name,v in d.items():
  if isinstance(v,dict) and 'n' in v:assert v['n']==(120 if 'Kvasir' in name else 612)
ledger=dict(document=str(DEST),status='draft_for_author_review',date='2026-11-18',sources=records,
 sample_paragraph_sources={'1':['E01','E03','E12','E13'],'2':['E04','E05','E07','E08','E09'],'3':['E01','E03'],'4':['E03','E09','E12'],'5':['E06','E10','E12'],'6':['E13','E14']},
 automated_checks={'split_train':880,'split_val':120,'split_overlap':0,'core_summary_files':len(result_files),'template_unchanged':True,'preserve_only_package_parts_unchanged':True},
 unresolved=['Author and supervisor review','Run-specific implementation reconciliation','Original-paper citation verification','Complete metric audit','Final schematic and qualitative figure creation'])
(OUT/f'{DEST.stem}_sources.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
md=['# FYP Thesis Writing Plan and Study Design','Draft for author review; 2026-11-18.']
md+=['## Research questions']+[f'**{label}** {question}' for label,question in research_questions]
for title,purpose,subs,display in chapters:
 md+=['\n## '+title,purpose]
 for label,description in subs:
  md+=['\n### '+label+'\n'+description]
  if label=='3.1 Study design':
   md+=['\n'.join(['| 研究问题 | 对应实验 |','|---|---|']+
                 [f'| {rq} | {experiment} |' for rq,experiment in study_mapping])]
 md+=[display]
md+=['\n## 3.1 Study design']+sample
(WORK/f'{DEST.stem}_content.md').write_text('\n\n'.join(md),encoding='utf-8')
artifact=f'''# Template execution contract

Reference: {REF}
SHA256: {ref_hash}
Reference render: manuscript/working/qa/template/reference.pdf (12 pages, Microsoft Word).
Reference has two A4 sections, a cover, abstract, contents, declarations, acknowledgement,
chapter examples, references and formatting appendix. Rendered all pages to PNG contact sheet.
Actual margins around 3.17 cm horizontally and 2.54 cm vertically differ from written rules.
Actual Normal is 10 pt, Heading 1 Arial 14 pt; written rules specify Times New Roman
12 pt body, 16/14 pt headings, 2 cm left/right and 2.5 cm top/bottom margins.

This review uses the written template rules for font sizes and margins, A4 portrait,
black headings, justified body and retained footer parts. Chinese planning text uses SimSun.
Source style IDs Normal, Heading1, Heading2, Heading3 and Caption are reused.
Body rhythm: 1.15 lines, 6 pt after, no first-line indent, widow control.
Explicit breaks separate overview, chapter outlines, sample, and review notes.
The review is intentionally not a full thesis: cover placeholders, declarations, sample
references, figure, TOC and formatting appendix are removed from body content.
Only the final body section is reused; section count becomes one intentionally.
No author, approval, signature, or publication-ready status is inferred.

Editable parts: word/document.xml (all body slots replaced with review content; last sectPr
retained with documented geometry changes and decimal pagination), word/styles.xml (selected
paragraph styles), word/settings.xml (updateFields). Preserve-only: every other OPC part,
including all relationships, media, numbering, custom XML, headers/footers and metadata.
Builder verifies byte equality for every preserve-only part and unchanged reference hash.
The output keeps unused original package assets rather than deleting opaque components.
No TOC or cross-reference cache is introduced; inherited PAGE fields update on Word export.
QA uses native Microsoft Word PDF export, then Poppler PNGs, because packaged render_docx.py
was attempted and failed: soffice.exe is absent. The native Word export succeeded.
Final page and structural audit are recorded in qa/review_02/layout_audit.json.
'''
(WORK/f'{DEST.stem}_template.md').write_text(artifact,encoding='utf-8')
print(json.dumps({'docx':str(DEST),'sample_words':sum(len(t.split()) for t in sample),'core_summaries_checked':len(result_files),'template_sha256':ref_hash},indent=2))
