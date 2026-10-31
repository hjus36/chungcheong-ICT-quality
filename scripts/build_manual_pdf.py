"""Optional report generation; needs reportlab and a Korean TrueType font."""
from pathlib import Path
import os
import re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
fontdir=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
pdfmetrics.registerFont(TTFont('Korean',str(fontdir/'malgun.ttf')))
pdfmetrics.registerFont(TTFont('KoreanBold',str(fontdir/'malgunbd.ttf')))
pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='KoreanBold')
INK=colors.HexColor('#213d4c'); TEAL=colors.HexColor('#107f73')
body=ParagraphStyle('Body',fontName='Korean',fontSize=10,leading=17,wordWrap='CJK',textColor=INK,spaceAfter=10)
title=ParagraphStyle('Title',parent=body,fontName='KoreanBold',fontSize=24,leading=32,spaceAfter=22)
tabletext=ParagraphStyle('Table',parent=body,fontSize=9,leading=14,spaceAfter=0)
code=ParagraphStyle('Code',parent=body,fontSize=9,leading=16,backColor=colors.HexColor('#eff5f3'),borderPadding=10,spaceAfter=16,spaceBefore=8)

def inline(s):
    s=escape(s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`(.*?)`',r'<font color="#107f73">\1</font>',s)
    return s

def footer(canvas,doc):
    w,h=A4
    canvas.setStrokeColor(colors.HexColor('#dbe5e8'));canvas.line(43,45,w-43,45)
    canvas.setFont('Korean',8);canvas.setFillColor(colors.HexColor('#8498a2'))
    canvas.drawString(43,30,'무한루프 · 충청권 ICT 현장적용 프로젝트 · 2026-11-01')
    canvas.drawRightString(w-43,30,f'{doc.page} / 5')
    canvas.setFillColor(TEAL);canvas.rect(43,h-37,24,3,fill=1,stroke=0)
    canvas.setFont('Korean',8);canvas.drawString(75,h-38,'제조공정 품질 분석 MVP  |  사용자 매뉴얼 v1.0')

def main():
    source=(ROOT/'docs/08_사용자_설명서.md').read_text(encoding='utf-8')
    sections=source.split('\n## ')[1:]
    assert len(sections)==5
    story=[]
    for index,section in enumerate(sections):
        heading,*lines=section.strip().splitlines()
        story.append(Paragraph(f'{index+1:02d} / 사용 가이드',ParagraphStyle('Index',parent=body,fontSize=9,textColor=TEAL,spaceAfter=12)))
        story.append(Paragraph(heading.split('. ',1)[-1],title))
        i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line: i+=1;continue
            if line.startswith('```'):
                block=[];i+=1
                while i<len(lines) and not lines[i].strip().startswith('```'): block.append(lines[i]);i+=1
                story.append(Paragraph('<br/>'.join(escape(s) for s in block),code));i+=1;continue
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    line=lines[i].strip()
                    if not re.match(r'^\|[\s:|\-]+$',line): rows.append([Paragraph(inline(c.strip()),tabletext) for c in line.strip('|').split('|')])
                    i+=1
                widths=[125,384] if len(rows[0])==2 else None
                t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f2ef')),('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),
                    ('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dfe7ea'))]))
                story.extend([t,Spacer(1,15)]);continue
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('|','```')):
                paragraph.append(lines[i].strip());i+=1
            story.append(Paragraph(inline(' '.join(paragraph)),body))
        if index<4: story.append(PageBreak())
    out=ROOT/'output/pdf/무한루프_MVP_사용자매뉴얼.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    document=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=63,bottomMargin=61,
        title='무한루프 MVP 사용자 매뉴얼',author='무한루프팀')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__=='__main__': main()
