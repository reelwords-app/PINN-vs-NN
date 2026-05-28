"""Generate editable PINN architecture PowerPoint slide."""

import pptx
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_CONNECTOR_TYPE
import pptx.oxml.ns as nsmap

# ── Slide setup ────────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(16)
prs.slide_height = Inches(9)

slide  = prs.slides.add_slide(prs.slide_layouts[6])  # blank
shapes = slide.shapes

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = RGBColor(0xF8, 0xF9, 0xFA)   # near-white background
BLUE    = RGBColor(0x1F, 0x77, 0xB4)   # input nodes
TEAL    = RGBColor(0x17, 0xBE, 0xCF)   # embedding blocks
PURPLE  = RGBColor(0x6A, 0x5A, 0xCD)   # hidden layers
GREEN   = RGBColor(0x2C, 0xA0, 0x2C)   # output / physics
ORANGE  = RGBColor(0xD6, 0x27, 0x28)   # loss / IC/BC
GREY    = RGBColor(0xAA, 0xAA, 0xAA)   # divider / arrows
DARK    = RGBColor(0x22, 0x22, 0x22)   # text
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)

def rgb(c): return c

# ── Helper: add rectangle with centred text ────────────────────────────────────
def box(left, top, width, height,
        text='', fill=BLUE, text_color=WHITE,
        font_size=11, bold=False, line_color=None, line_width=Pt(1),
        wrap=True, valign='middle', halign=PP_ALIGN.CENTER):
    shape = shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE if False else 1,  # MSO_AUTO_SHAPE_TYPE.RECTANGLE = 1
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()   # no border

    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.auto_size = None
    # vertical alignment
    from pptx.enum.text import MSO_ANCHOR
    tf.vertical_anchor = {'middle': MSO_ANCHOR.MIDDLE,
                           'top':    MSO_ANCHOR.TOP,
                           'bottom': MSO_ANCHOR.BOTTOM}[valign]

    p = tf.paragraphs[0]
    p.alignment = halign
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = text_color
    run.font.name = 'Calibri'
    return shape

def label(left, top, text, font_size=10, color=DARK, bold=False, halign=PP_ALIGN.CENTER):
    """Transparent text label."""
    txb = shapes.add_textbox(Inches(left), Inches(top), Inches(3), Inches(0.35))
    tf  = txb.text_frame
    tf.word_wrap = False
    p   = tf.paragraphs[0]
    p.alignment = halign
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.name  = 'Calibri'
    return txb

def wide_label(left, top, width, text, font_size=10, color=DARK, bold=False,
               halign=PP_ALIGN.LEFT, wrap=True):
    txb = shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1.5))
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = halign
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.name  = 'Calibri'
    return txb

def arrow(x1, y1, x2, y2, color=GREY, width=Pt(1.5)):
    """Simple connector line (pptx connector)."""
    from pptx.util import Emu
    connector = shapes.add_connector(
        MSO_CONNECTOR_TYPE.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    connector.line.color.rgb = color
    connector.line.width = width
    return connector

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
bg = shapes.add_shape(1, Inches(0), Inches(0), Inches(16), Inches(9))
bg.fill.solid(); bg.fill.fore_color.rgb = BG
bg.line.fill.background()

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
box(0.2, 0.1, 15.6, 0.55,
    text='PINN Surrogate Architecture — 1D Polymer Flood (Pelican Lake HP-6)',
    fill=DARK, font_size=16, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL  — Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════
# Panel header
box(0.2, 0.75, 7.6, 0.4, text='Neural-Network Architecture',
    fill=PURPLE, font_size=13, bold=True)

# ── INPUT NODES ───────────────────────────────────────────────────────────────
Y_inp = 1.35
box(0.3,  Y_inp, 1.5, 0.45, text='X  (norm)', fill=BLUE, font_size=11, bold=True)
box(1.95, Y_inp, 1.5, 0.45, text='T  (norm)', fill=BLUE, font_size=11, bold=True)
box(3.6,  Y_inp, 1.8, 0.45, text='C_pi  (inj. conc.)', fill=BLUE, font_size=11, bold=True)
label(0.25, 1.82, 'Inputs', font_size=9, color=GREY)

# ── EMBEDDING LAYERS ──────────────────────────────────────────────────────────
Y_emb = 2.2
box(0.3,  Y_emb, 1.5, 0.55, text='Dense 64\n+ Tanh', fill=TEAL, font_size=10)
box(1.95, Y_emb, 1.5, 0.55, text='Dense 64\n+ Tanh', fill=TEAL, font_size=10)
label(0.25, 2.77, 'Space & Time Embeddings', font_size=9, color=GREY)

# Arrows: inputs → embeddings
arrow(1.05, Y_inp+0.45, 1.05, Y_emb)
arrow(2.70, Y_inp+0.45, 2.70, Y_emb)

# ── CONCAT ────────────────────────────────────────────────────────────────────
Y_cat = 3.05
box(0.3, Y_cat, 3.1, 0.45, text='Concatenate  [64 + 64 + 1] = 129 units',
    fill=RGBColor(0x80,0x80,0x80), font_size=10)
# arrows: embeddings → concat
arrow(1.05, Y_emb+0.55, 1.05, Y_cat)
arrow(2.70, Y_emb+0.55, 2.70, Y_cat)
# Cpi direct to concat
arrow(4.50, Y_inp+0.45, 4.50, 3.27)   # Cpi goes down
arrow(4.50, 3.27,       3.40, 3.27)   # Cpi goes left to concat

# ── HIDDEN LAYERS ─────────────────────────────────────────────────────────────
Y_h = [3.7, 4.55, 5.4]
labels_h = ['Dense 128 → BN → Tanh → Dropout(0.1)',
            'Dense 128 → BN → Tanh → Dropout(0.1)',
            'Dense  64 → BN → Tanh → Dropout(0.1)']
for i, (y, lbl) in enumerate(zip(Y_h, labels_h)):
    box(0.3, y, 3.1, 0.55, text=lbl, fill=PURPLE, font_size=10)
    if i == 0:
        arrow(1.85, Y_cat+0.45, 1.85, y)
    else:
        arrow(1.85, Y_h[i-1]+0.55, 1.85, y)

label(0.25, 6.0, 'Hidden Layers', font_size=9, color=GREY)

# ── OUTPUT LAYER ──────────────────────────────────────────────────────────────
Y_out = 6.15
box(0.3, Y_out, 3.1, 0.55,
    text='Dense 2 → Sigmoid\n[Ŝw_norm ,  Ĉp_norm]  ∈ (0,1)',
    fill=GREEN, font_size=10)
arrow(1.85, Y_h[-1]+0.55, 1.85, Y_out)
label(0.25, 6.73, 'Output (normalised)', font_size=9, color=GREY)

# ── DENORMALISE BLOCK ─────────────────────────────────────────────────────────
Y_den = 7.0
box(0.3, Y_den, 3.1, 0.7,
    text='Denormalise\nSw = Swi + Ŝw·(1−Swi−Sor)\nCp = Ĉp · Cp,max',
    fill=RGBColor(0x4C,0x72,0xB0), font_size=9)
arrow(1.85, Y_out+0.55, 1.85, Y_den)

# ── VERTICAL DIVIDER ──────────────────────────────────────────────────────────
arrow(8.05, 0.75, 8.05, 8.85, color=GREY, width=Pt(1))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Physics & Loss
# ══════════════════════════════════════════════════════════════════════════════
box(8.15, 0.75, 7.65, 0.4, text='Physics Constraints & Loss Function',
    fill=ORANGE, font_size=13, bold=True)

# ── PHYSICAL TRANSFORMATION ───────────────────────────────────────────────────
box(8.15, 1.25, 7.65, 0.35, text='Physical Transformation  (from Sw, Cp)',
    fill=RGBColor(0x55,0x55,0x55), font_size=11, bold=True)

phys_text = (
    "Se  =  (Sw − Swc) / (1 − Swc − Sor)\n"
    "krw =  0.10 · Se^nw          [nw trainable]\n"
    "kro =  1.00 · (1−Se)^no      [no trainable]\n"
    "μw  =  μwi · (1 + r·Cp + s·Cp² + t·Cp³)     [r,s,t trainable]\n"
    "μo  =  1650 cp               [fixed]\n"
    "fw  =  (krw/μw) / (krw/μw + kro/μo)\n"
    "vp  =  Q·Tref / (A·φ·L)     [fixed]"
)
box(8.15, 1.65, 7.65, 2.0, text=phys_text,
    fill=RGBColor(0xFF,0xFF,0xE8), text_color=DARK, font_size=10,
    line_color=ORANGE, line_width=Pt(1), valign='top', halign=PP_ALIGN.LEFT)

# ── PDE RESIDUALS ─────────────────────────────────────────────────────────────
box(8.15, 3.75, 7.65, 0.35, text='PDE Residuals  (collocation points)',
    fill=RGBColor(0x55,0x55,0x55), font_size=11, bold=True)

pde_text = (
    "R_BL  =  ∂Sw/∂T  +  vp · ∂fw/∂X  =  0\n\n"
    "R_Cp  =  ∂(Sw·Cp)/∂T  +  vp · ∂(fw·Cp)/∂X  =  0"
)
box(8.15, 4.15, 7.65, 1.1, text=pde_text,
    fill=RGBColor(0xFF,0xFF,0xE8), text_color=DARK, font_size=11,
    line_color=ORANGE, line_width=Pt(1), valign='middle', halign=PP_ALIGN.LEFT)

# ── IC / BC ───────────────────────────────────────────────────────────────────
box(8.15, 5.35, 7.65, 0.35, text='Initial & Boundary Conditions',
    fill=RGBColor(0x55,0x55,0x55), font_size=11, bold=True)

icbc_text = (
    "IC  (T=0, all X):   Sw = Sw,init = 0.36 ,  Cp = 0\n"
    "BC  (X=0, all T):   Sw = 1−Sor    ,  Cp = Cp,inj(T)"
)
box(8.15, 5.75, 7.65, 0.8, text=icbc_text,
    fill=RGBColor(0xFF,0xFF,0xE8), text_color=DARK, font_size=10,
    line_color=ORANGE, line_width=Pt(1), valign='middle', halign=PP_ALIGN.LEFT)

# ── DATA LOSS ─────────────────────────────────────────────────────────────────
box(8.15, 6.65, 7.65, 0.35, text='Data Loss  (producer well observations)',
    fill=RGBColor(0x55,0x55,0x55), font_size=11, bold=True)

data_text = (
    "L_data  =  MSE(WC_pred , WC_obs)  +  MSE(q_o,pred , q_o,obs)\n\n"
    "where  WC = fw(Sw, Cp) ,   q_o = Q·(1−WC)"
)
box(8.15, 7.05, 7.65, 1.0, text=data_text,
    fill=RGBColor(0xFF,0xFF,0xE8), text_color=DARK, font_size=10,
    line_color=ORANGE, line_width=Pt(1), valign='middle', halign=PP_ALIGN.LEFT)

# ── TOTAL LOSS ────────────────────────────────────────────────────────────────
box(8.15, 8.15, 7.65, 0.55,
    text='L_total  =  w₁·L_BL + w₂·L_Cp + w₃·L_data + w₄·L_IC + w₅·L_BC',
    fill=ORANGE, font_size=11, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINABLE PARAMS LEGEND (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
box(0.2, 8.15, 7.6, 0.55,
    text='Trainable physics params:  nw, no, r, s, t  (polymer viscosity)  |  '
         'Fixed:  krw=0.10, kro=1.00, vp, μo=1650 cp',
    fill=RGBColor(0x33,0x33,0x33), font_size=10)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out = '/home/user/PINN-vs-NN/PINN_Architecture.pptx'
prs.save(out)
print(f'Saved: {out}')
