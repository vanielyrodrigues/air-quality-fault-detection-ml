"""Diagrama do fluxo retrospectivo implementado."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def gerar():
    pasta = Path(__file__).resolve().parents[1]/'figuras'
    pasta.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    boxes = {
        'entrada': (.5, .94, .82, .075, 'Entrada: leituras com data e hora'),
        'grade': (.5, .83, .82, .075, 'Organização na grade de dez minutos'),
        'lacuna': (.18, .63, .30, .105, 'Lacuna\nGrade sem interpolação'),
        'preparo': (.68, .72, .55, .075, 'Preparação da série\nRegistro das posições ausentes'),
        'persist': (.52, .57, .28, .105, 'Regras de persistência\nStuck e stuck-at-zero'),
        'ml': (.84, .57, .28, .105, 'Janelas e classificadores\nOscilação e Queda'),
        'consolida': (.5, .40, .90, .12, 'Consolidação das sinalizações\nUnião dos intervalos positivos por categoria\nPreservação da origem e das sobreposições'),
        'saida': (.5, .18, .90, .18, 'Saída por instante e intervalo\nSem sinalização / Sinalização de uma categoria\nSinalização de múltiplas categorias\nDados insuficientes / Leitura ausente abaixo do limiar')}
    for x, y, w, h, texto in boxes.values():
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.006', edgecolor='#234967', facecolor='#f2f6f9', linewidth=1.4))
        ax.text(x, y, texto, ha='center', va='center', fontsize=11, linespacing=1.6)
    edges = [('entrada', 'grade'), ('grade', 'lacuna'), ('grade', 'preparo'), ('preparo', 'persist'), ('preparo', 'ml'), ('lacuna', 'consolida'), ('persist', 'consolida'), ('ml', 'consolida'), ('consolida', 'saida')]
    for a, b in edges:
        x, y, w, h, _ = boxes[a]; xx, yy, ww, hh, _ = boxes[b]
        if a == 'grade':
            x = xx
        ax.add_patch(FancyArrowPatch((x, y-h/2-.008), (xx, yy+hh/2+.008), arrowstyle='-|>', mutation_scale=14, color='#234967', linewidth=1.2))
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(pasta/f'fluxo_integrado.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    gerar()
