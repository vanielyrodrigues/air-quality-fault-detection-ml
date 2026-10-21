"""Relatórios por janela e cobertura retrospectiva de eventos simulados."""
import os
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
NAMES = {'random_forest': 'Random Forest', 'mlp': 'MLP', 'xgboost': 'XGBoost', 'catboost': 'CatBoost'}
FAULT = {'oscilacao': 'Oscilação', 'queda': 'Queda'}

def overlap(a, b, c, d):
    return max(a, c) <= min(b, d)

def assess(root, preds):
    counts = []
    detail = []
    for (model, fault, r), w in preds[preds.cenario.eq('com_injecao')].groupby(['modelo', 'falha', 'rodada']):
        w = w.sort_values('Datetime_ini').copy()
        ev = pd.read_csv(root / 'bases' / f'rodada_{r}' / 'teste_eventos_injetados.csv', parse_dates=['inicio', 'fim'])
        ev = ev[ev.falha.eq(fault)].sort_values('inicio')
        positive = w[w.predicao.eq(1)]
        reached = 0
        covered = 0
        for e in ev.itertuples():
            candidates = [p for p in positive.itertuples() if overlap(p.Datetime_ini, p.Datetime_fim, e.inicio, e.fim)]
            evaluated = any((overlap(p.Datetime_ini, p.Datetime_fim, e.inicio, e.fim) for p in w.itertuples()))
            reached += bool(candidates)
            covered += evaluated
            detail.append(dict(modelo=model, falha=fault, rodada=r, evento_id=e.evento_id, inicio=e.inicio, fim=e.fim, alguma_janela_positiva=bool(candidates), janelas_positivas_sobrepostas=len(candidates), avaliado=evaluated))
        groups = []
        for k, p in enumerate(w.itertuples()):
            if p.predicao:
                if k == 0 or not w.iloc[k - 1].predicao:
                    groups.append([])
                groups[-1].append(p)
        ext = sum((not any((overlap(p.Datetime_ini, p.Datetime_fim, e.inicio, e.fim) for p in g for e in ev.itertuples())) for g in groups))
        counts.append(dict(modelo=model, falha=fault, rodada=r, injetadas=len(ev), com_sinalizacao=reached, sem_sinalizacao=len(ev) - reached, sem_janela_avaliada=len(ev) - covered, grupos_sinalizados=len(groups), alertas_externos=ext))
        assert reached <= len(ev) and covered == len(ev)
    return (pd.DataFrame(counts), pd.DataFrame(detail))

def export_table(frame, path, caption, label, columns):
    header = ' & '.join(columns.values()) + ' \\\\'
    lines = ['\\begin{table}[H]', '\\centering', '\\footnotesize', '\\caption{' + caption + '}', '\\label{' + label + '}', '\\begin{tabular}{l' + 'r' * (len(columns) - 1) + '}', '\\toprule', header, '\\midrule']
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col == 'modelo':
                val = NAMES[val]
            elif isinstance(val, (float, np.floating)):
                val = f'{val:.3f}'.replace('.', ',')
            else:
                val = str(val)
            if col == 'f1_teste' and row[col] == frame[col].max():
                val = '\\textbf{' + val + '}'
            vals.append(val)
        lines.append(' & '.join(vals) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\fonte{Elaborada pela autora.}', '\\end{table}']
    path.write_text('\n'.join(lines) + '\n')

def run(root, col, var):
    root = Path(root)
    assert (root / 'CONCLUIDO.json').exists()
    from auditar_resultados import run as auditar
    auditar(root)
    vl = {'PM25': 'PM$_{2,5}$', 'PM10': 'PM$_{10}$'}.get(var, var)
    out = root / 'relatorio'
    out.mkdir(exist_ok=True)
    chosen = pd.read_csv(root / 'configuracoes_congeladas.csv')
    val = pd.read_csv(root / 'validacao_todas_configuracoes.csv')
    final = pd.read_csv(root / 'resultados_finais.csv')
    preds = pd.concat([pd.read_csv(p, parse_dates=['Datetime_ini', 'Datetime_fim']) for p in sorted((root / 'predicoes').glob('*.csv'))])
    preds.to_csv(out / 'predicoes_todas.csv', index=False)
    counts, detail = assess(root, preds)
    counts.to_csv(out / 'cobertura_eventos.csv', index=False)
    detail.to_csv(out / 'eventos_individuais.csv', index=False)
    clean = final[final.cenario.eq('com_injecao')]
    avg = clean.groupby(['modelo', 'falha'], as_index=False).agg(f1_teste=('f1', 'mean'), dp_f1_teste=('f1', 'std'), precisao=('precision', 'mean'), recall=('recall', 'mean'), acuracia_balanceada=('balanced_accuracy', 'mean'))
    summary = chosen.merge(avg, on=['modelo', 'falha'])
    summary.to_csv(out / 'sintese_modelos.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for f in FAULT:
        vv = chosen[chosen.falha.eq(f)]
        export_table(vv, out / f'selecao_{f}.tex', f'Configurações fixas selecionadas pelo F1 médio da validação para {FAULT[f]} em {vl}.', f'tab:{var.lower()}_selecao_{f}', {'modelo': 'Modelo', 'janela': 'Janela', 'avanco': 'Avanço', 'f1_medio': 'F1 médio val.', 'dp_f1': 'DP F1 val.'})
        ss = summary[summary.falha.eq(f)]
        export_table(ss, out / f'teste_{f}.tex', f'Desempenho das configurações fixas no teste com injeção para {FAULT[f]} em {vl}.', f'tab:{var.lower()}_teste_{f}', {'modelo': 'Modelo', 'f1_teste': 'F1 médio', 'dp_f1_teste': 'DP F1', 'precisao': 'Precisão', 'recall': 'Recall'})
        cc = counts[counts.falha.eq(f)]
        export_table(cc, out / f'eventos_{f}.tex', f'Cobertura retrospectiva dos eventos de {FAULT[f]} em {vl}: sobreposição com alguma janela positiva.', f'tab:{var.lower()}_eventos_{f}', {'modelo': 'Modelo', 'rodada': 'Rodada', 'injetadas': 'Injetadas', 'com_sinalizacao': 'Com sinal.', 'sem_sinalizacao': 'Sem sinal.', 'alertas_externos': 'Alertas externos'})
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for ax, m in zip(axs.flat, NAMES):
            q = val[val.modelo.eq(m) & val.falha.eq(f) & val.elegivel.eq(True)]
            grid = q.pivot(index='janela', columns='avanco', values='f1_medio').reindex(index=range(20, 101, 10), columns=range(5, 101, 5))
            im = ax.imshow(grid.to_numpy(), origin='lower', aspect='auto', vmin=0, vmax=1, cmap='Blues', extent=[2.5, 102.5, 15, 105])
            c = vv[vv.modelo.eq(m)].iloc[0]
            ax.scatter([c.avanco], [c.janela], marker='*', s=110, c='#b22222', label='Configuração selecionada')
            ax.set_title(NAMES[m])
            ax.set_xlabel('Avanço (leituras)')
            ax.set_ylabel('Janela (leituras)')
            ax.set_xticks([5, 25, 50, 75, 100])
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.legend(fontsize=8, loc='upper left')
        fig.colorbar(im, ax=axs.ravel().tolist(), label='F1 médio na validação', shrink=0.8)
        fig.savefig(out / f'validacao_{f}.pdf', bbox_inches='tight')
        plt.close(fig)
        fig, axs = plt.subplots(5, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1]})
        base = pd.read_csv(root / 'bases/rodada_1/teste.csv', parse_dates=['Datetime'])
        axs[0].plot(base.Datetime, base[col], color='#666666', lw=0.8, label='Leituras com injeção')
        mask = base.label.eq(f)
        axs[0].scatter(base.loc[mask, 'Datetime'], base.loc[mask, col], color='#e69f00', s=9, label='Pontos da falha-alvo injetada')
        axs[0].set_xlim(base.Datetime.min(), base.Datetime.max())
        axs[0].set_ylabel({'PM25': 'PM₂,₅ (µg/m³)', 'PM10': 'PM₁₀ (µg/m³)', 'Temperatura': 'Temperatura (°C)', 'Umidade': 'Umidade (%)'}[var])
        axs[0].legend(fontsize=8)
        for ax, m in zip(axs[1:], NAMES):
            d = preds[preds.modelo.eq(m) & preds.falha.eq(f) & preds.rodada.eq(1) & preds.cenario.eq('com_injecao')].sort_values('Datetime_fim')
            ax.plot(d.Datetime_fim, d.rotulo, color='black', lw=0.7, label='Rótulo da janela')
            for positive, color, name in [(1, '#bf3030', 'Predição positiva'), (0, '#2563a6', 'Predição negativa')]:
                q = d[d.predicao.eq(positive)]
                ax.scatter(q.Datetime_fim, q.predicao, s=19, c=color, label=name, zorder=3)
            ax.set_ylabel(NAMES[m])
            ax.set_ylim(-0.18, 1.18)
            ax.set_yticks([0, 1])
            ax.grid(alpha=0.15)
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        axs[-1].set_xlabel('Instante final da janela; classificação referente à janela inteira')
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out / f'temporal_{f}_r1.pdf', bbox_inches='tight')
        plt.close(fig)
        for m in NAMES:
            d = clean[clean.modelo.eq(m) & clean.falha.eq(f)]
            cm = np.array([[d.tn.sum(), d.fp.sum()], [d.fn.sum(), d.tp.sum()]])
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            ax.imshow(cm, cmap='Blues')
            for i in range(2):
                for k in range(2):
                    ax.text(k, i, str(int(cm[i, k])), ha='center', va='center', color='white' if cm[i, k] > cm.max() / 2 else 'black')
            ax.set_xticks([0, 1], ['Negativa', 'Positiva'])
            ax.set_yticks([0, 1], ['Negativa', 'Positiva'])
            ax.set_xlabel('Predição')
            ax.set_ylabel('Rótulo de referência')
            fig.tight_layout()
            fig.savefig(out / f'cm_{m}_{f}.pdf')
            plt.close(fig)
    text = 'Resultados por janela e cobertura retrospectiva de eventos simulados. Configurações selecionadas na validação. As cinco rodadas compartilham os mesmos períodos; as sinalizações sem injeção não têm diagnóstico físico independente.\n'
    (out / 'LEIA_PRIMEIRO.md').write_text(text)
    print('Relatório gerado:', out)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resultados', required=True)
    p.add_argument('--coluna', required=True)
    p.add_argument('--variavel', required=True)
    a = p.parse_args()
    run(a.resultados, a.coluna, a.variavel)
