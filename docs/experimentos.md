# Registro de experimentos

> Documente cada execução relevante: data, configuração, hiperparâmetros, dados usados,
> métricas obtidas e observações. Isso fundamenta as seções de resultados da E2, E3 e E4.

Todas as execuções abaixo usam o mesmo holdout diagnóstico: 24 falas do médico em inglês, da partição de teste
(4 sem ato, 8 com atributo, 12 outras), SHA-256 `1699057205767186c67020b5dd682c79d0c1305140cac921cd16c756c3725283`,
definido em `evidencias/manifesto_dados.json`. Métricas: F1 micro por conjunto exato (ato = categoria + valor +
atributos; item = categoria + valor; atributo = categoria + valor + atributo), validade de JSON e acerto exato por fala.
Saída inválida conta os atos esperados como falsos negativos. O holdout foi inspecionado no diagnóstico do
experimento 01; **não é teste cego**.

## Baseline

| Data | Descrição | Configuração | Métricas | Observações |
|---|---|---|---|---|
| 03/11/2026 | B0 · regras | Vocabulário de valores dos rótulos das 8.940 falas de treino + padrões regex para 5 atributos; busca na fala atual | F1 ato 0,061 · item 0,121 · atributo 0,167 · JSON 100% · exato 12,5% · 0,001 s/fala | `evidencias/metricas_b0.json`. Previsões idênticas ao reexecutar |
| 03/11/2026 | B1 · Qwen3-4B-Instruct-2507 4-bit sem adaptador (exp. 01) | Prompt zero-shot com o vocabulário CMAS, histórico de 2 falas, temperatura 0, 192 tokens, parser estrito | F1 ato 0,000 · item 0,098 · atributo 0,000 · JSON 75% · exato 8,3% · 1,47 s/fala | Template comunitário. `evidencias/experimento_01_template_original/metricas_b1.json` |
| 03/11/2026 | B1 · mesmo modelo, template oficial (exp. 02) | Igual ao anterior, com o template oficial do Instruct-2507 | F1 ato 0,000 · item 0,098 · atributo 0,000 · JSON 75% · exato 8,3% · 1,41 s/fala | `evidencias/metricas_b1.json`. Das 6 saídas inválidas, 5 usam atributo fora do vocabulário e 1 foi truncada em 192 tokens; o modelo lista até 15 atributos por ato |

## Versões ajustadas (fine-tuning)

| Data | Descrição | Hiperparâmetros | Dados | Métricas | Observações |
|---|---|---|---|---|---|
| 03/11/2026 | LoRA-01 · template comunitário | LoRA rank 8, scale 16, dropout 0, últimas 4 camadas (q/k/v/o/gate/up/down), Adam lr 1e-4, 40 passos, lote 1, max 512 tokens, prompt mascarado, semente 42 | Piloto: 256 treino / 16 validação (filtro de 512 tokens) | JSON 0% (24/24 inválidas) · F1 0,000 em todas as unidades · perda de validação 5,930 → 0,450 | O template inseria `<think>…</think>` no alvo; depois do bloco, as 24 respostas eram `[]`. `evidencias/experimento_01_template_original/` |
| 03/11/2026 | LoRA-02 · template oficial | Iguais aos do LoRA-01 | Mesmos IDs do piloto (`src/configs/piloto_ids.json`) | F1 ato/item/atributo 0,000 · JSON 100% · exato 16,7% · 0,32 s/fala · perda de validação 2,980 → 0,510 · 22,3 s de treino · pico RSS 2,36 GiB | Respondeu `[]` nas 24 falas: os 4 acertos são as falas sem ato. Resultado negativo para extração. `evidencias/execucao_treino.json`, `evidencias/metricas_lora.json` |
