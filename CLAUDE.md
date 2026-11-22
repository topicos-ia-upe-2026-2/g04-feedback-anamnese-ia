# Instruções para assistentes de IA

Projeto aplicado de **Tópicos Avançados em Inteligência Artificial** (2026.2, Prof. Marcondes Ricarte), grupo G04:
Leonardo Marcelo e André Lucas. Repositório: `topicos-ia-upe-2026-2/g04-feedback-anamnese-ia`.

## Projeto

A partir do áudio da anamnese que o estudante de medicina faz com um paciente real, o sistema transcreve a entrevista,
identifica o que cada pergunta do estudante investigou e devolve um feedback por interação: acertos, o que precisa
melhorar e o que deixou de investigar. **Não diagnostica, não prescreve e não recomenda conduta.**

- Cliente: Marília Costa, docente de semiologia. A evidência inicial é a entrevista registrada em
  `evidencias/entrevista_cliente.md` do repositório.
- Artigo-base: MediTOD (Saley et al., EMNLP 2024), esquema CMAS. Extrair atos `inquire` das falas do entrevistador é
  uma **tarefa adaptada**; não é a NLU do artigo, e o F1 0,7139 do artigo não é comparável.
- Modelo: Qwen3-4B-Instruct-2507 (Apache-2.0), conversão `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revisão
  `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, com LoRA em MLX (Apple M5, 16 GiB). Whisper large-v3-turbo sem ajuste
  na transcrição (a partir da E3).
- Dados: MediTOD `raw_data/` no commit `0b7a87c3553c9056bc8371b0809811003f94b261`; áudios simulados de Fareez et al.
  (CC0). O português é uma hipótese por tradução (decisão D10).
- A fonte da verdade para o problema, o escopo e as decisões D1–D10 é `docs/E1-definicao.md` e `docs/decisoes.md`
  do repositório. Não contradiga esses documentos; se algo mudar, registre como nova decisão.

## Entregas

| Etapa | Prazo | Conteúdo |
|---|---|---|
| E1 · Definição | 11/11 | Problema, artigo, modelo, dados e plano (PR #2) |
| E2 · Modelo e dados | 22/11 + alinhamento 23/11 | Dados preparados, baseline e primeira execução de fine-tuning |
| E3 · Aplicação | 29/11 + alinhamento 30/11 | Protótipo integrado e resultados experimentais |
| E4 · Apresentação | 07/12 | Demonstração, defesa técnica e entrega final |

Requisitos da E2:

- **Dados:** limpeza, rotulagem, pré-processamento e partições de treino, validação e teste justificadas.
- **Baseline:** referência comparável nas mesmas condições de teste.
- **Fine-tuning inicial:** código, configuração, parâmetros ajustados, hiperparâmetros, ambiente e evidências
  (logs, curvas, métricas).
- **Resultados e obstáculos:** primeiros resultados, limitações e plano até a E3.
- **Material:** instruções de execução, origem dos dados, breve relatório de progresso e pesos/adaptadores disponíveis
  de forma permitida pela licença.

A revisão do professor no PR #2 pede ainda para a E2: baseline few-shot, tradução da partição de teste para o
português, distribuição de rótulos e de listas vazias por partição, piloto de 200–500 iterações, `docs/experimentos.md`
preenchido, testes na CI e exceções no `.gitignore` para os JSON de evidência.

## Fluxo de Git

- Uma branch por entrega (`deliverable-e1`, `deliverable-e2`, ...) e um PR para `main` com o template da etapa.
  Só o professor aprova e mescla. Não renomeie branch com PR aberto.
- O repositório contém apenas o que a entrega atual pede.
- Commits pequenos, com mensagem descritiva em português. Não adicione linhas de coautoria ou de atribuição a
  ferramentas de IA em commits ou PRs.
- Arquivos acima de 50 MB são bloqueados pela CI.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `docs/E1-definicao.md`, `docs/decisoes.md` | Definição do projeto e registro de decisões (fonte da verdade) |
| `docs/E2-modelo-dados.md` e `.pdf` | Relatório de progresso da E2 (o PDF é exportado do Markdown) |
| `docs/experimentos.md`, `docs/relatorio.md` | Registro de execuções e relatório técnico acumulado |
| `src/motor/` | `data.py` (preparação), `model.py` (prompt, parser, MLX), `train.py` (LoRA), `evaluate.py` (B0/B1/LoRA e métricas) |
| `src/configs/` | Hiperparâmetros, template oficial do Qwen, IDs fixos do piloto, registro de proveniência |
| `src/tests/` | Regressões de alinhamento, vazamento, parser e métricas |
| `evidencias/` | Entrevista da cliente (E1); manifestos, métricas, logs, curva e revisão técnica (E2) |
| `src/data/`, `src/artifacts/` | Dados e adaptadores **apenas locais** (ignorados pelo Git) |

O código calcula os caminhos a partir de `src/` e grava evidências em `evidencias/` na raiz. Os hashes de
`src/motor/*.py` estão registrados em `evidencias/execucao_treino.json`: alterar esses arquivos exige nova rodada.

Comandos, sempre partindo de `src/`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ml.txt
export PYTHONPATH="$PWD" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check --target-version py39 motor tests configs
```

## Regras de trabalho

- **Nunca apresente estimativa como resultado medido.** Todo número de relatório deve sair de um arquivo de evidência
  (`metricas_*.json`, `execucao_treino.json`, logs) e ser conferível por hash.
- Não versione dados do MediTOD, traduções, previsões nem adaptadores: a licença das anotações não foi identificada
  (D6). Versione código, configurações, IDs, estatísticas agregadas e logs sem texto do corpus.
- Não grave caminhos absolutos de máquina, e-mails ou dados pessoais em arquivos versionados. O repositório é público.
- Saída inválida do modelo é erro explícito. Não conserte JSON, não remova markdown e não repita a geração para
  melhorar a validade.
- B0, B1 e LoRA (e qualquer novo baseline) usam as mesmas amostras, o mesmo prompt, temperatura 0, `max_tokens` 192 e
  o mesmo parser.
- O holdout de 24 falas já foi inspecionado no diagnóstico do template. Não é teste cego: não escolha
  hiperparâmetros por ele, e use validação ou um novo conjunto retido.
- Uma nova rodada de treino recebe identificação, configuração e métricas próprias, sem sobrescrever as evidências
  anteriores. Reexecutar os scripts substitui arquivos de mesmo nome: copie antes.
- Não afirme validação clínica, avaliação humana, eficácia pedagógica, autorização de licença ou aceite do professor
  que não tenham ocorrido.
