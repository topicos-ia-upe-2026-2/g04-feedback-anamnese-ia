# Experimento local da Etapa 2

Extração retrospectiva de atos `inquire` de falas do médico no MediTOD, em inglês.
Não reproduz a tarefa NLU de falas do paciente nem a métrica do artigo. Não há
validação humana, tradução ou avaliação clínica neste piloto.

## Ambiente e execução

Requer macOS em Apple Silicon para MLX/Metal. O ambiente medido usa Python 3.12;
o núcleo sem MLX mantém sintaxe compatível com Python 3.9. Os pesos ocupam cerca
de 2,3 GB no cache Hugging Face. A lista exata de dependências está em
`requirements-ml.txt`.

Partindo desta pasta (`src/`):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ml.txt
export PYTHONPATH="$PWD"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check --target-version py39 motor tests configs
.venv/bin/python -m motor.data
.venv/bin/python -m motor.evaluate --variant b0
.venv/bin/python -m motor.evaluate --variant b1
.venv/bin/python -m motor.train
.venv/bin/python -m motor.evaluate --variant lora
.venv/bin/python configs/registrar_proveniencia.py
```

Os comandos acima reproduzem a primeira adaptação (LoRA-02). As rodadas de correção usam pastas próprias
(`data/<run>/`, `artifacts/<run>/`, `../evidencias/<run>/`), sem sobrescrever as anteriores:

```sh
.venv/bin/python -m motor.train --config configs/lora04.yaml --run lora04
.venv/bin/python -m motor.select --run lora04        # F1 de extração por checkpoint na validação
.venv/bin/python -m motor.evaluate --variant lora --adapter artifacts/lora04/melhor --evidence ../evidencias/lora04 --tag holdout
.venv/bin/python -m motor.evaluate --variant lora --adapter artifacts/lora04/melhor \
  --rows data/prepared/test_rows.jsonl --evidence ../evidencias/teste_completo --tag lora04
```

Sem o tokenizer do modelo no cache do Hugging Face, os testes terminam com 17
aprovados e 1 ignorado (o teste de integração do template); com o cache, 18 aprovados.

Os comandos de dados e modelo baixam as revisões fixadas no código. Arquivos
brutos existentes são conferidos contra SHA-256 conhecido. As variáveis acima
forçam download por HTTP, porque o download via Xet travou na execução registrada.
Não é necessário token ou serviço de inferência remoto. Downloads públicos
transferem os arquivos de origem; a inferência e o treino ocorrem localmente.

**Reexecutar substitui os artefatos e relatórios de mesmo nome.** Faça uma cópia
de `src/artifacts/` e `evidencias/` antes de repetir caso queira preservar esta execução.

## Partições, piloto e evidências

O corpus completo é preparado. A divisão é por grupos de diálogos com conteúdo
idêntico: SHA-256 da sequência ordenada de `(speaker, text)`. IDs diferentes de
uma entrevista duplicada ficam na mesma partição. O manifesto e as listas de
IDs ficam em `../evidencias/`; dados textuais ficam em `data/` e não são versionados.

O piloto filtra exemplos cujo template completo, incluindo a resposta, tem no
máximo 512 tokens. Isso evita cortar o alvo durante treinamento. São sorteadas
256 amostras elegíveis de treino e 16 de validação na primeira preparação; seus
IDs são fixados em `configs/piloto_ids.json` e preservados na execução corrigida.
**40 atualizações de lote1
não equivalem a treinar em todas as 256 amostras**. Quatro camadas finais recebem
LoRA de rank8 e scale16, com taxa1e-4 e prompt mascarado na perda. A configuração
está em `configs/lora_piloto.yaml`; não houve busca de hiperparâmetros.

O holdout é escolhido antes de B1/LoRA e permanece fixo para B0/B1/LoRA: 24 falas
em inglês, estratificadas em 4 sem ato, 8 com atributo e 12 outras positivas.
Essa amostra diagnóstica pequena não estima o desempenho populacional. A
seleção e os hashes podem ser auditados em `manifesto_dados.json` e
`manifesto_piloto.json`.

B0 usa um léxico obtido de **todos** os rótulos de treino, enquanto LoRA usa o
subconjunto piloto. O orçamento de exemplos não é equalizado. B1 e LoRA usam o
mesmo prompt, histórico de duas falas, temperatura0, limite192 tokens e parser.
Nenhum reparo de JSON, remoção de markdown ou repetição para melhorar a validade
é realizado. Falhas ficam explícitas; não são interpretadas como ausência de
investigação.

As métricas usam conjuntos exatos por fala: ato=`(slot, value, atributos)`,
slot/valor=`(slot, value)` e atributo=`(slot, value, atributo)`. Normalizam-se
caixa e espaços; sinônimos e valores compostos permanecem distintos. O F1 macro
usa os mesmos slots presentes no gold para todos os sistemas. Saída inválida
conta todos os atos gold como falsos negativos. Apresente F1 sempre junto com
validade de JSON e acerto exato por turno, pois F1 isolado não penaliza uma
saída inválida num turno cujo gold é vazio.

`../evidencias/treinamento.log` e `execucao_treino.json` registram perdas, tempo,
configuração, versões, memória RSS amostrada, pico MLX informado pelo treinador
e hashes. `metricas_{b0,b1,lora}.json` contém métricas agregadas e IDs; saídas
textuais por amostra ficam apenas em `artifacts/predictions_*.jsonl`, ignorado
pelo Git. RSS e alocação MLX são medidas diferentes, e não devem ser somadas.
Na tentativa original arquivada, B1 foi iniciado enquanto o modelo ainda baixava:
seus campos `load_seconds` e `total_seconds` incluem a espera pelo cache. A rodada
corrigida usa o modelo já no cache. As latências por
fala em `latency_seconds` excluem carregamento e download e são a medida de
inferência comparável. A validação no treino é realizada antes das atualizações
1, 20 e 40; os eventos registram corretamente 0, 19 e 39 atualizações concluídas.
Cada passagem de validação sorteia quatro das 16 amostras; sua perda não usa um
miniconjunto fixo. O holdout de geração permanece fixo nas três comparações.
`configs/registrar_proveniencia.py` acrescenta aos relatórios os hashes observados
após as execuções, valida a integridade de código, checkpoint e predições, e
registra o momento dessa observação. Não altera as predições originais.

## Correção do template e limite do experimento

A primeira tentativa treinou com o template do conversor MLX, que inseria
automaticamente `<think>...</think>` antes das respostas supervisionadas. As
24 saídas LoRA falharam no parser JSON estrito. O experimento completo foi
preservado em `../evidencias/experimento_01_template_original/`; textos, dados e
pesos dessa tentativa ficam em `artifacts/experimento_01_template_original/`.

Foi aplicada a versão oficial do template de `Qwen/Qwen3-4B-Instruct-2507`,
revisão `cdbee75f17c01a7cc42f958dc650907174af0554`, nas etapas de preparação,
treinamento e inferência. Ela é verificada por SHA-256 antes de uso. O teste real
confere que a continuação supervisionada contém somente JSON e EOS; a auditoria
verificou todos os 272 alvos piloto e os 24 prefixos de inferência. Nenhum
prefixo de inferência mudou; os marcadores indevidos deixaram de ser ensinados.
O parser continua estrito e não remove marcadores ou conserta respostas.
A licença Apache-2.0 desse template de terceiros está preservada em
`configs/THIRD_PARTY_LICENSE_QWEN.txt`; a origem e revisão constam em
`../evidencias/template_fonte.json`. Essa cópia de licença se refere ao material
Qwen e não altera a licença do restante do projeto.

B1 e LoRA foram reexecutados, mantendo holdout, IDs de treino/validação e
hiperparâmetros. **O holdout já foi inspecionado durante o diagnóstico; esta
rodada é diagnóstica e não constitui teste cego.** Não houve busca de
hiperparâmetros, revisão humana ou conjunto independente adicional.

O artigo/repositório do MediTOD não comprovam licença de redistribuição dos
dados. Não se presume autorização por uso acadêmico. Raw, derivados e
adaptadores permanecem locais e ignorados; confirmar direitos com os titulares
é pendência. O modelo base e sua conversão MLX declaram Apache-2.0.

## Uso programático

```python
from motor.model import get_extractor

extract = get_extractor("artifacts/lora_piloto")  # uma carga por backend
acts = extract("When did the cough start?", history=[])
```

O contrato também está disponível como
`extract_turn(text, history=None, adapter_path=None)`. Sem adaptador, usa B1.
Retorna lista de `{slot, value, attributes}`; falhas lançam exceção explícita.
Um objeto `MLXExtractor` pode ser mantido pela aplicação para reutilizar a carga.
