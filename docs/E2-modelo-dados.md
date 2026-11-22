# E2 · Dados preparados e primeira adaptação

## Extrator de perguntas para feedback educacional de anamnese

**Disciplina:** Tópicos Avançados em Inteligência Artificial · 2026.2  
**Professor:** Marcondes Ricarte  
**Grupo:** G04 · **Integrantes:** Leonardo Marcelo e André Lucas  
**Execução local:** 03/11/2026  
**Versionamento previsto:** 22/11/2026 · **Alinhamento:** 23/11/2026

## 1. Objetivo e recorte da entrega

Esta etapa implementa a preparação dos dados e a primeira adaptação de Qwen3-4B-Instruct-2507 para extrair atos de investigação das falas do entrevistador. A comparação usa uma baseline por regras, o modelo original e o modelo com adaptador LoRA. As entradas e os rótulos de avaliação são os mesmos para os três sistemas.

O experimento é um piloto textual em inglês. Não inclui áudio e transcrição, aplicação, tradução para o português (hipótese D10 da E1), avaliação de omissões por caso ou avaliação humana do feedback. Esses componentes dependem das próximas etapas. A entrega inicial exigida pelo enunciado permite uma execução pequena; por isso, o relatório distingue o corpus preparado do subconjunto efetivamente utilizado no treino.

A tarefa é inspirada no CMAS do MediTOD. Ela não reproduz a NLU publicada no artigo-base, que avalia informações do paciente. As métricas abaixo são locais, baseadas em correspondência exata de rótulos, e não podem ser comparadas diretamente ao F1 0,7139 do artigo.

## 2. Origem, acesso e integridade dos dados

### 2.1 Fonte congelada

Foram adquiridos `raw_data/dialogs.json` e `raw_data/annotations.json` do repositório https://github.com/dair-iitd/MediTOD, fixando a revisão:

```text
0b7a87c3553c9056bc8371b0809811003f94b261
```

| Arquivo | Tamanho em bytes | SHA-256 |
|---|---|---|
| dialogs.json | 4.440.978 | 9d79ecaa10e8da630f9accdb52a1be9a7b9bd812b9e3d19b77579e320a4123b6 |
| annotations.json | 4.330.472 | 828a802da0c5d1c42e92ac15b0fb79b2d34c20e78c9d1c70e1b4423424982c59 |

O arquivo `evidencias/manifesto_dados.json` registra URLs, revisão, hashes, contagens, normalizações e seleção da avaliação. `evidencias/splits_dialogos.json` preserva a lista de identificadores por partição. Os dados brutos e derivados permanecem em `src/data/`, ignorado pelo Git.

### 2.2 Condições de uso e compartilhamento

Não foi identificada uma licença específica para as anotações do MediTOD no repositório ou no artigo. A licença CC0 dos diálogos de Fareez não resolve automaticamente as condições de redistribuição das anotações adicionais.

Os arquivos baixados e os adaptadores resultantes estão preservados apenas no ambiente local. O material preparado para versionamento contém código próprio, configurações, identificadores, estatísticas e logs técnicos. Não inclui os textos do corpus ou pesos derivados. Esclarecer as condições de compartilhamento desses derivados continua pendente: o pedido de autorização aos autores previsto na decisão D6 ainda não foi enviado, e não se presume autorização por finalidade acadêmica. Os pesos-base Qwen e a conversão MLX utilizada informam licença Apache-2.0.

## 3. Preparação, limpeza e rotulagem

### 3.1 Unidade de aprendizado

Cada amostra corresponde a uma fala do entrevistador (`doctor` no corpus), acompanhada de até duas falas anteriores. O modelo recebe somente o texto e os papéis dos falantes. O alvo contém os atos `inquire` anotados naquela fala; rótulos do histórico não são fornecidos como entrada.

A preparação verifica que os conjuntos de IDs dos dois JSON coincidem, que não há identificador de fala duplicado no diálogo e que todos os `uid` anotados correspondem às falas existentes. Turnos do paciente podem fornecer contexto, mas não são usados como alvos de investigação do estudante.

Foram identificadas 11.097 falas do entrevistador e 10.070 atos `inquire` nessas falas. A fonte contém 10.082 atos dessa intenção em todos os falantes; os 12 restantes estão associados a falas do paciente e foram excluídos da tarefa. Não houve anotação humana adicional nesta execução: os alvos derivam das anotações existentes, com transformações declaradas.

### 3.2 Transformações

| Operação | Tratamento executado |
|---|---|
| Categoria | Preservar o vocabulário fechado identificado nos atos de investigação |
| Valor | Preservar texto do rótulo em inglês; normalizar espaços; valor ausente vira string vazia |
| Atributos | Selecionar campos booleanos verdadeiros terminados em `check` |
| Erro de grafia | Unificar `frquency check` em `frequency check` |
| Valores compostos | Preservar o valor composto; não inferir separação sem regra validada |
| Fala sem investigação anotada | Manter exemplo com lista vazia como alvo |
| Comprimento | Filtrar exemplos longos no piloto, em vez de cortar o alvo |
| Português e anonimização | Nenhuma tradução ou anonimização foi executada nesta etapa; o MediTOD é composto de entrevistas simuladas |

A preparação preserva `where check` e `location check` como rótulos distintos nesta rodada. Não agrega categorias por inferência clínica. Essas escolhas favorecem rastreabilidade à anotação, mas podem penalizar respostas semanticamente próximas na métrica exata.

### 3.3 Duplicatas e prevenção de vazamento

A inspeção encontrou 18 pares de diálogos com texto integral idêntico e identificadores diferentes. Portanto, os 231 IDs representam 213 sequências textuais distintas. Uma divisão apenas por ID colocava cópias da mesma entrevista em treino e teste; essa divisão foi descartada antes da execução dos modelos de linguagem.

A divisão final agrupa cada sequência ordenada de pares `(speaker, text)` por SHA-256 e mantém todas as cópias na mesma partição. Os grupos são embaralhados com semente 42. Aproximadamente 80% ficam no treino e 10% em cada uma das outras partições. Como alguns grupos contêm dois IDs, as proporções de IDs não são exatamente 80/10/10.

| Partição | Grupos textuais únicos | IDs de diálogo | Falas do entrevistador |
|---|---|---|---|
| Treino | 171 | 186 | 8.940 |
| Validação | 21 | 24 | 1.143 |
| Teste | 21 | 21 | 1.014 |
| Total | 213 | 231 | 11.097 |

Não há grupo de diálogo integral compartilhado entre as partições. Perguntas genéricas e saudações ainda podem repetir-se em entrevistas diferentes; isso não significa que toda frase do teste seja linguisticamente inédita. A revisão encontrou 15 entradas idênticas ao treino no teste completo, considerando fala atual e duas falas de contexto, mas nenhuma delas pertence às 24 amostras selecionadas para esta avaliação. Os IDs duplicados foram agrupados, não eliminados; assim, as contagens de falas não representam observações independentes em todos os casos.

### 3.4 Piloto de treinamento e avaliação retida

O corpus completo foi preparado, mas o piloto selecionou 256 amostras de treino e 16 de validação. A seleção considera o tamanho tokenizado da sequência completa, incluindo instrução, contexto, saída supervisionada e marcadores do modelo. A seleção inicial, com semente 42, excluiu 33 exemplos longos do treino e 5 da validação. Após a correção do template descrita na seção 7, os mesmos IDs foram mantidos: seus comprimentos máximos são 491 e 407 tokens, abaixo do limite de 512. Com o template corrigido, 31 exemplos do universo de treino e 5 do de validação excedem esse limite; os dois exemplos que passaram a caber não foram acrescentados ao piloto.

O conjunto de avaliação contém 24 amostras dos diálogos de teste: 4 sem ato, 8 com atributo e 12 com investigação sem atributo. Essa composição foi fixada antes da comparação final. Ela favorece a inspeção de tipos diferentes de erro e não estima, sem viés de amostragem, a frequência de erros em todo o corpus. As amostras de avaliação não são usadas em treino nem em cálculo da perda de validação.

`evidencias/manifesto_piloto.json` registra IDs selecionados, limites e hashes dos arquivos utilizados. A preparação dos 8.940 exemplos de treino não significa que todos foram usados: a execução inicial tem apenas 40 passos com lote 1, sem uma época completa sequer sobre os 256 exemplos selecionados.

## 4. Modelo, ambiente e configuração

### 4.1 Origem dos pesos

| Campo | Valor |
|---|---|
| Modelo original | Qwen/Qwen3-4B-Instruct-2507 |
| Conversão utilizada | mlx-community/Qwen3-4B-Instruct-2507-4bit |
| Revisão da conversão | 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b |
| Hardware | Apple M5, arm64, 16 GiB de memória |
| Ambiente | Python 3.12 em ambiente isolado; MLX/Metal |
| Adaptação | LoRA nas últimas quatro camadas, com pesos-base em quatro bits congelados |

As fichas dos modelos estão em https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 e https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit. Os hashes locais e as versões efetivamente usadas constam no manifesto de execução. A quantização foi obtida da conversão comunitária; não foi produzida localmente nesta etapa.

O template de conversa da conversão comunitária foi substituído pelo template oficial da variante Instruct, revisão `cdbee75f17c01a7cc42f958dc650907174af0554`. A origem e o hash estão em `evidencias/template_fonte.json`; o arquivo utilizado é `src/configs/chat_template_oficial.jinja`. A correção elimina marcadores de raciocínio que o template comunitário acrescentava aos alvos, em desacordo com o contrato de JSON puro. Ela é aplicada tanto ao treino quanto à inferência, mantendo o parser estrito.

### 4.2 Hiperparâmetros

| Parâmetro | Configuração do piloto |
|---|---|
| Passos de treino | 40 |
| Lote | 1 |
| Comprimento máximo | 512 tokens |
| Camadas adaptadas | Últimas 4 |
| Módulos adaptados | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` das camadas 32–35 (padrão do MLX-LM) |
| Parâmetros treináveis | 1.835.008 (0,046% de 4.022M), conferidos no log e no arquivo do adaptador |
| Rank LoRA | 8 |
| Scale MLX | 16,0; parâmetro da implementação MLX, não denominado alpha no relatório |
| Dropout | 0,0 |
| Taxa de aprendizado | 0,0001 |
| Otimizador | Adam |
| Máscara do prompt | Ativa; o alvo supervisionado é a resposta |
| Gradient checkpointing | Ativo |
| Semente | 42 |
| Avaliação de perda | Até 4 lotes de validação por avaliação, conforme log |
| Decodificação | Temperatura 0 e limite de 192 tokens, igual para B1 e LoRA |

A configuração está em `src/configs/lora_piloto.yaml`. Não houve busca em grade ou seleção de modelo baseada no teste. Este piloto testa a viabilidade da execução e produz uma primeira comparação. Uma perda de treino menor, isoladamente, não demonstra melhora de extração nem utilidade pedagógica.

## 5. Baselines e protocolo de métricas

### 5.1 Sistemas comparados

**B0, regras:** dicionário construído somente a partir dos valores dos rótulos de treino, com busca de termos na pergunta atual e padrões simples para alguns atributos. Não usa rótulos do teste. É uma referência de busca textual limitada, não um avaliador semiológico.

**B1, modelo original:** mesmo Qwen quantizado, sem adaptador. Recebe a instrução, vocabulário de categorias e atributos, histórico e pergunta atual. Não recebe exemplos rotulados no prompt.

**LoRA, modelo adaptado:** mesmo modelo-base e mesmo prompt, com os adaptadores da primeira execução. O código carrega os pesos localmente e valida a resposta sem acrescentar reparos específicos a uma variante.

### 5.2 Unidades comparadas

A avaliação compara conjuntos por fala, com normalização de espaços e caixa de texto nos valores. As três unidades são:

1. Ato completo: `(categoria, valor, conjunto de atributos)`.
2. Item: `(categoria, valor)`.
3. Atributo: `(categoria, valor, atributo)`.

Para cada unidade, TP é uma unidade presente no alvo e na previsão, FP aparece apenas na previsão e FN aparece apenas no alvo. A agregação micro soma essas contagens antes de calcular precisão, recall e F1. Sinônimos clínicos não recebem equivalência automática. Uma resposta semanticamente aceitável pode ser penalizada por diferença lexical; os resultados devem ser lidos com esse limite.

```text
precisão = TP / (TP + FP)
recall   = TP / (TP + FN)
F1       = 2 × precisão × recall / (precisão + recall)
```

Divisões sem denominador recebem zero por convenção explícita. A acurácia exata por turno exige igualdade de todos os atos daquela fala, incluindo exemplos sem atos. A taxa de JSON válido exige tanto sintaxe válida quanto conformidade com as categorias e os tipos do esquema.

Uma saída inválida fica registrada como erro, não como uma previsão válida de lista vazia. Ela produz falsos negativos para os atos esperados e falha nas métricas de validade e de acerto exato. Se o alvo estiver vazio, não existe ato para contar como FN; por isso, F1 deve ser apresentado junto com validade e acurácia exata, evitando esconder falhas de formato.

## 6. Resultados medidos

### 6.1 Execução registrada

O treinamento terminou com código de saída zero e 40 passos. A primeira e a última perdas de treino reportadas foram 1,569 e 0,620. As perdas de validação inicial e da última leitura foram 2,980 e 0,510. As janelas de treino contêm exemplos diferentes; a curva não representa a avaliação repetida de um conjunto fixo de treinamento. A última validação ocorreu após 39 atualizações, antes da última atualização do adaptador. Cada avaliação sorteia quatro lotes dentre os 16 exemplos de validação; a composição pode mudar, portanto a variação da perda inclui esse efeito de amostragem.

| Medida | Valor observado |
|---|---|
| Duração do processo registrado, incluindo instrumentação | 22,27 s |
| Pico RSS do processo filho, amostrado a cada 0,2 s | 2,36 GiB |
| Pico de alocação MLX informado nos eventos | 3,30 GB decimais |
| Python | 3.12.13 |
| MLX / MLX-LM | 0.32.4 / 0.32.2 |

RSS e memória alocada pelo MLX medem coisas diferentes e não devem ser somados. O RSS amostrado pode deixar de capturar um pico entre duas leituras. Os tempos de download e preparação anterior ao início do processo não estão incluídos.

### 6.2 Comparação na mesma avaliação retida

| Sistema | F1 ato | F1 item | F1 atributo | JSON válido | Acerto exato | s/fala |
|---|---|---|---|---|---|---|
| B0 · regras | 0,061 | 0,121 | 0,167 | 100,0% | 12,5% | 0,001 |
| B1 · original | 0,000 | 0,098 | 0,000 | 75,0% | 8,3% | 1,412 |
| LoRA | 0,000 | 0,000 | 0,000 | 100,0% | 16,7% | 0,324 |

A latência é a média por fala, sem o carregamento inicial dos pesos. A amostra contém 24 falas de 13 diálogos, com 20 atos de referência e 10 atributos. B0 constrói seu dicionário com todos os rótulos de treino; LoRA utiliza o piloto. O orçamento de supervisão de B0 e LoRA não é equalizado.

| Sistema | TP / FP / FN de ato | Precisão de ato | Recall de ato | Macro-F1 de item |
|---|---|---|---|---|
| B0 | 1 / 12 / 19 | 0,077 | 0,050 | 0,033 |
| B1 | 0 / 21 / 20 | 0,000 | 0,000 | 0,098 |
| LoRA | 0 / 0 / 20 | 0,000 | 0,000 | 0,000 |

O macro-F1 usa as mesmas seis categorias presentes na referência para todos os sistemas. Categorias sem suporte no holdout não são avaliadas por essa média.

O F1 de ato permaneceu igual após o ajuste. A diferença LoRA menos B1 foi 0,000; para F1 de item, a diferença foi -0,098. Trata-se de uma diferença observada neste piloto; não foi feito teste de significância e não se demonstrou generalização ou eficácia pedagógica.

Na rodada inicial arquivada, o tempo total de B1 incluiu espera pelo download. Na comparação atual, a tabela utiliza somente a latência individual de geração por fala, medida depois do carregamento; não compara download com inferência.

![Figura 1. Perdas registradas no treino e F1 de item na avaliação piloto de 24 falas em inglês.](../evidencias/curvas_resultados.png)

### 6.3 Rastreabilidade dos resultados

Os números foram lidos de `metricas_b0.json`, `metricas_b1.json`, `metricas_lora.json` e `execucao_treino.json`, que registram o hash do holdout, das previsões, do log, dos eventos e dos adaptadores. `src/configs/registrar_proveniencia.py` confere esses vínculos. Previsões individuais e saídas brutas ficam no diretório local `src/artifacts/`, permitindo inspecionar os erros sem publicar textos do corpus.

A curva e as métricas demonstram uma execução real. O tamanho da amostra, os erros observados e a correspondência lexical exata não permitem concluir que o sistema esteja pronto para avaliar estudantes.

### 6.4 Análise dos erros e conclusão da primeira adaptação

Na rodada corrigida, o adaptador retornou lista vazia em todas as 24 falas. O formato passou a ser válido, mas as 20 referências positivas não foram recuperadas. Os quatro acertos exatos correspondem apenas às quatro falas cujo alvo era vazio. Assim, a melhora de validade, de 75% no modelo original para 100% no adaptado, não representa melhora de extração.

O F1 de item caiu de aproximadamente 0,098 em B1 para zero com LoRA. A baseline por regras obteve aproximadamente 0,121. A latência menor do adaptador também não é vantagem útil: produzir apenas uma lista vazia exige menos tokens do que tentar identificar os atos. O resultado desta primeira adaptação é negativo para a qualidade da tarefa, apesar de o pipeline de treinamento e avaliação ter sido executado.

O comportamento observado é uma concentração das previsões na saída vazia. Ele não foi introduzido pela correção do template: na primeira rodada, as 24 respostas também continham `[]` depois do bloco de raciocínio que o parser rejeitou. Também não se explica por desbalanceamento dos alvos, pois só 37 dos 256 exemplos de treino do piloto (14%) têm lista vazia. A causa não foi isolada. As hipóteses restantes são o número de atualizações (40 passos com lote 1, ou seja, no máximo 40 dos 256 exemplos vistos), a combinação de taxa de aprendizado e escala do LoRA e a adaptação restrita a quatro camadas, a investigar com o conjunto de validação. Não se deve atribuí-lo exclusivamente ao porte do modelo, concluir que a abordagem é inviável ou aumentar o treino até obter um número desejado no mesmo teste.

Para a próxima rodada, a prioridade é comparar a distribuição de exemplos positivos e negativos efetivamente consumidos, acompanhar extração em validação além da perda e escolher checkpoints pela qualidade das previsões. Depois disso, será necessário avaliar em novas entrevistas retidas, pois este holdout já foi inspecionado durante o diagnóstico. Não há evidência nesta entrega de um modelo pronto para feedback educacional.

## 7. Obstáculos e limites de interpretação

A preparação revelou cópias integrais de diálogos com IDs diferentes. O agrupamento por conteúdo corrigiu o vazamento da primeira divisão antes do treino e da comparação final. O download inicial via Xet também apresentou interrupção de progresso; a aquisição foi retomada por HTTP, mantendo a revisão fixa e a verificação de integridade.

Uma primeira rodada de 40 atualizações foi concluída, mas seu adaptador produziu zero respostas válidas nas 24 amostras. A investigação encontrou a causa da invalidez no template comunitário: ele inseria um bloco vazio de raciocínio antes do JSON no texto efetivamente supervisionado. O modelo aprendeu esse formato e o parser o rejeitou. Depois do bloco, as 24 respostas eram `[]`. A perda de validação havia caído, demonstrando por que a curva, sozinha, não comprovava sucesso da extração.

Essa rodada foi preservada em `evidencias/experimento_01_template_original/`, com seus pesos e previsões em `src/artifacts/experimento_01_template_original/`. A correção usa o template oficial e inclui uma regressão sobre a continuação realmente tokenizada, não apenas sobre o campo textual das mensagens. Todos os 272 alvos do piloto corrigido foram conferidos quanto ao formato JSON. Os IDs dos 256 exemplos de treino e dos 16 de validação foram mantidos, assim como os 40 passos e demais hiperparâmetros.

Os resultados da seção 6 correspondem à rodada com o template corrigido. O mesmo holdout foi reutilizado após a inspeção necessária ao diagnóstico; portanto, esta avaliação não deve ser descrita como teste cego. Não houve ajuste de hiperparâmetros para buscar ganho no holdout. A correção remove uma inconsistência de software, e uma avaliação futura independente será necessária para sustentar conclusões de generalização.

O principal limite experimental é a escala: 40 passos e 24 exemplos retidos. A amostra estratificada contém mais exemplos de caracterização do que uma seleção aleatória simples poderia conter e não representa toda a distribuição de uso. Não se pode concluir, a partir dela, que o modelo funciona para todas as perguntas, especialidades, estudantes ou situações de entrevista.

O ajuste e a avaliação são em inglês, os rótulos provêm do corpus e os valores não passaram por revisão clínica da dupla ou da cliente nesta execução. As métricas medem correspondência com esses rótulos. Elas não validam o roteiro pedagógico, não medem omissões clínicas e não demonstram efeito no aprendizado. A tradução para português e a avaliação de áudio continuam fora desta entrega.

Os dados e adaptadores permanecem locais por pendência das condições de redistribuição das anotações. Os manifestos e instruções permitem conferir a execução e reconstruí-la, mas não devem ser apresentados como licença de terceiros ou aprovação de compartilhamento.

## 8. Continuidade proposta até E3

| Ação | Evidência esperada | Prioridade |
|---|---|---|
| Inspecionar erros de categorias, valores e atributos | Amostra revisada com motivo e referência textual | Antes de aumentar treino |
| Investigar previsões sempre vazias | Distribuição de alvos consumidos e métricas de extração na validação | Primeira correção do modelo |
| Aumentar passos e seleção de treino usando validação | Curvas e configuração nova; checkpoint escolhido por validação | Próxima rodada técnica |
| Ampliar a avaliação retida | Métricas por categoria e por diálogo, com incerteza documentada | Antes de afirmar generalização |
| Preparar conjunto em português com revisão independente | Origem, partições e anotações auditáveis | Antes de avaliar público brasileiro |
| Revisar roteiro com a docente | Registro de itens pertinentes e limites | Antes de feedback educacional |
| Esclarecer licença das anotações e derivados | Condição documentada de uso e compartilhamento | Antes de redistribuir dados/adaptadores |
| Integrar transcrição, revisão e feedback | Fluxo executável e análise de falhas do modelo | E3, ainda não implementada |

As atividades não foram realizadas nesta etapa e não possuem agendamento presumido. A análise dos resultados deste piloto servirá para definir uma próxima configuração; qualquer nova rodada deve possuir identificação e métricas próprias, preservando os registros atuais.

## 9. Reprodução, artefatos e material de alinhamento

O ponto de entrada técnico é `src/README.md`. Os modelos de linguagem dependem de Apple Silicon e Metal nesta implementação; isso não constitui suporte testado a Windows, Linux ou CUDA.

| Artefato | Finalidade |
|---|---|
| `src/motor/data.py` | Aquisição e preparação com validação dos dados |
| `src/motor/model.py` | Prompt, parser e carregamento do modelo |
| `src/motor/train.py` | Seleção do piloto e execução do LoRA |
| `src/motor/evaluate.py` | Baselines e cálculo das métricas |
| `src/configs/lora_piloto.yaml` | Hiperparâmetros fixos |
| `src/configs/chat_template_oficial.jinja` e `piloto_ids.json` | Template corrigido e exemplos fixados entre as rodadas |
| `src/configs/registrar_proveniencia.py` | Vincular métricas, modelo e adaptador por hash após as avaliações |
| `src/requirements-ml.txt` | Dependências resolvidas |
| `evidencias/manifesto_dados.json` | Fonte, hashes, contagens e divisão |
| `evidencias/manifesto_piloto.json` | Exemplos e comprimentos usados no piloto |
| `evidencias/execucao_treino.json` e `treinamento.log` | Ambiente, execução, perdas, duração e adaptadores |
| `evidencias/metricas_b0.json`, `metricas_b1.json`, `metricas_lora.json` | Métricas medidas sobre o mesmo conjunto |
| `src/artifacts/lora_piloto/` | Adaptadores locais; ignorados pelo Git |
| `src/artifacts/predictions_*.jsonl` | Previsões e erros por amostra; arquivos locais |
| `evidencias/revisao_tecnica.md` | Conferência independente de dados e avaliação |

## 10. Matriz de atendimento da E2

| Exigência do enunciado | Material |
|---|---|
| Limpeza, rotulagem e pré-processamento | Seção 3; código e manifesto |
| Partições justificadas | Seção 3.3; IDs e grupos de conteúdo |
| Baseline comparável | Seção 5; métricas B0 e B1 no mesmo holdout |
| Primeiro fine-tuning e hiperparâmetros | Seções 4 e 6; configuração, log e adaptadores locais |
| Ambiente e evidências de execução | Manifesto de execução e instruções |
| Resultados sem estimativas apresentadas como medições | Seção 6, vinculada aos JSON e logs |
| Obstáculos e plano de correção | Seções 7 e 8 |
| Pesos ou adaptadores e condições de acesso | Diretório local e restrição de redistribuição explicitada |

## 11. Referências principais

Saley, V. V. et al. (2024). MediTOD: An English Dialogue Dataset for Medical History Taking with Comprehensive Annotations. EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936

Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. https://arxiv.org/abs/2106.09685

Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314

Qwen Team. (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388. Modelo: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

MLX-LM. Documentação de LoRA, parâmetros e execução. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

A fundamentação educacional, a evidência da cliente e as condições das fontes estão na E1 (`docs/E1-definicao.md`) e em `referencias.bib`, na raiz do repositório.
