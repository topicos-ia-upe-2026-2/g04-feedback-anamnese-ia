# Revisão técnica independente da E2

**Data:** 03/11/2026.  
**Escopo:** `motor/data.py`, `model.py`, `evaluate.py`, `train.py`, testes, arquivos preparados e manifestos. Revisão automatizada e por inspeção de código; não é validação clínica ou avaliação humana do produto.

## Achados e correções verificadas

| Achado inicial | Consequência | Estado após revisão |
|---|---|---|
| Divisão somente por ID, apesar de cópias integrais com IDs diferentes | Vazamento de entrevistas entre treino e teste | Corrigido: divisão por hash da sequência completa de falante e texto; teste de regressão acrescentado |
| `data/pilot/` não abrangido inicialmente pelos padrões de exclusão | Risco de incluir dados derivados no versionamento | Corrigido: `data/` e `artifacts/` ignorados pelo Git; conferência com `git check-ignore` |
| Parser JSON aceitava duas ocorrências da mesma chave | A última chave substituía silenciosamente a anterior | Corrigido: rejeição explícita de chaves repetidas; teste de regressão |
| Template comunitário acrescentava bloco `<think>` à continuação supervisionada | O treino ensinava uma saída incompatível com o parser JSON; a primeira rodada teve 24 respostas inválidas | Corrigido com template oficial fixado e aplicado à preparação, treino e inferência; primeira rodada preservada |
| Macro-F1 usava categorias presentes em gold ou predição de cada sistema | Denominador variável entre sistemas | Corrigido: categorias com suporte no gold, fixas para todos; falsos positivos fora delas continuam no micro e em `per_slot` |
| Arquivo bruto em cache poderia ser aceito sem confrontar hash esperado | Proveniência declarada poderia divergir do conteúdo local | Corrigido: SHA-256 esperado verificado antes da preparação; teste de alteração do arquivo |

O problema de vazamento era material: no split inicial, `412:30` reproduzia a entrada e o alvo de `312:30`, enquanto `524:8` reproduzia `424:8`. Não se devem usar as métricas desse split como resultado final. Os artefatos atuais foram regenerados com a divisão por conteúdo.

## Dados e separação dos conjuntos

A comparação independente de todas as sequências `(speaker, text)` encontrou **231 IDs, 213 conteúdos únicos e 18 pares de cópias integrais**. Após a correção, nenhum conteúdo completo atravessa partições.

| Partição | IDs de diálogo | Grupos de conteúdo | Falas do entrevistador |
|---|---:|---:|---:|
| Treino | 186 | 171 | 8.940 |
| Validação | 24 | 21 | 1.143 |
| Teste | 21 | 21 | 1.014 |

Há **zero interseções de IDs** entre as três partições. Os 11.097 turnos de entrevistador produzem 10.070 atos `inquire`. Os 12 atos restantes dos 10.082 encontrados no arquivo-fonte pertencem a outros falantes e não entram no alvo. Essa diferença está registrada no manifesto.

A divisão por diálogo não torna toda pergunta inédita. Comparando a entrada exata `(fala atual, duas falas anteriores)`, entre entrevistas de conteúdo completo diferente:

- Validação: 26 entradas também presentes no treino, das quais 6 possuem alvo não vazio.
- Teste completo: 15 entradas também presentes no treino, das quais 1 possui alvo não vazio.
- Holdout diagnóstico de 24 exemplos: **nenhuma entrada também presente no treino**.

Grande parte da repetição residual corresponde a perguntas iniciais genéricas; existe também uma pergunta clínica repetida no teste. Isso deve ser distinguido do vazamento de entrevistas integrais. O agrupamento não identifica casos clínicos semelhantes, paráfrases ou possível exposição dos pesos pré-treinados ao corpus.

Os hashes de todos os arquivos preparados e dos dois arquivos do piloto foram recalculados e conferem com os manifestos. Hash do holdout revisado: `1699057205767186c67020b5dd682c79d0c1305140cac921cd16c756c3725283`.

## Alinhamento, tokenização e contrato

O alinhamento usa `uid`, não a posição da anotação na lista. Falas do paciente fornecem contexto, mas não produzem exemplos supervisionados de perguntas do entrevistador. IDs duplicados, anotações ausentes e IDs desconhecidos são tratados como erro. A entrada não contém os rótulos-alvo nem falas futuras.

Na primeira conferência, foram verificados os comprimentos e o alinhamento dos prefixos, mas o conteúdo decodificado da continuação supervisionada ainda não havia sido confrontado com o contrato JSON. A execução revelou a lacuna: o template comunitário adicionava `<think>\n\n</think>\n\n` antes do alvo. Esse problema foi demonstrado diretamente pela decodificação, sem nova inferência.

Após a correção, foram executados o tokenizer Hugging Face e o wrapper real de tokenizer MLX, sem carregar pesos, sobre todos os exemplos selecionados. Os tokens de ambos coincidem, e cada continuação supervisionada é exatamente o JSON de referência seguido por `<|im_end|>\n`, sem bloco de raciocínio. O template provém de [Qwen3-4B-Instruct-2507, revisão cdbee75f17c01a7cc42f958dc650907174af0554](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json), com hash preservado em `template_fonte.json`.

| Medida | Treino piloto | Validação piloto |
|---|---:|---:|
| Exemplos | 256 | 16 |
| Maior sequência completa | 491 tokens | 407 tokens |
| Menor alvo após o prompt | 3 tokens | 3 tokens |
| Divergências entre prefixo de treino e prompt de inferência | 0 | 0 |

O limite configurado é 512 tokens. Nenhum alvo desses arquivos é truncado, e todos os JSON de referência passam pelo parser. O maior JSON de referência do holdout ocupa 31 tokens, abaixo do teto de geração de 192; isso não garante que respostas geradas terminem antes do teto.

O filtro atual exclui 31 amostras de treino e 5 de validação. A primeira rodada excluía 33 e 5; para isolar a correção de software, foram preservados os mesmos 256 e 16 IDs, assim como todos os hiperparâmetros e o holdout. O piloto favorece sequências curtas e não demonstra capacidade em entrevistas longas.

## Métricas e interpretação

Os testes cobrem alinhamento, isolamento de cópias integrais, integridade dos dados brutos, erros explícitos de JSON, macro-F1 fixo, denominadores de recall e a continuação efetivamente supervisionada. A execução final verificada terminou com **18 testes aprovados**, com o tokenizer fixado no cache local; sem ele, o teste de integração do template é ignorado e os demais 17 passam.

Saída inválida permanece `None`. Quando o gold possui atos, eles contam como falsos negativos; quando o gold é vazio, não existe ato para contar como falso negativo. Nesse segundo caso, a validade estrutural e a acurácia exata do turno registram a falha. Por exemplo, uma previsão inválida para gold vazio e uma previsão correta para um turno não vazio podem produzir F1 micro 1,0, validade 0,5 e acurácia exata 0,5. É obrigatório interpretar essas medidas conjuntamente. Se todas as referências forem vazias, o código convenciona F1 0; a acurácia exata distingue sucesso de falha.

O macro-F1 considera somente categorias suportadas pelo gold e usa zero quando a divisão é indefinida. Categorias adicionais alucinadas afetam o micro e a tabela por categoria. Essa convenção deve acompanhar qualquer tabela de resultados.

O B0 usa vocabulário extraído das 8.940 falas de treino, enquanto o piloto LoRA seleciona 256 exemplos desse conjunto. São orçamentos de supervisão diferentes. A comparação controlada principal é Qwen sem adaptador versus Qwen com adaptador nas mesmas entradas; B0 é referência adicional.

O holdout foi estratificado por rótulo para conter 4 turnos vazios, 8 com atributos e 12 outros. Seus resultados são diagnósticos, não estimativa representativa da prevalência do corpus. Correspondência lexical exata também penaliza sinônimos e diferenças nos valores compostos. Nenhuma dessas métricas mede omissão clínica, aprendizagem ou qualidade pedagógica.

## Comandos e procedimentos executados

Todos os comandos abaixo partem de `src/`. A revisão não alterou o código de implementação; os problemas foram comunicados ao implementador e reavaliados após suas correções.

```sh
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q tests/test_data_model.py
git check-ignore data/raw/dialogs.json data/prepared/train.jsonl data/pilot/train.jsonl artifacts/predictions_b0.jsonl
```

A auditoria independente de cópias e entradas compartilhadas foi executada por leitura dos JSON locais, agrupando diálogos por `tuple((t['speaker'], t['text']) for t in utterances)`, sem chamar a função de split da implementação. Para cada grupo, foi confrontada sua presença em treino, validação e teste. As entradas foram comparadas pela serialização de `[text, history]` com chaves ordenadas.

A verificação final de tokenização usou `AutoTokenizer.from_pretrained` e `mlx_lm.utils.load_tokenizer` sobre a pasta local da revisão `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, aplicando `configure_tokenizer` a ambos. Para cada amostra, comparou `apply_chat_template(messages)` com o prefixo produzido por `apply_chat_template(messages[:-1], add_generation_prompt=True)`, verificou comprimento até 512 e exigiu que `decode(full[len(prefix):])` fosse igual ao conteúdo JSON original mais `<|im_end|>\n`. Não foram carregados pesos ou executada inferência nessa conferência.

## Estado da execução do modelo

Foram conferidas as duas execuções reais. A primeira realizou 40 atualizações em 29,9909 segundos, mas o template incorreto provocou falha de formato em todas as 24 previsões LoRA. Seus registros estão preservados em `experimento_01_template_original/`, com artefatos correspondentes mantidos localmente. Essa rodada não foi tratada como resultado final de extração.

A segunda execução preservou configuração, IDs do piloto e holdout, corrigindo o template. Terminou com código zero e 40 atualizações em **22,2670 segundos**. O pico MLX registrado foi **3,2992 GB decimais** e o pico RSS amostrado foi **2.530.852.864 bytes**; são medidas distintas. A revisão verificou os hashes dos manifestos de dados e piloto, log, eventos, arquivos de código, template e todos os adaptadores. O adaptador final coincide com o checkpoint de 40 atualizações e com a proveniência das previsões LoRA:

```text
83d98600ac303238a352f31b53e6ed6dc4fd97d3b71f9bcd78fa1dc5d8fadff2
```

Os metadados de proveniência acrescentados depois da inferência estão identificados como conferência posterior em `provenance_note`; os arquivos de previsões permaneceram inalterados.

Os resultados abaixo foram recalculados independentemente, sem chamar `score_predictions`: cada previsão foi associada ao ID e ao gold do holdout, os conjuntos foram reconstruídos e TP/FP/FN, F1 micro, macro-F1, validade e acerto exato foram confrontados com os JSON publicados. Todos conferem.

| Sistema | F1 ato | F1 item | F1 atributo | JSON válido | Acerto exato |
|---|---:|---:|---:|---:|---:|
| B0 | 0,060606 | 0,121212 | 0,166667 | 24/24 | 3/24 |
| B1 | 0,000000 | 0,097561 | 0,000000 | 18/24 | 2/24 |
| LoRA corrigido | 0,000000 | 0,000000 | 0,000000 | 24/24 | 4/24 |

**O LoRA corrigido devolveu `[]` nas 24 amostras.** O LoRA da primeira rodada também respondia `[]` em todas elas, depois do bloco `<think>` que o parser rejeitou; o colapso para lista vazia, portanto, não decorre da correção do template. Os quatro acertos são justamente os turnos com referência vazia; os 20 atos esperados ficaram como falsos negativos. A correção de formato resolveu o defeito do template, mas este piloto não demonstrou melhora da extração. F1 de item caiu em relação a B1. A latência menor de respostas vazias não comprova vantagem útil do modelo.

Os eventos de validação são anteriores às atualizações 1, 20 e 40: correspondem a **0, 19 e 39 atualizações concluídas**, com perdas 2,979584; 0,653559; 0,510435. O adaptador final recebeu 40 atualizações. Cada avaliação reamostra quatro lotes entre os 16 exemplos disponíveis; a variação de composição impede interpretar a curva como medição repetida de um conjunto fixo. Os eventos de treino registram perdas médias de janelas distintas, de 1,568914 na primeira janela a 0,619920 na última. Essa redução não contradiz o recall zero observado na geração.

Como o holdout foi inspecionado durante o diagnóstico do defeito, a segunda rodada é uma reavaliação técnica no mesmo conjunto, sem condição de teste cego. Não houve ajuste de hiperparâmetros para elevar as métricas. Uma próxima tentativa de melhorar extração deve usar validação e reservar avaliação ainda não inspecionada.

A apresentação da última perda de validação foi corrigida para 39 atualizações. README e relatório E2 foram conferidos quanto à separação entre execução local, resultados medidos, trabalho futuro e validação humana não realizada.
