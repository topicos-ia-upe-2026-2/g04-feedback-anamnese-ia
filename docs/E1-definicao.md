# E1 · Definição do projeto

## Motor de IA para feedback educacional de anamnese

**Disciplina:** Tópicos Avançados em Inteligência Artificial · 2026.2  
**Professor:** Marcondes Ricarte  
**Integrantes:** Leonardo Marcelo e André Lucas  
**Cliente indicada:** Marília Costa, docente de medicina  
**Versão revisada:** 03/11/2026 · **Grupo:** G04 · **Entrega E1:** 17/11/2026 (prazo prorrogado; original 11/11/2026), por Pull Request

## 1. Resumo da proposta

O projeto investiga o uso de um modelo de linguagem pré-treinado para identificar quais perguntas um estudante fez durante uma anamnese simulada. Esses registros deverão alimentar um feedback em português com acertos, pontos de melhoria e etapas investigativas não identificadas. A finalidade é educacional: a aplicação não oferecerá diagnóstico, prescrição ou recomendação de conduta sobre o paciente.

A fundamentação técnica é o MediTOD, conjunto de diálogos de coleta de história clínica anotados com o esquema CMAS. O projeto adapta esse esquema para uma tarefa própria: extrair atos de investigação das falas do entrevistador. O modelo escolhido é Qwen3-4B-Instruct-2507, com ajuste de adaptadores LoRA sobre pesos quantizados em quatro bits. A primeira execução será um experimento piloto local, com divisão por diálogo, baseline sem ajuste e avaliação no mesmo conjunto retido.

O desenvolvimento está organizado em quatro entregas. Esta entrega cobre a E1; a E2 está planejada nas seções 6 e 8. Áudio, interface, avaliação pedagógica com pessoas e demonstração pertencem ao planejamento das entregas seguintes.

## 2. Problema, cliente e evidência inicial

### 2.1 Contexto e público beneficiado

Marília Costa descreve um problema da preceptoria em ambulatório universitário: um docente precisa acompanhar vários estudantes e, ao mesmo tempo, atender à demanda assistencial. A observação da entrevista e o feedback individual dependem da disponibilidade do preceptor. O estudante pode terminar uma atividade sem saber quais partes da investigação deixou de realizar.

O usuário principal é o estudante de medicina que pratica anamnese. O preceptor é beneficiário indireto, pois poderá consultar um registro das perguntas e revisar o feedback antes de utilizá-lo no ensino. A proposta parte de uma cliente real indicada no documento fornecido; não pressupõe parceria institucional, contratação ou aceite formal da solução.

### 2.2 Processo atual e necessidade

| Processo atual | Limitação relevante | Benefício pretendido |
|---|---|---|
| Observação direta e comentários do preceptor | Dependem da presença e do tempo disponível | Organizar evidências da entrevista para facilitar a revisão |
| Autoavaliação do estudante | Pode divergir da avaliação externa | Mostrar trechos concretos que sustentam o retorno |
| Avaliações práticas em momentos específicos | Não acompanham necessariamente cada treino | Permitir revisão de entrevistas simuladas entre avaliações |

O problema específico deste projeto é a dificuldade de obter um registro frequente e verificável das etapas de investigação realizadas pelo estudante. A hipótese de benefício é que esse registro ajude a orientar a prática seguinte. Ganho de aprendizagem, redução da carga do preceptor e satisfação do usuário são objetivos futuros; não são resultados já demonstrados.

### 2.3 Evidência disponível e limites

A evidência inicial é o relato escrito “Motor de IA especializado em educação médica: o problema e as características da solução”, atribuído a Marília Costa e fornecido pelo usuário. Sua transcrição consta em `evidencias/relato_cliente.md`. O documento descreve a necessidade, o fluxo desejado e as qualidades esperadas do feedback. Trata-se de um relato documental, não de uma entrevista conduzida pela dupla nem de uma observação independente do ambulatório.

Haring et al. (2017) estudam manifestações observáveis do raciocínio clínico durante a anamnese; Spencer (2003) discute ensino no ambiente clínico. Cheung et al. (2019) e Alex et al. (2024) apresentam barreiras e experiências relacionadas à observação direta. Esses trabalhos ajudam a situar o problema, mas não medem o processo específico da cliente.

A literatura sobre autoavaliação precisa ser interpretada com cuidado. Blanch-Hartigan (2011) encontrou capacidade moderada de autoavaliação de estudantes, enquanto Davis et al. (2006) analisaram médicos. Não se deve transformar essas conclusões em uma afirmação universal de que todo estudante avalia mal o próprio desempenho. A proposta utiliza o feedback externo como complemento verificável à reflexão do aluno.

Não há entrevista agendada, consentimento assinado ou validação clínica comprovada nos materiais recebidos. Uma conversa com a cliente e a revisão do roteiro são atividades propostas para a continuidade do trabalho. Não constituem evidências já coletadas.

## 3. Solução e delimitação de escopo

### 3.1 Entrada, processamento e saída

Na E2, a entrada é textual: histórico recente e fala atual do entrevistador em um diálogo simulado, com identificação de falante fornecida pelo corpus. A saída experimental é uma lista estruturada de atos `inquire`, contendo categoria, valor e atributos investigados.

Para a aplicação planejada, a entrada será uma transcrição identificada por falante ou um arquivo de áudio de anamnese simulada. O áudio deverá passar por transcrição local e revisão dos falantes antes da análise. O Whisper, sozinho, não identifica de forma confiável quem é estudante ou paciente; a diarização não será presumida como funcionalidade pronta.

```text
E2: histórico + fala do entrevistador
    → Qwen sem ajuste ou com adaptador LoRA
    → atos de investigação em JSON
    → comparação com rótulos retidos e métricas

E3/E4, planejadas:
áudio → transcrição → revisão dos falantes e identificadores
    → extração → comparação com roteiro do caso
    → acertos, melhorias e etapas não identificadas
```

O feedback futuro deverá permitir consultar o trecho da entrevista associado a cada afirmação. A comparação com um roteiro será determinística, mas sua entrada vem de um modelo sujeito a erro. Portanto, uma comparação determinística não torna a conclusão infalível. Saídas inválidas, baixa confiança na transcrição ou falantes não revisados deverão impedir conclusões categóricas sobre omissões.

### 3.2 Exemplo ilustrativo, não resultado medido

Considere a pergunta “Quando a tosse começou?”. Uma saída esperada é:

```json
[
  {
    "slot": "symptom",
    "value": "cough",
    "attributes": ["when check"]
  }
]
```

Esse ato sustenta um retorno como: “Você investigou o início da tosse. Na próxima prática, revise os demais aspectos previstos no roteiro do caso.” Ele não permite afirmar que a anamnese está completa, que o paciente possui uma doença ou que um procedimento é indicado. A equivalência entre a pergunta em português e os rótulos em inglês também precisa ser avaliada; o exemplo apenas explica a estrutura.

### 3.3 Entregas previstas

| Etapa | Escopo |
|---|---|
| E1 | Problema e evidência documental; artigo; modelo; dados; plano de adaptação, avaliação e responsabilidades |
| E2 | Aquisição e preparo reproduzível de texto; partições; baseline; primeiro LoRA; logs, pesos locais e métricas observadas |
| E3, futura | Integração da extração ao fluxo de uso, rastreabilidade e resultados ampliados |
| E4, futura | Demonstração, defesa técnica e entrega final revisada pela dupla |

Não fazem parte do protótipo: diagnóstico, tratamento, apoio à decisão clínica, pacientes reais, treinamento do zero, integração com prontuários, implantação multiusuário ou validação de eficácia educacional em larga escala. O primeiro recorte é o de entrevistas simuladas predominantemente respiratórias.

## 4. Fundamentação científica

### 4.1 Artigo-base

**Título:** MediTOD: An English Dialogue Dataset for Medical History Taking with Comprehensive Annotations.  
**Autores:** Vishal Vivek Saley, Goonjan Saha, Rocktim Jyoti Das, Dinesh Raghu e Mausam.  
**Publicação:** EMNLP, 2024, p. 16843–16877.  
**DOI:** https://doi.org/10.18653/v1/2024.emnlp-main.936  
**Artigo e material:** https://aclanthology.org/2024.emnlp-main.936/  
**Repositório:** https://github.com/dair-iitd/MediTOD

O artigo apresenta diálogos de coleta de história clínica e o Comprehensive Medical Attribute Schema (CMAS). A anotação distingue atos como `inquire`, para perguntas, e `inform`, para informações, e associa categorias e atributos às falas. O esquema permite representar, por exemplo, que o entrevistador perguntou pelo início de um sintoma.

O artigo também avalia tarefas de compreensão, política de diálogo e geração. A NLU descrita no artigo se concentra na fala mais recente do paciente. O F1 0,7139 atribuído ao modelo Llama3-8B supervisionado corresponde à tarefa e ao protocolo dos autores. Nosso projeto avalia perguntas já proferidas pelo entrevistador, com outra divisão e outro tratamento de correspondência entre rótulos. Não é uma reprodução da NLU original e seus números não serão comparados diretamente aos nossos.

### 4.2 Componentes utilizados e adaptações

| Componente | Uso no projeto | Mudança ou limite |
|---|---|---|
| Esquema CMAS | Categorias e atributos da extração | Foco retrospectivo em `inquire` |
| Falas e anotações textuais abertas | Preparação local e experimento inicial | Licença das anotações não identificada; sem redistribuição no pacote |
| Ajuste supervisionado de modelo pré-treinado | Aprender a emitir estrutura a partir do contexto | Qwen 4B e LoRA em hardware local |
| Avaliação publicada | Referência de desenho experimental | Métricas próprias explícitas, sem alegar equivalência numérica |
| Política e geração de perguntas | Não utilizadas | O sistema não entrevista no lugar do aluno |
| Canonicalização UMLS | Não utilizada | Evitar dependência de acesso e licença UMLS |

O código de métricas original não será simplesmente copiado: ele inclui aspectos de canonicalização e correspondência que não equivalem à nossa avaliação local por conjuntos de rótulos. A correspondência metodológica está em usar contexto de diálogo, supervisão estruturada e um modelo pré-treinado adaptado, não em conservar a tarefa inteira do artigo.

### 4.3 Referências complementares

Hu et al. (2022) fundamentam LoRA: ajustar matrizes de baixa dimensão mantendo os pesos-base congelados. Dettmers et al. (2023) apresentam QLoRA e mostram a viabilidade de adaptação sobre modelos quantizados. A implementação MLX adotada utiliza LoRA sobre pesos em quatro bits; não se presume equivalência exata ao formato NF4 ou à receita original de QLoRA.

Van Veen et al. (2024) estudam adaptação de modelos para sumarização clínica e avaliação humana em dimensões distintas. Essa referência orienta a separação entre correção e qualidade da comunicação; não demonstra que LoRA será superior no nosso problema. Para a forma do feedback, Bastos e Castro et al. (2021) e van de Ridder et al. (2015) sustentam a necessidade de considerar conteúdo, contexto e forma de entrega.

## 5. Modelo pré-treinado e recursos

### 5.1 Modelo que será adaptado

| Campo | Definição |
|---|---|
| Nome e versão | Qwen3-4B-Instruct-2507 |
| Desenvolvedor e pesos originais | Qwen; https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Distribuição para execução local | https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit |
| Licença informada nas fichas | Apache-2.0 |
| Tarefa original | Geração de texto e seguimento de instruções multilíngue |
| Tarefa adaptada | Contexto e pergunta do entrevistador → lista de atos `inquire` |
| Parâmetros ajustados | Matrizes LoRA em parte das camadas; pesos-base congelados |
| Proveniência reproduzível | Revisão resolvida, arquivos e configuração registrados na E2 |

O modelo não é o Llama3-8B do artigo-base. A substituição reduz a demanda de memória e mantém a formulação de geração supervisionada de estrutura. O suporte multilíngue declarado pelo Qwen torna o português uma possibilidade a testar, mas não comprova qualidade nesse idioma após ajuste em inglês.

### 5.2 Ambiente e viabilidade

O ambiente disponível foi inspecionado em 03/11/2026: Apple M5, arquitetura arm64 e 16 GiB de memória. O experimento usa MLX/Metal e ambiente Python 3.12 isolado em `.venv/` na raiz deste repositório (ignorado pelo Git). As dependências do experimento serão registradas em `requirements.txt` na E2.

A configuração piloto limita tamanho de sequência, lote e número de camadas adaptadas. Tempo e memória serão medidos no log da execução. Não se presume que um modelo de 4B sempre caiba em 11 GB nem que a duração de treino seja fixa. Em caso de falta de memória, a ordem de redução é contexto, camadas adaptadas e tamanho da execução, registrando a configuração efetivamente usada.

### 5.3 Transcrição planejada

O componente de áudio previsto é Whisper large-v3-turbo, disponível em https://huggingface.co/openai/whisper-large-v3-turbo, cuja ficha consultada informa licença MIT. Ele não será ajustado na E2 e não compõe a avaliação deste experimento textual. Sua qualidade em português e seu custo no fluxo completo deverão ser medidos quando houver áudio de teste adequado.

## 6. Dados-base, origem e tratamento

### 6.1 Inventário

| Fonte | Conteúdo e acesso | Licença ou condição | Papel |
|---|---|---|---|
| MediTOD, `raw_data/dialogs.json` e `raw_data/annotations.json` | JSON público em repositório GitHub; 231 diálogos alinhados | Não foi identificado arquivo LICENSE nem declaração CC BY-SA para as anotações no artigo | Dados textuais do experimento local |
| Fareez et al. (2022), Figshare, item 16550013 | Coleção de entrevistas simuladas; arquivo Data.zip | CC0 informado na página/API do item | Fonte de áudio planejada; não necessária à primeira adaptação textual |
| Exemplos em português do projeto | Exemplos didáticos/sintéticos quando criados | Origem e finalidade devem acompanhar cada arquivo | Testes técnicos; não substituem validação humana |
| Futuras gravações da dupla | Ainda não coletadas | Exigem definição de consentimento e forma de compartilhamento | Avaliação futura de áudio e português |

Dados de Fareez: https://springernature.figshare.com/articles/dataset/Collection_of_simulated_medical_exams/16550013. Artigo: https://doi.org/10.1038/s41597-022-01423-1.

A licença CC0 dos diálogos de origem não determina, por si só, a licença das anotações adicionais do MediTOD. A ausência de licença não deve ser descrita como autorização automática para uso acadêmico ou redistribuição. Esta entrega mantém os arquivos de dados e os adaptadores derivados apenas no ambiente local; o pacote compartilhável contém código próprio, configurações e estatísticas agregadas. O compartilhamento de derivados requer esclarecer as condições com os titulares. Nenhum contato foi enviado em nome da dupla.

### 6.2 Formato e rótulos

Cada diálogo possui falas com identificador, texto e falante. As anotações possuem atos com `uid`, intenção, categoria, valor e atributos. A preparação deve casar identificadores, conferir cobertura do alinhamento e gerar uma amostra por fala do entrevistador, inclusive quando a resposta esperada for lista vazia.

Os rótulos utilizados são os atos `inquire`. Informações espontâneas do paciente não comprovam que uma pergunta específica foi realizada. Ao mesmo tempo, o sistema não deve punir automaticamente o estudante por não repetir informação já oferecida: a presença da informação e a evidência de investigação são conceitos distintos, a serem tratados no roteiro pedagógico futuro.

Os arquivos de divisão publicados no repositório contêm apenas um identificador em cada partição (`115`, `317` e `407`). Eles não recuperam a divisão completa dos experimentos do artigo. Por isso, o projeto utilizará divisão própria, determinística e documentada por listas de diálogo. Diálogos textualmente idênticos com identificadores diferentes devem permanecer no mesmo grupo, impedindo a repetição integral de uma entrevista entre treino, validação e teste.

### 6.3 Preparação planejada

1. Fixar revisão do repositório-fonte e calcular SHA-256 dos arquivos adquiridos.
2. Verificar alinhamento entre diálogo, falante, identificador de fala e anotação.
3. Normalizar variantes de grafia declaradas, sem inferir atributos não anotados.
4. Agrupar diálogos idênticos pelo conteúdo e separar os grupos em treino, validação e teste antes de gerar amostras por fala.
5. Construir histórico limitado, removendo qualquer rótulo-alvo da entrada.
6. Selecionar piloto de treino e amostra de avaliação por regra e semente registradas.
7. Exportar dados em formato compatível com `mlx-lm` e registrar contagens e exclusões.

A E2 detalha o procedimento realmente executado. Não será anunciada tradução integral para português nem revisão de vinte diálogos sem que essas atividades sejam realizadas. A avaliação inicial em inglês mede a adaptação ao corpus; a generalização ao público brasileiro permanece uma questão experimental.

## 7. Roteiro de cobertura e feedback previstos

O CMAS oferece sinais para avaliar perguntas sobre sintomas, antecedentes, medicações, hábitos, história familiar, ocupação, exposições e viagens. O roteiro de cada caso deverá escolher quais desses itens são pertinentes. A frequência de um item no corpus não o torna obrigatório em toda entrevista.

| Categoria CMAS | Interpretação educacional proposta |
|---|---|
| `symptom` e atributos | Caracterização de queixa e investigação de outros sintomas previstos no caso |
| `medical history` | Exploração de antecedentes |
| `medication` | Investigação de medicamentos relatados ou em uso |
| `habit` | Perguntas sobre hábitos pertinentes ao caso |
| `family history` | Exploração da história familiar |
| `occupation`, `exposure`, `travel` | Contexto ocupacional, ambiental e de deslocamentos quando pertinente |

Esse mapeamento é uma proposta computacional, ainda sem validação clínica da cliente. Alergias, vacinação e uma revisão completa de sistemas não estão adequadamente representadas como categorias próprias neste recorte; o sistema não poderá alegar avaliar a anamnese inteira.

Na aplicação futura, cada item poderá ser classificado como investigação identificada, investigação parcial, investigação não identificada, informação espontânea presente, não aplicável ou avaliação inconclusiva. Uma resposta inválida do modelo não pode ser convertida em “o estudante não perguntou”. Os três blocos pedagógicos deverão reconhecer condutas observáveis, descrever lacunas com cautela e indicar uma oportunidade de prática, sem julgamento pessoal do estudante.

## 8. Adaptação e avaliação

### 8.1 Hipótese e execução inicial

A hipótese técnica é que o ajuste de adaptadores melhore a emissão de atos de investigação em relação ao mesmo modelo sem ajuste. O experimento compara esses sistemas usando a mesma entrada, saída esperada e política de validação do JSON. A E2 exige uma primeira execução verificável, não superioridade garantida nem o melhor modelo possível.

O treino utiliza entropia cruzada sobre a saída supervisionada. Os hiperparâmetros e o número de passos efetivos constam no arquivo de configuração da E2. O conjunto de validação acompanha o treinamento; a avaliação de teste não orienta a escolha de hiperparâmetros nesta rodada. Experimentos posteriores devem registrar separadamente alterações motivadas pela inspeção do teste.

### 8.2 Baselines e métricas

| Sistema ou medida | Finalidade |
|---|---|
| Qwen original, sem adaptador | Baseline principal: isolar o efeito do primeiro ajuste |
| Regras simples, quando executadas | Referência adicional para explicitar limites da busca textual |
| Qwen com LoRA | Sistema adaptado, avaliado nas mesmas amostras |
| Precisão, recall e F1 micro dos rótulos | Medir falsos positivos e falsos negativos da extração |
| Validade estrutural de JSON | Medir quantas respostas seguem o contrato sem correção oportunista |
| Perda de treino/validação | Evidenciar a execução; não confundir perda menor com melhor produto |
| Latência e memória | Verificar viabilidade no equipamento utilizado |

A definição exata das unidades comparadas e do tratamento de saídas inválidas acompanhará os resultados da E2. As métricas de omissão, qualidade pedagógica, WER e eficácia de aprendizagem não serão inferidas do F1 de extração. Dependem de outros conjuntos de dados e protocolos.

### 8.3 Avaliação futura do produto

Para omissões, será necessário um roteiro por caso e referência anotada independentemente. Devem ser reportados tanto recall de omissão quanto precisão: deixar de apontar uma lacuna e acusar uma lacuna inexistente são falhas diferentes. Para feedback, propõe-se rubrica de especificidade, correção e tom pedagógico, com avaliações independentes e registro de divergências. As notas só poderão ser publicadas depois da avaliação por pessoas.

Não há limiar clínico validado para aceitar este protótipo. A primeira meta é executar e medir corretamente. Para avançar ao uso educacional com a cliente, será preciso examinar erros, idioma, rastreabilidade e adequação pedagógica, mesmo que a métrica agregada melhore.

## 9. Organização, riscos e cronograma

### 9.1 Divisão proposta de atividades

| Frente | Responsável proposto | Revisão proposta |
|---|---|---|
| Dados, ambiente e ajuste LoRA | Leonardo Marcelo | André Lucas |
| Roteiro pedagógico e protocolo de avaliação | André Lucas | Leonardo Marcelo |
| Integração e rastreabilidade | Leonardo Marcelo | André Lucas |
| Análise de erros, documentação e demonstração | Ambos | Revisão cruzada |
| Conversa com cliente e defesa técnica | Ambos | Participação a confirmar na execução |

A tabela é uma proposta de trabalho. Não certifica participação já realizada. Os integrantes devem revisar o material e conseguir explicar e reproduzir os experimentos que apresentarão.

### 9.2 Riscos e resposta prevista

| Risco | Consequência | Resposta |
|---|---|---|
| Licença das anotações não esclarecida | Restrição ao compartilhamento de derivados | Manter dados e adaptadores locais; registrar a fonte e esclarecer condições antes de redistribuir |
| Piloto pequeno e desbalanceado | Métrica instável e pouco representativa | Publicar contagens, composição e exemplos de erro; ampliar avaliação depois |
| Ajuste em inglês | Desempenho em português desconhecido | Avaliar corpus próprio revisado em etapa posterior |
| Falha de extração ou transcrição | Feedback incorreto | Evidências por trecho e estado inconclusivo na aplicação futura |
| Memória insuficiente | Interrupção de treino | Limitar contexto, lote e camadas; registrar configuração final |
| Roteiro sem revisão especializada | Omissões pedagógicas indevidas | Identificar como proposta e revisar com docente antes do uso educacional |

### 9.3 Datas do enunciado

| Entrega | Versionamento | Encontro |
|---|---|---|
| E1 · Definição | 17/11/2026 (prorrogado; original 11/11/2026) | Não previsto |
| E2 · Modelo e dados | 22/11/2026 | Alinhamento em 23/11/2026 |
| E3 · Aplicação | 29/11/2026 | Alinhamento em 30/11/2026 |
| E4 · Apresentação | 07/12/2026 | Apresentação em 07/12/2026 |

O enunciado exige Pull Requests para avaliação. A organização dos arquivos nesta máquina não equivale a envio, aceite ou apresentação. Os textos de submissão e as instruções de reprodução acompanham o material, mas o estado de publicação precisa ser conferido separadamente.

## 10. Matriz de atendimento da E1

| Exigência | Evidência nesta entrega |
|---|---|
| Problema, beneficiário e evidência inicial | Seção 2 e relato documental da cliente |
| Entrada, saída e escopo | Seção 3 |
| Artigo com identificação e método | Seção 4 |
| Modelo, versão, pesos, licença e relação com artigo | Seção 5 |
| Dados, acesso, formato, rótulos, volume e condições | Seção 6; manifesto e inventário na E2 |
| Adaptação, recursos, baseline e métricas | Seções 5 e 8 |
| Riscos, atividades e viabilidade | Seção 9 e primeira execução registrada na E2 |

## 11. Referências

Saley, V. V.; Saha, G.; Das, R. J.; Raghu, D.; Mausam. (2024). MediTOD: An English Dialogue Dataset for Medical History Taking with Comprehensive Annotations. EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936

Fareez, F. et al. (2022). A dataset of simulated patient-physician medical interviews with a focus on respiratory cases. Scientific Data, 9, 313. https://doi.org/10.1038/s41597-022-01423-1

Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. https://arxiv.org/abs/2106.09685

Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314

Qwen Team. (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388. Ficha da variante utilizada: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

Van Veen, D. et al. (2024). Adapted large language models can outperform medical experts in clinical text summarization. Nature Medicine, 30, 1134–1142. https://doi.org/10.1038/s41591-024-02855-5

Haring, C. M. et al. (2017). Observable phenomena that reveal medical students' clinical reasoning ability during expert assessment of their history taking: a qualitative study. BMC Medical Education, 17, 147. https://doi.org/10.1186/s12909-017-0983-3

Spencer, J. (2003). Learning and teaching in the clinical environment. BMJ, 326, 591–594. https://doi.org/10.1136/bmj.326.7389.591

Cheung, W. J. et al. (2019). Barriers and Enablers to Direct Observation of Trainees' Clinical Performance: A Qualitative Study Using the Theoretical Domains Framework. Academic Medicine, 94, 101–114. https://doi.org/10.1097/ACM.0000000000002396

Alex, C. P. et al. (2024). Exploring Medical Student Experiences With Direct Observation During the Pediatric Clerkship. Academic Medicine, 99, 997–1006. https://doi.org/10.1097/ACM.0000000000005747

Blanch-Hartigan, D. (2011). Medical students' self-assessment of performance: Results from three meta-analyses. Patient Education and Counseling, 84, 3–9. https://doi.org/10.1016/j.pec.2010.06.037

Davis, D. A. et al. (2006). Accuracy of Physician Self-assessment Compared With Observed Measures of Competence. JAMA, 296, 1094–1102. https://doi.org/10.1001/jama.296.9.1094

Bastos e Castro, M. A. et al. (2021). The Use of Feedback in Improving the Knowledge, Attitudes and Skills of Medical Students: a Systematic Review and Meta-analysis of Randomized Controlled Trials. Medical Science Educator, 31, 2093–2104. https://doi.org/10.1007/s40670-021-01443-3

van de Ridder, J. M. M. et al. (2015). Variables that affect the process and outcome of feedback, relevant for medical training: a meta-review. Medical Education, 49, 658–673. https://doi.org/10.1111/medu.12744

Gordon, M. et al. (2024). A scoping review of artificial intelligence in medical education: BEME Guide No. 84. Medical Teacher, 46, 446–470. https://doi.org/10.1080/0142159X.2024.2314198

Varughese, E. et al. (2026). Medical Students' Perceptions of Artificial Intelligence Use in Medical Practice: A Systematic Review. Medical Science Educator, 36, 1003–1014. https://doi.org/10.1007/s40670-025-02630-2

Costa, M. (2026). Motor de IA especializado em educação médica: o problema e as características da solução. Relato documental fornecido para o projeto. Bibliografia completa e metadados: `referencias.bib`, na pasta principal do projeto.
