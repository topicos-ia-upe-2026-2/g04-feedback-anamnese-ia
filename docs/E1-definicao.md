# E1 · Definição do projeto

## Motor de IA para feedback educacional de anamnese

**Disciplina:** Tópicos Avançados em Inteligência Artificial · 2026.2  
**Professor:** Marcondes Ricarte  
**Integrantes:** Leonardo Marcelo e André Lucas  
**Cliente indicada:** Marília Costa, docente de medicina  
**Versão revisada:** 15/11/2026 · **Grupo:** G04 · **Entrega E1:** 17/11/2026 (prazo prorrogado; original 11/11/2026), por Pull Request

## Resumo da proposta

O projeto recebe o áudio da anamnese que um estudante de medicina conduz com um paciente real no ambulatório e devolve ao estudante um feedback sobre cada interação e pergunta da entrevista, fechado em três blocos: o que ele acertou, o que errou ou precisa melhorar e o que deixou de fazer. A finalidade é educacional: a aplicação não oferecerá diagnóstico, hipótese diagnóstica, prescrição ou recomendação de conduta sobre o paciente.

O áudio é transcrito localmente pelo Whisper, sem ajuste. O componente adaptado é o modelo de linguagem Qwen3-4B-Instruct-2507, ajustado com adaptadores LoRA sobre pesos quantizados em quatro bits para identificar o que cada pergunta do estudante investigou. A fundamentação técnica é o MediTOD, conjunto de diálogos de coleta de história clínica anotados com o esquema CMAS. Como os dados públicos estão em inglês, a hipótese para os próximos passos é tratar a base com traduções para o português.

O documento segue os seis tópicos pedidos na E1: problema e beneficiário (seção 1), solução e escopo (seção 2), artigo-base (seção 3), modelo pré-treinado (seção 4), conjunto de dados-base (seção 5) e plano de adaptação e avaliação (seção 6). A seção 7 relaciona o projeto ao critério de aceite.

## 1. Problema e beneficiário

### 1.1 Cliente e contexto

Marília Costa é docente de medicina. Na disciplina de semiologia, os estudantes vão ao ambulatório universitário atender pacientes reais e fazem a anamnese: a entrevista em que, a partir da queixa principal, investigam a história do paciente. Segundo a cliente, é uma habilidade investigativa, desenvolvida com prática e retorno. O exame físico, que vem depois da anamnese, não faz parte deste projeto. A cliente é real; a proposta não pressupõe parceria institucional, contratação ou aceite formal da solução.

### 1.2 Necessidade

O ideal, segundo a cliente, é o professor acompanhar a entrevista e dizer ao estudante o que ele acertou, o que errou e o que deixou de fazer, para que ele corrija na próxima anamnese. Isso não acontece de forma adequada: um professor responde por mais de vinte alunos, além da rotina do consultório, e não consegue ver o que cada um faz nem dar esse retorno individual. Quando o retorno existe, é verbal, pouco consistente e tardio.

| Processo atual | Limitação relevante | Benefício pretendido |
|---|---|---|
| Professor observa e comenta a entrevista | Um professor para mais de vinte alunos; depende da presença e do tempo | Feedback para cada anamnese gravada, não só para as que o professor acompanha |
| Autoavaliação do estudante | Pode divergir da avaliação externa | Mostrar os trechos da entrevista que sustentam cada observação |
| Avaliações práticas pontuais | Não acompanham cada atendimento | Retorno logo após a anamnese, a tempo da próxima |

O problema específico deste projeto é que o estudante termina a anamnese sem um retorno individual e verificável sobre o que investigou e o que deixou de investigar. A hipótese de benefício é que esse retorno ajude a melhorar a entrevista seguinte. Ganho de aprendizagem e redução da carga do professor são objetivos; não são resultados já demonstrados.

### 1.3 Perfil do usuário

O usuário principal é o estudante de medicina em formação clínica, cursando semiologia, que conduz anamneses com pacientes reais no ambulatório e precisa saber o que investigou bem e o que deixou de investigar. O professor (preceptor) é beneficiário indireto, pois poderá consultar o registro e revisar o feedback antes de utilizá-lo no ensino.

### 1.4 Evidência inicial

A evidência inicial é uma entrevista com a cliente, feita por mensagens de voz, complementada por um vídeo explicativo enviado por ela; ambos estão registrados em `evidencias/entrevista_cliente.md`. A cliente descreve o cenário (ambulatório, paciente real, disciplina de semiologia), o feedback esperado (acertos, erros e o que deixou de fazer), a causa do problema (um professor para mais de vinte alunos, com retorno pouco consistente e tardio) e a solução que imagina: captar o áudio da entrevista, transformá-lo em texto e analisá-lo com os conhecimentos de anamnese.

Haring et al. (2017) estudam manifestações observáveis do raciocínio clínico durante a anamnese; Spencer (2003) discute ensino no ambiente clínico. Cheung et al. (2019) e Alex et al. (2024) apresentam barreiras e experiências relacionadas à observação direta, e Bastos e Castro et al. (2021) reúnem evidências do efeito do feedback no aprendizado de estudantes de medicina. Gordon et al. (2024) mapeiam aplicações de IA na educação médica, e Varughese et al. (2026) revisam a percepção de estudantes sobre IA na prática médica. Esses trabalhos ajudam a situar o problema, mas não medem o processo específico da cliente.

A literatura sobre autoavaliação precisa ser interpretada com cuidado. Blanch-Hartigan (2011) encontrou capacidade moderada de autoavaliação de estudantes, enquanto Davis et al. (2006) analisaram médicos. Não se deve transformar essas conclusões em uma afirmação universal de que todo estudante avalia mal o próprio desempenho. A proposta utiliza o feedback externo como complemento verificável à reflexão do aluno.

A entrevista representa a visão de uma docente. A dupla não observou o ambulatório, e o roteiro de itens que o feedback vai cobrar ainda precisa ser validado com a cliente.

## 2. Solução e escopo

### 2.1 Por que esta solução

A escolha da solução se apoia nas três bases que a E1 pede. A **necessidade da cliente** é que o estudante saiba o que acertou, errou e deixou de fazer, num contexto em que o professor não acompanha todos; por isso a solução gera feedback automático para cada anamnese, revisável pelo professor. A **evidência do processo atual** mostra que a anamnese é uma entrevista investigativa, conduzida por perguntas, que pode ser gravada; por isso a entrada é o áudio e a unidade de análise é cada pergunta do estudante. O **artigo-base** mostra que as perguntas de uma anamnese podem ser representadas por atos `inquire` com categoria e atributos (CMAS) e que um LLM ajustado por supervisão aprende a extraí-los; por isso "o que cada pergunta investigou" se torna uma tarefa de extração mensurável, e o feedback resulta da comparação com um roteiro, não de texto livre gerado pelo modelo.

### 2.2 Entrada

Gravação de áudio da anamnese conduzida pelo estudante com um paciente real, restrita à entrevista, sem o exame físico.

### 2.3 Saída

Feedback ao estudante, em português, em duas partes. A primeira avalia cada interação: para cada pergunta do estudante, apresenta o trecho transcrito, o que ela investigou e como ficou cada item do roteiro tocado por ela. A segunda resume a anamnese nos três blocos pedidos pela cliente: o que o estudante acertou (itens do roteiro investigados), o que errou ou precisa melhorar (itens abordados sem a caracterização esperada, como perguntar pela tosse sem investigar início, duração ou fatores de piora) e o que deixou de fazer (itens do roteiro sem nenhuma pergunta identificada).

Cada afirmação do feedback aponta o trecho da entrevista que a sustenta. A comparação com o roteiro é determinística, mas sua entrada vem de um modelo sujeito a erro. Portanto, saídas inválidas, baixa confiança na transcrição ou falantes não revisados geram "avaliação inconclusiva", e nunca uma conclusão categórica de que o estudante não perguntou.

### 2.4 Fluxo de uso

```text
áudio da anamnese (estudante + paciente real)
    → transcrição local (Whisper, sem ajuste)
    → separação das falas e revisão de quem falou
    → Qwen3-4B + LoRA: o que cada pergunta investigou, em JSON (CMAS)
    → comparação determinística com o roteiro de anamnese do caso
    → feedback por interação + acertos, a melhorar, deixou de fazer
```

O estudante grava a anamnese com consentimento do paciente. O Whisper, sozinho, não identifica de forma confiável quem é estudante ou paciente; por isso, o estudante ou o professor confirma a separação das falas antes da análise. O professor pode revisar o feedback antes de liberá-lo ao estudante.

### 2.5 Do rótulo ao feedback

O roteiro de cada caso segue a anamnese ensinada em semiologia e escolhe quais itens do CMAS são pertinentes. A frequência de um item no corpus não o torna obrigatório em toda entrevista. O mapeamento abaixo é uma proposta da dupla, ainda sem validação da cliente.

| Categoria CMAS | Item do roteiro de anamnese |
|---|---|
| `symptom` e atributos | Queixa principal, caracterização dos sintomas e outros sintomas previstos no caso |
| `medical history` | Antecedentes pessoais |
| `medication` | Medicamentos relatados ou em uso |
| `habit` | Hábitos pertinentes ao caso |
| `family history` | História familiar |
| `occupation`, `exposure`, `travel` | Contexto ocupacional, ambiental e de deslocamentos quando pertinente |

Cada item do roteiro poderá ser classificado como investigação identificada, investigação parcial, investigação não identificada, informação espontânea presente, não aplicável ou avaliação inconclusiva. Alergias, vacinação e uma revisão completa de sistemas não têm categoria própria no CMAS; o sistema não poderá alegar avaliar a anamnese inteira.

Considere a pergunta do estudante “Quando a tosse começou?”. Uma saída esperada do extrator é:

```json
[
  {
    "slot": "symptom",
    "value": "cough",
    "attributes": ["when check"]
  }
]
```

O feedback dessa interação poderia ser: “Nesta pergunta você investigou o início da tosse. O roteiro do caso também prevê duração e fatores de piora, que não apareceram na entrevista.” Os rótulos ficam no vocabulário do CMAS, em inglês, e o feedback os traduz para os termos do roteiro em português. O exemplo é ilustrativo, não resultado medido, e não permite afirmar que a anamnese está completa, que o paciente possui uma doença ou que um procedimento é indicado.

### 2.6 O que será e o que não será implementado

| Será implementado até a apresentação | Não será implementado |
|---|---|
| Preparo dos dados do MediTOD, baseline e ajuste LoRA do extrator de perguntas | Avaliação do exame físico |
| Versão em português dos dados por tradução, para medir e reduzir a perda no idioma (hipótese) | Diagnóstico, hipótese diagnóstica, tratamento ou apoio à decisão clínica |
| Transcrição local do áudio (Whisper, sem ajuste) com revisão de quem falou | Separação automática e confiável de falantes (diarização) |
| Extração do que cada pergunta investigou e comparação com um roteiro de anamnese | Treinamento do zero ou ajuste do Whisper |
| Feedback por interação e resumo (acertos, a melhorar, deixou de fazer), com trechos | Integração com prontuário, implantação multiusuário ou validação de eficácia educacional em larga escala |
| Demonstração do fluxo completo com uma anamnese simulada | Painel da turma para o professor e histórico do aluno ao longo do curso |

O desenvolvimento e a demonstração usam áudios de entrevistas simuladas (Fareez et al., 2022, CC0), para não expor pacientes. Gravações com pacientes reais só entram com consentimento do paciente e do estudante e autorização do ambulatório. O primeiro recorte é o de queixas predominantemente respiratórias, perfil do corpus.

## 3. Artigo-base

### 3.1 Identificação

**Título:** MediTOD: An English Dialogue Dataset for Medical History Taking with Comprehensive Annotations.  
**Autores:** Vishal Vivek Saley, Goonjan Saha, Rocktim Jyoti Das, Dinesh Raghu e Mausam.  
**Ano e publicação:** 2024, EMNLP, p. 16843–16877.  
**DOI:** https://doi.org/10.18653/v1/2024.emnlp-main.936  
**Artigo e material:** https://aclanthology.org/2024.emnlp-main.936/  
**Repositório:** https://github.com/dair-iitd/MediTOD

### 3.2 Método

O artigo apresenta diálogos de coleta de história clínica e o Comprehensive Medical Attribute Schema (CMAS). A anotação distingue atos como `inquire`, para perguntas, e `inform`, para informações, e associa categorias e atributos às falas. O esquema permite representar, por exemplo, que o entrevistador perguntou pelo início de um sintoma.

O artigo também avalia tarefas de compreensão, política de diálogo e geração, com modelos ajustados por supervisão. A NLU descrita no artigo se concentra na fala mais recente do paciente. O F1 0,7139 atribuído ao modelo Llama3-8B supervisionado corresponde à tarefa e ao protocolo dos autores. Nosso projeto avalia perguntas já proferidas pelo entrevistador, com outra divisão e outro tratamento de correspondência entre rótulos. Não é uma reprodução da NLU original e seus números não serão comparados diretamente aos nossos.

### 3.3 Componentes utilizados ou adaptados

| Componente | Uso no projeto | Mudança ou limite |
|---|---|---|
| Esquema CMAS | Categorias e atributos do que cada pergunta investigou | Foco retrospectivo em `inquire` |
| Falas e anotações textuais abertas | Preparação local e ajuste do extrator | Licença das anotações não identificada; sem redistribuição |
| Ajuste supervisionado de modelo pré-treinado | Aprender a emitir estrutura a partir do contexto | Qwen 4B e LoRA em hardware local |
| Avaliação publicada | Referência de desenho experimental | Métricas próprias explícitas, sem alegar equivalência numérica |
| Política e geração de perguntas | Não utilizadas | O sistema não entrevista no lugar do aluno |
| Canonicalização UMLS | Não utilizada | Evitar dependência de acesso e licença UMLS |

O código de métricas original não será simplesmente copiado: ele inclui aspectos de canonicalização e correspondência que não equivalem à nossa avaliação local por conjuntos de rótulos.

Hu et al. (2022) fundamentam LoRA: ajustar matrizes de baixa dimensão mantendo os pesos-base congelados. Dettmers et al. (2023) apresentam QLoRA e mostram a viabilidade de adaptação sobre modelos quantizados. A implementação MLX adotada utiliza LoRA sobre pesos em quatro bits; não se presume equivalência exata ao formato NF4 ou à receita original de QLoRA. Van Veen et al. (2024) estudam adaptação de modelos para sumarização clínica e avaliação humana em dimensões distintas; essa referência orienta a separação entre correção e qualidade da comunicação. Para a forma do feedback, Bastos e Castro et al. (2021) e van de Ridder et al. (2015) sustentam a necessidade de considerar conteúdo, contexto e forma de entrega.

## 4. Modelo pré-treinado

### 4.1 Modelo que será adaptado

| Campo | Definição |
|---|---|
| Nome e versão | Qwen3-4B-Instruct-2507 (revisão `cdbee75`) |
| Origem dos pesos | Qwen; https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Distribuição para execução local | https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit (revisão `50d4277`) |
| Licença | Apache-2.0 |
| Tarefa original | Geração de texto e seguimento de instruções multilíngue |
| Tarefa adaptada | Contexto e pergunta do estudante → lista de atos `inquire` em JSON |
| Parâmetros ajustados | Matrizes LoRA em parte das camadas; pesos-base congelados |

### 4.2 Relação com o artigo e correspondência metodológica

O modelo não é o Llama3-8B do artigo-base. A substituição reduz a demanda de memória para caber em 16 GiB de memória unificada. A correspondência metodológica está mantida: um LLM pré-treinado, decoder-only e ajustado por instrução, recebe o contexto do diálogo e é ajustado por supervisão para gerar a estrutura CMAS. O suporte multilíngue declarado pelo Qwen é o que torna viável a hipótese do português (seção 5.2), sem comprovar qualidade nesse idioma. Se o professor pedir redução de escopo, o plano B é o Qwen3-1.7B com a mesma receita.

### 4.3 Componente de transcrição

O componente de transcrição é o Whisper large-v3-turbo, disponível em https://huggingface.co/openai/whisper-large-v3-turbo, cuja ficha consultada informa licença MIT e cuja tarefa original é o reconhecimento de fala multilíngue (Radford et al., 2023). Ele é usado sem ajuste e não faz parte da avaliação do extrator; sua qualidade na fala do ambulatório deverá ser medida em áudio de teste.

## 5. Conjunto de dados-base

### 5.1 Fonte, acesso, licença, formato, volume e rótulos

A fonte principal é o repositório público do MediTOD (https://github.com/dair-iitd/MediTOD), arquivos `raw_data/dialogs.json` e `raw_data/annotations.json`, revisão `0b7a87c`. Cada diálogo possui falas com identificador, texto e falante; as anotações possuem atos com `uid`, intenção, categoria, valor e atributos. Na medição de 12/11/2026 há 231 diálogos, 22.317 falas (11.097 do médico) e 10.082 atos `inquire` em 13 categorias, com predomínio de `symptom` (5.527).

Os áudios vêm de Fareez et al. (2022), Figshare, item 16550013 (https://springernature.figshare.com/articles/dataset/Collection_of_simulated_medical_exams/16550013): 272 entrevistas simuladas no formato OSCE, em inglês, com áudio mp3 e transcrição em texto, em arquivo único Data.zip de cerca de 1,03 GB, licença CC0. A maioria dos casos é respiratória.

O repositório do MediTOD não tem arquivo LICENSE, e não foi identificada declaração de licença para as anotações. A licença CC0 dos diálogos de origem não determina, por si só, a licença das anotações adicionais. A ausência de licença não deve ser descrita como autorização automática para uso acadêmico ou redistribuição. Por isso, dados, traduções e adaptadores derivados ficam apenas no ambiente local, e a dupla vai pedir autorização aos autores antes de qualquer redistribuição.

Os rótulos utilizados são os atos `inquire`, que marcam o que o entrevistador perguntou: o sinal de que o feedback por interação precisa. Informações espontâneas do paciente não comprovam que uma pergunta específica foi realizada. Ao mesmo tempo, o sistema não deve punir automaticamente o estudante por não repetir informação já oferecida: a presença da informação e a evidência de investigação são conceitos distintos, tratados no roteiro de anamnese.

### 5.2 Representatividade do cenário e tradução para o português

Os dados públicos cobrem bem a tarefa, com perguntas de anamnese e rótulos finos, mas não o idioma nem o contexto da cliente: são entrevistas simuladas, em inglês, de estudantes canadenses. Como hipótese para os próximos passos, a base será tratada com traduções para o português. As falas dos diálogos do MediTOD serão traduzidas preservando o identificador e o alinhamento de cada fala; os rótulos continuam no vocabulário do CMAS, em inglês, e por isso a anotação vale para as duas versões. A tradução da partição de teste servirá para medir quanto o extrator perde do inglês para o português nas mesmas amostras, e a tradução do treino servirá para testar se incluí-la no ajuste reduz essa perda. Uma amostra das traduções será revisada manualmente, com atenção à terminologia médica e aos atributos investigados, e o método de tradução e a taxa de revisão serão registrados.

Texto traduzido não reproduz a fala espontânea de um ambulatório brasileiro, com coloquialismos, variação regional e ruído. A tradução aproxima o cenário, mas não o substitui: a validação final pede gravações reais em português, quando autorizadas.

| Fonte | Conteúdo e acesso | Licença ou condição | Papel |
|---|---|---|---|
| MediTOD, `raw_data/` | JSON público em repositório GitHub; 231 diálogos alinhados, 10.082 atos `inquire` | Sem LICENSE identificado | Treino e teste do extrator, em inglês |
| Tradução do MediTOD para o português | Gerada pela dupla, localmente | Derivada do MediTOD: não redistribuída | Medir e reduzir a perda em português (hipótese) |
| Fareez et al. (2022) | 272 entrevistas simuladas com áudio mp3 e transcrição | CC0 | Testar e demonstrar o fluxo a partir do áudio sem expor pacientes |
| Gravações de anamneses com pacientes reais | Ainda não coletadas | Consentimento do paciente e do estudante, autorização do ambulatório e anonimização | Validar o fluxo no cenário da cliente, quando autorizadas |

### 5.3 Preparação e divisão

Os arquivos de divisão publicados no repositório contêm apenas um identificador em cada partição (`115`, `317` e `407`). Eles não recuperam a divisão completa dos experimentos do artigo. Por isso, o projeto utilizará divisão própria, determinística e documentada por listas de diálogo. Há 18 pares de diálogos textualmente idênticos com identificadores diferentes; eles devem permanecer no mesmo grupo, impedindo a repetição integral de uma entrevista entre treino, validação e teste. Como os diálogos do MediTOD vêm das entrevistas do Fareez, os áudios usados na demonstração e na avaliação do fluxo serão só os da partição de teste; a correspondência entre os identificadores das duas bases será conferida na preparação.

1. Fixar revisão do repositório-fonte e calcular SHA-256 dos arquivos adquiridos.
2. Verificar alinhamento entre diálogo, falante, identificador de fala e anotação.
3. Normalizar variantes de grafia declaradas, sem inferir atributos não anotados.
4. Agrupar diálogos idênticos pelo conteúdo e separar os grupos em treino, validação e teste antes de gerar amostras por fala.
5. Construir histórico limitado, removendo qualquer rótulo-alvo da entrada.
6. Selecionar piloto de treino e amostra de avaliação por regra e semente registradas.
7. Gerar a versão traduzida das partições e revisar uma amostra.
8. Exportar dados em formato compatível com `mlx-lm` e registrar contagens e exclusões.

## 6. Plano de adaptação e avaliação

### 6.1 O que será ajustado

A hipótese técnica é que o ajuste de adaptadores melhore a identificação do que cada pergunta investigou em relação ao mesmo modelo sem ajuste, com a mesma entrada, saída esperada e política de validação do JSON. Serão ajustados adaptadores LoRA em parte das camadas do Qwen3-4B-Instruct-2507, sobre pesos quantizados em quatro bits, com os pesos-base congelados. O treino utiliza entropia cruzada sobre a saída supervisionada, primeiro com o treino em inglês e, se a hipótese do português se confirmar, com inglês e tradução. Os hiperparâmetros e o número de passos ficam num arquivo de configuração versionado junto com o código de treino. O Whisper não é ajustado.

### 6.2 Recursos computacionais

O ambiente disponível foi inspecionado em 03/11/2026: Apple M5, arquitetura arm64 e 16 GiB de memória unificada, com MLX/Metal e Python 3.12 isolado em `.venv/` (ignorado pelo Git). Não há GPU em nuvem nem serviço pago. A configuração piloto limita tamanho de sequência, lote e número de camadas adaptadas. Tempo e memória serão medidos no log da execução; não se presume que um modelo de 4B sempre caiba na memória disponível nem que a duração de treino seja fixa. Em caso de falta de memória, a ordem de redução é contexto, camadas adaptadas e tamanho da execução, registrando a configuração efetivamente usada.

### 6.3 Baseline e métricas

| Sistema ou medida | Finalidade |
|---|---|
| Qwen original, sem adaptador | Baseline principal: isolar o efeito do ajuste |
| Regras simples, quando executadas | Referência adicional para explicitar limites da busca textual |
| Qwen com LoRA | Sistema adaptado, avaliado nas mesmas amostras |
| Precisão, recall e F1 micro dos rótulos, com F1 por categoria | Medir falsos positivos e falsos negativos da extração sem esconder categorias raras |
| As mesmas métricas no teste traduzido para o português | Medir quanto o extrator perde no idioma da cliente |
| Validade estrutural de JSON | Medir quantas respostas seguem o contrato sem correção oportunista |
| Perda de treino/validação | Evidenciar a execução; não confundir perda menor com melhor produto |
| Latência e memória | Verificar viabilidade no equipamento utilizado |

O conjunto de validação acompanha o treinamento; a avaliação de teste não orienta a escolha de hiperparâmetros, e alterações motivadas pela inspeção do teste devem ser registradas separadamente. As saídas inválidas são contabilizadas como erro e reportadas à parte. Qualidade da transcrição (WER), acerto das omissões apontadas, qualidade pedagógica e eficácia de aprendizagem não serão inferidos do F1 de extração. Para omissões, será necessário um roteiro por caso e referência anotada independentemente, reportando tanto recall quanto precisão: deixar de apontar uma lacuna e acusar uma lacuna inexistente são falhas diferentes. Para o feedback, propõe-se rubrica de especificidade, correção e tom pedagógico, com avaliações independentes e registro de divergências.

### 6.4 Riscos

| Risco | Consequência | Resposta |
|---|---|---|
| Gravação de paciente real sem as autorizações | Exposição de dado de saúde | Desenvolver com áudio simulado; gravação real só com consentimento e autorização; processamento local; áudio fora do repositório |
| Licença das anotações não esclarecida | Restrição ao compartilhamento de derivados | Manter dados, traduções e adaptadores locais; pedir autorização aos autores antes de redistribuir |
| Ajuste em inglês, uso em português | Desempenho em português desconhecido | Medir no teste traduzido; incluir a tradução no treino se a perda for grande |
| Tradução altera o sentido clínico ou os atributos | Rótulos deixam de corresponder ao texto | Revisão manual por amostra, com atenção à terminologia; descartar falas inconsistentes |
| Texto traduzido não representa a fala real | Resultado otimista em relação ao ambulatório | Declarar o limite; validar com gravações reais autorizadas |
| Falha de transcrição ou de extração | Feedback incorreto | Revisão de falantes, evidências por trecho e estado inconclusivo |
| Piloto pequeno e desbalanceado | Métrica instável e pouco representativa | Publicar contagens, composição e exemplos de erro; ampliar avaliação depois |
| Memória insuficiente | Interrupção de treino | Limitar contexto, lote e camadas; plano B Qwen3-1.7B |
| Roteiro sem revisão especializada | Omissões apontadas indevidamente | Identificar como proposta e revisar com a cliente antes do uso educacional |

### 6.5 Divisão das atividades

| Frente | Responsável proposto | Revisão proposta |
|---|---|---|
| Dados, ambiente e ajuste LoRA | Leonardo Marcelo | André Lucas |
| Tradução para o português e revisão da amostra | Ambos | Revisão cruzada |
| Roteiro de anamnese e protocolo de avaliação do feedback | André Lucas | Leonardo Marcelo |
| Transcrição, integração e rastreabilidade | Leonardo Marcelo | André Lucas |
| Análise de erros, documentação e demonstração | Ambos | Revisão cruzada |
| Validação do roteiro com a cliente e defesa técnica | Ambos | Revisão cruzada |

Os integrantes devem conseguir explicar e reproduzir os experimentos que apresentarão.

## 7. Critério de aceite

| Critério | Como o projeto atende |
|---|---|
| Problema específico | Retorno individual e verificável sobre o que o estudante investigou na anamnese com paciente real (seção 1.2) |
| Método fundamentado | Esquema CMAS e ajuste supervisionado do MediTOD; LoRA (Hu et al., 2022) sobre modelo quantizado (Dettmers et al., 2023) (seção 3) |
| Pesos acessíveis | Qwen3-4B-Instruct-2507 (Apache-2.0) e Whisper large-v3-turbo (MIT), públicos no Hugging Face (seção 4) |
| Caminho viável de preparação dos dados | MediTOD público com 231/231 diálogos alinhados, divisão por grupo, tradução para o português e 8 passos de preparo (seção 5) |
| Caminho viável de treinamento | LoRA em quatro bits num Apple M5 de 16 GiB, com ordem de redução e plano B de 1,7B (seção 6) |
| Caminho viável de demonstração | Áudio simulado do Fareez (partição de teste) → transcrição → extração → feedback por interação (seções 2 e 5.3) |

## Referências

Saley, V. V.; Saha, G.; Das, R. J.; Raghu, D.; Mausam. (2024). MediTOD: An English Dialogue Dataset for Medical History Taking with Comprehensive Annotations. EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936

Fareez, F. et al. (2022). A dataset of simulated patient-physician medical interviews with a focus on respiratory cases. Scientific Data, 9, 313. https://doi.org/10.1038/s41597-022-01423-1

Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. https://arxiv.org/abs/2106.09685

Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314

Qwen Team. (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388. Ficha da variante utilizada: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML. https://arxiv.org/abs/2212.04356

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

Costa, M. (2026). Entrevista concedida a André Lucas sobre o ensino de anamnese e a proposta do projeto [mensagens de voz e vídeo]. Registro em `evidencias/entrevista_cliente.md`.
