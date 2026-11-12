# Motor de IA para feedback educacional de anamnese

Projeto aplicado da disciplina **Tópicos Avançados em Inteligência Artificial** — 2026.2 · Prof. Marcondes Ricarte

| Grupo | Integrantes |
|---|---|
| GXX <!-- TODO: número do grupo --> | Leonardo Marcelo (@LeonardoMarcelo), André Lucas (@andredutradev) |

**Etapa atual:** E1 · Definição (só documentação). Documento completo: [`docs/E1-definicao.md`](docs/E1-definicao.md) · [PDF](docs/E1-definicao.pdf).

> **Em uma frase:** um modelo de linguagem pré-treinado (Qwen3-4B-Instruct-2507), ajustado com LoRA sobre o corpus
> MediTOD, identifica quais perguntas o estudante de medicina fez numa anamnese simulada, para gerar feedback
> educacional verificável. **Não diagnostica, não prescreve e não recomenda conduta.**

| Ficha rápida | |
|---|---|
| Cliente | Marília Costa, docente de medicina (preceptoria em ambulatório universitário) |
| Usuário principal | Estudante de medicina que pratica anamnese; preceptor como beneficiário indireto |
| Artigo-base | Saley et al. (2024). *MediTOD*. EMNLP. [doi:10.18653/v1/2024.emnlp-main.936](https://doi.org/10.18653/v1/2024.emnlp-main.936) |
| Modelo | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0), LoRA sobre pesos 4-bit em MLX |
| Dados | MediTOD `raw_data/` (diálogos + anotações CMAS) · Fareez et al. (2022), CC0, para áudio futuro |
| Baseline | Mesmo Qwen sem adaptador (+ regras simples, se executadas) |
| Métricas | Precisão, recall e F1 micro dos rótulos `inquire`; validade de JSON; perda; latência e memória |

## Cliente, problema e beneficiário

**Contexto.** Marília Costa descreve um problema da preceptoria em ambulatório universitário: um docente precisa
acompanhar vários estudantes e, ao mesmo tempo, atender à demanda assistencial. A observação da entrevista e o
feedback individual dependem da disponibilidade do preceptor. O estudante pode terminar uma atividade sem saber
quais partes da investigação deixou de realizar.

**Quem se beneficia.** O usuário principal é o estudante de medicina que pratica anamnese. O preceptor é
beneficiário indireto: poderá consultar um registro das perguntas e revisar o feedback antes de usá-lo no ensino.
A proposta parte de uma cliente real; não pressupõe parceria institucional, contratação ou aceite formal da solução.

**Como é resolvido hoje e o que a aplicação muda:**

| Processo atual | Limitação relevante | Benefício pretendido |
|---|---|---|
| Observação direta e comentários do preceptor | Dependem da presença e do tempo disponível | Organizar evidências da entrevista para facilitar a revisão |
| Autoavaliação do estudante | Pode divergir da avaliação externa | Mostrar trechos concretos que sustentam o retorno |
| Avaliações práticas em momentos específicos | Não acompanham necessariamente cada treino | Permitir revisão de entrevistas simuladas entre avaliações |

**Problema específico:** a dificuldade de obter um registro **frequente e verificável** das etapas de investigação
realizadas pelo estudante. A hipótese de benefício é que esse registro ajude a orientar a prática seguinte. Ganho de
aprendizagem, redução da carga do preceptor e satisfação do usuário são objetivos futuros, não resultados demonstrados.

**Evidência inicial.** Relato escrito *"Motor de IA especializado em educação médica: o problema e as
características da solução"*, de Marília Costa, transcrito em [`evidencias/relato_cliente.md`](evidencias/relato_cliente.md).
Ele descreve a necessidade, o fluxo desejado e as qualidades esperadas do feedback. É um relato documental: não é
uma entrevista conduzida pela dupla nem uma observação independente do ambulatório.

A literatura situa o problema sem medir o processo específico da cliente: Haring et al. (2017) sobre raciocínio
clínico observável na anamnese; Spencer (2003) sobre ensino no ambiente clínico; Cheung et al. (2019) e Alex et al.
(2024) sobre barreiras à observação direta. Sobre autoavaliação, Blanch-Hartigan (2011) encontrou capacidade
**moderada** em estudantes. Por isso o feedback externo é tratado como complemento verificável à reflexão do aluno,
e não como prova de que o estudante avalia mal o próprio desempenho.

**Limites da evidência:** não há entrevista agendada, consentimento assinado nem validação clínica nos materiais
recebidos. Uma conversa com a cliente e a revisão do roteiro estão planejadas para as próximas etapas.

## Solução e escopo

<!-- Explique a entrada, a saída e o fluxo de uso. Liste o que SERÁ e o que NÃO SERÁ
implementado até a apresentação final. -->

## Artigo-base

- **Título:**
- **Autores:**
- **Ano:**
- **DOI/URL:**
- **Componentes utilizados/adaptados:** <!-- método, arquitetura, protocolo experimental -->

## Modelo pré-treinado

- **Nome e versão:**
- **Origem dos pesos:** <!-- link (ex.: Hugging Face) -->
- **Licença:**
- **Tarefa original:**
- **Relação com o artigo-base:** <!-- justificativa, se não for o mesmo modelo do artigo -->

## Conjunto de dados-base

- **Fonte e acesso:** <!-- link + como obter -->
- **Licença/autorização:**
- **Formato e volume:**
- **Rótulos e adequação ao problema:**

> Arquivos de dados grandes **não** devem ser versionados no repositório.
> Use este espaço para o link (Google Drive, Hugging Face, etc.) e as instruções de download.

## Plano de adaptação e avaliação

- **O que será ajustado:** <!-- full fine-tuning, parcial ou adaptadores (LoRA etc.) -->
- **Recursos computacionais:** <!-- GPU/Colab/Kaggle, tempo estimado -->
- **Baseline:**
- **Métricas:**
- **Riscos:**
- **Divisão das atividades entre os integrantes:**

## Como executar

```bash
# 1. Criar ambiente
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Se aplicável) baixar dados e pesos conforme as seções acima

# 3. Executar
# python src/train.py ...
# python src/app.py ...
```

## Resultados

<!-- Preencher a partir da E2: baseline, configuração do ajuste, métricas,
exemplos de sucesso e falha, comparação final. Detalhes em docs/experimentos.md -->

## Limitações e uso responsável

<!-- Limitações conhecidas, anonimização de dados sensíveis, cuidados de uso,
custo/latência quando relevante -->

## Referências

<!-- Artigo-base e demais referências citadas (formato ABNT ou APA) -->
