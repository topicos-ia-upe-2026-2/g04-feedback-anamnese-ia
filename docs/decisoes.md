# Registro de decisões do projeto

> Cada decisão relevante de projeto (troca de modelo, redução de escopo, mudança de dados,
> escolha de aplicação) deve ser registrada aqui com data, decisão tomada e justificativa.
> Mudanças relevantes de problema, artigo, modelo ou dados devem ser comunicadas ao professor
> antes de comprometer as próximas entregas.

| Data | Decisão | Justificativa | Impacto nas entregas |
|---|---|---|---|
| 03/11/2026 | **D1 · Problema:** extrair as perguntas (atos `inquire`) feitas pelo estudante numa anamnese simulada, como base de feedback educacional | Necessidade relatada pela cliente: registro frequente e verificável das etapas de investigação. Mantém o sistema longe de diagnóstico ou conduta | Define a tarefa da E2 (extração) e o produto da E3 (feedback com rastreabilidade) |
| 03/11/2026 | **D2 · Artigo-base:** MediTOD (Saley et al., EMNLP 2024) | Corpus público com diálogos de coleta de história clínica e anotação fina de perguntas por categoria e atributo (CMAS) | Esquema de rótulos e dados da E2 vêm daqui |
| 03/11/2026 | **D3 · Modelo:** Qwen3-4B-Instruct-2507 no lugar do Llama3-8B do artigo | Cabe em 16 GiB de memória unificada com 4 bits; Apache-2.0; instrução multilíngue (hipótese para português). Mantém a formulação do artigo: geração supervisionada de estrutura | Treino e inferência locais na E2 e E3; sem custo de nuvem |
| 03/11/2026 | **D4 · Método de ajuste:** LoRA sobre pesos 4-bit com `mlx-lm` (Apple M5) | Só o hardware local disponível; LoRA (Hu et al., 2022) e adaptação sobre modelo quantizado (Dettmers et al., 2023) mantêm o custo baixo. Não presumimos equivalência exata ao QLoRA/NF4 | Configuração e logs da E2; ordem de redução se faltar memória: contexto → camadas → tamanho da execução |
| 03/11/2026 | **D5 · Partições próprias por grupo de diálogo** | Os arquivos de divisão do MediTOD têm 1 id cada (`115`, `317`, `407`). Agrupar diálogos idênticos evita vazamento entre treino e teste | Split determinístico e documentado na E2 |
| 03/11/2026 | **D6 · Dados e adaptadores só locais** | O repositório MediTOD não tem LICENSE e não há licença identificada para as anotações; o CC0 de Fareez et al. não cobre as anotações | Repositório guarda código, configs e estatísticas agregadas; redistribuição só após esclarecer a licença |
| 03/11/2026 | **D7 · Sem canonicalização UMLS e sem tarefas de política/geração** | A versão canonicalizada exige licença UMLS; o sistema não deve entrevistar no lugar do aluno | Avaliação por conjunto de rótulos próprio, sem comparação numérica com o artigo |
| 12/11/2026 | **D8 · Plano B de modelo:** Qwen3-1.7B com a mesma receita, caso o professor peça redução de escopo ou a memória não comporte o 4B | A receita (dados, prompt, LoRA, métricas) não depende do tamanho do modelo | Troca é comunicada ao professor antes da E2 |
