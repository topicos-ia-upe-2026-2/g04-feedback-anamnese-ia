# Relatório técnico — Motor de IA para feedback educacional de anamnese

**Grupo:** G03 | **Integrantes:** Leonardo Marcelo, André Lucas | **2026.2**

> Estado: **E1**. As seções 1–5 resumem a definição do projeto (texto completo em [`E1-definicao.md`](E1-definicao.md));
> as seções 6–8 serão escritas a partir da E2 com resultados medidos.

## 1. Problema e contexto

Cliente: Marília Costa, docente de medicina. Na preceptoria de ambulatório universitário, o docente acompanha vários
estudantes enquanto atende à demanda assistencial, e a observação da entrevista e o feedback individual dependem da
disponibilidade dele. O estudante pode terminar a atividade sem saber o que deixou de investigar.

Beneficiário principal: estudante de medicina que pratica anamnese; indireto: o preceptor, que revisa o registro.
Problema específico: falta de um registro **frequente e verificável** das etapas de investigação feitas pelo estudante.
Resultado esperado: feedback em português com acertos, pontos de melhoria e etapas não identificadas, cada um ligado
ao trecho da entrevista, sem diagnóstico nem conduta. Evidência: relato documental da cliente
([`../evidencias/relato_cliente.md`](../evidencias/relato_cliente.md)) e literatura sobre observação direta e
autoavaliação (Haring et al., 2017; Spencer, 2003; Cheung et al., 2019; Alex et al., 2024; Blanch-Hartigan, 2011).

## 2. Fundamentação científica

Artigo-base: Saley, V. V. et al. (2024). *MediTOD: An English Dialogue Dataset for Medical History Taking with
Comprehensive Annotations*. EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936.

Usamos o esquema CMAS (atos, categorias e atributos) e a ideia de ajustar um LLM pré-treinado para emitir estrutura a
partir do diálogo. Diferente do artigo, extraímos os atos `inquire` das falas do **entrevistador**, com divisão própria e
sem canonicalização UMLS; por isso os números não são comparáveis aos do artigo (F1 0,7139 do Llama3-8B na NLU).
Apoio: LoRA (Hu et al., 2022), adaptação sobre modelos quantizados (Dettmers et al., 2023), LLMs adaptados no domínio
clínico (Van Veen et al., 2024) e forma do feedback (Bastos e Castro et al., 2021; van de Ridder et al., 2015).

## 3. Modelo e adaptação

Qwen3-4B-Instruct-2507 (Apache-2.0), em 4 bits via `mlx-community/Qwen3-4B-Instruct-2507-4bit`. Ajuste por adaptadores
LoRA em parte das camadas, pesos-base congelados, entropia cruzada sobre a saída supervisionada, com `mlx-lm` num Apple
M5 com 16 GiB. Troca em relação ao Llama3-8B do artigo justificada pela memória disponível, mantendo a formulação.
Plano B: Qwen3-1.7B. Ver [`model_card.md`](model_card.md) e [`decisoes.md`](decisoes.md).

## 4. Dados

MediTOD `raw_data/` (231 diálogos, 22.317 falas, 10.082 atos `inquire` em 13 categorias, inglês, sem licença identificada para as anotações → uso apenas local). Partições
próprias por grupo de diálogo para evitar vazamento (há 18 pares de diálogos idênticos → 213 grupos). Preparo planejado em 7 passos (revisão fixada + SHA-256,
alinhamento, normalização, agrupamento e split, histórico sem rótulo-alvo, piloto com semente, exportação `mlx-lm`).
Áudio futuro: Fareez et al. (2022), CC0. Ver [`data_card.md`](data_card.md).

## 5. Experimentos

Protocolo fixado na E1 ([`experimentos.md`](experimentos.md)): baseline = mesmo Qwen sem adaptador (+ regras simples,
quando executadas); métricas = precisão, recall e F1 micro dos rótulos `inquire`, validade do JSON, perda, latência e
memória; teste não usado para escolher hiperparâmetros. **Sem execuções até aqui.**

## 6. Aplicação

*A preencher na E3.* Planejado: áudio ou transcrição → revisão de falantes → extração → comparação com o roteiro do caso
→ feedback com trechos, com estado "inconclusivo" quando a entrada não for confiável.

## 7. Resultados e análise de falhas

*A preencher a partir da E2.*

## 8. Limitações e uso responsável

Finalidade só educacional; sem diagnóstico, conduta ou pacientes reais. Ajuste em inglês com desempenho em português
desconhecido. Cobertura parcial da anamnese pelo CMAS. Dados e adaptadores derivados não redistribuídos. Mapeamento
pedagógico ainda sem validação da cliente.

## 9. Referências

Lista completa no [README](../README.md#referências) e em [`../referencias.bib`](../referencias.bib).
