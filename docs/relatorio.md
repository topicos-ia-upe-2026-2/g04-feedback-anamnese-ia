# Relatório técnico — Motor de IA para feedback educacional de anamnese

**Grupo:** G04 | **Integrantes:** Leonardo Marcelo, André Lucas | **2026.2**

## 1. Problema e contexto

Ver E1, seção 1 (`docs/E1-definicao.md`): docente de semiologia sem condição de dar retorno individual a mais de vinte estudantes; o sistema devolve feedback por pergunta da anamnese.

## 2. Fundamentação científica

MediTOD (Saley et al., EMNLP 2024, doi:10.18653/v1/2024.emnlp-main.936). Usamos o esquema CMAS para extrair atos `inquire` das falas do entrevistador, uma tarefa adaptada: o artigo avalia a NLU das falas do paciente, e seus números não são comparáveis aos nossos.

## 3. Modelo e adaptação

Qwen3-4B-Instruct-2507 (Apache-2.0), conversão `mlx-community/Qwen3-4B-Instruct-2507-4bit` (revisão `50d4277`). LoRA rank 8 nas projeções de atenção e MLP das últimas 4 camadas (1.835.008 parâmetros treináveis), pesos-base em 4 bits congelados. Apple M5, 16 GiB, MLX 0.32.4 / MLX-LM 0.32.2. Detalhes em `docs/E2-modelo-dados.md`, seção 4.

## 4. Dados

MediTOD `raw_data/` no commit `0b7a87c`, conferido por SHA-256; anotações sem licença identificada (D6). Amostra = fala do médico + 2 falas anteriores; alvo = atos `inquire`. Partição 80/10/10 por grupos de diálogos com conteúdo idêntico (18 pares duplicados, 213 grupos): 8.940 / 1.143 / 1.014 falas. Detalhes e manifesto em `docs/E2-modelo-dados.md`, seções 2 e 3.

## 5. Experimentos

E2: B0 (regras), B1 (Qwen sem adaptador) e LoRA no mesmo holdout de 24 falas, com o mesmo prompt, decodificação e parser. Registro completo em `docs/experimentos.md`.

## 6. Aplicação

<!-- Descrição do protótipo, fluxo de uso e demonstração -->

## 7. Resultados e análise de falhas

Primeiros resultados (E2, holdout diagnóstico): F1 de item 0,121 (B0), 0,098 (B1) e 0,000 (LoRA-02). O LoRA-02 gerou JSON válido em todas as falas, mas sempre lista vazia; a primeira rodada já colapsava para `[]`. A causa não foi isolada; o plano de correção até a E3 está em `docs/E2-modelo-dados.md`, seção 8.

## 8. Limitações e uso responsável

<!-- Limitações, dados sensíveis/anonimização, cuidados de uso -->

## 9. Referências
