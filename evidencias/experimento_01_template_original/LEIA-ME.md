# Tentativa original preservada

Esta pasta contém a primeira execução real, anterior à correção do template.
O adaptador completou 40 atualizações, mas suas 24 respostas falharam no parser
JSON. Depois do bloco `<think>`, todas eram `[]`: o colapso para lista vazia já
existia antes da correção. Os resultados aqui são históricos; os arquivos no diretório superior
identificam a rodada corrigida.

Diagnóstico: o template do modelo convertido MLX
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, revisão
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, acrescentava o bloco vazio
`<think>\n\n</think>\n\n` à resposta do assistente ao construir cada alvo
supervisionado. O modelo aprendeu esse prefixo incompatível com JSON estrito.
Mudar `enable_thinking=False` não o removia nesse template.

A correção usa o template oficial Instruct-2507, sem esse bloco, em preparação,
treino e inferência. O parser não foi flexibilizado. A inspeção do holdout foi
usada para diagnosticar o defeito; a nova rodada é, portanto, diagnóstica e não
cega. IDs de treino/validação, holdout e hiperparâmetros foram mantidos.

`codigo_executado/` preserva os quatro módulos usados na tentativa. O checkpoint,
as predições originais e os dados piloto permanecem no caminho local ignorado
`../../src/artifacts/experimento_01_template_original/`. Hashes originais
podem ser confrontados com esses arquivos; não houve substituição de predições
dentro do experimento histórico. Os caminhos absolutos do manifesto e do log foram substituídos por `<projeto>`
em 22/11/2026; os hashes originais estão no campo `sanitizacao` do manifesto.
