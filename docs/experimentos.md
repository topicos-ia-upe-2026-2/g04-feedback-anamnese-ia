# Registro de experimentos

> Documente cada execução relevante: data, configuração, hiperparâmetros, dados usados,
> métricas obtidas e observações. Isso fundamenta as seções de resultados da E2, E3 e E4.

## Protocolo definido na E1 (antes de qualquer execução)

Fixado agora para que a escolha de métricas e partições não seja influenciada pelos resultados.

| Item | Definição |
|---|---|
| Tarefa | Histórico limitado + fala atual do entrevistador → lista JSON de atos `inquire` (`slot`, `value`, `attributes`) |
| Unidade de amostra | Uma fala do entrevistador; inclui falas cuja resposta esperada é lista vazia |
| Partições | Treino / validação / teste por **grupo de diálogo** (diálogos idênticos no mesmo grupo), divisão determinística com semente registrada |
| Sistemas comparados | (1) Qwen3-4B-Instruct-2507 sem adaptador · (2) regras simples por busca textual, quando executadas · (3) Qwen3-4B + LoRA |
| Mesmas condições | Mesmo prompt, mesmas amostras de teste, mesma regra de validação do JSON para todos os sistemas |
| Métricas principais | Precisão, recall e F1 **micro** sobre os rótulos `inquire` |
| Métricas de apoio | Taxa de JSON válido (sem correção oportunista), perda de treino/validação, latência por amostra, pico de memória |
| Saídas inválidas | Contabilizadas como erro e reportadas à parte; a regra exata acompanha os resultados da E2 |
| Uso do teste | O teste **não** orienta hiperparâmetros nesta rodada; alterações motivadas pelo teste são registradas separadamente |
| Fora desta avaliação | Omissões, qualidade pedagógica, WER e eficácia de aprendizagem (exigem outros dados e protocolos) |
| Registro obrigatório | Revisão do MediTOD e SHA-256 dos arquivos, revisão dos pesos (HF), configuração, semente, contagens e exclusões |

## Baseline

| Data | Descrição | Configuração | Métricas | Observações |
|---|---|---|---|---|
| E2 (planejado) | Qwen3-4B-Instruct-2507 sem adaptador | `mlx-community/Qwen3-4B-Instruct-2507-4bit`, mesmo prompt do LoRA | — | Ainda não executado |
| E2 (planejado, opcional) | Regras simples por busca textual | Palavras-chave por categoria CMAS | — | Ainda não executado |

## Versões ajustadas (fine-tuning)

| Data | Descrição | Hiperparâmetros | Dados | Métricas | Observações |
|---|---|---|---|---|---|
| E2 (planejado) | Primeiro LoRA (piloto) | A definir no arquivo de configuração da E2 (camadas, rank, lr, passos, tamanho de sequência, lote) | Piloto de treino do MediTOD, split próprio | — | Ainda não executado |
