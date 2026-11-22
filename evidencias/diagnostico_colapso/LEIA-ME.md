# Diagnóstico do colapso do LoRA-02 para lista vazia

O LoRA-02 respondeu `[]` nas 24 falas do holdout. Esta pasta registra a investigação de 22/11/2026, feita com o
modelo e o adaptador fixados, mudando uma variável por vez. Números em `resultados.json`; scripts em `scripts/`
(rodar de `src/`; saídas em `src/artifacts/diagnostico/`, local).

1. **Reprodução.** O LoRA-02 reavaliado repetiu exatamente as 24 saídas. Retreinado com a mesma configuração (A),
   gerou curvas de perda quase iguais e o mesmo colapso.
2. **Primeiro token.** Lista vazia é um único token (`[]`); lista com atos começa com `[`. O modelo original
   prevê `[` com probabilidade 0,999 nas falas com pergunta. Após 20 passos, `[]` já tinha 0,53; após 40, 0,96,
   inclusive nos 35 exemplos de treino com pergunta que o modelo viu.
3. **Separação.** O LoRA-02 ainda ordena falas com e sem pergunta (AUC 0,85), mas todas ficam acima de 0,9: o
   problema é um deslocamento global, não falta de informação.
4. **Teste B (sem alvos vazios, 40 passos):** o colapso some. Os exemplos vazios causam o deslocamento.
5. **Teste C (com alvos vazios, 400 passos):** respostas vazias indevidas caem para 3%. Com 40 passos o treino
   parou no pico do deslocamento.

**Correção adotada:** treinar por mais passos e escolher o checkpoint pela extração na validação
(`src/motor/select.py`), não pela perda nem pelo último passo. Ver LoRA-03 e LoRA-04 em `docs/experimentos.md`.
