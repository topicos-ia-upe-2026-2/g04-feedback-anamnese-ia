# TODO — pendências do projeto

Lista viva de pendências. Cada item sai daqui quando vira commit, decisão em `docs/decisoes.md` ou resultado em
`docs/experimentos.md`.

## Antes de fechar a E1 (prazo 17/11/2026)

- [ ] **Transcrever o relato da cliente na íntegra** em [`evidencias/relato_cliente.md`](evidencias/relato_cliente.md),
  separando o texto da cliente do resumo da dupla. Hoje o arquivo só tem o resumo (pesa no critério "evidência inicial", 0,6 pt).
- [ ] Revisão cruzada do README pelos dois integrantes (ambos precisam conseguir defender cada escolha).

## Pendências de validação (não bloqueiam a E1)

- [ ] Conversar com a cliente e revisar com ela o mapeamento CMAS → interpretação educacional.
- [ ] Decidir como o roteiro trata as categorias que o mapeamento ainda não cobre: `residence` (244 atos),
  `symptom (dermatology)` (160), `medical test` (106), `other` (80), `basic information` (9).
- [ ] Esclarecer a licença das anotações do MediTOD com os autores **antes** de redistribuir qualquer derivado.

## E2 · Modelo e dados (22/11/2026)

- [ ] `src/`: script de aquisição + manifesto (revisão `0b7a87c`, SHA-256, contagens, exclusões) que reproduza as
  estatísticas do [`docs/data_card.md`](docs/data_card.md).
- [ ] Normalização de grafia declarada (ex.: `frquency check` → `frequency check`) sem inferir atributos.
- [ ] Decidir e registrar o que fazer com os 12 atos `inquire` em falas que não são do médico.
- [ ] Divisão determinística por grupo de diálogo (213 grupos; 18 pares idênticos) com semente registrada.
- [ ] Geração de amostras por fala do médico com histórico limitado e sem rótulo-alvo na entrada; exportar para `mlx-lm`.
- [ ] Prompt e contrato JSON únicos para baseline e LoRA; regra de validação de saída e tratamento de inválidos.
- [ ] Baseline: Qwen3-4B sem adaptador (e regras simples, se der tempo) nas mesmas amostras.
- [ ] Primeiro LoRA: arquivo de configuração versionado, logs, curvas de perda, tempo e pico de memória.
- [ ] Métricas: P/R/F1 micro + F1 por categoria + taxa de JSON válido + latência.
- [ ] Fixar versões em `requirements.txt` e instruções de execução no README.
- [ ] Preencher `docs/experimentos.md` e a seção "Resultados" com números medidos (sem estimativas).

## E3 · Aplicação (29/11/2026)

- [ ] Fluxo: transcrição/áudio → revisão de falantes → extração → comparação com roteiro → feedback com trechos.
- [ ] Estado "avaliação inconclusiva" para saída inválida, transcrição pouco confiável ou falantes não revisados.
- [ ] Avaliar Whisper large-v3-turbo em português com áudio de teste autorizado; medir custo e latência.
- [ ] Exemplos em português (sintéticos ou gravados com consentimento) para teste técnico.
- [ ] Teste de tarefa ou feedback de ao menos um representante do público-alvo.

## E4 · Apresentação (07/12/2026)

- [ ] Slides, demonstração de um caso completo, vídeo de contingência e relatório final em `docs/relatorio.md`.

## Depois do merge de cada etapa

- [ ] Criar a tag da etapa (`git tag e1 && git push origin e1`), como pede o `CONTRIBUTING.md`.
