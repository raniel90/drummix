# drummix — design

Data: 2026-11-02 · Status: aprovado no brainstorming, aguardando revisão do spec

## 1. Contexto e objetivo

O cover de bateria de *The Hornet* (Yellowjackets) foi mixado ad hoc com ffmpeg e scripts soltos. Funcionou, mas com falhas de processo: uma soma de stems falhou em silêncio (word-splitting no zsh) e entregou um final inalterado; a verificação da entrega não existia; versões intermediárias se perderam ao limpar pastas temporárias.

`drummix` torna esse trabalho reprodutível: uma CLI Python que transforma uma gravação de bateria (Yamaha EAD10, separada em stems pelo Moises) numa mixagem completa com a banda, pronta para vídeo, e uma skill do Claude Code que conduz a conversa e as iterações. Deve servir a vários estilos — jazz fusion (Yellowjackets) e rock anos 90 (Foo Fighters, Stone Temple Pilots).

### Fora do escopo (v1)
- Separação de stems (continua manual no Moises).
- MIDI do trigger da EAD10 como entrada.
- Corte vertical 9:16 do vídeo.
- Regravação com mais microfones (não há equipamento).
- Matchering (substituído por EQ-match próprio).

## 2. Entradas e saídas

**Entradas por música** (copiadas para `songs/<slug>/input/` na primeira execução, com hash):
- 7 stems do Moises da bateria: kick, snare, toms, hi-hat, cymbals, other, other_kit (WAV 44,1 kHz). Nome no padrão `…-<Tom>-<BPM>bpm-<A4>hz.wav`.
- Faixa sem bateria (Moises) e música original (referência).
- Opcional: vídeo do Rec'n'Share.

**Saídas por render** em `songs/<slug>/output/vN/` (N incrementa; nunca sobrescreve):
- `final.wav` (24-bit, −14 LUFS, true peak ≤ −1 dBTP).
- `stems/` — um WAV por elemento (bumbo, sample do bumbo, caixa, sample da caixa, toms, chimbal, pratos, other_kit, sala, plate, banda), todos começando em 0:00 na linha do tempo da bateria/vídeo; soma com faders em 0 dB = premaster.
- `drums.mid` — golpes detectados (36 bumbo, 38 caixa), tempo do projeto = BPM detectado.
- `report.md` / `report.json` — medições antes/depois vs referência, avisos, hashes.
- `video.mp4` (se houver vídeo).
- `compare.html` na raiz da música, cobrindo original, todas as versões e simulação de celular.

## 3. Configuração em camadas

`profiles/<estilo>.yaml` → `songs/<slug>/song.yaml` (só overrides). Ajustar ("caixa +2 dB") é editar uma linha e rodar `drummix render` de novo.

| Parâmetro | `fusion` (validado em The Hornet) | `rock` (ponto de partida, calibrar no 1º cover) |
|---|---|---|
| Bateria vs banda | −2 LU | −1 LU |
| Match de timbre vs referência | 65% | 50% |
| Ganho da caixa | +2 dB | +2 dB |
| Sample do bumbo | sintético afinado no tom, −9 dB vs bumbo real | sintético com clique ou pack, −6 dB |
| Sample da caixa | só golpes acima do limiar, −10 dB | golpes acima do limiar, −4 dB |
| Compressão paralela (NY) | 15% | 30% |
| Saturação | leve | média |
| Sala | Voxengo "Small Drum Room", envio baixo | Voxengo "Nice Drum Room" + plate mais longo |
| Master | −14 LUFS / −1 dBTP | −14 LUFS / −1 dBTP |

Princípio do perfil fusion: bumbo contido (o usuário achou +2 dB "muito pesado").

## 4. Arquitetura

```
drummix/
  pyproject.toml              uv, Python 3.12
  drummix/
    cli.py                    drummix new | render | compare | video
    song.py                   carrega perfil + overrides, valida, tom/BPM do nome
    audio.py                  único ponto de I/O: soundfile p/ WAV; ffmpeg via lista
                              de argumentos (nunca shell) p/ m4a/vídeo; LUFS, TP, hash
    stages/                   funções puras: (Stems, params) -> Stems; sem I/O
      ingest.py align.py clean.py transients.py samples.py tone_match.py
      dynamics.py space.py band.py balance.py master.py
    dsp/                      blocos reutilizáveis: biquads, envelopes (numba),
                              eq dinâmico com chave, gerador de IR plate, kick sintético
    deliver.py                vN, escrita, MIDI, relatório, verificação
    compare.py                compare.html (Web Audio próprio) + simulação de celular
  profiles/fusion.yaml rock.yaml
  assets/irs/                 Voxengo + LICENSE-voxengo.txt
  assets/samples/<pack>/      manifest.yaml (camadas de velocity + round robin) + WAVs
  tests/                      sinais sintéticos
  .claude/skills/drummix/SKILL.md
  songs/<slug>/               song.yaml versionado; input/ e output/ fora do git
```

**Dependências:** numpy, scipy, numba, pedalboard (efeitos, `BrickwallLimiter(true_peak=True)`, `Convolution`), pyloudnorm (LUFS/LRA), pyebur128 (conferência de true peak), librosa (onsets da caixa, Plano 2), soundfile (WAV 24-bit), pretty_midi, pyyaml. ffmpeg do sistema. Licenças: pedalboard é GPLv3 — sem impacto em projeto pessoal não distribuído.

**Convenção de arrays:** internamente `(samples, channels)` float64; conversão para `(channels, samples)` só na fronteira com pedalboard.

**Determinismo:** mesma entrada + mesmo YAML → mesmo áudio (seeds fixas em ruído, round robin e IR sintético).

## 5. Pipeline

0. **ingest** — copia entradas para `input/` com hash; extrai tom e BPM do nome; decodifica m4a/mp4 via ffmpeg (aplica o priming AAC de 2112 amostras pelo edit list); descarta stems vazios (< −60 LUFS) com aviso.
1. **align** — bateria de referência = original − sem bateria. Correlação de onsets entre o take e a referência em janelas ao longo da música; as curvas de correlação das janelas são somadas (média por lag) e o pico global dá o atraso da banda — a música repete por compasso e uma janela isolada costuma preferir ±1–2 compassos por margem estreita (The Hornet: só 8 de 16 janelas acertam; a soma dá 11,73 s a 0,111 contra 13,58 s a 0,079). Os melhores lags por janela servem só ao relatório: janelas que concordam com o pico (±50 ms), desvio delas e correlação média delas; aviso se correlação fraca ou atraso instável (> 30 ms). A linha do tempo da bateria/vídeo manda: a banda é deslocada.
2. **clean** — HP/LP por peça; gate nos toms; grave do bumbo em mono; estreitar chimbal; resíduo (soma dos stems vs take) reportado.
3. **transients** — modelador por peça: envelope rápido (≈1/20 ms) − lento (≈25/150 ms) → ganho de ataque e sustain; numba.
4. **samples** — bumbo: detector de fluxo de energia no grave (< 150 Hz, subida ≥ 12 dB em 8 ms), validado em The Hornet (544 golpes), rodando no stem **cru** — o compressor da limpeza gera redisparos ~13 dB abaixo (597 detecções no limpo, 176 espúrias); a amostra é alinhada em fase contra o stem limpo. Caixa (Plano 2): onsets com librosa (`backtrack`, `hop` 128); velocity em dB do pico pós-onset; guarda de retrigger (flams); filtro de bleed (energia do stem da caixa vs bumbo no golpe). Bumbo: kick sintético afinado na tônica (~55 Hz para Lá), alinhado em fase por correlação. Caixa: camadas de velocity + round robin do pack; abaixo do limiar, só som acústico (ghost notes intactas). MIDI com 36/38.
5. **tone_match** — curva em bandas de oitava (31,5 Hz–16 kHz, níveis relativos ao total) do barramento vs bateria de referência; alvo = X% da diferença. Itera só no barramento somado (EQ é linear) e aplica o EQ final acumulado, uma vez, idêntico em cada stem (mantém a soma). A cada iteração remove-se o deslocamento comum do passo (mediana — um ganho uniforme é invisível em níveis relativos, e sem isso uma banda presa no limite arrasta as outras); a correção **acumulada** é limitada a ±6 dB por banda, com reforço de sub ≤ +3 dB e corte de agudos ≤ −3 dB.
6. **dynamics** — compressão por peça (pedalboard); NY parallel no barramento; de-harsh dinâmico em pratos/chimbal (shelf acima de ~5 kHz atuando só em picos); saturação paralela no bumbo com HP (harmônicos 100–400 Hz para celular) + saturação leve no barramento.
7. **space** — envio para IR de sala (Voxengo); plate sintético (1,2–1,8 s, low cut 150 Hz) só na caixa.
8. **band** — EQ dinâmico na banda com chave: bumbo → fundamental, caixa → 2–5 kHz; −2 a −3 dB; ataque rápido, release 50–150 ms; mais cortes estáticos estreitos opcionais.
9. **balance** — ganho da bateria para ficar X LU abaixo da banda (medido só onde ambos tocam).
10. **master** — EQ-match da mixagem inteira vs original (X%, padrão 30%); glue compressor; ganho para −14 LUFS; `BrickwallLimiter(ceiling −1, true_peak)`; conferência com pyebur128, reajuste iterativo até ±0,3 LU.
11. **deliver** — grava `output/vN/`; verifica: arquivos existem, durações iguais, soma dos stems == premaster (erro < −90 dB), LUFS/TP dentro do alvo, hash do final diferente da versão anterior quando o YAML mudou. Vídeo: `ffmpeg -i video -i final.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k -t <duração do vídeo> -movflags +faststart` (+ `-tag:v hvc1` se HEVC); sync conferido por correlação entre o áudio original do vídeo e o novo (0 ± 1 ms).

## 6. Página de comparação

`compare.html` local, JS próprio (~150 linhas, sem dependências): decodifica todas as versões, toca todas sincronizadas no mesmo playhead, alterna com rampa de 5 ms (sem clique), ganhos igualados por LUFS calculados no Python e gravados em JSON. Botão "celular" aplica a simulação: mono, HP 4ª ordem em 400 Hz, +3 dB em 3 kHz, LP 11 kHz, limitador leve.

## 7. Skill do Claude Code

`.claude/skills/drummix/SKILL.md`: ao iniciar uma música nova, pergunta só o que difere do perfil (estilo, arquivos, uso); roda `drummix new` + `render`; apresenta o relatório; traduz pedidos de ouvido ("caixa mais alta", "muito pesado") em mudanças no `song.yaml`; renderiza nova versão; lembra de reimportar no GarageBand (ele copia o áudio para dentro do projeto). Nunca edita `input/`; nunca apaga versões.

## 8. Erros e robustez

- Falha alta, nunca silenciosa: todo subprocess checa returncode; nenhuma montagem de comando via shell.
- Avisos no relatório: alinhamento fraco/instável, TP acima do alvo, resíduo alto, golpes suspeitos (intervalo < 60 ms), stems vazios, LRA muito baixa (sobrecompressão).
- Render interrompido não deixa `vN` parcial (escreve em `vN.tmp` e renomeia).

## 9. Testes

Sinais sintéticos com pytest:
- align recupera atraso conhecido (±1 ms) e sinaliza atraso instável.
- onsets acham golpes e ghost notes plantados; retrigger não duplica flams.
- transient shaper aumenta o crest factor com ataque positivo.
- EQ dinâmico só reduz a banda quando a chave toca.
- limiter respeita true peak ≤ −1 dBTP; master atinge −14 ± 0,3 LUFS.
- soma dos stems == premaster.
- deliver nunca sobrescreve e não deixa versão parcial.
- mux de vídeo preserva sincronia.
- E2E curto: música sintética de 20 s passa pelo pipeline completo.

**Regressão com material real — `songs/the-hornet`** (fontes em `~/Downloads`, copiadas pelo `drummix new`):
- Entradas: pasta `The Hornet - Audio-A minor-131bpm-441hz/` (7 stems), `The Hornet - Sem bateria.wav`, `The Hornet - Original.wav`, `The Hornet - Audio.m4a` (take bruto).
- Baseline: `The Hornet - Mix Cover/The Hornet - Cover Final.wav` (sha1 `9c8a284fb5ee…`, −13,9 LUFS, −1,2 dBTP) — a versão aprovada de ouvido na sessão manual; entra no `compare.html` como "manual".
- Critérios: atraso da banda 11,73 ± 0,03 s; ~544 golpes de bumbo (±10%); final −14 ± 0,3 LUFS, TP ≤ −1 dBTP; bateria −2 ± 0,5 LU vs banda; aprovação final de ouvido no A/B contra o baseline.

## 10. Questões em aberto

- Pack de samples para rock: escolher no primeiro cover de rock (Salamander, DrumGizmo ou SM MegaReaper), com licença verificada antes de entrar no git.
