# drummix

Mixagem reprodutível de covers de bateria. Recebe a bateria gravada (separada em stems pelo [Moises](https://moises.ai)), a faixa sem bateria e a música original, e entrega a música completa mixada e masterizada, junto com os stems processados para o GarageBand, o MIDI dos golpes do bumbo e um relatório com as medições.

Foi feito para gravações com a Yamaha EAD10 (um único microfone estéreo no bumbo). Funciona com qualquer gravação de bateria separada em stems no padrão de nomes do Moises.

## O que ele faz

1. **Alinha** a bateria com a banda, correlacionando os ataques da sua gravação com a bateria original (original − sem bateria).
2. **Limpa** cada peça: filtros, gate nos toms, EQ, compressão, largura estéreo.
3. **Reforça o bumbo** com um sample sintético afinado na tônica da música e alinhado em fase. Gera também o MIDI dos golpes.
4. **Aproxima o timbre** da bateria do disco original numa porcentagem configurável (por oitava, com limite de ±6 dB).
5. **Sala** curta, **equilíbrio** bateria × banda em LU e **master** em −14 LUFS / −1 dBTP (true peak).
6. **Entrega versionada** em `output/vN/`: nunca sobrescreve, grava em `vN.tmp` e renomeia. Verifica se a soma dos stems bate com o premaster, se volume e pico estão no alvo, e recusa um final idêntico ao anterior quando a configuração mudou.

## Uso

Requer Python 3.12, [uv](https://docs.astral.sh/uv/) e ffmpeg.

```bash
uv sync
uv run drummix new minha-musica \
  --stems "pasta com os stems do Moises" \
  --band "Musica - Sem bateria.wav" \
  --original "Musica - Original.wav" \
  --style fusion
uv run drummix render minha-musica
```

Para ajustar, edite `songs/minha-musica/song.yaml` (só o que difere do perfil) e rode `render` de novo. Exemplo, para deixar a caixa 2 dB mais alta:

```yaml
style: fusion
key: A minor
bpm: 131
params:
  clean:
    snare:
      gain_db: 4.0
```

Os perfis ficam em `profiles/`. O `fusion.yaml` foi calibrado de ouvido num cover de jazz fusion.

O áudio (`songs/*/input`, `songs/*/output`) fica fora do git.

## Testes

```bash
uv run pytest                 # suíte com áudio sintético (~15 s)
uv run pytest -m regression   # regressão com áudio real, se os arquivos existirem
```

## Roadmap

- **Plano 2:** modelador de transientes, reforço da caixa com samples, compressão paralela, saturação, controle de aspereza nos pratos, IRs reais + plate, EQ dinâmico na banda acionado pela bateria, master guiada pela referência, perfil `rock`.
- **Plano 3:** página de comparação A/B no mesmo volume com simulação de celular, troca do áudio no vídeo do Rec'n'Share, skill do Claude Code.

Design e planos estão em [`docs/superpowers/`](docs/superpowers/).

## Licença

GPL-3.0. O drummix usa o [pedalboard](https://github.com/spotify/pedalboard) (GPLv3).
