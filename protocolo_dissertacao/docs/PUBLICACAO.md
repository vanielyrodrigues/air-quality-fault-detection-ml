# Publicação da versão da dissertação

1. Extrair o pacote e localizar a pasta `protocolo_dissertacao`.
2. Abrir GitHub Desktop e autenticar a conta proprietária do repositório.
3. Escolher File > Clone repository > URL e informar `https://github.com/vanielyrodrigues/air-quality-fault-detection-ml`.
4. Escolher uma pasta no computador e concluir Clone. Se já houver uma cópia local atualizada, selecioná-la e executar Fetch origin/Pull origin antes de copiar arquivos.
5. Criar uma branch em Branch > New branch, com nome `dissertacao-final-2026`.
6. Em Repository > Show in Explorer, abrir a pasta do repositório clonado.
7. Copiar a pasta inteira `protocolo_dissertacao` para dentro da pasta do repositório. Não copiar a pasta externa que contém este guia e não enviar somente o ZIP.
8. Conferir na aba Changes se aparecem `protocolo_dissertacao/src`, `dados`, `resultados`, `docs`, `README.md` e `requirements.txt`. Não apagar os arquivos históricos da raiz nesta etapa.
9. No campo Summary, escrever `Adiciona protocolo e resultados finais da dissertação`.
10. Clicar em Commit to dissertacao-final-2026 e depois Publish branch/Push origin.
11. Abrir o repositório no navegador, criar a pull request da branch para a branch padrão, revisar a lista de arquivos e concluir Merge pull request.
12. Na branch padrão, abrir `protocolo_dissertacao/README.md` e verificar a navegação para código, dados e resultados.
13. Copiar o identificador do commit incorporado. Opcionalmente criar uma Release com uma tag, por exemplo `dissertacao-2026-v1`, apontando para esse commit.
14. Informar na dissertação a identificação efetiva dessa versão, sem inventar o hash ou anunciar uma tag ainda não criada.

## Conferência antes de anunciar a publicação completa

- O pacote inclui a base de entrada consolidada `dados_ordenados.csv`, mas não os CSV individuais originais de aquisição. Se o apêndice promete esses CSV, acrescentá-los separadamente em `dados/originais` antes de afirmar que estão publicados, ou ajustar o texto para descrever exatamente a base disponibilizada.
- A organização adiciona uma pasta própria ao repositório existente e preserva os arquivos anteriores. Atualizar o README da raiz com um link para `protocolo_dissertacao/README.md` facilita localizar o protocolo final.
- Não executar os comandos de treinamento apenas para enviar o material; resultados e verificações já estão incluídos.

## Alternativa pelo navegador

GitHub aceita até 100 arquivos por envio e até 25 MiB por arquivo no navegador. Como o pacote tem mais de 100 arquivos, GitHub Desktop evita dividir a estrutura em muitos envios. Se usar Upload files, publicar lotes de até 100 arquivos preservando a hierarquia e conferir a pasta de destino a cada envio.

Documentação:
- https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
