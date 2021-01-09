# Aprendizado de Representações em Grafos por Redes Neurais: uma Análise Comparativa
Projeto de conclusão de curso de graduação no Centro Federal de Educação Tecnológica Celso Suckow da Fonseca (CEFET/RJ).

- Nome do curso: Sistemas para Internet
- Título do projeto: Aprendizado de Representações em Grafos por Redes Neurais: uma Análise Comparativa
- Nome completo do autor: Augusto José Moreira da Fonseca
- Nomes do orientador: Eduardo Bezerra da Silva, D.Sc.
- Data da defesa: 12/2020

## Estrutura de pastas
* `/monografia`: Monografia completa e formatada (incluindo as correções solicitadas pela Banca e a ficha catalográfica elaborada com a orientação da biblioteca).
* `/bibliografia`: Toda bibliografia utilizada no projeto que esteja disponível em meio magnético.
* `/src`: Código fonte dos experimentos realizados.
* `/apresentacao`: Slides da apresentação do projeto.

## Requisitos
É recomendado executar este projeto por meio do _docker container_ configurado. Para tal, é necessário ter o `Docker` instalado.


## Configuração do ambiente
Para criar e executar o _docker container_, execute os comandos abaixo em um terminal a partir da pasta `src`:

```
docker build -t rep-learning .
docker run -it --rm -v "$(pwd):/home/rep_learning" rep-learning bash
```

No container, instale os pacotes necessários com o comando abaixo:

```
pip install -r requirements.txt
```

## Conjuntos de Dados
Os conjuntos de dados empregados neste projeto são o `Cora`, `Pubmed` e `Reddit`. Apesar de existirem diversas fontes para download na internet, é necessário baixá-los deste [repositório](https://drive.google.com/file/d/1nYj0dzFYVvfsaXi294W476L_ptr92dHS/view?usp=sharing) para garantir que estejam no formato esperado deste projeto. Após fazer o download, mova o arquivo para a pasta `src/datasets` e execute o comando abaixo em um terminal:

```
tar -xvf data.tar.xz
```
