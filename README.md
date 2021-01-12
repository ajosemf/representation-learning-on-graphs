# Aprendizado de Representações em Grafos por Redes Neurais: uma Análise Comparativa
Projeto de conclusão de curso de graduação no Centro Federal de Educação Tecnológica Celso Suckow da Fonseca (CEFET/RJ).

- Nome do curso: Sistemas para Internet
- Título do projeto: Aprendizado de Representações em Grafos por Redes Neurais: uma Análise Comparativa
- Nome completo do autor: Augusto José Moreira da Fonseca
- Nome do orientador: Eduardo Bezerra da Silva, D. Sc.
- Data da defesa: 12/2020

## Estrutura de pastas
* `/monografia`: Monografia completa e formatada.
* `/bibliografia`: Bibliografia disponível utilizada no projeto.
* `/src`: Código fonte dos experimentos realizados.
* `/apresentacao`: Slides da apresentação do projeto.

## Visão Geral
Pesquisas recentes têm buscado aplicar redes neurais artificiais a dados estruturados em grafos com o objetivo de treinar modelos capazes de inferir padrões desconhecidos. Apesar das _Graph Neural Networks_ (GNN) serem atualmente a abordagem comum para o treinamento destes modelos, sua desvantagem encontra-se no custo computacional em termos de tempo de treinamento e consumo de memória. O custo computacional das GNNs tipicamente está relacionado ao aprendizado de representações dos objetos, que incorpora não só as informações do próprio objeto mas também as informações dos objetos relacionados. _Autoencoder Neural Networks_ (AENN) são redes que podem ser treinadas para aprender representações com baixo custo computacional, porém com a desvantagem de não incorporar as relações entre os objetos. Este trabalho realizou uma análise experimental comparativa utilizando as duas arquiteturas de redes neurais mencionadas para geração de representações em grafos em tarefas de classificação. Avaliamos essas arquiteturas em termos de tempo de treinamento e inferência, consumo de memória e poder preditivo, utilizando os conjuntos de dados `Cora`, `Pubmed` e `Reddit`. O objetivo foi explorar o _trade-off_ entre custo computacional e poder preditivo entre as arquiteturas citadas. Empregamos o [ClusterGCN](https://github.com/google-research/google-research/tree/master/cluster_gcn) como arquitetura GNN e uma implementação própria de AENN. Os resultados apresentaram custos computacionais de duas a quatro vezes menor para a arquitetura AENN. Apesar do poder preditivo ser maior para o ClusterGCN, o resultado obtido pela AENN no conjunto de dados `Pubmed` foi muito próximo. Este trabalho pôde constatar que o emprego de AENNs no aprendizado de representações pode ser mais adequado ou suficiente na solução de determinados problemas.

## Requisitos
- É recomendado executar este projeto por meio do _docker container_ configurado. Para tal, é necessário ter o [Docker](https://www.docker.com/) instalado.
- O consumo de memória RAM para executar os experimentos pode ultrapassar os 32 GiB. É recomendando processá-los em computadores com 64 GiB RAM.

## Configuração do ambiente
- Para criar e executar o _docker container_, execute os comandos abaixo em um terminal a partir da pasta `src`:

```
docker build -t rep-learning .
docker run -it --rm -v "$(pwd):/home/rep_learning" rep-learning bash
```

- No container, instale os pacotes necessários com o comando abaixo:

```
pip install -r requirements.txt
```

## Conjuntos de Dados
- Os conjuntos de dados empregados neste projeto são o `Cora`, `Pubmed` e `Reddit`. Apesar de existirem diversas fontes para download na internet, é necessário baixá-los deste [repositório](https://drive.google.com/file/d/1nYj0dzFYVvfsaXi294W476L_ptr92dHS/view?usp=sharing) para garantir que estejam no formato esperado deste projeto. 
- Após fazer o download, mova o arquivo para a pasta `src/datasets` e execute o comando abaixo em um terminal:

```
tar -xvf data.tar.xz
```

## Experimento
- Os melhores hiperparâmetros para cada par arquitetura/conjunto de dados foram obtidos pelo método _grid search_. Consulte a [monografia](monografia/monografia.pdf) para maiores detalhes.
- Para executar os experimentos (já com os melhores hiperparâmetros fixados), entre nas pastas `src/autoencoder` e `src/cluster_gcn` e execute em um terminal:

```
./run_mprof.sh
```

## Análise
- A análise dos experimentos é apresentada no [jupyter notebook](src/analysis/analysis.ipynb) disponível na pasta `src/analysis`.
