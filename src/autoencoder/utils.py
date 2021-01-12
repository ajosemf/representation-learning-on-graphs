import json
import numpy as np

from tensorflow import gfile
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler


def load_graphsage_data(dataset_path, dataset_name, normalize=True):
    """Load GraphSAGE data."""
    
    # carregando os arquivos
    graph_json = json.load(
        gfile.Open('{0}/{1}/{1}-G.json'.format(dataset_path, dataset_name)))

    graph_nx = json_graph.node_link_graph(graph_json)
    
    id_map = json.load(
        gfile.Open('{0}/{1}/{1}-id_map.json'.format(dataset_path, dataset_name)))

    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
    
    class_map = json.load(
        gfile.Open('{0}/{1}/{1}-class_map.json'.format(dataset_path, dataset_name)))

    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
                 for k, v in class_map.items()}
    
    # removendo nós que não possuem classes mapeadas
    broken_count = 0
    to_remove = []
    for node in graph_nx.nodes():
        if node not in id_map:
            to_remove.append(node)
            broken_count += 1

    for node in to_remove:
        graph_nx.remove_node(node)

    print('Removed {} nodes that lacked proper annotations due to networkx versioning issues'.
          format(broken_count))

    
    # carregando features
    feats = np.load(gfile.Open(
        '{0}/{1}/{1}-feats.npy'.format(dataset_path, dataset_name),
        'rb')).astype(np.float32)
    print('Loaded data, now preprocessing..')
    
    
    # separando conjuntos train, val, test
    num_data = len(id_map)

    val_ids = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['val']],
        dtype=np.int32)

    test_ids = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['test']],
        dtype=np.int32)

    is_train = np.ones((num_data), dtype=np.bool)

    is_train[val_ids] = False
    is_train[test_ids] = False

    train_ids = np.array([n for n in range(num_data) if is_train[n]],
                          dtype=np.int32)

    
    # processando o target
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
        labels[id_map[k], class_map[k]] = 1
        

    # Normalizando as features
    normalize = True
    if normalize:
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    print('Data processed')
    
    return num_data, feats, labels, train_ids, val_ids, test_ids