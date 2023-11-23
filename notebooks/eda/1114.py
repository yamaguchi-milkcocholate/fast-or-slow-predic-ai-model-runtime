import numpy as np
import pandas as pd
from pathlib import Path


rootdir = Path().resolve().parent
inputdir = rootdir / "data" / "predict-ai-model-runtime"

fileobj = np.load(
    "/home/yamaguchi/kaggle/data/predict-ai-model-runtime/npz_all/npz/layout/nlp/default/train/albert_en_base_batch_size_16_test.npz"
)
node_feat = fileobj["node_feat"]
edge_index = fileobj["edge_index"]

graph = {i: [] for i in range(node_feat.shape[0])}
for edge in edge_index:
    graph[edge[0]].append(edge[1])

# print(graph)
exit()


def dfs_all_paths(graph, node, visited, path, all_paths):
    """
    深さ優先探索を用いてすべての経路を取得する関数

    Parameters:
    - graph: 隣接リスト形式の有向グラフ
    - node: 現在のノード
    - visited: ノードの訪問状態を保持する辞書
    - path: 現在の探索パスを表すリスト
    - all_paths: すべての経路を格納するリスト
    """
    visited[node] = True
    path.append(node)

    # ゴールノードに到達した場合、現在の経路をコピーして結果リストに追加
    if not graph[node]:
        all_paths.append(path.copy())

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs_all_paths(graph, neighbor, visited, path, all_paths)

    # バックトラックして、探索パスから現在のノードを削除
    path.pop()
    visited[node] = False


# グラフの隣接リストを定義
graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": ["F"], "F": []}

# ノードの訪問状態を管理する辞書
visited = {node: False for node in graph}

# DFSを開始するノードを指定
start_node = "A"

# すべての経路を格納するリスト
all_paths = []

# DFSを実行
dfs_all_paths(graph, start_node, visited, [], all_paths)

# 結果を出力
print("すべての経路:")
for path in all_paths:
    print(path)
print(visited)
