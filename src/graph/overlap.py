import json
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from scipy.spatial.distance import cosine
from heapq import heappush, heappop
from grakel import Graph, kernels
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

# 假设这个 llm 函数是由外部提供的，输入是 prompt，输出是自然语言描述的相对位置
def llm(prompt):
    # 这里只是一个示例实现，实际情况下你需要调用真实的 LLM 服务
    # 例如：return "The position of node B relative to node A is [1, 0]."
    return "The position of node B relative to node A is [1, 0]."

class GraphMatcher:
    def __init__(self, graph1, graph2, llm=None):
        self.graph1 = graph1
        self.graph2 = graph2
        self.llm = llm

        G1 = nx.DiGraph()
        for node in graph1['nodes']:
            G1.add_node(node['id'], position=node['position'])
        for edge in graph1['edges']:
            G1.add_edge(edge['source'], edge['target'], type=edge['type'])

        G2 = nx.DiGraph()
        for node in graph2['nodes']:
            G2.add_node(node['id'])  # No position assigned initially
        for edge in graph2['edges']:
            G2.add_edge(edge['source'], edge['target'], type=edge['type'])
        self.G1 = G1
        self.G2 = G2
        self.matching_threshold = 0.9

    def node_similarity(self, G1, G2, similarity_matrix):
        nodes_G1 = set(G1.nodes)
        nodes_G2 = set(G2.nodes)
        common_nodes = nodes_G1.intersection(nodes_G2) #just nodes with the same names

        if len(nodes_G2) == 0 or similarity_matrix.shape[0] == 0:
            node_overlap_ratio = 0.0
        else:
            node_overlap_ratio = similarity_matrix.mean()

        degree_sim = 0.0
        if self.common_pairs:
            for pair in self.common_pairs:
                degree_G1 = G1.degree(list(G1.nodes)[pair["scene_idx"]])
                degree_G2 = G2.degree(list(G2.nodes)[pair["goal_idx"]])
                degree_diff = abs(degree_G1 - degree_G2)
                max_degree = max(degree_G1, degree_G2)
                if max_degree > 0:
                    degree_sim += 1.0 - (degree_diff / max_degree)
            degree_sim /= len(self.common_pairs)

        node_sim = (node_overlap_ratio + degree_sim) / 2.0
        return node_sim

    def edge_similarity(self, G1, G2, similarity_matrix):
        edges_G1 = {frozenset((u, v, frozenset(data.items()))) for u, v, data in G1.edges(data=True)}
        edges_G2 = {frozenset((u, v, frozenset(data.items()))) for u, v, data in G2.edges(data=True)}
        common_edges = edges_G1.intersection(edges_G2)

        if len(edges_G2) == 0:
            return 1.0
        elif similarity_matrix.shape[0] == 0:
            return 0.0
        else:
            return similarity_matrix.mean()

    def graph_edit_distance_heuristic(self, G1, G2):
        node_diff = abs(len(G1) - len(G2))
        edge_diff = abs(G1.number_of_edges() - G2.number_of_edges())
        return node_diff + edge_diff

    def apply_operation(self, operation, G, **kwargs):
        if operation == 'add_node':
            G.add_node(kwargs['node'])
        elif operation == 'remove_node':
            G.remove_node(kwargs['node'])
        elif operation == 'add_edge':
            G.add_edge(*kwargs['edge'])
        elif operation == 'remove_edge':
            G.remove_edge(*kwargs['edge'])

    def overlap(self, goal_node_features, goal_edge_features, scene_node_features, scene_edge_features, black_list_node, black_list_edge):
        G1 = self.G1
        G2 = self.G2

        scene_node_degrees = torch.zeros(scene_node_features.shape[0], 1)
        for idx, nodei in enumerate(G1):
            scene_node_degrees[idx,0] = G1.degree(nodei)
        scene_node_features = torch.cat([scene_node_features, scene_node_degrees], dim = 1)
        scene_node_features = F.normalize(scene_node_features, p=2, dim=1)

        goal_node_degrees = torch.zeros(goal_node_features.shape[0], 1)
        for idx, nodei in enumerate(G2):
            goal_node_degrees[idx,0] = G2.degree(nodei)
        goal_node_features = torch.cat([goal_node_features, goal_node_degrees], dim = 1)
        goal_node_features = F.normalize(goal_node_features, p=2, dim=1)

        scene_edge_features = F.normalize(scene_edge_features, p=2, dim=1)
        goal_edge_features = F.normalize(goal_edge_features, p=2, dim=1)

        node_similarity_matrix = torch.matmul(scene_node_features, goal_node_features.T)
        node_similarity_matrix_threshold = node_similarity_matrix.masked_fill(node_similarity_matrix<self.matching_threshold, -1)
        node_similarity_matrix_threshold[black_list_node] = -1
        # print("<><><><><><> node_similarity_matrix:", node_similarity_matrix)
        # print("<><><><><><> node_similarity_matrix_threshold:", node_similarity_matrix_threshold)

        edge_similarity_matrix = torch.matmul(scene_edge_features, goal_edge_features.T)
        edge_similarity_matrix[black_list_edge] = -1

        self.common_pairs = self.find_common_nodes(G1, G2, similarity_matrix = node_similarity_matrix_threshold)

        node_sim = self.node_similarity(G1, G2, similarity_matrix=node_similarity_matrix)
        edge_sim = self.edge_similarity(G1, G2, similarity_matrix=edge_similarity_matrix)

        # print("<><><><><><> node_similarity:", node_sim)
        # print("<><><><><><> edge_similarity:", edge_sim)

        combined_sim = (node_sim + edge_sim) / 2
        return combined_sim

    def find_common_nodes(self, G1, G2, similarity_matrix):
        """ Find common nodes between two graphs """
        cost_matrix = -similarity_matrix.detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid_matches = []
        self.common_nodes_goal = []
        self.common_nodes_scene = []
        self.scene_node_to_goal_node = {}
        self.goal_node_to_scene_node = {}
        for r, c in zip(row_ind, col_ind):
            if similarity_matrix[r, c] > 0:
                valid_matches.append({"scene_idx":r, "goal_idx":c, "scene_captain":list(G1.nodes)[r], "goal_captain":list(G2.nodes)[c], "weight":similarity_matrix[r, c]})
                self.common_nodes_goal.append(list(G2.nodes)[c])
                self.common_nodes_scene.append(list(G1.nodes)[r])
                self.scene_node_to_goal_node[list(G1.nodes)[r]] = list(G2.nodes)[c]
                self.goal_node_to_scene_node[list(G2.nodes)[c]] = list(G1.nodes)[r]
        # print(f"<><><><><><> valid_matches", valid_matches)
        return valid_matches # G1.nodes only contains the captain of each node

    def calculate_relative_positions(self, graph, common_nodes):
        """ Calculate relative positions of nodes within a graph using LLM for unknown positions """
        relative_positions = {}
        if not common_nodes:
            return relative_positions

        # Select the first common node as the reference point
        ref_node = list(common_nodes)[0]

        # Calculate the relative positions for all other common nodes
        for node in common_nodes:
            if node == ref_node:
                continue
            # Use LLM to predict the relative position
            prompt = f"Given the following information: {ref_node} and {node}. Please provide the relative position of {node} with respect to {ref_node} in the format [x, y]."
            response = llm(prompt)
            # Parse the response to get the relative position
            try:
                rel_pos_str = response.split("[")[1].split("]")[0]
                rel_pos = [float(coord.strip()) for coord in rel_pos_str.split(",")]
                key = tuple((node, ref_node))
                relative_positions[key] = rel_pos
            except (IndexError, ValueError) as e:
                print(f"Failed to parse LLM response: {response}")
                print(f"Error: {e}")

        positions = {}
        ref_node = next(iter(relative_positions))[1]
        positions[ref_node] = [0, 0]
        for node_c in relative_positions.keys():
            positions[node_c[0]] = relative_positions[node_c]

        return positions

    def predict_remaining_node_positions(self, common_pairs, positions, scene_graph):
        if len(common_pairs) < 2:
            raise ValueError("At least two common nodes are required to predict the position of other nodes")

        common_nodes_scene = [list(scene_graph.nodes)[cpi['scene_idx']] for cpi in common_pairs]

        ref_points = sorted(list(common_nodes_scene))[:2]
        ref_point_1, ref_point_2 = ref_points

        ref_pos_1 = np.array(scene_graph.nodes[ref_point_1]['position'])
        ref_pos_2 = np.array(scene_graph.nodes[ref_point_2]['position'])

        ref_vec_scene = ref_pos_2 - ref_pos_1
        try:
            ref_vec_subgraph = np.array(positions[self.scene_node_to_goal_node[ref_point_1]]) - np.array(positions[self.scene_node_to_goal_node[ref_point_2]])
        except KeyError as e:
            print(f"KeyError: {e}")
            return {}

        if np.allclose(ref_vec_subgraph, 0):
            ref_vec_subgraph = np.array([1000., 1000.])

        angle = np.arctan2(ref_vec_scene[1], ref_vec_scene[0]) - np.arctan2(ref_vec_subgraph[1], ref_vec_subgraph[0])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        scale_factor = np.linalg.norm(ref_vec_scene) / np.linalg.norm(ref_vec_subgraph)

        predicted_positions = {}
        for node, rel_pos in positions.items():
            if node not in ref_points:
                relative_position = np.array(rel_pos) - np.array(positions[self.scene_node_to_goal_node[ref_point_1]])
                transformed_pos = np.dot(rotation_matrix, np.array(relative_position) * scale_factor) + ref_pos_1
                predicted_positions[node] = transformed_pos

        if len(predicted_positions) > 0:
            rel_pos_list = []
            for node, rel_pos in predicted_positions.items():
                rel_pos_list.append(rel_pos)
            position = sum(rel_pos_list) / len(rel_pos_list)
        else:
            rel_pos_list = [ref_pos_1, ref_pos_2]
            position = sum(rel_pos_list) / len(rel_pos_list)
        position = list(position)
        return position
