import networkx as nx
import matplotlib.pyplot as plt


class ShowGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.pos = [(0, 0), (-2, -2), (0, -2), (2, -2), (1, -4),
                    (2, -4), (1.5, -6), (2.5, -6), (3, -4), (3, -6), (3.5, -6)]
        self.count = 0
        self.labels = {}

    def add_node(self, parent: list, node: str, nodes: dict):
        self.graph.add_node(node, pos=self.pos[self.count])
        self.labels[node] = node
        self.count += 1
        if len(parent) > 0:
            self.graph.add_edge(node, parent[0], path=parent[1])
        for key in nodes.keys():
            self.graph.add_node(self.count, pos=self.pos[self.count])
            self.graph.add_edge(node, self.count, path=key)
            self.labels[self.count] = nodes.get(key)
            self.count += 1

    def draw_graph(self):
        node_pos = nx.get_node_attributes(self.graph, 'pos')
        path = nx.get_edge_attributes(self.graph, 'path')
        nx.draw_networkx(self.graph, node_pos, node_color='darkturquoise', node_size=1200, with_labels=False)
        nx.draw_networkx_labels(self.graph, node_pos, self.labels)
        nx.draw_networkx_edges(self.graph, node_pos, width=2, edge_color='darkturquoise')
        nx.draw_networkx_edge_labels(self.graph, node_pos, edge_color='peru', edge_labels=path)
        plt.axis('off')
        plt.show()
