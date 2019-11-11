from classes.id3 import Id3
from classes.networkx import ShowGraph
import operator


def recursion(parent_node, parent_routes):
    if len(id3.data_headers) < 1:
        return
    data = id3.data
    for key in parent_routes.keys():
        print("Current Path: ", key)
        id3.data = data[data.eval(parent_node) == key]
        selected_node, routes, finished = id3_work()
        node = selected_node if selected_node is not None else 'END!'
        graph.add_node([parent_node, key], node, finished)
        if selected_node is not None:
            recursion(selected_node, dict(sorted(routes.items(), key=operator.itemgetter(0), reverse=True)))
    return


def id3_work():
    target_entropy = get_classifier_entropy()
    selected_root_node, delete_rows = id3.root_node_election(target_entropy)
    print("The selected root node is: ", selected_root_node)

    delete_row = {}
    routes = {}
    if selected_root_node in delete_rows:
        for row, (entropy, value) in delete_rows[selected_root_node].items():
            if entropy == 1 or entropy == 0:
                delete_row[row] = value
            else:
                routes[row] = entropy
    print("Routes Next: ", routes)
    print("Routes End: ", delete_row)
    if selected_root_node is None:
        return selected_root_node, routes, delete_row
    id3.data = id3.reduce_training_data_set(id3.data, delete_row.keys(), selected_root_node)
    id3.data_headers.pop(id3.data_headers.index(selected_root_node))
    return selected_root_node, routes, delete_row


def get_classifier_entropy():
    entropy, not_used = id3.entropy(id3.root)
    return entropy


graph = ShowGraph()
id3 = Id3()
root_node, root_routes, finish = id3_work()
graph.add_node([], root_node, finish)
recursion(root_node, root_routes)
graph.draw_graph()
