from . import uct_helper_utils
from . import uct_parameters

from operator import attrgetter
from math import  sqrt, log
from random import uniform


class TreeNode(object):
    __tree_size = 0  # TODO keeps the overal tree size for debugging

    def __init__(self, edge_action=None, parent_node=None):
        self.__tree_size += 1 # debugging purpose
        self.win_rate = 0.0
        self.visit_count = 1.0
        self.depth = (parent_node.depth+1) if parent_node is not None else 0
        self.edge_action = edge_action  # set this to the edge action (function pointer respective c++) that generates self node
        self.parent_node = parent_node
        self.current_max_child_size = len(uct_helper_utils.all_actions) # TODO set this will change based on environment changes
        self.available_actions = []
        self.children_nodes = []  # this should be filled on demand for memory efficiency

    def add_child_node(self,new_edge_action):
        new_born = TreeNode(new_edge_action,self) # pass self as parent of new node
        self.children_nodes.append(new_born) # actually adds the new child to the children list
        return new_born

    def is_full_explored(self): # Currenly fixed to full action_space TODO this must be called after populating the number of moves
        return len(self.children_nodes) == len(self.available_actions) # max_children_size is dynamic

    def best_ucb_child(self):
        return max(self.children_nodes, key=ucb_value)

    def best_visit_child(self):
        return max(self.children_nodes, key=attrgetter('visit_count'))

    def _tree_printer(self):
        print('depth at ', self.depth, 'action ', self.edge_action)
        if len(self.children_nodes) > 0:
            for i in range(len(self.children_nodes)):
                print("it has ", len(self.children_nodes), 'kids')
                if (self.depth < 3):
                    self.treePrinter(self.children_nodes[i])


def ucb_value(node): #TODO populate difference action selection methods for experiments later
    return node.win_rate / node.visit_count + uct_parameters.C * sqrt(log(node.parent_node.visit_count) / node.visit_count)
