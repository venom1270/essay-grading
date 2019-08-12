def get_parse_tree_height(root):
    if not list(root.children):
        return 1
    else:
        return 1 + max(get_parse_tree_height(x) for x in root.children)