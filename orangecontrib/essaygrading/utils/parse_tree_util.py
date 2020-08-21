def get_parse_tree_height(root):
    """
    Util function to calculate sentence parse tree height.
    :param root: root sentence element.
    :return: sentence tree height.
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(get_parse_tree_height(x) for x in root.children)
