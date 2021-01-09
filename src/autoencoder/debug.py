"""Author: Augusto Jose
   Utility functions for debugging
"""


def customdir(obj, hidden_attributes=True):
    """custom python dir(), does not return magic methods

    Arguments:
        obj {object} -- [description]

    Keyword Arguments:
        hidden_attributes {bool} -- if False, does not return hidden attributes too (default: {True})

    Returns:
        list -- list of methods and attributes of obj
    """
    prefix = '__' if hidden_attributes else '_'
    return [m for m in dir(obj) if not m.startswith(prefix)]


def findingdir(obj, substring):
    """returns all dir(obj) outputs that matchs with substring

    Arguments:
        obj {object} -- [description]
        substring {str} -- substring to be found

    Returns:
        list -- list of outputs matched
    """
    return [m for m in dir(obj) if m.find(substring) != -1]


def list_to_commented_text(l):
    """Converts a list to commented text
    Example:
        a = ['a', 'b', 'c']
        list_to_commented_text(a)

        >> '# 0 - a# 1 - b# 2 - c'

        One can transform this output pressing "Enter" button before each # to get this:
        # 0 - a
        # 1 - b
        # 2 - c

    Arguments:
        l {list} -- List to be transformed

    Returns:
        str -- list transformed as string
    """
    rows = ['# ' + (str(k) + ' - ' + v) for k,v in enumerate(l)]
    return ''.join(rows)
