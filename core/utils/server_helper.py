import socket

def in_server():
    host_name = socket.gethostname().split(".")[0]
    return "elim" not in host_name


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except:
        return False