import os
import sys
import signal
import socket
import argparse
from nicegui import ui, app
from ifcview.lib.interactions import GuiInteraction
from ifcview import __version__

display_msg = """
===============================================================================

Browser-based viewer for imaging flow cytometry (IFC) data stored in HDF5

===============================================================================

Type: ifcview to run the software
Exit the software by pressing: Ctrl + C

===============================================================================
"""


def handle_shutdown(main_app):
    """
    Handle the shutdown process of the application.
    """
    if main_app is not None:
        main_app.shutdown()
    print("\n===============")
    print(" Exit the app!")
    print("===============\n")


def signal_handler(*args):
    """
    Handle the signal for graceful shutdown of the application.

    Parameters
    ----------
    *args
        Variable length argument list.
    """
    sys.exit(0)


def check_port(port):
    """
    Check if a given port is available.

    Parameters
    ----------
    port : int
        The port number to check.

    Returns
    -------
    bool
        True if the port is already in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=display_msg,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("--port", type=int, default=8180,
                        help="Specify the port to run ifcview on "
                             "(default: 8180)")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to start the application.
    """
    args = parse_args()
    if check_port(args.port):
        print("\n!!! Port {} is already in use. Please specify a "
              "different port using --port !!!\n".format(args.port))
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)  # Back-up shutdown
    try:
        ifcview_app = None

        @ui.page('/')
        def main_page():
            global ifcview_app
            ifcview_app = GuiInteraction()

        app.on_shutdown(lambda: handle_shutdown(ifcview_app))
        os.environ["NO_NETIFACES"] = "True"
        app.on_startup(
            lambda: print("Access ifcview at urls: {}".format(
                app.urls.union())))
        ui.run(reload=False, title="IFC HDF5 Viewer", port=args.port,
               show_welcome_message=False)
    except Exception as error:
        print(f"An error occurred: {error}")
        sys.exit(0)


# Only the parent launch (``__main__``) starts the server. ``run.cpu_bound``
# spawns worker processes that re-import this module as ``__mp_main__``; those
# must NOT re-enter main() (it would try to rebind the port and break the
# process pool), so ``__mp_main__`` is deliberately excluded here.
if __name__ == "__main__":
    main()
