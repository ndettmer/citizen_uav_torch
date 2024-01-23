import sys

from PySide6.QtWidgets import QApplication

from SampleSelector.selector import SelectorWidget


def iter_files(data_dir, argv):
    """
    Load ImageFolder and metadata.csv
    add a column to metadata called "hand_picked"
    iterate through the samples and show them in a window with 2 option buttons (with hot keys)
      1. pick sample
      2. ditch sample
    if a sample is picked hand_picked for this sample is set to True
    if a sample is ditched hand_picked for this sample is set to False
    """

    app = QApplication(argv)
    selector = SelectorWidget(data_dir)
    selector.show()

    sys.exit(app.exec())
