import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QComboBox, QCompleter


def polygon2bbox(pts):
    x = np.min(pts[:, 0])
    y = np.min(pts[:, 1])
    w = np.max(pts[:, 0]) - x
    h = np.max(pts[:, 1]) - y
    return [int(x), int(y), int(w), int(h)]

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label

def completion(word_list, widget, i=True):
    """ Autocompletion of sender and subject """
    word_set = set(word_list)
    completer = QCompleter(word_set)
    if i:
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    else:
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    widget.setCompleter(completer)


class Autocomplete(QComboBox):
    def __init__(self, items, parent=None, i=False, allow_duplicates=True):
        super(Autocomplete, self).__init__(parent)
        self.items = items
        self.insensitivity = i
        self.allowDuplicates = allow_duplicates
        self.init()

    def init(self):
        self.setEditable(True)
        self.setDuplicatesEnabled(self.allowDuplicates)
        self.addItems(self.items)
        self.setAutocompletion(self.items, i=self.insensitivity)

    def setAutocompletion(self, items, i):
        completion(items, self, i)
