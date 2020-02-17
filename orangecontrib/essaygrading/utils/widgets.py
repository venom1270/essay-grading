'''
Taken from: https://github.com/biolab/orange3-text/blob/master/orangecontrib/text/widgets/utils/widgets.py
'''


import os

from AnyQt.QtWidgets import (QComboBox, QWidget, QHBoxLayout,
                         QSizePolicy, QLineEdit, QDoubleSpinBox,
                         QSpinBox, QTextEdit, QDateEdit, QGroupBox,
                             QPushButton, QStyle, QFileDialog, QLabel,
                             QGridLayout, QCheckBox, QStackedLayout)

from AnyQt.QtCore import pyqtSignal, Qt, QSize


from orangecontrib.text.corpus import get_sample_corpora_dir


class FileWidget(QWidget):
    on_open = pyqtSignal(str)

    # TODO consider removing directory_aliases since it is not used any more
    def __init__(self, dialog_title='', dialog_format='',
                 start_dir=os.path.expanduser('~/'),
                 icon_size=(12, 20), minimal_width=200,
                 browse_label='Browse', on_open=None,
                 reload_button=True, reload_label='Reload',
                 recent_files=None, directory_aliases=None,
                 allow_empty=True, empty_file_label='(none)'):
        """ Creates a widget with a button for file loading and
        an optional combo box for recent files and reload buttons.
        Args:
            dialog_title (str): The title of the dialog.
            dialog_format (str): Formats for the dialog.
            start_dir (str): A directory to start from.
            icon_size (int, int): The size of buttons' icons.
            on_open (callable): A callback function that accepts filepath as the only argument.
            reload_button (bool): Whether to show reload button.
            reload_label (str): The text displayed on the reload button.
            recent_files (List[str]): List of recent files.
            directory_aliases (dict): An {alias: dir} dictionary for fast directories' access.
            allow_empty (bool): Whether empty path is allowed.
        """
        super().__init__()
        self.dialog_title = dialog_title
        self.dialog_format = dialog_format
        self.start_dir = start_dir

        # Recent files should also contain `empty_file_label` so
        # when (none) is selected this is stored in settings.
        self.recent_files = recent_files if recent_files is not None else []
        self.directory_aliases = directory_aliases or {}
        self.allow_empty = allow_empty
        self.file_combo = None
        self.empty_file_label = empty_file_label
        if self.empty_file_label not in self.recent_files \
                and (self.allow_empty or not self.recent_files):
            self.recent_files.append(self.empty_file_label)

        self.check_existence()
        self.on_open.connect(on_open)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if recent_files is not None:
            self.file_combo = QComboBox()
            self.file_combo.setMinimumWidth(minimal_width)
            self.file_combo.activated[int].connect(self.select)
            self.update_combo()
            layout.addWidget(self.file_combo)

        self.browse_button = QPushButton(browse_label)
        self.browse_button.setFocusPolicy(Qt.NoFocus)
        self.browse_button.clicked.connect(self.browse)
        self.browse_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.setIconSize(QSize(*icon_size))
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.browse_button)

        if reload_button:
            self.reload_button = QPushButton(reload_label)
            self.reload_button.setFocusPolicy(Qt.NoFocus)
            self.reload_button.clicked.connect(self.reload)
            self.reload_button.setIcon(self.style()
                                       .standardIcon(QStyle.SP_BrowserReload))
            self.reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.reload_button.setIconSize(QSize(*icon_size))
            layout.addWidget(self.reload_button)

    def browse(self, start_dir=None):
        start_dir = start_dir or self.start_dir
        path, _ = QFileDialog().getOpenFileName(self, self.dialog_title,
                                                start_dir, self.dialog_format)

        if path and self.recent_files is not None:
            if path in self.recent_files:
                self.recent_files.remove(path)
            self.recent_files.insert(0, path)
            self.update_combo()

        if path:
            self.open_file(path)

    def select(self, n):
        if self.file_combo is None:
            return
        name = self.file_combo.currentText()
        if name == self.empty_file_label:
            del self.recent_files[n]
            self.recent_files.insert(0, self.empty_file_label)
            self.update_combo()
            self.open_file(self.empty_file_label)
        elif name in self.directory_aliases:
            self.browse(self.directory_aliases[name])
        elif n < len(self.recent_files):
            name = self.recent_files[n]
            del self.recent_files[n]
            self.recent_files.insert(0, name)
            self.update_combo()
            self.open_file(self.recent_files[0])

    def update_combo(self):
        """ Sync combo values to the changes in self.recent_files. """
        if self.recent_files is not None and self.file_combo is not None:
            self.file_combo.clear()
            for i, file in enumerate(self.recent_files):
                # remove (none) when we have some files and allow_empty=False
                if file == self.empty_file_label and \
                        not self.allow_empty and len(self.recent_files) > 1:
                    del self.recent_files[i]
                else:
                    self.file_combo.addItem(os.path.split(file)[1])

            for alias in self.directory_aliases.keys():
                self.file_combo.addItem(alias)

    def reload(self):
        if self.recent_files:
            self.select(0)

    def check_existence(self):
        if self.recent_files:
            to_remove = []
            for file in self.recent_files:
                doc_path = os.path.join(get_sample_corpora_dir(), file)
                exists = any(os.path.exists(f) for f in [file, doc_path])
                if file != self.empty_file_label and not exists:
                    to_remove.append(file)
            for file in to_remove:
                self.recent_files.remove(file)

    def open_file(self, path):
        self.on_open.emit(path if path != self.empty_file_label else '')

    def get_selected_filename(self):
        if self.recent_files:
            return self.recent_files[0]
        else:
            return self.empty_file_label
