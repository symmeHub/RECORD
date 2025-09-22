"""
Extended Qt widgets and helpers used by the GUI.

Includes a checkable combo box, a file-picking line edit composite,
and small popup helpers for confirmations and errors.
"""

from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    Qt,
    QSize,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QGradient,
    QGuiApplication,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
    QStandardItem,
)
from PyQt6.QtWidgets import QMessageBox, QComboBox


class CheckableComboBox(QComboBox):
    """QComboBox that supports multiple selection via checkable items."""

    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.closeOnLineEditClick = False
        self.lineEdit().installEventFilter(self)
        self.model().dataChanged.connect(self.updateLineEditField)

    # def eventFilter(self, widget, event: QEvent) -> bool:
    #     if widget == self.lineEdit():
    #         if event.type() == QEvent.Type.MouseButtonPress:
    #             if self.closeOnLineEditClick:
    #                 print("hide")
    #                 self.hidePopup()
    #             else:
    #                 print("show")
    #                 self.showPopup()
    #                 self.closeOnLineEditClick = True
    #             return True
    #     return super().eventFilter(widget, event)

    def addItems(self, items, itemList=None):
        """Add a list of items with optional userData list."""
        itemList = [] if itemList is None else itemList
        for indx, text in enumerate(items):
            try:
                data = itemList[indx]
            except IndexError:
                data = None
            self.addItem(text, data)

    def addItem(self, text, userData=None):
        """Insert a checkable item with optional userData."""
        item = QStandardItem()
        item.setText(text)
        if not userData is None:
            item.setData(userData)

        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def updateLineEditField(self):
        """Reflect checked items as a comma-separated list in the line edit."""
        text_container = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                text_container.append(self.model().item(i).text())

        text_string = ", ".join(text_container)
        self.lineEdit().setText(text_string)

    def get_selected_items(self):
        """Return the list of checked item texts."""
        selected_items = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                selected_items.append(self.model().item(i).text())
        return selected_items

    def checkItemFromName(self, name):
        """Check the item whose text matches name (if found)."""
        for i in range(self.model().rowCount()):
            if self.model().item(i).text() == name:
                self.model().item(i).setCheckState(Qt.CheckState.Checked)
                break

    def uncheckItemFromName(self, name):
        """Uncheck the item whose text matches name (if found)."""
        for i in range(self.model().rowCount()):
            if self.model().item(i).text() == name:
                self.model().item(i).setCheckState(Qt.CheckState.Unchecked)
                break


class CustomLinedit(QtWidgets.QWidget):
    """Composite widget: QLineEdit with a button to open a file dialog."""

    _dial_msg = "Select a file"

    def __init__(
        self,
        parent=None,
        dial_root_dir: str = "",
        allowed_extensions: str = "",
        *args,
        **kwargs,
    ):
        super(CustomLinedit, self).__init__(parent)
        self._dial_root_dir = dial_root_dir
        self._allowed_extensions = allowed_extensions
        self.ui_setup()
        self.connections()

    def ui_setup(self):
        """Set up the internal layout and child widgets."""
        # Add a hori layout to the main layout
        self.horiLayout = QtWidgets.QHBoxLayout(self)
        # Add a line edit and a button to the hori layout
        self.lineEdit = QtWidgets.QLineEdit()
        self.button = QtWidgets.QPushButton("...")
        self.horiLayout.addWidget(self.lineEdit)
        self.horiLayout.addWidget(self.button)
        self.horiLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.horiLayout.setContentsMargins(0, 0, 0, 0)

    def connections(self):
        """Connect the browse button to the dialog-opening slot."""
        # Connect the button to a slot that will open a dialog box
        self.button.clicked.connect(self.openDialogBox)

    # Opens a dialog box
    def openDialogBox(self):
        """Open a QFileDialog and set the selected path in the line edit."""
        self.seleted_fname = QtWidgets.QFileDialog.getOpenFileName(
            self, self._dial_msg, self._dial_root_dir, self._allowed_extensions
        )
        # Update the line edit field with the selected file name
        self.lineEdit.setText(self.seleted_fname[0])

    @property
    def text(self):
        """Return the current text of the line edit."""
        return self.lineEdit.text()

    @property
    def dial_msg(self):
        """Dialog title/message string."""
        return self._dial_msg

    @dial_msg.setter
    def dial_msg(self, msg):
        self._dial_msg = msg

    @property
    def dial_root_dir(self):
        """Root directory used by the file dialog."""
        return self._dial_root_dir

    @dial_root_dir.setter
    def dial_root_dir(self, root_dir):
        # check if the root dir is a directory and set it as the root dir
        # for the dialog box
        import os

        if os.path.isdir(root_dir):
            self._dial_root_dir = root_dir
        else:
            raise ValueError(f"The specified directory: {root_dir} is not a directory")

    @property
    def allowed_extensions(self):
        """Allowed file extensions filter used by the file dialog."""
        return self._allowed_extensions

    @allowed_extensions.setter
    def allowed_extensions(self, extensions):
        self._allowed_extensions = extensions


def show_popup(title: str, msg: str, parent=None):
    """Confirmation popup returning True for Yes, False for No."""
    # Crée une boîte de dialogue de type "Question"
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setWindowTitle(title)
    msg_box.setText(msg)

    # Ajout des boutons "Yes" et "No"
    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)

    # Affiche la boîte de dialogue et capture la réponse
    result = msg_box.exec()

    if result == QMessageBox.StandardButton.Yes:
        if parent:
            if hasattr(parent, "logger"):
                parent.logger.info("User chose: Yes")
        return True
    else:
        if parent:
            if hasattr(parent, "logger"):
                parent.logger.info("User chose: No")
        return False


def error_popup(title: str, msg: str, parent=None):
    """Error popup displaying a critical message with an OK button."""
    # Crée une boîte de dialogue de type "Question"
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(msg)

    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.setDefaultButton(QMessageBox.StandardButton.Ok)

    # Affiche la boîte de dialogue et capture la réponse
    result = msg_box.exec()
    if parent:  # type: ignore
        if hasattr(parent, "logger"):
            parent.logger.info("User aknowledge error")


if __name__ == "__main__":
    pass
