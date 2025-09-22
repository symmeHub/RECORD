"""
Console log dock that captures Python logging output and renders ANSI-colored
messages as styled HTML in a read-only text widget.
"""

from record.gui.ui_interfaces.DockWidget_console_ui_mod import Ui_DockWidget_console

from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import (
    QDockWidget,
    QPlainTextEdit,
    QTableWidgetItem,
    QStyledItemDelegate,
    QLineEdit,
)
import re
import html
import logging

# Table de correspondance ANSI vers HTML
ANSI_COLOR_MAP = {
    "30": "black",
    "31": "red",
    "32": "green",
    "33": "yellow",
    "34": "blue",
    "35": "magenta",
    "36": "cyan",
    "37": "white",
    "90": "gray",
    "91": "lightcoral",
    "92": "lightgreen",
    "93": "lightyellow",
    "94": "lightblue",
    "95": "violet",
    "96": "lightcyan",
    "97": "white",
}


def ansi_to_html(text):
    """Convert ANSI escape sequences to styled HTML, with optional centering.

    Replaces newlines by <br>, maps color codes to CSS colors, and ensures
    open spans are properly closed.
    """
    text = html.escape(text)  # Échappe les caractères spéciaux (<, >, etc.)

    # Détection du centrage avec [[CENTER]]
    is_centered = False
    if "#CENTER" in text:
        is_centered = True
        text = text.replace("#CENTER", "", 1)  # On enlève la balise

    # Remplacement des retours à la ligne par <br>
    text = text.replace("\r\n", "<br>").replace("\n", "<br>")

    ansi_escape = re.compile(r"\033\[(\d+(?:;\d+)*)m")

    open_tags = []

    def replace_match(match):
        nonlocal open_tags
        codes = match.group(1).split(";")

        if "0" in codes:  # Réinitialisation des styles
            open_tags.clear()
            return "</span>"

        color = None
        for code in codes:
            if code in ANSI_COLOR_MAP:
                color = ANSI_COLOR_MAP[code]

        if color:
            open_tags.append(color)
            return f'<span style="color: {color};">'

        return ""

    text = ansi_escape.sub(replace_match, text)

    # Fermeture des balises ouvertes restantes
    while open_tags:
        text += "</span>"
        open_tags.pop()

    # Si le texte doit être centré, on l'encapsule dans un <div>
    if is_centered:
        text = f'<div style="text-align: center;">{text}</div>'

    return text


class QPlainTextEditLogger(logging.Handler, QtCore.QObject):
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self)

        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

        self.widget.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #202020;
                color: #B0B0B0;
                font-size: 12px;
                border: none;
            }
        """
        )

        # Connect signal to GUI-safe slot
        self.log_signal.connect(self._append_log)

    def emit(self, record):
        """Format the record to HTML and emit a signal for thread-safe append."""
        msg = self.format(record)
        html_msg = ansi_to_html(msg)
        self.log_signal.emit(html_msg)  # Call in GUI thread

    @QtCore.pyqtSlot(str)
    def _append_log(self, html_msg):
        """Append an HTML message to the text widget (GUI thread)."""
        self.widget.appendHtml(html_msg)


class ConsoleLogDockWidget(QDockWidget):
    """Dock widget wrapping a QTextEdit with ANSI-to-HTML logging handler."""

    def __init__(self, parent):
        super(ConsoleLogDockWidget, self).__init__(parent)

        # UI Setups
        self.ui = Ui_DockWidget_console()
        self.ui.setupUi(self)
        self.init_Ui_Settings()

        # Connection Setups
        self.connections()

    def init_Ui_Settings(self):
        """Set styles and add the logger widget into the layout."""
        self.setStyleSheet("QDockWidget::title" "{" "background : darkgray;" "}")
        self.ui.textEdit_console = QPlainTextEditLogger(self)
        self.ui.textEdit_console.widget.verticalScrollBar().setValue(
            self.ui.textEdit_console.widget.verticalScrollBar().maximum()
        )
        self.ui.verticalLayout.addWidget(self.ui.textEdit_console.widget)

    def connections(self):
        pass


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        """Emit written text as a Qt signal."""
        self.textWritten.emit(str(text))

    def flush(self):
        """No-op for compatibility with file-like API."""
        pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    DockWidget = ConsoleLogDockWidget(None)
    # ui = Ui_DockWidget_IMU_server_manager()
    # ui.setupUi(DockWidget)
    DockWidget.show()
    DockWidget.resize(400, 300)
    sys.exit(app.exec())
