"""In-place Qt language bindings and consistently localized dialogs."""
from PySide6.QtCore import Qt, QObject, QTranslator, QLibraryInfo, QSignalBlocker, Signal
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QAbstractButton,
    QComboBox, QGroupBox, QTabWidget, QWidget, QMessageBox as QtMessageBox,
    QFileDialog as QtFileDialog)
from i18n import EN, message, render


class QtTranslations(QTranslator):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.chinese = QTranslator(self)
        self.chinese.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.TranslationsPath))

    def isEmpty(self):
        return False

    def translate(self, context, sourceText, disambiguation=None, n=-1):
        if self.owner.language == "en":
            return sourceText
        return self.chinese.translate(context, sourceText, disambiguation, n)


class UiLanguage(QObject):
    changed = Signal(str)

    def __init__(self, owner):
        super().__init__(owner)
        self.language = "zh"
        self._bindings = {}
        self.qt_translator = QtTranslations(self)
        QApplication.instance().installTranslator(self.qt_translator)
        owner.destroyed.connect(self._uninstall)

    def _uninstall(self):
        application = QApplication.instance()
        if application:
            application.removeTranslator(self.qt_translator)

    def text(self, source, **values):
        return render(message(source, **values) if values else source, self.language)

    def bind(self, widget, value, prop="text", index=None):
        key = (widget, prop, index)
        self._bindings[key] = value
        self._apply(key, value)

    def forget(self, widget):
        for key in list(self._bindings):
            if key[0] is widget:
                del self._bindings[key]

    def _apply(self, key, value):
        widget, prop, index = key
        text = render(value, self.language)
        blocker = QSignalBlocker(widget)
        try:
            # Retranslation must not clear a currently displayed detection frame.
            if prop == "text" and isinstance(widget, QLabel) and not widget.pixmap().isNull():
                return
            if prop == "itemText":
                widget.setItemText(index, text)
            elif prop == "tabText":
                widget.setTabText(index, text)
            elif prop == "status":
                widget.showMessage(text)
            else:
                getattr(widget, "set" + prop[0].upper() + prop[1:])(text)
        finally:
            blocker.unblock()

    def capture(self, root):
        """Capture owned static text, never editable content or annotation data."""
        for widget in [root, *root.findChildren(QWidget)]:
            properties = ["toolTip"]
            if isinstance(widget, QLabel):
                properties += ["text"]
            elif isinstance(widget, QAbstractButton):
                properties += ["text"]
            elif isinstance(widget, QLineEdit):
                properties += ["placeholderText"]
            elif isinstance(widget, QGroupBox):
                properties += ["title"]
            if widget is root:
                properties += ["windowTitle"]
            for prop in properties:
                key = (widget, prop, None)
                source = getattr(widget, prop)()
                if key not in self._bindings and source in EN:
                    self.bind(widget, source, prop)
                    if prop == "text" and isinstance(widget, QLabel):
                        widget.setWordWrap(True)
            if isinstance(widget, QComboBox):
                for index in range(widget.count()):
                    key = (widget, "itemText", index)
                    source = widget.itemText(index)
                    if key not in self._bindings and source in EN:
                        self.bind(widget, source, "itemText", index)
            if isinstance(widget, QTabWidget):
                for index in range(widget.count()):
                    if widget.tabText(index) in EN:
                        self.bind(widget, widget.tabText(index), "tabText", index)

    def set_language(self, language):
        if language not in ("zh", "en"):
            raise ValueError("Unsupported interface language")
        if language == self.language:
            return
        self.language = language
        # Reinstall so Qt's standard dialog strings also get a LanguageChange.
        application = QApplication.instance()
        application.removeTranslator(self.qt_translator)
        application.installTranslator(self.qt_translator)
        for key, value in list(self._bindings.items()):
            self._apply(key, value)
        self.changed.emit(language)


def language_for(parent):
    widget = parent
    while widget is not None:
        candidate = getattr(widget, "ui_language", None)
        if not isinstance(candidate, UiLanguage):
            candidate = getattr(getattr(widget, "app", None), "ui_language", None)
        if isinstance(candidate, UiLanguage):
            return candidate
        widget = widget.parent() if isinstance(widget, QWidget) else None
    return None


class QMessageBox(QtMessageBox):
    """Keep call sites/test mocks familiar, but localize standard buttons too."""

    @staticmethod
    def _show(parent, title, text, icon, buttons, default):
        translator = language_for(parent)
        dialog = QtMessageBox(parent)
        dialog.setTextFormat(Qt.PlainText)
        dialog.setIcon(icon)
        dialog.setStandardButtons(buttons)
        dialog.setDefaultButton(default)
        def update(*_):
            language = translator.language if translator else "zh"
            dialog.setWindowTitle(render(title, language))
            dialog.setText(render(text, language))
            for flag, zh, en in ((QtMessageBox.Ok, "确定", "OK"), (QtMessageBox.Cancel, "取消", "Cancel"),
                                 (QtMessageBox.Yes, "是", "Yes"), (QtMessageBox.No, "否", "No")):
                if dialog.button(flag):
                    dialog.button(flag).setText(en if language == "en" else zh)
        update()
        if translator:
            translator.changed.connect(update)
        try:
            return dialog.exec()
        finally:
            if translator:
                translator.changed.disconnect(update)
            dialog.deleteLater()

    @staticmethod
    def information(parent, title, text, buttons=QtMessageBox.Ok, defaultButton=QtMessageBox.NoButton):
        return QMessageBox._show(parent, title, text, QtMessageBox.Information, buttons, defaultButton)

    @staticmethod
    def warning(parent, title, text, buttons=QtMessageBox.Ok, defaultButton=QtMessageBox.NoButton):
        return QMessageBox._show(parent, title, text, QtMessageBox.Warning, buttons, defaultButton)

    @staticmethod
    def critical(parent, title, text, buttons=QtMessageBox.Ok, defaultButton=QtMessageBox.NoButton):
        return QMessageBox._show(parent, title, text, QtMessageBox.Critical, buttons, defaultButton)

    @staticmethod
    def question(parent, title, text, buttons=QtMessageBox.Yes | QtMessageBox.No, defaultButton=QtMessageBox.No):
        return QMessageBox._show(parent, title, text, QtMessageBox.Question, buttons, defaultButton)


class QFileDialog(QtFileDialog):
    """Use Qt dialogs, not OS-language dialogs, for consistent UI language."""

    @staticmethod
    def _texts(parent, caption, file_filter=""):
        translator = language_for(parent)
        language = translator.language if translator else "zh"
        return render(caption, language), render(file_filter, language)

    @staticmethod
    def getOpenFileName(parent=None, caption="", dir="", filter="", **kwargs):
        caption, filter = QFileDialog._texts(parent, caption, filter)
        kwargs["options"] = kwargs.get("options", QtFileDialog.Options()) | QtFileDialog.DontUseNativeDialog
        return QtFileDialog.getOpenFileName(parent, caption, dir, filter, **kwargs)

    @staticmethod
    def getSaveFileName(parent=None, caption="", dir="", filter="", **kwargs):
        caption, filter = QFileDialog._texts(parent, caption, filter)
        kwargs["options"] = kwargs.get("options", QtFileDialog.Options()) | QtFileDialog.DontUseNativeDialog
        return QtFileDialog.getSaveFileName(parent, caption, dir, filter, **kwargs)

    @staticmethod
    def getExistingDirectory(parent=None, caption="", dir="", **kwargs):
        caption, _ = QFileDialog._texts(parent, caption)
        kwargs["options"] = kwargs.get("options", QtFileDialog.ShowDirsOnly) | QtFileDialog.DontUseNativeDialog
        return QtFileDialog.getExistingDirectory(parent, caption, dir, **kwargs)
