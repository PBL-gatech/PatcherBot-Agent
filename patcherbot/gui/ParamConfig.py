"""Reusable configuration editors, independent of camera and hardware GUI classes."""
import ast
import functools
import logging
import param
import qtawesome as qta
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from patcherbot.configs.Config import NumberWithUnit

__all__ = ["ParamConfig", "ElidedLabel"]

class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text, minimum_width=200, *args, **kwds):
        self.minimum_width = minimum_width
        self.text = text
        super(ElidedLabel, self).__init__(*args, **kwds)

    def minimumSizeHint(self):
        return QtCore.QSize(self.minimum_width,
                            super(ElidedLabel, self).minimumSizeHint().height())

    def resizeEvent(self, event):
        metric = QtGui.QFontMetrics(self.font())
        elidedText = metric.elidedText(self.text, QtCore.Qt.ElideRight,
                                       self.width())
        self.setText(elidedText)


class ParamConfig(QtWidgets.QWidget):
    value_changed_signal = QtCore.pyqtSignal('QString', object)

    def __init__(self, config, show_name=False, *, parent=None, build_ui=True):
        super().__init__(parent=parent)
        self.config = config
        self.value_widgets = {}
        self.config._value_changed = self.value_changed
        self.value_changed_signal.connect(self.display_changed_value)
        if build_ui:
            self._build_config_ui(show_name)

    def _build_config_ui(self, show_name=False):
        config = self.config
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        top_row = QtWidgets.QHBoxLayout()
        if show_name:
            self.title = QtWidgets.QLabel(config.name)
            self.title.setStyleSheet('font-weight: bold;')
            top_row.addWidget(self.title)
        else:
            top_row.setAlignment(Qt.AlignRight)
        self.load_button = QtWidgets.QToolButton(clicked=self.load_config)
        self.load_button.setIcon(qta.icon('fa.upload'))
        top_row.addWidget(self.load_button)
        self.save_button = QtWidgets.QToolButton(clicked=self.save_config)
        self.save_button.setIcon(qta.icon('fa.download'))
        top_row.addWidget(self.save_button)
        layout.addLayout(top_row)
        for category, params in config.categories:
            layout.addWidget(self._create_config_group(category, params))
        self.setLayout(layout)

    def _create_config_group(self, category, parameter_names):
        box = QtWidgets.QGroupBox(category)
        rows = QtWidgets.QVBoxLayout()
        for name in parameter_names:
            parameter = self.config.param[name]
            row = QtWidgets.QHBoxLayout()
            label = ElidedLabel(parameter.doc)
            label.setToolTip(parameter.doc)
            row.addWidget(label, stretch=1)
            row.addWidget(self._create_value_widget(name))
            if isinstance(parameter, NumberWithUnit):
                row.addWidget(QtWidgets.QLabel(parameter.unit))
            rows.addLayout(row)
        box.setLayout(rows)
        return box

    def _create_value_widget(self, param_name, *, multiline=False):
        config = self.config
        param_obj = config.param[param_name]
        if multiline and not isinstance(param_obj, param.String):
            raise ValueError("Multiline editors require a string parameter.")
        if isinstance(param_obj, NumberWithUnit):
            value_widget = QtWidgets.QDoubleSpinBox()
            magnitude = param_obj.magnitude
            value_widget.setMinimum(param_obj.bounds[0] / magnitude)
            value_widget.setMaximum(param_obj.bounds[1] / magnitude)
            value_widget.setValue(getattr(config, param_name) / magnitude)
            value_widget.valueChanged.connect(
                functools.partial(self.set_numerical_value_with_unit, param_name, magnitude))
        elif isinstance(param_obj, param.Number):
            value_widget = QtWidgets.QDoubleSpinBox()
            value_widget.setMinimum(param_obj.bounds[0])
            value_widget.setMaximum(param_obj.bounds[1])
            value_widget.setValue(getattr(config, param_name))
            value_widget.valueChanged.connect(
                functools.partial(self.set_numerical_value, param_name))
        elif isinstance(param_obj, param.Boolean):
            value_widget = QtWidgets.QCheckBox()
            value_widget.setChecked(getattr(config, param_name))
            value_widget.stateChanged.connect(
                functools.partial(self.set_boolean_value, param_name, value_widget))
        elif isinstance(param_obj, (param.Selector)):
            value_widget = QtWidgets.QComboBox()
            value_widget.addItems([str(o) for o in param_obj.objects])
            current = getattr(config, param_name)
            if current in param_obj.objects:
                value_widget.setCurrentIndex(param_obj.objects.index(current))
            value_widget.currentIndexChanged.connect(
                functools.partial(self.set_selector_value, param_name, param_obj.objects))
        elif isinstance(param_obj, param.String) and multiline:
            value_widget = QtWidgets.QPlainTextEdit()
            value_widget.setPlainText(str(getattr(config, param_name)))
            value_widget.textChanged.connect(
                lambda: self.set_string_value(param_name, value_widget.toPlainText()))
        elif isinstance(param_obj, param.String):
            value_widget = QtWidgets.QLineEdit()
            value_widget.setText(str(getattr(config, param_name)))
            value_widget.textChanged.connect(
                functools.partial(self.set_string_value, param_name))
        elif isinstance(param_obj, param.List):
            value_widget = QtWidgets.QLineEdit()
            value_widget.setText(self._format_list_value(getattr(config, param_name)))
            value_widget.editingFinished.connect(
                functools.partial(self.set_list_value, param_name, value_widget, param_obj))
        elif isinstance(param_obj, param.Tuple):         
            value_widget = QtWidgets.QLineEdit()          
            value_widget.setReadOnly(True)                
            value_widget.setEnabled(False)               
            value_widget.setText(str(getattr(config, param_name))) 
        else:
            value_widget = QtWidgets.QLineEdit()
            value_widget.setReadOnly(True)
            value_widget.setEnabled(False)
            value_widget.setText(str(getattr(config, param_name)))
        value_widget.setToolTip(param_obj.doc)
        value_widget.setObjectName(param_name)
        self.value_widgets[param_name] = value_widget
        return value_widget

    def value_changed(self, key, value):
        """Relay parameter updates coming from the Config object.
        Numeric parameters are scaled by their unit magnitude; non‑numeric
        (e.g. Selector / Boolean) are forwarded unchanged."""
        if key not in self.value_widgets:
            return

        param_obj  = self.config.param[key]
        magnitude  = getattr(param_obj, 'magnitude', 1)

        # Only scale numeric values; leave strings / bools intact
        if isinstance(value, (int, float)):
            self.value_changed_signal.emit(key, value / magnitude)
        else:
            self.value_changed_signal.emit(key, value)

    @QtCore.pyqtSlot('QString', object)
    def display_changed_value(self, key, value):
        widget = self.value_widgets.get(key)
        if widget is None:
            return
        blocked = widget.blockSignals(True)
        try:
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                if widget.toPlainText() != str(value):
                    widget.setPlainText(str(value))
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = self._format_list_value(value) if isinstance(value, list) else str(value)
                if widget.text() != text:
                    widget.setText(text)
        finally:
            widget.blockSignals(blocked)


    def set_numerical_value(self, name, value):
        setattr(self.config, name, value)

    def set_numerical_value_with_unit(self, name, magnitude, value):
        setattr(self.config, name, value * magnitude)

    def set_boolean_value(self, name, widget):
        setattr(self.config, name, widget.isChecked())

    def set_selector_value(self, name, options, index):
        if 0 <= index < len(options):
            setattr(self.config, name, options[index])

    def set_string_value(self, name, value):
        setattr(self.config, name, value)

    def _format_list_value(self, value):
        if isinstance(value, (list, tuple)):
            return ', '.join(str(v) for v in value)
        return str(value)

    def _parse_list_value(self, text, item_type=None):
        cleaned = (text or "").strip()
        if cleaned == "":
            parsed = []
        else:
            try:
                literal = ast.literal_eval(cleaned)
                if isinstance(literal, (list, tuple)):
                    parsed = list(literal)
                elif isinstance(literal, str):
                    parsed = [literal]
                else:
                    parsed = [literal]
            except Exception:
                parsed = [item.strip() for item in cleaned.split(",") if item.strip()]

        if item_type is None:
            return parsed

        coerced = []
        for item in parsed:
            if item_type is str:
                coerced.append(str(item))
            elif item_type is int:
                coerced.append(int(item))
            elif item_type is float:
                coerced.append(float(item))
            elif item_type is bool:
                if isinstance(item, bool):
                    coerced.append(item)
                else:
                    token = str(item).strip().lower()
                    if token in ("1", "true", "yes", "y", "on"):
                        coerced.append(True)
                    elif token in ("0", "false", "no", "n", "off"):
                        coerced.append(False)
                    else:
                        raise ValueError(f"Invalid boolean list entry: {item}")
            else:
                coerced.append(item_type(item))
        return coerced

    def set_list_value(self, name, widget, param_obj):
        try:
            item_type = getattr(param_obj, "item_type", None)
            parsed = self._parse_list_value(widget.text(), item_type=item_type)
            setattr(self.config, name, parsed)
            widget.blockSignals(True)
            widget.setText(self._format_list_value(parsed))
            widget.blockSignals(False)
        except Exception:
            logging.getLogger(__name__).warning(
                "Invalid list value for '%s': %s", name, widget.text(), exc_info=True
            )
            widget.blockSignals(True)
            widget.setText(self._format_list_value(getattr(self.config, name)))
            widget.blockSignals(False)


    def save_config(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save configuration",
                                                            filter='Configuration files (*.yaml)',
                                                            options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename:
            try:
                self.config.to_file(filename)
            except Exception as ex:
                err = f'Could not save configuration to file "{filename}"'
                logging.getLogger(__name__).exception(err)
                QtWidgets.QMessageBox.warning(self, 'Saving failed', err + '\n' + str(ex), QtWidgets.QMessageBox.Ok)

    def load_config(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load configuration",
                                                            filter='Configuration files (*.yaml)',
                                                            options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename:
            try:
                self.config.from_file(filename)
            except Exception as ex:
                err = f'Could not load configuration from file "{filename}"'
                logging.getLogger(__name__).exception(err)
                QtWidgets.QMessageBox.warning(self, 'Loading failed', err + '\n' + str(ex), QtWidgets.QMessageBox.Ok)
