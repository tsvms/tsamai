"""The TsamAI desktop app.

One window: pick a model in the top bar, type below, read above. The v3
checkpoint chats; v1/v2 continue whatever you start writing. I type in
sans-serif and the model answers in serif — that's the whole "chat bubble"
system, and I like it better than bubbles.

Generation runs on a QThread so the window never freezes; Enter sends,
Shift+Enter is a newline, and the send arrow turns into a stop square while
the model is writing.
"""

import os
import sys

import torch
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sample import load_model

# checkpoint -> (picker label, chat mode?, input placeholder)
MODELS = {
    "checkpoints/v3.pt": ("TsamAI chat", True, "Message TsamAI…"),
    "checkpoints/v2.pt": ("TsamAI stories", False, "Write a beginning… e.g. Once upon a time"),
    "checkpoints/ckpt.pt": ("TsamAI shakespeare", False, "Write a beginning… e.g. ROMEO:"),
}

EOT = "<|endoftext|>"
STOP_STR = "\nUser:"  # chat: the model starting the *user's* next line = done

INK = "#101216"
SURFACE = "#16191f"
LINE = "#23272f"
TEXT = "#e6e4dd"
MUTED = "#7a7f8a"
AMBER = "#e0a458"

UI_FONT = "Adwaita Sans"
SERIF_FONT = "DejaVu Serif"

STYLE = f"""
* {{ font-family: "{UI_FONT}", "Cantarell", "DejaVu Sans"; font-size: 13px; }}
QMainWindow, QWidget {{ background: {INK}; color: {TEXT}; }}

#topbar {{ border-bottom: 1px solid {LINE}; }}
#wordmark {{ font-size: 15px; font-weight: 700; color: {AMBER}; }}

QComboBox {{
    background: transparent; border: none; color: {MUTED};
    padding: 6px 10px; font-weight: 600;
}}
QComboBox:hover {{ color: {TEXT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {LINE};
    selection-background-color: {AMBER}; selection-color: {INK};
    padding: 4px;
}}

QToolButton {{
    background: transparent; border: none; color: {MUTED};
    padding: 6px 10px; font-weight: 600;
}}
QToolButton:hover {{ color: {TEXT}; }}
QToolButton:checked {{ color: {AMBER}; }}

#tuner {{ border-bottom: 1px solid {LINE}; }}
QLabel.knob {{ color: {MUTED}; font-size: 12px; }}
QLabel.knobValue {{ color: {TEXT}; font-size: 12px; min-width: 34px; }}
#modelInfo {{ color: {MUTED}; font-size: 11px; }}

QTextEdit {{
    background: {INK}; border: none; padding: 4px 10px;
    selection-background-color: {AMBER}; selection-color: {INK};
}}

#inputwrap {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; }}
#inputwrap:focus-within {{ border-color: #3a4050; }}
QPlainTextEdit {{
    background: transparent; border: none; padding: 2px;
    font-size: 14px; color: {TEXT};
}}

QPushButton#send {{
    background: {AMBER}; color: {INK}; border: none; border-radius: 9px;
    min-width: 34px; max-width: 34px; min-height: 34px; max-height: 34px;
    font-size: 15px; font-weight: 700;
}}
QPushButton#send:hover {{ background: #eab56e; }}
QPushButton#send:disabled {{ background: {LINE}; color: {MUTED}; }}
QPushButton#send[stopping="true"] {{ background: {SURFACE}; color: #c96a5a;
    border: 1px solid #c96a5a; }}

QSlider {{ min-width: 110px; }}
QSlider::groove:horizontal {{ height: 3px; background: {LINE}; border-radius: 1px; }}
QSlider::handle:horizontal {{
    background: {MUTED}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: {AMBER}; }}
QSlider::sub-page:horizontal {{ background: {MUTED}; border-radius: 1px; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3a4050; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


class GenerateWorker(QThread):
    """Streams tokens out of model.generate. In chat mode I hold back the
    last few characters before emitting them — if the model starts writing
    the user's next turn ("\\nUser:") it gets cut before anyone sees it."""

    piece = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, model, tokenizer, device, prompt, tokens,
                 temperature, top_k, stop_str=None, eot_id=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt = prompt
        self.tokens = tokens
        self.temperature = temperature
        self.top_k = top_k
        self.stop_str = stop_str
        self.eot_id = eot_id
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            ids = self.tokenizer.encode(self.prompt) or self.tokenizer.encode("\n") or [0]
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            holdback = len(self.stop_str) + 4 if self.stop_str else 0
            pending = ""
            for token_id in self.model.generate(
                idx, self.tokens, temperature=self.temperature, top_k=self.top_k
            ):
                if self._stop:
                    break
                if self.eot_id is not None and token_id == self.eot_id:
                    break
                pending += self.tokenizer.decode_token(token_id)
                if self.stop_str and self.stop_str in pending:
                    pending = pending[: pending.index(self.stop_str)]
                    break
                if len(pending) > holdback:
                    cut = len(pending) - holdback
                    self.piece.emit(pending[:cut])
                    pending = pending[cut:]
            self.piece.emit(pending)
            self.finished_ok.emit()
        except Exception as e:  # show it in the window, don't take the app down
            self.failed.emit(str(e))


class Knob(QWidget):
    """One quiet slider with a label and a live value."""

    def __init__(self, label, minimum, maximum, value, scale=1.0, fmt="{:.2f}"):
        super().__init__()
        self.scale = scale
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name = QLabel(label)
        name.setProperty("class", "knob")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.val = QLabel()
        self.val.setProperty("class", "knobValue")
        self.slider.valueChanged.connect(
            lambda raw: self.val.setText(fmt.format(raw * self.scale))
        )
        self.val.setText(fmt.format(value * scale))
        row.addWidget(name)
        row.addWidget(self.slider)
        row.addWidget(self.val)

    def value(self):
        return self.slider.value() * self.scale


class InputBox(QPlainTextEdit):
    """Single-to-few-line input. Enter submits, Shift+Enter breaks a line."""

    submitted = Signal()

    def __init__(self):
        super().__init__()
        self.setTabChangesFocus(True)
        self.document().setDocumentMargin(8)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textChanged.connect(self._resize)
        self._resize()

    def _resize(self):
        rows = min(max(self.document().blockCount(), 1), 5)
        line = self.fontMetrics().lineSpacing()
        self.setFixedHeight(int(rows * line + 24))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class TsamAIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TsamAI")
        self.resize(860, 640)
        self.worker = None
        self.model = None
        self.chat_history: list[tuple[str, str]] = []  # (speaker, text)

        self.available = {p: m for p, m in MODELS.items() if os.path.exists(p)}
        if not self.available:
            QMessageBox.critical(
                self, "TsamAI",
                "No trained model found.\n\nRun  python train.py  first."
            )
            sys.exit(1)

        self._fmt_user_label = QTextCharFormat()
        self._fmt_user_label.setForeground(QColor(AMBER))
        self._fmt_user_label.setFontWeight(QFont.DemiBold)
        self._fmt_user = QTextCharFormat()
        self._fmt_user.setForeground(QColor(MUTED))
        self._fmt_ai = QTextCharFormat()
        serif = QFont(SERIF_FONT, 13)
        serif.setStyleHint(QFont.Serif)
        self._fmt_ai.setFont(serif)
        self._fmt_ai.setForeground(QColor(TEXT))

        self._build_ui()
        self.picker.setCurrentIndex(0)
        self._load_selected_model()

    # ---------- UI ----------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # top bar
        topbar = QWidget()
        topbar.setObjectName("topbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(20, 10, 14, 10)
        wordmark = QLabel("TsamAI")
        wordmark.setObjectName("wordmark")
        self.picker = QComboBox()
        for path, (label, _, _) in self.available.items():
            self.picker.addItem(label, userData=path)
        self.picker.currentIndexChanged.connect(self._load_selected_model)
        self.clear_btn = QToolButton()
        self.clear_btn.setText("clear")
        self.clear_btn.clicked.connect(self._clear)
        self.tune_btn = QToolButton()
        self.tune_btn.setText("tune")
        self.tune_btn.setCheckable(True)
        self.tune_btn.toggled.connect(lambda on: self.tuner.setVisible(on))
        top.addWidget(wordmark)
        top.addStretch()
        top.addWidget(self.picker)
        top.addWidget(self.clear_btn)
        top.addWidget(self.tune_btn)
        col.addWidget(topbar)

        # tuner (hidden until "tune")
        self.tuner = QWidget()
        self.tuner.setObjectName("tuner")
        tun = QHBoxLayout(self.tuner)
        tun.setContentsMargins(20, 8, 20, 10)
        tun.setSpacing(24)
        self.temp_ctl = Knob("temperature", 10, 200, 80, scale=0.01)
        self.topk_ctl = Knob("top-k", 1, 200, 50, scale=1, fmt="{:.0f}")
        self.len_ctl = Knob("length", 50, 2000, 400, scale=1, fmt="{:.0f}")
        self.info = QLabel("")
        self.info.setObjectName("modelInfo")
        tun.addWidget(self.temp_ctl)
        tun.addWidget(self.topk_ctl)
        tun.addWidget(self.len_ctl)
        tun.addStretch()
        tun.addWidget(self.info)
        self.tuner.setVisible(False)
        col.addWidget(self.tuner)

        # transcript
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setDocumentMargin(20)
        col.addWidget(self.view, stretch=1)

        # input row
        bottom = QWidget()
        brow = QHBoxLayout(bottom)
        brow.setContentsMargins(20, 8, 20, 16)
        brow.setSpacing(10)
        wrap = QWidget()
        wrap.setObjectName("inputwrap")
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(10, 4, 10, 4)
        self.input = InputBox()
        self.input.submitted.connect(self._submit)
        wl.addWidget(self.input)
        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("send")
        self.send_btn.clicked.connect(self._submit_or_stop)
        brow.addWidget(wrap, stretch=1)
        brow.addWidget(self.send_btn, alignment=Qt.AlignBottom)
        col.addWidget(bottom)

    # ---------- model handling ----------

    def _load_selected_model(self):
        path = self.picker.currentData()
        if path is None:
            return
        try:
            self.model, self.tokenizer, self.meta = load_model(path)
        except Exception as e:
            QMessageBox.warning(self, "TsamAI", f"Failed to load model:\n{e}")
            return
        _, self.chat_mode, placeholder = self.available[path]
        self.input.setPlaceholderText(placeholder)
        eot_ids = self.tokenizer.encode(EOT)
        self.eot_id = eot_ids[0] if len(eot_ids) == 1 else None
        self.info.setText(
            f"{self.model.num_params()/1e6:.1f}M params · {self.meta['device'].upper()}"
            f" · val loss {self.meta['val_loss']:.3f}"
        )
        self._clear()

    def _clear(self):
        self.chat_history = []
        self.view.clear()

    # ---------- transcript rendering ----------

    def _append(self, text: str, fmt: QTextCharFormat):
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, fmt)
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

    # ---------- generation ----------

    def _submit_or_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        else:
            self._submit()

    def _submit(self):
        if self.worker is not None and self.worker.isRunning():
            return
        text = self.input.toPlainText().strip()
        if not text or self.model is None:
            return
        self.input.clear()

        if self.chat_mode:
            if self.chat_history:
                self._append("\n\n", self._fmt_ai)
            self._append("you  ", self._fmt_user_label)
            self._append(text + "\n", self._fmt_user)
            self.chat_history.append(("User", text))
            prompt = (
                "\n".join(f"{s}: {t}" for s, t in self.chat_history) + "\nTsamAI:"
            )
            stop_str, self._reply = STOP_STR, []
        else:
            self.view.clear()
            self._append(text, self._fmt_user)
            self._append("", self._fmt_ai)
            prompt, stop_str, self._reply = text, None, []

        self.worker = GenerateWorker(
            self.model, self.tokenizer, self.meta["device"], prompt,
            int(self.len_ctl.value()), self.temp_ctl.value(),
            int(self.topk_ctl.value()), stop_str=stop_str, eot_id=self.eot_id,
        )
        self.worker.piece.connect(self._on_piece)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.send_btn.setText("■")
        self.send_btn.setProperty("stopping", "true")
        self._repolish(self.send_btn)
        self.picker.setEnabled(False)
        self.worker.start()

    def _on_piece(self, piece: str):
        if self.chat_mode and not self._reply:
            piece = piece.lstrip()
            if not piece:
                return
        self._reply.append(piece)
        self._append(piece, self._fmt_ai)

    def _done(self):
        if self.chat_mode:
            reply = "".join(self._reply).strip()
            self.chat_history.append(("TsamAI", reply))
        self.send_btn.setText("➤")
        self.send_btn.setProperty("stopping", "false")
        self._repolish(self.send_btn)
        self.picker.setEnabled(True)
        self.input.setFocus()

    def _failed(self, message: str):
        self._done()
        QMessageBox.warning(self, "TsamAI", f"Generation failed:\n{message}")

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    default = QFont(UI_FONT, 10)
    default.setStyleHint(QFont.SansSerif)
    app.setFont(default)
    window = TsamAIWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
