import PyQt5.Qt as Qt

class LeftRightAnnotator:
    
    def __init__(self, rw):
        self.rw = rw
        self.current_idx = rw.flipbook.current_page_idx
        self.rw.flipbook.focused_page_idx = 0
        self.directions = {}
        self.actions = []
        self.add_action('left', Qt.Qt.Key_Left, self.left)
        self.add_action('right', Qt.Qt.Key_Right, self.right)

    def add_action(self, name, key, function):
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)

    def left(self):
        self._log_direction('left')
    
    def right(self):
        self._log_direction('right')

    def _log_direction(self, direction):
        self.directions[self.rw.flipbook.current_page.name] = direction
        self.rw.flipbook.focus_next_page()
