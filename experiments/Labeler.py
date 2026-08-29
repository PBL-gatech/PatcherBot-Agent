from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QHBoxLayout,
    QFrame, QMessageBox, QShortcut, QInputDialog, QListWidget
)
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QPen, QColor, QKeySequence
import os
import json
import sys

class ResizableRectItem(QGraphicsRectItem):
    """Custom QGraphicsRectItem that can be resized and holds a label. Used to make a bounding box."""
    def __init__(self, rect, label="", parent=None):
        super().__init__(rect, parent)
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.handle_size = 8.0
        self.handles = {}
        self._current_handle = None
        self._resizing = False
        self.update_handles()
        self.label = label  # Label for the bounding box

    def update_handles(self):
        """Update the positions of the resize handles."""
        s = self.handle_size
        rect = self.rect()
        self.handles = {
            'top_left': QRectF(rect.topLeft().x() - s/2, rect.topLeft().y() - s/2, s, s),
            'top_right': QRectF(rect.topRight().x() - s/2, rect.topRight().y() - s/2, s, s),
            'bottom_left': QRectF(rect.bottomLeft().x() - s/2, rect.bottomLeft().y() - s/2, s, s),
            'bottom_right': QRectF(rect.bottomRight().x() - s/2, rect.bottomRight().y() - s/2, s, s),
        }

    def paint(self, painter, option, widget=None):
        """Paint the rectangle and its resize handles."""
        super().paint(painter, option, widget)
        if self.isSelected():
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect())
            # Draw handles
            painter.setBrush(QColor(255, 0, 0))
            for handle in self.handles.values():
                painter.drawRect(handle)

    def hoverMoveEvent(self, event):
        """Change cursor when hovering over resize handles."""
        cursor = Qt.ArrowCursor
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                if 'left' in key or 'right' in key:
                    if 'top' in key or 'bottom' in key:
                        cursor = Qt.SizeFDiagCursor
                break
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """Determine if a resize handle is pressed."""
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                self._current_handle = key
                self._resizing = True
                break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Resize the rectangle based on the handle being dragged."""
        if self._resizing and self._current_handle:
            rect = self.rect()
            delta = event.scenePos() - event.lastScenePos()
            if 'left' in self._current_handle:
                new_left = rect.left() + delta.x()
                if new_left < 0:
                    new_left = 0
                rect.setLeft(new_left)
            if 'right' in self._current_handle:
                new_right = rect.right() + delta.x()
                if new_right > self.scene().width():
                    new_right = self.scene().width()
                rect.setRight(new_right)
            if 'top' in self._current_handle:
                new_top = rect.top() + delta.y()
                if new_top < 0:
                    new_top = 0
                rect.setTop(new_top)
            if 'bottom' in self._current_handle:
                new_bottom = rect.bottom() + delta.y()
                if new_bottom > self.scene().height():
                    new_bottom = self.scene().height()
                rect.setBottom(new_bottom)
            self.setRect(rect)
            self.update_handles()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Reset resizing flags."""
        self._resizing = False
        self._current_handle = None
        super().mouseReleaseEvent(event)

class CustomGraphicsScene(QGraphicsScene):
    """Custom QGraphicsScene to handle drawing of bounding boxes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_point = QPointF()
        self.current_rect_item = None

    def mousePressEvent(self, event):
        """Start drawing a new rectangle only if click is not on an existing item."""
        if event.button() == Qt.LeftButton:
            items = self.items(event.scenePos())
            if not any(isinstance(item, ResizableRectItem) for item in items):
                self.drawing = True
                self.start_point = event.scenePos()
                self.current_rect_item = ResizableRectItem(QRectF(self.start_point, self.start_point))
                pen = QPen(QColor(255, 0, 0), 2)
                self.current_rect_item.setPen(pen)
                self.addItem(self.current_rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update the size of the rectangle being drawn."""
        if self.drawing and self.current_rect_item:
            rect = QRectF(self.start_point, event.scenePos()).normalized()
            self.current_rect_item.setRect(rect)
            self.current_rect_item.update_handles()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish drawing the rectangle and assign label."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_rect_item:
                rect = self.current_rect_item.rect()
                if rect.width() < 10 or rect.height() < 10:
                    self.removeItem(self.current_rect_item)
                else:
                    # Assign label
                    parent = self.parent()  # Reference to ImageLabeler
                    if parent.default_label:
                        label = parent.default_label
                        self.current_rect_item.label = label
                        self.current_rect_item.setToolTip(label)
                        parent.add_label_to_list(self.current_rect_item)
                        parent.save_bounding_boxes(parent.image_paths[parent.current_index])  # Save immediately
                    else:
                        # Prompt for label if no default_label is set
                        label, ok = QInputDialog.getText(None, "Input Label", "Enter label for the bounding box:")
                        if ok and label.strip():
                            self.current_rect_item.label = label.strip()
                            self.current_rect_item.setToolTip(label.strip())
                            parent.default_label = label.strip()  # Save as default label
                            parent.add_label_to_list(self.current_rect_item)
                            parent.save_bounding_boxes(parent.image_paths[parent.current_index])  # Save immediately
                        else:
                            # If no label is provided, remove the box
                            self.removeItem(self.current_rect_item)
            self.current_rect_item = None
        super().mouseReleaseEvent(event)

class ImageLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeler")
        self.setGeometry(50, 50, 1000, 800)  # Increased width to accommodate new button

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Horizontal layout for image and side list
        self.top_layout = QHBoxLayout()

        # Graphics scene and view for bounding boxes
        self.scene = CustomGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setMinimumSize(800, 600)

        # Side list for labeled bounding boxes
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(300)  # Fixed width for the side list

        # Add the view and list widget to the top layout
        self.top_layout.addWidget(self.view)
        self.top_layout.addWidget(self.list_widget)
        self.main_layout.addLayout(self.top_layout)

        # Info pane layout for displaying messages to the user
        self.info_label = QLabel("No info yet!")
        self.info_label.setAlignment(Qt.AlignLeft)
        self.info_label.setFixedHeight(20)
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.addWidget(self.info_label)
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setFrameShadow(QFrame.Sunken)

        self.main_layout.addWidget(self.info_frame)

        # Add buttons for navigation and directory selection at the bottom
        self.buttons_layout = QHBoxLayout()

        self.load_button = QPushButton("Select Data Directory")
        self.load_button.clicked.connect(self.open_directory)

        self.prev_button = QPushButton("Previous Image")
        self.prev_button.clicked.connect(self.show_previous_image)

        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.show_next_image)

        # Add the new buttons: Delete Selected Box and Delete All Boxes
        self.delete_selected_button = QPushButton("Delete Selected Box")
        self.delete_selected_button.clicked.connect(self.delete_selected_box)

        self.delete_all_button = QPushButton("Delete All Boxes")
        self.delete_all_button.clicked.connect(self.delete_all_boxes)

        # *** New Button: Change Label Name ***
        self.change_label_button = QPushButton("Change Label Name")
        self.change_label_button.clicked.connect(self.change_label_name)
        # *** End of New Button ***

        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.load_button)
        self.buttons_layout.addWidget(self.next_button)
        self.buttons_layout.addWidget(self.delete_selected_button)
        self.buttons_layout.addWidget(self.delete_all_button)
        self.buttons_layout.addWidget(self.change_label_button)  # Add new button to layout

        self.main_layout.addLayout(self.buttons_layout)

        # Set initial state
        self.image_paths = []
        self.labels_path = ''
        self.current_index = 0
        self.default_label = None  # To store the default label after first input

        self.label_ID = {
            "in_focus" : 0,
            "above_plane": 1,
            "below_plane": 2,
            "not_detected": 3
        } # parameterise the labels from words to numbers

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.show_next_image)

    def refresh_list_widget(self):
        """Refresh the sidebar list_widget to reflect current labels."""
        self.list_widget.clear()
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                self.add_label_to_list(item)

    def add_label_to_list(self, rect_item):
        """Add a labeled bounding box to the side list."""
        label_text = f"Label: {rect_item.label}, " \
                     f"x: {rect_item.rect().x():.4f}, " \
                     f"y: {rect_item.rect().y():.4f}, " \
                     f"w: {rect_item.rect().width():.4f}, " \
                     f"h: {rect_item.rect().height():.4f}"
        self.list_widget.addItem(label_text)

    def remove_label_from_list(self, rect_item):
        """Remove a labeled bounding box from the side list."""
        # Iterate in reverse to safely remove items while iterating
        for index in reversed(range(self.list_widget.count())):
            item = self.list_widget.item(index)
            if (f"Label: {rect_item.label}," in item.text() and
                f"x: {rect_item.rect().x():.4f}," in item.text() and
                f"y: {rect_item.rect().y():.4f}," in item.text() and
                f"w: {rect_item.rect().width():.4f}," in item.text() and
                f"h: {rect_item.rect().height():.4f}" in item.text()):
                self.list_widget.takeItem(index)
                break

    def open_directory(self):
        """Open a directory with 'P_DET_IMAGES' and 'P_DET_LABELS' folders."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            images_dir = os.path.join(directory, 'P_DET_IMAGES')
            labels_dir = os.path.join(directory, 'P_DET_LABELS')

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                self.load_images_from_directory(images_dir)
                self.labels_path = labels_dir
                self.info_label.setText(f"Loaded images and labels from {directory}.")
                if self.image_paths:
                    self.current_index = 0
                    self.display_image(self.image_paths[self.current_index])
            else:
                QMessageBox.warning(self, "Directory Error",
                                    "Selected folder must contain 'P_DET_IMAGES' and 'P_DET_LABELS' subfolders.")

    def load_images_from_directory(self, directory):
        """Load images from the 'P_DET_IMAGES' folder."""
        if not os.path.exists(directory):
            self.info_label.setText(f"Directory {directory} does not exist")
            return

        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not self.image_paths:
            self.info_label.setText(f"No images found in {directory}")
            return

        self.current_index = 0
        self.info_label.setText(f"Loaded {len(self.image_paths)} images.")

    def display_image(self, image_path):
        """Display the image on the QGraphicsView using QGraphicsScene."""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image file {image_path} does not exist.")
            return  # Exit if image path is invalid

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Unable to load image.")
        else:
            # Clear the scene to remove any previous content
            self.scene.clear()

            # Add the image to the QGraphicsScene
            pixmap_item = self.scene.addPixmap(pixmap)
            pixmap_item.setZValue(-1)  # Ensure the image is at the back

            # Fit the image to the view
            self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            self.info_label.setText(f"Displaying {os.path.basename(image_path)}")

            # Load bounding boxes for the image
            self.load_bounding_boxes(image_path)

    def load_bounding_boxes(self, image_path, format="txt"):
        """Load bounding boxes from the corresponding label file."""
        extension = '.json' if format == "json" else '.txt'
        label_file = os.path.join(
            self.labels_path,
            os.path.basename(image_path).rsplit('.', 1)[0] + extension
        )

        self.list_widget.clear()  # Clear the side list

        if os.path.exists(label_file):
            try:
                if format == "json":
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        bounding_boxes = data.get('bounding_boxes', [])
                        # Update labels to use the label_ID dictionary
                        for box in bounding_boxes:
                            label = box.get('label', '')
                            if label in self.label_ID:
                                box['label'] = self.label_ID[label]  # Replace string label with numeric ID

                else:
                    bounding_boxes = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                label_num, x_center, y_center, width, height = map(float, parts[1:])
                                label = next((key for key, val in self.label_ID.items() if val == int(parts[0])), "")
                                bounding_boxes.append({
                                    'label': label,
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height
                                })

                # Handle the default label (preserving across images)
                if not self.default_label:
                    for box in bounding_boxes:
                        label = box.get('label', '')
                        if label:
                            self.default_label = label
                            break

                # Convert YOLO format to pixel coordinates and add to the scene
                pixmap = QPixmap(image_path)
                image_width = pixmap.width()
                image_height = pixmap.height()

                for box in bounding_boxes:
                    label = box.get('label', '')
                    x_center = box.get('x_center', 0)
                    y_center = box.get('y_center', 0)
                    width = box.get('width', 0)
                    height = box.get('height', 0)

                    x = (x_center - width / 2) * image_width
                    y = (y_center - height / 2) * image_height
                    w = width * image_width
                    h = height * image_height

                    rect = QRectF(x, y, w, h)
                    rect_item = ResizableRectItem(rect, label)
                    rect_item.setPen(QPen(QColor(255, 0, 0), 2))
                    self.scene.addItem(rect_item)
                    self.add_label_to_list(rect_item)

                self.info_label.setText(f"Loaded labels for {os.path.basename(image_path)}")

            except (json.JSONDecodeError, ValueError):
                QMessageBox.warning(self, "Error", f"Invalid format in {label_file}.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load bounding boxes: {e}")
        else:
            self.info_label.setText(f"No labels found for {os.path.basename(image_path)}")

    def save_bounding_boxes(self, image_path, format="txt"):
        """Save bounding boxes to the corresponding label file in YOLO format."""
        label_file = os.path.join(
            self.labels_path,
            os.path.basename(image_path).rsplit('.', 1)[0] + '.json'
        )
        bounding_boxes = []
        pixmap = QPixmap(image_path)
        image_width = pixmap.width()
        image_height = pixmap.height()

        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                rect = item.rect()
                # Calculate YOLO format
                x_center = (rect.x() + rect.width() / 2) / image_width
                y_center = (rect.y() + rect.height() / 2) / image_height
                width = rect.width() / image_width
                height = rect.height() / image_height
                bounding_boxes.append({
                    'label': item.label,
                    'x_center': round(x_center, 6),
                    'y_center': round(y_center, 6),
                    'width': round(width, 6),
                    'height': round(height, 6)
                })
        if format == "json":
            extension = '.json' 
        else:
            extension = '.txt'
        label_file = os.path.join(self.labels_path, os.path.basename(image_path).rsplit('.', 1)[0] + extension)

        try:
            with open(label_file, 'w') as f:
                if format == "json":
                    json.dump({'bounding_boxes': bounding_boxes}, f, indent=4)
                else:
                    for box in bounding_boxes:
                        label_number = self.label_ID.get(box['label'], -1)
                        if label_number != -1:
                            f.write(f"{label_number} {box['x_center']} {box['y_center']} {box['width']} {box['height']}\n")

            self.info_label.setText(f"Saved labels for {os.path.basename(image_path)}")
        except IOError as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save labels: {e}")

    def show_previous_image(self):
        """Show the previous image and load its bounding boxes."""
        if self.current_index > 0:
            current_image = self.image_paths[self.current_index]
            self.save_bounding_boxes(current_image)  # Save before switching
            self.current_index -= 1
            self.display_image(self.image_paths[self.current_index])
        else:
            QMessageBox.information(self, "Start of List", "This is the first image.")

    def show_next_image(self):
        """Show the next image and load its bounding boxes."""
        if self.current_index < len(self.image_paths) - 1:
            current_image = self.image_paths[self.current_index]
            self.save_bounding_boxes(current_image)  # Save before switching
            self.current_index += 1
            self.display_image(self.image_paths[self.current_index])
        else:
            QMessageBox.information(self, "End of List", "This is the last image.")

    def delete_selected_box(self):
        """Delete the currently selected bounding box."""
        selected_items = self.scene.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "No box is selected to delete.")
            return
        for item in selected_items:
            if isinstance(item, ResizableRectItem):
                self.remove_label_from_list(item)
                self.scene.removeItem(item)
        # Save changes after deletion
        if self.image_paths:
            current_image = self.image_paths[self.current_index]
            self.save_bounding_boxes(current_image)
        self.info_label.setText("Selected box(es) deleted.")

    def delete_all_boxes(self):
        """Delete all bounding boxes in the current image."""
        items_to_remove = [item for item in self.scene.items() if isinstance(item, ResizableRectItem)]
        if not items_to_remove:
            QMessageBox.information(self, "No Boxes", "There are no boxes to delete.")
            return
        for item in items_to_remove:
            self.remove_label_from_list(item)
            self.scene.removeItem(item)
        # Save changes after deletion
        if self.image_paths:
            current_image = self.image_paths[self.current_index]
            self.save_bounding_boxes(current_image)
        self.info_label.setText("All boxes deleted.")

    def change_label_name(self):
        """Change the label name for the selected box and all subsequent boxes."""
        selected_items = self.scene.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a box to change its label.")
            return

        # Prompt the user to enter a new label
        new_label, ok = QInputDialog.getText(self, "Change Label Name", "Enter new label name:")
        if not ok or not new_label.strip():
            QMessageBox.information(self, "No Input", "No label name was entered.")
            return

        new_label = new_label.strip()

        # Get the selected box
        selected_item = selected_items[0]
        if not isinstance(selected_item, ResizableRectItem):
            QMessageBox.warning(self, "Invalid Selection", "Selected item is not a bounding box.")
            return

        # Update the label for the selected box and all subsequent boxes
        boxes = [item for item in self.scene.items() if isinstance(item, ResizableRectItem)]
        # Sort boxes based on their creation order (assuming last items are newer)
        boxes_sorted = sorted(boxes, key=lambda x: self.scene.items().index(x), reverse=True)
        try:
            selected_index = boxes_sorted.index(selected_item)
        except ValueError:
            QMessageBox.warning(self, "Error", "Selected box not found.")
            return

        # Only update the selected box and boxes created after it
        for box in boxes_sorted[:selected_index + 1]:
            box.label = new_label
            box.setToolTip(new_label)

        # Update the default_label for future boxes
        self.default_label = new_label

        # Refresh the sidebar to reflect label changes
        self.refresh_list_widget()

        # Save the updated labels to the file
        if self.image_paths:
            current_image = self.image_paths[self.current_index]
            self.save_bounding_boxes(current_image)

        self.info_label.setText(f"Label changed to '{new_label}' for selected and subsequent boxes.")

    def update_list_widget(self, rect_item):
        """Update the corresponding list widget entry for a rect_item."""
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            # To ensure accurate matching, split the text and compare positions
            item_parts = item.text().split(", ")
            if len(item_parts) >= 1 and item_parts[0].startswith("Label:"):
                current_label = item_parts[0].split("Label:")[1].strip()
                x = float(item_parts[1].split("x:")[1].strip())
                y = float(item_parts[2].split("y:")[1].strip())
                w = float(item_parts[3].split("w:")[1].strip())
                h = float(item_parts[4].split("h:")[1].strip())
                if (current_label == rect_item.label and
                    abs(x - rect_item.rect().x()) < 0.0001 and
                    abs(y - rect_item.rect().y()) < 0.0001 and
                    abs(w - rect_item.rect().width()) < 0.0001 and
                    abs(h - rect_item.rect().height()) < 0.0001):
                    # Reconstruct the label text with updated label
                    label_text = f"Label: {rect_item.label}, " \
                                 f"x: {rect_item.rect().x():.4f}, " \
                                 f"y: {rect_item.rect().y():.4f}, " \
                                 f"w: {rect_item.rect().width():.4f}, " \
                                 f"h: {rect_item.rect().height():.4f}"
                    self.list_widget.item(index).setText(label_text)
                    break

    def closeEvent(self, event):
        """Prompt to save bounding boxes on exit."""
        reply = QMessageBox.question(self, 'Quit',
                                     "Do you want to save changes before quitting?",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                     QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            if self.image_paths:
                current_image = self.image_paths[self.current_index]
                self.save_bounding_boxes(current_image)
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    labeler = ImageLabeler()
    labeler.show()
    sys.exit(app.exec_())