# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:33:29 2023

@author: Michel Gordillo
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QVBoxLayout, QWidget)


class MainWindow(QTabWidget):
    def __init__(
        self,
        title="UAD Aircraft Developer Tool",
        window_size=(600, 400),
        *args,
        **kwargs,
    ):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.create_global_variables()

        self.setWindowTitle(title)
        self.resize(*window_size)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        self.addTab(self.tab3, "Tab 3")
        self.setCurrentIndex(0)
        self.tab2.setEnabled(False)
        self.tab3.setEnabled(False)

        # Create the exit button and layout

        # self.exit_layout.addWidget(self.exit_button, alignment=Qt.AlignLeft | Qt.AlignBottom)

        # Tab 1 layout
        tab1_layout = QVBoxLayout()

        # Create the button
        self.button_enabler = QPushButton("Click me to enable tabs 2 and 3")
        # Connect to (function)
        self.button_enabler.clicked.connect(self.enable_tabs)
        # Add to layout
        tab1_layout.addWidget(self.button_enabler)

        self.XML_file_Button = QPushButton("Select an file")
        self.XML_file_Button.clicked.connect(self.get_xml_path)
        tab1_layout.addWidget(self.XML_file_Button)

        # Create the exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close_app)

        tab1_layout.addWidget(
            self.exit_button, alignment=Qt.AlignRight | Qt.AlignBottom
        )

        # Tab 1 set layout
        self.tab1.setLayout(tab1_layout)

        # Tab 2 layout
        tab2_layout = QVBoxLayout()

        # Create text and button side by side
        self.hbox, self.button1, self.line1 = self.create_bilateral(
            button_text="File Explorer (XML)",
            function=self.get_xml_path,
            line_text="Insert XML file path",
        )

        # Add the hbox to the vbox
        tab2_layout.addLayout(self.hbox)

        # Create text and button side by side
        self.hbox2, self.button2, self.line2 = self.create_bilateral(
            button_text="Set Airfoil Folder",
            function=self.get_airfoil_folder_path,
            line_text="Insert airfoil folder path",
        )

        # Add the hbox to the vbox
        tab2_layout.addLayout(self.hbox2)

        # Create text and button side by side
        self.hbox3, self.button3, self.line3 = self.create_bilateral(
            button_text="Set Working Directory",
            function=self.get_working_directory,
            line_text="Insert airfoil folder path",
        )

        # Add the hbox to the vbox
        tab2_layout.addLayout(self.hbox3)

        # self.label2 = QLabel("Tab 2")
        # tab2_layout.addWidget(self.label2)

        # tab2_layout.addWidget(self.exit_button)

        self.tab2.setLayout(tab2_layout)

        # Tab 3 layout
        tab3_layout = QVBoxLayout()
        self.label3 = QLabel("Tab 3")
        tab3_layout.addWidget(self.label3)
        self.tab3.setLayout(tab3_layout)

    def create_global_variables(self):
        self.XMLfile = None
        self.AIRFOIL_FOLDER_PATH = None
        self.WORKING_DIRECTORY = None
        self.OP_POINT_FILE = None
        self.POLAR_FILE = None

    def create_bilateral(
        self,
        button_text="Click me to enable tabs 2 and 3",
        function=lambda: None,
        line_text="Insert XML file path",
    ):
        # Create a QHBoxLayout
        hbox = QHBoxLayout()
        # Add widgets to the hbox
        # Create the button
        button1 = QPushButton(button_text)
        # Connect to (function)
        button1.clicked.connect(function)

        # Line Edit
        line1 = QLineEdit()
        line1.setText(line_text)

        # Add to layout
        hbox.addWidget(button1)
        hbox.addWidget(line1)

        return hbox, button1, line1

    def get_xml_path(self, file_path):
        self.XMLfile = get_file(file_path=None, extension=".xml")
        self.line1.setText(self.XMLfile)

    def get_airfoil_folder_path(self, file_path):
        self.AIRFOIL_FOLDER_PATH = get_folder_path()
        self.line2.setText(self.AIRFOIL_FOLDER_PATH)

    def get_working_directory(self, file_path):
        self.WORKING_DIRECTORY = get_folder_path()
        self.line3.setText(self.AIRFOIL_FOLDER_PATH)

    def enable_tabs(self):
        self.tab2.setEnabled(True)
        self.tab3.setEnabled(True)

    def close_app(self):
        self.close()
        QApplication.exit()

    def closeEvent(self, event):
        self.close_app()


def get_file(file_path=None, extension=None):

    filters = {
        ".txt": "Text Files (*.txt);;All Files (*)",
        ".xml": "XML files (*.xml);;All files (*)",
    }

    if extension in filters:
        filter_str = filters[extension]
    elif isinstance(extension, str):
        filter_str = f"Custom files(*{extension});;All Files (*)"
    else:
        print(" filter was unspecified, reading all files")
        filter_str = "All files (*)"

    if file_path is None or not os.path.exists(file_path):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select a file", "", filter_str, options=options
        )
        return file_path
    else:
        print("An error flew by")
        raise (Exception())


def get_folder_path():
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
    return folder_path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowState(Qt.WindowActive)
    # main_window.setWindowFlags(Qt.WindowStaysOnTopHint)
    main_window.show()
    sys.exit(app.exec_())
