import ast
import os


class PyFileParser(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.classes = {}
        self.current_class = None

    def parse(self):
        with open(self.filepath, "r") as file:
            root = ast.parse(file.read(), self.filepath)
            self.visit(root)

        return self.classes

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.classes[self.current_class] = []
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class is not None:
            # Only consider methods, not standalone functions
            self.classes[self.current_class].append(node.name)
        self.generic_visit(node)


def simulate_tree(startpath, indent="  "):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent_level = indent * level
        print(f"{indent_level}{os.path.basename(root)}/")
        subindent = indent * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
            if f.endswith(".py"):
                try:
                    parser = PyFileParser(os.path.join(root, f))
                    classes = parser.parse()
                    for cls, methods in classes.items():
                        print(f"{subindent*2}[Class] {cls}")
                        for method in methods:
                            print(f"{subindent*3}- {method}")
                except SyntaxError as e:
                    print(f"{subindent*2}Error parsing {f}: {e}")


start_dir = "src"  # Starting directory, '.' for current
simulate_tree(start_dir)
