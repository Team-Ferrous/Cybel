import os
import ast
from typing import List


class Visualizer:
    """
    Generates visual representations of code and workflows.
    """

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def generate_class_diagram(self, path: str) -> str:
        """
        Generates a Mermaid class diagram for a given Python file.
        """
        full_path = os.path.join(self.root_dir, path)
        if not os.path.exists(full_path):
            return f"Error: {path} not found."

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            mermaid = "classDiagram\n"
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    mermaid += f"    class {node.name} {{\n"
                    # Add methods
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef):
                            args = [a.arg for a in sub.args.args if a.arg != "self"]
                            mermaid += f"        +{sub.name}({', '.join(args)})\n"
                    mermaid += "    }\n"

                    # Add simple inheritance
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            mermaid += f"    {base.id} <|-- {node.name}\n"

            return mermaid
        except Exception as e:
            return f"Error generating diagram: {str(e)}"

    def generate_flowchart(self, steps: List[str]) -> str:
        """
        Generates a Mermaid flowchart for a list of steps.
        """
        mermaid = "graph TD\n"
        for i in range(len(steps)):
            node_id = f"step{i}"
            mermaid += f"    {node_id}[{steps[i]}]\n"
            if i > 0:
                mermaid += f"    step{i-1} --> {node_id}\n"
        return mermaid
