from core.visualizer import Visualizer


def visualize_code(path: str, diagram_type: str = "class"):
    viz = Visualizer()
    if diagram_type == "class":
        return viz.generate_class_diagram(path)
    return "Unsupported diagram type."
