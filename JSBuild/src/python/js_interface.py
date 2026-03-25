# sparc_server.py
import sys
import subprocess
from   python.newbie_pipeline  import run_newbie_pipeline
from   python.sparc_server     import generate_sdf_local
from   fast_graphrag           import generate_grag
from   thrml_graph_generator   import generate_thrml, run_query
from   thrml_merkle_pygad      import build_mtree, generate_thrml_from_mtree, predict_ga_from_graph, run_image as _run_image, run_text as _run_text
from   oqtopus_graph_generator import QuantumCircuitGraph, QuantumNode, extract_semantic_graph, run
from   oqtopus_merkle_pygad    import build_merkle_quantum_graph, run_merkle_quantum_query


class PreswaldClient:
    def __init__(self, project_path="."):
        self.project_path = project_path
        self.process = None

    def init(self, name):
        subprocess.run(
            ["preswald", "init", name],
            cwd=self.project_path,
            check=True
        )

    def run(self):
        if self.process:
            print("Preswald already running.")
            return

        self.process = subprocess.Popen(
            ["preswald", "run"],
            cwd=self.project_path
        )

        print("Preswald server started.")

    def export(self):
        subprocess.run(
            ["preswald", "export"],
            cwd=self.project_path,
            check=True
        )

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("Preswald server stopped.")

class SaguaroClient:
    def __init__(self):
        self.initialized = False

    def init(self):
        subprocess.run(["saguaro", "init"], check=True)
        self.initialized = True

    def index(self, path="."):
        subprocess.run(["saguaro", "index", "--path", path], check=True)

    def query(self, text, k=5):
        result = subprocess.run(
            ["saguaro", "query", text, "--k", str(k)],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
# for HNLF to Cybel Compatibility
# also useful for cross model interactions and testing
def tf_to_pytorch(keras_model):
    import tf2onnx
    from onnx2pytorch import ConvertModel

    # Convert the Keras model to ONNX format
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model)
    
    # Convert ONNX model to PyTorch
    pytorch_model = ConvertModel(onnx_model)
    return pytorch_model

def pytorch_to_tf(pytorch_model, input_shape):
    import torch
    import onnx
    from onnx_tf.backend import prepare

    dummy_input = torch.randn(*input_shape)

    # Export PyTorch model to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    # Load ONNX model
    onnx_model = onnx.load("model.onnx")

    # Convert ONNX → TensorFlow
    tf_rep = prepare(onnx_model)

    # Returns TensorFlow representation
    return tf_rep

def makeONNX_Model(model):
    import torch
    dummy = torch.randn(1, 128)
    onx_mdl = torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        opset_version=17
    )

#Execute in Python
def run_mlir_runtime():
    import iree.runtime as ireert
    import numpy as np

    config = ireert.Config("local-task")
    vm_module = ireert.load_vm_module("model.vmfb", config)
    input = np.random.randn(1,128).astype(np.float32)
    result = vm_module.main(input)


import subprocess
from pathlib import Path


def onnx_to_mlir(onnx_path: str, mlir_path: str | None = None):
    """
    Convert ONNX model to MLIR using onnx-mlir.
    """
    onnx_path = Path(onnx_path)

    if mlir_path is None:
        mlir_path = onnx_path.with_suffix(".mlir")

    cmd = [
        "onnx-mlir",
        str(onnx_path),
        "-emit-mlir",
        f"-o={mlir_path}",
    ]

    subprocess.run(cmd, check=True)
    return str(mlir_path)


def mlir_to_vmfb(mlir_path: str, vmfb_path: str | None = None, backend="llvm-cpu"):
    """
    Compile MLIR to an IREE runtime module (.vmfb).
    """
    mlir_path = Path(mlir_path)

    if vmfb_path is None:
        vmfb_path = mlir_path.with_suffix(".vmfb")

    cmd = [
        "iree-compile",
        str(mlir_path),
        f"--iree-hal-target-backends={backend}",
        "-o",
        str(vmfb_path),
    ]
    subprocess.run(cmd, check=True)
    return str(vmfb_path)

def run_saguaro(working_dir, text):
    saguaro = SaguaroClient()
    if not saguaro.initialized:
        saguaro.init()
    saguaro.index(path=working_dir)  # Index specified directory
    results = saguaro.query(text)
    print("Saguaro Query Results:", results)

def run_preswald(working_dir, command):
    preswald = PreswaldClient()
    preswald.project_path = working_dir
    if command == "init":
        preswald.init()
    elif command == "run":
        preswald.run()
    elif command == "export":
        preswald.export()
    elif command == "stop":
        preswald.stop()

# Only these functions are callable via JS
_allowed_commands = {
    "call-sparc":                  lambda args:              generate_sdf_local(args[0], args[1], args[2]),
    "call-newbie":                 lambda prompt:            run_newbie_pipeline(prompt),
    "call-preswald":               lambda args:              run_preswald(args[0], args[1]),
    "call-saguaro":                lambda working_dir, text: run_saguaro(working_dir, text),
    "run-thrml-query":             lambda args:              run_query(args[0], args[1], args[2], args[3]),
    "run-image-thrml-ga":          lambda args:              _run_image(args[0], args[1], args[2]),
    "run-text-thrml-ga":           lambda args:              _run_text(args[0], args[1], args[2]),

    "generate-grag":               lambda text:              generate_grag(text),
    "extract-semantic-graph":      lambda text:              extract_semantic_graph(None, text),  # LLM integration would go here
    "generate-mtree":              lambda inputs:            build_mtree(inputs),
    "generate-thrml-graph":        lambda args:              generate_thrml_from_mtree(args[0], args[1]),  # Placeholder for actual graph generation
    "run-quantum-query":           lambda args:              run(args[0], args[1], args[2], args[3]),
    "generate-quantum-graph":      lambda _:                 QuantumCircuitGraph(nodes=[QuantumNode(f"q{i}") for i in range(5)]),

    "predict-ga-from-thrml-graph": lambda args:              predict_ga_from_graph(args[0], args[1]),
    "build_merkle_quantum_graph":  lambda args:              build_merkle_quantum_graph(args[0], args[1]),
    "run-merkle-quantum-query-ga": lambda args:              run_merkle_quantum_query(args[0], args[1], args[2])
}

def main():
    args = sys.argv[1:]

    # Determine which command
    command = None
    param   = None
    if "--call-sparc" in args:
        command = "call-sparc"
        uid    = args[args.index("--call-sparc") + 1]
        prompt = args[args.index("--call-sparc") + 2]
        n      = args[args.index("--call-sparc") + 3]
        param = (uid, prompt, n)
    elif "--call-newbie" in args:
        command = "call-newbie"
        param = args[args.index("--call-newbie") + 1]
    elif "--call-saguaro" in args:
        command = "call-saguaro"
        working_dir = args[args.index("--call-saguaro") + 1]
        text =        args[args.index("--call-saguaro") + 2]
        param = (working_dir, text)
    elif "--generate-grag" in args:
        command = "generate-grag"
        text = args[args.index("--generate-grag") + 1]
        param = text
    elif "--extract-semantic-graph" in args:
        command = "extract-semantic-graph"
        text = args[args.index("--extract-semantic-graph") + 1]
        param = text
    elif "--discover-entity-types" in args:
        command = "discover-entity-types"
        text = args[args.index("--discover-entity-types") + 1]
        param = text
    elif "--discover-relationships" in args:
        command = "discover-relationships"
        text = args[args.index("--discover-relationships") + 1]
        param = text
    elif "--discover-queries" in args:
        command = "discover-queries"
        text = args[args.index("--discover-queries") + 1]
        param = text
    elif "--generate-mtree" in args:
        command = "generate-mtree"
        inputs = args[args.index("--generate-mtree") + 1]
        param = inputs
    elif "--generate-thrml-graph" in args:
        command = "generate-thrml-graph"
        arg1 = args[args.index("--generate-thrml-graph") + 1]
        arg2 = args[args.index("--generate-thrml-graph") + 2]
        param = (arg1, arg2)
    elif "--generate-quantum-graph" in args:
        command = "generate-quantum-graph"
        param = None
    elif "--run-quantum-query" in args:
        command = "run-quantum-query"
        arg1 = args[args.index("--run-quantum-query") + 1]
        arg2 = args[args.index("--run-quantum-query") + 2]
        arg3 = args[args.index("--run-quantum-query") + 3]
        arg4 = args[args.index("--run-quantum-query") + 4]
        param = (arg1, arg2, arg3, arg4)
    elif "--run-thrml-query" in args:
        command = "run-thrml-query"
        arg1 = args[args.index("--run-thrml-query") + 1]
        arg2 = args[args.index("--run-thrml-query") + 2]
        arg3 = args[args.index("--run-thrml-query") + 3]
        arg4 = args[args.index("--run-thrml-query") + 4]
        param = (arg1, arg2, arg3, arg4)
    elif "--run-image-thrml-ga" in args:
        command = "run-image-thrml-ga"
        param = args[args.index("--run-image-thrml-ga") + 1]
    elif "--run-text-thrml-ga" in args:
        command = "run-text-thrml-ga"
        param = args[args.index("--run-text-thrml-ga") + 1]
    elif "--predict-ga-from-thrml-graph" in args:
        command = "predict-ga-from-thrml-graph"
        arg1 = args[args.index("--predict-ga-from-thrml-graph") + 1]
        arg2 = args[args.index("--predict-ga-from-thrml-graph") + 2]
        param = (arg1, arg2)
    elif "--build_merkle_quantum_graph" in args:
        command = "build_merkle_quantum_graph"
        arg1 = args[args.index("--build_merkle_quantum_graph") + 1]
        arg2 = args[args.index("--build_merkle_quantum_graph") + 2]
        param = (arg1, arg2)
    elif "--run-merkle-quantum-query-ga" in args:
        command = "run-merkle-quantum-query-ga"
        arg1 = args[args.index("--run-merkle-quantum-query-ga") + 1]
        arg2 = args[args.index("--run-merkle-quantum-query-ga") + 2]
        arg3 = args[args.index("--run-merkle-quantum-query-ga") + 3]
        param = (arg1, arg2, arg3)
    else:
        print("No valid command provided. Allowed: --run-image <path>, --run-text <text>")
        sys.exit(1)

    # Execute only allowed commands
    func = _allowed_commands.get(command)
    if func:
        func(param)
    else:
        print(f"Command {command} is not allowed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
