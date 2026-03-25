# web service interfaces
import supabase       as S

# tools
import os
import sys
import yaml
from   tempfile import NamedTemporaryFile
from   dotenv   import load_dotenv, dotenv_values 

# machine learning components
import pygad # Artificial Machine Learning
from   merkly.mtree import MerkleTree   
from   typing import Callable
import numpy
from   PIL import Image
import argparse

import dct 
import dwt 

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument("--dct",                       help="Perform Discrete Cosine  Transform instead of FFT",              type=int)
parser.add_argument("--dwt",                       help="Perform Discrete Wavelet Transform instead of FFT",              type=int)
parser.add_argument("-t",  "--target",             help="Select target image for GA to train towards",                    type=int)
parser.add_argument("-m",  "--output_trie",        help="Designate output file for generated Merkle Trie",                type=int)
parser.add_argument("-p",  "--output_predictions", help="Designate output file for generated Predictions",                type=int)
parser.add_argument("-i",  "--output_image",       help="Designate output filename for generated image",                  type=int)
args = parser.parse_args()
#print(args.square**2)

with open('GA_Options.yaml', 'r') as f:
    GAData = yaml.load(f, Loader=yaml.SafeLoader)

# accessing supabase
url: str           = os.getenv("SUPABASE_URL")
key: str           = os.getenv("SUPABASE_KEY")
supabase: S.Client = S.create_client(url, key)

mhash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y
function_inputs = [4, -2, 3.5, 5, -11,-4.7]
desired_output  = 44
bytes_inputs    = ['a', 'b', 'c', 'd', 'e', 'f']
mtree           = MerkleTree(bytes_inputs, mhash_function)

#Example Case:
#y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
#where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

# choose any hash function that is of type (bytes, bytes) -> bytes
# my_hash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y

# # create a Merkle Tree
# mtree = MerkleTree(['a', 'b', 'c', 'd'], my_hash_function)

# # show original input
# assert mtree.raw_leaves == ['a', 'b', 'c', 'd']

# # hashed leaves
# assert mtree.leaves == [b'a', b'b', b'c', b'd']

# # shorted hashed leaves
# assert mtree.short_leaves == [b'a', b'b', b'c', b'd']

# Supabase Functions (Auth, Database, Datastore)
def query():
    data = supabase.auth.get_session()
    if data is not None:
        response = supabase.table('metadata').select("associated_emotions").execute()
    return response

def insert(id:int, mem:[str], pred:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('metadata').insert({"id": id, "solution": mem, "prediction":  pred}).execute()

def update(id:int, mem:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('metadata').update({"associated_memories": mem}).eq('id', id).execute()

def upload(bucket_name, destination, source):
    res  = supabase.storage.from_(bucket_name).update(destination, source)
    return res
    
def download(bucket_name, file_name, dl_path):
    with open(file_name, 'wb+') as f:
        res = supabase.storage.from_(bucket_name).download(dl_path)
        f.write(res)

def register_user(email: str, password: str):
   supabase.auth.sign_up({
    "email":    email,
    "password": password
    })
   
def login_user(email, password):
    access_token = supabase.auth.sign_in_with_password({"email": email, "password": password})
    localStorage.setItem("access_token", supabase.auth.get_session())
    index = load_from_cloud_data_bucket('MatterBucket', f'{st.session_state.username.split("@")[0]}/conversation_master.pdf', st.session_state["model"])
  
def logout_user():
    supabase.auth.sign_out()

#functions
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

def run():
    input_image  = []
    target_image = Image.open("target.jpg").convert('L')  # grayscale
    fitness_function = fitness_func
    num_genes = len(function_inputs)

    # Decide transform function
    if args.dct:
        transform_func = dct.Encode
        print("Using DCT transform")
    elif args.dwt:
        transform_func = dwt.Encode
        print("Using DWT transform")
    else:
        transform_func = lambda im_array: numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im_array)))
        print("Using default FFT transform")

    for x in range(0, GAData['NUM_ITERATIONS']):
        ga_instance = pygad.GA(
            num_generations=GAData['NUM_GENERATIONS'],
            num_parents_mating=GAData['NUM_PARENTS_MATING'],
            fitness_func=fitness_function,
            sol_per_pop=GAData['SOL_PER_POP'],
            num_genes=num_genes,
            init_range_low=GAData['INIT_RANGE_LOW'],
            init_range_high=GAData['INIT_RANGE_HIGH'],
            parent_selection_type=GAData['PARENT_SELECTION_TYPE'],
            keep_parents=GAData['KEEP_PARENTS'],
            crossover_type=GAData['CROSSOVER_TYPE'],
            mutation_type=GAData['MUTATION_TYPE'],
            mutation_percent_genes=GAData['MUTATION_PERCENT_GENES']
        )
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")

        prediction = numpy.sum(numpy.array(function_inputs) * solution)
        print(f"Predicted output based on the best solution : {prediction}")

        # Transform image
        im_array = numpy.array(target_image)
        visual = transform_func(im_array)

        # If using FFT, scale by prediction
        if not args.dct and not args.dwt:
            visual = numpy.log(visual + 1e-8) * prediction

        # Normalize for image output
        visual = (visual - visual.min()) / (visual.max() - visual.min()) * 255
        input_image = Image.fromarray(visual.astype(numpy.uint8))
        input_image.save('out.bmp')

        # Write predictions
        with open("predictions.txt", "w+") as file1:
            file1.write(f"Solution Index:        {solution_idx}\n")
            file1.write(f"Best Soln. Parameters: {solution}\n")
            file1.write(f"Best Prediction:       {prediction}\n\n")

        # Merkle tree leaves
        with open("predictions_mtree.txt", "w+") as file1:
            file1.write(f"Solution Index (mtree): {mtree.raw_leaves}\n")

        # Insert into Supabase
        insert(GAData['NUM_ITERATIONS'], solution.tolist(), [str(prediction)])

'''
# X, Y, and Z location to set
default_cube.location = (0.0, 0.0, 0.0)
# Set the keyframe with that location, and which frame.
default_cube.keyframe_insert(data_path="location", frame=1)

# do it again!
default_cube.location = (3.0, 2.0, 1.0)
# setting it for frame 10
default_cube.keyframe_insert(data_path="location", frame=10)
'''