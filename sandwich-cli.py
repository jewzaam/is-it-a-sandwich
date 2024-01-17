from fastai.vision.all import *
import sys

# the warnings in the libraries used can get old, do not show them
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) <= 1:
    print("HELP: provide a single argument with a path to an image")
    sys.exit(-1)
image = sys.argv[1]

learn = load_learner('sandwich.pkl')

result = learn.predict(image)

print(f"The image '{image}' is: {result[0]} ({int(100*result[2][result[1]])}%)")

