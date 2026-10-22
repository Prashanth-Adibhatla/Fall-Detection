# Compatibility shim for Python 3.12+
# tflite_runtime has no pre-built wheel for 3.12; tensorflow-cpu ships the same API.
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import load_delegate
