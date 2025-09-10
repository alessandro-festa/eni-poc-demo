import warnings
import randomname

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

import functools
import inspect
import os
import textwrap

import openai

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

# Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001/"
os.environ["OPENAI_API_KEY"] = "ollama" 
# Or directly in code
mlflow.set_tracking_uri("http://localhost:5001/")
client = openai.OpenAI(
    base_url = 'http://localhost:31434/v1',
    api_key='ollama', # required, but unused
)
name = randomname.get_name()
mlflow.set_experiment(name)

instruction = [
  {
      "role": "system",
      "content": (
          "As an AI specializing in code review, your task is to analyze and critique the submitted code. For each code snippet, provide a detailed review that includes: "
          "1. Identification of any errors or bugs. "
          "2. Suggestions for optimizing code efficiency and structure. "
          "3. Recommendations for enhancing code readability and maintainability. "
          "4. Best practice advice relevant to the code's language and functionality. "
          "Your feedback should help the user improve their coding skills and understand best practices in software development."
      ),
  },
  {"role": "user", "content": "Review my code and suggest improvements: {code}"},
]

# Define the model signature that will be used for both the base model and the eventual custom pyfunc implementation later.
signature = ModelSignature(
  inputs=Schema([ColSpec(type="string", name=None)]),
  outputs=Schema([ColSpec(type="string", name=None)]),
  params=ParamSchema(
      [
          ParamSpec(name="max_tokens", default=500, dtype="long"),
          ParamSpec(name="temperature", default=0, dtype="float"),
      ]
  ),
)

# Log the base OpenAI model with the included instruction set (prompt)
with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="qwen3:0.6b",
      task=openai.chat.completions,
      name="base_model",
      messages=instruction,
      signature=signature,
  )

# Custom pyfunc implementation that applies text and code formatting to the output results from the OpenAI model
class CodeHelper(PythonModel):
  def __init__(self):
      self.model = None

  def load_context(self, context):
      self.model = mlflow.pyfunc.load_model(context.artifacts["model_path"])

  @staticmethod
  def _format_response(response):
      formatted_output = ""
      in_code_block = False

      for item in response:
          lines = item.split("")
          for line in lines:
              # Check for the start/end of a code block
              if line.strip().startswith("```"):
                  in_code_block = not in_code_block
                  formatted_output += line + ""
                  continue

              if in_code_block:
                  # Don't wrap lines inside code blocks
                  formatted_output += line + ""
              else:
                  # Wrap lines outside of code blocks
                  wrapped_lines = textwrap.fill(line, width=80)
                  formatted_output += wrapped_lines + ""

      return formatted_output

  def predict(self, context, model_input, params):
      # Call the loaded OpenAI model instance to get the raw response
      raw_response = self.model.predict(model_input, params=params)

      # Return the formatted response so that it is easier to read
      return self._format_response(raw_response)

# Define the location of the base model that we'll be using within our custom pyfunc implementation
artifacts = {"model_path": model_info.model_uri}

with mlflow.start_run():
  helper_model = mlflow.pyfunc.log_model(
      name="code_helper",
      python_model=CodeHelper(),
      input_example=["x = 1"],
      signature=signature,
      artifacts=artifacts,
  )

loaded_helper = mlflow.pyfunc.load_model(helper_model.model_uri)