import torch
import transformers

# define a model id. In huggingface it is formated as user_name/model_name.
# this code will work with any huggingface model: https://huggingface.co/models?pipeline_tag=text-generation
# but its important to know that the average computer does not have the specifications to run anything beyond a few billion parameters.

# in this code, the amount of RAM used is the number of parametersmultiplied by 2
# this is because each parameter/number is using 16 bits, or 2 bytes

model_id = "LiquidAI/LFM2.5-1.2B-Instruct"

"""
define the generation pipeline. this includes:
1. the objective of the ai model (text generation)
2. the model id (in our case, LiquidAI/LFM2.5-1.2B-Instruct)
3. which device to run the model. Its best to use a graphics card if available
4. what quantization to run the model in, or how many bits should be used for each number in the weights/parameters (in our case, bfloat16/bf16 is 16 bits)
"""

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map=None,
    dtype=torch.bfloat16
)

# define a system prompt
system_prompt = "You are a helpful assistant."

# format the system prompt as a dictionary.
system_prompt = {"role": "system", "content": system_prompt}

# create a list representing all previous messages in the conversation apply the system prompt.
messages = [system_prompt]

while True:
    # ask the user for a prompt
    prompt = input("Prompt: ")

    # if the prompt is 'exit', halt the loop
    if prompt == "exit":
        break

    # otherwise, format the prompt as a dictionary.
    prompt = {"role": "user", "content": prompt}

    # add the prompt into a list containing all the previous information
    messages.append(prompt)

    # generate the response given the previous messages and the prompt
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )

    # remove the format the response is in, and turn it into a string
    outputs = outputs[0]["generated_text"][-1]["content"]

    # print the respone
    print("Response: ", outputs)

    # format the response into a dictionary
    outputs = {"role": "assistant", "content": outputs}

    # add the response to the list of all previous messages
    messages.append(outputs)

    # repeat until the user's prompt is exit
