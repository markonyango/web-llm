import {
  pipeline,
  type DeviceType,
  type TextGenerationOutput,
} from "@huggingface/transformers";

export async function load(
  device: DeviceType = "wasm",
  system_prompt: string,
  user_prompt: string,
) {
  const output_element = document.getElementById("output")!;

  // Allocate pipeline
  let generator: any = undefined;
  try {
    output_element.textContent = "Loading model...";
    generator = await pipeline(
      "text-generation",
      "onnx-community/Qwen2.5-0.5B-Instruct",
      { dtype: "q4", device, progress_callback: console.log },
    );
  } catch (error) {
    console.log("Error occurred while loading model:", error);
    output_element.textContent =
      "Error occurred while loading model\n" + JSON.stringify(error);
    return;
  }

  const messages = [
    {
      role: "system",
      content: system_prompt,
    },
    {
      role: "user",
      content: user_prompt,
    },
  ];

  output_element.textContent = "Prompting model...";
  const output = (await generator(messages, {
    max_new_tokens: 1024,
  })) as TextGenerationOutput[];

  const system = document.createElement("p");
  const user = document.createElement("p");
  const assistant = document.createElement("p");

  system.textContent = output[0].generated_text[0].content;
  user.textContent = output[0].generated_text[1].content;
  assistant.innerHTML = output[0].generated_text[2].content;

  output_element.textContent = "";
  output_element.appendChild(system);
  output_element.appendChild(user);
  output_element.appendChild(assistant);
}

export async function img_classifier(device: DeviceType = "wasm") {
  const elem = document.getElementById("output2")!;
  elem.textContent = "";
  const status = document.createElement("span");
  elem.appendChild(status);

  status.textContent = "Loading model...";
  const classifier = await pipeline(
    "image-classification",
    "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    { device },
  );

  const url =
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg";

  let img = document.createElement("img");
  img.src = url;
  img.width = 400;
  elem.appendChild(img);

  status.textContent = "Classifying image...";
  const output = await classifier(url);

  let out_elem = document.createElement("span");
  out_elem.textContent = JSON.stringify(output, undefined, 2);
  status.textContent = "";
  elem.appendChild(out_elem);
}
