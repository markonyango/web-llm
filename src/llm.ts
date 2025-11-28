import { pipeline, type DeviceType } from "@huggingface/transformers";

export async function load(device: DeviceType = "wasm") {
  const output_element = document.getElementById("output")!;

  // Allocate pipeline
  let generator: any = undefined;
  try {
    output_element.textContent = "Loading model...";
    generator = await pipeline(
      "text-generation",
      "onnx-community/Qwen2.5-0.5B-Instruct",
      { dtype: "q4", device },
    );
  } catch (error) {
    console.log("Error occurred while loading model:", error);
    output_element.textContent =
      "Error occurred while loading model\n" + JSON.stringify(error);
    return;
  }
  
  const messages = [
    { role: "system", content: "You are a senior Angular developer" },
    {
      role: "user",
      content: "What is the difference between a Promise and an Observable?",
    },
  ];

  output_element.textContent = "Prompting model...";
  const output = await generator(messages, { max_new_tokens: 1024 });


  const system = document.createElement("p");
  const user = document.createElement("p");
  const assistant = document.createElement("p");

  system.textContent = output[0].generated_text[0].content;
  user.textContent = output[0].generated_text[1].content;
  assistant.innerHTML = output[0].generated_text[2].content;

  output_element.textContent = '';
  output_element.appendChild(system);
  output_element.appendChild(user);
  output_element.appendChild(assistant);
}

export async function img_classifier(device: DeviceType = "wasm") {
  const classifier = await pipeline(
    "image-classification",
    "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    { device },
  );

  const url =
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg";
  const elem = document.getElementById("output2")!;

  let img = document.createElement("img");
  img.src = url;
  elem.appendChild(img);

  const output = await classifier(url);

  let out_elem = document.createElement("span");
  out_elem.textContent = JSON.stringify(output, undefined, 2);
  elem.appendChild(out_elem);
}
