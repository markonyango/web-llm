import "./style.css";
import { img_classifier, load } from "./llm.ts";
import type { DeviceType } from "@huggingface/transformers";

const button = document.getElementById("magic");

button?.addEventListener("click", (event) => {
  const device = (document.getElementById("device") as HTMLSelectElement)
    ?.value as DeviceType;
  const prompt_form = document.getElementById("prompt_form") as HTMLFormElement;

  const form_data = new FormData(prompt_form);
  const system_prompt = form_data.get("system_prompt");
  const user_prompt = form_data.get("user_prompt");

  console.log(system_prompt, user_prompt);

  event.preventDefault();

  Promise.allSettled([
    load(
      device,
      system_prompt?.toString() ?? "",
      user_prompt?.toString() ?? "",
    ),
    img_classifier(device),
  ]);
});
