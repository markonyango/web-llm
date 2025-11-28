import "./style.css";
import { img_classifier, load } from "./llm.ts";
import type { DeviceType } from "@huggingface/transformers";

const button = document.getElementById('magic');

button?.addEventListener('click', event => {
    const device = (document.getElementById('device') as HTMLSelectElement)?.value as DeviceType;
    event.preventDefault();
    
    Promise.allSettled([
        load(device),
        img_classifier(device)
    ]);
})

