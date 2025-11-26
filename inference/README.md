#  Running the Alt-Text Generation Model 
1. Create a virtual environment  
    ```bash
    python3 -m venv venv
    ```
2. Activate the virtual environment  
    ```bash
    source venv/bin/activate
    ```
3. Install packages  
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers
    pip install pillow
    pip install pyyaml    
    ```
4. Update the existing `config.yaml` file to reflect the correct model choice.

5. Run the inference.py script
    ```bash
    python3 inference/scripts/inference.py
    ```

6. View the generated captions in terminal or json.
    Output will be saved to:
    ```
    ./outputs/captions_filled.json
    ```

## nlpconnect/vit-gpt2-image-captioning [1]

This model uses two transformer models: 
1. Vision Transformer (ViT) - image encoder:
 The input image is split up into flattened patches, which also have positional data [2]. This sequence of patches are provided as an input into a standard transform encoder and used to pre-train the model with image labels [2]. 
2. GPT- 2 - language model decoder: The architecture of GPT is made up of only decoders [3]. The input text is broken into small units, often called tokens and each is mapped to a numerical ID [3]. These IDs are converted into vectors that represent the semantic meaning of the tokens and are passed into the consecutive decoder blocks to generate text one token at a time [3]. 

The ViT extracts features from the inputted image and GPT-2 produces a description of these features. Though this model is fast, often the produced alt text doesn't communicate the correct context and purpose of the image. 

* Transformer architecture is a type of neural network that can be used for natural language processing tasks. 
* Note Transformers are non sequential, which is one way they differ from Recurrent Neural Networks (RNNs) [2]. Additionally, transforms can access past information, while RNNs can only access previous state [2]. 

## Salesforce/blip-image-captioning-base [5]
This model uses a Bootstrapping Language-Image Pre-training (BLIP) for Unified Vision-Language Understanding and Generation

This is new Vision-Language Pre-training framework can be used for both vision-language understanding and generation tasks, while previous models focused on one or the other [5]. Model have been able to improve their performance by increasing the size of the dataset used for training, by including noisy image-text pairs found on the internet [5]. BLIP has been able to reduce the noise on these image-text pairs by using a caption generator and a filter, to generate new captions and filter defective ones [5]. 

BLIP uses a similar architecture to the nlpconnect/vit-gpt2-image-captioning as it uses a Vision Encoder (ViT) and a text encoder [6].


### References 
[1] “Nlpconnect/VIT-GPT2-image-captioning · hugging face,” nlpconnect/vit-gpt2-image-captioning · Hugging Face, https://huggingface.co/nlpconnect/vit-gpt2-image-captioning (accessed Nov. 25, 2025). 


[2] D. Shah, “Vision Transformer: What it is & how it works [2024 guide],” V7, https://www.v7labs.com/blog/vision-transformer-guide (accessed Nov. 25, 2025). 


[3] R. Lamsal, “How llms work: A beginner’s guide to decoder-only Transformers,” Langformers Blog, https://blog.langformers.com/how-llms-work/?utm_source=chatgpt.com (accessed Nov. 25, 2025). 


[4] Comment et al., “Transformers in machine learning,” GeeksforGeeks, https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/ (accessed Nov. 25, 2025). 

[5] “Salesforce/Blip-image-captioning-base · hugging face,” Salesforce/blip-image-captioning-base · Hugging Face, https://huggingface.co/Salesforce/blip-image-captioning-base (accessed Nov. 25, 2025). 

[6] A. Sabir, Paper summary: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation | by Ahmed Sabir | Medium, https://ahmed-sabir.medium.com/paper-summary-blip-bootstrapping-language-image-pre-training-for-unified-vision-language-c1df6f6c9166 (accessed Nov. 26, 2025). 