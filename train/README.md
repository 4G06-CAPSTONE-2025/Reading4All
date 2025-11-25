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

## nlpconnect/vit-gpt2-image-captioning

This model uses two transformer models: 
1. Vision Transformer (ViT) - image encoder:
 The input image is split up into flattened patches, which also have positional data. This sequence of patches are provided as an input into a standard transform encoder and used to pre-train the model with image labels. 
2. GPT- 2 - language model decoder: The architecture of GPT is made up of only decoders. The input text is broken into small units, often called tokens and each is mapped to a numerical ID. These IDs are converted into vectors that represent the semantic meaning of the tokens and are passed into the consecutive decoder blocks to generate text one token at a time. 

The ViT extracts features from the inputted image and GPT-2 produces a description of these features. Though this model is fast, often the produced alt text doesn't communicate the correct context and purpose of the image. 

* Transformer architecture is a type of neural network that can be used for natural language processing tasks. 
* Note Transformers are non sequential, which is one way they differ from Recurrent Neural Networks (RNNs). Additionally, transforms can access past information, while RNNs can only access previous state. 

## Salesforce/blip-image-captioning-base
This model uses a Bootstrapping Language-Image Pre-training (BLIP) for Unified Vision-Language Understanding and Generation

This is new Vision-Language Pre-training framework can be used for both vision-language understanding and generation tasks, while previous models focused on one or the other. Model have been able to improve their performance by increasing the size of the dataset used for training, by including noisy image-text pairs found on the internet. BLIP has been able to reduce the noise on these image-text pairs by using a caption generator and a filter, to generate new captions and filter defective ones. 

BLIP uses a similar architecture to the nlpconnect/vit-gpt2-image-captioning as it uses a Vision Encoder (ViT) and a text encoder.