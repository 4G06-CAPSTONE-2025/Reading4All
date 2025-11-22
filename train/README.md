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

5. Go to the `train` folder  
    ```bash
    cd train
    ```
6. Run the script
    ```bash
    python3 scripts/train.py
    ```

7. View the generated captions in terminal or json.
    Output will be saved to:
    ```
    ./outputs/captions_filled.json
    ```