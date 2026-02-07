## How to use Hugging Face

### How to Push with Git 
Allows to alter handler.py, replace models, etc. 

**Step 1**: Install the Hugging Face Hub client library to upload models 
```bash
pip install huggingface_hub
```

**Step 2:** Enable large file support for uploading models
```bash
git lfs install
```

**Step 3:** Authenticate your machine with Hugging Face account 
- This step will ask for a token. Use the reading4all-write token on hugging face (account -> settings -> access tokens)
- Press Y for yes 
```bash
huggingface-cli login
```

**Step 4**: Clone the Hugging Face Repo 
```bash
git clone https://huggingface.co/reading4all/generate-alt-text
```
Once it has been cloned successfully, check to see if you can see the current files (handler.py, etc.)
```bash
cd generate-alt-text
ls  
```

To push any new changes in repo under terminal: 
```bash
git add <file(s) changed/added>
git commit -m "commit message here" 
git push
```
Then you should see files updated on Hugging Face. 

---

### Endpoints on Git 

- Endpoints are stored on *Inference Endpoints*
- Pause when its not in use 
- ```Endpoint URL``` is what we use to access inference. 
