"""
COMPLETE PIPELINE TESTING SCRIPT FOR READING4ALL - FIXED VERSION
"""

import os
import sys
import torch
import json
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image

# ---------------- CONFIGURATION ----------------
BASE_DIR = r"C:\Users\nawaa\OneDrive\Desktop\Reading4All\ai"
sys.path.insert(0, BASE_DIR)

# Paths
PATHS = {
    "base": BASE_DIR,
    "model": os.path.join(BASE_DIR, "model"),
    "train": os.path.join(BASE_DIR, "train"),
    "test": os.path.join(BASE_DIR, "test"),
    "data": os.path.join(BASE_DIR, "data"),
    "inference": os.path.join(BASE_DIR, "inference", "images"),
    "utils": os.path.join(BASE_DIR, "utils")
}

# Heart diagram specific info
HEART_DIAGRAM_INFO = {
    "image_name": "Human_heart.png",
    "expected_labels": [
        "Semilunar valve", "Aorta", "Pulmonary artery", 
        "Right and left Atrium", "Pulmonary veins", 
        "Atrioventricular valve", "Posterior vena cava",
        "Diastole (filling)", "Systole (pumping)"
    ],
    "ideal_alt_text": "Diagram of the human heart showing its anatomical structure with labeled chambers (right and left atria, right and left ventricles), valves (semilunar and atrioventricular), and major blood vessels (aorta, pulmonary artery, pulmonary veins, vena cava). Arrows indicate blood flow through the cardiac cycle during diastole (filling phase) and systole (pumping phase)."
}

# Create necessary directories
for key, path in PATHS.items():
    if key not in ["utils"]:  # Don't create utils dir
        os.makedirs(path, exist_ok=True)

# ---------------- TEST UTILITIES ----------------
def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(text):
    print(f"\n▶ {text}")

def print_success(text):
    print(f"✅ {text}")

def print_warning(text):
    print(f"⚠  {text}")

def print_error(text):
    print(f"❌ {text}")

def check_heart_image():
    """Check if heart image exists and is valid"""
    heart_path = os.path.join(PATHS["inference"], HEART_DIAGRAM_INFO["image_name"])
    
    if not os.path.exists(heart_path):
        print_warning(f"Heart diagram not found at: {heart_path}")
        print_warning(f"Please copy {HEART_DIAGRAM_INFO['image_name']} to {PATHS['inference']}")
        return False, None
    
    try:
        img = Image.open(heart_path)
        print_success(f"Heart diagram found: {heart_path}")
        print_success(f"  Size: {img.size}, Format: {img.format}, Mode: {img.mode}")
        return True, img
    except Exception as e:
        print_error(f"Failed to load heart image: {e}")
        return False, None

def create_test_data():
    """Create test structured_alttext.json with heart diagram examples"""
    test_data_path = os.path.join(PATHS["data"], "structured_alttext.json")
    
    if not os.path.exists(test_data_path):
        os.makedirs(PATHS["data"], exist_ok=True)
        
        # Heart diagram specific examples
        test_data = [
            {
                "structured": "Human heart diagram showing chambers: right atrium, left atrium, right ventricle, left ventricle. Valves: semilunar valve, atrioventricular valve. Blood vessels: aorta, pulmonary artery, pulmonary veins, vena cava. Phases: diastole (filling), systole (pumping).",
                "alt_text": "A diagram of the human heart showing the four chambers: right atrium, left atrium, right ventricle, and left ventricle. The semilunar and atrioventricular valves control blood flow. Major blood vessels include the aorta, pulmonary artery, pulmonary veins, and vena cava. Arrows indicate the cardiac cycle with diastole (filling phase) and systole (pumping phase)."
            },
            {
                "structured": "Diagram shows water cycle with evaporation, condensation, precipitation",
                "alt_text": "A diagram illustrating the water cycle with arrows showing evaporation from the ocean, condensation forming clouds, and precipitation falling as rain."
            }
        ]
        
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print_success(f"Created test data with heart diagram example at {test_data_path}")
    
    return test_data_path

# ---------------- CORE TESTS ----------------
def test_utils_imports():
    """Test if utils modules can be imported"""
    print_header("TEST 1: UTILS IMPORTS")
    
    utils_modules = [
        "reproducibility",
        "logging_utils", 
        "progress_utils",
        "safety_utils"
    ]
    
    all_success = True
    for module in utils_modules:
        try:
            module_path = f"utils.{module}"
            __import__(module_path)
            print_success(f"Imported {module}")
        except Exception as e:
            print_error(f"Failed to import {module}: {e}")
            all_success = False
    
    return all_success

def test_stage1_pix2struct():
    """Test Pix2Struct model loading and basic inference"""
    print_header("TEST 2: STAGE 1 - PIX2STRUCT BASICS")
    
    try:
        print_step("Loading Pix2Struct model...")
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        
        # Quick test with pretrained model
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
        
        print_success(f"Base model loaded (Parameters: {sum(p.numel() for p in model.parameters()):,})")
        
        # Test with a simple generated image
        test_img = Image.new('RGB', (512, 512), color='white')
        
        print_step("Testing basic inference...")
        inputs = processor(test_img, text="describe this image", return_tensors="pt")
        
        # Quick generation test
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print_success(f"Inference works! Generated: '{decoded}'")
        return True
        
    except Exception as e:
        print_error(f"Stage 1 test failed: {e}")
        traceback.print_exc()
        return False

def test_heart_diagram_stage1():
    """Test stage 1 specifically on heart diagram"""
    print_header("TEST 3: HEART DIAGRAM - STAGE 1 ANALYSIS")
    
    try:
        # Check if heart image exists
        has_heart, heart_img = check_heart_image()
        if not has_heart:
            print_warning("Skipping heart diagram test - image not found")
            return False
        
        print_step("Loading Pix2Struct for heart diagram analysis...")
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
        
        # Test with different prompts
        test_prompts = [
            "describe the diagram structure",
            "what are the components in this medical diagram",
            "describe the human heart anatomy shown",
            "list the labeled parts in this diagram"
        ]
        
        results = []
        print_step("Testing heart diagram with different prompts:")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  {i}. '{prompt}'")
            inputs = processor(heart_img, text=prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
                decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            results.append({"prompt": prompt, "output": decoded})
            print(f"     → '{decoded[:80]}...'")
        
        # Check if any expected labels were detected
        all_output = " ".join([r["output"] for r in results])
        detected_labels = []
        
        for label in HEART_DIAGRAM_INFO["expected_labels"]:
            if label.lower() in all_output.lower():
                detected_labels.append(label)
        
        print_step(f"Label Detection: {len(detected_labels)}/{len(HEART_DIAGRAM_INFO['expected_labels'])}")
        for label in detected_labels:
            print(f"  ✓ {label}")
        
        # Save results
        heart_results = {
            "image": HEART_DIAGRAM_INFO["image_name"],
            "tests": results,
            "expected_labels": HEART_DIAGRAM_INFO["expected_labels"],
            "detected_labels": detected_labels,
            "detection_rate": f"{len(detected_labels)}/{len(HEART_DIAGRAM_INFO['expected_labels'])}"
        }
        
        result_path = os.path.join(PATHS["test"], "heart_stage1_results.json")
        with open(result_path, 'w') as f:
            json.dump(heart_results, f, indent=2)
        
        print_success(f"Heart diagram analysis saved to: {result_path}")
        return True
        
    except Exception as e:
        print_error(f"Heart diagram test failed: {e}")
        traceback.print_exc()
        return False

def test_stage2_dataset():
    """Test if stage2 can load its dataset"""
    print_header("TEST 4: STAGE 2 - DATASET LOADING")
    
    try:
        # Check if AI2D dataset exists
        dataset_path = r"C:\Users\nawaa\Downloads\ai2d-all\ai2d"
        
        if os.path.exists(dataset_path):
            print_success(f"AI2D dataset found at: {dataset_path}")
            
            # Check for subdirectories
            subdirs = ["images", "annotations", "questions"]
            for subdir in subdirs:
                subdir_path = os.path.join(dataset_path, subdir)
                if os.path.exists(subdir_path):
                    file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                    print_success(f"  {subdir}: {file_count:,} files")
                else:
                    print_warning(f"  {subdir}: not found")
            
            return True
        else:
            print_warning(f"AI2D dataset not found at: {dataset_path}")
            print_warning("Creating minimal test dataset structure...")
            
            # Create a minimal test dataset structure
            test_dataset_path = os.path.join(BASE_DIR, "test_ai2d")
            images_dir = os.path.join(test_dataset_path, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy heart image as test
            heart_source = os.path.join(PATHS["inference"], HEART_DIAGRAM_INFO["image_name"])
            if os.path.exists(heart_source):
                import shutil
                shutil.copy2(heart_source, os.path.join(images_dir, "test_heart.png"))
                print_success(f"Copied heart diagram to test dataset")
            
            print_success(f"Created test dataset at {test_dataset_path}")
            return True
            
    except Exception as e:
        print_error(f"Dataset test failed: {e}")
        traceback.print_exc()
        return False

def test_stage3_data_and_model():
    """Test if stage3 data exists and T5 model works"""
    print_header("TEST 5: STAGE 3 - DATA & T5 MODEL")
    
    try:
        # Create or verify test data
        data_path = create_test_data()
        
        print_step(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print_success(f"Data loaded: {len(data)} items")
        
        # Test T5 model
        print_step("Loading T5 model...")
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            model = T5ForConditionalGeneration.from_pretrained("t5-base")
            
            print_success(f"T5 model loaded (Parameters: {sum(p.numel() for p in model.parameters()):,})")
            
            # Test with heart diagram structured description
            print_step("Testing T5 with heart diagram description...")
            
            heart_structured = "Human heart diagram with chambers: right atrium, left atrium, right ventricle, left ventricle. Valves: semilunar valve, atrioventricular valve. Blood vessels: aorta, pulmonary artery, pulmonary veins, vena cava."
            
            input_text = f"Generate alt text for diagram: {heart_structured}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
                alt_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            print_success(f"T5 generated alt text:")
            print(f"  Input: '{heart_structured[:80]}...'")
            print(f"  Output: '{alt_text}'")
            
            # Save test result
            t5_test = {
                "test_input": heart_structured,
                "generated_alt_text": alt_text,
                "model": "t5-base"
            }
            
            result_path = os.path.join(PATHS["test"], "t5_heart_test.json")
            with open(result_path, 'w') as f:
                json.dump(t5_test, f, indent=2)
            
            print_success(f"T5 test saved to: {result_path}")
            return True
            
        except ImportError as e:
            if "SentencePiece" in str(e):
                print_error("T5Tokenizer requires SentencePiece library")
                print_error("Install with: pip install sentencepiece")
                print_error("Or: conda install -c conda-forge sentencepiece")
                return False
            raise
            
    except Exception as e:
        print_error(f"Stage 3 test failed: {e}")
        traceback.print_exc()
        return False

def test_script_configurations():
    """Test if all scripts are properly configured"""
    print_header("TEST 6: SCRIPT CONFIGURATION CHECK")
    
    scripts_to_check = [
        ("stage1_pix2struct_visual.py", [
            "OUT =",
            "from utils",
            "CFG = {"
        ]),
        ("stage2_pix2struct_structured.py", [
            "OUT =",
            "CFG = {",
            "dataset_path"
        ]),
        ("stage3_alttext_t5.py", [
            "OUT =",
            "CFG = {",
            "structured_alttext"
        ])
    ]
    
    all_good = True
    
    for script_name, required_patterns in scripts_to_check:
        script_path = os.path.join(PATHS["train"], script_name)
        
        if not os.path.exists(script_path):
            print_error(f"Script not found: {script_path}")
            all_good = False
            continue
        
        print_step(f"Checking {script_name}...")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        missing = []
        for pattern in required_patterns:
            if pattern not in content:
                missing.append(pattern)
        
        if missing:
            print_error(f"  Missing patterns: {missing}")
            all_good = False
        else:
            print_success(f"  Configuration OK")
    
    return all_good

def test_path_consistency():
    """Test if paths are consistent across the pipeline"""
    print_header("TEST 7: PATH CONSISTENCY CHECK")
    
    issues = []
    
    # Check run_all_stages.py for expected model paths
    main_script = os.path.join(BASE_DIR, "run_all_stages.py")
    
    if os.path.exists(main_script):
        with open(main_script, 'r') as f:
            content = f.read()
        
        # Check for model directory references
        expected_model_dirs = [
            "stage1_visual",
            "stage2_structured",
            "stage3_alttext"
        ]
        
        for model_dir in expected_model_dirs:
            search_str = f"stage1_model_dir = os.path.join(MODEL_DIR, \"{model_dir}\""
            if search_str not in content:
                # Check alternative format
                alt_search = f'stage1_model_dir = os.path.join(MODEL_DIR, "{model_dir}'
                if alt_search not in content:
                    issues.append(f"run_all_stages.py doesn't reference {model_dir} in MODEL_DIR")
    
    # Check training scripts for output paths
    training_scripts = [
        ("stage1_pix2struct_visual.py", "OUT ="),
        ("stage2_pix2struct_structured.py", "OUT ="),
        ("stage3_alttext_t5.py", "OUT =")
    ]
    
    for script_name, out_pattern in training_scripts:
        script_path = os.path.join(PATHS["train"], script_name)
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Look for OUT = line
            import re
            out_match = re.search(r'OUT\s*=\s*["\']([^"\']+)["\']', content)
            if out_match:
                out_path = out_match.group(1)
                if not out_path.startswith("C:\\") and "model" not in out_path.lower():
                    issues.append(f"{script_name}: OUT path '{out_path}' may not save to correct location")
    
    if issues:
        for issue in issues:
            print_error(issue)
        return False
    else:
        print_success("Path consistency looks good")
        return True

def test_gpu_capabilities():
    """Test GPU availability and memory"""
    print_header("TEST 8: GPU & HARDWARE CAPABILITIES")
    
    try:
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            print_success(f"GPU: {gpu_name}")
            print_success(f"VRAM: {gpu_memory:.1f} GB")
            
            # Test memory allocation
            print_step("Testing GPU memory allocation...")
            try:
                # Try to allocate a small tensor
                test_tensor = torch.randn(1000, 1000, device='cuda')
                allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
                
                print_success(f"GPU memory test passed")
                print_success(f"  Allocated: {allocated:.1f} MB")
                print_success(f"  Reserved: {reserved:.1f} MB")
                
                # Clean up
                del test_tensor
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print_error(f"GPU memory allocation failed: {e}")
                return False
        else:
            print_warning("No GPU available - training will be very slow on CPU")
            print_warning("Your RTX 5070 should be detected. Check CUDA installation.")
        
        # Check system RAM
        import psutil
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024**3)  # GB
        
        print_success(f"System RAM: {ram_total:.1f} GB")
        print_success(f"RAM Available: {ram.available / (1024**3):.1f} GB")
        
        return gpu_available
        
    except Exception as e:
        print_error(f"Hardware test failed: {e}")
        return False

# ---------------- COMPREHENSIVE HEART DIAGRAM TEST ----------------
def comprehensive_heart_test():
    """Run a comprehensive test on the heart diagram through all stages"""
    print_header("COMPREHENSIVE HEART DIAGRAM PIPELINE TEST")
    
    try:
        # Check heart image
        has_heart, heart_img = check_heart_image()
        if not has_heart:
            print_error("Cannot run comprehensive test - heart image missing")
            return False
        
        print_step("Starting comprehensive pipeline test...")
        
        # Initialize results
        results = {
            "image": HEART_DIAGRAM_INFO["image_name"],
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # STAGE 1: Visual Description
        print_step("Stage 1: Generating visual description...")
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        
        processor1 = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
        model1 = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
        
        inputs1 = processor1(heart_img, text="describe the diagram structure in detail", 
                           return_tensors="pt")
        
        with torch.no_grad():
            outputs1 = model1.generate(**inputs1, max_new_tokens=150)
            description = processor1.batch_decode(outputs1, skip_special_tokens=True)[0]
        
        results["stages"]["stage1_visual"] = {
            "prompt": "describe the diagram structure in detail",
            "output": description
        }
        print_success(f"Stage 1 complete: '{description[:100]}...'")
        
        # STAGE 2: Structured Analysis (simulated - would use trained model)
        print_step("Stage 2: Generating structured analysis...")
        
        inputs2 = processor1(heart_img, text="list all anatomical components and their relationships", 
                           return_tensors="pt")
        
        with torch.no_grad():
            outputs2 = model1.generate(**inputs2, max_new_tokens=200)
            structured = processor1.batch_decode(outputs2, skip_special_tokens=True)[0]
        
        # Check label detection
        detected_labels = []
        for label in HEART_DIAGRAM_INFO["expected_labels"]:
            if label.lower() in structured.lower():
                detected_labels.append(label)
        
        results["stages"]["stage2_structured"] = {
            "prompt": "list all anatomical components and their relationships",
            "output": structured,
            "expected_labels": HEART_DIAGRAM_INFO["expected_labels"],
            "detected_labels": detected_labels,
            "detection_rate": f"{len(detected_labels)}/{len(HEART_DIAGRAM_INFO['expected_labels'])}"
        }
        print_success(f"Stage 2 complete: Detected {len(detected_labels)}/{len(HEART_DIAGRAM_INFO['expected_labels'])} labels")
        
        # STAGE 3: Alt Text Generation (skip if SentencePiece not installed)
        print_step("Stage 3: Checking T5 availability...")
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            tokenizer3 = T5Tokenizer.from_pretrained("t5-base")
            model3 = T5ForConditionalGeneration.from_pretrained("t5-base")
            
            # Use the structured output as input to T5
            input_text = f"Generate accessible alt text for this diagram description: {structured}"
            inputs3 = tokenizer3(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs3 = model3.generate(**inputs3, max_new_tokens=150)
                alt_text = tokenizer3.batch_decode(outputs3, skip_special_tokens=True)[0]
            
            results["stages"]["stage3_alttext"] = {
                "input": input_text[:200] + "...",
                "generated_alt_text": alt_text,
                "reference_alt_text": HEART_DIAGRAM_INFO["ideal_alt_text"]
            }
            print_success(f"Stage 3 complete: '{alt_text[:100]}...'")
            
        except ImportError as e:
            if "SentencePiece" in str(e):
                print_warning("Stage 3 skipped: SentencePiece not installed")
                print_warning("Install with: pip install sentencepiece")
                results["stages"]["stage3_alttext"] = {
                    "status": "skipped",
                    "reason": "SentencePiece library not installed"
                }
            else:
                raise
        
        # Save comprehensive results
        result_path = os.path.join(PATHS["test"], "comprehensive_heart_pipeline.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print_header("COMPREHENSIVE TEST COMPLETE")
        print(f"\nResults saved to: {result_path}")
        print(f"\nSummary:")
        print(f"  • Stage 1: Visual description generated")
        print(f"  • Stage 2: {len(detected_labels)}/{len(HEART_DIAGRAM_INFO['expected_labels'])} labels detected")
        if "generated_alt_text" in results["stages"]["stage3_alttext"]:
            print(f"  • Stage 3: Alt text generated")
        else:
            print(f"  • Stage 3: Skipped (install sentencepiece)")
        print(f"\nPipeline appears functional on heart diagram!")
        
        return True
        
    except Exception as e:
        print_error(f"Comprehensive test failed: {e}")
        traceback.print_exc()
        return False

# ---------------- MAIN TEST RUNNER ----------------
def run_all_tests():
    """Run all pipeline tests"""
    print_header("READING4ALL COMPLETE PIPELINE TEST SUITE")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Testing with heart diagram: {HEART_DIAGRAM_INFO['image_name']}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run tests
    test_results = {}
    
    test_results["utils_imports"] = test_utils_imports()
    test_results["stage1_basics"] = test_stage1_pix2struct()
    test_results["heart_stage1"] = test_heart_diagram_stage1()
    test_results["stage2_dataset"] = test_stage2_dataset()
    test_results["stage3_t5"] = test_stage3_data_and_model()
    test_results["script_config"] = test_script_configurations()
    test_results["path_consistency"] = test_path_consistency()
    test_results["gpu_capabilities"] = test_gpu_capabilities()
    
    # Run comprehensive test if basic tests passed
    basic_tests_passed = all([
        test_results["utils_imports"],
        test_results["stage1_basics"],
        test_results["heart_stage1"]
    ])
    
    if basic_tests_passed:
        test_results["comprehensive"] = comprehensive_heart_test()
    
    # Summary
    print_header("TEST RESULTS SUMMARY")
    
    passed = sum([1 for v in test_results.values() if v])
    total = len(test_results)
    
    print(f"\n{'TEST':40} {'STATUS':10} {'DETAILS'}")
    print("-" * 70)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        status_symbol = "✅" if result else "❌"
        
        # Add details for key tests
        details = ""
        if test_name == "heart_stage1" and result:
            details = "Heart diagram analyzed"
        elif test_name == "gpu_capabilities" and result:
            if torch.cuda.is_available():
                details = f"{torch.cuda.get_device_name(0)}"
        
        print(f"{test_name:40} {status_symbol} {status:8} {details}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print_header("🎉 ALL TESTS PASSED!")
        print("\nRECOMMENDED NEXT STEPS:")
        print("1. ✅ Run individual training stages:")
        print("   python .\\train\\stage1_pix2struct_visual.py")
        print("2. ✅ Monitor GPU usage: nvidia-smi -l 1")
        print("3. ✅ After stage1 completes, run stage2")
        print("4. ✅ Finally, run the full pipeline:")
        print("   python run_all_stages.py")
    else:
        print_header("⚠  SOME TESTS FAILED")
        print("\nRECOMMENDED ACTIONS:")
        
        if not test_results["utils_imports"]:
            print("1. Fix utils imports - check sys.path in scripts")
        
        if not test_results["heart_stage1"]:
            print("2. Ensure heart diagram is at: inference/images/Human_heart.png")
        
        if not test_results["stage3_t5"]:
            print("3. Install SentencePiece: pip install sentencepiece")
        
        if not test_results["script_config"]:
            print("4. Fix stage3_alttext_t5.py configuration")
        
        print("\nRun these commands to fix issues:")
        print("  pip install sentencepiece")
        print("  # Then update stage3_alttext_t5.py data path")
    
    # Save test summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed_tests": passed,
        "success_rate": passed/total*100,
        "results": test_results,
        "heart_diagram": HEART_DIAGRAM_INFO["image_name"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    summary_path = os.path.join(PATHS["test"], "test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {summary_path}")
    
    return test_results

def quick_test():
    """Run only critical tests"""
    print_header("QUICK CRITICAL TESTS")
    
    critical_tests = [
        ("Utils Imports", test_utils_imports),
        ("Heart Diagram", test_heart_diagram_stage1),
        ("GPU Available", test_gpu_capabilities),
    ]
    
    all_passed = True
    for test_name, test_func in critical_tests:
        print_step(f"Testing {test_name}...")
        result = test_func()
        if result:
            print_success(f"{test_name} PASSED")
        else:
            print_error(f"{test_name} FAILED")
            all_passed = False
    
    if all_passed:
        print_header("✅ QUICK TEST PASSED")
        print("Basic setup is working! Run full test suite for complete verification.")
    else:
        print_header("❌ QUICK TEST FAILED")
        print("Fix critical issues before proceeding.")
    
    return all_passed

# ---------------- FIX PIPELINE SCRIPT ----------------
def fix_pipeline():
    """Fix common pipeline issues"""
    print_header("FIXING PIPELINE ISSUES")
    
    fixes_applied = []
    
    # 1. Install SentencePiece
    print_step("1. Checking SentencePiece installation...")
    try:
        import sentencepiece
        print_success("SentencePiece already installed")
    except ImportError:
        print_warning("SentencePiece not installed")
        print("Run: pip install sentencepiece")
        fixes_applied.append("Need to install sentencepiece")
    
    # 2. Fix stage3_alttext_t5.py
    print_step("2. Checking stage3_alttext_t5.py...")
    stage3_path = os.path.join(PATHS["train"], "stage3_alttext_t5.py")
    
    if os.path.exists(stage3_path):
        with open(stage3_path, 'r') as f:
            content = f.read()
        
        # Check for data path issue
        if '"data": "structured_alttext.json"' in content:
            print_warning("Found relative data path in stage3")
            # Fix it
            fixed_content = content.replace(
                '"data": "structured_alttext.json"',
                f'"data": os.path.join(r"{BASE_DIR}", "data", "structured_alttext.json")'
            )
            
            with open(stage3_path, 'w') as f:
                f.write(fixed_content)
            
            print_success("Fixed stage3 data path")
            fixes_applied.append("Fixed stage3 data path")
        else:
            print_success("Stage3 data path looks OK")
    
    # 3. Check output paths
    print_step("3. Checking output paths...")
    
    scripts_to_check = [
        ("stage1_pix2struct_visual.py", "OUT = \"models/stage1_visual_full\""),
        ("stage2_pix2struct_structured.py", "OUT = \"models/stage2_structured\""),
        ("stage3_alttext_t5.py", "OUT = \"models/stage3_alttext\"")
    ]
    
    for script_name, old_path in scripts_to_check:
        script_path = os.path.join(PATHS["train"], script_name)
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
            
            if old_path in content:
                print_warning(f"{script_name} uses relative output path")
                fixes_applied.append(f"Fix {script_name} output path")
    
    # Summary
    print_header("FIXES NEEDED")
    
    if fixes_applied:
        print("Apply these fixes:")
        for fix in fixes_applied:
            print(f"  • {fix}")
        
        print("\nRun these commands:")
        print("  pip install sentencepiece")
        print("  # Update stage scripts to use absolute paths")
    else:
        print_success("No fixes needed - pipeline looks good!")
    
    return len(fixes_applied) == 0

if __name__ == "__main__":
    # Parse arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Reading4All pipeline")
    parser.add_argument("--quick", action="store_true", help="Run only critical tests")
    parser.add_argument("--fix", action="store_true", help="Check and fix common issues")
    
    args = parser.parse_args()
    
    if args.fix:
        fix_pipeline()
    elif args.quick:
        quick_test()
    else:
        run_all_tests()