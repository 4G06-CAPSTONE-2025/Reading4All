from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import re


MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def build_prompt(alt_text: str) -> str:
    return f"""Edit the alt text below.
Do not summarize.
Do not remove details.
Do not add details.

Input: {alt_text}
Output:"""

def polish_alt_text(alt_text: str) -> str:
    prompt = build_prompt(alt_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def post_process_alt_text(alt_text):

        # removes leading/trailing whitespace
        alt_text = alt_text.strip()

        # replaces multiple spaces (or tabs/newlines) with a single space
        alt_text = re.sub(r"\s+", " ", alt_text)

        # remove repeated consecutive words, case-insensitive
        alt_text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", alt_text, flags=re.IGNORECASE)

        # remove repeated punctuations
        alt_text = re.sub(r"([.!?,;:]){2,}", r"\1", alt_text)

        # capitalize first character if needed
        if alt_text and alt_text[0].islower():
            alt_text = alt_text[0].upper() + alt_text[1:]

        # ensure sentence ends with punctuation
        if alt_text and alt_text[-1] not in ".!?":
            alt_text += "."

        # capitalize the first letter of each sentence
        sentences = re.split(r'([.!?]\s*)', alt_text)  # keep punctuation
        capitalized_sentences = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            capitalized_sentences.append(sentence)

            # add back punctuation if it exists
            if i + 1 < len(sentences):
                capitalized_sentences.append(sentences[i + 1])

        alt_text = "".join(capitalized_sentences)

        return alt_text

if __name__ == "__main__":
    sample_alt_text = "the angular momentum of a star is shown as a hexagonal lattice with angular velocity. at the center of the star, an arrow labeled f points upward and downward arrow labeled p indicates the direction of motion. in the direction of the star travels from its center to a point on its own side, while a curved arrow labeled f is labeled r. the diagram illustrates how angular momentum changes over time."
    cleaned_alt_text = post_process_alt_text(sample_alt_text)
    polished = polish_alt_text(cleaned_alt_text)

    print("Original:\n", sample_alt_text)
    print("Cleaned:\n", cleaned_alt_text)
    print("Polished:\n", polished)