import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

system_prompt = """
 You are an e-commerce product copywriter. Your task is to write a single
 persuasive product description based ONLY on the provided product 
 information.
 
 STRICT RULES: 
 
 1. LENGTH: Write exactly 50 to 90 words. Count carefully. 
 
 2. GROUNDING - this is the most critical rule:
- ONLY use facts explicitly stated in the provided product name,
 attributes, material, and warranty. 
- You MAY infer physical properties from materials (e.g., steel =
 durable, cotton = soft, aluminum = lightweight).
- You MAY infer functional capabilities from attributes (e.g., Bluetooth
 5.2 = wireless connectivity, 30 hr battery = long listening sessions).
- You MUST NOT invent features, specs, or claims not in the input (e.g.,
 do not add "waterproof", color options, or certifications unless provided). 
- You MUST NOT add lifestyle or use-case claims not in the input (e.g.,
 do not say "perfect for hiking", "eco-friendly choice", or "great gift
 idea"). 
- Generic marketing phrases with no factual claim are fine (e.g., "you\'ll
love it", "a great choice"). 
 
 3. TONE - 4 friendly and credible:
- Write in a warm, benefit-oriented voice. Speak TO the customer about
 what the product does for them. 
- Stay confident but measured. No superlatives ("the BEST ever"), no
 pressure tactics ("buy NOW!"), no stacked intensifiers ("truly absolutely 
 incredible"). 
- Do not write like a dry spec sheet. Connect features to benefits.
 
 4. FLUENCY- natural, cohesive prose:
- Write a unified paragraph, not bullet points or a list of disconnected 
 facts.
- Vary sentence structure. Group related features together. Build a
 logical flow (what it is -> what it\'s made of -> what that means for the
 customer).
- It should sound like a human copywriter, not a reformatted spec list.
 
 5. GRAMMAR: Use correct spelling, punctuation, and grammar throughout. Zero 
 errors.

 OUTPUT: Return ONLY the product description. No titles, labels, headers, or 
 extra commentary.
"""

def main():
    # Initiating LLM client
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY")
    )

    # Read CSV into dataframe
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "products_dataset.csv"))

    results = []

    for _, row in df.iterrows():
        user_msg = f"""Product: {row['product_name']}
            Attributes: {row['Product_attribute_list']} 
            Material: {row['material']}
            Warranty: {row['warranty']}"""

        start = time.time()
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg} 
            ],
            temperature=0.4
        )
        
        latency_ms = (time.time() - start) * 1000

        results.append({
            "product_name": row["product_name"],
            "generated_description": response.choices[0].message.content, 
            "latency_ms": round(latency_ms),
            "input_tokens": response.usage.prompt_tokens, 
            "output_tokens": response.usage.completion_tokens,
        })
        
    # Store results
    results_df = pd.DataFrame(results)

    for criterion in ["Fluency", "Grammar", "Tone", "Length", "Grounding",      
    "Latency", "Cost"]:                                                         
        results_df[criterion] = ""
    results_df["final_score"] = ""                                              
                    
    results_df.to_excel(os.path.join(SCRIPT_DIR, "assignment_01.xlsx"), index=False)

if __name__ == "__main__":
    main()